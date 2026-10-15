"""Independent cut enumeration and real HiGHS/Gurobi baseline checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas
import pytest

import classical_baselines as cb
import network_interdiction as ni
import network_interdiction_plot as plot
import run_benchmark as rb
from test_gurobi_trajectory import require_gurobi

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['tiny_a', 'tiny_b', 'tiny_c', 'tiny_d'])
@pytest.mark.parametrize('budget', [0, 1, 2])
def test_lp_bound_and_repaired_cut_bracket_enumerated_optimum(name, budget):
    graph, source, sink = ni.read_flow_file(ROOT / 'testcases' / f'{name}.flow')
    result = cb.lp_round_repair(graph, source, sink, budget)
    optimum, _ = rb.brute_force_nip_optimum(graph, source, sink, budget)
    assert result['lower_bound'] <= optimum
    assert result['lower_bound'] == pytest.approx(result['lp_objective'])
    assert result['flow'] >= optimum
    assert len(set(result['interdicted_edges'])) == budget
    assert result['flow'] == rb.max_flow(graph, source, sink, result['interdicted_edges'])
    assert sum(result['fractional_x']) <= budget + 1e-7


@pytest.mark.parametrize('seed', range(8))
def test_bound_is_valid_on_cyclic_fractional_capacity_networks(seed):
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(5))
    for u in range(5):
        for v in range(5):
            if u != v and rng.random() < 0.3:
                graph.add_edge(u, v, weight=float(rng.integers(1, 20)) * 0.13)
    budget = min(2, len(graph.edges))
    result = cb.lp_round_repair(graph, 0, 4, budget)
    optimum, _ = rb.brute_force_nip_optimum(graph, 0, 4, budget)
    assert result['lower_bound'] <= optimum + 1e-14
    assert result['flow'] >= optimum - 1e-14


def test_rounding_passes_fractional_solution_to_the_same_repair(monkeypatch):
    graph = nx.DiGraph()
    # Deliberately insert edges in reverse sorted order.
    graph.add_weighted_edges_from([(1, 2, 5), (0, 2, 3), (0, 1, 5)], weight='weight')
    actual = cb.linprog
    seen = []
    def altered(*args, **kwargs):
        result = actual(*args, **kwargs)
        result.x[:3] = [0.51, 0.49, 0.50]
        return result
    def repair(g, bits, s, t, b):
        seen.append(bits)
        return [(0, 1)], rb.max_flow(g, s, t, [(0, 1)])
    monkeypatch.setattr(cb, 'linprog', altered)
    monkeypatch.setattr(ni, 'improve_dwave_solution', repair)
    result = cb.lp_round_repair(graph, 0, 2, 1)
    assert seen == [[1, 0, 1]]
    assert result['rounded_edges'] == [(0, 1), (1, 2)]
    assert result['interdicted_edges'] == [(0, 1)]


def test_fractional_relaxation_has_a_real_integrality_and_rounding_gap():
    # Two capacity-10 entrance arcs feed three capacity-4 exit arcs through
    # a complete high-capacity layer. With B=1 the best cut removes one exit
    # (flow 8). Mixing B=0 (flow 12) and B=2 (flow 0) gives LP value 6.
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [(0, u, 10) for u in (1, 2)]
        + [(u, v, 100) for u in (1, 2) for v in (3, 4, 5)]
        + [(v, 6, 4) for v in (3, 4, 5)])
    result = cb.lp_round_repair(graph, 0, 6, 1)
    optimum, _ = rb.brute_force_nip_optimum(graph, 0, 6, 1)
    assert optimum == 8
    assert result['lower_bound'] == pytest.approx(6)
    assert result['fractional_x'][:2] == pytest.approx([0.5, 0.5])
    assert result['rounded_edges'] == [(0, 1), (0, 2)]
    assert len(result['interdicted_edges']) == 1
    assert result['flow'] == 10
    assert result['flow'] > optimum > result['lower_bound']
    assert cb.relative_gap(result['flow'], result['lower_bound']) == 0.4


def test_dual_certificate_does_not_trust_inexact_stationarity():
    graph, source, sink = ni.read_flow_file(ROOT / 'testcases/tiny_a.flow')
    _, cost, matrix, rhs, bounds = cb._lp_model(graph, source, sink, 1)
    result = cb.linprog(cost, A_ub=matrix, b_ub=rhs, bounds=bounds, method='highs')
    perturbed = result.ineqlin.marginals + np.linspace(-0.2, 0.1, len(rhs))
    bound = cb.box_dual_bound(cost, matrix, rhs, bounds, perturbed)
    assert 0 <= bound <= 5
    assert bound < 5


def test_solver_failure_cannot_be_reported_as_a_bound(monkeypatch):
    graph, source, sink = ni.read_flow_file(ROOT / 'testcases/tiny_a.flow')
    monkeypatch.setattr(cb, 'linprog', lambda *a, **k: SimpleNamespace(
        success=False, message='time limit'))
    with pytest.raises(RuntimeError, match='time limit'):
        cb.lp_round_repair(graph, source, sink, 1)


@pytest.mark.parametrize('budget', [-1, 0.5, True, 5])
def test_invalid_budget_refused_before_solver(budget, monkeypatch):
    graph, source, sink = ni.read_flow_file(ROOT / 'testcases/tiny_a.flow')
    monkeypatch.setattr(cb, 'linprog', lambda *a, **k: pytest.fail('solver called'))
    with pytest.raises(ValueError, match='budget'):
        cb.lp_round_repair(graph, source, sink, budget)


def test_lp_cli_records_bounds_gaps_and_plot_series(tmp_path):
    out = tmp_path / 'lp.csv'
    assert rb.main(['--exact', 'brute', '--solvers', 'sa', '--baselines', 'lp',
                    '--case', str(ROOT / 'testcases/tiny_a.flow') + ':1:6',
                    '--out', str(out)]) == 0
    frame = pandas.read_csv(out, comment='#')
    row = frame.iloc[0]
    assert row.LP_lower_bound == pytest.approx(5)
    assert row.LP_Round_Repair >= 5
    assert row.Lower_bound == row.LP_lower_bound
    for name in ('SA_NIP', 'SA_NIP_Greedy', 'LP_Round_Repair'):
        if not pandas.isna(row[name]):
            assert row[name + '_gap'] == pytest.approx((row[name] - 5) / row[name])
    assert pandas.isna(row['QA_NIP(advantage1)_gap'])
    assert pandas.isna(row['Lagrangian_lower_bound'])
    assert 'exact-rational dual bound' in out.read_text()
    assert plot.available_baselines(frame) == ['LP-Round-Repair']
    assert plot.main(['--csv', str(out), '--outdir', str(tmp_path)]) == 0
    assert (tmp_path / 'lp_flow_reduction.pdf').stat().st_size > 0


def test_all_solver_and_baseline_bars_do_not_overlap_adjacent_instances(tmp_path, monkeypatch):
    bars = []
    save = plot.plt.savefig
    def capture(path):
        bars.extend(plot.plt.gca().patches)
        save(path)
    monkeypatch.setattr(plot.plt, 'savefig', capture)
    series = {name: [0.4, 0.7] for name in plot.SERIES_COLORS}
    output = tmp_path / 'all_series.pdf'
    plot.draw(series, ['first instance', 'second instance'], 'Flow reduction',
              (0, 1.1), output)
    assert len(bars) == 16
    # Matplotlib emits one rectangle per instance for each series. Compare
    # the actual rendered extents, independently of the position helper.
    first_group, second_group = bars[::2], bars[1::2]
    assert max(bar.get_x() + bar.get_width() for bar in first_group) < min(
        bar.get_x() for bar in second_group)
    assert output.stat().st_size > 0


def test_lp_is_available_when_linear_qubo_penalty_has_no_strict_window():
    row = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 3, 'auto',
                      'brute', ['sa'], baselines=['lp'])
    assert row['lambda'] == ''
    assert row['LP_Round_Repair'] == 0
    assert row['Lower_bound'] == 0
    assert row['LP_Round_Repair_gap'] == 0
    assert row['SA_NIP_gap'] == ''


def test_zero_flow_gap_and_inconsistent_bound():
    assert cb.relative_gap(0, 0) == 0
    assert cb.relative_gap(10, 7) == 0.3
    with pytest.raises(ValueError, match='exceeds'):
        cb.relative_gap(1, 2)


def test_lagrangian_preserves_each_valid_bound_and_feasible_incumbent():
    require_gurobi()
    graph, source, sink = ni.read_flow_file(ROOT / 'testcases/tiny_a.flow')
    result = ni.solve_nip_gurobi_relaxation(graph, source, sink, 1, 0, 15)
    assert result['lower_bound'] == pytest.approx(5)
    assert result['feasible_flow'] == 5
    assert rb.max_flow(graph, source, sink, result['interdicted_edges']) == 5
    assert result['iterations']
    for item in result['iterations']:
        exact_lagrangian = min(rb.brute_force_nip_optimum(graph, source, sink, k)[0]
                               + item['lambda'] * (k - 1) for k in range(5))
        assert item['lower_bound'] <= 5 + 1e-8
        assert item['lower_bound'] == pytest.approx(exact_lagrangian)


def test_over_budget_lagrangian_solution_is_not_a_feasible_result():
    require_gurobi()
    graph, source, sink = ni.read_flow_file(ROOT / 'testcases/tiny_a.flow')
    result = ni.solve_nip_gurobi_relaxation(graph, source, sink, 0, 0, 0,
                                           max_iterations=1)
    assert result['iterations'][0]['num_interdicted'] > 0
    assert result['feasible_flow'] is None
    assert result['interdicted_edges'] == []
    assert result['lower_bound'] == 0


def test_lagrangian_csv_serializes_iteration_bounds():
    require_gurobi()
    row = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 6, 'brute', [],
                      baselines=['lp', 'lagrangian'])
    assert row['Lagrangian_Feasible'] == 5
    assert row['Lagrangian_lower_bound'] == pytest.approx(5)
    assert row['Lower_bound'] == pytest.approx(5)
    assert row['Lagrangian_Feasible_gap'] == pytest.approx(0)
    assert json.loads(row['Lagrangian_iterations'])


def test_negative_lagrangian_multiplier_is_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        ni.solve_nip_gurobi_relaxation(nx.DiGraph(), 0, 1, 0, -1, 2)

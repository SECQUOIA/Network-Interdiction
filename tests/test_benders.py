"""Independent max-flow and interdiction oracles for the Benders baseline."""
import itertools
import json
from pathlib import Path

import networkx as nx
import pytest
import pandas as pd

import benders
import network_interdiction as ni
import run_benchmark as rb
from test_gurobi_trajectory import require_gurobi

ROOT = Path(__file__).resolve().parents[1]


def tiny(name='tiny_a'):
    return ni.read_flow_file(ROOT / 'testcases' / f'{name}.flow')


def gap_graph():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [(0, u, 10) for u in (1, 2)]
        + [(u, v, 100) for u in (1, 2) for v in (3, 4, 5)]
        + [(v, 6, 4) for v in (3, 4, 5)])
    return graph, 0, 6


@pytest.mark.parametrize('name', ['tiny_a', 'tiny_b', 'tiny_c', 'tiny_d'])
@pytest.mark.parametrize('budget', [0, 1, 2])
def test_benders_matches_independent_interdiction_enumeration(name, budget):
    require_gurobi()
    graph, source, sink = tiny(name)
    result = benders.solve(graph, source, sink, budget)
    optimum, _ = rb.brute_force_nip_optimum(graph, source, sink, budget)
    assert result['status'] == 'optimal'
    assert result['flow'] == optimum
    assert result['lower_bound'] == pytest.approx(optimum)
    assert len(set(result['interdicted_edges'])) <= budget
    assert result['flow'] == rb.max_flow(graph, source, sink, result['interdicted_edges'])
    bounds = [item['lower_bound'] for item in result['history']]
    incumbents = [item['best_flow'] for item in result['history']]
    assert bounds == sorted(bounds)
    assert incumbents == sorted(incumbents, reverse=True)


def test_every_generated_cut_is_a_global_bound_and_the_oracle_cut_is_tight():
    require_gurobi()
    graph, source, sink = tiny()
    result = benders.solve(graph, source, sink, 1, record_cuts=True)
    assert result['cuts']
    # Four interdiction variables: all 16 assignments, including budgets
    # outside this solve, independently check global validity of every cut.
    assert len(graph.edges) == 4
    for count in range(5):
        for removed in itertools.combinations(graph.edges, count):
            actual = rb.max_flow(graph, source, sink, removed)
            for cut in result['cuts']:
                assert cut['value'] - sum(cut['coefficients'][e] for e in removed) <= actual
            value, coefficients = benders.flow_cut(graph, source, sink, removed)
            assert value == actual
            assert all(coefficients[e] == 0 for e in removed)
            assert value - sum(coefficients[e] for e in removed) == actual


def test_benders_closes_the_gap_left_by_the_continuous_lp():
    require_gurobi()
    graph, source, sink = gap_graph()
    result = benders.solve(graph, source, sink, 1)
    optimum, _ = rb.brute_force_nip_optimum(graph, source, sink, 1)
    assert optimum == result['flow'] == 8
    assert result['status'] == 'optimal'
    assert result['lower_bound'] == pytest.approx(8)
    assert result['iterations'] > 1


def test_cyclic_antiparallel_network_preserves_flow_cut_bounds():
    require_gurobi()
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 6), (1, 2, 9), (2, 3, 8), (3, 4, 7),
                                   (1, 0, 3), (2, 1, 4), (3, 2, 6), (4, 3, 2),
                                   (0, 2, 5), (1, 3, 6), (2, 4, 3)])
    result = benders.solve(graph, 0, 4, 1, record_cuts=True)
    assert result['flow'] == rb.brute_force_nip_optimum(graph, 0, 4, 1)[0]
    assert result['status'] == 'optimal'
    # Independently examine 67 cuts, including larger budgets than this solve.
    for count in range(3):
        for removed in itertools.combinations(graph.edges, count):
            actual = rb.max_flow(graph, 0, 4, removed)
            for cut in result['cuts']:
                assert cut['value'] - sum(cut['coefficients'][e] for e in removed) <= actual


def test_iteration_limit_preserves_feasible_result_and_a_valid_unclosed_bound():
    require_gurobi()
    graph, source, sink = gap_graph()
    result = benders.solve(graph, source, sink, 1, max_iterations=1)
    assert result['status'] == 'iteration_limit'
    assert result['iterations'] == 1
    assert result['lower_bound'] <= 8 <= result['flow']
    assert result['lower_bound'] < result['flow']
    assert rb.max_flow(graph, source, sink, result['interdicted_edges']) == result['flow']


def test_deadline_exhausted_by_initial_flow_dispatches_no_master(monkeypatch):
    graph, source, sink = tiny()
    ticks = iter([0, 2, 2, 2])
    monkeypatch.setattr(benders.time, 'perf_counter', lambda: next(ticks))
    monkeypatch.setattr(ni, '_gurobi_model', lambda: pytest.fail('master dispatched'))
    result = benders.solve(graph, source, sink, 1, time_limit=1)
    assert result['status'] == 'time_limit'
    assert result['flow'] == rb.max_flow(graph, source, sink)
    assert result['interdicted_edges'] == []
    assert result['lower_bound'] == 0
    assert result['iterations'] == 0


@pytest.mark.parametrize('kwargs', [dict(time_limit=0), dict(time_limit=float('inf')),
                                     dict(max_iterations=0), dict(max_iterations=True)])
def test_invalid_resource_limits_fail_before_dispatch(kwargs, monkeypatch):
    monkeypatch.setattr(benders, 'flow_cut', lambda *a: pytest.fail('flow dispatched'))
    with pytest.raises(ValueError):
        benders.solve(*tiny(), 1, **kwargs)


def test_fractional_capacities_are_explicitly_unsupported():
    graph, source, sink = tiny()
    graph[0][1]['weight'] = 0.5
    with pytest.raises(ValueError, match='integer capacities'):
        benders.solve(graph, source, sink, 1)


def test_csv_cli_preserves_limits_bounds_and_backend_independent_reference(tmp_path):
    require_gurobi()
    graph, source, sink = gap_graph()
    path = tmp_path / 'gap.flow'
    path.write_text(f'{len(graph)}\n{source} {sink}\n' + ''.join(
        f'{u} {v} {data["weight"]}\n' for u, v, data in graph.edges(data=True)))
    out = tmp_path / 'limited.csv'
    assert rb.main(['--exact', 'brute', '--solvers', 'none',
                    '--baselines', 'lp,benders', '--benders-max-iterations', '1',
                    '--benders-time-limit', '12', '--case', f'{path}:1:auto',
                    '--out', str(out)]) == 0
    row = pd.read_csv(out, comment='#').iloc[0]
    assert row.Opt_NIP == 8
    assert row.Benders_status == 'iteration_limit'
    assert row.Benders_iterations == 1
    assert row.Benders_lower_bound <= 8 <= row.Benders
    assert row.LP_lower_bound == 6
    assert row.Lower_bound == max(row.LP_lower_bound, row.Benders_lower_bound)
    assert row.Benders_gap == pytest.approx((row.Benders - row.Lower_bound) / row.Benders)
    assert row.Benders_time_sec >= 0
    assert rb.max_flow(graph, source, sink, json.loads(row.Benders_interdicted_edges)) == row.Benders
    assert '12s cooperative wall limit, 1 master iterations' in out.read_text()


def test_unrun_benders_cells_remain_empty():
    row = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 6, 'brute', [])
    for column in ['Benders', 'Benders_lower_bound', 'Benders_status',
                   'Benders_iterations', 'Benders_history', 'Benders_time_sec',
                   'Benders_gap', 'Benders_interdicted_edges']:
        assert row[column] == ''


@pytest.mark.parametrize('option,value', [('--benders-time-limit', '0'),
                                         ('--benders-time-limit', 'inf'),
                                         ('--benders-max-iterations', '0')])
def test_invalid_cli_limit_fails_before_any_case_work(option, value, monkeypatch):
    monkeypatch.setattr(rb, 'run_case', lambda *a, **k: pytest.fail('case dispatched'))
    with pytest.raises(SystemExit) as error:
        rb.main(['--baselines', 'benders', option, value,
                 '--case', 'testcases/tiny_a.flow:1:auto', '--out', 'unused.csv'])
    assert error.value.code == 2

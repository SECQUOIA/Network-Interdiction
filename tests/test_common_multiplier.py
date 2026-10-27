"""Independent allocation and whole-network checks for a shared budget price."""
import itertools
import json
import random
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

import decomposition as dc
import run_benchmark as rb
import run_decomposition_benchmark as rdb


def exact_allocation_value(curves, budget):
    target = min(budget, sum(len(curve) - 1 for curve in curves))
    return min(sum(curve[k] for curve, k in zip(curves, allocation))
               for allocation in itertools.product(*(range(len(c)) for c in curves))
               if sum(allocation) == target)


def test_shared_price_improves_its_initial_repaired_allocation():
    curves = [[20, 13, 1, 1], [20, 20, 19, 4], [20, 13, 13, 8]]
    result = dc.allocate_budget_multiplier(curves, 3)
    assert result['history'][0]['surrogate_value'] == 44
    assert exact_allocation_value(curves, 3) == result['surrogate_value'] == 34
    assert result['surrogate_lower_bound'] == 34
    assert result['status'] == 'optimal_surrogate'
    assert result['iterations'] > 1
    assert any(row['budget_violation'] != 0 for row in result['history'])


@pytest.mark.parametrize('budget', [0, 1, 2, 3, 4, 9])
def test_tied_linear_curves_repair_to_the_available_budget(budget):
    curves = [[10, 5, 0], [10, 5, 0]]
    result = dc.allocate_budget_multiplier(curves, budget)
    assert sum(result['budgets']) == min(budget, 4)
    assert result['status'] == 'optimal_surrogate'
    assert result['surrogate_value'] == result['surrogate_lower_bound'] == exact_allocation_value(curves, budget)


def test_nonconvex_curve_reports_its_true_duality_gap():
    # At B=1 the integer optimum is 9, but the lower convex envelope is 5.
    # No single price selects k=1; stopping at 100 steps must not claim exactness.
    result = dc.allocate_budget_multiplier([[10, 9, 0]], 1)
    assert result['budgets'] == [1] and result['surrogate_value'] == 9
    assert 4.9 < result['surrogate_lower_bound'] <= 5
    assert result['status'] == 'iteration_limit' and result['iterations'] == 100
    assert all(row['requested_budgets'] in ([0], [2]) for row in result['history'])


def test_limited_run_returns_feasible_allocation_and_valid_bounds():
    curves = [[20, 13, 1, 1], [20, 20, 19, 4], [20, 13, 13, 8]]
    result = dc.allocate_budget_multiplier(curves, 3, max_iterations=1)
    assert result['status'] == 'iteration_limit'
    assert result['iterations'] == 1
    assert result['surrogate_lower_bound'] <= 34 <= result['surrogate_value']
    assert sum(result['budgets']) == 3


def test_bounds_bracket_exhaustive_allocations_and_history_is_monotone():
    rng = random.Random(39)
    for _ in range(40):
        curves = []
        for _ in range(rng.randint(1, 4)):
            curve = [20]
            for _ in range(rng.randint(0, 4)):
                curve.append(rng.randint(0, curve[-1]))
            curves.append(curve)
        budget = rng.randint(0, 7)
        result = dc.allocate_budget_multiplier(curves, budget, max_iterations=30)
        optimum = exact_allocation_value(curves, budget)
        assert result['surrogate_lower_bound'] <= optimum <= result['surrogate_value']
        allocation = result['budgets']
        assert sum(allocation) == min(budget, sum(len(c)-1 for c in curves))
        assert sum(c[k] for c, k in zip(curves, allocation)) == result['surrogate_value']
        bounds = [row['best_surrogate_lower_bound'] for row in result['history']]
        values = [row['best_surrogate_value'] for row in result['history']]
        assert bounds == sorted(bounds) and values == sorted(values, reverse=True)
        for row in result['history']:
            assert row['surrogate_lower_bound'] <= optimum <= row['surrogate_value']
            assert sum(row['repaired_budgets']) == sum(allocation)
        if result['status'] == 'optimal_surrogate':
            assert result['surrogate_value'] == optimum


def test_binary_float_certificate_rounds_outward():
    curves = [[0.3, 0.2, 0.0], [0.3, 0.1, 0.0]]
    result = dc.allocate_budget_multiplier(curves, 2)
    exact_curves = [[Fraction(x) for x in c] for c in curves]
    optimum = exact_allocation_value(exact_curves, 2)
    assert Fraction(result['surrogate_lower_bound']) <= optimum
    assert Fraction(result['surrogate_value']) >= optimum


def test_empty_and_flat_curves_need_no_price_updates():
    for curves, budget, value in [([], 0, 0), ([[5], [3]], 2, 8), ([[7, 7], [2, 2]], 1, 9)]:
        result = dc.allocate_budget_multiplier(curves, budget)
        assert result['status'] == 'optimal_surrogate' and result['iterations'] == 1
        assert result['surrogate_value'] == result['surrogate_lower_bound'] == value


@pytest.mark.parametrize('curves,budget,options', [([[]], 0, {}), ([[1, 2]], 1, {}),
    ([[float('nan')]], 0, {}), ([[-1]], 0, {}), ([[1]], -1, {}),
    ([[1]], True, {}), ([[1]], 0, {'max_iterations': 0})])
def test_invalid_controls_or_curves_are_rejected(curves, budget, options):
    with pytest.raises(ValueError):
        dc.allocate_budget_multiplier(curves, budget, **options)


def test_finite_individual_curves_cannot_overflow_the_exported_bound():
    with pytest.raises(ValueError, match='finite export range'):
        dc.allocate_budget_multiplier([[1e308], [1e308]], 0)


def test_invalid_iteration_limit_does_not_solve_any_piece(monkeypatch):
    monkeypatch.setattr(dc, 'piece_curves', lambda *a, **k: pytest.fail('piece work dispatched'))
    with pytest.raises(ValueError, match='max_iterations'):
        dc.common_multiplier_split(None, None, None, 1, None, None, max_iterations=0)


def test_incomplete_piece_oracle_cannot_supply_an_exact_curve(monkeypatch):
    class IncompleteModel:
        Params = SimpleNamespace()
        Status = dc.ni.GRB.TIME_LIMIT
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def optimize(self):
            pass
    model = IncompleteModel()
    monkeypatch.setattr(dc.ni, '_build_nip_milp', lambda *a: (model, {}))
    sub = SimpleNamespace(graph=None, src=0, tgt=1, interdictable=[(0, 1)])
    with pytest.raises(RuntimeError, match='did not prove optimality'):
        dc.gurobi_piece_solver(sub, 1)


def test_priced_allocation_is_scored_on_the_real_combined_network():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 9), (1, 3, 9), (0, 2, 4), (2, 3, 4)])
    pieces = [{0, 1}, {2, 3}]
    result = dc.common_multiplier_split(graph, 0, 3, 1, pieces, dc.brute_force_piece_solver)
    cuts, flow = dc.solve_decomposed(graph, 0, 3, pieces, result['budgets'],
                                    dc.brute_force_piece_solver)
    assert len(cuts) <= 1
    assert flow == rb.max_flow(graph, 0, 3, cuts) == 4
    # Sum of each independently scored piece's flow is not global flow.
    assert result['surrogate_lower_bound'] == result['surrogate_value'] > flow
    assert dc.split_budget(graph, 0, 3, 1, pieces, 'multiplier',
                           dc.brute_force_piece_solver) == result['budgets']


def test_cli_rejects_invalid_price_budget_before_loading_graph(monkeypatch):
    monkeypatch.setattr(dc.ni, 'read_flow_file', lambda *a: pytest.fail('graph loaded'))
    with pytest.raises(SystemExit) as error:
        dc.main(['unused.flow', '--pieces', '2', '--budget', '1',
                 '--split', 'multiplier', '--multiplier-iterations', '0'])
    assert error.value.code == 2


def test_benchmark_limits_and_surrogate_diagnostics_reach_real_allocation(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(dc, 'gurobi_piece_solver', dc.brute_force_piece_solver)
    instance = Path(__file__).resolve().parents[1] / 'testcases/tiny_a.flow'
    cases = tmp_path / 'cases.csv'
    cases.write_text(f'instance,B,Opt_NIP,Greedy,maxflow\n{instance},1,5,5,15\n')
    out = tmp_path / 'results.csv'
    assert rdb.main(['--cases', str(cases), '--settings', 'pieces4',
                     '--rules', 'multiplier', '--modes', 'sequential',
                     '--max-terminals', '0', '--multiplier-iterations', '1',
                     '--out', str(out)]) == 0
    frame = pd.read_csv(out, comment='#')
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.rule == 'multiplier' and row['mode'] == 'sequential'
    history = json.loads(row.multiplier_history)
    assert len(history) == 1
    curves = json.loads(row.allocation_curves)
    assert row.surrogate_lower_bound <= exact_allocation_value(curves, 1) <= row.surrogate_value
    graph, source, sink = dc.ni.read_flow_file(instance)
    cuts = json.loads(row.interdicted_edges)
    assert len(cuts) <= 1
    assert row.flow == rb.max_flow(graph, source, sink, cuts)
    assert 'max_terminals: 0; multiplier_iterations: 1' in out.read_text()


@pytest.mark.parametrize('option,value', [('--rules', 'unknown'), ('--modes', 'bad'),
    ('--settings', 'pieces4,pieces4'), ('--max-terminals', '13'),
    ('--multiplier-iterations', '0')])
def test_benchmark_invalid_controls_fail_before_loading_cases(monkeypatch, option, value):
    monkeypatch.setattr(rdb.pd, 'read_csv', lambda *a, **k: pytest.fail('cases loaded'))
    with pytest.raises(SystemExit) as error:
        rdb.main(['--out', 'unused.csv', option, value])
    assert error.value.code == 2


def test_benchmark_flushes_completed_variant_before_later_failure(tmp_path, monkeypatch):
    import csv
    instance = Path(__file__).resolve().parents[1] / 'testcases/tiny_a.flow'
    cases = tmp_path / 'cases.csv'
    cases.write_text(f'instance,B,Opt_NIP,Greedy,maxflow\n{instance},1,5,5,15\n')
    out = tmp_path / 'partial.csv'
    def rows(*args, **kwargs):
        yield ('mincut', 'sequential', [1, 0, 0, 0], 5, 0.1, [(0, 1)], {})
        with out.open() as stream:
            saved = list(csv.DictReader(line for line in stream if not line.startswith('#')))
        assert len(saved) == 1 and saved[0]['flow'] == '5'
        raise RuntimeError('later variant failed')
    monkeypatch.setattr(rdb, 'case_rows', rows)
    with pytest.raises(RuntimeError, match='later variant'):
        rdb.main(['--cases', str(cases), '--settings', 'pieces4',
                  '--max-terminals', '0', '--out', str(out)])

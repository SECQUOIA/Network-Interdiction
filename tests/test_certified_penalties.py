"""Independent truth-table and min-cut checks for the certified encoding."""
import itertools

import dimod
import networkx as nx
import pytest

import certified_penalties as cp
import network_interdiction as ni


def graph():
    g = nx.DiGraph()
    g.add_weighted_edges_from([(0, 1, 5), (1, 3, 5), (0, 2, 5), (2, 3, 5)])
    return g


def direct_terms(g, sample, B, allowed):
    """Evaluate the normalized polynomial, independently of PyQUBO."""
    pi = {v: (1 if v == 0 else 0 if v == 3 else sample[f'pi_{v}']) for v in g}
    k = raw = q = 0
    for i, (u, v) in enumerate(sorted(g.edges())):
        a, b = pi[u], pi[v]
        if (u, v) not in allowed:
            raw += g[u][v]['weight'] * a * (1-b)
            continue
        x, c, w = (sample[f'{name}_{i}'] for name in ('x', 'gamma', 'w'))
        k += x
        raw += g[u][v]['weight'] * (1-x) * c
        q += w * (a+b+c-1) + b*c-b-c+1
    return k, raw, (k-B)**2, q


@pytest.mark.parametrize('B,allowed', [(0, None), (1, None), (4, None),
                                     (1, {(0, 1), (1, 3)})])
def test_every_binary_assignment_has_the_claimed_energy_and_separation(B, allowed):
    g = graph()
    allowed = set(g.edges()) if allowed is None else allowed
    bqm, cert = cp.certified_squared_qubo(B, g, 0, 3, allowed)
    assert bqm.num_variables <= 14  # cap before exhaustive allocation
    samples = dimod.ExactSolver().sample(bqm)
    feasible, infeasible = [], []
    for datum in samples.data(['sample', 'energy'], sorted_by=None):
        k, raw, p, q = direct_terms(g, datum.sample, B, allowed)
        energy = datum.energy + cert.normalized_offset
        assert energy == raw + cert.cardinality*p + cert.dual*q
        assert 0 <= raw <= cert.objective_upper_bound and q >= 0
        (feasible if k == B and q == 0 else infeasible).append(energy)
    assert max(feasible) <= cert.objective_upper_bound < min(infeasible)
    # Independent original NIP: remove every allowed subset of size <= B.
    opt = float('inf')
    for k in range(B+1):
        for edges in itertools.combinations(allowed, k):
            residual = g.copy()
            residual.remove_edges_from(edges)
            opt = min(opt, nx.maximum_flow_value(residual, 0, 3, capacity='weight'))
    assert min(feasible) == opt


def test_squared_form_encodes_a_budget_with_no_strict_linear_window():
    from run_benchmark import lambda_window, choose_lambda
    assert choose_lambda(*lambda_window([10, 5, 0], 1)) is None
    bqm, cert = cp.certified_squared_qubo(1, graph(), 0, 3)
    assert dimod.ExactSolver().sample(bqm).first.energy + cert.normalized_offset == 5


def test_default_linear_model_and_new_clique():
    g = graph()
    default = ni.nip_to_qubo(1, g, 0, 3, 21, 21)
    explicit = ni.nip_to_qubo(1, g, 0, 3, 21, 21, budget_penalty='linear')
    squared, _ = cp.certified_squared_qubo(1, g, 0, 3)
    assert default == explicit
    assert set(default.variables) == set(squared.variables)
    for i, j in itertools.combinations(range(4), 2):
        assert default.get_quadratic(f'x_{i}', f'x_{j}', default=0) == 0
        assert squared.get_quadratic(f'x_{i}', f'x_{j}') == 42


@pytest.mark.parametrize('budget', [-1, 5, 1.5, True])
def test_bad_budget_rejected_before_compilation(budget, monkeypatch):
    monkeypatch.setattr(ni, 'nip_to_qubo', lambda *a, **kw: pytest.fail('compiled'))
    with pytest.raises(ValueError, match='budget'):
        cp.certified_squared_qubo(budget, graph(), 0, 3)


@pytest.mark.parametrize('capacity', [-1, 1.5, float('nan'), True, 2**53])
def test_uncertifiable_capacities_rejected(capacity, monkeypatch):
    g = graph()
    g[0][1]['weight'] = capacity
    monkeypatch.setattr(ni, 'nip_to_qubo', lambda *a, **kw: pytest.fail('compiled'))
    with pytest.raises(ValueError, match='capacities|safety bound'):
        cp.certified_squared_qubo(1, g, 0, 3)


def test_invalid_graph_and_terminals():
    with pytest.raises(ValueError, match='directed'):
        cp.certified_squared_qubo(1, graph().to_undirected(), 0, 3)
    with pytest.raises(ValueError, match='consecutive'):
        cp.certified_squared_qubo(1, nx.relabel_nodes(graph(), {3: 4}), 0, 4)
    with pytest.raises(ValueError, match='distinct'):
        cp.certified_squared_qubo(1, graph(), 0, 0)
    with pytest.raises(ValueError, match='arcs'):
        cp.certified_squared_qubo(1, graph(), 0, 3, [(3, 0)])
    with pytest.raises(ValueError, match='budget_penalty'):
        ni.nip_to_qubo(1, graph(), 0, 3, 21, 21, budget_penalty='typo')


def test_zero_capacities_and_no_interdictable_arcs():
    g = nx.DiGraph()
    g.add_weighted_edges_from([(0, 1, 0)])
    bqm, cert = cp.certified_squared_qubo(0, g, 0, 1, [])
    assert cert.objective_upper_bound == 0 and cert.cardinality == cert.dual == 1
    assert bqm.num_variables == 0 and bqm.offset == 0
    bqm, cert = cp.certified_squared_qubo(1, g, 0, 1)
    best = dimod.ExactSolver().sample(bqm).first
    assert best.energy + cert.normalized_offset == 0
    assert best.sample['x_0'] == 1


def test_comparison_size_guard_precedes_compilation(tmp_path, monkeypatch):
    import run_penalty_comparison as comparison
    path = tmp_path / 'large.flow'
    path.write_text('52\n0 51\n' + ''.join(f'{i} {i+1} 1\n' for i in range(51)))
    monkeypatch.setattr(comparison, 'certified_squared_qubo',
                        lambda *a: pytest.fail('compiled oversized input'))
    with pytest.raises(ValueError, match='bounded comparison'):
        comparison.run_case(path, 1)


def test_actual_gurobi_comparison_on_empty_window_fixture(tmp_path):
    import gurobipy as gp
    import run_penalty_comparison as comparison
    path = tmp_path / 'equal.flow'
    path.write_text('4\n0 3\n0 1 5\n1 3 5\n0 2 5\n2 3 5\n')
    try:
        row = comparison.run_case(path, 1)
    except gp.GurobiError as error:
        pytest.skip(f'Gurobi unavailable: {error}')
    assert row['nip_optimum'] == row['decoded_flow'] == 5
    assert row['qubo_energy'] + row['certificate']['normalized_offset'] == 5
    assert row['linear_window']['strict_window'] is False
    assert row['squared_couplers'] - row['linear_couplers'] == 6

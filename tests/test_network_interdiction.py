"""
Smoke/regression tests for the core NIP pipeline (issue #7).

These deliberately use a tiny, hand-built instance with a known optimum
rather than anything from testcases/, so the expected answer can be
verified by hand (see build_two_path_graph docstring) instead of trusting
another solver run.
"""
import networkx as nx
import dimod
import gurobipy
import pytest

import network_interdiction as ni


def build_two_path_graph():
    """
    Two edge-disjoint src->tgt paths:
        0 (src) --10--> 1 --10--> 3 (tgt)   path A, bottleneck capacity 10
        0 (src) --5---> 2 --5---> 3 (tgt)   path B, bottleneck capacity 5

    Baseline max flow = 10 + 5 = 15. With a budget of B=1 interdicted edge,
    cutting either edge of path A leaves only path B (flow drops to 5);
    cutting either edge of path B leaves only path A (flow drops to 10);
    leaving everything alone keeps flow at 15. So the unique optimal
    objective for B=1 is 5, achieved by interdicting (0,1) or (1,3) (not
    both -- only one edge is allowed).
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_edge(0, 1, weight=10)
    g.add_edge(1, 3, weight=10)
    g.add_edge(0, 2, weight=5)
    g.add_edge(2, 3, weight=5)
    return g, 0, 3


def test_maxflow_baseline():
    g, src, tgt = build_two_path_graph()
    flow_value, _ = nx.maximum_flow(g, src, tgt, capacity='weight')
    assert flow_value == 15


def test_read_flow_file_roundtrip(tmp_path):
    path = tmp_path / "toy.flow"
    path.write_text("4\n0 3\n0 1 10\n1 3 10\n0 2 5\n2 3 5\n")
    g, src, tgt = ni.read_flow_file(str(path))
    assert (src, tgt) == (0, 3)
    assert set(g.nodes()) == {0, 1, 2, 3}
    assert g[0][1]['weight'] == 10
    assert g[1][3]['weight'] == 10
    assert g[0][2]['weight'] == 5
    assert g[2][3]['weight'] == 5


def _require_gurobi():
    """Skip a test instead of failing it when no usable Gurobi license is
    available in this environment (e.g. a host-locked license file that
    doesn't match the current machine -- see issue #10)."""
    try:
        m = gurobipy.Model("license-check")
        m.setParam('OutputFlag', 0)
        m.optimize()
    except gurobipy.GurobiError as e:
        pytest.skip(f"no usable Gurobi license in this environment: {e}")


def test_solve_nip_gurobi_finds_known_optimum():
    _require_gurobi()
    g, src, tgt = build_two_path_graph()
    obj_val, interdicted_edges = ni.solve_nip_gurobi(g, src, tgt, B=1)
    assert obj_val == 5
    assert interdicted_edges == [(0, 1)] or interdicted_edges == [(1, 3)]


def test_nip_to_qubo_builds_expected_variables():
    g, src, tgt = build_two_path_graph()
    bqm = ni.nip_to_qubo(1, g, src, tgt, cardinality_penalty_coeff=7, dual_penalty_coeff=8)
    assert isinstance(bqm, dimod.BinaryQuadraticModel)
    assert bqm.vartype is dimod.BINARY
    # 4 edges * 3 (x, gamma, w) = 12, plus pi_i for the 2 non-src/tgt nodes.
    # pi_src/pi_tgt are created (Binary(f'pi_{i}')) but nip_to_qubo always
    # substitutes them with literal constants (1/0) at every use site, so
    # they never appear in any term and PyQUBO correctly drops them from
    # the compiled BQM -- they're not missing, they're structurally unused.
    expected = set()
    for i in range(4):
        expected.update({f'x_{i}', f'gamma_{i}', f'w_{i}'})
    non_terminal_nodes = set(g.nodes()) - {src, tgt}
    expected.update(f'pi_{i}' for i in non_terminal_nodes)
    assert set(bqm.variables) == expected


def test_qubo_ground_state_matches_known_optimum():
    """
    Brute-forces the BQM's exact ground state (16 binary variables --
    small enough for dimod.ExactSolver) and checks it reproduces the same
    optimum as test_solve_nip_gurobi_finds_known_optimum, without needing
    Gurobi.

    The penalty coefficients (7, 8) were found by sweeping candidate
    values against this fixture, not derived from a general formula --
    that derivation is the subject of issue #2. Notably, *larger*
    coefficients (e.g. 20/21) make this specific test fail: the
    cardinality term in nip_to_qubo() is linear in (sum(x) - B), not
    squared, so an overly large coefficient rewards interdicting nothing
    at all rather than penalizing only budget violations. That's exactly
    the kind of subtlety issue #2's bound derivation needs to nail down.
    """
    g, src, tgt = build_two_path_graph()
    B = 1
    bqm = ni.nip_to_qubo(B, g, src, tgt, cardinality_penalty_coeff=7, dual_penalty_coeff=8)
    sampleset = dimod.ExactSolver().sample(bqm)
    best = sampleset.first.sample

    edge_list = sorted(g.edges(), key=lambda e: (e[0], e[1]))
    interdicted = [edge_list[i] for i in range(len(edge_list)) if best[f'x_{i}'] == 1]

    g_copy = g.copy()
    g_copy.remove_edges_from(interdicted)
    flow_value, _ = nx.maximum_flow(g_copy, src, tgt, capacity='weight')

    assert len(interdicted) == B
    assert flow_value == 5


def test_improve_dwave_solution_repairs_to_budget():
    g, src, tgt = build_two_path_graph()
    edge_list = sorted(g.edges(), key=lambda e: (e[0], e[1]))
    B = 1

    # Start from "no edges interdicted" (x all zero) and check the greedy
    # repair adds exactly one edge and finds the known optimum.
    x_empty = [0, 0, 0, 0]
    interdicted, objective = ni.improve_dwave_solution(g, x_empty, src, tgt, B)
    assert len(interdicted) == B
    assert objective == 5

    # Start from "too many edges interdicted" (all four) and check the
    # greedy repair removes down to the budget.
    x_full = [1, 1, 1, 1]
    interdicted, objective = ni.improve_dwave_solution(g, x_full, src, tgt, B)
    assert len(interdicted) == B


@pytest.mark.parametrize("fn_name", ["solve_nip_qubo", "solve_nip_qubo_repair_once_end"])
def test_unsupported_qubo_paths_raise_clearly(fn_name):
    g, src, tgt = build_two_path_graph()
    fn = getattr(ni, fn_name)
    with pytest.raises(NotImplementedError, match="nip_cubic.jl"):
        fn("toy", g, 1, 10, src, tgt, B=1)

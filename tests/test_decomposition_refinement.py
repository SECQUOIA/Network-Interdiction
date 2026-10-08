"""
Tests for the decomposition's newer parts (issue #14): the 'marginal' budget
split and its allocation, 'iterated' solving, refine_split(), the Gurobi
piece solver, and fitting pieces to a qubit budget.

Every expected flow is checked against brute-force optima of the original
graph (run_benchmark.brute_force_nip_optimum), never against the
decomposition's own numbers. The three small fixtures were found by
searching seeded random graphs for a case where the new method changes the
answer; each is pinned as a literal graph, so a version of the method that
does nothing fails here.
"""
import itertools
import os
import random

import networkx as nx
import pytest

import decomposition as dc
import network_interdiction as ni
import run_benchmark as rb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gurobi_usable():
    """A Gurobi license that can solve a small model (the CI trial can)."""
    try:
        import gurobipy
        model = gurobipy.Model()
        model.setParam('OutputFlag', 0)
        model.addVars(10)
        model.optimize()
        return True
    except Exception:
        return False


needs_gurobi = pytest.mark.skipif(not gurobi_usable(),
                                  reason='no usable Gurobi license')


def graph(edges, num_nodes):
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    return g


def optimum(g, src, tgt, B):
    return rb.brute_force_nip_optimum(g, src, tgt, B)[0]


# Seed 15: with one cut each, piece {0, 3, 5} goes first and cuts an arc that
# is only right while the other piece has not cut; re-solving it afterwards
# reaches the optimum, 0.
ITERATED = (graph([(0, 1, 6), (0, 3, 1), (1, 0, 9), (1, 2, 6), (1, 4, 8),
                   (2, 3, 6), (2, 4, 8), (2, 5, 4), (3, 0, 8), (3, 4, 6),
                   (3, 5, 2), (5, 0, 3), (5, 4, 7)], 6),
            0, 5, [{0, 3, 5}, {1, 2, 4}], 2)

# Seed 99: the min-cut rule gives piece {0, 2, 3} two cuts and {1, 4, 5} one;
# the pieces' own curves say the reverse, which reaches the optimum, 0.
MARGINAL = (graph([(0, 2, 9), (0, 3, 4), (0, 4, 2), (1, 5, 2), (2, 3, 9),
                   (3, 1, 6), (3, 2, 7), (3, 4, 4), (3, 5, 4), (4, 1, 4),
                   (4, 3, 4), (5, 1, 3), (5, 4, 2)], 6),
            0, 5, [{0, 2, 3}, {1, 4, 5}], 3)

# Seed 23: a one-and-one split leaves 2 whatever the pieces cut; both cuts
# in piece {0, 3, 4, 7} leave 0.
REFINE = (graph([(0, 4, 5), (0, 7, 2), (1, 6, 8), (2, 5, 8), (3, 4, 8),
                 (3, 5, 6), (3, 6, 1), (4, 3, 1), (4, 6, 2), (5, 1, 4),
                 (5, 4, 5), (5, 7, 3), (6, 2, 1), (6, 3, 9), (7, 3, 1)], 8),
          0, 7, [{0, 3, 4, 7}, {1, 2, 5, 6}], 2)


# --- allocate_budget ----------------------------------------------------------

def test_allocation_takes_a_nonconvex_curve_whole():
    # One cut at a time, B's 4 beats A's 1 first; but A's second cut is worth
    # 9, so both cuts in A save 10 against B's best of 5.
    assert dc.allocate_budget([[10, 9, 0], [10, 6, 5]], 2) == [2, 0]


def test_allocation_is_the_best_of_every_split():
    rng = random.Random(4)
    for _ in range(200):
        curves = []
        for _piece in range(rng.randint(1, 4)):
            flows = [20]
            for _cut in range(rng.randint(0, 4)):
                flows.append(rng.randint(0, flows[-1]))
            curves.append(flows)
        B = rng.randint(0, 6)
        budgets = dc.allocate_budget(curves, B)
        total = min(B, sum(len(c) - 1 for c in curves))
        assert sum(budgets) == total
        best = max(sum(c[0] - c[k] for c, k in zip(curves, split))
                   for split in itertools.product(
                       *[range(len(c)) for c in curves])
                   if sum(split) == total)
        assert sum(c[0] - c[k] for c, k in zip(curves, budgets)) == best


# --- the three behaviours -----------------------------------------------------

def test_iterated_solving_lets_early_pieces_answer_later_ones():
    g, s, t, pieces, B = ITERATED
    budgets = dc.split_budget(g, s, t, B, pieces, 'mincut')
    assert budgets == [1, 1]
    solve = dc.brute_force_piece_solver
    _, sequential = dc.solve_decomposed(g, s, t, pieces, budgets, solve)
    cut, iterated = dc.solve_decomposed(g, s, t, pieces, budgets, solve,
                                        'iterated')
    assert sequential == 1
    assert iterated == optimum(g, s, t, B) == 0
    assert rb.max_flow(g, s, t, cut) == 0 and len(cut) <= B


def test_marginal_split_follows_the_pieces_own_curves():
    g, s, t, pieces, B = MARGINAL
    solve = dc.brute_force_piece_solver
    mincut = dc.split_budget(g, s, t, B, pieces, 'mincut')
    marginal = dc.split_budget(g, s, t, B, pieces, 'marginal', solve)
    assert (mincut, marginal) == ([2, 1], [1, 2])
    assert dc.solve_decomposed(g, s, t, pieces, mincut, solve)[1] == 2
    assert dc.solve_decomposed(g, s, t, pieces, marginal, solve)[1] == \
        optimum(g, s, t, B) == 0


def test_refine_moves_budget_where_it_is_worth_more():
    g, s, t, pieces, B = REFINE
    solve = dc.brute_force_piece_solver
    for split in ([1, 1], [0, 2]):
        assert dc.solve_decomposed(g, s, t, pieces, split, solve,
                                   'iterated')[1] == 2
    budgets, cut, flow = dc.refine_split(g, s, t, pieces, [1, 1], solve)
    assert budgets == [2, 0]
    assert flow == optimum(g, s, t, B) == 0
    assert rb.max_flow(g, s, t, cut) == 0


def random_graph(num_nodes, num_arcs, seed):
    rng = random.Random(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    while g.number_of_edges() < num_arcs:
        u, v = rng.sample(range(num_nodes), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=rng.randint(1, 9))
    return g


def test_refinements_never_lose_and_never_beat_the_optimum():
    solve = dc.brute_force_piece_solver
    for seed in range(25):
        n = 6 + seed % 3
        g = random_graph(n, n + 5, seed)
        s, t = 0, n - 1
        pieces = dc.partition_graph(g, 2, seed=0)
        B = 2
        best = optimum(g, s, t, B)
        budgets = dc.split_budget(g, s, t, B, pieces, 'mincut')
        _, sequential = dc.solve_decomposed(g, s, t, pieces, budgets, solve)
        _, iterated = dc.solve_decomposed(g, s, t, pieces, budgets, solve,
                                          'iterated')
        _, _, refined = dc.refine_split(g, s, t, pieces, budgets, solve)
        assert best <= refined <= iterated <= sequential


# --- the Gurobi piece solver --------------------------------------------------

@needs_gurobi
def test_gurobi_piece_solver_matches_brute_force():
    for seed in range(15):
        n = 7
        g = random_graph(n, 12, seed)
        pieces = dc.partition_graph(g, 2, seed=0)
        for index in range(2):
            sub = dc.piece_subproblem(g, 0, n - 1, pieces, index)
            for budget in (1, 2):
                brute = dc.brute_force_piece_solver(sub, budget)
                exact = dc.gurobi_piece_solver(sub, budget)
                assert len(exact) <= budget
                assert rb.max_flow(g, 0, n - 1, exact) == \
                    rb.max_flow(g, 0, n - 1, brute)


@needs_gurobi
def test_restricted_milp_cuts_only_allowed_arcs():
    g, s, t, _, _ = MARGINAL
    allowed = [(3, 5), (5, 1)]
    _, cut = ni.solve_nip_gurobi(g, s, t, 2, interdictable=allowed)
    assert set(cut) <= set(allowed)
    best = min(rb.max_flow(g, s, t, arcs) for k in range(3)
               for arcs in itertools.combinations(allowed, k))
    assert rb.max_flow(g, s, t, cut) == best


# --- fitting pieces to a device ------------------------------------------------

@pytest.mark.parametrize('name, pieces', [('50.planarflow', 4),
                                          ('50_5.2dnnflow', 4)])
def test_piece_qubo_size_matches_the_built_qubo(name, pieces):
    g, s, t = ni.read_flow_file(os.path.join(REPO_ROOT, 'testcases', name))
    parts = dc.partition_graph(g, pieces, seed=0)
    for index in range(pieces):
        sub = dc.piece_subproblem(g, s, t, parts, index)
        assert dc.piece_qubo_size(sub) == len(
            dc.piece_qubo(sub, 1, 1.0).variables)


def test_partition_to_fit_uses_the_fewest_pieces_that_fit():
    g, s, t = ni.read_flow_file(os.path.join(REPO_ROOT, 'testcases',
                                             '50_5.2dnnflow'))
    limit = 400
    pieces, sizes = dc.partition_to_fit(g, s, t, limit)
    assert max(sizes) <= limit
    assert sizes == [dc.piece_qubo_size(dc.piece_subproblem(g, s, t, pieces, i))
                     for i in range(len(pieces))]
    fewer = dc.partition_graph(g, len(pieces) - 1, seed=0)
    assert max(dc.piece_qubo_size(dc.piece_subproblem(g, s, t, fewer, i))
               for i in range(len(fewer))) > limit


def test_partition_to_fit_reports_a_limit_nothing_meets():
    g, s, t, _, _ = MARGINAL
    with pytest.raises(ValueError, match='fits 3 qubits'):
        dc.partition_to_fit(g, s, t, 3)

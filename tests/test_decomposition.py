"""
Tests for decomposition.py (issue #14): partitioning, arc ownership, the two
budget-split rules, the exact contraction of zero-budget zones, the per-piece
QUBO, and combining the pieces' interdictions.

Exactness is checked against max flows of the original graph, never against
the reduced network the code builds, so a contraction that is merely
self-consistent cannot pass.
"""
import itertools
import os
import random

import dimod
import neal
import networkx as nx
import pytest

import decomposition as dc
import network_interdiction as ni

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_testcase(name):
    return ni.read_flow_file(os.path.join(REPO_ROOT, 'testcases', name))


def surviving_flow(di_g, src, tgt, removed=()):
    remaining = di_g.copy()
    remaining.remove_edges_from(removed)
    return nx.maximum_flow_value(remaining, src, tgt, capacity='weight')


def build_four_paths():
    """
    Four parallel s-t paths, split into P0 = {0, 1, 2} and P1 = {3, 4, 5}:

        path 1   0 --2--> 1 --5--> 5     bottleneck (0,1), owned by P0
        path 2   0 --2--> 2 --5--> 5     bottleneck (0,2), owned by P0
        path 3   0 --2--> 3 --5--> 5     bottleneck (0,3), owned by P0: its
                                         tail is in P0, its head in P1
        path 4   0 --5--> 4 --3--> 5     bottleneck (4,5), owned by P1

    plus (4,0) into the source and (5,1) out of the sink, which carry no s-t
    flow. Max flow 2+2+2+3 = 9, and the minimum cut is the four
    bottlenecks: three owned by P0, one by P1 (by the heads it would be two
    and two). Local max flows, by hand:
      P0: the source feeds (0,1) 2 and (0,2) 2, and drains to the super-sink
          through (0,3) 2 + (0,4) 5 = 7, (1,5) 5 and (2,5) 5 -> 2+2+7 = 11
      P1: the super-source feeds 3 with 2 and 4 with 5 -> min(2,5) +
          min(5,3) = 5. (1,5) and (2,5) enter P1 at the sink itself, so they
          cross no arc P1 owns: they count for P0, as its outflow, not for P1.
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(6))
    for u, v, w in [(0, 1, 2), (0, 2, 2), (0, 3, 2), (0, 4, 5),
                    (1, 5, 5), (2, 5, 5), (3, 5, 5), (4, 5, 3),
                    (4, 0, 4), (5, 1, 3)]:
        g.add_edge(u, v, weight=w)
    return g, 0, 5, [{0, 1, 2}, {3, 4, 5}]


def build_series_middle():
    """
    s=0 -> 1, then a middle zone {1, 2, 3}, then 3 -> t=4:

        0 --10--> 1 --4--> 2 --6--> 3 --10--> 4
                  1 --3-----------> 3

    The piece is {0, 4} (source and sink), so it owns only (0, 1); the
    middle and its exit arc (3, 4) carry no budget. The middle is entered
    only at 1 and left only at 3, and its max flow from 1 to 3 is 4 + 3 = 7,
    so exactly contracted it is one arc of capacity 7 -- the "constant
    capacity" of the slides.
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(5))
    for u, v, w in [(0, 1, 10), (1, 2, 4), (2, 3, 6), (1, 3, 3), (3, 4, 10)]:
        g.add_edge(u, v, weight=w)
    return g, 0, 4, [{0, 4}, {1, 2, 3}]


def assert_subproblem_is_exact(di_g, src, tgt, pieces, index, subsets,
                               interdicted=(), max_terminals=12):
    """
    For every set of the piece's arcs `subsets` yields, the reduced network
    without them must have the original graph's max flow without them (and
    without `interdicted`). Returns the subproblem.
    """
    sub = dc.piece_subproblem(di_g, src, tgt, pieces, index, interdicted,
                              max_terminals)
    reduced = {original: arc for arc, original in sub.arc_map.items()}
    base = di_g.copy()
    base.remove_edges_from(interdicted)
    for arcs in subsets(sorted(reduced)):
        assert surviving_flow(sub.graph, sub.src, sub.tgt,
                              [reduced[arc] for arc in arcs]) == \
            surviving_flow(base, src, tgt, arcs), (index, arcs)
    return sub


def all_subsets_up_to(size):
    def subsets(arcs):
        for count in range(min(size, len(arcs)) + 1):
            yield from itertools.combinations(arcs, count)
    return subsets


def random_subsets(count, max_size, seed):
    def subsets(arcs):
        rng = random.Random(seed)
        for _ in range(count):
            yield tuple(rng.sample(arcs, rng.randint(0, min(max_size,
                                                            len(arcs)))))
    return subsets


# --- partition and ownership -------------------------------------------------

@pytest.mark.parametrize('name', [
    '50.planarflow',
    # Bisection alone leaves all four pieces of this one disconnected.
    '100_6.2dnnflow',
])
def test_partition_covers_every_node_once_in_connected_pieces(name):
    g, _, _ = load_testcase(name)
    pieces = dc.partition_graph(g, 4)

    assert len(pieces) == 4
    assert sorted(node for piece in pieces for node in piece) == \
        sorted(g.nodes())
    undirected = g.to_undirected()
    assert all(nx.is_connected(undirected.subgraph(piece))
               for piece in pieces)
    assert pieces == dc.partition_graph(g, 4)


def test_stray_parts_of_a_piece_join_the_neighbour_they_touch_most():
    """
    On the path 0-1-2-3-4 plus a node 5 joined to 2, 3 and 4, the piece
    {0, 1, 5} is in two parts. The stray part {5} shares two edges with
    {2, 3} and one with {4}, so it joins {2, 3}.
    """
    graph = nx.path_graph(5)
    graph.add_edges_from([(5, 2), (5, 3), (5, 4)])
    pieces = dc._make_contiguous(graph, [{0, 1, 5}, {2, 3}, {4}])
    assert pieces == [{0, 1}, {2, 3, 5}, {4}]


def test_arcs_are_owned_by_the_piece_of_their_tail():
    g, _, _, pieces = build_four_paths()
    assert dc.arc_owners(g, pieces) == {
        (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0, (1, 5): 0, (2, 5): 0,
        (3, 5): 1, (4, 5): 1, (4, 0): 1, (5, 1): 1,
    }


@pytest.mark.parametrize('pieces', [
    [{0, 1, 2}, {2, 3, 4, 5}],     # node 2 twice
    [{0, 1, 2}, {3, 4}],           # node 5 missing
    [{0, 1, 2}, {3, 4, 5}, set()],  # empty piece
])
def test_pieces_must_partition_the_nodes(pieces):
    g, _, _, _ = build_four_paths()
    with pytest.raises(ValueError):
        dc.arc_owners(g, pieces)


# --- budget split ------------------------------------------------------------

def test_mincut_split_counts_the_min_cut_arcs_each_piece_owns():
    g, src, tgt, pieces = build_four_paths()
    assert surviving_flow(g, src, tgt) == 9
    assert dc.mincut_arc_counts(g, src, tgt, pieces) == [3, 1]
    # quotas 3.75 and 1.25; the leftover unit goes to the larger remainder
    assert dc.split_budget(g, src, tgt, 5, pieces, 'mincut') == [4, 1]


def test_local_split_follows_the_local_max_flows():
    g, src, tgt, pieces = build_four_paths()
    assert dc.local_max_flows(g, src, tgt, pieces) == [11, 5]
    # quotas 55/16 and 25/16, i.e. 3.44 and 1.56
    assert dc.split_budget(g, src, tgt, 5, pieces, 'local') == [3, 2]


def test_unknown_split_strategy_is_rejected():
    g, src, tgt, pieces = build_four_paths()
    with pytest.raises(ValueError, match='split strategy'):
        dc.split_budget(g, src, tgt, 2, pieces, 'even')


@pytest.mark.parametrize('B, weights, expected', [
    (7, [3, 1], [5, 2]),       # quotas 5.25, 1.75
    (2, [3, 1], [2, 0]),       # tied remainders: the larger weight wins
    (2, [1, 3], [0, 2]),       # quotas 0.5, 1.5: the tie goes to weight 3
    (1, [2, 2], [1, 0]),       # tied weights too: the lower index wins
    (3, [0, 5, 0], [0, 3, 0]),
    (4, [0, 0], [0, 0]),       # nothing carries flow: nothing to spend on
])
def test_apportion_by_largest_remainder(B, weights, expected):
    assert dc.apportion(B, weights) == expected


def test_apportion_always_spends_the_budget_within_one_of_each_quota():
    rng = random.Random(0)
    for _ in range(200):
        weights = [rng.randint(0, 20) for _ in range(rng.randint(1, 6))]
        weights[0] += 1
        B = rng.randint(0, 40)
        shares = dc.apportion(B, weights)
        assert sum(shares) == B
        for share, weight in zip(shares, weights):
            quota = B * weight / sum(weights)
            assert quota - 1 < share < quota + 1


# --- exact contraction and the per-piece subproblem --------------------------

def test_zero_budget_middle_contracts_to_one_constant_capacity_arc():
    g, src, tgt, pieces = build_series_middle()
    sub = assert_subproblem_is_exact(g, src, tgt, pieces, 0,
                                     all_subsets_up_to(2))

    assert sub.contracted
    assert sub.free_terminals == 2
    assert {(u, v): d['weight'] for u, v, d in sub.graph.edges(data=True)} \
        == {(0, 1): 10, (1, 2): 7, (2, 3): 10}
    assert sub.arc_map == {(0, 1): (0, 1)}


def test_piece_subproblems_are_exact_on_random_graphs():
    """
    Random graphs and random partitions into 2-3 pieces, each piece checked
    against the uninterdicted graph and with two arcs of other pieces
    already cut (the sequential mode): every subset of up to two of the
    piece's arcs must leave the original max flow.
    """
    rng = random.Random(1)
    checked = contracted_smaller = 0
    for _ in range(30):
        n = rng.randint(5, 9)
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        for u, v in itertools.permutations(range(n), 2):
            if rng.random() < 0.35:
                g.add_edge(u, v, weight=rng.randint(1, 9))
        order = list(range(n))
        rng.shuffle(order)
        cuts = sorted(rng.sample(range(1, n), rng.randint(1, 2)))
        pieces = [set(order[a:b]) for a, b in zip([0] + cuts, cuts + [n])]
        for index in range(len(pieces)):
            others = [arc for arc in g.edges() if arc[0] not in pieces[index]]
            for interdicted in [(), others[:2]]:
                sub = assert_subproblem_is_exact(
                    g, 0, n - 1, pieces, index, all_subsets_up_to(2),
                    interdicted)
                checked += 1
                contracted_smaller += sub.graph.number_of_nodes() < n
    assert checked > 100
    # The contraction has to have merged something for this to test it.
    assert contracted_smaller > checked // 4


@pytest.mark.parametrize('name', ['50.planarflow', '50_5.2dnnflow'])
def test_piece_subproblems_are_exact_on_testcases(name):
    g, src, tgt = load_testcase(name)
    pieces = dc.partition_graph(g, 4)
    contracted = 0
    for index in range(len(pieces)):
        sub = assert_subproblem_is_exact(g, src, tgt, pieces, index,
                                         random_subsets(15, 12, index))
        contracted += sub.contracted
    assert contracted >= 2


def test_piece_owns_exactly_its_useful_arcs():
    """The interdictable arcs are the piece's own, minus arcs into the source
    and out of the sink, which carry no s-t flow."""
    g, src, tgt, pieces = build_four_paths()
    sub = dc.piece_subproblem(g, src, tgt, pieces, 1)
    assert sorted(sub.arc_map.values()) == [(3, 5), (4, 5)]
    assert sorted(sub.graph.nodes()) == list(range(sub.graph.number_of_nodes()))


def build_source_behind_bottleneck():
    """
    0 --1--> 1 --5--> 2 --5--> 3 with the piece {2, 3}: the source sits
    inside the rest of the graph, behind the bottleneck (0,1). Max flow 1.
    Were the source neither pinned to the source side nor kept as a
    terminal, it would share a class with 1 and the bottleneck would vanish
    (flow 5).
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    for u, v, w in [(0, 1, 1), (1, 2, 5), (2, 3, 5)]:
        g.add_edge(u, v, weight=w)
    return g, 0, 3, [{2, 3}, {0, 1}]


def build_sink_beside_dead_end():
    """
    0 --5--> 1 --5--> 2 --1--> 3, plus a dead end 2 --5--> 4, with the piece
    {0, 1}: the sink sits inside the rest of the graph. Max flow 1. The dead
    end 4 reaches no terminal; were the sink neither pinned to the sink side
    nor kept as a terminal, the two would share a class and (2,4) would
    count as flow into the sink (flow 5).
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(5))
    for u, v, w in [(0, 1, 5), (1, 2, 5), (2, 3, 1), (2, 4, 5)]:
        g.add_edge(u, v, weight=w)
    return g, 0, 3, [{0, 1}, {2, 3, 4}]


@pytest.mark.parametrize('build', [build_source_behind_bottleneck,
                                   build_sink_beside_dead_end])
def test_source_and_sink_inside_the_zone_stay_on_their_sides(build):
    g, src, tgt, pieces = build()
    assert surviving_flow(g, src, tgt) == 1
    sub = assert_subproblem_is_exact(g, src, tgt, pieces, 0,
                                     all_subsets_up_to(2))
    assert sub.contracted


def test_too_many_terminals_skips_the_contraction_but_stays_exact():
    g, src, tgt = load_testcase('50.planarflow')
    pieces = dc.partition_graph(g, 4)
    index = next(i for i in range(4) if dc.piece_subproblem(
        g, src, tgt, pieces, i).contracted)
    sub = assert_subproblem_is_exact(g, src, tgt, pieces, index,
                                     random_subsets(10, 8, 0), max_terminals=0)
    assert not sub.contracted
    assert sub.graph.number_of_nodes() == g.number_of_nodes()


# --- the per-piece QUBO ------------------------------------------------------

def test_nip_to_qubo_default_is_every_arc_interdictable():
    g, src, tgt = load_testcase('tiny_b.flow')
    default = ni.nip_to_qubo(2, g, src, tgt, 6.0, 7.0)
    explicit = ni.nip_to_qubo(2, g, src, tgt, 6.0, 7.0,
                              interdictable=list(g.edges()))
    assert default == explicit


def test_fixed_arcs_get_no_variables():
    """
    tiny_a with only (0,1) interdictable: x/gamma/w for arc 0 of the sorted
    list, pi for the two inner nodes, and nothing for the other three arcs.
    """
    g, src, tgt = load_testcase('tiny_a.flow')
    bqm = ni.nip_to_qubo(1, g, src, tgt, 2.0, 3.0, interdictable=[(0, 1)])
    assert set(bqm.variables) == {'x_0', 'gamma_0', 'w_0', 'pi_1', 'pi_2'}


def test_interdictable_arcs_must_be_in_the_graph():
    g, src, tgt = load_testcase('tiny_a.flow')
    with pytest.raises(ValueError, match='not in the graph'):
        ni.nip_to_qubo(1, g, src, tgt, 2.0, 3.0, interdictable=[(3, 0)])


# dimod.ExactSolver enumerates all 2**n states, so a regression that bloats a
# QUBO would exhaust memory rather than fail; past this many variables the
# exact-sampling tests refuse instead.
MAX_EXACT_VARIABLES = 20


def exact_sample(bqm):
    assert len(bqm.variables) <= MAX_EXACT_VARIABLES, len(bqm.variables)
    return dimod.ExactSolver().sample(bqm)


def decode(bqm, di_g):
    """The arcs the exact ground state of `bqm` interdicts."""
    best = exact_sample(bqm).first.sample
    edges = sorted(di_g.edges())
    return [edges[i] for i in range(len(edges))
            if best.get(f'x_{i}', 0) == 1]


def test_restricted_qubo_ground_state_is_the_restricted_optimum():
    """
    tiny_a (paths of capacity 10 and 5) with only the 5-path's arcs
    interdictable at B=1: the best cut leaves 10, not the unrestricted 5.
    f = 15, 10, 10 for 0, 1, 2 cuts, so lambda must lie in (0, 5).
    """
    g, src, tgt = load_testcase('tiny_a.flow')
    bqm = ni.nip_to_qubo(1, g, src, tgt, 2.5, 3.5,
                         interdictable=[(0, 2), (2, 3)])
    chosen = decode(bqm, g)
    assert len(chosen) == 1 and chosen[0] in [(0, 2), (2, 3)]
    assert surviving_flow(g, src, tgt, chosen) == 10


def test_piece_qubo_ground_state_is_the_piece_optimum():
    """
    P1 of the four paths may cut (3,5) (saving 2) or (4,5) (saving 3):
    f = 9, 6, 4 for 0, 1, 2 cuts, so at B=1 lambda must lie in (2, 3).
    """
    g, src, tgt, pieces = build_four_paths()
    sub = dc.piece_subproblem(g, src, tgt, pieces, 1)
    chosen = [sub.arc_map[arc] for arc in decode(dc.piece_qubo(sub, 1, 2.5),
                                                sub.graph)]
    assert chosen == [(4, 5)]
    assert surviving_flow(g, src, tgt, chosen) == 6


@pytest.mark.parametrize('name', ['50.planarflow', '50_0.15.erdosflow'])
def test_every_piece_qubo_is_smaller_than_the_whole_graph_qubo(name):
    """
    The point of the task: each piece's QUBO has x/gamma/w only for its own
    arcs (x for exactly those), so it is smaller than the whole graph's even
    where the rest of the graph is left uncontracted (every piece of the
    Erdos-Renyi instance has too many terminals).
    """
    g, src, tgt = load_testcase(name)
    whole = len(ni.nip_to_qubo(4, g, src, tgt, 1.0, 2.0).variables)
    pieces = dc.partition_graph(g, 4)
    owners = dc.arc_owners(g, pieces)
    for index in range(len(pieces)):
        sub = dc.piece_subproblem(g, src, tgt, pieces, index)
        variables = dc.piece_qubo(sub, 1, 1.0).variables
        useful = [arc for arc, owner in owners.items() if owner == index
                  and arc[1] != src and arc[0] != tgt]
        assert sum(1 for v in variables if v.startswith('x_')) == len(useful)
        assert len(variables) < whole


def test_contraction_shrinks_the_piece_qubo():
    g, src, tgt = load_testcase('50.planarflow')
    pieces = dc.partition_graph(g, 4)
    compared = 0
    for index in range(len(pieces)):
        sub = dc.piece_subproblem(g, src, tgt, pieces, index)
        if sub.contracted:
            full = dc.piece_subproblem(g, src, tgt, pieces, index,
                                       max_terminals=0)
            assert len(dc.piece_qubo(sub, 1, 1.0).variables) < \
                len(dc.piece_qubo(full, 1, 1.0).variables)
            compared += 1
    assert compared >= 2


# --- solving and combining ---------------------------------------------------

def test_sequential_pieces_respond_to_earlier_ones():
    """
    Budgets [3, 1] on the four paths. P0 alone cuts (0,1), (0,2) and (0,4)
    (the first best triple), leaving path 3 and a flow of 2. Solved
    independently, P1 then cuts (4,5) -- best against the uninterdicted
    graph, but path 4 is already cut -- and the flow stays 2. Solved after
    P0, it cuts (3,5) and the flow drops to 0, the optimum for 4 cuts.
    """
    g, src, tgt, pieces = build_four_paths()
    arcs, flow = dc.solve_decomposed(g, src, tgt, pieces, [3, 1],
                                     dc.brute_force_piece_solver,
                                     mode='independent')
    assert sorted(arcs) == [(0, 1), (0, 2), (0, 4), (4, 5)]
    assert flow == 2

    arcs, flow = dc.solve_decomposed(g, src, tgt, pieces, [3, 1],
                                     dc.brute_force_piece_solver,
                                     mode='sequential')
    assert sorted(arcs) == [(0, 1), (0, 2), (0, 4), (3, 5)]
    assert flow == 0


@pytest.mark.parametrize('call', [
    lambda g, s, t, p: dc.partition_graph(g, 0),
    lambda g, s, t, p: dc.partition_graph(g, g.number_of_nodes() + 1),
    lambda g, s, t, p: dc.piece_subproblem(g, s, t, p, 0, [(1, 0)]),
    lambda g, s, t, p: dc.solve_decomposed(
        g, s, t, p, [1, 1], dc.brute_force_piece_solver, mode='parallel'),
    lambda g, s, t, p: dc.solve_decomposed(
        g, s, t, p, [1], dc.brute_force_piece_solver),
])
def test_bad_arguments_are_rejected(call):
    g, src, tgt, pieces = build_four_paths()
    with pytest.raises(ValueError):
        call(g, src, tgt, pieces)


def test_zero_budget_pieces_are_not_solved():
    g, src, tgt, pieces = build_four_paths()
    solved = []

    def record(sub, budget):
        solved.append(budget)
        return []

    dc.solve_decomposed(g, src, tgt, pieces, [0, 2], record)
    assert solved == [2]


@pytest.mark.parametrize('returned', [
    [(4, 5), (3, 5)],   # over the budget of 1
    [(0, 1)],           # another piece's arc
    [(4, 5), (4, 5)],   # the same arc twice
])
def test_a_piece_may_only_cut_its_own_arcs_within_budget(returned):
    g, src, tgt, pieces = build_four_paths()
    with pytest.raises(ValueError, match='its own arcs'):
        dc.solve_decomposed(g, src, tgt, pieces, [0, 1],
                            lambda sub, budget: returned)


def test_qubo_piece_solver_returns_the_ground_state_cut():
    g, src, tgt, pieces = build_four_paths()
    solver = dc.qubo_piece_solver(2.5, sampler=exact_sample)
    sub = dc.piece_subproblem(g, src, tgt, pieces, 1)
    assert solver(sub, 1) == [(4, 5)]


def test_qubo_piece_solver_repairs_samples_within_the_piece():
    """
    At lambda = 0.5 the ground state cuts both of P1's arcs, one over its
    budget; the repair keeps the better one. Candidates are restricted to
    P1's arcs, so it cannot reach for P0's.
    """
    g, src, tgt, pieces = build_four_paths()
    sub = dc.piece_subproblem(g, src, tgt, pieces, 1)
    assert len(decode(dc.piece_qubo(sub, 1, 0.5), sub.graph)) == 2

    solver = dc.qubo_piece_solver(0.5, sampler=exact_sample)
    assert solver(sub, 1) == [(4, 5)]


def test_improve_dwave_solution_adds_only_candidate_edges():
    """tiny_a: unrestricted, greedy cuts the 10-path; limited to the 5-path's
    arcs, it has to cut one of those."""
    g, src, tgt = load_testcase('tiny_a.flow')
    edges, flow = ni.improve_dwave_solution(g, [0, 0, 0, 0], src, tgt, 1)
    assert flow == 5
    edges, flow = ni.improve_dwave_solution(
        g, [0, 0, 0, 0], src, tgt, 1, candidate_edges=[(0, 2), (2, 3)])
    assert edges in ([(0, 2)], [(2, 3)])
    assert flow == 10


def test_annealed_decomposition_stays_within_each_piece_budget():
    g, src, tgt = load_testcase('50.planarflow')
    pieces = dc.partition_graph(g, 2)
    budgets = dc.split_budget(g, src, tgt, 4, pieces, 'local')
    sampler = neal.SimulatedAnnealingSampler()
    solver = dc.qubo_piece_solver(
        30.0, sampler=lambda bqm: sampler.sample(bqm, num_reads=20, seed=7))
    arcs, flow = dc.solve_decomposed(g, src, tgt, pieces, budgets, solver)

    owners = dc.arc_owners(g, pieces)
    for index, budget in enumerate(budgets):
        assert sum(1 for arc in arcs if owners[arc] == index) == budget
    assert flow == surviving_flow(g, src, tgt, arcs)
    assert flow < surviving_flow(g, src, tgt)


def test_cli_prints_one_row_per_piece(capsys):
    assert dc.main([os.path.join(REPO_ROOT, 'testcases', '50.planarflow'),
                    '--pieces', '2', '--budget', '4']) == 0
    lines = capsys.readouterr().out.splitlines()
    whole = int(lines[0].split('whole-graph QUBO ')[1].split()[0])
    assert whole == 912
    rows = [line.split() for line in lines[3:]]
    assert [int(row[0]) for row in rows] == [0, 1]
    assert sum(int(row[3]) for row in rows) == 4
    assert all(0 < int(row[7]) < whole for row in rows)

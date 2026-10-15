"""
Decomposition-based refinement for larger NIP instances (issue #14).

The whole-graph QUBO nip_to_qubo() builds has V + 3E - 2 variables, and that
is what caps the instances a NISQ device can take. This module splits an
instance into pieces whose QUBOs are smaller, following the approach sketched
on slides 11-12 of the S-T_NIP deck:

1. partition_graph() cuts the unweighted graph into contiguous pieces.
2. Every arc is owned by the piece its tail lies in, so a separator arc
   belongs to the piece its source is in (arc_owners()).
3. split_budget() divides the budget B between the pieces by one of the
   slides' two rules: in proportion to the arcs of the global minimum cut
   each piece owns ('mincut'), or from each piece's local max flow with a
   super-source/super-sink attached to its boundary vertices ('local'; read
   here as in proportion to it). A third rule, 'marginal', solves each
   piece for 0..B cuts first and gives the budget to the pieces whose cuts
   save the most (marginal_split()). 'multiplier' instead tunes one shared
   budget price by subgradient ascent on those same per-piece curves.
4. piece_subproblem() builds the network one piece is solved on. Only the
   piece's own arcs may be interdicted there, so the rest of the graph
   carries no budget and is contracted exactly (contract_zone());
   piece_qubo() turns it into a QUBO in which only the piece's arcs carry
   x/gamma/w variables. The slide prototype kept the whole graph and only
   restricted which arcs could be cut, which left the QUBO its full size.
5. solve_decomposed() solves every piece that has a budget and combines the
   interdictions: independently, sequentially, or 'iterated' (re-solving
   each piece against the others until nothing improves).
6. refine_split() moves budget between pieces while that lowers the flow,
   and partition_to_fit() picks the fewest pieces whose QUBOs fit a device
   (156 qubits for IBM Heron).

What is exact and what is not
-----------------------------
Each piece's subproblem is exact: for every set X of the piece's arcs, the
reduced network with X removed has the same max flow as the original graph
with X removed (and with whatever earlier pieces interdicted). A per-piece
QUBO whose ground state is its optimum -- lambda inside that piece's window,
see #2 -- therefore solves the piece exactly.

The decomposition as a whole is a heuristic. A common budget price optimizes
only a separable approximation to the allocation, and nonconvex curves can
leave a duality gap even for that approximation. 'marginal' solves that
allocation exactly by dynamic programming; 'multiplier' retains its own
price history and surrogate gap. Neither gives a bound on global NIP flow.
refine_split() repairs what treating the pieces' savings as additive gets
wrong. run_decomposition_benchmark.py measures every rule against the exact
optimum; the saved three-case comparison also uses the Benders sample.
Full-cohort and real-network comparisons remain under #14/#15 and #1.

Command line
------------
Print the per-piece QUBO sizes next to the whole-graph size:

    python decomposition.py testcases/50.planarflow --pieces 4 --budget 12
"""
import argparse
import itertools
import math
import numbers
import sys
from dataclasses import dataclass
from fractions import Fraction

import networkx as nx

import network_interdiction as ni
import solve_qubo_via_dwave

SPLIT_STRATEGIES = ('mincut', 'local', 'marginal', 'multiplier')
SOLVE_MODES = ('sequential', 'independent', 'iterated')

# Rounds of re-solving every piece in 'iterated' mode, and budget moves
# refine_split() tries, before stopping even without convergence.
DEFAULT_MAX_ROUNDS = 10
DEFAULT_MAX_MOVES = 20

# Free terminals past which contract_zone() leaves a zone as it is. The exact
# contraction solves one minimum cut per source/sink labelling of the zone's
# terminals, so its cost doubles with each terminal: 2**12 = 4096 cuts take
# 2-9 s on the testcases, for zones of 25-94 nodes.
DEFAULT_MAX_TERMINALS = 12

# The same count of lowest-energy samples qubo_verify_parameters() repairs.
DEFAULT_REPAIRED_SAMPLES = 20

# Node labels for the auxiliary source and sink of the local networks. Flow
# files number their nodes, so strings cannot collide with a real node.
_SUPER_SOURCE = 'super_source'
_SUPER_SINK = 'super_sink'


@dataclass
class PieceSubproblem:
    """
    The network one piece is solved on (built by piece_subproblem()).

    graph           reduced network, nodes numbered 0..n-1 as nip_to_qubo()
                    requires, capacities in the 'weight' attribute
    src, tgt        source and sink in `graph`
    interdictable   arcs of `graph` the piece may interdict: its own arcs
    arc_map         each interdictable arc -> the original arc it stands for
    free_terminals  terminals of the rest of the graph other than the
                    source/sink: the contraction solves 2**free_terminals
                    minimum cuts
    contracted      whether the rest of the graph was contracted (False when
                    it had more than max_terminals free terminals)
    """
    graph: nx.DiGraph
    src: int
    tgt: int
    interdictable: list
    arc_map: dict
    free_terminals: int
    contracted: bool


def partition_graph(di_g, num_pieces, seed=0):
    """
    Split the nodes of `di_g` into `num_pieces` contiguous pieces.

    The graph is partitioned unweighted and undirected, as on the slides:
    adjacency matters, capacities and arc directions do not. The largest
    piece is bisected with networkx's Kernighan-Lin heuristic, which keeps
    the number of edges between the halves small, until there are
    `num_pieces` pieces; then every piece the bisection left in several
    parts is made contiguous by handing its smaller parts to the
    neighbouring piece they share the most edges with. Bisection halves the
    largest piece, so a count that is not a power of two gives pieces of
    unequal size. (Any other partition, METIS's for instance, can be passed
    to the rest of this module instead.)

    Returns a list of node sets ordered by their smallest node, reproducible
    from `seed`.
    """
    num_nodes = di_g.number_of_nodes()
    if not 1 <= num_pieces <= num_nodes:
        raise ValueError(
            f"num_pieces must be between 1 and {num_nodes}, got {num_pieces}")
    undirected = di_g.to_undirected()
    pieces = [set(di_g.nodes())]
    while len(pieces) < num_pieces:
        pieces.sort(key=lambda piece: (-len(piece), min(piece)))
        largest = pieces.pop(0)
        half_a, half_b = nx.community.kernighan_lin_bisection(
            undirected.subgraph(largest), weight=None, seed=seed)
        pieces.extend([set(half_a), set(half_b)])
    return sorted(_make_contiguous(undirected, pieces), key=min)


def _make_contiguous(undirected, pieces):
    """
    Move each part of a piece other than its largest to the piece it shares
    the most edges with (ties to the lower index), until every piece is
    connected. Each move merges the part into a neighbour, so the total
    number of parts falls and the loop ends. A part with no edge to another
    piece -- possible only if the graph itself is disconnected -- stays.
    """
    pieces = [set(piece) for piece in pieces]
    piece_of = {node: index for index, piece in enumerate(pieces)
                for node in piece}
    moved = True
    while moved:
        moved = False
        for index in range(len(pieces)):
            parts = sorted(nx.connected_components(
                undirected.subgraph(pieces[index])),
                key=lambda part: (-len(part), min(part)))
            for part in parts[1:]:
                shared = {}
                for node in part:
                    for neighbour in undirected[node]:
                        other = piece_of[neighbour]
                        if other != index:
                            shared[other] = shared.get(other, 0) + 1
                if not shared:
                    continue
                target = max(sorted(shared), key=lambda other: shared[other])
                pieces[index] -= part
                pieces[target] |= part
                for node in part:
                    piece_of[node] = target
                moved = True
    return pieces


def _piece_index(di_g, pieces):
    """Map every node to the index of its piece; `pieces` must partition the nodes."""
    piece_of = {}
    for index, piece in enumerate(pieces):
        if not piece:
            raise ValueError(f"piece {index} is empty")
        for node in piece:
            if node in piece_of:
                raise ValueError(
                    f"node {node} is in pieces {piece_of[node]} and {index}")
            piece_of[node] = index
    if set(piece_of) != set(di_g.nodes()):
        raise ValueError("pieces do not cover exactly the nodes of the graph")
    return piece_of


def arc_owners(di_g, pieces):
    """
    The index of the piece owning each arc: the piece its tail lies in, so an
    arc between two pieces belongs to the piece its source is in.
    """
    piece_of = _piece_index(di_g, pieces)
    return {(u, v): piece_of[u] for u, v in di_g.edges()}


def mincut_arc_counts(di_g, src_node, tgt_node, pieces):
    """
    How many arcs of a minimum s-t cut of the uninterdicted graph each piece
    owns. The cut is the one networkx returns, whose source side holds every
    node that cannot reach the sink in the residual network -- a unique
    choice, so the counts do not depend on the max-flow algorithm.
    """
    owners = arc_owners(di_g, pieces)
    _, (source_side, _) = nx.minimum_cut(di_g, src_node, tgt_node,
                                         capacity='weight')
    counts = [0] * len(pieces)
    for u, v in di_g.edges():
        if u in source_side and v not in source_side:
            counts[owners[(u, v)]] += 1
    return counts


def local_max_flows(di_g, src_node, tgt_node, pieces):
    """
    Each piece's max flow with the rest of the graph replaced by a
    super-source and a super-sink attached to its boundary vertices.

    The super-source feeds every vertex of the piece that an arc enters from
    outside, with the total capacity of those arcs, and the source itself
    with unlimited capacity if the piece holds it; every vertex an arc leaves
    the piece from drains to the super-sink with the total capacity of those
    arcs, as does the sink with unlimited capacity. Arcs into the source and
    out of the sink carry no s-t flow and are left out, and so are arcs that
    enter the piece at the sink itself: they cross no arc the piece owns, so
    the piece could not cut that flow (their tail's piece counts them as its
    outflow). A local max flow is an estimate: it assumes the rest of the
    graph can supply and absorb whatever the boundary arcs carry, and it
    goes stale as soon as a neighbouring piece is interdicted.
    """
    piece_of = _piece_index(di_g, pieces)
    flows = []
    for index, piece in enumerate(pieces):
        local = nx.DiGraph()
        local.add_nodes_from(piece)
        local.add_nodes_from([_SUPER_SOURCE, _SUPER_SINK])
        for u, v, data in di_g.edges(data=True):
            if v == src_node or u == tgt_node:
                continue
            inside_u = piece_of[u] == index
            if v == tgt_node and not inside_u:
                continue
            inside_v = piece_of[v] == index
            if inside_u and inside_v:
                local.add_edge(u, v, weight=data['weight'])
            elif inside_v:
                _add_capacity(local, _SUPER_SOURCE, v, data['weight'])
            elif inside_u:
                _add_capacity(local, u, _SUPER_SINK, data['weight'])
        # No 'weight' attribute: networkx reads the arc as uncapacitated.
        if src_node in piece:
            local.add_edge(_SUPER_SOURCE, src_node)
        if tgt_node in piece:
            local.add_edge(tgt_node, _SUPER_SINK)
        flows.append(nx.maximum_flow_value(local, _SUPER_SOURCE, _SUPER_SINK,
                                           capacity='weight'))
    return flows


def _add_capacity(di_g, u, v, capacity):
    if di_g.has_edge(u, v):
        di_g[u][v]['weight'] += capacity
    else:
        di_g.add_edge(u, v, weight=capacity)


def apportion(B, weights):
    """
    Split the integer B in proportion to `weights` by largest remainder: each
    piece gets the whole part of its quota, and the units left over go to
    the largest fractional parts (ties to the larger weight, then the lower
    index). All zero when every weight is zero, since there is then nothing
    to spend the budget on.
    """
    if B < 0:
        raise ValueError(f"budget must be non-negative, got {B}")
    total = sum(weights)
    if total <= 0:
        return [0] * len(weights)
    quotas = [Fraction(B) * Fraction(weight) / Fraction(total)
              for weight in weights]
    shares = [int(quota) for quota in quotas]
    order = sorted(range(len(weights)),
                   key=lambda i: (-(quotas[i] - shares[i]), -weights[i], i))
    for i in order[:B - sum(shares)]:
        shares[i] += 1
    return shares


def split_budget(di_g, src_node, tgt_node, B, pieces, strategy='mincut',
                 solve_piece=None, max_terminals=DEFAULT_MAX_TERMINALS,
                 multiplier_iterations=100):
    """
    Split the budget B between `pieces`: one integer per piece, summing to B
    unless no piece has any weight (then all zero, see apportion()).

    'mincut'    in proportion to the arcs of the global minimum cut each
                piece owns (mincut_arc_counts())
    'local'     in proportion to each piece's local max flow
                (local_max_flows()). The slides derive this split from the
                local max flows; allocating in proportion to them is this
                module's reading of that rule.
    'marginal'  by what each piece's cuts are worth: marginal_split(), with
                `solve_piece` (gurobi_piece_solver by default) measuring
                every piece's best flow for 0..B cuts
    'multiplier' the same curves, but allocated by one subgradient-tuned
                price; common_multiplier_split() also returns its history

    'mincut' and 'local' are proportional rules decided from the graph
    alone; 'marginal' and 'multiplier' solve the pieces first. All are decided before
    the pieces are combined; refine_split() revises a split afterwards.
    """
    if strategy == 'mincut':
        weights = mincut_arc_counts(di_g, src_node, tgt_node, pieces)
    elif strategy == 'local':
        weights = local_max_flows(di_g, src_node, tgt_node, pieces)
    elif strategy == 'marginal':
        budgets, _curves = marginal_split(
            di_g, src_node, tgt_node, B, pieces,
            solve_piece if solve_piece is not None else gurobi_piece_solver,
            max_terminals)
        return budgets
    elif strategy == 'multiplier':
        result = common_multiplier_split(
            di_g, src_node, tgt_node, B, pieces,
            solve_piece if solve_piece is not None else gurobi_piece_solver,
            max_terminals=max_terminals, max_iterations=multiplier_iterations)
        return result['budgets']
    else:
        raise ValueError(
            f"unknown split strategy {strategy!r}, expected one of "
            f"{', '.join(SPLIT_STRATEGIES)}")
    return apportion(B, weights)


def zone_terminals(di_g, src_node, tgt_node, zone):
    """
    The nodes through which `zone` meets the rest of the graph: zone nodes
    an arc crosses the zone's boundary at, plus the source and the sink if
    the zone holds them.
    """
    zone = set(zone)
    terminals = set()
    for u, v in di_g.edges():
        if (u in zone) != (v in zone):
            terminals.add(u if u in zone else v)
    return terminals | ({src_node, tgt_node} & zone)


def contract_zone(di_g, src_node, tgt_node, zone,
                  max_terminals=DEFAULT_MAX_TERMINALS):
    """
    Exact contraction of `zone`, a set of nodes none of whose internal arcs
    may be interdicted.

    Returns {node: class} over the zone, where the nodes of a class can be
    merged into one without changing the max flow of the graph under any
    interdiction of arcs outside the zone; or None when the zone has more
    than `max_terminals` free terminals (terminals other than the source and
    sink), since the check costs 2**free_terminals minimum cuts.

    Why it is exact: the zone meets the rest of the graph only at its
    terminals (zone_terminals()), so a minimum cut of the whole graph,
    restricted to the zone's internal arcs, is a minimum cut of the zone
    given the side each terminal is on. The zone therefore matters only
    through g(L), its minimum internal cut for each labelling L of the
    terminals as source side or sink side (the source always on the source
    side, the sink on the sink side). This solves the zone's minimum cut
    for every L, taking the cut with the largest source side (unique), and
    records which side each node lands on. Nodes that land on the same side
    in all of those cuts form a class. Every recorded cut survives merging
    the classes with its value intact, so the merged zone's minimum cut for
    L is at most g(L); merging can only restrict cuts, so it is at least
    g(L). The two are equal for every L, and so is the max flow, whatever is
    interdicted outside.

    This is the contraction #14 says is exact only for a piece with no
    budget: a zone whose arcs could be cut has a g that depends on the
    interdiction, so merging it would lose its internal structure. For a
    zone with two terminals, g is just the zone's max flow between them in
    each direction -- the "constant capacity" #14 mentions; with more
    terminals it is a small network over them rather than one number. The
    merged network is exact but not always the smallest possible one.
    Whether a zone gets contracted at all depends on its terminal count,
    i.e. on how small a separator the partition found: pieces of the k-NN
    testcases mostly stay under the default limit, pieces of the
    Erdos-Renyi ones (random graphs, no small separators) do not.
    """
    zone = set(zone)
    terminals = zone_terminals(di_g, src_node, tgt_node, zone)
    fixed_sources = {src_node} & zone
    fixed_sinks = {tgt_node} & zone
    free = sorted(terminals - fixed_sources - fixed_sinks)
    if len(free) > max_terminals:
        return None

    network = nx.DiGraph(di_g.subgraph(zone))
    network.add_nodes_from([_SUPER_SOURCE, _SUPER_SINK])
    ordered = sorted(zone)
    signature = {node: [] for node in ordered}
    for labels in itertools.product((True, False), repeat=len(free)):
        sources = fixed_sources | {node for node, is_source in zip(free, labels)
                                   if is_source}
        sinks = fixed_sinks | {node for node, is_source in zip(free, labels)
                               if not is_source}
        # No 'weight' attribute: the terminals are tied to their side with
        # uncapacitated arcs, so only zone arcs can be cut.
        attachments = ([(_SUPER_SOURCE, node) for node in sources]
                       + [(node, _SUPER_SINK) for node in sinks])
        network.add_edges_from(attachments)
        _, (source_side, _) = nx.minimum_cut(network, _SUPER_SOURCE,
                                             _SUPER_SINK, capacity='weight')
        network.remove_edges_from(attachments)
        for node in ordered:
            signature[node].append(node in source_side)

    classes = {}
    for node in ordered:
        classes.setdefault(tuple(signature[node]), node)
    return {node: classes[tuple(signature[node])] for node in ordered}


def piece_subproblem(di_g, src_node, tgt_node, pieces, index, interdicted=(),
                     max_terminals=DEFAULT_MAX_TERMINALS):
    """
    The network piece `index` is solved on, as a PieceSubproblem.

    Arcs in `interdicted` (what earlier pieces cut, when solving
    sequentially) are removed first, and so are arcs into the source and out
    of the sink, which never carry s-t flow. The piece keeps every node and
    arc; the rest of the graph -- no budget in this subproblem -- is merged
    by contract_zone(), or kept as it is past `max_terminals`. Either way the
    subproblem is exact: removing any set of the piece's arcs leaves the
    reduced network the max flow of the original graph with the same arcs
    removed.
    """
    piece = set(pieces[index])
    _piece_index(di_g, pieces)
    unknown = [arc for arc in interdicted if not di_g.has_edge(*arc)]
    if unknown:
        raise ValueError(f"interdicted arcs not in the graph: {unknown}")

    working = di_g.copy()
    working.remove_edges_from(interdicted)
    working.remove_edges_from(list(working.in_edges(src_node))
                              + list(working.out_edges(tgt_node)))

    zone = set(working.nodes()) - piece
    free_terminals = len(zone_terminals(working, src_node, tgt_node, zone)
                         - {src_node, tgt_node})
    classes = contract_zone(working, src_node, tgt_node, zone, max_terminals)
    contracted = classes is not None
    if not contracted:
        classes = {node: node for node in zone}

    # Number the merged nodes 0..n-1 in order of their smallest member, as
    # nip_to_qubo() indexes pi by node label.
    group = {node: ('piece', node) for node in piece}
    group.update({node: ('zone', classes[node]) for node in zone})
    smallest = {}
    for node, key in group.items():
        smallest[key] = min(node, smallest.get(key, node))
    label = {key: i for i, key in
             enumerate(sorted(smallest, key=lambda key: smallest[key]))}
    node_map = {node: label[group[node]] for node in working.nodes()}

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(label)))
    arc_map = {}
    for u, v, data in working.edges(data=True):
        a, b = node_map[u], node_map[v]
        if a == b:
            continue
        if u in piece:
            # The piece's own arcs never share an endpoint pair: their heads
            # outside the piece are terminals, and terminals never merge.
            assert not graph.has_edge(a, b), (u, v)
            arc_map[(a, b)] = (u, v)
        _add_capacity(graph, a, b, data['weight'])

    return PieceSubproblem(
        graph=graph, src=node_map[src_node], tgt=node_map[tgt_node],
        interdictable=sorted(arc_map), arc_map=arc_map,
        free_terminals=free_terminals, contracted=contracted)


def piece_qubo(sub, budget, cardinality_penalty_coeff, dual_penalty_coeff=None):
    """
    The QUBO for one piece: nip_to_qubo() on the reduced network with only
    the piece's arcs interdictable. `dual_penalty_coeff` defaults to
    cardinality_penalty_coeff + 1, the pairing the rest of the pipeline
    uses; it has to exceed the cardinality coefficient (#2). Its ground state
    is the piece's optimum when the cardinality coefficient lies in that
    piece's lambda window, which is the piece's own, not the whole graph's.
    """
    if dual_penalty_coeff is None:
        dual_penalty_coeff = cardinality_penalty_coeff + 1
    return ni.nip_to_qubo(budget, sub.graph, sub.src, sub.tgt,
                          cardinality_penalty_coeff, dual_penalty_coeff,
                          interdictable=sub.interdictable)


def _surviving_flow(di_g, src_node, tgt_node, interdicted):
    remaining = di_g.copy()
    remaining.remove_edges_from(interdicted)
    return nx.maximum_flow_value(remaining, src_node, tgt_node,
                                 capacity='weight')


def brute_force_piece_solver(sub, budget):
    """
    The piece's exact optimum, by trying every set of min(budget, arcs) of
    its arcs (cutting more never raises a max flow). Returns original arcs.
    Exhaustive, so only for small pieces -- a reference for the QUBO path.
    """
    size = min(budget, len(sub.interdictable))
    best_flow, best_arcs = None, ()
    for arcs in itertools.combinations(sub.interdictable, size):
        flow = _surviving_flow(sub.graph, sub.src, sub.tgt, arcs)
        if best_flow is None or flow < best_flow:
            best_flow, best_arcs = flow, arcs
    return [sub.arc_map[arc] for arc in best_arcs]


def gurobi_piece_solver(sub, budget):
    """
    The piece's exact optimum from the NIP MILP on its reduced network with
    only its arcs interdictable (Gurobi license required). It scales to
    pieces brute force cannot, so it is the exact per-piece reference for
    the benchmarks: with it, the only error left is the decomposition's.
    Requests zero MIP gap and requires OPTIMAL, subject to Gurobi's numerical
    tolerances. Interrupted or incomplete solves do not supply exact curves.
    Returns original arcs.
    """
    size = min(budget, len(sub.interdictable))
    if size == 0:
        return []
    model, x = ni._build_nip_milp(sub.graph, sub.src, sub.tgt, size,
                                  sub.interdictable)
    with model:
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 0
        model.optimize()
        if model.Status != ni.GRB.OPTIMAL:
            raise RuntimeError(f'piece oracle did not prove optimality: status {model.Status}')
        return [sub.arc_map[arc] for arc in sub.interdictable if x[arc].X > 0.5]


def piece_curves(di_g, src_node, tgt_node, B, pieces, solve_piece,
                 interdicted=(), max_terminals=DEFAULT_MAX_TERMINALS):
    """
    For every piece, its best surviving flow (of the whole graph) when only
    that piece cuts k = 0, 1, ... of its arcs, up to B or its arc count,
    with `interdicted` already removed. curve[0] is the same for every piece.
    """
    owned = [0] * len(pieces)
    for owner in arc_owners(di_g, pieces).values():
        owned[owner] += 1
    base = _surviving_flow(di_g, src_node, tgt_node, interdicted)
    curves = []
    for index, piece_owned in enumerate(owned):
        sub = piece_subproblem(di_g, src_node, tgt_node, pieces, index,
                               interdicted, max_terminals)
        curve = [base]
        for k in range(1, min(B, len(sub.interdictable)) + 1):
            cut = list(interdicted) + list(solve_piece(sub, k))
            curve.append(_surviving_flow(di_g, src_node, tgt_node, cut))
        curves.append(curve)
    return curves


def marginal_split(di_g, src_node, tgt_node, B, pieces, solve_piece,
                   max_terminals=DEFAULT_MAX_TERMINALS):
    """
    Split B by what each piece's cuts are worth (#14's common-multiplier
    task). Each piece's saving s_P(k) = f(0) - f_P(k) is measured with
    piece_curves(), and the split maximizes the summed savings
    sum_P s_P(B_P) subject to sum_P B_P = B (or every arc, if fewer), by
    dynamic programming over the pieces.

    One multiplier for all pieces, implemented by common_multiplier_split(),
    relaxes this allocation through its Lagrangian; convex curves have no
    duality gap, while nonconvex curves can. This function uses exact dynamic
    programming instead. Treating savings as additive is the heuristic part: two pieces
    that cut the same flow path save less together than apart, which
    'iterated' solving and refine_split() correct afterwards.

    Returns (budgets, curves).
    """
    curves = piece_curves(di_g, src_node, tgt_node, B, pieces, solve_piece,
                          max_terminals=max_terminals)
    return allocate_budget(curves, B), curves


def allocate_budget(curves, B):
    """
    The split maximizing sum_P (curve_P[0] - curve_P[B_P]) subject to
    sum_P B_P = min(B, sum of the curves' lengths - 1), by dynamic
    programming over the pieces: exact for any curves, convex or not.
    Ties keep the allocation found first, which favours earlier pieces
    taking fewer cuts.
    """
    total = min(B, sum(len(curve) - 1 for curve in curves))
    # best[b] = (summed saving, budgets) over the pieces seen so far with b
    # cuts used in all.
    best = {0: (0, [])}
    for curve in curves:
        saving = [curve[0] - flow for flow in curve]
        step = {}
        for used, (value, budgets) in best.items():
            for k in range(min(len(curve) - 1, total - used) + 1):
                candidate = (value + saving[k], budgets + [k])
                if used + k not in step or candidate[0] > step[used + k][0]:
                    step[used + k] = candidate
        best = step
    return best[total][1]


def _validate_multiplier_limits(B, max_iterations):
    if not isinstance(B, numbers.Integral) or isinstance(B, bool) or B < 0:
        raise ValueError('budget must be a nonnegative integer')
    if (not isinstance(max_iterations, numbers.Integral)
            or isinstance(max_iterations, bool) or max_iterations < 1):
        raise ValueError('max_iterations must be a positive integer')


def allocate_budget_multiplier(curves, B, *, max_iterations=100):
    """Tune one budget price for the separable per-piece allocation surrogate.

    Minimize sum_i f_i(k_i), sum_i k_i <= B, over the supplied finite,
    nonincreasing curves. At price lambda >= 0, each piece independently
    minimizes f_i(k) + lambda*k. Their total count minus B is a supergradient
    of the concave dual. Projected ascent uses M/(G*sqrt(t)), where M is
    the largest marginal drop and G bounds the count violation. A price
    above M selects zero cuts, so [0,M] contains a dual optimum.

    Each requested allocation is repaired deterministically to the available
    budget, retaining the best feasible surrogate value and best dual bound.
    Integer/binary-float inputs and prices are evaluated with Fraction; the
    exported lower bound rounds down, and the feasible value rounds up.
    `optimal_surrogate` requires an exactly closed rational gap. Nonconvex
    curves can have a positive duality gap even after tuning converges.

    THESE ARE NOT BOUNDS ON GLOBAL INTERDICTION. Per-piece savings overlap;
    the combined cut must still be scored on the actual original network.
    """
    _validate_multiplier_limits(B, max_iterations)
    exact = []
    for curve in curves:
        if not len(curve):
            raise ValueError('each curve must contain at least its zero-budget flow')
        try:
            valid = all(math.isfinite(value) and value >= 0 for value in curve)
        except (TypeError, OverflowError):
            valid = False
        if not valid:
            raise ValueError('curve values must be finite and nonnegative')
        values = [Fraction(int(value)) if isinstance(value, numbers.Integral)
                  else Fraction(float(value)) for value in curve]
        if any(a < b for a, b in zip(values, values[1:])):
            raise ValueError('flow curves must be nonincreasing')
        exact.append(values)
    if sum((curve[0] for curve in exact), Fraction(0)) > Fraction(sys.float_info.max):
        raise ValueError('summed surrogate flow exceeds the finite export range')
    capacity = sum(len(curve) - 1 for curve in exact)
    target = min(int(B), capacity)
    scale = max((a - b for curve in exact for a, b in zip(curve, curve[1:])),
                default=Fraction(0))
    violation_bound = max(1, target, capacity - target)
    price = scale if target == 0 else Fraction(0)

    def rounded(value, direction):
        result = float(value)
        if (direction < 0 and Fraction(result) > value
                or direction > 0 and Fraction(result) < value):
            result = math.nextafter(result, direction * math.inf)
        return result

    def repair(requested):
        allocation = requested.copy()
        while sum(allocation) > target:
            i = min((i for i, k in enumerate(allocation) if k),
                    key=lambda i: (exact[i][allocation[i]-1] - exact[i][allocation[i]], i))
            allocation[i] -= 1
        while sum(allocation) < target:
            i = min((i for i, k in enumerate(allocation) if k+1 < len(exact[i])),
                    key=lambda i: (exact[i][allocation[i]+1] - exact[i][allocation[i]], i))
            allocation[i] += 1
        return allocation

    best_value, best_lower = None, Fraction(-1)
    best_allocation, allocation_price, bound_price = None, None, None
    history = []
    status = 'iteration_limit'
    for iteration in range(1, max_iterations + 1):
        requested = [min(range(len(curve)), key=lambda k: (curve[k] + price*k, k))
                     for curve in exact]
        violation = sum(requested) - target
        dual = sum((curve[k] + price*k for curve, k in zip(exact, requested)),
                   Fraction(0)) - price*target
        # Nonnegativity is another valid bound and avoids exporting a very
        # negative dual value outside the float range at an excessive price.
        dual = max(Fraction(0), dual)
        allocation = repair(requested)
        value = sum((curve[k] for curve, k in zip(exact, allocation)), Fraction(0))
        if best_value is None or value < best_value:
            best_value, best_allocation, allocation_price = value, allocation, price
        if dual > best_lower:
            best_lower, bound_price = dual, price
        history.append({'iteration': iteration, 'multiplier': float(price),
                        'multiplier_exact': str(price),
                        'requested_budgets': requested, 'budget_violation': violation,
                        'repaired_budgets': allocation,
                        'surrogate_value': rounded(value, 1),
                        'surrogate_lower_bound': rounded(dual, -1),
                        'best_surrogate_value': rounded(best_value, 1),
                        'best_surrogate_lower_bound': rounded(best_lower, -1)})
        if best_value == best_lower:
            status = 'optimal_surrogate'
            break
        step = Fraction(float(scale) / (violation_bound * math.sqrt(iteration)))
        price = min(scale, max(Fraction(0), price + step*violation))
    return {'budgets': best_allocation, 'target_budget': target,
            'surrogate_value': rounded(best_value, 1),
            'surrogate_lower_bound': rounded(best_lower, -1),
            'allocation_multiplier': float(allocation_price),
            'bound_multiplier': float(bound_price),
            'status': status, 'iterations': len(history), 'history': history}


def common_multiplier_split(di_g, src_node, tgt_node, B, pieces, solve_piece,
                            max_terminals=DEFAULT_MAX_TERMINALS, *, max_iterations=100):
    """Measure per-piece curves, then tune one common allocation price.

    Returns allocation diagnostics plus the actual curves. The default
    piece solver used by split_budget is exact; with a heuristic solver the
    bounds describe only the supplied measured curves. All curve solves
    happen before price tuning and must be included in allocation timing.
    """
    _validate_multiplier_limits(B, max_iterations)
    curves = piece_curves(di_g, src_node, tgt_node, B, pieces, solve_piece,
                          max_terminals=max_terminals)
    result = allocate_budget_multiplier(curves, B, max_iterations=max_iterations)
    result['curves'] = curves
    return result


def refine_split(di_g, src_node, tgt_node, pieces, budgets, solve_piece,
                 mode='iterated', max_moves=DEFAULT_MAX_MOVES,
                 max_terminals=DEFAULT_MAX_TERMINALS):
    """
    Improve a split by moving one unit of budget between two pieces at a
    time, keeping the best move while it lowers the combined surviving
    flow (solve_decomposed() in `mode`). Stops when no move helps or after
    `max_moves` moves. Each round solves the decomposition once per ordered
    pair of pieces, so it is meant for a handful of pieces.

    Returns (budgets, interdicted arcs, surviving flow).
    """
    owned = [0] * len(pieces)
    for owner in arc_owners(di_g, pieces).values():
        owned[owner] += 1

    def evaluate(split):
        return solve_decomposed(di_g, src_node, tgt_node, pieces, split,
                                solve_piece, mode, max_terminals)

    budgets = list(budgets)
    interdicted, flow = evaluate(budgets)
    for _ in range(max_moves):
        best = None
        for give, take in itertools.permutations(range(len(pieces)), 2):
            if budgets[give] == 0 or budgets[take] >= owned[take]:
                continue
            candidate = list(budgets)
            candidate[give] -= 1
            candidate[take] += 1
            cut, cand_flow = evaluate(candidate)
            if cand_flow < flow and (best is None or cand_flow < best[2]):
                best = (candidate, cut, cand_flow)
        if best is None:
            break
        budgets, interdicted, flow = best
    return budgets, interdicted, flow


def piece_qubo_size(sub):
    """
    Variables of the piece's QUBO without building it: one pi per node of
    the reduced network other than the source and sink, and x, gamma and w
    for each arc the piece may cut (piece_qubo() gives the rest no
    variables). 0 when the piece has no arcs to cut.
    """
    if not sub.interdictable:
        return 0
    return sub.graph.number_of_nodes() - 2 + 3 * len(sub.interdictable)


def partition_to_fit(di_g, src_node, tgt_node, max_qubits, seed=0,
                     max_pieces=None, max_terminals=DEFAULT_MAX_TERMINALS):
    """
    The fewest pieces whose QUBOs all fit `max_qubits` variables (156 for an
    IBM Heron device), trying 1, 2, ... pieces with partition_graph().

    A piece's QUBO keeps one pi per node of its reduced network, so a piece
    whose surroundings cannot be contracted (more than `max_terminals`
    terminals) carries most of the graph's nodes however few arcs it owns,
    and may never fit; that raises ValueError, as does reaching
    `max_pieces` (default: the node count).

    Returns (pieces, sizes).
    """
    limit = max_pieces or di_g.number_of_nodes()
    # Some piece owns at least E / count arcs, 3 variables each, so fewer
    # than 3E / max_qubits pieces cannot fit; start there.
    first = max(1, -(-3 * di_g.number_of_edges() // max_qubits))
    for count in range(min(first, limit), limit + 1):
        pieces = partition_graph(di_g, count, seed=seed)
        sizes = [piece_qubo_size(piece_subproblem(
            di_g, src_node, tgt_node, pieces, index,
            max_terminals=max_terminals)) for index in range(len(pieces))]
        if max(sizes) <= max_qubits:
            return pieces, sizes
    raise ValueError(f'no partition into at most {limit} pieces fits '
                     f'{max_qubits} qubits')


def qubo_piece_solver(cardinality_penalty_coeff, dual_penalty_coeff=None,
                      sampler=None, num_samples=DEFAULT_REPAIRED_SAMPLES):
    """
    A piece solver that samples the piece's QUBO.

    `sampler(bqm)` returns a dimod SampleSet; by default the simulated
    annealer the rest of the pipeline uses (solve_bqm_dwave(bqm, 'sa')). The
    `num_samples` lowest-energy samples are decoded and each is repaired to
    min(budget, arcs) of the piece's own arcs by improve_dwave_solution(),
    the same protocol qubo_verify_parameters() follows for the whole graph;
    the best repaired sample is returned as original arcs.
    """
    if sampler is None:
        def sampler(bqm):
            return solve_qubo_via_dwave.solve_bqm_dwave(bqm, 'sa')

    def solve_piece(sub, budget):
        target = min(budget, len(sub.interdictable))
        if target == 0:
            return []
        bqm = piece_qubo(sub, budget, cardinality_penalty_coeff,
                         dual_penalty_coeff)
        sampleset = sampler(bqm)
        edge_list = sorted(sub.graph.edges())
        best_flow, best_arcs = None, []
        for datum in itertools.islice(
                sampleset.data(fields=['sample'], sorted_by='energy'),
                num_samples):
            x = [int(datum.sample.get(f'x_{i}', 0))
                 for i in range(len(edge_list))]
            arcs, flow = ni.improve_dwave_solution(
                sub.graph, x, sub.src, sub.tgt, target,
                candidate_edges=sub.interdictable)
            if best_flow is None or flow < best_flow:
                best_flow, best_arcs = flow, arcs
        return [sub.arc_map[arc] for arc in best_arcs]

    return solve_piece


def solve_decomposed(di_g, src_node, tgt_node, pieces, budgets, solve_piece,
                     mode='sequential', max_terminals=DEFAULT_MAX_TERMINALS,
                     max_rounds=DEFAULT_MAX_ROUNDS, flow_oracle=None):
    """
    Solve every piece with a positive budget and combine the interdictions.

    `solve_piece(sub, budget)` gets a PieceSubproblem and returns the
    original arcs to cut -- brute_force_piece_solver, or the function
    qubo_piece_solver() returns. Pieces are solved in list order.

    'sequential'   each piece is solved with the arcs earlier pieces chose
                   already removed, so it responds to them
    'independent'  every piece is solved against the uninterdicted graph,
                   so the pieces can be solved in parallel (on separate
                   devices, say) and their choices are only combined after
    'iterated'     'sequential', then every piece is re-solved against all
                   the other pieces' current cuts, keeping its new cuts when
                   they lower the combined flow, until a full round changes
                   nothing (at most `max_rounds` rounds). Sequential solving
                   only lets a piece answer the pieces before it; this lets
                   the early pieces answer the later ones too.

    Optional `flow_oracle(cut)` supplies the same whole-network score, allowing
    benchmark callers to count and limit scoring work. Exceptions propagate.
    Returns (interdicted arcs, max flow of `di_g` without them).
    """
    score = (flow_oracle if flow_oracle is not None else
             lambda cut: _surviving_flow(di_g, src_node, tgt_node, cut))
    if mode not in SOLVE_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of "
                         f"{', '.join(SOLVE_MODES)}")
    if len(budgets) != len(pieces):
        raise ValueError(f"{len(budgets)} budgets for {len(pieces)} pieces")
    owners = arc_owners(di_g, pieces)

    def solve(index, fixed):
        budget = budgets[index]
        sub = piece_subproblem(di_g, src_node, tgt_node, pieces, index, fixed,
                               max_terminals)
        chosen = list(solve_piece(sub, budget))
        if len(chosen) > budget or len(set(chosen)) != len(chosen) or any(
                owners.get(arc) != index for arc in chosen):
            raise ValueError(
                f"piece {index} returned {chosen}: a piece may cut at most "
                f"its budget ({budget}) of its own arcs")
        return chosen

    active = [index for index, budget in enumerate(budgets) if budget > 0]
    interdicted = []
    for index in active:
        fixed = () if mode == 'independent' else interdicted
        interdicted.extend(solve(index, fixed))
    flow = score(interdicted)
    if mode != 'iterated':
        return interdicted, flow

    for _ in range(max_rounds):
        improved = False
        for index in active:
            others = [arc for arc in interdicted if owners[arc] != index]
            candidate = others + solve(index, others)
            candidate_flow = score(candidate)
            if candidate_flow < flow:
                interdicted, flow, improved = candidate, candidate_flow, True
        if not improved:
            break
    return interdicted, flow


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Partition an NIP instance, split its budget, and '
                    'compare each piece\'s QUBO size with the whole graph\'s.')
    parser.add_argument('instance', help='flow file (see read_flow_file)')
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument('--pieces', type=int,
                      help='number of pieces to partition into')
    size.add_argument('--fit-qubits', type=int,
                      help='use the fewest pieces whose QUBOs fit this many '
                           'qubits (156 for IBM Heron)')
    parser.add_argument('--budget', type=int, required=True,
                        help='interdiction budget B to split')
    parser.add_argument('--split', choices=SPLIT_STRATEGIES, default='mincut',
                        help='budget split rule (default: mincut)')
    parser.add_argument('--multiplier-iterations', type=int, default=100,
                        help='shared-price updates for the multiplier split (default 100)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for the partition (default: 0)')
    parser.add_argument('--max-terminals', type=int,
                        default=DEFAULT_MAX_TERMINALS,
                        help='contract the rest of the graph only up to this '
                             'many free terminals (default: '
                             f'{DEFAULT_MAX_TERMINALS})')
    args = parser.parse_args(argv)
    if args.split == 'multiplier' and args.multiplier_iterations < 1:
        parser.error('--multiplier-iterations must be positive')

    di_g, src_node, tgt_node = ni.read_flow_file(args.instance)
    if args.fit_qubits:
        pieces, _sizes = partition_to_fit(di_g, src_node, tgt_node,
                                          args.fit_qubits, seed=args.seed,
                                          max_terminals=args.max_terminals)
        args.pieces = len(pieces)
    else:
        pieces = partition_graph(di_g, args.pieces, seed=args.seed)
    budgets = split_budget(di_g, src_node, tgt_node, args.budget, pieces,
                           args.split, max_terminals=args.max_terminals,
                           multiplier_iterations=args.multiplier_iterations)
    owners = arc_owners(di_g, pieces)
    # The coefficients do not change which variables a QUBO has.
    whole = len(ni.nip_to_qubo(args.budget, di_g, src_node, tgt_node,
                               1.0, 2.0).variables)

    print(f'{args.instance}: V={di_g.number_of_nodes()} '
          f'E={di_g.number_of_edges()} B={args.budget}, whole-graph QUBO '
          f'{whole} variables')
    print(f'{args.pieces} pieces (seed {args.seed}), {args.split} split, '
          f'max {args.max_terminals} free terminals')
    header = ('piece', 'nodes', 'arcs', 'budget', 'terminals', 'contracted',
              'sub_nodes', 'qubo_vars')
    print(' '.join(f'{name:>10}' for name in header))
    for index, piece in enumerate(pieces):
        sub = piece_subproblem(di_g, src_node, tgt_node, pieces, index,
                               max_terminals=args.max_terminals)
        size = (len(piece_qubo(sub, budgets[index], 1.0).variables)
                if sub.interdictable else 0)
        owned = sum(1 for owner in owners.values() if owner == index)
        row = (index, len(piece), owned, budgets[index], sub.free_terminals,
               'yes' if sub.contracted else 'no',
               sub.graph.number_of_nodes(), size)
        print(' '.join(f'{value:>10}' for value in row))
    return 0


if __name__ == '__main__':
    sys.exit(main())

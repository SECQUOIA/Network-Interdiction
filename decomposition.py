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
   here as in proportion to it).
4. piece_subproblem() builds the network one piece is solved on. Only the
   piece's own arcs may be interdicted there, so the rest of the graph
   carries no budget and is contracted exactly (contract_zone());
   piece_qubo() turns it into a QUBO in which only the piece's arcs carry
   x/gamma/w variables. The slide prototype kept the whole graph and only
   restricted which arcs could be cut, which left the QUBO its full size.
5. solve_decomposed() solves every piece that has a budget and combines the
   interdictions.

What is exact and what is not
-----------------------------
Each piece's subproblem is exact: for every set X of the piece's arcs, the
reduced network with X removed has the same max flow as the original graph
with X removed (and with whatever earlier pieces interdicted). A per-piece
QUBO whose ground state is its optimum -- lambda inside that piece's window,
see #2 -- therefore solves the piece exactly.

The decomposition as a whole is a heuristic. The split is fixed before any
piece is solved, and a fixed split is optimal only if every piece's last
interdiction is worth the same (the common-multiplier condition in #14).
Replacing the split with one multiplier for all pieces, measuring what fits
on each device, comparing against Benders (#15) and evaluating on real
networks (#1) are the rest of #14.

Command line
------------
Print the per-piece QUBO sizes next to the whole-graph size:

    python decomposition.py testcases/50.planarflow --pieces 4 --budget 12
"""
import argparse
import itertools
import sys
from dataclasses import dataclass
from fractions import Fraction

import networkx as nx

import network_interdiction as ni
import solve_qubo_via_dwave

SPLIT_STRATEGIES = ('mincut', 'local')
SOLVE_MODES = ('sequential', 'independent')

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
    out of the sink carry no s-t flow and are left out. A local max flow is
    an estimate: it assumes the rest of the graph can supply and absorb
    whatever the boundary arcs carry, and it goes stale as soon as a
    neighbouring piece is interdicted.
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


def split_budget(di_g, src_node, tgt_node, B, pieces, strategy='mincut'):
    """
    Split the budget B between `pieces`: one integer per piece, summing to B
    unless no piece has any weight (then all zero, see apportion()).

    'mincut'  in proportion to the arcs of the global minimum cut each piece
              owns (mincut_arc_counts())
    'local'   in proportion to each piece's local max flow
              (local_max_flows()). The slides derive this split from the
              local max flows; allocating in proportion to them is this
              module's reading of that rule.

    Both are fixed splits, decided before any piece is solved and never
    revised.
    """
    if strategy == 'mincut':
        weights = mincut_arc_counts(di_g, src_node, tgt_node, pieces)
    elif strategy == 'local':
        weights = local_max_flows(di_g, src_node, tgt_node, pieces)
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
                     mode='sequential', max_terminals=DEFAULT_MAX_TERMINALS):
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

    Returns (interdicted arcs, max flow of `di_g` without them).
    """
    if mode not in SOLVE_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of "
                         f"{', '.join(SOLVE_MODES)}")
    if len(budgets) != len(pieces):
        raise ValueError(f"{len(budgets)} budgets for {len(pieces)} pieces")
    owners = arc_owners(di_g, pieces)
    interdicted = []
    for index, budget in enumerate(budgets):
        if budget <= 0:
            continue
        fixed = interdicted if mode == 'sequential' else ()
        sub = piece_subproblem(di_g, src_node, tgt_node, pieces, index, fixed,
                               max_terminals)
        chosen = list(solve_piece(sub, budget))
        if len(chosen) > budget or len(set(chosen)) != len(chosen) or any(
                owners.get(arc) != index for arc in chosen):
            raise ValueError(
                f"piece {index} returned {chosen}: a piece may cut at most "
                f"its budget ({budget}) of its own arcs")
        interdicted.extend(chosen)
    return interdicted, _surviving_flow(di_g, src_node, tgt_node, interdicted)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Partition an NIP instance, split its budget, and '
                    'compare each piece\'s QUBO size with the whole graph\'s.')
    parser.add_argument('instance', help='flow file (see read_flow_file)')
    parser.add_argument('--pieces', type=int, required=True,
                        help='number of pieces to partition into')
    parser.add_argument('--budget', type=int, required=True,
                        help='interdiction budget B to split')
    parser.add_argument('--split', choices=SPLIT_STRATEGIES, default='mincut',
                        help='budget split rule (default: mincut)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for the partition (default: 0)')
    parser.add_argument('--max-terminals', type=int,
                        default=DEFAULT_MAX_TERMINALS,
                        help='contract the rest of the graph only up to this '
                             'many free terminals (default: '
                             f'{DEFAULT_MAX_TERMINALS})')
    args = parser.parse_args(argv)

    di_g, src_node, tgt_node = ni.read_flow_file(args.instance)
    pieces = partition_graph(di_g, args.pieces, seed=args.seed)
    budgets = split_budget(di_g, src_node, tgt_node, args.budget, pieces,
                           args.split)
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

"""
Mid-size NIP instances: between the tiny_* cases (14-22 qubits) and the
testcases/ families (912-2812 qubits), sized so the whole QUBO fits a
156-qubit IBM device (issues #1, #17).

The QUBO has V - 2 + 3E variables, so there is nothing to put on a QPU or a
simulator between `tiny_c` (22) and `50.planarflow` (912). This module makes
that range: three families mirroring testcases/, at sizes whose QUBO has
30-156 variables, written to testcases/mid/ with a manifest of their exact
optima (results/mid_instances.csv).

Families
--------
planar  Delaunay triangulation of n random points in the unit square. The
        LEDA inputs behind testcases/*.planarflow are triangulations too.
erdos   G(n, p) Erdős–Rényi, resampled until connected.
knn     every point joined to its k nearest neighbours (testcases/*.2dnnflow).

From graph to flow network
--------------------------
The same steps for every family, with one seeded generator, so an instance
is reproduced exactly from (family, n, parameter, seed):

1. Each undirected edge gets a random direction.
2. Until the network is strongly connected, reverse arcs are added one at a
   time, in random order. (The testcases/ generators add every reverse arc
   at once, which makes almost every edge two-way and doubles E; one at a
   time keeps E, and so the qubit count, smaller.)
3. Capacities are uniform integers in [15, 99], as in testcases/.
4. Source and sink are the ordered pair with the most arc-disjoint paths
   between them, ties going to the pair farthest apart. On graphs this
   small a pair chosen for distance alone usually has a source with one or
   two arcs out, so one or two cuts disconnect it and no budget is worth
   benchmarking; with k disjoint paths, every B < k leaves flow. testcases/
   uses the two highest-degree nodes, which on the random families puts the
   optimal cut next to s or t (slide 12, #1); the manifest records, per
   budget, whether that happens here.
"""
import argparse
import contextlib
import csv
import io
import itertools
import json
import os
import sys

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

import network_interdiction as ni
import run_benchmark as rb

FAMILIES = ('planar', 'erdos', 'knn')
MID_DIR = os.path.join('testcases', 'mid')
IBM_QUBITS = 156

# (family, n, parameter, seed) for every committed instance. The parameter is
# unused for planar, p for erdos and k for knn. Chosen for three QUBO sizes
# per family, about 40, 80 and 116-149 variables (39/90/116 planar,
# 40/78/149 Erdős–Rényi, 40/81/142 k-NN). Planar stops at 116: 14 and 15
# points already give 159 and 157.
CATALOG = [
    ('planar', 5, None, 0), ('planar', 8, None, 0), ('planar', 13, None, 0),
    ('erdos', 6, 0.5, 0), ('erdos', 8, 0.45, 0), ('erdos', 13, 0.25, 0),
    ('knn', 6, 2, 0), ('knn', 11, 2, 0), ('knn', 18, 2, 0),
]

MANIFEST_COLUMNS = [
    'instance', 'family', 'V', 'E', 'qubits', 'B', 'maxflow', 'f_curve',
    'lambda_lo', 'lambda_hi', 'lambda', 'Opt_NIP', 'optimal_cut', 'Greedy',
    'cut_touches_st',
]


def undirected_graph(family, n, parameter, rng):
    """The family's undirected graph on nodes 0..n-1 (connected)."""
    if family == 'planar':
        points = rng.random((n, 2))
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for simplex in Delaunay(points).simplices:
            graph.add_edges_from(itertools.combinations(sorted(simplex), 2))
        return graph
    if family == 'erdos':
        while True:
            graph = nx.gnp_random_graph(n, parameter,
                                        seed=int(rng.integers(2 ** 31)))
            if nx.is_connected(graph):
                return graph
    if family == 'knn':
        while True:
            points = rng.random((n, 2))
            graph = nx.Graph()
            graph.add_nodes_from(range(n))
            for i in range(n):
                distance = np.linalg.norm(points - points[i], axis=1)
                for j in np.argsort(distance)[1:parameter + 1]:
                    graph.add_edge(i, int(j))
            if nx.is_connected(graph):
                return graph
    raise ValueError(f'unknown family {family!r}, expected one of {FAMILIES}')


def flow_network(graph, rng):
    """
    (di_g, src, tgt): orient, add reverse arcs until strongly connected,
    draw capacities, and pick a diameter pair as source and sink.
    """
    di_g = nx.DiGraph()
    di_g.add_nodes_from(graph.nodes())
    for u, v in sorted(graph.edges()):
        if rng.random() < 0.5:
            u, v = v, u
        di_g.add_edge(u, v)
    reverse = [(v, u) for u, v in di_g.edges()]
    order = rng.permutation(len(reverse))
    for index in order:
        if nx.is_strongly_connected(di_g):
            break
        di_g.add_edge(*reverse[index])
    for u, v in sorted(di_g.edges()):
        di_g[u][v]['weight'] = int(rng.integers(15, 100))
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    connectivity = {pair: nx.edge_connectivity(di_g, *pair)
                    for pair in itertools.permutations(sorted(di_g.nodes()), 2)}
    src, tgt = max(sorted(connectivity), key=lambda pair: (
        connectivity[pair], lengths[pair[0]][pair[1]]))
    return di_g, src, tgt


def make_instance(family, n, parameter=None, seed=0):
    """The flow network for one catalog entry: (di_g, src, tgt)."""
    rng = np.random.default_rng([FAMILIES.index(family), n, seed])
    return flow_network(undirected_graph(family, n, parameter, rng), rng)


def instance_name(family, n, parameter, seed):
    suffix = '' if parameter is None else f'_{parameter}'
    return f'{family}_{n}{suffix}_s{seed}.flow'


def write_flow_file(path, di_g, src, tgt):
    """The format read_flow_file() reads: node count, 's t', 'u v cap' lines."""
    with open(path, 'w') as handle:
        handle.write(f'{di_g.number_of_nodes()}\n{src} {tgt}\n')
        for u, v in sorted(di_g.edges()):
            handle.write(f"{u} {v} {di_g[u][v]['weight']}\n")


def qubits(di_g):
    """Variables of nip_to_qubo()'s QUBO: V - 2 + 3E."""
    return di_g.number_of_nodes() - 2 + 3 * di_g.number_of_edges()


def interesting_budgets(curve, max_budget=3):
    """
    Budgets B in 1..max_budget worth benchmarking: the lambda window has an
    interior (so the QUBO's ground state is the answer) and f(B) > 0 (the
    budget does not simply cut the sink off).
    """
    budgets = []
    for B in range(1, max_budget + 1):
        if B >= len(curve) or curve[B] == 0:
            break
        lo, hi = rb.lambda_window(curve, B)
        if rb.choose_lambda(lo, hi) is not None:
            budgets.append(B)
    return budgets


def manifest_rows(path, family, max_budget=3):
    """One manifest row per interesting budget of the instance at `path`."""
    di_g, src, tgt = ni.read_flow_file(path)
    curve = rb.optimum_curve('gurobi', di_g, src, tgt, max_budget)
    rows = []
    for B in interesting_budgets(curve, max_budget):
        lo, hi = rb.lambda_window(curve, B)
        _, cut = ni.solve_nip_gurobi(di_g, src, tgt, B)
        rows.append({
            'instance': path.replace(os.sep, '/'), 'family': family,
            'V': di_g.number_of_nodes(), 'E': di_g.number_of_edges(),
            'qubits': qubits(di_g), 'B': B,
            'maxflow': rb.max_flow(di_g, src, tgt),
            'f_curve': json.dumps(curve), 'lambda_lo': lo, 'lambda_hi': hi,
            'lambda': rb.choose_lambda(lo, hi), 'Opt_NIP': curve[B],
            'optimal_cut': json.dumps(sorted([list(arc) for arc in cut])),
            'Greedy': rb.greedy_from_scratch(di_g, src, tgt, B),
            'cut_touches_st': any(src in arc or tgt in arc for arc in cut),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out-dir', default=MID_DIR)
    parser.add_argument('--manifest', default=os.path.join('results', 'mid_instances.csv'))
    parser.add_argument('--max-budget', type=int, default=4)
    args = parser.parse_args(argv)

    # Read before anything is written, as in run_qaoa_benchmark.py.
    commit = rb.repo_commit()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for family, n, parameter, seed in CATALOG:
        di_g, src, tgt = make_instance(family, n, parameter, seed)
        if qubits(di_g) > IBM_QUBITS:
            raise ValueError(f'{family} n={n}: {qubits(di_g)} qubits, over {IBM_QUBITS}')
        path = os.path.join(args.out_dir, instance_name(family, n, parameter, seed))
        write_flow_file(path, di_g, src, tgt)
        with contextlib.redirect_stdout(io.StringIO()):  # Gurobi prints every solve
            found = manifest_rows(path, family, args.max_budget)
        rows.extend(found)
        print(f'{path}: V={di_g.number_of_nodes()} E={di_g.number_of_edges()} '
              f'qubits={qubits(di_g)} budgets={[row["B"] for row in found]}')
    with open(args.manifest, 'w', newline='') as handle:
        handle.write('# Generated by instances.py -- see its docstring.\n')
        handle.write(f"# command: {' '.join(sys.argv)}\n")
        handle.write(f'# repo commit: {commit}\n')
        handle.write('# exact optima from Gurobi (solve_nip_gurobi); lambda = '
                     'auto, the midpoint of the exact window.\n')
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())

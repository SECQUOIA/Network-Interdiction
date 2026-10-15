"""Bounded offline minor-embedding overhead on ideal P16/Z12 topology models.

No DWaveSampler, credentials, live working graph or quantum job is involved.
Each retained mapping is independently checked as a graph minor certificate.
An unsuccessful heuristic attempt is not an infeasibility proof.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import dwave_networkx as dnx
import minorminer
import networkx as nx

import network_interdiction as ni
from run_benchmark import parse_case


def topology(name):
    # dwave-graphs currently requires NumPy >=2, incompatible with this repo's
    # pyqubo NumPy<2 contract. Retain the already-declared compatible generator.
    if name == 'P16':
        return dnx.pegasus_graph(16, fabric_only=True)
    if name == 'Z12':
        return dnx.zephyr_graph(12, t=4)
    raise ValueError('topology must be P16 or Z12')


def interaction_graph(bqm):
    """Include every variable, including isolates, but only nonzero couplings."""
    if not all(math.isfinite(v) for v in [bqm.offset, *bqm.linear.values(), *bqm.quadratic.values()]):
        raise ValueError('BQM coefficients must be finite')
    graph = nx.Graph()
    graph.add_nodes_from(bqm.variables)
    graph.add_edges_from((u, v) for (u, v), value in bqm.quadratic.items() if value != 0)
    return graph


def graph_digest(graph):
    """Hash labeled nodes and edges, including isolates (CLI labels: str/int)."""
    nodes = sorted(graph, key=lambda x: (type(x).__name__, repr(x)))
    index = {node: i for i, node in enumerate(nodes)}
    edges = sorted(tuple(sorted((index[u], index[v]))) for u, v in graph.edges)
    return hashlib.sha256(json.dumps([nodes, edges], separators=(',', ':')).encode()).hexdigest()


def audit_embedding(source, target, embedding):
    """Check coverage, chain membership/connectivity/disjointness, and couplers."""
    if set(embedding) != set(source):
        raise ValueError('embedding must cover every logical variable exactly')
    occupied = set()
    target_nodes = set(target)
    chains = {}
    for variable, chain in embedding.items():
        nodes = set(chain)
        if not nodes or len(nodes) != len(chain) or not nodes <= target_nodes:
            raise ValueError('chain is empty, duplicated, or outside the target')
        if occupied & nodes:
            raise ValueError('chains overlap')
        if not nx.is_connected(target.subgraph(nodes)):
            raise ValueError('chain is disconnected')
        occupied.update(nodes)
        chains[variable] = nodes
    for u, v in source.edges:
        if not any(neighbor in chains[v] for qubit in chains[u] for neighbor in target[qubit]):
            raise ValueError('logical interaction has no inter-chain coupler')
    lengths = [len(chain) for chain in chains.values()]
    return {'physical_qubits': len(occupied), 'overhead_ratio': len(occupied)/len(source),
            'max_chain_length': max(lengths), 'mean_chain_length': sum(lengths)/len(lengths),
            'chain_length_histogram': {str(k): lengths.count(k) for k in sorted(set(lengths))}}


def embedding_attempt(source, target, *, seed=0, timeout=5):
    """One seeded, single-thread minorminer call with one restart allowance."""
    if (isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or
            not 0 <= seed < 2**32):
        raise ValueError('seed must be an integer in 0..2**32-1')
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    if not source or source.is_directed() or target.is_directed() or nx.number_of_selfloops(source):
        raise ValueError('source must be nonempty and graphs must be undirected without source loops')
    result = dict(seed=int(seed), timeout_sec=timeout, tries=1, threads=1,
                  logical_variables=len(source), logical_interactions=source.number_of_edges(),
                  source_graph_sha256=graph_digest(source), target_graph_sha256=graph_digest(target),
                  target_qubits=len(target), target_couplers=target.number_of_edges(),
                  physical_qubits=None, overhead_ratio=None, max_chain_length=None,
                  mean_chain_length=None, chain_length_histogram=None, embedding=None,
                  embedding_time_sec=None, qpu_time_sec=None)
    if len(source) > len(target):
        result['status'] = 'too_many_logical_variables'
        return result
    start = time.perf_counter()
    mapping = minorminer.find_embedding(source, target, random_seed=int(seed),
                                       timeout=timeout, tries=1, threads=1)
    result['embedding_time_sec'] = time.perf_counter()-start
    if not mapping:
        result['status'] = 'not_found_within_budget'
        return result
    result.update(audit_embedding(source, target, mapping))
    result.update(status='embedded', embedding=[{'variable': v, 'chain': sorted(mapping[v])}
                                               for v in source])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', type=parse_case, required=True)
    parser.add_argument('--topology', action='append', choices=['P16', 'Z12'])
    parser.add_argument('--seed', action='append', type=int)
    parser.add_argument('--timeout', type=float, default=5,
                        help='seconds per minorminer call; use an outer process limit as well')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if any(lam == 'auto' or not math.isfinite(lam) or lam <= 0 or budget < 0
           for _, budget, lam in args.case):
        parser.error('provide a nonnegative budget and explicit finite positive lambda')
    seeds = args.seed or [0]
    if not math.isfinite(args.timeout) or args.timeout <= 0 or any(not 0 <= s < 2**32 for s in seeds):
        parser.error('timeout must be positive and finite; seeds must be in 0..2**32-1')
    report = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                         ('profile_embedding.py', 'network_interdiction.py', 'run_benchmark.py')},
        'argv': list(argv) if argv is not None else sys.argv[1:],
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('minorminer', 'dwave-networkx', 'networkx', 'dimod', 'pyqubo', 'numpy')},
        'threads': {name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'scope': 'Ideal topology models only, no defective/live working graph, '
                 'sampler, account or QPU. A failed heuristic attempt does not prove infeasibility.',
        'topologies': {}, 'rows': [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.topology or ['P16', 'Z12']:
        target = topology(name)
        report['topologies'][name] = dict(qubits=len(target), couplers=target.number_of_edges(),
                                         graph_sha256=graph_digest(target))
        for path, budget, lam in args.case:
            graph, source, sink = ni.read_flow_file(path)
            if budget > len(graph.edges):
                raise ValueError('budget exceeds number of edges')
            start = time.perf_counter()
            bqm = ni.nip_to_qubo(budget, graph, source, sink, lam, lam+1)
            logical = interaction_graph(bqm)
            construction = time.perf_counter()-start
            for seed in seeds:
                row = embedding_attempt(logical, target, seed=seed, timeout=args.timeout)
                row.update(instance=str(path), budget=budget, cardinality_penalty=lam,
                           dual_penalty=lam+1, topology=name,
                           input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                           bqm_graph_build_time_sec=construction)
                report['rows'].append(row)
                args.out.write_text(json.dumps(report, indent=2)+'\n')
                print(f"{name} {path} seed={seed}: {row['status']}, "
                      f"{row['logical_variables']} logical / {row['physical_qubits']} physical", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

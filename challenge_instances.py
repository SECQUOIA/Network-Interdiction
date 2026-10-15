"""Deterministic, offline conversions for the bounded challenge cohort.

All outputs use integer capacities and the existing .flow format. The manifest
also records the allowed arc set: it must travel with restricted synthetic cases.
See testcases/challenge/README.md for units, attribution, and modeling limits.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import gzip
import hashlib
import json
from pathlib import Path
import random
import re

import networkx as nx

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'testcases/challenge'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transport(text):
    """TNTP directed capacities, rounded to the nearest integer (half up)."""
    graph = nx.DiGraph()
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        u, v = map(int, fields[:2])
        capacity = int(Decimal(fields[2]).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        if u == v or graph.has_edge(u, v) or capacity <= 0:
            raise ValueError('expected positive capacities and unique directed arcs')
        graph.add_edge(u, v, weight=capacity)
    return graph


def contacts(text):
    """Each unordered pair accumulates 20 seconds per unique contact interval.

    Emit two independently removable directed arcs with that capacity. This is
    an aggregate contact-flow proxy, not a temporal epidemic or person-removal model.
    """
    graph, seen = nx.DiGraph(), set()
    for line in text.splitlines():
        timestamp, u, v, a, b = line.split()
        u, v = sorted((int(u), int(v)))
        key = int(timestamp), u, v
        if a != '4A' or b != '4A' or u == v or key in seen:
            raise ValueError('expected unique within-class 4A contacts')
        seen.add(key)
        capacity = graph.get_edge_data(u, v, {}).get('weight', 0) + 20
        graph.add_edge(u, v, weight=capacity)
        graph.add_edge(v, u, weight=capacity)
    return graph


def power(text):
    """Read active IEEE14 branches; rateA MVA becomes a directed flow proxy.

    Two independently removable directions per physical line, NOT coupled line
    outages; no voltage, reactance, Kirchhoff, generation, or demand constraints.
    No MATLAB code is evaluated. The retained PGLib ratings are model estimates.
    """
    match = re.search(r'mpc\.branch\s*=\s*\[(.*?)\];', text, re.S)
    if not match:
        raise ValueError('missing MATPOWER branch matrix')
    graph = nx.DiGraph()
    for line in match[1].splitlines():
        fields = line.split('%', 1)[0].strip().rstrip(';').split()
        if not fields:
            continue
        if len(fields) != 13:
            raise ValueError('expected thirteen MATPOWER branch fields')
        u, v = map(int, fields[:2])
        if Decimal(fields[10]) == 0:
            continue
        cap = Decimal(fields[5])
        if cap <= 0 or cap != int(cap) or u == v or graph.has_edge(u, v):
            raise ValueError('expected positive integer ratings and no parallel lines')
        graph.add_edge(u, v, weight=int(cap))
        graph.add_edge(v, u, weight=int(cap))
    return graph


def relabel_pair(graph):
    """Use the first maximum-hop-distance pair with at least three incident arcs.

    Select using topology only, before solving interdiction. Sort original IDs
    and map them to consecutive integers for PyQUBO compatibility.
    """
    if not graph or not nx.is_strongly_connected(graph):
        raise ValueError('real-network conversion must be strongly connected')
    distances = [(d, -u, -v) for u, lengths in nx.all_pairs_shortest_path_length(graph)
                 for v, d in lengths.items() if u != v and graph.out_degree(u) >= 3
                 and graph.in_degree(v) >= 3]
    if not distances:
        raise ValueError('no terminal pair has at least three outgoing/incoming arcs')
    _, minus_source, minus_sink = max(distances)
    ids = sorted(graph)
    mapping = {old: new for new, old in enumerate(ids)}
    return (nx.relabel_nodes(graph, mapping), mapping[-minus_source],
            mapping[-minus_sink], ids)


def covering(n, probability, seed):
    """Minimum-union covering stressor, with all removable arcs internal.

    For each edge {u,v} of a seeded simple graph, create s->a->b->u,v;
    capacities are 2,2,2,2. Vertex->t capacity is 1. Only a->b may be cut.
    For a retained set R of original edges the max flow is |union(R)|:
    each incident vertex can receive one unit along any retained incident edge;
    an edge serves at most two vertices. Thus deleting m-r arcs asks for r
    edges with the smallest union of endpoints. It is a covering stressor,
    not a replication of published run times or a guaranteed hard instance.
    """
    if not isinstance(n, int) or not 2 <= n <= 40 or not 0 < probability <= 1:
        raise ValueError('covering generator requires 2..40 vertices and 0<p<=1')
    rng = random.Random(seed)
    original = [(u, v) for u in range(n) for v in range(u + 1, n)
                if rng.random() < probability]
    graph = nx.DiGraph()
    source, sink = 0, 1
    graph.add_nodes_from(range(2 + n + 2 * len(original)))
    graph.add_weighted_edges_from((v + 2, sink, 1) for v in range(n))
    allowed = []
    for i, (u, v) in enumerate(original):
        a, b = 2 + n + 2 * i, 3 + n + 2 * i
        graph.add_weighted_edges_from([(source, a, 2), (a, b, 2),
                                       (b, u + 2, 2), (b, v + 2, 2)])
        allowed.append((a, b))
    return graph, source, sink, allowed, original


def flow_bytes(graph, source, sink):
    lines = [str(len(graph)), f'{source} {sink}']
    lines += [f'{u} {v} {int(graph[u][v]["weight"])}' for u, v in sorted(graph.edges())]
    return ('\n'.join(lines) + '\n').encode()


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    specs = [('ema', 'road', 'EMA_net.tntp', transport, [1, 3]),
             ('school4a', 'contact', 'primaryschool_4A.tsv.gz', contacts, [1, 3]),
             ('ieee14', 'power', 'pglib_opf_case14_ieee.m', power, [1, 2])]
    cases = []
    for name, domain, filename, convert, budgets in specs:
        path = DATA / 'sources' / filename
        raw = path.read_bytes()
        text = (gzip.decompress(raw) if path.suffix == '.gz' else raw).decode()
        graph, source, sink, ids = relabel_pair(convert(text))
        out = destination / f'{name}.flow'
        out.write_bytes(flow_bytes(graph, source, sink))
        cases.append({'name': name, 'family': domain, 'file': out.name,
                      'sha256': digest(out), 'source_file': filename,
                      'source_sha256': digest(path), 'original_node_ids': ids,
                      'source': source, 'sink': sink, 'budgets': budgets,
                      'interdictable': None})
    for n, retained in [(12, 6), (18, 10), (24, 15), (30, 21)]:
        graph, source, sink, allowed, original = covering(n, .5, 0)
        name = f'cover_n{n}_s0'
        out = destination / f'{name}.flow'
        out.write_bytes(flow_bytes(graph, source, sink))
        cases.append({'name': name, 'family': 'covering', 'file': out.name,
                      'sha256': digest(out), 'source': source, 'sink': sink,
                      'budgets': [len(allowed) - retained], 'interdictable': allowed,
                      'generator': {'n': n, 'probability': .5, 'seed': 0,
                                    'retained_edges': retained, 'original_edges': original}})
    for name, family in [('50.planarflow', 'planar'), ('50_0.15.erdosflow', 'erdos'),
                         ('50_5.2dnnflow', 'knn')]:
        path = ROOT / 'testcases' / name
        cases.append({'name': name, 'family': family, 'file': '../' + name,
                      'sha256': digest(path), 'budgets': [3, 6], 'interdictable': None})
    (destination / 'manifest.json').write_text(json.dumps({'schema': 1, 'cases': cases}, indent=2) + '\n')


def load_cases(manifest=DATA / 'manifest.json'):
    """Validate all inputs before a caller dispatches a solver."""
    import network_interdiction as ni
    manifest = Path(manifest)
    result = []
    for case in json.loads(manifest.read_text())['cases']:
        path = manifest.parent / case['file']
        if digest(path) != case['sha256']:
            raise ValueError(f'input hash mismatch: {path}')
        graph, source, sink = ni.read_flow_file(path)
        if len(graph) > 1000 or graph.number_of_edges() > 4000:
            raise ValueError('cohort input exceeds 1000 nodes or 4000 arcs')
        if set(graph) != set(range(len(graph))) or source == sink:
            raise ValueError('expected consecutive vertices and distinct terminals')
        allowed = (set(graph.edges()) if case['interdictable'] is None else
                   {tuple(e) for e in case['interdictable']})
        if allowed - set(graph.edges()):
            raise ValueError('unknown allowed arc')
        for budget in case['budgets']:
            if type(budget) is not int or not 0 <= budget <= len(allowed):
                raise ValueError('invalid budget')
            result.append((case, graph, source, sink, budget, allowed))
    if len(result) > 24:
        raise ValueError('cohort exceeds 24 cases')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='output directory (not the source directory)')
    args = parser.parse_args()
    generate(args.out)

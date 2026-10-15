"""Audit retained embeddings using Ocean's separate certificate verifier."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from dwave.embedding import verify_embedding
import networkx as nx
import pytest

import network_interdiction as ni
import profile_embedding as pe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/embedding_resources.json'


@pytest.mark.parametrize('filename,count,cases_count,seeds,timeout', [
    ('embedding_resources', 80, 20, (0, 1), 5), ('embedding_large_resources', 6, 3, (0,), 30)])
def test_embedding_campaign_provenance_and_coverage(filename, count, cases_count, seeds, timeout):
    data = ROOT/f'results/{filename}.json'
    report = json.loads(data.read_text())
    manifest = json.loads((ROOT/f'results/{filename}.manifest.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == manifest['report_sha256']
    assert report['source_commit'] == manifest['source_commit']
    assert len(report['rows']) == count
    cases = {r['instance'] for r in report['rows']}
    assert len(cases) == cases_count
    assert {(r['instance'], r['topology'], r['seed']) for r in report['rows']} == {
        (case, target, seed) for case in cases for target in ('P16', 'Z12') for seed in seeds}
    for row in report['rows']:
        assert hashlib.sha256((ROOT/row['instance']).read_bytes()).hexdigest() == row['input_sha256']
        assert row['tries'] == row['threads'] == 1 and row['timeout_sec'] == timeout
        assert row['qpu_time_sec'] is None


@pytest.mark.parametrize('filename', ['embedding_resources', 'embedding_large_resources'])
def test_every_saved_chain_is_a_valid_minor_certificate(filename):
    report = json.loads((ROOT/f'results/{filename}.json').read_text())
    graphs = {}
    for name in report['topologies']:
        target = graphs[name] = pe.topology(name)
        assert pe.graph_digest(target) == report['topologies'][name]['graph_sha256']
    sources = {}
    for row in report['rows']:
        if row['instance'] not in sources:
            network, src, sink = ni.read_flow_file(ROOT/row['instance'])
            bqm = ni.nip_to_qubo(row['budget'], network, src, sink,
                                row['cardinality_penalty'], row['dual_penalty'])
            source = nx.Graph()
            source.add_nodes_from(bqm.variables)
            source.add_edges_from((u, v) for (u, v), coefficient in bqm.quadratic.items() if coefficient != 0)
            sources[row['instance']] = source
        source, target = sources[row['instance']], graphs[row['topology']]
        assert pe.graph_digest(source) == row['source_graph_sha256']
        assert pe.graph_digest(target) == row['target_graph_sha256']
        assert len(source) == row['logical_variables']
        assert source.number_of_edges() == row['logical_interactions']
        if row['status'] == 'embedded':
            mapping = {entry['variable']: entry['chain'] for entry in row['embedding']}
            assert len(mapping) == len(row['embedding']) == len(source)
            assert set(mapping) == set(source)
            assert verify_embedding(mapping, source, target)
            lengths = [len(chain) for chain in mapping.values()]
            occupied = {q for chain in mapping.values() for q in chain}
            assert len(occupied) == sum(lengths) == row['physical_qubits']
            assert row['overhead_ratio'] == len(occupied)/len(source)
            assert row['mean_chain_length'] == sum(lengths)/len(lengths)
            assert row['max_chain_length'] == max(lengths)
            assert row['chain_length_histogram'] == {str(k): v for k, v in Counter(lengths).items()}
            assert row['embedding_time_sec'] >= 0
        else:
            assert row['status'] in ('not_found_within_budget', 'too_many_logical_variables')
            assert all(row[key] is None for key in ('embedding', 'physical_qubits',
                'overhead_ratio', 'max_chain_length', 'mean_chain_length', 'chain_length_histogram'))
            if row['status'] == 'too_many_logical_variables':
                assert len(source) > len(target) and row['embedding_time_sec'] is None
            else:
                assert row['embedding_time_sec'] >= 0


def test_longer_searches_do_not_overwrite_short_failures():
    short = json.loads(DATA.read_text())['rows']
    long = json.loads((ROOT/'results/embedding_large_resources.json').read_text())['rows']
    assert sum(row['status'] == 'embedded' for row in long) == 4
    for row in long:
        previous = [r for r in short if r['instance'] == row['instance'] and r['topology'] == row['topology']]
        assert len(previous) == 2 and all(r['status'] == 'not_found_within_budget' for r in previous)
        assert all(r['source_graph_sha256'] == row['source_graph_sha256'] and
                   r['target_graph_sha256'] == row['target_graph_sha256'] for r in previous)

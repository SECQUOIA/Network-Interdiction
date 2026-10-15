"""Independent flow audits of the saved SA/Aer same-case sample."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import networkx as nx
import pytest

import network_interdiction as ni

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/shared_sa_qaoa.csv'
with DATA.open() as stream:
    ROWS = list(csv.DictReader(line for line in stream if not line.startswith('#')))


def test_shared_sample_provenance_and_coverage():
    manifest = json.loads((ROOT / 'results/shared_sa_qaoa.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest['csv_sha256']
    assert manifest['source_commit'] in DATA.read_text()
    assert [(r['instance'], r['B']) for r in ROWS] == [
        ('testcases/tiny_a.flow', '1'), ('testcases/tiny_b.flow', '2')]
    for path, digest in manifest['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['instance'])
def test_exact_reference_and_every_selected_raw_cut(row):
    graph, source, sink = ni.read_flow_file(ROOT/row['instance'])
    budget = int(row['B'])
    def flow(edges):
        residual = graph.copy()
        residual.remove_edges_from(edges)
        return nx.maximum_flow_value(residual, source, sink, capacity='weight',
                                     flow_func=nx.algorithms.flow.edmonds_karp)
    optimum = min(flow(edges) for size in range(budget+1)
                  for edges in itertools.combinations(graph.edges, size))
    assert float(row['Opt_NIP']) == optimum == 5
    assert float(row['maxflow']) == flow([])
    for solver in ('SA', 'QAOA'):
        selected = json.loads(row[f'{solver}_raw_samples'])
        assert len(selected) == int(row[f'{solver}_raw_samples_examined']) == 20
        assert sum(record['budget_feasible'] for record in selected) == int(row[f'{solver}_raw_feasible_count'])
        assert selected == json.loads(row[f'{solver}_pre_greedy_samples'])
        for record in selected:
            cut = [tuple(edge) for edge in record['interdicted_edges']]
            assert set(cut) <= set(graph.edges) and len(cut) == len(set(cut))
            assert record['flow'] == flow(cut)
            assert record['budget_feasible'] == (len(cut) <= budget)
            assert record['energy'] >= float(row['Opt_QUBO'])
            assert record['num_occurrences'] >= 1
        assert min(record['flow'] for record in selected if record['budget_feasible']) == float(row[f'{solver}_NIP']) == optimum
        assert float(row[f'{solver}_NIP_Greedy']) == optimum
        assert float(row[f'{solver}_QUBO']) == float(row['Opt_QUBO'])
    assert all(value == '' for key, value in row.items() if key.startswith(('QCI_', 'QA_')))
    metadata = json.loads(row['QAOA_metadata'])
    assert metadata['num_qubits'] == len(metadata['variables']) == int(row['V']) - 2 + 3*int(row['E'])
    assert metadata['shots'] == 2500 and metadata['evaluations'] == metadata['maxiter'] == 20
    assert metadata['seed'] == 0 and metadata['reps'] == 1
    assert float(row['QAOA_time_sec']) >= sum(metadata[key] for key in
        ('build_time_sec', 'train_time_sec', 'sample_time_sec'))

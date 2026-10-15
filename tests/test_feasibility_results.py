"""Recompute the new saved SA sample's counts and all 60 selected cut flows."""
import csv
import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

import network_interdiction as ni

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results/raw_feasibility_sa.csv'
with CSV.open() as stream:
    ROWS = list(csv.DictReader(line for line in stream if not line.startswith('#')))
CASES = {('testcases/50.planarflow', 5), ('testcases/50_0.15.erdosflow', 5),
         ('testcases/50_5.2dnnflow', 4)}


def test_saved_sample_covers_documented_cases_and_pinned_inputs():
    assert len(ROWS) == len(CASES)
    assert {(row['instance'], int(row['B'])) for row in ROWS} == CASES
    manifest = json.loads((ROOT / 'results/raw_feasibility_sa.json').read_text())
    assert manifest['source_commit'] in CSV.read_text()
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == manifest['csv_sha256']
    assert set(manifest['input_sha256']) == {path for path, _ in CASES}
    for path, digest in manifest['input_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['instance'])
def test_saved_records_independently_reproduce_feasibility_and_surviving_flow(row):
    graph, source, sink = ni.read_flow_file(ROOT / row['instance'])
    records = json.loads(row['SA_raw_samples'])
    assert len(records) == int(row['SA_raw_samples_examined']) == 20
    assert records == json.loads(row['SA_pre_greedy_samples'])
    assert [r['energy'] for r in records] == sorted(r['energy'] for r in records)
    assert [r['rank'] for r in records] == list(range(1, 21))
    feasible_flows = []
    for record in records:
        edges = [tuple(edge) for edge in record['interdicted_edges']]
        assert len(set(edges)) == len(edges)
        assert set(edges) <= set(graph.edges)
        assert record['num_occurrences'] >= 1
        feasible = len(edges) <= int(row['B'])
        assert feasible == record['budget_feasible']
        residual = graph.copy()
        residual.remove_edges_from(edges)
        flow = nx.maximum_flow_value(residual, source, sink, capacity='weight',
                                     flow_func=nx.algorithms.flow.edmonds_karp)
        assert flow == record['flow']
        if feasible:
            feasible_flows.append(flow)
    assert len(feasible_flows) == int(row['SA_raw_feasible_count'])
    assert row['SA_raw_feasible_count'] == row['SA_pre_greedy_feasible_count']
    assert row['SA_raw_samples_examined'] == row['SA_pre_greedy_samples_examined']
    assert records[0]['energy'] == float(row['SA_QUBO'])
    if feasible_flows:
        assert min(feasible_flows) == float(row['SA_NIP'])
    else:
        assert row['SA_NIP'] == ''
    assert all(value == '' for key, value in row.items() if key.startswith('QA_'))

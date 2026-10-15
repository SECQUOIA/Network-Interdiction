"""Audit the saved three-family sample without rerunning licensed solvers."""
import csv
import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

import network_interdiction as ni

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results/classical_baselines_sample.csv'
CASES = {('testcases/50.planarflow', 5),
         ('testcases/50_0.15.erdosflow', 5),
         ('testcases/50_5.2dnnflow', 4)}


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith('#')))


def test_sample_has_exactly_the_documented_cases_and_pinned_inputs():
    sample = rows(CSV)
    assert len(sample) == len(CASES)
    assert {(row['instance'], int(row['B'])) for row in sample} == CASES
    manifest = json.loads((ROOT / 'results/classical_baselines_sample.json').read_text())
    assert manifest['source_commit'] in CSV.read_text()
    assert set(manifest['input_sha256']) == {path for path, _ in CASES}
    for path, expected in manifest['input_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == manifest['csv_sha256']


@pytest.mark.parametrize('row', rows(CSV), ids=lambda row: row['instance'])
def test_saved_benders_cut_and_certificate_match_independent_flow_and_prior_reference(row):
    graph, source, sink = ni.read_flow_file(ROOT / row['instance'])
    removed = [tuple(edge) for edge in json.loads(row['Benders_interdicted_edges'])]
    assert len(set(removed)) == len(removed) <= int(row['B'])
    assert set(removed) <= set(graph.edges)
    assert len(graph) == int(row['V']) and len(graph.edges) == int(row['E'])
    graph.remove_edges_from(removed)
    # Edmonds-Karp provides a separate max-flow algorithm from the producing
    # NetworkX default preflow-push; no Benders master is run by this audit.
    value = nx.maximum_flow_value(graph, source, sink, capacity='weight',
                                 flow_func=nx.algorithms.flow.edmonds_karp)
    assert value == float(row['Benders'])
    prior = next(old for old in rows(ROOT / 'results/slide_instances_sa.csv')
                 if old['instance'] == row['instance'] and old['B'] == row['B'])
    optimum = float(prior['Opt_NIP'])
    assert optimum == float(row['Opt_NIP'])
    assert float(row['LP_lower_bound']) <= optimum <= float(row['LP_Round_Repair'])
    assert float(row['Benders_lower_bound']) <= optimum <= value
    assert row['Benders_status'] == 'optimal'
    assert float(row['Benders_lower_bound']) == value
    history = json.loads(row['Benders_history'])
    assert len(history) == int(row['Benders_iterations'])
    assert [step['lower_bound'] for step in history] == sorted(step['lower_bound'] for step in history)
    assert [step['best_flow'] for step in history] == sorted((step['best_flow'] for step in history), reverse=True)
    assert history[-1]['best_flow'] == value
    assert history[-1]['time_sec'] <= float(row['Benders_time_sec'])
    assert float(row['Gurobi_total_time_sec']) >= float(row['Gurobi_time_sec'])
    assert float(row['SA_total_time_sec']) >= float(row['SA_time_sec']) + float(row['SA_repair_time_sec'])
    for key in ('Reference_time_sec', 'LP_time_sec', 'LP_repair_time_sec', 'Greedy_time_sec'):
        assert float(row[key]) > 0
    for key in ('QA_NIP(advantage1)', 'QA_NIP(advantage1)_Greedy', 'QA_QUBO(advantage1)',
                'Lagrangian_Feasible', 'Random_Greedy'):
        assert row[key] == ''

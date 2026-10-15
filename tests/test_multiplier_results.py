"""Audit saved combined cuts and exact-price allocation bounds independently."""
import csv
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import networkx as nx
import pytest

import network_interdiction as ni

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results/common_multiplier_comparison.csv'


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith('#')))


ROWS = read(CSV)
REFERENCES = {(r['instance'], r['B']): r
              for r in read(ROOT / 'results/classical_baselines_sample.csv')}


def test_comparison_covers_each_rule_mode_and_pinned_reference_case():
    expected = {(instance, budget, rule, mode) for instance, budget in REFERENCES
                for rule in ('mincut', 'local', 'marginal', 'multiplier')
                for mode in ('independent', 'sequential', 'iterated')}
    assert {(r['instance'], r['B'], r['rule'], r['mode']) for r in ROWS} == expected
    assert len(ROWS) == len(expected) == 36
    manifest = json.loads((ROOT / 'results/common_multiplier_comparison.json').read_text())
    assert manifest['source_commit'] in CSV.read_text()
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == manifest['csv_sha256']
    for path, digest in manifest['input_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('row', ROWS, ids=lambda r: f"{r['instance']}-{r['rule']}-{r['mode']}")
def test_combined_interdictions_are_feasible_and_match_independent_flow(row):
    graph, source, sink = ni.read_flow_file(ROOT / row['instance'])
    cut = [tuple(edge) for edge in json.loads(row['interdicted_edges'])]
    assert len(cut) == len(set(cut)) <= int(row['B'])
    assert set(cut) <= set(graph.edges)
    graph.remove_edges_from(cut)
    flow = nx.maximum_flow_value(graph, source, sink, capacity='weight',
                                 flow_func=nx.algorithms.flow.edmonds_karp)
    assert flow == float(row['flow'])
    reference = REFERENCES[row['instance'], row['B']]
    assert float(row['optimum']) == float(reference['Opt_NIP']) <= flow
    budgets = json.loads(row['budgets'])
    assert len(budgets) == int(row['pieces']) == 4
    assert all(k >= 0 for k in budgets) and sum(budgets) <= int(row['B'])
    assert float(row['ratio']) == pytest.approx(
        (float(row['maxflow']) - flow) / (float(row['maxflow']) - float(row['optimum'])),
        abs=5e-7)
    assert float(row['time_sec']) >= 0 and float(row['partition_sec']) >= 0
    if row['rule'] != 'multiplier':
        assert all(row[field] == '' for field in ('allocation_status', 'surrogate_value',
            'surrogate_lower_bound', 'allocation_curves', 'multiplier_history'))


@pytest.mark.parametrize('row', [r for r in ROWS if r['rule'] == 'multiplier' and r['mode'] == 'sequential'],
                         ids=lambda r: r['instance'])
def test_every_recorded_price_is_a_bound_on_exhaustive_allocation(row):
    curves = json.loads(row['allocation_curves'])
    target = int(row['B'])
    optimum = min(sum(c[k] for c, k in zip(curves, allocation))
                  for allocation in itertools.product(*(range(len(c)) for c in curves))
                  if sum(allocation) == target)
    budgets = json.loads(row['budgets'])
    assert sum(c[k] for c, k in zip(curves, budgets)) == float(row['surrogate_value'])
    assert float(row['surrogate_lower_bound']) <= optimum <= float(row['surrogate_value'])
    history = json.loads(row['multiplier_history'])
    assert 1 <= len(history) <= 100
    assert any(item['repaired_budgets'] == budgets for item in history)
    for item in history:
        price = Fraction(item['multiplier_exact'])
        assert price >= 0
        lower = max(Fraction(0), sum(min(Fraction(c[k]) + price*k
                                        for k in range(len(c))) for c in curves) - price*target)
        assert Fraction(item['surrogate_lower_bound']) <= lower <= optimum
        allocation = item['repaired_budgets']
        assert sum(allocation) == target
        value = sum(c[k] for c, k in zip(curves, allocation))
        assert value >= optimum
        assert item['surrogate_value'] == value
        assert sum(item['requested_budgets']) - target == item['budget_violation']
    assert history[-1]['best_surrogate_lower_bound'] == float(row['surrogate_lower_bound'])
    if row['allocation_status'] == 'optimal_surrogate':
        assert float(row['surrogate_lower_bound']) == float(row['surrogate_value']) == optimum
    else:
        assert row['allocation_status'] == 'iteration_limit' and len(history) == 100

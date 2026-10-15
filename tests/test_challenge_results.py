"""Audit retained observations without rerunning the expensive campaign."""
import json
from pathlib import Path

import networkx as nx
import pytest

import challenge_instances as ci

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('artifact,manifest', [
    ('challenge_profile.json', 'manifest.json'),
    ('challenge_legacy_profile.json', 'legacy_manifest.json')])
def test_every_solver_witness_and_bound_in_the_retained_screen(artifact, manifest):
    data = json.loads((ROOT / 'results' / artifact).read_text())
    assert data['attempts'] == [{'attempt': 1, 'status': 'completed', 'completed_solver_calls': 32}]
    assert len(data['cases']) == 16
    assert data['manifest_sha256'] == ci.digest(ci.DATA / manifest)
    inputs = {(spec['name'], budget): (graph, s, t, allowed)
              for spec, graph, s, t, budget, allowed in ci.load_cases(ci.DATA / manifest)}
    for row in data['cases']:
        graph, s, t, allowed = inputs[(row['name'], row['B'])]
        results = [row['milp'], row['benders']]
        for result in results:
            cuts = [result] + result.get('incumbents', [])
            for point in cuts:
                removed = [tuple(edge) for edge in point['interdicted_edges']]
                assert len(set(removed)) == len(removed) <= row['B']
                assert set(removed) <= allowed
                residual = graph.copy()
                residual.remove_edges_from(removed)
                actual = nx.maximum_flow_value(residual, s, t, capacity='weight')
                assert actual == point['flow']
            incident = sum(s in edge or t in edge for edge in result['interdicted_edges'])
            assert incident == result['terminal_incident_count']
            assert result['any_terminal_incident'] == bool(incident)
            assert result['all_terminal_incident'] == bool(incident and incident == len(result['interdicted_edges']))
            assert result['lower_bound'] <= min(r['flow'] for r in results) + 1e-7
            if result['status'] == 'optimal':
                assert result['flow'] == pytest.approx(result['lower_bound'], abs=1e-7)
            for other in results:
                if other['status'] == 'optimal':
                    assert result['lower_bound'] <= other['flow'] + 1e-7 <= result['flow'] + 1e-7
            assert result['node_count'] >= 0
        if row['classification'] == 'hard_for_both_measured_solvers':
            assert all(r['runtime_sec'] >= 5 for r in results)
            assert row['allowed_only_interior'] or any(r['status'] == 'optimal' for r in results)
            assert not any(r['any_terminal_incident'] and r['status'] == 'optimal' for r in results)
        for point in row['milp']['trajectory']:
            assert point['lower_bound'] <= min(r['flow'] for r in results) + 1e-7


def test_covering_observations_have_an_independent_endpoint_union_certificate():
    data = json.loads((ROOT / 'results/challenge_profile.json').read_text())
    manifest = {c['name']: c for c in json.loads((ci.DATA / 'manifest.json').read_text())['cases']}
    for row in data['cases']:
        if row['family'] != 'covering':
            continue
        case = manifest[row['name']]
        edges = case['generator']['original_edges']
        allowed = [tuple(edge) for edge in case['interdictable']]
        for result in (row['milp'], row['benders']):
            removed = {tuple(edge) for edge in result['interdicted_edges']}
            endpoints = {v for arc, edge in zip(allowed, edges) if arc not in removed for v in edge}
            assert result['flow'] == len(endpoints)
            remaining = len(edges) - len(removed)
            assert remaining <= len(endpoints) * (len(endpoints) - 1) // 2

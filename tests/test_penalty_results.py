"""Audit retained solver witnesses against the original graph and compiled BQM."""
import hashlib
import itertools
import json
from pathlib import Path

import networkx as nx

from certified_penalties import certified_squared_qubo
import network_interdiction as ni

ROOT = Path(__file__).resolve().parents[1]


def test_retained_comparison_has_complete_cohort_and_valid_ground_witnesses():
    data = json.loads((ROOT / 'results/certified_penalties.json').read_text())
    assert len(data['rows']) == 10 and data['failures'] == []
    assert len({(r['instance'], r['budget']) for r in data['rows']}) == 10
    for row in data['rows']:
        path = ROOT / row['instance']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['input_sha256']
        graph, source, sink = ni.read_flow_file(path)
        bqm, certificate = certified_squared_qubo(row['budget'], graph, source, sink)
        assert set(row['sample']) == set(bqm.variables)
        assert set(row['sample'].values()) <= {0, 1}
        assert abs(bqm.energy(row['sample']) - row['qubo_energy']) < 1e-6
        assert abs(row['qubo_energy'] - row['qubo_bound']) < 1e-6
        assert abs(row['nip_optimum'] - row['nip_bound']) < 1e-6
        assert abs(row['qubo_energy'] + certificate.normalized_offset - row['nip_optimum']) < 1e-6
        selected = {edge for i, edge in enumerate(sorted(graph.edges()))
                    if row['sample'][f'x_{i}']}
        assert selected == {tuple(e) for e in row['interdicted_edges']}
        assert len(selected) == row['budget']
        residual = graph.copy()
        residual.remove_edges_from(selected)
        assert nx.minimum_cut_value(residual, source, sink, capacity='weight') == row['decoded_flow']
        assert row['decoded_flow'] == round(row['nip_optimum'])
        assert row['squared_couplers'] == sum(c != 0 for c in bqm.quadratic.values())
        m = graph.number_of_edges()
        assert row['squared_couplers'] - row['linear_couplers'] == m*(m-1)//2
        if row['linear_window'] is not None:
            # Independently recover f(k) on the <=6-arc saved fixtures.
            assert m <= 6
            for k, expected in enumerate(row['linear_window']['curve']):
                optimum = float('inf')
                for edges in itertools.combinations(graph.edges(), k):
                    r = graph.copy()
                    r.remove_edges_from(edges)
                    optimum = min(optimum, nx.minimum_cut_value(r, source, sink, capacity='weight'))
                assert optimum == expected


def test_campaign_keeps_the_failed_export_and_bounded_total_work():
    ledger = json.loads((ROOT / 'results/certified_penalties_attempts.json').read_text())
    assert [a['outcome'] for a in ledger['attempts']] == ['export_failed', 'completed']
    assert sum(a['completed_case_solves'] for a in ledger['attempts']) == 22
    assert ledger['max_solver_seconds_per_case_solve'] == 20
    assert ledger['attempts'][1]['result_sha256'] == hashlib.sha256(
        (ROOT / 'results/certified_penalties.json').read_bytes()).hexdigest()

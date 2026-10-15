"""Independent conversion, covering, bounds, and profiling checks."""
import gzip
import itertools
import json
from pathlib import Path

import networkx as nx
import pytest

import benders
import challenge_instances as ci
import run_challenge_profile as profile
from test_gurobi_trajectory import require_gurobi

pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")


def surviving(graph, s, t, removed):
    copy = graph.copy()
    copy.remove_edges_from(removed)
    return nx.maximum_flow_value(copy, s, t, capacity='weight')


def test_transport_decimal_rounding_and_direction():
    graph = ci.transport('<HEADER>\n~ comment\n1 2 2.5 ;\n2 1 9.49 ;\n')
    assert graph[1][2]['weight'] == 3
    assert graph[2][1]['weight'] == 9


def test_contact_duration_is_not_contact_count_or_doubled():
    graph = ci.contacts('20 1 2 4A 4A\n40 2 1 4A 4A\n40 1 3 4A 4A\n')
    assert graph[1][2]['weight'] == graph[2][1]['weight'] == 40
    assert graph[1][3]['weight'] == 20
    with pytest.raises(ValueError, match='unique'):
        ci.contacts('20 1 2 4A 4A\n20 2 1 4A 4A\n')


def test_power_uses_rateA_and_excludes_inactive_branch():
    graph = ci.power('mpc.branch = [\n1 2 0 0 0 7 9 10 0 0 1 -30 30;\n'
                     '2 3 0 0 0 8 9 10 0 0 0 -30 30;\n];')
    assert set(graph.edges()) == {(1, 2), (2, 1)}
    assert graph[1][2]['weight'] == 7


def test_conversion_regenerates_all_flow_bytes_and_manifest(tmp_path):
    ci.generate(tmp_path)
    expected = json.loads((ci.DATA / 'manifest.json').read_text())
    assert json.loads((tmp_path / 'manifest.json').read_text()) == expected
    for case in expected['cases']:
        if not case['file'].startswith('../'):
            assert (tmp_path / case['file']).read_bytes() == (ci.DATA / case['file']).read_bytes()
    assert {c['family'] for c in expected['cases']} >= {'road', 'contact', 'power'}


def test_covering_flow_equals_endpoint_union_for_every_retained_subset():
    graph, s, t, allowed, edges = ci.covering(4, 1, 0)
    assert len(edges) == 6
    assert all(s not in e and t not in e for e in allowed)
    # All 64 states, with no MILP solver or builder used for the oracle.
    for bits in itertools.product([0, 1], repeat=6):
        removed = [e for e, bit in zip(allowed, bits) if bit]
        endpoints = {v for edge, bit in zip(edges, bits) if not bit for v in edge}
        assert surviving(graph, s, t, removed) == len(endpoints)


@pytest.mark.parametrize('budget', [0, 1, 3, 6])
def test_bounded_solvers_match_independent_restricted_oracle(budget):
    require_gurobi()
    graph, s, t, allowed, _ = ci.covering(4, 1, 0)
    optimum = min(surviving(graph, s, t, removed)
                  for removed in itertools.combinations(allowed, budget))
    for result in [profile.milp_profile(graph, s, t, budget, set(allowed), time_limit=10),
                   benders.solve(graph, s, t, budget, interdictable=allowed, time_limit=10)]:
        assert result['status'] == 'optimal'
        assert result['flow'] == optimum
        assert result['lower_bound'] == pytest.approx(optimum)
        assert set(result['interdicted_edges']) <= set(allowed)
        assert result['node_count'] >= 0


def test_short_time_limit_still_returns_auditable_bounds_and_witness():
    require_gurobi()
    graph, s, t, allowed, _ = ci.covering(8, .5, 0)
    for result in [profile.milp_profile(graph, s, t, 3, set(allowed), time_limit=.00001),
                   benders.solve(graph, s, t, 3, interdictable=allowed, time_limit=.00001)]:
        assert result['status'] == 'time_limit'
        assert result['flow'] == surviving(graph, s, t, result['interdicted_edges'])
        assert 0 <= result['lower_bound'] <= result['flow']


def test_hardness_needs_both_strain_and_evidence_of_interior_optimum():
    slow = {'status': 'time_limit', 'runtime_sec': 20, 'any_terminal_incident': False}
    fast = dict(slow, status='optimal', runtime_sec=.1)
    terminal = dict(slow, status='optimal', any_terminal_incident=True)
    assert profile.classify(slow, slow, {(2, 3)}, 0, 1) == 'hard_for_both_measured_solvers'
    assert profile.classify(slow, slow, {(0, 3)}, 0, 1).startswith('unclassified')
    assert profile.classify(fast, slow, {(2, 3)}, 0, 1).startswith('not_hard')
    assert profile.classify(terminal, slow, {(0, 3)}, 0, 1).startswith('excluded')


def test_invalid_budget_and_resource_limits_fail_before_solver(monkeypatch):
    graph, s, t, allowed, _ = ci.covering(4, 1, 0)
    def forbidden(*args, **kwargs):
        pytest.fail('solver dispatched before validation')
    monkeypatch.setattr(profile.ni, '_gurobi_model', forbidden)
    with pytest.raises(ValueError):
        profile.milp_profile(graph, s, t, 1, set(allowed), time_limit=61)
    with pytest.raises(ValueError):
        benders.solve(graph, s, t, len(allowed) + 1, interdictable=allowed)
    graph.add_nodes_from(range(1001))
    with pytest.raises(ValueError):
        profile.milp_profile(graph, s, t, 1, set(allowed))


def test_profile_counts_actual_compiled_variables():
    graph, s, t, allowed, _ = ci.covering(4, 1, 0)
    case = {'name': 'test', 'family': 'covering', 'file': 'test.flow', 'sha256': 'test'}
    row = profile.profile_case(case, graph, s, t, 2, set(allowed), 10)
    assert row['qubits'] == len(graph) - 2 + 3 * len(allowed)
    assert row['allowed_only_interior']

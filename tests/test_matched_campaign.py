"""Campaign gates bind references and prevent oversized simulator dispatch."""
import json

import pytest

import challenge_instances as ci
import run_matched_benchmark as matched
import run_simulator_scaling as scaling


def test_references_bind_manifests_including_allowed_arc_sets(tmp_path, monkeypatch):
    reference = json.loads((matched.ROOT/'results/challenge_profile.json').read_text())
    reference['manifest_sha256'] = 'changed'
    path = tmp_path/'reference.json'
    path.write_text(json.dumps(reference))
    monkeypatch.setattr(matched, 'PAIRS', [('testcases/challenge/manifest.json',str(path))])
    with pytest.raises(ValueError, match='stale reference manifest'):
        matched.cases_and_references()


def test_retained_reference_selection_has_exact_inputs_and_optimal_witnesses():
    references = matched.cases_and_references()
    for key in matched.SELECTION:
        ref = references[key]
        assert ref['archived']['milp']['status'] == 'optimal'
        assert ref['archived']['sha256'] == ref['case']['sha256']


@pytest.mark.parametrize('method', ['penalty_statevector','xy_statevector','penalty_mps','xy_mps'])
def test_hard_case_refused_before_sampler_or_reference_dispatch(method, monkeypatch):
    import solve_qubo_via_qaoa
    import run_challenge_profile
    monkeypatch.setattr(solve_qubo_via_qaoa,'solve_bqm_qaoa',lambda *a,**k:pytest.fail('sampler dispatched'))
    monkeypatch.setattr(run_challenge_profile,'milp_profile',lambda *a,**k:pytest.fail('reference dispatched'))
    result=scaling.worker('testcases/challenge/cover_n30_s0.flow',192,method)
    assert result['status']=='qubit_guard_refused'
    assert result['qubits']==1095


def test_fit_compiles_actual_restricted_pieces_and_schema(monkeypatch):
    import profile_embedding
    monkeypatch.setattr(matched,'hard_embedding_attempt',lambda source,target,**kw:
                        {'status':'not_found_within_budget','logical_variables':len(source)})
    g,s,t,allowed,_=ci.covering(4,1,0)
    result=matched.fit(g,s,t,3,set(allowed))
    assert result['status']=='completed'
    assert len(result['rows'])==5
    assert not result['all_pieces_embedded']
    assert result['rows'][0]['allowed_arcs']==6
    assert sum(row['allowed_arcs'] for row in result['rows'][1:])==6
    assert all(row['qci_schema']['status']=='accepted_offline' for row in result['rows'])


def slow_embedding_child(connection, source, target):
    import time
    time.sleep(10)


def test_embedding_outer_limit_contains_native_overrun(monkeypatch):
    import time
    import networkx as nx
    monkeypatch.setattr(matched,'_embedding_child',slow_embedding_child)
    start=time.perf_counter()
    result=matched.hard_embedding_attempt(nx.path_graph(3),nx.path_graph(4),seconds=.05)
    assert result['status']=='embedding_outer_time_limit'
    assert time.perf_counter()-start<2
    assert result['logical_variables']==3


def test_actual_embedding_process_returns_an_auditable_mapping():
    import networkx as nx
    from profile_embedding import audit_embedding
    source,target=nx.path_graph(3),nx.path_graph(5)
    result=matched.hard_embedding_attempt(source,target)
    assert result['status']=='embedded'
    check=audit_embedding(source,target,{r['variable']:r['chain'] for r in result['embedding']})
    assert check['physical_qubits']==result['physical_qubits']

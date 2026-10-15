"""Independent minor-certificate checks and actual bounded minorminer calls."""
import dimod
import networkx as nx
import pytest

import profile_embedding as pe


def test_source_retains_isolates_but_omits_zero_couplers():
    bqm = dimod.BQM({'a': 1, 'b': 0, 'isolated': 0}, {('a', 'b'): 2, ('b', 'isolated'): 0}, 0, 'BINARY')
    graph = pe.interaction_graph(bqm)
    assert set(graph) == {'a', 'b', 'isolated'}
    assert set(graph.edges) == {('a', 'b')}
    result = pe.embedding_attempt(graph, nx.path_graph(5), seed=0, timeout=1)
    assert result['status'] == 'embedded' and result['physical_qubits'] == 3
    assert len(result['embedding']) == 3


@pytest.mark.parametrize('mapping,reason', [
    ({'a': [0]}, 'cover'),
    ({'a': [], 'b': [1]}, 'empty'),
    ({'a': [0, 0], 'b': [1]}, 'duplicated'),
    ({'a': [9], 'b': [1]}, 'outside'),
    ({'a': [0, 1], 'b': [1]}, 'overlap'),
    ({'a': [0, 2], 'b': [1]}, 'disconnected'),
    ({'a': [0], 'b': [3]}, 'coupler'),
])
def test_invalid_certificates_are_rejected(mapping, reason):
    with pytest.raises(ValueError, match=reason):
        pe.audit_embedding(nx.Graph([('a', 'b')]), nx.path_graph(4), mapping)


def test_complete_triangle_certificate_and_statistics():
    source, target = nx.complete_graph(3), nx.cycle_graph(4)
    result = pe.embedding_attempt(source, target, seed=0, timeout=1)
    assert result['status'] == 'embedded'
    assert result['physical_qubits'] == 4
    assert result['max_chain_length'] == 2
    assert result['overhead_ratio'] == pytest.approx(4/3)
    assert result['chain_length_histogram'] == {'1': 2, '2': 1}
    mapping = {row['variable']: row['chain'] for row in result['embedding']}
    # Contracting all chains must yield the source triangle.
    contracted = nx.Graph()
    contracted.add_nodes_from(source)
    reverse = {q: v for v, chain in mapping.items() for q in chain}
    contracted.add_edges_from((reverse[u], reverse[v]) for u, v in target.edges
                             if reverse[u] != reverse[v])
    assert nx.is_isomorphic(contracted, source)


def test_size_refusal_and_failure_are_distinct(monkeypatch):
    monkeypatch.setattr(pe.minorminer, 'find_embedding', lambda *a, **k: pytest.fail('solver dispatched'))
    result = pe.embedding_attempt(nx.path_graph(4), nx.path_graph(3))
    assert result['status'] == 'too_many_logical_variables'
    assert result['embedding_time_sec'] is None
    monkeypatch.setattr(pe.minorminer, 'find_embedding', lambda *a, **k: {})
    result = pe.embedding_attempt(nx.complete_graph(4), nx.path_graph(4))
    assert result['status'] == 'not_found_within_budget'
    assert result['physical_qubits'] is None and result['embedding'] is None


def test_seed_and_work_controls_reach_native_boundary(monkeypatch):
    calls = []
    def finder(source, target, **options):
        calls.append(options)
        return {0: [0], 1: [1]}
    monkeypatch.setattr(pe.minorminer, 'find_embedding', finder)
    pe.embedding_attempt(nx.path_graph(2), nx.path_graph(3), seed=71, timeout=0.25)
    assert calls == [dict(random_seed=71, timeout=0.25, tries=1, threads=1)]


def test_invalid_controls_and_invalid_return_fail_closed(monkeypatch):
    monkeypatch.setattr(pe.minorminer, 'find_embedding', lambda *a, **k: pytest.fail('solver dispatched'))
    for options in [dict(seed=-1), dict(seed=True), dict(timeout=0), dict(timeout=float('nan'))]:
        with pytest.raises(ValueError):
            pe.embedding_attempt(nx.path_graph(2), nx.path_graph(3), **options)
    monkeypatch.setattr(pe.minorminer, 'find_embedding', lambda *a, **k: {0: [0]})
    with pytest.raises(ValueError, match='cover'):
        pe.embedding_attempt(nx.path_graph(2), nx.path_graph(3))


def test_target_identities_and_seeded_repeatability():
    for name, sizes in [('P16', (5640, 40484)), ('Z12', (4800, 45864))]:
        target = pe.topology(name)
        assert (len(target), target.number_of_edges()) == sizes
        source = nx.complete_graph(5)
        a = pe.embedding_attempt(source, target, seed=3, timeout=5)
        b = pe.embedding_attempt(source, target, seed=3, timeout=5)
        assert a['status'] == b['status'] == 'embedded'
        assert a['embedding'] == b['embedding']
        assert a['qpu_time_sec'] is None

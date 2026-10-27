"""The uniform benchmark uses actual Aer shots with the original BQM labels."""
import importlib.util
import json
import os
from pathlib import Path

import dimod
import numpy as np
import pytest

import run_benchmark as rb
import solve_qubo_via_qaoa as sampler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def aer():
    if any(importlib.util.find_spec(name) is None for name in ('qiskit', 'qiskit_aer')):
        if os.environ.get('REQUIRE_QAOA_BACKENDS'):
            pytest.fail('Qiskit/Aer required in this lane')
        pytest.skip('optional Qiskit/Aer backend')
    import qaoa_qiskit
    return qaoa_qiskit


@pytest.mark.parametrize('options', [{'shots': 0}, {'reps': 0}, {'maxiter': 3},
    {'reps': 2, 'maxiter': 5}, {'seed': -1}, {'max_qubits': 25}, {'reps': True}])
def test_invalid_options_fail_before_backend_import(options, monkeypatch):
    import builtins
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name in ('qaoa_common', 'qaoa_qiskit'):
            pytest.fail('backend work dispatched')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    with pytest.raises(ValueError):
        sampler.solve_bqm_qaoa(dimod.BQM({'a': 1}, {}, 0, 'BINARY'), **options)


def test_large_qaoa_case_fails_before_exact_reference_or_window(monkeypatch):
    monkeypatch.setattr(rb, 'optimum_curve', lambda *a: pytest.fail('exact curve dispatched'))
    monkeypatch.setattr(rb, 'exact_reference', lambda *a: pytest.fail('exact reference dispatched'))
    with pytest.raises(ValueError, match='1..13 variables'):
        rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 'auto', 'brute', ['qaoa'],
                    qaoa_options={'max_qubits': 13})


def test_aer_samples_match_independent_cost_phases_and_bit_order(aer):
    bqm = dimod.BQM({'last': -3, 'first': 1, 'middle': 2},
                    {('last', 'first'): -5, ('first', 'middle'): 1.5}, 11, 'BINARY')
    samples = sampler.solve_bqm_qaoa(bqm, shots=8192, maxiter=8, seed=7)
    meta = samples.info['qaoa_metadata']
    variables = meta['variables']
    bits = np.array([[(i >> q) & 1 for q in range(3)] for i in range(8)])
    energies = bqm.energies((bits, variables))
    beta, gamma = meta['theta']
    # Constant phase is immaterial. Direct BQM energies avoid the Ising converter.
    psi = np.exp(-1j*gamma*energies/meta['hamiltonian_scale']) / np.sqrt(8)
    for q in range(3):
        psi = np.cos(beta)*psi - 1j*np.sin(beta)*psi[np.arange(8) ^ (1 << q)]
    expected = np.abs(psi)**2
    observed = np.zeros(8)
    for record in samples.data():
        index = sum(int(record.sample[v]) << q for q, v in enumerate(variables))
        observed[index] += record.num_occurrences / 8192
        assert record.energy == bqm.energy(record.sample)
    assert observed == pytest.approx(expected, abs=0.025)
    assert meta['expected_energy'] == pytest.approx(float(expected @ energies), abs=1e-10)
    assert sum(samples.record.num_occurrences) == 8192
    assert 1 <= meta['evaluations'] <= 8
    assert meta['num_qubits'] == 3 and meta['two_qubit_gates'] == 2
    assert all(meta[name] >= 0 for name in ('build_time_sec', 'train_time_sec', 'sample_time_sec'))


def test_same_seed_reproduces_angles_and_counts(aer):
    bqm = dimod.BQM({'a': -3, 'b': 1}, {('a', 'b'): 2}, 5, 'BINARY')
    first = sampler.solve_bqm_qaoa(bqm, shots=128, maxiter=6, seed=9)
    second = sampler.solve_bqm_qaoa(bqm, shots=128, maxiter=6, seed=9)
    assert first.info['qaoa_metadata']['theta'] == second.info['qaoa_metadata']['theta']
    assert np.array_equal(first.record, second.record)


def test_qaoa_benchmark_scores_actual_shots_against_shared_reference(aer):
    row = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 6, 'brute', ['qaoa'],
                      sa_seed=0, qaoa_options={'shots': 128, 'maxiter': 4})
    assert row['Opt_NIP'] == 5
    assert row['QAOA_NIP_Greedy'] >= row['Opt_NIP']
    assert row['QAOA_QUBO'] >= row['Opt_QUBO']
    records = json.loads(row['QAOA_raw_samples'])
    assert 1 <= len(records) <= 20
    graph, src, tgt = rb.ni.read_flow_file(ROOT / 'testcases/tiny_a.flow')
    for record in records:
        assert record['flow'] == rb.max_flow(graph, src, tgt, record['interdicted_edges'])
        assert record['budget_feasible'] == (len(record['interdicted_edges']) <= 1)
    assert row['QAOA_raw_samples'] == row['QAOA_pre_greedy_samples']
    meta = json.loads(row['QAOA_metadata'])
    assert meta['shots'] == 128 and meta['evaluations'] <= 4
    assert row['QAOA_time_sec'] >= sum(meta[field] for field in
                                     ('build_time_sec', 'train_time_sec', 'sample_time_sec'))
    assert all(row[c] == '' for solver in ('sa', 'qa', 'qci') for c in rb.SOLVER_COLUMNS[solver])


def test_unselected_qaoa_and_empty_window_remain_unmeasured(monkeypatch, tmp_path):
    from test_lambda_window import write_two_equal_paths
    monkeypatch.setattr(sampler, 'solve_bqm_qaoa', lambda *a, **k: pytest.fail('QAOA dispatched'))
    path = tmp_path / 'equal.flow'
    write_two_equal_paths(path)
    for instance, lam, solvers in [(path, 'auto', ['qaoa']),
                                  (ROOT / 'testcases/tiny_a.flow', 6, [])]:
        row = rb.run_case(str(instance), 1, lam, 'brute', solvers)
        assert all(value == '' for key, value in row.items() if key.startswith('QAOA_'))


@pytest.mark.parametrize('options', [
    {'method':'mps', 'max_qubits':257, 'max_bond_dimension':8},
    {'method':'mps', 'max_bond_dimension':None},
    {'method':'mps', 'max_bond_dimension':65},
    {'method':'mps', 'max_bond_dimension':True},
    {'method':'mps', 'max_bond_dimension':8, 'reps':4},
    {'method':'mps', 'max_bond_dimension':8, 'shots':10001},
    {'method':'statevector', 'max_bond_dimension':8}])
def test_mps_limits_precede_optional_backend_work(options, monkeypatch):
    import builtins
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name in ('qaoa_common','qaoa_qiskit'):
            pytest.fail('backend work dispatched')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(ValueError):
        sampler.solve_bqm_qaoa(dimod.BQM({'a':1},{},0,'BINARY'), **options)


def test_actual_capped_mps_matches_small_statevector_energy_and_labels(aer):
    bqm = dimod.BQM({'third':-3,'first':1,'second':2},
                    {('third','first'):-5,('first','second'):1.5},11,'BINARY')
    exact = sampler.solve_bqm_qaoa(bqm, shots=1024, maxiter=8, seed=7, order='rcm')
    approx = sampler.solve_bqm_qaoa(bqm, shots=1024, maxiter=8, seed=7, order='rcm',
                                   method='mps', max_bond_dimension=8, max_qubits=256)
    em,am = [r.info['qaoa_metadata'] for r in [exact,approx]]
    assert am['expected_energy'] == pytest.approx(em['expected_energy'], abs=1e-9)
    assert am['theta'] == pytest.approx(em['theta'], abs=1e-9)
    assert am['max_bond_dimension']==8 and am['approximate']
    assert am['mps_truncation_threshold']==1e-16
    assert am['variables']==em['variables']
    assert sum(approx.record.num_occurrences)==1024
    for record in approx.data():
        assert record.energy==bqm.energy(record.sample)


@pytest.mark.parametrize('options',[{'max_qubits':1201},{'max_bond_dimension':9},{'reps':2},
                                    {'maxiter':5},{'shots':257},{'max_bond_dimension':'8'}])
def test_extended_mps_enforces_its_tighter_resource_envelope(options):
    controls=dict(method='mps',extended_mps=True,max_qubits=1200,max_bond_dimension=8,
                  reps=1,shots=256,maxiter=4)
    controls.update(options)
    with pytest.raises(ValueError):
        sampler.validate_options(**controls)


def test_extended_mps_dense_bqm_refused_before_backend_import(monkeypatch):
    import builtins
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name in ('qaoa_common','qaoa_qiskit'):
            pytest.fail('backend dispatched')
        return original(name,*args,**kwargs)
    # 317*316/2 = 50,086, above the dedicated compiled-coupling envelope.
    bqm=dimod.BQM({i:1 for i in range(317)},
                  {(i,j):1 for i in range(317) for j in range(i)},0,'BINARY')
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(ValueError,match='50000'):
        sampler.solve_bqm_qaoa(bqm,method='mps',extended_mps=True,max_qubits=1200,
                              max_bond_dimension=8,maxiter=4,shots=256)


def test_extended_mps_matches_small_default_mps(aer):
    bqm=dimod.BQM({'a':-2,'b':1},{('a','b'):3},0,'BINARY')
    controls=dict(method='mps',max_bond_dimension=8,maxiter=4,shots=128,seed=3)
    first=sampler.solve_bqm_qaoa(bqm,max_qubits=256,**controls)
    second=sampler.solve_bqm_qaoa(bqm,extended_mps=True,max_qubits=1200,**controls)
    assert np.array_equal(first.record,second.record)
    assert first.info['qaoa_metadata']['theta']==second.info['qaoa_metadata']['theta']

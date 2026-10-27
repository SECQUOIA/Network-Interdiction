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

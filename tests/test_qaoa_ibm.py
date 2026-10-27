"""Actual Runtime local jobs and routed classical-bit semantics, without HTTP."""
import importlib.util
import json
import os
from types import SimpleNamespace

import dimod
import numpy as np
import pytest

import qaoa_ibm as ibm


@pytest.fixture
def sdk(monkeypatch):
    if any(importlib.util.find_spec(name) is None for name in
           ('qiskit', 'qiskit_aer', 'qiskit_ibm_runtime')):
        if os.environ.get('REQUIRE_IBM_BACKEND'):
            pytest.fail('IBM SDK and Aer required in this lane')
        pytest.skip('optional IBM Runtime dependencies')
    import requests
    monkeypatch.setattr(requests.sessions.Session, 'request',
                        lambda *a, **k: pytest.fail('HTTP is forbidden in offline IBM tests'))
    import qiskit_ibm_runtime.executor_sampler
    return qiskit_ibm_runtime.executor_sampler


@pytest.fixture
def bqm():
    return dimod.BQM({'last': -3, 'first': 1, 'middle': 2},
                    {('last', 'first'): -5, ('first', 'middle'): 1.5,
                     ('middle', 'last'): 2}, 11, 'BINARY')


@pytest.mark.parametrize('options', [dict(theta=[1]), dict(theta=[float('nan'), 1]),
    dict(theta=[1, 2], shots=0), dict(theta=[1, 2], max_depth=True),
    dict(theta=[1, 2], result_timeout_sec=0), dict(theta=[1, 2], max_execution_time_sec=10801)])
def test_invalid_controls_before_optional_import(options, bqm):
    with pytest.raises(ValueError):
        ibm.solve_bqm_ibm(bqm, backend=None, **options)


def test_routed_counts_keep_logical_bit_order_and_original_energies(sdk, bqm, monkeypatch):
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit_aer import AerSimulator
    import qiskit.transpiler.preset_passmanagers as managers
    target = GenericBackendV2(5, coupling_map=[[i, j] for i in range(5) for j in range(5)
        if abs(i-j) == 1], basis_gates=['id', 'rz', 'sx', 'x', 'cx'], noise_info=False, seed=0)
    backend = AerSimulator.from_backend(target)
    original = managers.generate_preset_pass_manager
    def placed(**kwargs):
        return original(initial_layout=[4, 2, 0], **kwargs)
    monkeypatch.setattr(managers, 'generate_preset_pass_manager', placed)
    theta = [0.31, 0.73]
    samples = ibm.solve_bqm_ibm(bqm, backend=backend, theta=theta, shots=8192)
    meta = samples.info['qaoa_metadata']
    assert meta['final_layout'] != [0, 1, 2]
    assert meta['active_physical_qubits'] > len(bqm)
    variables = meta['variables']
    bits = np.array([[(i >> q) & 1 for q in range(3)] for i in range(8)])
    energies = bqm.energies((bits, variables))
    psi = np.exp(-1j*theta[1]*energies/meta['hamiltonian_scale'])/np.sqrt(8)
    for q in range(3):
        psi = np.cos(theta[0])*psi - 1j*np.sin(theta[0])*psi[np.arange(8) ^ (1 << q)]
    observed = np.zeros(8)
    for record in samples.data():
        index = sum(int(record.sample[v]) << q for q, v in enumerate(variables))
        observed[index] += record.num_occurrences / 8192
        assert record.energy == bqm.energy(record.sample)
    assert observed == pytest.approx(np.abs(psi)**2, abs=0.025)
    assert meta['job_id'] and meta['execution'] == 'local'
    assert meta['runtime_usage'] is None and meta['usage_status'] == 'not_applicable'
    assert meta['shots'] == 8192


def test_real_fake_backend_job_and_target_instruction_audit(sdk, bqm):
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    backend = FakeManilaV2()
    circuit, variables, meta = ibm.prepare_circuit(bqm, backend, theta=[0.2, 0.4])
    for item in circuit.data:
        if item.operation.name != 'barrier':
            assert backend.target.instruction_supported(operation_name=item.operation.name,
                qargs=tuple(circuit.find_bit(q).index for q in item.qubits))
    assert circuit.num_clbits == len(variables) == len(bqm)
    samples = ibm.solve_bqm_ibm(bqm, backend=backend, theta=[0.2, 0.4], shots=128)
    assert sum(samples.record.num_occurrences) == 128
    assert meta['two_qubit_gates'] > 3


def test_oversize_and_depth_fail_before_sampler(sdk, bqm, monkeypatch):
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    monkeypatch.setattr(sdk, 'Sampler', lambda **k: pytest.fail('sampler dispatched'))
    backend = FakeManilaV2()
    with pytest.raises(ValueError, match='1..5 variables'):
        ibm.solve_bqm_ibm(dimod.BQM({i: 1 for i in range(6)}, {}, 0, 'BINARY'),
                          backend=backend, theta=[0.2, 0.4])
    with pytest.raises(ValueError, match='routed depth'):
        ibm.solve_bqm_ibm(bqm, backend=backend, theta=[0.2, 0.4], max_depth=1)


@pytest.mark.parametrize('failure', ['timeout', 'error', 'malformed'])
def test_failures_preserve_receipt_without_resubmitting(sdk, bqm, monkeypatch, failure):
    from qiskit_aer import AerSimulator
    calls = []
    class Job:
        def job_id(self): return 'retained-job-17'
        def in_final_state(self): return failure != 'timeout'
        def result(self):
            if failure == 'error': raise ValueError('untrusted response secret')
            return [SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: {'11': 128})))]
    class Sampler:
        def __init__(self, **kwargs): pass
        def run(self, pubs, shots):
            calls.append((pubs, shots))
            return Job()
    monkeypatch.setattr(sdk, 'Sampler', Sampler)
    with pytest.raises(RuntimeError, match='retained-job-17') as error:
        ibm.solve_bqm_ibm(bqm, backend=AerSimulator(), theta=[0.2, 0.4], shots=128,
                          result_timeout_sec=0.001)
    assert len(calls) == 1 and 'untrusted response secret' not in str(error.value)


def test_shared_benchmark_uses_runtime_samples_and_exact_reference(sdk):
    from qiskit_aer import AerSimulator
    import run_benchmark as rb
    row = rb.run_case('testcases/tiny_a.flow', 1, 6, 'brute', ['qaoa'], sa_seed=0,
        qaoa_options=dict(engine='ibm', backend=AerSimulator(), theta=[0.3, 0.7], shots=128))
    assert row['Opt_NIP'] == 5 and row['QAOA_NIP_Greedy'] >= 5
    meta = json.loads(row['QAOA_metadata'])
    assert meta['execution'] == 'local' and meta['method'] == 'runtime-fixed-angle'
    assert meta['shots'] == 128 and meta['num_qubits'] == 14
    assert row['QAOA_time_sec'] >= meta['transpile_time_sec'] + meta['sample_time_sec']
    records = json.loads(row['QAOA_raw_samples'])
    graph, source, sink = rb.ni.read_flow_file('testcases/tiny_a.flow')
    for record in records:
        assert record['flow'] == rb.max_flow(graph, source, sink, record['interdicted_edges'])
    assert row['QAOA_raw_samples'] == row['QAOA_pre_greedy_samples']


def test_runtime_size_guard_precedes_exact_reference(sdk, monkeypatch):
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    import run_benchmark as rb
    monkeypatch.setattr(rb, 'exact_reference', lambda *a: pytest.fail('reference dispatched'))
    monkeypatch.setattr(rb, 'optimum_curve', lambda *a: pytest.fail('window dispatched'))
    with pytest.raises(ValueError, match='1..5 variables'):
        rb.run_case('testcases/tiny_a.flow', 1, 'auto', 'brute', ['qaoa'],
            qaoa_options=dict(engine='ibm', backend=FakeManilaV2(), theta=[0.3, 0.7]))


def test_active_routing_qubits_guard_local_simulation(sdk, bqm, monkeypatch):
    from qiskit_aer import AerSimulator
    monkeypatch.setattr(ibm, 'prepare_circuit', lambda *a, **k:
                        (None, list(bqm.variables), {'active_physical_qubits': 25}))
    monkeypatch.setattr(sdk, 'Sampler', lambda **k: pytest.fail('sampler dispatched'))
    with pytest.raises(ValueError, match='active routed qubits'):
        ibm.solve_bqm_ibm(bqm, backend=AerSimulator(), theta=[0.2, 0.4])


def test_cli_missing_angles_fails_before_service(sdk, monkeypatch, tmp_path):
    import qiskit_ibm_runtime
    import run_benchmark as rb
    monkeypatch.setattr(qiskit_ibm_runtime, 'QiskitRuntimeService',
                        lambda: pytest.fail('service created'))
    with pytest.raises(SystemExit) as error:
        rb.main(['--case', 'testcases/tiny_a.flow:1:6', '--solvers', 'qaoa',
                 '--qaoa-engine', 'ibm', '--ibm-backend', 'explicit-device',
                 '--out', str(tmp_path/'out.csv')])
    assert error.value.code == 2

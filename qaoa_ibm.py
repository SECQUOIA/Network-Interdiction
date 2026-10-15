"""Fixed-angle penalty-QAOA through IBM Runtime's client-side Sampler.

Preparation is offline. Sampling submits one job to the supplied backend; a
FakeBackendV2 or Aer backend uses the SDK's local mode. No service/account is
created here, and there is no optimization loop or automatic job resubmission.
"""
import importlib.metadata
import math
import numbers
import time
from types import SimpleNamespace

import dimod
import numpy as np

from solve_qubo_via_qaoa import validate_bqm


def is_local_backend(backend):
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
    from qiskit_aer import AerSimulator
    return isinstance(backend, (FakeBackendV2, AerSimulator))


def validate_options(*, theta, reps=1, shots=2500, seed_transpiler=0,
                     max_qubits=127, max_depth=10000, result_timeout_sec=300,
                     max_execution_time_sec=300):
    for name, value, low, high in (
        ('reps', reps, 1, 100), ('shots', shots, 1, 1000000),
        ('seed_transpiler', seed_transpiler, 0, 2**32-1),
        ('max_qubits', max_qubits, 1, 1000), ('max_depth', max_depth, 1, 1000000),
        ('max_execution_time_sec', max_execution_time_sec, 1, 10800),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not low <= value <= high:
            raise ValueError(f'IBM {name} must be an integer in {low}..{high}')
    angles = np.asarray(theta, dtype=float)
    if angles.shape != (2*reps,) or not np.all(np.isfinite(angles)):
        raise ValueError('IBM theta must contain 2*reps finite angles: beta layers, then gamma layers')
    if isinstance(result_timeout_sec, bool) or not math.isfinite(result_timeout_sec) or result_timeout_sec <= 0:
        raise ValueError('IBM result_timeout_sec must be finite and positive')


def prepare_circuit(bqm, backend, *, theta, reps=1, seed_transpiler=0,
                    max_qubits=127, max_depth=10000):
    """Return a measured ISA circuit, original labels, and resource metadata.

    Measurements are inserted BEFORE routing. Classical bit i therefore remains
    the ith BQM variable, regardless of initial placement, swaps or idle ancillas.
    The depth guard covers the parameterized circuit before binding angles.
    Transpilation allocates circuits, never a statevector, and contacts no service.
    """
    validate_options(theta=theta, reps=reps, seed_transpiler=seed_transpiler,
                     max_qubits=max_qubits, max_depth=max_depth)
    validate_bqm(bqm, min(max_qubits, backend.num_qubits))
    start = time.perf_counter()
    import qaoa_common as qc
    from qaoa_qiskit import cost_operator
    from qiskit.circuit.library import qaoa_ansatz
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    variables = list(bqm.variables)
    h, J, offset = qc.ising_from_bqm(bqm, variables)
    scale = float(max(np.max(np.abs(h)), max(map(abs, J.values()), default=0)))
    if scale == 0 or not all(math.isfinite(v) for v in [*h, *J.values(), offset]):
        raise ValueError('IBM QAOA needs a nonconstant finite Hamiltonian')
    problem = SimpleNamespace(h=h, J=J, offset=offset, scale=scale, num_qubits=len(variables))
    circuit = qaoa_ansatz(cost_operator(problem), reps=int(reps))
    names = [p.name for p in circuit.parameters]
    if names != [f'β[{i}]' for i in range(reps)] + [f'γ[{i}]' for i in range(reps)]:
        raise RuntimeError('unexpected QAOA parameter ordering')
    circuit.measure_all()
    manager = generate_preset_pass_manager(backend=backend, optimization_level=1,
                                          seed_transpiler=int(seed_transpiler))
    isa = manager.run(circuit)
    depth = isa.depth()
    if depth > max_depth:
        raise ValueError(f'IBM routed depth {depth} exceeds max_depth={max_depth}')
    active = {isa.find_bit(q).index for item in isa.data
              if item.operation.name != 'barrier' for q in item.qubits}
    metadata = {
        'backend': backend.name, 'method': 'runtime-fixed-angle',
        'num_qubits': len(variables), 'variables': variables,
        'reps': int(reps), 'theta': [float(x) for x in theta],
        'angle_order': 'beta layers, then gamma layers', 'hamiltonian_scale': scale,
        'seed_transpiler': int(seed_transpiler), 'optimization_level': 1,
        'max_qubits': int(max_qubits), 'max_depth': int(max_depth),
        'target_qubits': backend.num_qubits, 'circuit_register_qubits': isa.num_qubits,
        'active_physical_qubits': len(active), 'circuit_depth': depth,
        'two_qubit_gates': sum(len(item.qubits) == 2 and item.operation.name != 'barrier'
                               for item in isa.data),
        'operation_counts': dict(isa.count_ops()),
        'final_layout': isa.layout.final_index_layout() if isa.layout else list(range(len(variables))),
        'classical_bit_order': 'BQM variables; routing is already accounted for by measurements',
        'transpile_time_sec': time.perf_counter() - start,
        'qiskit': importlib.metadata.version('qiskit'),
        'qiskit_ibm_runtime': importlib.metadata.version('qiskit-ibm-runtime'),
    }
    return isa.assign_parameters(list(theta)), variables, metadata


def solve_bqm_ibm(bqm, *, backend, theta, reps=1, shots=2500, seed_transpiler=0,
                  max_qubits=127, max_depth=10000, result_timeout_sec=300,
                  max_execution_time_sec=300):
    """Sample supplied angles once; return original BQM energies and a receipt.

    Result polling has a cooperative deadline (individual SDK/network calls may
    overrun it). A timeout preserves the job ID, does not cancel the job, and
    never resubmits it. The service execution-time cap excludes queue time.
    Runtime usage is service accounting, not circuit duration. Local runs leave
    that usage unmeasured even though the SDK returns placeholder zeros.
    """
    validate_options(theta=theta, reps=reps, shots=shots, seed_transpiler=seed_transpiler,
                     max_qubits=max_qubits, max_depth=max_depth,
                     result_timeout_sec=result_timeout_sec,
                     max_execution_time_sec=max_execution_time_sec)
    from qiskit_ibm_runtime.executor_sampler import Sampler
    local = is_local_backend(backend)
    # The fake backend can expand the register; Aer discards idle ancillas.
    # Bound active logical input before any local sampling dispatch.
    validate_bqm(bqm, min(max_qubits, backend.num_qubits, 24 if local else max_qubits))
    isa, variables, metadata = prepare_circuit(
        bqm, backend, theta=theta, reps=reps, seed_transpiler=seed_transpiler,
        max_qubits=max_qubits, max_depth=max_depth)
    if local and metadata['active_physical_qubits'] > 24:
        raise ValueError('local Runtime sampling requires at most 24 active routed qubits')
    options = {'max_execution_time': int(max_execution_time_sec),
               'twirling': {'enable_gates': False, 'enable_measure': False},
               'dynamical_decoupling': {'enable': False}}
    if local:
        options['simulator'] = {'seed_simulator': int(seed_transpiler)}
    sampler = Sampler(mode=backend, options=options)
    start = time.perf_counter()
    job = sampler.run([isa], shots=int(shots))
    job_id = job.job_id()
    try:
        deadline = start + result_timeout_sec
        while not job.in_final_state():
            if time.perf_counter() >= deadline:
                raise TimeoutError('result deadline exceeded')
            time.sleep(min(0.1, max(0, deadline-time.perf_counter())))
        counts = job.result()[0].data.meas.get_counts()
        if not counts or sum(counts.values()) != shots:
            raise ValueError('unexpected total shot count')
        for key, count in counts.items():
            if (len(key) != len(variables) or set(key) - {'0', '1'} or
                    isinstance(count, bool) or not isinstance(count, numbers.Integral) or count <= 0):
                raise ValueError('invalid bitstring or occurrence count')
        samples = dimod.SampleSet.from_samples_bqm(
            ([tuple(int(bit) for bit in reversed(key)) for key in counts], variables),
            bqm, num_occurrences=list(counts.values()))
    except Exception as error:
        raise RuntimeError(f'IBM job {job_id}: result unavailable or invalid ({type(error).__name__}); '
                           'inspect this job before submitting another') from None
    metadata.update(job_id=job_id, execution='local' if local else 'hardware',
                    shots=int(shots), sample_time_sec=time.perf_counter()-start,
                    result_timeout_sec=result_timeout_sec,
                    max_execution_time_sec=int(max_execution_time_sec),
                    runtime_usage=None, usage_status='not_applicable' if local else 'unavailable')
    if not local:
        try:
            usage = job.metrics().get('usage', {})
            metadata['runtime_usage'] = {key: usage[key] for key in
                ('status', 'seconds', 'quantum_seconds', 'qpu_charge_time_seconds') if key in usage}
            metadata['usage_status'] = usage.get('status', 'unspecified')
        except Exception:
            # Metrics retrieval must not discard already validated samples.
            pass
    samples.info['qaoa_metadata'] = metadata
    return samples

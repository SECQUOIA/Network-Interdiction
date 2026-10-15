"""Sample a binary BQM with Aer QAOA and optional fixed-weight XY mixing.

This adapter trains and samples only. Exact NIP references and greedy repair
belong to run_benchmark/qubo_verify_parameters, so they are shared with SA/QA/QCI
and are not silently repeated inside the sampler. No IBM service is contacted.
"""
import importlib.metadata
import math
import numbers
import time
from types import SimpleNamespace

import dimod
import numpy as np


def validate_options(*, reps=1, shots=2500, maxiter=100, seed=0, max_qubits=24):
    """Reject invalid resource controls before any optional backend import."""
    for name, value in [('reps', reps), ('shots', shots), ('maxiter', maxiter),
                        ('max_qubits', max_qubits), ('seed', seed)]:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f'QAOA {name} must be an integer')
    if reps < 1 or shots < 1 or maxiter < 2*reps + 2:
        raise ValueError('QAOA needs reps/shots >=1 and maxiter >=2*reps+2 (COBYLA minimum)')
    if not 1 <= max_qubits <= 24:
        raise ValueError('this statevector adapter requires max_qubits in 1..24')
    if not 0 <= seed < 2**32:
        raise ValueError('QAOA seed must be in 0..2**32-1')


def validate_bqm(bqm, max_qubits):
    """Validate input size/domain without constructing a circuit."""
    if bqm.vartype is not dimod.BINARY or not 1 <= len(bqm) <= max_qubits:
        raise ValueError(f'QAOA requires a BINARY BQM with 1..{max_qubits} variables')
    if not all(math.isfinite(value) for value in
               [bqm.offset, *bqm.linear.values(), *bqm.quadratic.values()]):
        raise ValueError('QAOA coefficients and offset must be finite')


def solve_bqm_qaoa(bqm, *, reps=1, shots=2500, maxiter=100, seed=0, max_qubits=24,
                   fixed_weight_variables=None, hamming_weight=None, initial_state='basis'):
    """Return raw SampleSet records with original BQM energies and run metadata.

    Uses one COBYLA start (the existing linear ramp), exact Aer statevector
    expectations, and seeded Aer shots. The best <=20 distinct observed records
    are selected later by the shared NIP scorer. The 24-qubit hard cap is a local
    memory guard, not a hardware-fit claim. All three stage times are wall time;
    building includes imports and Hamiltonian/circuit construction, training
    includes expectation evaluation, and sampling includes the shot job.

    An optional fixed-weight variable register selects the XY mixer. Its
    Hamming weight is enforced by state preparation and gates; the caller
    supplies the appropriate objective and is responsible for any remaining
    penalties. This does not silently change the BQM or its energy units.
    """
    validate_options(reps=reps, shots=shots, maxiter=maxiter, seed=seed, max_qubits=max_qubits)
    validate_bqm(bqm, max_qubits)
    start = time.perf_counter()
    import qaoa_common as qc
    from qaoa_qiskit import QiskitBackend

    variables = list(bqm.variables)
    h, J, offset = qc.ising_from_bqm(bqm, variables)
    scale = float(max(np.max(np.abs(h)), max(map(abs, J.values()), default=0)))
    if scale == 0 or not all(math.isfinite(v) for v in [*h, *J.values(), offset]):
        raise ValueError('QAOA needs a nonconstant Hamiltonian with finite coefficients')
    # QiskitBackend's cost protocol only needs these fields, not NIP references.
    problem = SimpleNamespace(h=h, J=J, offset=offset, scale=scale, num_qubits=len(variables))
    mixer = None
    if fixed_weight_variables is not None:
        from qaoa_xy import FixedWeightMixer
        labels = tuple(fixed_weight_variables)
        if any(v not in variables for v in labels):
            raise ValueError('fixed-weight variables must belong to the BQM')
        mixer = FixedWeightMixer(tuple(variables.index(v) for v in labels),
                                 hamming_weight, initial_state)
        mixer.validate(len(variables))
    elif hamming_weight is not None or initial_state != 'basis':
        raise ValueError('fixed-weight settings require a variable register')
    backend = QiskitBackend(problem, int(reps), fixed_weight=mixer)
    build_time = time.perf_counter() - start
    start = time.perf_counter()
    theta, value, evaluations = qc.optimize_angles(
        backend.normalized_expectation, int(reps), seed=int(seed), maxiter=int(maxiter))
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    counts = backend.sample(theta, int(shots), int(seed))
    sample_time = time.perf_counter() - start
    metadata = {
        'backend': 'qiskit-aer', 'method': 'statevector', 'reps': int(reps),
        'shots': int(shots), 'seed': int(seed), 'maxiter': int(maxiter),
        'max_qubits': int(max_qubits), 'num_qubits': len(variables),
        'variables': variables, 'theta': [float(t) for t in theta],
        'angle_order': 'beta layers, then gamma layers', 'hamiltonian_scale': scale,
        'evaluations': int(evaluations), 'expected_energy': float(value*scale + offset),
        'build_time_sec': build_time, 'train_time_sec': train_time,
        'sample_time_sec': sample_time, 'circuit_depth': backend.circuit.depth(),
        'two_qubit_gates': sum(len(item.qubits) == 2 for item in backend.circuit.data),
        'circuit_basis': 'h,rz,rzz,rx,ry,cx; unconstrained connectivity; no measurement',
        'qiskit': importlib.metadata.version('qiskit'),
        'qiskit_aer': importlib.metadata.version('qiskit-aer'),
    }
    if mixer is not None:
        metadata['mixer'] = 'parity-partitioned XY ring on register; X elsewhere'
        metadata['fixed_weight_variables'] = list(labels)
        metadata['fixed_weight_qubits'] = list(mixer.qubits)
        metadata['hamming_weight'] = int(mixer.weight)
        metadata['initial_state'] = mixer.initial
    return dimod.SampleSet.from_samples_bqm(
        (list(counts), variables), bqm, num_occurrences=list(counts.values()),
        info={'qaoa_metadata': metadata})

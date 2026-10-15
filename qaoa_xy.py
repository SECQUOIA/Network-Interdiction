"""Fixed-weight, parity-partitioned XY mixers for a selected qubit register.

Each pair uses exp(-i beta (XX+YY)); the ordered pair product preserves
weight exactly but is not generally the exponential of the summed ring
Hamiltonian. Other qubits retain the ordinary X mixer.
"""
from dataclasses import dataclass
import itertools
import math
from numbers import Integral

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation

MAX_DICKE_QUBITS = 12


@dataclass(frozen=True)
class FixedWeightMixer:
    qubits: tuple
    weight: int
    initial: str = 'basis'

    def validate(self, num_qubits):
        qubits = tuple(self.qubits)
        if not qubits or any(isinstance(q, bool) or not isinstance(q, Integral)
                             or not 0 <= q < num_qubits for q in qubits):
            raise ValueError('fixed-weight register needs valid qubit indices')
        if len(set(qubits)) != len(qubits):
            raise ValueError('fixed-weight qubits must be distinct')
        if isinstance(self.weight, bool) or not isinstance(self.weight, Integral) \
                or not 0 <= self.weight <= len(qubits):
            raise ValueError('fixed weight must be an integer from zero to register size')
        if self.initial not in ('basis', 'dicke'):
            raise ValueError("fixed-weight initial state must be 'basis' or 'dicke'")
        if self.initial == 'dicke' and len(qubits) > MAX_DICKE_QUBITS:
            raise ValueError(f'dense Dicke preparation is limited to {MAX_DICKE_QUBITS} qubits')
        if num_qubits == len(qubits) == 1:
            raise ValueError('a one-qubit fixed register has no mixing gate')
        return tuple(map(int, qubits))


def ring_pairs(qubits):
    """Even adjacent pairs, odd adjacent pairs, then the closing edge."""
    q = tuple(qubits)
    if len(q) < 2:
        return []
    if len(q) == 2:
        return [(q[0], q[1])]
    return (list(zip(q[0:-1:2], q[1::2]))
            + list(zip(q[1:-1:2], q[2::2])) + [(q[-1], q[0])])


def fixed_weight_ansatz(problem, reps, mixer):
    """Build the cost/XY circuit without any exact-reference or service call.

    `basis` sets the first B register bits. `dicke` prepares the uniform
    weight-B state using Qiskit's generic state synthesis, bounded to 12
    register qubits before allocating 2**m amplitudes. This is not an
    implementation of a polynomial-size Dicke preparation algorithm.
    """
    if isinstance(reps, bool) or not isinstance(reps, Integral) or reps < 1:
        raise ValueError('reps must be a positive integer')
    qubits = mixer.validate(problem.num_qubits)
    register = set(qubits)
    circuit = QuantumCircuit(problem.num_qubits)
    if mixer.initial == 'basis' or mixer.weight in (0, len(qubits)):
        for q in qubits[:int(mixer.weight)]:
            circuit.x(q)
    else:
        amplitudes = np.zeros(1 << len(qubits), dtype=complex)
        amplitude = 1 / math.sqrt(math.comb(len(qubits), int(mixer.weight)))
        for selected in itertools.combinations(range(len(qubits)), int(mixer.weight)):
            amplitudes[sum(1 << i for i in selected)] = amplitude
        circuit.append(StatePreparation(amplitudes), qubits)
    for q in range(problem.num_qubits):
        if q not in register:
            circuit.h(q)
    betas, gammas = ParameterVector('β', int(reps)), ParameterVector('γ', int(reps))
    for beta, gamma in zip(betas, gammas):
        for i, coeff in enumerate(problem.h):
            if coeff != 0:
                circuit.rz(2*gamma*coeff/problem.scale, i)
        for (i, j), coeff in problem.J.items():
            circuit.rzz(2*gamma*coeff/problem.scale, i, j)
        for i, j in ring_pairs(qubits):
            circuit.rxx(2*beta, i, j)
            circuit.ryy(2*beta, i, j)
        for q in range(problem.num_qubits):
            if q not in register:
                circuit.rx(2*beta, q)
    return circuit

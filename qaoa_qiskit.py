"""
Penalty-QAOA for the NIP QUBO in Qiskit (issues #3, #16; the Aer step of #17).

The cost Hamiltonian is the Ising form of nip_to_qubo()'s QUBO, built by
qaoa_common.build_problem(), as a SparsePauliOp; the circuit is Qiskit's
qaoa_ansatz() of it with the standard X mixer. Training uses the exact
expectation values of Aer's EstimatorV2, and the trained circuit is scored
from its Aer statevector and from SamplerV2 shots. Conventions (qubit order,
angle order [beta..., gamma...], Hamiltonian scaling) are in qaoa_common.

cost_operator() deliberately has no identity term: the constant offset is
added back in QUBO units by qaoa_common, and qaoa_training_pipeline's
trainers (TransferTrainer, ScipyTrainer) reject operators that carry one. The
operator they need for parameter transfer (#16) is cost_operator(problem)
as is.

Command line (needs requirements-qaoa.txt):

    python qaoa_qiskit.py testcases/tiny_a.flow --budget 1 --reps 2
"""
import sys

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2

import qaoa_common

# The gates qaoa_ansatz() of a Z/ZZ operator reduces to. Transpiling once to
# these keeps the cost layer as RZ/RZZ, which Aer simulates directly.
BASIS_GATES = ['h', 'rz', 'rzz', 'rx', 'cx']


def cost_operator(problem):
    """(H_C - offset) / scale as a SparsePauliOp on problem.num_qubits."""
    scale = problem.scale
    terms = [('Z', [i], coeff / scale)
             for i, coeff in enumerate(problem.h) if coeff != 0]
    terms += [('ZZ', [i, j], coeff / scale)
              for (i, j), coeff in problem.J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=problem.num_qubits)


class QiskitBackend:
    """
    The QAOA circuit for one problem and depth, simulated with Aer.

    `estimator` defaults to Aer's EstimatorV2; any V2 estimator that accepts
    this circuit can be passed instead (an IBM device would also need the
    circuit transpiled to its ISA, #17). Sampling uses Aer's SamplerV2,
    seeded per call.
    """
    name = 'qiskit'

    def __init__(self, problem, reps, estimator=None):
        self.problem = problem
        self.reps = reps
        self.operator = cost_operator(problem)
        ansatz = qaoa_ansatz(self.operator, reps=reps)
        # qaoa_ansatz() names its parameters β[0..p-1] and γ[0..p-1], and
        # circuit.parameters sorts them so: a theta vector binds in the
        # [beta..., gamma...] order qaoa_common uses.
        names = [parameter.name for parameter in ansatz.parameters]
        expected = [f'β[{l}]' for l in range(reps)] + \
                   [f'γ[{l}]' for l in range(reps)]
        if names != expected:
            raise RuntimeError(f'unexpected qaoa_ansatz parameters {names}')
        self.circuit = transpile(ansatz, basis_gates=BASIS_GATES,
                                 optimization_level=0)
        self.estimator = estimator if estimator is not None else EstimatorV2()
        self.simulator = AerSimulator(method='statevector')

    def normalized_expectation(self, theta):
        """Exact <(H_C - offset) / scale> at theta = [beta..., gamma...]."""
        job = self.estimator.run([(self.circuit, self.operator, list(theta))])
        return float(job.result()[0].data.evs)

    def probabilities(self, theta):
        """The 2**n outcome probabilities at theta, little-endian."""
        circuit = self.circuit.assign_parameters(list(theta))
        circuit.save_statevector()
        state = self.simulator.run(circuit).result().get_statevector()
        return np.abs(np.asarray(state)) ** 2

    def sample(self, theta, shots, seed=None):
        """
        {per-qubit bit tuple: count} from `shots` measurements. Qiskit's
        bitstrings put qubit 0 last, so each is reversed.
        """
        measured = self.circuit.measure_all(inplace=False)
        sampler = SamplerV2(seed=seed)
        result = sampler.run([(measured, list(theta))], shots=shots).result()
        counts = result[0].data.meas.get_counts()
        return {tuple(int(bit) for bit in reversed(key)): count
                for key, count in counts.items()}


def main(argv=None):
    return qaoa_common.cli(
        lambda problem, reps, args: QiskitBackend(problem, reps),
        'Penalty-QAOA for the NIP QUBO, simulated with Qiskit Aer.', argv)


if __name__ == '__main__':
    sys.exit(main())

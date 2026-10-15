"""
QAOA for the NIP QUBO in Qiskit (issues #3, #16; the Aer step of #17).

The cost Hamiltonian is the Ising form of nip_to_qubo()'s QUBO, built by
qaoa_common.build_problem(), as a SparsePauliOp; the circuit is Qiskit's
qaoa_ansatz() of it with the standard X mixer by default. An opt-in
FixedWeightMixer preserves a selected register's budget. Training uses the exact
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
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2

import qaoa_common

# The gates qaoa_ansatz() of a Z/ZZ operator reduces to. Transpiling once to
# these keeps the cost layer as RZ/RZZ, which Aer simulates directly.
BASIS_GATES = ['h', 'rz', 'rzz', 'rx', 'ry', 'cx']

# Aer simulation methods: the exact statevector (up to qaoa_common's
# EXACT_QUBITS for the exact scores, about 30 qubits in memory) and the
# matrix product state, whose cost grows with entanglement rather than
# qubit count. Capping its bond dimension makes it approximate.
METHODS = {'statevector': 'statevector', 'mps': 'matrix_product_state'}


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
    The QAOA circuit for one problem and depth, simulated with Aer's V2
    primitives. qaoa_ibm.py supplies fixed-angle Runtime sampling after routing
    the same cost circuit to a device ISA (#17).

    `method` is 'statevector' (exact) or 'mps' (matrix product state).
    `max_bond_dimension` caps the MPS bond dimension: None keeps it exact,
    and so exponential in the entanglement; a cap such as 64 keeps it
    polynomial at a truncation error. On a 40-qubit k-NN instance at p = 2
    a cap of 64 changed <H_C>/scale by 6e-5 (relative 1e-5) and cut the
    time from 71 s to 5 s. MPS wants a qubit order that keeps couplings
    short: build the problem with order='rcm'.

    `warm_start_angles`, when supplied, gives one Ry angle per qubit in
    [0, pi] and replaces the X mixer by sin(angle) X + cos(angle) Z on
    that qubit. regularized_warm_start() supplies these angles. At pi/2
    this reduces to the standard initial state and mixer.

    `fixed_weight` accepts qaoa_xy.FixedWeightMixer: its register uses a
    weight-preserving XY ring and a feasible initial state; all other qubits
    use X mixing. It cannot be combined with the product warm start.
    """
    name = 'qiskit'

    def __init__(self, problem, reps, method='statevector',
                 max_bond_dimension=None, warm_start_angles=None, fixed_weight=None):
        if method not in METHODS:
            raise ValueError(f'unknown method {method!r}, expected one of '
                             f'{tuple(METHODS)}')
        self.problem = problem
        self.reps = reps
        self.method = method
        self.backend_options = {'method': METHODS[method]}
        if max_bond_dimension is not None:
            if method != 'mps':
                raise ValueError('max_bond_dimension applies to method mps')
            self.backend_options[
                'matrix_product_state_max_bond_dimension'] = max_bond_dimension
        # Exact probabilities need the statevector.
        self.exact_probabilities = method == 'statevector'
        self.operator = cost_operator(problem)
        if fixed_weight is not None:
            if warm_start_angles is not None:
                raise ValueError('fixed-weight mixing cannot use product warm-start angles')
            from qaoa_xy import fixed_weight_ansatz
            ansatz = fixed_weight_ansatz(problem, reps, fixed_weight)
        elif warm_start_angles is None:
            ansatz = qaoa_ansatz(self.operator, reps=reps)
        else:
            angles = np.asarray(warm_start_angles, dtype=float)
            if angles.shape != (problem.num_qubits,) or not np.all(
                    np.isfinite(angles) & (angles >= 0) & (angles <= np.pi)):
                raise ValueError('warm-start angles must be in [0, pi], one per qubit')
            ansatz = QuantumCircuit(problem.num_qubits)
            betas, gammas = ParameterVector('β', reps), ParameterVector('γ', reps)
            for i, angle in enumerate(angles):
                ansatz.ry(float(angle), i)
            for beta, gamma in zip(betas, gammas):
                for i, coeff in enumerate(problem.h):
                    if coeff != 0:
                        ansatz.rz(2 * gamma * coeff / problem.scale, i)
                for (i, j), coeff in problem.J.items():
                    ansatz.rzz(2 * gamma * coeff / problem.scale, i, j)
                for i, angle in enumerate(angles):
                    # exp(-i beta (sin(angle) X + cos(angle) Z)).
                    ansatz.ry(-float(angle), i)
                    ansatz.rz(2 * beta, i)
                    ansatz.ry(float(angle), i)
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
        self.estimator = EstimatorV2(
            options={'backend_options': dict(self.backend_options)})
        self.simulator = AerSimulator(method='statevector')

    def normalized_expectation(self, theta):
        """
        <(H_C - offset) / scale> at theta = [beta..., gamma...]: exact for
        the statevector and an uncapped MPS, approximate for a capped one.
        """
        job = self.estimator.run([(self.circuit, self.operator, list(theta))])
        return float(job.result()[0].data.evs)

    def probabilities(self, theta):
        """The 2**n outcome probabilities at theta, little-endian."""
        if not self.exact_probabilities:
            raise ValueError('exact probabilities need method statevector')
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
        sampler = SamplerV2(seed=seed, options={
            'backend_options': dict(self.backend_options)})
        result = sampler.run([(measured, list(theta))], shots=shots).result()
        counts = result[0].data.meas.get_counts()
        return {tuple(int(bit) for bit in reversed(key)): count
                for key, count in counts.items()}


def main(argv=None):
    def simulator_options(parser):
        parser.add_argument('--method', choices=tuple(METHODS),
                            default='statevector',
                            help='Aer method: exact statevector, or matrix '
                                 'product state for more qubits')
        parser.add_argument('--bond-dimension', type=int,
                            help='cap on the MPS bond dimension (default: '
                                 'none, exact)')

    return qaoa_common.cli(
        lambda problem, reps, args: QiskitBackend(
            problem, reps, args.method, args.bond_dimension),
        'Penalty-QAOA for the NIP QUBO, simulated with Qiskit Aer.', argv,
        extra_args=simulator_options)


if __name__ == '__main__':
    sys.exit(main())

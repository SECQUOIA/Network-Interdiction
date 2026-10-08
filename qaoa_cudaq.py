"""
Penalty-QAOA for the NIP QUBO in CUDA-Q (issues #3, #16).

The same circuit as qaoa_qiskit.py, written as a CUDA-Q kernel: Hadamards,
then p layers of RZ (local fields), CNOT-RZ-CNOT (couplings) and RX (mixer).
Training uses cudaq.observe(), which is exact on the statevector simulators;
the trained kernel is scored from cudaq.get_state() and from cudaq.sample()
shots. Conventions (qubit order, angle order [beta..., gamma...],
Hamiltonian scaling) are in qaoa_common, so for the same theta this kernel
and the Qiskit circuit prepare the same state.

CUDA-Q picks its simulator target at import: `nvidia` (cuStateVec) when a GPU
is visible, else `qpp-cpu`. --target overrides it.

Command line (needs requirements-qaoa.txt):

    python qaoa_cudaq.py testcases/tiny_a.flow --budget 1 --reps 2
"""
import sys

import cudaq
import numpy as np

import qaoa_common

# CUDA-Q targets that hold the full statevector, so get_state() can give
# exact probabilities.
STATEVECTOR_TARGETS = {'qpp-cpu', 'nvidia', 'nvidia-fp64', 'nvidia-mgpu'}


@cudaq.kernel
def nip_qaoa_kernel(num_qubits: int, reps: int, z_qubits: list[int],
                    z_coeffs: list[float], zz_first: list[int],
                    zz_second: list[int], zz_coeffs: list[float],
                    thetas: list[float]):
    """
    QAOA state for (H_C - offset) / scale given as its Z terms
    (z_coeffs[k] on qubit z_qubits[k]) and ZZ terms (zz_coeffs[k] on
    zz_first[k], zz_second[k]); thetas = [beta..., gamma...].
    """
    qubits = cudaq.qvector(num_qubits)
    h(qubits)
    for layer in range(reps):
        beta = thetas[layer]
        gamma = thetas[reps + layer]
        # exp(-i gamma c Z) = RZ(2 gamma c)
        for k in range(len(z_qubits)):
            rz(2.0 * gamma * z_coeffs[k], qubits[z_qubits[k]])
        # exp(-i gamma c Z Z) = CNOT, RZ(2 gamma c) on the target, CNOT
        for k in range(len(zz_first)):
            x.ctrl(qubits[zz_first[k]], qubits[zz_second[k]])
            rz(2.0 * gamma * zz_coeffs[k], qubits[zz_second[k]])
            x.ctrl(qubits[zz_first[k]], qubits[zz_second[k]])
        # exp(-i beta X) = RX(2 beta)
        for i in range(num_qubits):
            rx(2.0 * beta, qubits[i])


def spin_operator(problem):
    """(H_C - offset) / scale as a cudaq.SpinOperator."""
    scale = problem.scale
    terms = [coeff / scale * cudaq.spin.z(i)
             for i, coeff in enumerate(problem.h) if coeff != 0]
    terms += [coeff / scale * cudaq.spin.z(i) * cudaq.spin.z(j)
              for (i, j), coeff in problem.J.items()]
    operator = terms[0]
    for term in terms[1:]:
        operator += term
    return operator


class CudaqBackend:
    """
    The QAOA kernel for one problem and depth, on the active CUDA-Q target.

    Exact probabilities (cudaq.get_state) come only from a statevector
    target; on a tensor-network target such as 'tensornet-mps' (GPU only)
    run_qaoa() scores from observe() and shots instead.
    """
    name = 'cudaq'

    def __init__(self, problem, reps):
        self.problem = problem
        self.reps = reps
        self.exact_probabilities = (
            cudaq.get_target().name in STATEVECTOR_TARGETS)
        self.operator = spin_operator(problem)
        scale = problem.scale
        z_terms = [(i, coeff / scale) for i, coeff in enumerate(problem.h)
                   if coeff != 0]
        zz_terms = [(i, j, coeff / scale)
                    for (i, j), coeff in problem.J.items()]
        self._fixed_args = (
            problem.num_qubits, reps,
            [i for i, _ in z_terms], [c for _, c in z_terms],
            [i for i, _, _ in zz_terms], [j for _, j, _ in zz_terms],
            [c for _, _, c in zz_terms])

    def _args(self, theta):
        return self._fixed_args + ([float(t) for t in theta],)

    def normalized_expectation(self, theta):
        """Exact <(H_C - offset) / scale> at theta = [beta..., gamma...]."""
        return float(cudaq.observe(nip_qaoa_kernel, self.operator,
                                   *self._args(theta)).expectation())

    def probabilities(self, theta):
        """
        The 2**n outcome probabilities at theta. CUDA-Q indexes its state
        vector little-endian (qubit i is bit i of the index), as qaoa_common
        does.
        """
        state = cudaq.get_state(nip_qaoa_kernel, *self._args(theta))
        return np.abs(np.asarray(state)) ** 2

    def sample(self, theta, shots, seed=None):
        """
        {per-qubit bit tuple: count} from `shots` measurements. CUDA-Q's
        bitstrings put qubit 0 first, so they are read in order.
        """
        if seed is not None:
            cudaq.set_random_seed(seed)
        counts = cudaq.sample(nip_qaoa_kernel, *self._args(theta),
                              shots_count=shots)
        return {tuple(int(bit) for bit in key): count
                for key, count in counts.items()}


def main(argv=None):
    def target_option(parser):
        parser.add_argument('--target',
                            help="CUDA-Q target, e.g. 'qpp-cpu' or 'nvidia' "
                                 "(default: CUDA-Q's choice for this machine)")

    def make_backend(problem, reps, args):
        if args.target:
            cudaq.set_target(args.target)
        return CudaqBackend(problem, reps)

    return qaoa_common.cli(
        make_backend, 'Penalty-QAOA for the NIP QUBO, simulated with CUDA-Q.',
        argv, extra_args=target_option)


if __name__ == '__main__':
    sys.exit(main())

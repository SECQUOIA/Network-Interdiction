"""
Regularized warm starts for the NIP Hamiltonian (issue #16).

Adapts equations (9), (18) of https://arxiv.org/abs/2603.10191 to the
repository's *minimization* and normalized Ising convention. Writing
z_i = cos(angle_i), minimize

    sum h_i z_i / scale + sum J_ij z_i z_j / scale + strength sum z_i**2

over [-1, 1]^n. The last term differs from -strength sum sin(angle_i)**2
only by a constant. Strength is dimensionless and is separate from NIP's
budget penalty lambda. This is a local classical optimization, not a bound.
The Qiskit backend prepares Ry(angle) and uses the matching tilted mixer.
"""
import numpy as np
from scipy.optimize import minimize


def warm_start_objective(problem, z, strength):
    """Regularized product-state energy and analytic gradient, without offset."""
    z = np.asarray(z, dtype=float)
    h = problem.h / problem.scale
    value = h @ z + strength * (z @ z)
    gradient = h + 2 * strength * z
    for (i, j), coeff in problem.J.items():
        coeff /= problem.scale
        value += coeff * z[i] * z[j]
        gradient[i] += coeff * z[j]
        gradient[j] += coeff * z[i]
    return float(value), gradient


def regularized_warm_start(problem, strength=0.6, seed=0, restarts=3):
    """Per-qubit Ry angles from the best converged L-BFGS-B start."""
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('warm-start strength must be positive and finite')
    if restarts < 0:
        raise ValueError('restarts must be nonnegative')
    rng = np.random.default_rng(seed)
    starts = [np.zeros(problem.num_qubits)]
    starts += [rng.uniform(-1, 1, problem.num_qubits)
               for _ in range(restarts)]
    results = [minimize(
        lambda z: warm_start_objective(problem, z, strength), start,
        jac=True, method='L-BFGS-B', bounds=[(-1, 1)] * problem.num_qubits,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        for start in starts]
    converged = [result for result in results if result.success]
    if not converged:
        raise RuntimeError('regularized warm-start optimization did not converge')
    best = min(converged, key=lambda result: result.fun)
    return np.arccos(np.clip(best.x, -1, 1))

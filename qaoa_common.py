"""
Penalty-QAOA for the NIP QUBO: the parts shared by the Qiskit and CUDA-Q
backends (issues #3 and #16, first task of each).

qaoa_qiskit.py and qaoa_cudaq.py each build and simulate the QAOA circuit in
their own framework. Everything else lives here, so the two backends optimize
exactly the same Hamiltonian, use the same angle convention, and are scored by
the same code:

- build_problem() builds the QUBO nip_to_qubo() returns for an instance and
  budget, with lambda chosen from its exact window (`auto`, as in
  run_benchmark.py; issue #2), and converts it to an Ising Hamiltonian.
- run_qaoa() trains the angles against a backend's exact expectation value,
  then scores the trained state twice: exactly, from its probability vector,
  and from shots, decoded and repaired the way the annealer passes are
  (qubo_verify_parameters()).

Qubits and bits
---------------
Qubit i carries QUBO variable problem.variables[i]: x_0..x_{E-1}, then gamma,
w, and pi (the families nip_to_qubo() creates, each in index order). A
measured bit b_i = 1 means that variable is 1, so Z_i = 1 - 2 b_i and
x_i = (1 - Z_i) / 2. With that substitution the QUBO energy is the diagonal
Hamiltonian

    H_C = offset + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

and H_C equals the QUBO energy on every basis state. A probability vector is
indexed little-endian, as both Qiskit and CUDA-Q index state vectors: basis
state k has b_i = (k >> i) & 1. Samples are exchanged as tuples of per-qubit
bits, (b_0, b_1, ...); each backend converts its own bitstring order.

Angle convention
----------------
Both backends prepare

    |psi(beta, gamma)> = prod_{l=1..p} exp(-i beta_l sum_i X_i)
                                       exp(-i gamma_l (H_C - offset) / scale)
                         |+>^n

that is, RZ(2 gamma h_i / scale), RZZ(2 gamma J_ij / scale) and RX(2 beta)
gates, the convention of Qiskit's qaoa_ansatz(). The parameter vector is
theta = [beta_1..beta_p, gamma_1..gamma_p], the order qaoa_ansatz() sorts its
parameters in and qaoa_training_pipeline stores angles in, so trained angles
can be handed to either without reordering. `scale` is the largest |h_i| or
|J_ij|. Dividing by it puts every instance's gamma on the same O(1) scale,
whatever its capacities and lambda, which is what makes angles comparable
across instances (#16). Energies reported here are in QUBO units, not scaled.

What the scores mean
--------------------
p_ground    probability of measuring a QUBO ground state
p_optimal   probability of measuring a state whose x bits are an optimal
            interdiction: at most B arcs, leaving the optimal surviving flow.
            Every ground state is one, and other states with the same x but
            unsatisfied gamma/w/pi bits are too, so p_optimal >= p_ground.
approx_ratio  (E_max - <H_C>) / (E_max - E_min): 1 at the ground state, 0 at
            the highest-energy state
uniform_*   the same scores for the uniform superposition (p = 0): <H_C> is
            then `offset`, since every <Z> vanishes, and p_optimal is the
            fraction of x patterns that are optimal interdictions

Statevector simulation needs 2**n amplitudes, so build_problem() refuses
QUBOs over max_qubits (24 by default, run_benchmark.py's exact-solver cap):
the tiny_* instances (14 to 22 qubits) and nothing larger.
"""
import argparse
import contextlib
import io
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
from scipy.optimize import minimize

import network_interdiction as ni
import run_benchmark as rb

DEFAULT_MAX_QUBITS = 24

# Largest problem whose 2**n basis states are enumerated for exact scores
# (energies(), optimal_mask()). Larger problems are scored from shots.
EXACT_QUBITS = 24

# Variable families in the order qubits are assigned to them.
VARIABLE_FAMILIES = ('x', 'gamma', 'w', 'pi')
QUBIT_ORDERS = ('family', 'rcm')

# Samples decoded and repaired per run, as qubo_verify_parameters() decodes
# the annealer's 20 lowest-energy samples.
REPAIRED_SAMPLES = 20

# Energies within this of the minimum count as ground states. QUBO
# coefficients are sums of capacities and lambda, so energies differ by far
# more than this when they differ at all.
ENERGY_TOL = 1e-6


def variable_order(bqm):
    """
    The QUBO variables in qubit order: x, gamma, w, then pi, each by index.

    nip_to_qubo() names variables `<family>_<index>`; ordering them by family
    and index (rather than by however the BQM stores them) keeps qubit i on
    the same variable from run to run, which matters once angles or samples
    are stored.
    """
    return sorted(bqm.variables, key=_family_key)


def _family_key(label):
    family, index = label.rsplit('_', 1)
    return VARIABLE_FAMILIES.index(family), int(index)


def rcm_order(bqm):
    """
    The QUBO variables in reverse Cuthill-McKee order of their interaction
    graph, which keeps coupled variables close together.

    Matrix-product-state simulation pays for every coupling by how many
    qubits it spans, so this order is what makes it practical. In
    variable_order(), one p = 1 evaluation of a 78-qubit instance had not
    finished after ten minutes; in this order a 40-qubit k-NN instance takes
    0.1 s, and a 53-qubit planar one 4.5 s with the bond dimension capped at
    64. The order is deterministic across Python hash seeds: the traversal
    uses stable integer node IDs instead of randomized string hashes.
    """
    variables = variable_order(bqm)
    position = {label: i for i, label in enumerate(variables)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(variables)))
    graph.add_edges_from(sorted(
        tuple(sorted((position[u], position[v]))) for u, v in bqm.quadratic))
    return [variables[i] for i in nx.utils.reverse_cuthill_mckee_ordering(graph)]


def ising_from_bqm(bqm, variables):
    """
    (h, J, offset) of the diagonal Hamiltonian equal to `bqm`'s energy.

    Substitutes x_i = (1 - Z_i) / 2 into the binary model
    E(x) = c + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j:

        a_i x_i      = a_i / 2 - (a_i / 2) Z_i
        b_ij x_i x_j = (b_ij / 4) (1 - Z_i - Z_j + Z_i Z_j)

    h is an array indexed by qubit, J a dict {(i, j): J_ij} with i < j.
    This is the opposite sign convention to dimod's to_ising() (which uses
    spin s = 2x - 1 = -Z), because a measured 1 must mean x = 1.
    """
    bqm = bqm.change_vartype('BINARY', inplace=False)
    position = {label: i for i, label in enumerate(variables)}
    h = np.zeros(len(variables))
    J = {}
    offset = bqm.offset
    for label, a in bqm.linear.items():
        i = position[label]
        h[i] -= a / 2
        offset += a / 2
    for (u, v), b in bqm.quadratic.items():
        i, j = sorted((position[u], position[v]))
        h[i] -= b / 4
        h[j] -= b / 4
        J[(i, j)] = J.get((i, j), 0.0) + b / 4
        offset += b / 4
    return h, {pair: c for pair, c in J.items() if c != 0}, float(offset)


@dataclass
class NIPQaoaProblem:
    """
    One NIP instance and budget, as the Ising Hamiltonian a QAOA backend
    simulates. Built by build_problem(); see the module docstring for the
    qubit, bit and angle conventions.
    """
    instance: str
    di_g: nx.DiGraph
    src_node: int
    tgt_node: int
    B: int
    lam: float
    lambda_window: tuple
    bqm: object
    variables: list
    h: np.ndarray
    J: dict
    offset: float
    opt_nip: float
    ground_energy: float = None
    interdictable: object = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def num_qubits(self):
        return len(self.variables)

    @property
    def exact(self):
        """Small enough to enumerate every basis state (EXACT_QUBITS)."""
        return self.num_qubits <= EXACT_QUBITS

    @property
    def maxflow(self):
        """Max flow with nothing interdicted."""
        if 'maxflow' not in self._cache:
            self._cache['maxflow'] = rb.max_flow(self.di_g, self.src_node,
                                                 self.tgt_node)
        return self._cache['maxflow']

    def _require_exact(self, what):
        if not self.exact:
            raise ValueError(
                f'{what} enumerates all 2**{self.num_qubits} basis states; '
                f'only problems of up to {EXACT_QUBITS} qubits are scored '
                f'exactly, larger ones from shots')

    @property
    def edge_list(self):
        """Arcs in the order x_0, x_1, ... index them."""
        return rb.sorted_edges(self.di_g)

    @property
    def scale(self):
        """Largest |h_i| or |J_ij|; the circuit simulates (H_C - offset) / scale."""
        return float(max(np.max(np.abs(self.h)),
                         max((abs(c) for c in self.J.values()), default=0.0)))

    @property
    def x_qubits(self):
        """Qubits of the interdictable arcs, in edge-list order."""
        return [self.variables.index(f'x_{i}')
                for i, edge in enumerate(self.edge_list)
                if edge in self.interdictable_edges]

    @property
    def interdictable_edges(self):
        """Original arcs that have x variables; fixed contraction arcs do not."""
        if self.interdictable is None:
            return self.edge_list
        return [edge for edge in self.edge_list if edge in self.interdictable]

    def energies(self):
        """
        H_C on every basis state, little-endian (length 2**n), in QUBO units.
        Cached: it is 2**n floats and every exact score reads it.
        """
        self._require_exact('energies()')
        if 'energies' not in self._cache:
            index = np.arange(2 ** self.num_qubits, dtype=np.int64)

            def z(i):
                return 1 - 2 * ((index >> i) & 1)

            energy = np.full(index.shape, self.offset)
            for i, coeff in enumerate(self.h):
                if coeff != 0:
                    energy += coeff * z(i)
            for (i, j), coeff in self.J.items():
                energy += coeff * (z(i) * z(j))
            self._cache['energies'] = energy
        return self._cache['energies']

    def optimal_mask(self):
        """
        Boolean over basis states: is its x pattern an optimal interdiction?
        Only the E x bits decide it, so it costs 2**E max-flow solves.
        """
        self._require_exact('optimal_mask()')
        if 'optimal_mask' not in self._cache:
            edges = self.interdictable_edges
            num_edges = len(edges)
            optimal_pattern = np.zeros(2 ** num_edges, dtype=bool)
            for pattern in range(2 ** num_edges):
                cut = [edges[e] for e in range(num_edges)
                       if (pattern >> e) & 1]
                optimal_pattern[pattern] = (
                    len(cut) <= self.B and
                    rb.max_flow(self.di_g, self.src_node, self.tgt_node, cut)
                    == self.opt_nip)
            index = np.arange(2 ** self.num_qubits, dtype=np.int64)
            pattern = np.zeros(index.shape, dtype=np.int64)
            for e, qubit in enumerate(self.x_qubits):
                pattern |= ((index >> qubit) & 1) << e
            self._cache['optimal_mask'] = optimal_pattern[pattern]
        return self._cache['optimal_mask']

    def bits_energy(self, bits):
        """QUBO energy of one assignment given as per-qubit bits."""
        return float(self.bqm.energy(dict(zip(self.variables, bits))))

    def interdicted_edges(self, bits):
        """Arcs whose x bit is 1 in `bits` (per-qubit bits)."""
        return [edge for edge, qubit in zip(self.interdictable_edges, self.x_qubits)
                if bits[qubit] == 1]

    def x_vector(self, bits):
        """The x array improve_dwave_solution() takes, indexed by edge_list."""
        cut = set(self.interdicted_edges(bits))
        return np.array([int(edge in cut) for edge in self.edge_list], dtype=int)


def restricted_optimum_curve(di_g, src, tgt, interdictable, backend):
    """Exact f(k) for a contracted piece, including its fixed boundary arcs."""
    edges = sorted(interdictable)
    curve = []
    for k in range(len(edges) + 1):
        if backend == 'brute':
            value = min(rb.max_flow(di_g, src, tgt, cut)
                        for cut in itertools.combinations(edges, k))
        elif backend == 'gurobi':
            model, x = ni._build_nip_milp(di_g, src, tgt, k, edges)
            model.optimize()
            if model.Status != ni.GRB.OPTIMAL:
                raise RuntimeError('piece reference did not reach optimality')
            # Recompute the objective of the optimal discrete cut using
            # the same flow calculation as decoding/scoring. Solver
            # tolerance noise must not change equality with a fractional
            # optimum, and rounding would erase real capacity differences.
            cut = [edge for edge in edges if x[edge].X > 0.5]
            value = rb.max_flow(di_g, src, tgt, cut)
        else:
            raise ValueError(f'unknown exact backend: {backend}')
        curve.append(value)
    return curve


def problem_from_graph(di_g, src_node, tgt_node, B, lam='auto', instance='',
                       max_qubits=DEFAULT_MAX_QUBITS, exact_backend='brute',
                       order='family', interdictable=None):
    """
    The QAOA problem for one graph and budget. `lam` is a number, or 'auto'
    for the midpoint of the exact window (run_benchmark.choose_lambda()).

    `exact_backend` computes the optimal flows the window needs: 'brute'
    (enumeration, the tiny instances) or 'gurobi' (license required; the
    only practical one past a dozen arcs). A problem of up to EXACT_QUBITS
    qubits gets its ground energy by enumerating the Hamiltonian; a larger
    one from Gurobi's exact solve of the penalised model
    (run_benchmark.exact_reference()), which needs a license too. `order`
    is the qubit order: 'family' (variable_order()) or 'rcm' (rcm_order(),
    for matrix-product-state simulation).

    `interdictable` restricts x variables to a piece's owned arcs; other
    arcs retain their capacities. Its exact window uses the restricted
    NIP, and decoding/repair keep those fixed arcs intact.

    Raises ValueError when the window is empty (no lambda makes the QUBO's
    ground state the optimal interdiction, so there is nothing for QAOA to
    find), when a given `lam` does not encode the optimum, or when the QUBO
    is over `max_qubits`.
    """
    if order not in QUBIT_ORDERS:
        raise ValueError(f'unknown qubit order {order!r}, expected one of '
                         f'{QUBIT_ORDERS}')
    # Enforce the simulation limit before the combinatorial reference scan.
    # Compile with positive placeholder penalties to count the actual QUBO
    # variables, including auxiliaries, without first solving for lambda.
    if interdictable is not None:
        interdictable = frozenset(interdictable)
        if not interdictable.issubset(di_g.edges):
            raise ValueError('interdictable contains an arc outside the graph')
        if not 0 <= B <= len(interdictable):
            raise ValueError('piece budget exceeds its interdictable arcs')
    sizing_bqm = ni.nip_to_qubo(B, di_g, src_node, tgt_node, 1.0, 2.0,
                               interdictable=interdictable)
    if len(sizing_bqm.variables) > max_qubits:
        raise ValueError(
            f'{instance} B={B}: the QUBO has {len(sizing_bqm.variables)} variables, '
            f'over the statevector limit of {max_qubits} qubits')
    with contextlib.redirect_stdout(io.StringIO()):  # Gurobi prints every solve
        if interdictable is None:
            curve = rb.optimum_curve(exact_backend, di_g, src_node, tgt_node, B)
        else:
            curve = restricted_optimum_curve(
                di_g, src_node, tgt_node, interdictable, exact_backend)
    lo, hi = rb.lambda_window(curve, B)
    if lam == 'auto':
        lam = rb.choose_lambda(lo, hi)
        if lam is None:
            raise ValueError(
                f'{instance} B={B}: the lambda window ({lo}, {hi}) has no '
                f'interior, so no lambda makes the QUBO ground state the '
                f'optimal interdiction (issue #2)')
    lam = float(lam)
    if not math.isfinite(lam) or not lo < lam < hi:
        raise ValueError(
            f'{instance} B={B}: lambda={lam} is outside the window '
            f'({lo}, {hi}); a finite, strictly interior value is required')

    bqm = ni.nip_to_qubo(B, di_g, src_node, tgt_node, lam, lam + 1,
                        interdictable=interdictable)
    if len(bqm.variables) > max_qubits:
        raise ValueError(
            f'{instance} B={B}: the QUBO has {len(bqm.variables)} variables, '
            f'over the statevector limit of {max_qubits} qubits')
    variables = variable_order(bqm) if order == 'family' else rcm_order(bqm)
    h, J, offset = ising_from_bqm(bqm, variables)
    problem = NIPQaoaProblem(
        instance=instance, di_g=di_g, src_node=src_node, tgt_node=tgt_node,
        B=B, lam=lam, lambda_window=(lo, hi), bqm=bqm, variables=variables,
        h=h, J=J, offset=offset, opt_nip=curve[B],
        interdictable=interdictable)

    if problem.exact:
        energies = problem.energies()
        ground = energies <= energies.min() + ENERGY_TOL
        if not np.all(problem.optimal_mask()[ground]):
            raise ValueError(
                f'{instance} B={B}: lambda={lam} is outside the window '
                f'({lo}, {hi}); the QUBO ground state is not an optimal '
                f'interdiction')
        problem.ground_energy = float(energies.min())
    elif interdictable is not None:
        # Strictly inside the restricted window, every ground state has B
        # interdictions. The budget term is then zero and beta > lambda
        # enforces the cut constraints. nip_to_qubo's quadratized penalty
        # carries a constant -beta for each interdictable arc.
        problem.ground_energy = float(curve[B] - (lam + 1) * len(interdictable))
    else:
        # exact_reference() raises the same way when the penalised optimum
        # does not decode to an optimal interdiction.
        with contextlib.redirect_stdout(io.StringIO()):
            _, ground_energy = rb.exact_reference(
                'gurobi', di_g, src_node, tgt_node, B, lam)
        problem.ground_energy = float(ground_energy)
    return problem


def build_problem(instance, B, lam='auto', max_qubits=DEFAULT_MAX_QUBITS,
                  exact_backend='brute', order='family'):
    """problem_from_graph() for a flow file."""
    di_g, src_node, tgt_node = ni.read_flow_file(instance)
    return problem_from_graph(di_g, src_node, tgt_node, B, lam,
                              instance=str(instance), max_qubits=max_qubits,
                              exact_backend=exact_backend, order=order)


def linear_ramp(reps, delta=0.6):
    """
    Annealing-like starting angles, theta = [beta..., gamma...]: |gamma|
    rises and beta falls linearly over the p layers, both within delta
    (the usual TQA-style initialization).

    gamma starts negative. The mixer is exp(-i beta sum X) and |+>^n is the
    *highest* eigenstate of sum X, so the discretized anneal that lowers H_C
    has gamma and beta of opposite sign. In a scan on tiny_a (B = 1, 2) and
    tiny_d (B = 1), this start reached the same p = 1 optimum for every delta
    in 0.3..1.0, while the positive one failed at delta = 1.0 on all three;
    at p = 2 and 3 on tiny_a it ended lower in three of the four cases.
    """
    steps = (np.arange(reps) + 0.5) / reps
    return np.concatenate([delta * (1 - steps), -delta * steps])


def optimize_angles(objective, reps, seed=0, maxiter=300, restarts=0,
                    initial=None):
    """
    Minimize objective(theta) with COBYLA from a linear ramp (or `initial`),
    plus `restarts` uniformly random starts with beta, gamma in [0, pi/2).

    Returns (theta, value, evaluations), the best over all starts.
    """
    rng = np.random.default_rng(seed)
    starts = [linear_ramp(reps) if initial is None
              else np.asarray(initial, dtype=float)]
    starts += [rng.uniform(0, np.pi / 2, 2 * reps) for _ in range(restarts)]
    best, evaluations = None, 0
    for start in starts:
        result = minimize(objective, start, method='COBYLA',
                          options={'maxiter': maxiter, 'rhobeg': 0.2})
        evaluations += result.nfev
        if best is None or result.fun < best.fun:
            best = result
    return np.asarray(best.x), float(best.fun), evaluations


def score_probabilities(problem, probabilities):
    """
    Exact scores of a state from its probability vector (little-endian):
    expected_energy, approx_ratio, p_ground, p_optimal.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (2 ** problem.num_qubits,):
        raise ValueError(f'expected {2 ** problem.num_qubits} probabilities, '
                         f'got shape {probabilities.shape}')
    energies = problem.energies()
    e_min, e_max = float(energies.min()), float(energies.max())
    expected = float(probabilities @ energies)
    return {
        'expected_energy': expected,
        'approx_ratio': (e_max - expected) / (e_max - e_min),
        'p_ground': float(probabilities[energies <= e_min + ENERGY_TOL].sum()),
        'p_optimal': float(probabilities[problem.optimal_mask()].sum()),
    }


def score_samples(problem, counts, num_repaired=REPAIRED_SAMPLES):
    """
    Score measured samples the way qubo_verify_parameters() scores the
    annealer's: take the `num_repaired` lowest-energy distinct samples, and
    report

        best_sample_energy  lowest QUBO energy measured
        flow_before_repair  best surviving flow among those samples that
                            already respect the budget (inf if none does)
        flow_after_repair   best surviving flow after improve_dwave_solution()
                            repairs each to exactly B arcs
        interdicted_edges   the arcs achieving flow_after_repair
        shot_p_optimal      fraction of all shots that are optimal
                            interdictions (the shot estimate of p_optimal)
        p_feasible          fraction of shots cutting at most B arcs
        shot_mean_ratio     mean over all shots of the flow-reduction ratio
                            (maxflow - flow) / (maxflow - opt_nip), the
                            measure the rest of the repository scores
                            methods by (1 = optimal), with over-budget shots
                            scoring 0; no repair. If no reduction is
                            possible (for example B = 0), an optimal
                            feasible shot scores 1 and other shots score 0.

    `counts` maps per-qubit bit tuples to shot counts.
    """
    shots = sum(counts.values())
    energy = {bits: problem.bits_energy(bits) for bits in counts}
    flows = {}

    def surviving(bits):
        """Surviving flow of the shot's cut, or None when over budget."""
        cut = problem.interdicted_edges(bits)
        if len(cut) > problem.B:
            return None
        key = frozenset(cut)
        if key not in flows:
            flows[key] = rb.max_flow(problem.di_g, problem.src_node,
                                     problem.tgt_node, cut)
        return flows[key]

    lowest = sorted(counts, key=lambda bits: (energy[bits], bits))
    lowest = lowest[:num_repaired]
    before, after, best_edges = math.inf, math.inf, []
    for bits in lowest:
        flow = surviving(bits)
        if flow is not None:
            before = min(before, flow)
        repaired, flow = ni.improve_dwave_solution(
            problem.di_g, problem.x_vector(bits), problem.src_node,
            problem.tgt_node, problem.B,
            candidate_edges=problem.interdictable_edges)
        if flow < after:
            after, best_edges = flow, repaired

    optimal = feasible = 0
    ratio_sum = 0.0
    reduction = problem.maxflow - problem.opt_nip
    for bits, count in counts.items():
        flow = surviving(bits)
        if flow is None:
            continue
        feasible += count
        optimal += count if flow == problem.opt_nip else 0
        ratio_sum += count * ((problem.maxflow - flow) / reduction
                              if reduction else float(flow == problem.opt_nip))

    return {
        'best_sample_energy': energy[lowest[0]],
        'flow_before_repair': before,
        'flow_after_repair': after,
        'interdicted_edges': best_edges,
        'shot_p_optimal': optimal / shots,
        'p_feasible': feasible / shots,
        'shot_mean_ratio': ratio_sum / shots,
    }


def uniform_scores(problem):
    """
    Scores of the uniform superposition |+>^n (QAOA at p = 0). Its
    expected energy is `offset` at any size; the other scores need every
    basis state, so past EXACT_QUBITS they are NaN.
    """
    if not problem.exact:
        return {'uniform_expected_energy': problem.offset,
                'uniform_approx_ratio': math.nan,
                'uniform_p_ground': math.nan, 'uniform_p_optimal': math.nan}
    probabilities = np.full(2 ** problem.num_qubits, 2.0 ** -problem.num_qubits)
    scores = score_probabilities(problem, probabilities)
    return {f'uniform_{key}': value for key, value in scores.items()}


def gap_closed(problem, expected_energy):
    """
    How much of the energy gap between the uniform state and the ground
    state a state closes: (offset - <H_C>) / (offset - ground energy), 0 for
    random guessing and 1 for the ground state. approx_ratio measures from
    the highest-energy state instead, which breaks many penalties at once,
    so random guessing already scores 0.575 on tiny_a and 0.564 on tiny_d
    (B = 1); this starts at 0, and it needs only the ground energy, so it
    is defined at any size.
    """
    return (problem.offset - expected_energy) / (
        problem.offset - problem.ground_energy)


@dataclass
class QaoaResult:
    """Everything run_qaoa() measured; QUBO units unless noted."""
    backend: str
    instance: str
    B: int
    lam: float
    num_qubits: int
    reps: int
    theta: list
    evaluations: int
    train_time_sec: float
    expected_energy: float
    approx_ratio: float
    p_ground: float
    p_optimal: float
    uniform_expected_energy: float
    uniform_p_optimal: float
    shots: int
    sample_time_sec: float
    best_sample_energy: float
    flow_before_repair: float
    flow_after_repair: float
    interdicted_edges: list
    shot_p_optimal: float
    opt_nip: float
    opt_qubo: float
    greedy: float
    gap_closed: float = math.nan
    p_feasible: float = math.nan
    shot_mean_ratio: float = math.nan

    def to_json(self):
        """A JSON-safe dict (inf and NaN become None)."""
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value
        return {key: clean(value) for key, value in asdict(self).items()}


def run_qaoa(backend, problem, reps, shots=4096, seed=0, maxiter=300,
             restarts=0, theta=None):
    """
    Train a depth-`reps` QAOA on `backend` -- or take the given `theta`
    unchanged, as parameter transfer does -- then score it from `shots`
    samples, and exactly where that is possible.

    `backend` must offer
        name                          a short label
        normalized_expectation(theta) <(H_C - offset) / scale>
        sample(theta, shots, seed)    {per-qubit bit tuple: count}
        exact_probabilities           whether probabilities(theta), the
                                      2**n probability vector (little-
                                      endian), is available
    and must have been built for this `problem` and `reps`. The exact scores
    (approx_ratio, p_ground, p_optimal) need those probabilities and a
    problem of at most EXACT_QUBITS qubits; otherwise they are NaN and the
    expected energy is the backend's estimate (exact for statevector,
    approximate for a truncated matrix product state). Training time
    excludes building the circuit, as SA's excludes building the QUBO.
    """
    start = time.perf_counter()
    if theta is None:
        theta, _, evaluations = optimize_angles(
            backend.normalized_expectation, reps, seed=seed, maxiter=maxiter,
            restarts=restarts)
    else:
        theta, evaluations = np.asarray(theta, dtype=float), 0
        if theta.shape != (2 * reps,):
            raise ValueError(f'expected {2 * reps} angles for p = {reps}, '
                             f'got {theta.shape}')
    train_time = time.perf_counter() - start

    if problem.exact and getattr(backend, 'exact_probabilities', True):
        exact = score_probabilities(problem, backend.probabilities(theta))
    else:
        exact = {'expected_energy': problem.scale * float(
                     backend.normalized_expectation(theta)) + problem.offset,
                 'approx_ratio': math.nan, 'p_ground': math.nan,
                 'p_optimal': math.nan}
    start = time.perf_counter()
    counts = backend.sample(theta, shots, seed)
    sample_time = time.perf_counter() - start
    sampled = score_samples(problem, counts)
    uniform = uniform_scores(problem)

    return QaoaResult(
        backend=backend.name, instance=problem.instance, B=problem.B,
        lam=problem.lam, num_qubits=problem.num_qubits, reps=reps,
        theta=[float(t) for t in theta], evaluations=evaluations,
        train_time_sec=train_time, **exact,
        uniform_expected_energy=uniform['uniform_expected_energy'],
        uniform_p_optimal=uniform['uniform_p_optimal'],
        shots=shots, sample_time_sec=sample_time, **sampled,
        opt_nip=problem.opt_nip, opt_qubo=problem.ground_energy,
        greedy=ni.improve_dwave_solution(
            problem.di_g, np.zeros(len(problem.edge_list), dtype=int),
            problem.src_node, problem.tgt_node, problem.B,
            candidate_edges=problem.interdictable_edges)[1],
        gap_closed=gap_closed(problem, exact['expected_energy']))


def format_result(result):
    """A short human-readable report of a QaoaResult."""
    return '\n'.join([
        f'{result.backend} QAOA, p={result.reps}: {result.instance} B={result.B} '
        f'lambda={result.lam} ({result.num_qubits} qubits)',
        f'  trained in {result.evaluations} evaluations, '
        f'{result.train_time_sec:.2f} s; theta (beta..., gamma...) = '
        f'{np.round(result.theta, 4).tolist()}',
        f'  <H_C> = {result.expected_energy:.4f} (uniform '
        f'{result.uniform_expected_energy:.4f}, ground {result.opt_qubo:.4f}): '
        f'gap closed {result.gap_closed:.4f}, approximation ratio '
        f'{result.approx_ratio:.4f}',
        f'  P(ground state) = {result.p_ground:.4f}, '
        f'P(optimal interdiction) = {result.p_optimal:.4f} '
        f'(uniform {result.uniform_p_optimal:.4f})'
        + ('' if math.isfinite(result.p_optimal) else
           ' -- exact scores need at most '
           f'{EXACT_QUBITS} qubits and a statevector backend'),
        f'  {result.shots} shots: P(optimal) = {result.shot_p_optimal:.4f}, '
        f'P(within budget) = {result.p_feasible:.4f}, mean flow-reduction '
        f'ratio {result.shot_mean_ratio:.4f}; best surviving flow '
        f'{result.flow_before_repair} before repair, '
        f'{result.flow_after_repair} after (optimum {result.opt_nip}, '
        f'greedy {result.greedy}); arcs {result.interdicted_edges}',
    ])


def cli(make_backend, description, argv=None, extra_args=None):
    """
    Command line shared by qaoa_qiskit.py and qaoa_cudaq.py. `make_backend`
    is called as make_backend(problem, reps, args) after `extra_args(parser)`
    has added any backend-specific options.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('instance', help='flow file, e.g. testcases/tiny_a.flow')
    parser.add_argument('--budget', '-B', type=int, required=True)
    parser.add_argument('--lam', default='auto',
                        help="cardinality penalty, or 'auto' for the midpoint "
                             "of its exact window (default)")
    parser.add_argument('--reps', '-p', type=int, default=2,
                        help='QAOA depth p (default 2)')
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--maxiter', type=int, default=300,
                        help='COBYLA iterations per start (default 300)')
    parser.add_argument('--restarts', type=int, default=0,
                        help='random starts in addition to the linear ramp')
    parser.add_argument('--max-qubits', type=int, default=DEFAULT_MAX_QUBITS)
    parser.add_argument('--exact-backend', choices=('brute', 'gurobi'),
                        default='brute',
                        help='optimal flows for the lambda window (gurobi '
                             'past a dozen arcs; needs a license)')
    parser.add_argument('--order', choices=QUBIT_ORDERS,
                        help='qubit order (default: rcm for --method mps, '
                             'family otherwise)')
    parser.add_argument('--json', help='also write the result as JSON here')
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args(argv)

    lam = args.lam if args.lam == 'auto' else float(args.lam)
    order = args.order or ('rcm' if getattr(args, 'method', None) == 'mps'
                           else 'family')
    try:
        problem = build_problem(args.instance, args.budget, lam,
                                max_qubits=args.max_qubits,
                                exact_backend=args.exact_backend, order=order)
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    backend = make_backend(problem, args.reps, args)
    result = run_qaoa(backend, problem, args.reps, shots=args.shots,
                      seed=args.seed, maxiter=args.maxiter,
                      restarts=args.restarts)
    print(format_result(result))
    if args.json:
        with open(args.json, 'w') as handle:
            json.dump(result.to_json(), handle, indent=2)
    return 0

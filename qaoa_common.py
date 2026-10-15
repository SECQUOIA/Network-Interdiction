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

# Variable families in the order qubits are assigned to them.
VARIABLE_FAMILIES = ('x', 'gamma', 'w', 'pi')

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
    def key(label):
        family, index = label.rsplit('_', 1)
        return VARIABLE_FAMILIES.index(family), int(index)
    return sorted(bqm.variables, key=key)


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
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def num_qubits(self):
        return len(self.variables)

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
        """Qubit of x_i for each arc i of edge_list."""
        return [self.variables.index(f'x_{i}') for i in range(len(self.edge_list))]

    def energies(self):
        """
        H_C on every basis state, little-endian (length 2**n), in QUBO units.
        Cached: it is 2**n floats and every exact score reads it.
        """
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
        if 'optimal_mask' not in self._cache:
            num_edges = len(self.edge_list)
            optimal_pattern = np.zeros(2 ** num_edges, dtype=bool)
            for pattern in range(2 ** num_edges):
                cut = [self.edge_list[e] for e in range(num_edges)
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
        return [edge for edge, qubit in zip(self.edge_list, self.x_qubits)
                if bits[qubit] == 1]

    def x_vector(self, bits):
        """The x array improve_dwave_solution() takes, indexed by edge_list."""
        return np.array([bits[qubit] for qubit in self.x_qubits], dtype=int)


def problem_from_graph(di_g, src_node, tgt_node, B, lam='auto', instance='',
                       max_qubits=DEFAULT_MAX_QUBITS):
    """
    The QAOA problem for one graph and budget. `lam` is a number, or 'auto'
    for the midpoint of the exact window (run_benchmark.choose_lambda()).

    Raises ValueError when the window is empty (no lambda makes the QUBO's
    ground state the optimal interdiction, so there is nothing for QAOA to
    find), when a given `lam` does not encode the optimum, or when the QUBO
    is over `max_qubits`.
    """
    # Enforce the simulation limit before the combinatorial reference scan.
    # Compile with positive placeholder penalties to count the actual QUBO
    # variables, including auxiliaries, without first solving for lambda.
    sizing_bqm = ni.nip_to_qubo(B, di_g, src_node, tgt_node, 1.0, 2.0)
    if len(sizing_bqm.variables) > max_qubits:
        raise ValueError(
            f'{instance} B={B}: the QUBO has {len(sizing_bqm.variables)} variables, '
            f'over the statevector limit of {max_qubits} qubits')
    curve = rb.optimum_curve('brute', di_g, src_node, tgt_node, B)
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

    bqm = ni.nip_to_qubo(B, di_g, src_node, tgt_node, lam, lam + 1)
    if len(bqm.variables) > max_qubits:
        raise ValueError(
            f'{instance} B={B}: the QUBO has {len(bqm.variables)} variables, '
            f'over the statevector limit of {max_qubits} qubits')
    variables = variable_order(bqm)
    h, J, offset = ising_from_bqm(bqm, variables)
    problem = NIPQaoaProblem(
        instance=instance, di_g=di_g, src_node=src_node, tgt_node=tgt_node,
        B=B, lam=lam, lambda_window=(lo, hi), bqm=bqm, variables=variables,
        h=h, J=J, offset=offset, opt_nip=curve[B])

    energies = problem.energies()
    ground = energies <= energies.min() + ENERGY_TOL
    if not np.all(problem.optimal_mask()[ground]):
        raise ValueError(
            f'{instance} B={B}: lambda={lam} is outside the window '
            f'({lo}, {hi}); the QUBO ground state is not an optimal '
            f'interdiction')
    return problem


def build_problem(instance, B, lam='auto', max_qubits=DEFAULT_MAX_QUBITS):
    """problem_from_graph() for a flow file."""
    di_g, src_node, tgt_node = ni.read_flow_file(instance)
    return problem_from_graph(di_g, src_node, tgt_node, B, lam,
                              instance=str(instance), max_qubits=max_qubits)


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

    `counts` maps per-qubit bit tuples to shot counts.
    """
    shots = sum(counts.values())
    energy = {bits: problem.bits_energy(bits) for bits in counts}
    lowest = sorted(counts, key=lambda bits: (energy[bits], bits))
    lowest = lowest[:num_repaired]

    before, after, best_edges = math.inf, math.inf, []
    for bits in lowest:
        cut = problem.interdicted_edges(bits)
        if len(cut) <= problem.B:
            before = min(before, rb.max_flow(
                problem.di_g, problem.src_node, problem.tgt_node, cut))
        repaired, flow = ni.improve_dwave_solution(
            problem.di_g, problem.x_vector(bits), problem.src_node,
            problem.tgt_node, problem.B)
        if flow < after:
            after, best_edges = flow, repaired

    optimal = 0
    for bits, count in counts.items():
        cut = problem.interdicted_edges(bits)
        if len(cut) <= problem.B and rb.max_flow(
                problem.di_g, problem.src_node, problem.tgt_node,
                cut) == problem.opt_nip:
            optimal += count

    return {
        'best_sample_energy': energy[lowest[0]],
        'flow_before_repair': before,
        'flow_after_repair': after,
        'interdicted_edges': best_edges,
        'shot_p_optimal': optimal / shots,
    }


def uniform_scores(problem):
    """Scores of the uniform superposition |+>^n (QAOA at p = 0)."""
    probabilities = np.full(2 ** problem.num_qubits, 2.0 ** -problem.num_qubits)
    scores = score_probabilities(problem, probabilities)
    return {f'uniform_{key}': value for key, value in scores.items()}


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

    def to_json(self):
        """A JSON-safe dict (inf becomes None)."""
        def clean(value):
            if isinstance(value, float) and math.isinf(value):
                return None
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value
        return {key: clean(value) for key, value in asdict(self).items()}


def run_qaoa(backend, problem, reps, shots=4096, seed=0, maxiter=300,
             restarts=0):
    """
    Train a depth-`reps` QAOA on `backend`, then score it exactly and from
    `shots` samples.

    `backend` must offer
        name                          a short label
        normalized_expectation(theta) <(H_C - offset) / scale>, exact
        probabilities(theta)          the 2**n probability vector, little-endian
        sample(theta, shots, seed)    {per-qubit bit tuple: count}
    and must have been built for this `problem` and `reps`. Training time
    excludes building the circuit, as SA's excludes building the QUBO.
    """
    start = time.perf_counter()
    theta, _, evaluations = optimize_angles(
        backend.normalized_expectation, reps, seed=seed, maxiter=maxiter,
        restarts=restarts)
    train_time = time.perf_counter() - start

    exact = score_probabilities(problem, backend.probabilities(theta))
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
        opt_nip=problem.opt_nip, opt_qubo=float(problem.energies().min()),
        greedy=rb.greedy_from_scratch(problem.di_g, problem.src_node,
                                      problem.tgt_node, problem.B))


def format_result(result):
    """A short human-readable report of a QaoaResult."""
    return '\n'.join([
        f'{result.backend} QAOA, p={result.reps}: {result.instance} B={result.B} '
        f'lambda={result.lam} ({result.num_qubits} qubits)',
        f'  trained in {result.evaluations} evaluations, '
        f'{result.train_time_sec:.2f} s; theta (beta..., gamma...) = '
        f'{np.round(result.theta, 4).tolist()}',
        f'  <H_C> = {result.expected_energy:.4f} (uniform '
        f'{result.uniform_expected_energy:.4f}, ground {result.opt_qubo:.4f}), '
        f'approximation ratio {result.approx_ratio:.4f}',
        f'  P(ground state) = {result.p_ground:.4f}, '
        f'P(optimal interdiction) = {result.p_optimal:.4f} '
        f'(uniform {result.uniform_p_optimal:.4f})',
        f'  {result.shots} shots: P(optimal) = {result.shot_p_optimal:.4f}; '
        f'best surviving flow {result.flow_before_repair} before repair, '
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
    parser.add_argument('--json', help='also write the result as JSON here')
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args(argv)

    lam = args.lam if args.lam == 'auto' else float(args.lam)
    try:
        problem = build_problem(args.instance, args.budget, lam,
                                max_qubits=args.max_qubits)
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

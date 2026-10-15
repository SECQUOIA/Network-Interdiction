"""
Produce the results CSV that network_interdiction_plot.py reads (issue #6).

The plotting script used to read a `result_dwave_planar.csv` that was never
checked in and had no generating script, so it could not be run at all. This
module is that generating script: for each (instance, budget, penalty
coefficient) case it computes every column the plot consumes and writes them
as one CSV row.

Columns
-------
instance        path to the flow file the row was computed from
V, E, B         instance size and interdiction budget
lambda          cardinality penalty coefficient; nip_to_qubo() is called with
                (lambda, lambda+1) as (cardinality, dual) coefficients, the
                same pairing qubo_verify_parameters() uses
lambda_lo,      for a case run with lambda `auto`: the exact window of
lambda_hi       coefficients whose QUBO ground state is the optimal B-arc
                interdiction (see "Choosing lambda" below)
maxflow         max flow of the un-interdicted instance
Opt_NIP         optimal surviving flow under the budget (exact)
Opt_QUBO        exact ground-state energy of the QUBO at that lambda
SA_NIP          best surviving flow over the annealer's 20 best samples that
                already respect the budget
SA_NIP_Greedy   same, after improve_dwave_solution() repairs each sample to
                exactly B interdicted edges
SA_QUBO         lowest QUBO energy the annealer reached
QA_*            the same three columns for the D-Wave QPU path
QCI_NIP, QCI_NIP_Greedy, QCI_QUBO
                corresponding Dirac-3 binary-mode columns; optional QCI_*
                record diagnostics follow the same <=20-record protocol
QCI_time_sec    worker startup, authentication, upload, queue and sampling wall
                time; QCI_repair_time_sec measures diagnostics and greedy repair
QCI_device_usage_sec
                vendor-reported rounded billable device usage, NOT precise
                execution/QPU time; empty when unavailable or QCI not run
QCI_job_id, QCI_device, QCI_client_version
                identifies the actual submitted job, device and isolated SDK
QAOA_NIP, QAOA_NIP_Greedy, QAOA_QUBO
                corresponding Qiskit Aer penalty-QAOA results, with the same
                raw/pre-greedy diagnostics as other samplers
QAOA_time_sec   circuit building, angle training and shots; excludes BQM and
                exact-reference construction; QAOA_repair_time_sec is separate
QAOA_metadata   JSON: controls, angles, variable order, versions, stage wall times,
                evaluations and circuit depth/two-qubit gates before hardware routing
SA_raw_feasible_count, QA_raw_feasible_count
                budget-feasible records among <=20 lowest-energy raw samples;
                *_raw_samples_examined is the actual denominator
SA_raw_samples, QA_raw_samples
                JSON records with energies, occurrence counts, interdicted
                arcs, actual surviving flows and budget-feasibility flags
*_pre_greedy_*  corresponding diagnostics for the greedy-repair input pool;
                for QA this is after classical steepest descent, so it can
                differ from raw; SA has no descent stage
Greedy          surviving flow after interdicting B arcs greedily from an
                empty interdiction, one highest-impact arc at a time
Random_Greedy   best surviving flow over 20 random interdictions (each arc
                cut independently with probability B/E), each repaired to
                exactly B arcs by the same greedy pass
Gurobi_time_sec solver-only Runtime for a dedicated solve
                (--gurobi-threads threads, default 8)
SA_time_sec     sampler call only; SA_repair_time_sec is the decode,
                evaluation and repair of its 20 best samples
Gurobi_total_time_sec, SA_total_time_sec
                method construction, solve and scoring, in seconds
Reference_time_sec exact references, QUBO verification and any auto-lambda
                calibration, recorded separately from the method totals
Greedy_time_sec, Random_Greedy_time_sec, Lagrangian_time_sec, Benders_time_sec
                complete baseline pass including construction and scoring;
                LP_time_sec includes construction/solve/certification and
                LP_repair_time_sec separately includes rounding/repair/scoring
Gurobi_trajectory  JSON list of [seconds, surviving flow] for every incumbent
                Gurobi found in that dedicated solve (MIPSOL callback); only
                with --exact gurobi

`instance` and `lambda` are additions to what the plot reads; pandas ignores
them. The two baseline columns are optional for the plot: it draws them when
present and non-empty.

Exact-reference backends (--exact)
----------------------------------
gurobi  Opt_NIP from solve_nip_gurobi(), Opt_QUBO from
        gurobi_verify_parameters(). This is what the original runs used, and
        it is the only backend that scales to the testcases/*.planarflow
        instances -- but it needs a real Gurobi license (see README).
brute   Opt_NIP by enumerating every edge subset of size <= B, Opt_QUBO by
        dimod.ExactSolver over the whole BQM. No license required, but the
        BQM has 3E + V - 2 variables, so it is capped at --max-exact-vars
        (24 by default, ~20s and a few hundred MB). This is the backend that
        produced the checked-in sample CSV, which is why that sample uses the
        deliberately tiny testcases/tiny_*.flow instances.

Solver passes (--solvers)
-------------------------
`sa` needs no credentials. `qa` needs a D-Wave Leap token (see README); the
columns of a pass that was not run are left empty, and the plotting script
skips a series whose columns are empty rather than drawing empty bars.

Classical baselines (--baselines)
---------------------------------
`greedy` and `random` (both run by default; `none` disables them) are the
controls for the annealer passes (issue #15). The annealer pipeline decodes the
20 lowest-energy samples and repairs each with improve_dwave_solution(), so its
reported *_Greedy number mixes two contributions: what the sampler found, and
what the repair recovered. The baselines hold the repair fixed and replace the
sampler: `greedy` starts it from nothing, `random` starts it from 20 samples
that carry no information about the instance beyond the budget. An annealer
pass that does not beat these on an instance contributed nothing measurable
there. Both need no credentials. `random` is seeded (--seed) so a CSV is
reproducible.

Opt-in `lp` solves the continuous relaxation, rounds x >= 0.5, and applies
the same repair. `lagrangian` runs the existing binary Gurobi multiplier
search and retains its iteration bounds. LP_lower_bound and
Lagrangian_lower_bound are lower bounds on surviving flow; Lower_bound is
their maximum. Every available feasible result F gets an optional *_gap
column (F-Lower_bound)/F (zero flow has gap zero). Missing results stay
empty. LP_Round_Repair and Lagrangian_Feasible are feasible flows, not
bounds. The LP and Lagrangian times include construction; LP repair is
timed separately. See README for the formulations and numerical contracts.

`benders` uses a binary interdiction master and integer max-flow cuts.
Benders_status distinguishes a closed gap from a time/iteration limit;
Benders and Benders_lower_bound bracket the optimum even on an incomplete
run, subject to the master solver's numerical tolerances. The default
cooperative wall limit is 60 seconds, with at most 1000 master solves.

Choosing lambda (`auto`)
------------------------
The budget term in nip_to_qubo() is linear, lambda * (sum(x) - B), so the
QUBO ground state minimizes f(k) + lambda * k over the number k of interdicted
arcs, where f(k) is the best surviving flow using at most k arcs. It
interdicts exactly B arcs and attains f(B) if and only if

    max_{k>B} (f(B) - f(k)) / (k - B)  <  lambda  <  min_{k<B} (f(k) - f(B)) / (B - k)

(Proposition 3 of mfnip-qubo-research-notes; the dual coefficient lambda+1 is
always safe, Proposition 2). Writing `auto` for the lambda field of a --case
computes f with the --exact backend -- f(0..B) plus f(k) for k > B up to its
first zero, past which no k can tighten the lower end -- and uses the
midpoint of that window. A strict window exists only when B is a vertex of
the lower convex envelope of f. If lo > hi, B lies above that envelope; if
lo == hi, other cut counts tie at the only candidate lambda, even when f is
convex (for example f = [10, 5, 0] at B = 1). Neither case forces every
ground state to use exactly B cuts, so the row keeps the exact optimum and
the baselines but leaves lambda, Opt_QUBO and the solver columns blank.

Examples
--------
Regenerate the checked-in sample (no license or token needed):

    python run_benchmark.py --exact brute --solvers sa \\
        --case testcases/tiny_a.flow:1:auto \\
        --out results/sample_results_sa.csv

A real run over a planar instance (needs Gurobi, and a Leap token for qa):

    python run_benchmark.py --exact gurobi --solvers sa,qa \\
        --case testcases/50.planarflow:10:auto \\
        --out result_dwave_planar.csv
"""
import argparse
import csv
import datetime
import itertools
import json
import math
import subprocess
import sys
import time

import dimod
import networkx as nx
import numpy as np

import network_interdiction as ni
import classical_baselines as cb
import benders

# Every column network_interdiction_plot.py reads, in the order it reads them.
PLOT_COLUMNS = [
    'V', 'E', 'B',
    'Opt_NIP', 'Opt_QUBO',
    'SA_NIP', 'SA_NIP_Greedy', 'SA_QUBO',
    'QA_NIP(advantage1)', 'QA_NIP(advantage1)_Greedy', 'QA_QUBO(advantage1)',
    'maxflow',
]

# What run_benchmark.py actually writes: the plot's columns plus the instance
# path and lambda. The plot ignores those two, but without them a row cannot
# be traced back to -- or recomputed from -- the instance that produced it.
CSV_COLUMNS = ['instance'] + PLOT_COLUMNS[:3] + \
    ['lambda', 'lambda_lo', 'lambda_hi'] + PLOT_COLUMNS[3:]

SOLVER_COLUMNS = {
    'sa': ('SA_NIP', 'SA_NIP_Greedy', 'SA_QUBO'),
    'qa': ('QA_NIP(advantage1)', 'QA_NIP(advantage1)_Greedy',
           'QA_QUBO(advantage1)'),
    'qci': ('QCI_NIP', 'QCI_NIP_Greedy', 'QCI_QUBO'),
    'qaoa': ('QAOA_NIP', 'QAOA_NIP_Greedy', 'QAOA_QUBO'),
}

# Classical baseline passes -> the column each fills. Kept out of
# PLOT_COLUMNS: the plot treats them as optional, so CSVs written before they
# existed still plot.
BASELINE_COLUMNS = {
    'greedy': 'Greedy',
    'random': 'Random_Greedy',
    'lp': 'LP_Round_Repair',
    'lagrangian': 'Lagrangian_Feasible',
    'benders': 'Benders',
}

# Wall time per method, and Gurobi's incumbent trajectory (last: it is long).
TIME_COLUMNS = ['Gurobi_time_sec', 'SA_time_sec', 'SA_repair_time_sec',
                'Greedy_time_sec', 'Random_Greedy_time_sec', 'LP_time_sec',
                'LP_repair_time_sec', 'Lagrangian_time_sec', 'Benders_time_sec',
                'Gurobi_total_time_sec', 'SA_total_time_sec', 'Reference_time_sec',
                'QCI_time_sec', 'QCI_repair_time_sec',
                'QAOA_time_sec', 'QAOA_repair_time_sec']
QCI_METADATA_COLUMNS = ['QCI_device_usage_sec', 'QCI_job_id', 'QCI_device',
                        'QCI_client_version']
TRAJECTORY_COLUMN = 'Gurobi_trajectory'
FLOW_COLUMNS = [col for cols in SOLVER_COLUMNS.values() for col in cols[:2]] \
    + list(BASELINE_COLUMNS.values())
BOUND_COLUMNS = ['LP_lower_bound', 'LP_objective', 'Lagrangian_lower_bound',
                 'Lower_bound', 'Lagrangian_iterations', 'Benders_lower_bound',
                 'Benders_status', 'Benders_iterations', 'Benders_history',
                 'Benders_interdicted_edges'] \
    + [col + '_gap' for col in FLOW_COLUMNS]
SAMPLE_FIELDS = ['raw_feasible_count', 'raw_samples_examined', 'raw_samples',
                 'pre_greedy_feasible_count', 'pre_greedy_samples_examined',
                 'pre_greedy_samples']
SAMPLE_COLUMNS = [f'{solver.upper()}_{field}'
                  for solver in SOLVER_COLUMNS for field in SAMPLE_FIELDS]

CSV_COLUMNS = CSV_COLUMNS[:-1] + list(BASELINE_COLUMNS.values()) + \
    CSV_COLUMNS[-1:] + list(SOLVER_COLUMNS['qci']) + list(SOLVER_COLUMNS['qaoa']) + \
    TIME_COLUMNS + BOUND_COLUMNS + SAMPLE_COLUMNS + QCI_METADATA_COLUMNS + \
    ['QAOA_metadata', TRAJECTORY_COLUMN]

DEFAULT_MAX_EXACT_VARS = 24

# Matches the 20 lowest-energy samples qubo_verify_parameters() decodes, so
# the random baseline repairs as many starting points as the annealer pass.
RANDOM_BASELINE_SAMPLES = 20


def max_flow(di_g, src_node, tgt_node, interdicted_edges=()):
    """Max flow of `di_g` with `interdicted_edges` removed."""
    di_g_copy = di_g.copy()
    di_g_copy.remove_edges_from(interdicted_edges)
    flow_value, _ = nx.maximum_flow(di_g_copy, src_node, tgt_node,
                                    capacity='weight')
    return flow_value


def sorted_edges(di_g):
    """The edge order nip_to_qubo() indexes x_i/gamma_i/w_i by."""
    return sorted(di_g.edges(), key=lambda element: (element[0], element[1]))


def brute_force_nip_optimum(di_g, src_node, tgt_node, B):
    """
    Exact NIP optimum by enumerating every edge subset of size <= B.

    Returns (objective, interdicted_edges). Exhaustive, so it is only usable
    on the tiny instances the sample CSV is built from -- but it needs no
    solver license, which is what makes that sample regenerable by anyone.
    """
    edge_list = sorted_edges(di_g)
    best_obj = None
    best_edges = None
    for size in range(B + 1):
        for candidate in itertools.combinations(edge_list, size):
            obj = max_flow(di_g, src_node, tgt_node, candidate)
            if best_obj is None or obj < best_obj:
                best_obj, best_edges = obj, list(candidate)
    return best_obj, best_edges


def brute_force_qubo_optimum(di_g, src_node, tgt_node, B, curr_λ,
                             max_exact_vars=DEFAULT_MAX_EXACT_VARS):
    """
    Exact QUBO ground state via dimod.ExactSolver.

    Returns (energy, interdicted_edges) where interdicted_edges is the ground
    state decoded through the same x_i convention nip_to_qubo() encodes, so
    the caller can check the QUBO optimum really does encode the NIP optimum
    at this lambda. Raises ValueError past `max_exact_vars`, since the solver
    enumerates all 2**n assignments.
    """
    bqm = ni.nip_to_qubo(B, di_g, src_node, tgt_node, curr_λ, curr_λ + 1)
    if len(bqm.variables) > max_exact_vars:
        raise ValueError(
            f"QUBO has {len(bqm.variables)} variables, over the "
            f"--max-exact-vars limit of {max_exact_vars}; use "
            f"--exact gurobi for an instance this size")
    best = dimod.ExactSolver().sample(bqm).first
    edge_list = sorted_edges(di_g)
    interdicted = [edge_list[i] for i in range(len(edge_list))
                   if best.sample[f'x_{i}'] == 1]
    return best.energy, interdicted


def exact_reference(backend, di_g, src_node, tgt_node, B, curr_λ,
                    max_exact_vars=DEFAULT_MAX_EXACT_VARS):
    """
    Opt_NIP and Opt_QUBO for one case, via the requested backend.

    Both backends also cross-check that the QUBO optimum at this lambda
    decodes to a solution attaining the NIP optimum. When it does not, the
    penalty coefficient is wrong for this instance (issue #2) and every
    SA_*/QA_* ratio in that row would be measured against a QUBO whose
    optimum is not the answer, so this stops rather than emitting the row.
    """
    if backend == 'brute':
        opt_nip, _ = brute_force_nip_optimum(di_g, src_node, tgt_node, B)
        opt_qubo, qubo_edges = brute_force_qubo_optimum(
            di_g, src_node, tgt_node, B, curr_λ, max_exact_vars)
    elif backend == 'gurobi':
        opt_nip, _ = ni.solve_nip_gurobi(di_g, src_node, tgt_node, B)
        opt_qubo, _, qubo_edges = ni.gurobi_verify_parameters(
            di_g, src_node, tgt_node, B, curr_λ, curr_λ + 1)
    else:
        raise ValueError(f"unknown exact backend: {backend}")

    qubo_obj = max_flow(di_g, src_node, tgt_node, qubo_edges)
    if len(qubo_edges) > B or qubo_obj != opt_nip:
        raise ValueError(
            f"lambda={curr_λ} does not encode this instance: the QUBO "
            f"optimum interdicts {len(qubo_edges)} edges (budget {B}) for a "
            f"flow of {qubo_obj}, but the NIP optimum is {opt_nip}. Pick a "
            f"different lambda (see issue #2) rather than plotting ratios "
            f"against a QUBO whose optimum is not the answer.")
    return opt_nip, opt_qubo


def nip_optimum(backend, di_g, src_node, tgt_node, k):
    """f(k): the best surviving flow using at most k interdicted arcs."""
    if backend == 'brute':
        return brute_force_nip_optimum(di_g, src_node, tgt_node, k)[0]
    if backend == 'gurobi':
        # Capacities are integers (read_flow_file), so the optimum is too;
        # rounding strips MIP tolerance noise that would otherwise leak into
        # the window's difference quotients and the curve's zero test.
        return round(ni.solve_nip_gurobi(di_g, src_node, tgt_node, k)[0])
    raise ValueError(f"unknown exact backend: {backend}")


def optimum_curve(backend, di_g, src_node, tgt_node, B):
    """
    [f(0), f(1), ...] far enough to determine the lambda window at B.

    That is f(0..B), then f(k) for k > B up to and including its first zero:
    f is nonincreasing and nonnegative, so once f(k0) = 0 every larger k has
    f = 0 and a smaller slope (f(B) - 0) / (k - B) than k0 does, and cannot
    move the lower end of the window. f reaches 0 by the time every arc out
    of the source is cut, so the scan is short on any instance with a small
    source degree.
    """
    num_edges = len(di_g.edges())
    curve = []
    for k in range(num_edges + 1):
        curve.append(nip_optimum(backend, di_g, src_node, tgt_node, k))
        if k > B and curve[-1] == 0:
            break
    return curve


def lambda_window(curve, B):
    """
    (lo, hi): the open interval of cardinality coefficients lambda for which
    the QUBO ground state interdicts exactly B arcs and attains f(B), given
    curve = optimum_curve(...). Non-empty iff lo < hi, i.e. iff B is a vertex
    of the lower convex envelope of f. hi is inf when B = 0.
    """
    hi = min(((curve[k] - curve[B]) / (B - k) for k in range(B)),
             default=math.inf)
    lo = max(((curve[B] - curve[k]) / (k - B)
              for k in range(B + 1, len(curve))), default=0.0)
    return lo, hi


def choose_lambda(lo, hi):
    """
    The midpoint of the window, or None when it is empty. The endpoints
    themselves are ties (a degenerate ground state), so a strictly interior
    point is required.
    """
    if lo >= hi:
        return None
    if math.isinf(hi):
        return lo + 1.0
    return (lo + hi) / 2


def greedy_from_scratch(di_g, src_node, tgt_node, B):
    """
    Surviving max flow after interdicting B arcs greedily from nothing.

    This is improve_dwave_solution() started from the empty interdiction, so
    it adds the single highest-impact arc B times (ties go to the first arc
    in sorted order). Greedy is not optimal in general -- a single arc that
    cuts the most flow now can be the wrong first move for a pair -- but on
    the random families in testcases/ it frequently is, which is why it is
    the number an annealer result has to beat.
    """
    x = np.zeros(len(di_g.edges()), dtype=int)
    _, objective = ni.improve_dwave_solution(di_g, x, src_node, tgt_node, B)
    return objective


def random_greedy(di_g, src_node, tgt_node, B, rng,
                  num_samples=RANDOM_BASELINE_SAMPLES):
    """
    Best surviving flow over `num_samples` random interdictions, each
    repaired to exactly B arcs by improve_dwave_solution().

    Each arc is cut independently with probability B/E, so a sample has about
    the budgeted number of cuts but no information about which arcs matter.
    That is the null model for the annealer pass: same count of starting
    points, same repair, samples that know the budget and nothing else. (A
    uniform random bitstring would cut about E/2 arcs and make the repair do
    all the work, which measures the repair, not the sampler.)
    """
    num_edges = len(di_g.edges())
    p_cut = min(1.0, B / num_edges) if num_edges else 0.0
    best = math.inf
    for _ in range(num_samples):
        x = (rng.random(num_edges) < p_cut).astype(int)
        _, objective = ni.improve_dwave_solution(
            di_g, x, src_node, tgt_node, B)
        best = min(best, objective)
    return best


def gurobi_trajectory(di_g, src_node, tgt_node, B, threads):
    """
    (runtime_sec, JSON trajectory, optimal flow) from a dedicated Gurobi
    solve with the incumbent-recording callback.
    """
    result = ni.solve_nip_gurobi_trajectory(di_g, src_node, tgt_node, B,
                                            threads=threads)
    trajectory = json.dumps([[round(t, 6), round(float(flow), 6)]
                             for t, flow in result['trajectory']])
    return result['runtime_sec'], trajectory, result['objective']


def run_case(instance, B, curr_λ, exact_backend, solvers,
             max_exact_vars=DEFAULT_MAX_EXACT_VARS, baselines=(), rng=None,
             sa_seed=None, gurobi_threads=ni.DEFAULT_GUROBI_THREADS,
             benders_time_limit=60.0, benders_max_iterations=1000,
             qci_options=None, qaoa_options=None):
    """
    Compute one CSV row. `instance` is a path to a flow file; `curr_λ` is a
    float or 'auto' (pick it from the exact window, see the module
    docstring); `baselines` is any subset of BASELINE_COLUMNS' keys, and `rng`
    a numpy Generator for the random baseline (a fresh unseeded one if
    omitted). `sa_seed` seeds simulated annealing. With the gurobi backend
    the row also gets Gurobi's runtime and incumbent trajectory from a
    dedicated solve on `gurobi_threads` threads.
    """
    di_g, src_node, tgt_node = ni.read_flow_file(instance)

    if 'qaoa' in solvers:
        from solve_qubo_via_qaoa import validate_options, validate_bqm
        options = dict(qaoa_options or {})
        engine = options.pop('engine', 'aer')
        if engine == 'ibm':
            from qaoa_ibm import validate_options as validate_ibm_options, is_local_backend
            backend = options.pop('backend')
            validate_ibm_options(seed_transpiler=0 if sa_seed is None else sa_seed, **options)
            qubit_limit = min(options.get('max_qubits', 127), backend.num_qubits)
            if is_local_backend(backend):
                qubit_limit = min(qubit_limit, 24)
        elif engine == 'aer':
            validate_options(seed=0 if sa_seed is None else sa_seed, **options)
            qubit_limit = options.get('max_qubits', 24)
        else:
            raise ValueError(f'unknown QAOA engine {engine!r}')
        # Positive placeholders count actual compiled variables before an exact
        # reference or penalty-window scan can dispatch expensive solves.
        sizing_bqm = ni.nip_to_qubo(B, di_g, src_node, tgt_node, 1, 2)
        validate_bqm(sizing_bqm, qubit_limit)

    row = {column: '' for column in CSV_COLUMNS}
    row.update({
        'instance': instance,
        'V': len(di_g.nodes()),
        'E': len(di_g.edges()),
        'B': B,
        'maxflow': max_flow(di_g, src_node, tgt_node),
    })

    reference_start = time.perf_counter()
    if curr_λ == 'auto':
        curve = optimum_curve(exact_backend, di_g, src_node, tgt_node, B)
        lo, hi = lambda_window(curve, B)
        row.update({'lambda_lo': lo, 'lambda_hi': hi, 'Opt_NIP': curve[B]})
        curr_λ = choose_lambda(lo, hi)
        if curr_λ is None:
            reason = ('tied budgets' if lo == hi else
                      'B above the lower convex envelope')
            print(f'{instance} B={B}: lambda window ({lo}, {hi}) is empty -- '
                  f'{reason}, so no QUBO row; baselines only')
            solvers = []

    if curr_λ is not None:
        opt_nip, opt_qubo = exact_reference(
            exact_backend, di_g, src_node, tgt_node, B, curr_λ,
            max_exact_vars)
        row.update({'lambda': curr_λ, 'Opt_NIP': opt_nip,
                    'Opt_QUBO': opt_qubo})
    row['Reference_time_sec'] = time.perf_counter() - reference_start

    if exact_backend == 'gurobi':
        start = time.perf_counter()
        runtime, trajectory, objective = gurobi_trajectory(
            di_g, src_node, tgt_node, B, gurobi_threads)
        row['Gurobi_total_time_sec'] = time.perf_counter() - start
        if round(objective) != round(float(row['Opt_NIP'])):
            raise ValueError(
                f'{instance} B={B}: the trajectory solve found {objective}, '
                f'the exact reference {row["Opt_NIP"]}')
        row.update({'Gurobi_time_sec': runtime, TRAJECTORY_COLUMN: trajectory})

    for solver in solvers:
        start = time.perf_counter()
        options = {'qci_options': qci_options} if solver == 'qci' else {}
        if solver == 'qaoa':
            options = {'qaoa_options': qaoa_options}
        result = ni.qubo_verify_parameters(
            str(instance), di_g, curr_λ, src_node, tgt_node, B, solver=solver,
            seed=sa_seed, **options)
        if solver == 'sa':
            row['SA_total_time_sec'] = time.perf_counter() - start
            row['SA_time_sec'] = result['sample_time_sec']
            row['SA_repair_time_sec'] = result['postprocess_time_sec']
        if solver == 'qci':
            row['QCI_time_sec'] = result['sample_time_sec']
            row['QCI_repair_time_sec'] = result['postprocess_time_sec']
            info = result['sampler_info']
            for field in QCI_METADATA_COLUMNS:
                value = info.get(field.lower())
                row[field] = '' if value is None else value
        if solver == 'qaoa':
            row['QAOA_time_sec'] = result['sample_time_sec']
            row['QAOA_repair_time_sec'] = result['postprocess_time_sec']
            row['QAOA_metadata'] = json.dumps(result['sampler_info']['qaoa_metadata'])
        nip_col, greedy_col, qubo_col = SOLVER_COLUMNS[solver]
        before = result['best_obj_before_greedy']
        # inf means not one of the 20 best samples stayed inside the budget.
        # Leave it blank: the plot reads a 0 in this column as "this solver
        # was not run" (it special-cases 0 to a zero-height bar), so writing
        # a sentinel number here would be read as data.
        row[nip_col] = '' if math.isinf(before) else before
        row[greedy_col] = result['best_obj_after_greedy']
        row[qubo_col] = result['best_energy']
        for field in SAMPLE_FIELDS:
            value = result[field]
            row[f'{solver.upper()}_{field}'] = (json.dumps(value)
                                               if isinstance(value, list) else value)

    if 'greedy' in baselines:
        start = time.perf_counter()
        row[BASELINE_COLUMNS['greedy']] = greedy_from_scratch(
            di_g, src_node, tgt_node, B)
        row['Greedy_time_sec'] = time.perf_counter() - start
    if 'random' in baselines:
        start = time.perf_counter()
        row[BASELINE_COLUMNS['random']] = random_greedy(
            di_g, src_node, tgt_node, B,
            rng if rng is not None else np.random.default_rng())
        row['Random_Greedy_time_sec'] = time.perf_counter() - start

    bounds = []
    if 'lp' in baselines:
        result = cb.lp_round_repair(di_g, src_node, tgt_node, B)
        row.update({'LP_Round_Repair': result['flow'],
                    'LP_lower_bound': result['lower_bound'],
                    'LP_objective': result['lp_objective'],
                    'LP_time_sec': result['lp_time_sec'],
                    'LP_repair_time_sec': result['repair_time_sec']})
        bounds.append(result['lower_bound'])
    if 'lagrangian' in baselines:
        start = time.perf_counter()
        result = ni.solve_nip_gurobi_relaxation(
            di_g, src_node, tgt_node, B, 0, row['maxflow'])
        row.update({'Lagrangian_lower_bound': result['lower_bound'],
                    'Lagrangian_Feasible': (result['feasible_flow']
                                           if result['feasible_flow'] is not None else ''),
                    'Lagrangian_iterations': json.dumps(result['iterations']),
                    'Lagrangian_time_sec': time.perf_counter() - start})
        bounds.append(result['lower_bound'])
    if 'benders' in baselines:
        result = benders.solve(di_g, src_node, tgt_node, B,
                               time_limit=benders_time_limit,
                               max_iterations=benders_max_iterations)
        row.update({'Benders': result['flow'],
                    'Benders_lower_bound': result['lower_bound'],
                    'Benders_status': result['status'],
                    'Benders_iterations': result['iterations'],
                    'Benders_history': json.dumps(result['history']),
                    'Benders_interdicted_edges': json.dumps(result['interdicted_edges']),
                    'Benders_time_sec': result['runtime_sec']})
        bounds.append(result['lower_bound'])
    if bounds:
        row['Lower_bound'] = max(bounds)
        for column in FLOW_COLUMNS:
            if row[column] != '':
                row[column + '_gap'] = cb.relative_gap(row[column], row['Lower_bound'])

    return row


def parse_case(spec):
    """
    Parse a --case argument of the form PATH:BUDGET:LAMBDA, where LAMBDA is
    a number or `auto`.
    """
    parts = spec.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected PATH:BUDGET:LAMBDA, got {spec!r}")
    path, budget, lam = parts
    try:
        return path, int(budget), 'auto' if lam == 'auto' else float(lam)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"budget must be an int and lambda a float or 'auto', "
            f"got {spec!r}")


def parse_solvers(spec):
    if spec == 'none':
        return []
    solvers = [s.strip() for s in spec.split(',') if s.strip()]
    for solver in solvers:
        if solver not in SOLVER_COLUMNS:
            raise argparse.ArgumentTypeError(
                f"unknown solver {solver!r}, expected one of {tuple(SOLVER_COLUMNS)}")
    return solvers


def parse_baselines(spec):
    if spec == 'none':
        return []
    baselines = [b.strip() for b in spec.split(',') if b.strip()]
    for baseline in baselines:
        if baseline not in BASELINE_COLUMNS:
            raise argparse.ArgumentTypeError(
                f"unknown baseline {baseline!r}, expected one of "
                f"{', '.join(BASELINE_COLUMNS)}")
    return baselines


def repo_commit():
    """
    Current commit, for the CSV provenance header ('unknown' if git is not
    available). Marked '-dirty' when the working tree has uncommitted
    changes, since the commit alone would not then identify the code that
    produced the numbers.
    """
    def git(*command):
        return subprocess.run(['git', *command], capture_output=True,
                              text=True, check=True).stdout

    try:
        commit = git('rev-parse', 'HEAD').strip()
        if git('status', '--porcelain').strip():
            commit += '-dirty'
        return commit
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def provenance_lines(args, solvers, baselines, rows=()):
    """
    Comment header for the CSV. network_interdiction_plot.py already reads
    the CSV with comment='#', so these lines are stripped on load.
    """
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds')
    lines = [
        "Generated by run_benchmark.py -- see README, "
        '"Regenerating the results CSV".',
        f"command: {' '.join(sys.argv)}",
        f"repo commit: {repo_commit()}",
        f"generated (UTC): {generated}",
        f"exact backend: {args.exact}",
        f"solver passes: {', '.join(solvers) if solvers else 'none'}",
        f"baselines: {', '.join(baselines) if baselines else 'none'}"
        + (f" (seed {args.seed})" if 'random' in baselines else ''),
    ]
    if 'sa' in solvers:
        lines.append(f"simulated annealing seed: {args.seed}")
    if 'qci' in solvers:
        lines.append(f"QCI Dirac-3 binary integer mode; samples: {args.qci_samples}; "
                     f"relaxation schedule: {args.qci_relaxation_schedule}; "
                     f"worker timeout: {args.qci_timeout} seconds; no automatic job resubmission")
        lines.append('QCI_time_sec includes worker startup/authentication/upload/queue/results; '
                     'QCI_device_usage_sec is rounded billable usage, not precise QPU time')
    if 'qaoa' in solvers:
        if args.qaoa_engine == 'ibm':
            lines.append(f'QAOA: IBM Runtime fixed angles, backend={args.ibm_backend}, '
                         f'p={args.qaoa_reps}, shots={args.qaoa_shots}, '
                         f'max_qubits={args.qaoa_max_qubits}, transpiler seed={args.seed}; '
                         'no angle optimization; hardware submission')
            lines.append('QAOA_time_sec includes transpilation/submission/queue/results; '
                         'QAOA_metadata records job ID, routed resources and service usage '
                         '(accounting, not circuit duration); no automatic resubmission')
        else:
            lines.append(f'QAOA: Qiskit Aer statevector, p={args.qaoa_reps}, '
                         f'shots={args.qaoa_shots}, COBYLA maxiter={args.qaoa_maxiter}, '
                         f'max_qubits={args.qaoa_max_qubits}, seed={args.seed}; no hardware')
            lines.append('QAOA_time_sec includes circuit building/training/shots; '
                         'QAOA_metadata records stage times, angles and logical circuit resources')
    if solvers:
        lines.append('raw feasibility counts budget-feasible SampleSet records among '
                     'the <=20 lowest-energy raw records, not occurrence-weighted reads; '
                     'the examined denominator and record diagnostics are retained')
        lines.append('pre_greedy diagnostics describe the repair input pool; '
                     'QA includes classical steepest descent, SA/QCI/QAOA equal raw')
    if args.exact == 'gurobi':
        lines.append(f"Gurobi trajectory solve threads: {args.gurobi_threads}")
    if 'lp' in baselines:
        import scipy
        lines.append(f"LP: scipy {scipy.__version__}, HiGHS continuous relaxation; "
                     "round x >= 0.5 then greedy repair; exact-rational dual bound")
    if 'lagrangian' in baselines:
        lines.append(f"Lagrangian: binary Gurobi solves, "
                     f"{ni.DEFAULT_GUROBI_THREADS} threads, lambda interval [0,maxflow]; "
                     "best global solver bound, subject to solver tolerances")
    if 'benders' in baselines:
        lines.append(f"Benders: binary Gurobi master, {ni.DEFAULT_GUROBI_THREADS} "
                     f"threads, NetworkX integer max-flow cuts, "
                     f"{args.benders_time_limit:g}s cooperative wall limit, "
                     f"{args.benders_max_iterations} master iterations; "
                     "master bounds subject to solver tolerances")
    lines.append("*_total_time_sec includes one method's construction, solve and "
                 "scoring; Reference_time_sec separately covers exact references "
                 "and any auto-lambda calibration; legacy Gurobi_time_sec is solve-only")
    if any(name in baselines for name in ('lp', 'lagrangian', 'benders')):
        lines.append("*_gap = (feasible surviving flow - Lower_bound) / feasible "
                     "surviving flow; zero flow has gap 0; unrun/infeasible results are empty")
    if any(case[2] == 'auto' for case in args.cases):
        lines.append(
            "lambda = auto: the midpoint of the exact window "
            "(lambda_lo, lambda_hi); see the run_benchmark.py docstring.")
    empty = [f"{row['instance']} B={row['B']} ("
             + ('tied budgets' if row['lambda_lo'] == row['lambda_hi'] else
                'B above the lower convex envelope') + ')'
             for row in rows
             if row['lambda_lo'] != '' and row['lambda'] == '']
    if empty:
        lines.append(
            f"empty strict lambda window, baselines only: "
            f"{'; '.join(empty)}")
    missing = [s for s in SOLVER_COLUMNS if s not in solvers]
    if missing:
        lines.append(
            f"{'/'.join(sorted(missing))} columns are empty: that pass was "
            f"not run here. Re-run with --solvers to fill them.")
    return lines


def write_csv(path, rows, header_lines):
    with open(path, 'w', newline='') as handle:
        for line in header_lines:
            handle.write(f'# {line}\n')
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--case', dest='cases', action='append', required=True,
        type=parse_case, metavar='PATH:BUDGET:LAMBDA',
        help='instance file, interdiction budget and penalty coefficient; '
             'repeat for one CSV row per case')
    parser.add_argument(
        '--exact', choices=['gurobi', 'brute'], default='gurobi',
        help='backend for the Opt_NIP/Opt_QUBO columns (default: gurobi)')
    parser.add_argument(
        '--solvers', type=parse_solvers, default='sa', metavar='sa[,qa,qci,qaoa]',
        help="comma-separated passes from 'sa', 'qa', 'qci', 'qaoa', or 'none' "
             '(default: sa)')
    parser.add_argument('--qci-python', help='Python executable in the separate QCI '
                        'environment (default: QCI_PYTHON environment variable)')
    parser.add_argument('--qci-samples', type=int, default=20,
                        help='Dirac-3 samples, 1..100 (default: 20)')
    parser.add_argument('--qci-relaxation-schedule', type=int, choices=[1, 2, 3, 4],
                        default=1, help='Dirac-3 relaxation schedule (default: 1)')
    parser.add_argument('--qci-timeout', type=float, default=300,
                        help='worker wall-time limit in seconds; does not cancel a remote job')
    parser.add_argument('--qaoa-reps', type=int, default=1, help='QAOA layers (default: 1)')
    parser.add_argument('--qaoa-shots', type=int, default=2500, help='Aer shots (default: 2500)')
    parser.add_argument('--qaoa-maxiter', type=int, default=100,
                        help='COBYLA evaluation limit; at least 2*reps+2 (default: 100)')
    parser.add_argument('--qaoa-max-qubits', type=int, default=24,
                        help='variable limit; Aer permits at most 24 (default: 24)')
    parser.add_argument('--qaoa-engine', choices=['aer', 'ibm'], default='aer',
                        help='Aer training+sampling, or IBM Runtime fixed-angle sampling')
    parser.add_argument('--ibm-backend', help='explicit device name from a saved IBM Runtime account')
    parser.add_argument('--ibm-theta', type=float, nargs='+',
                        help='2*p fixed angles in beta-then-gamma order, same normalized Hamiltonian')
    parser.add_argument('--ibm-max-depth', type=int, default=10000)
    parser.add_argument('--ibm-timeout', type=float, default=300,
                        help='cooperative result-polling timeout; does not cancel remote jobs')
    parser.add_argument('--ibm-max-execution-time', type=int, default=300,
                        help='service execution cap in seconds, excluding queue time')
    parser.add_argument(
        '--baselines', type=parse_baselines, default='greedy,random',
        metavar='greedy,random,lp,lagrangian,benders',
        help="comma-separated classical baselines: greedy, random, lp, "
             "lagrangian, benders (last two need Gurobi), or "
             "'none' (default: greedy,random)")
    parser.add_argument('--benders-time-limit', type=float, default=60.0,
                        help='cooperative wall-clock seconds per Benders pass (default 60)')
    parser.add_argument('--benders-max-iterations', type=int, default=1000,
                        help='maximum master solves per Benders pass (default 1000)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='seed for simulated annealing, QAOA and the random baseline '
             '(default: 0)')
    parser.add_argument(
        '--gurobi-threads', type=int, default=ni.DEFAULT_GUROBI_THREADS,
        help="Threads for the timed Gurobi trajectory solve (default: 8; "
             "0 = Gurobi's automatic setting). Exact-reference solves use 8.")
    parser.add_argument(
        '--max-exact-vars', type=int, default=DEFAULT_MAX_EXACT_VARS,
        help='refuse --exact brute past this many QUBO variables '
             f'(default: {DEFAULT_MAX_EXACT_VARS})')
    parser.add_argument(
        '--out', required=True, metavar='PATH', help='CSV to write')
    args = parser.parse_args(argv)

    solvers = args.solvers if isinstance(args.solvers, list) else \
        parse_solvers(args.solvers)
    if 'qci' in solvers and (not 1 <= args.qci_samples <= 100
                            or not math.isfinite(args.qci_timeout) or args.qci_timeout <= 0):
        parser.error('QCI samples must be in 1..100 and timeout must be finite and positive')
    qci_options = dict(python_executable=args.qci_python, num_samples=args.qci_samples,
                       relaxation_schedule=args.qci_relaxation_schedule,
                       timeout_sec=args.qci_timeout)
    qaoa_options = dict(reps=args.qaoa_reps, shots=args.qaoa_shots,
                        maxiter=args.qaoa_maxiter, max_qubits=args.qaoa_max_qubits)
    if 'qaoa' in solvers:
        try:
            if args.qaoa_engine == 'ibm':
                from qaoa_ibm import validate_options as validate_ibm_options
                if not args.ibm_backend:
                    raise ValueError('--qaoa-engine ibm requires --ibm-backend')
                qaoa_options = dict(theta=args.ibm_theta, reps=args.qaoa_reps,
                    shots=args.qaoa_shots, max_qubits=args.qaoa_max_qubits,
                    max_depth=args.ibm_max_depth, result_timeout_sec=args.ibm_timeout,
                    max_execution_time_sec=args.ibm_max_execution_time)
                validate_ibm_options(seed_transpiler=args.seed, **qaoa_options)
            else:
                from solve_qubo_via_qaoa import validate_options
                validate_options(seed=args.seed, **qaoa_options)
        except ValueError as error:
            parser.error(str(error))
        if args.qaoa_engine == 'ibm':
            from qiskit_ibm_runtime import QiskitRuntimeService
            qaoa_options.update(engine='ibm', backend=QiskitRuntimeService().backend(args.ibm_backend))
    baselines = args.baselines if isinstance(args.baselines, list) else \
        parse_baselines(args.baselines)
    if 'benders' in baselines:
        if not math.isfinite(args.benders_time_limit) or args.benders_time_limit <= 0:
            parser.error('--benders-time-limit must be positive and finite')
        if args.benders_max_iterations < 1:
            parser.error('--benders-max-iterations must be positive')
    # One generator for the whole run, so a CSV is reproducible from its
    # command line and seed alone.
    rng = np.random.default_rng(args.seed)

    rows = []
    for instance, B, curr_λ in args.cases:
        print(f'=== {instance} B={B} lambda={curr_λ} ===')
        rows.append(run_case(instance, B, curr_λ, args.exact, solvers,
                             args.max_exact_vars, baselines, rng,
                             sa_seed=args.seed,
                             gurobi_threads=args.gurobi_threads,
                             benders_time_limit=args.benders_time_limit,
                             benders_max_iterations=args.benders_max_iterations,
                             qci_options=qci_options, qaoa_options=qaoa_options))

    write_csv(args.out, rows,
              provenance_lines(args, solvers, baselines, rows))
    print(f'Wrote {len(rows)} row(s) to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

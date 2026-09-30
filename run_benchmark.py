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
maxflow         max flow of the un-interdicted instance
Opt_NIP         optimal surviving flow under the budget (exact)
Opt_QUBO        exact ground-state energy of the QUBO at that lambda
SA_NIP          best surviving flow over the annealer's 20 best samples that
                already respect the budget
SA_NIP_Greedy   same, after improve_dwave_solution() repairs each sample to
                exactly B interdicted edges
SA_QUBO         lowest QUBO energy the annealer reached
QA_*            the same three columns for the D-Wave QPU path

`instance` and `lambda` are additions to what the plot reads; pandas ignores
them.

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

Examples
--------
Regenerate the checked-in sample (no license or token needed):

    python run_benchmark.py --exact brute --solvers sa \\
        --case testcases/tiny_a.flow:1:6 \\
        --out results/sample_results_sa.csv

A real run over a planar instance (needs Gurobi, and a Leap token for qa):

    python run_benchmark.py --exact gurobi --solvers sa,qa \\
        --case testcases/50.planarflow:12:36 \\
        --out result_dwave_planar.csv
"""
import argparse
import csv
import datetime
import itertools
import math
import subprocess
import sys

import dimod
import networkx as nx

import network_interdiction as ni

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
CSV_COLUMNS = ['instance'] + PLOT_COLUMNS[:3] + ['lambda'] + PLOT_COLUMNS[3:]

SOLVER_COLUMNS = {
    'sa': ('SA_NIP', 'SA_NIP_Greedy', 'SA_QUBO'),
    'qa': ('QA_NIP(advantage1)', 'QA_NIP(advantage1)_Greedy',
           'QA_QUBO(advantage1)'),
}

DEFAULT_MAX_EXACT_VARS = 24


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


def run_case(instance, B, curr_λ, exact_backend, solvers,
             max_exact_vars=DEFAULT_MAX_EXACT_VARS):
    """Compute one CSV row. `instance` is a path to a flow file."""
    di_g, src_node, tgt_node = ni.read_flow_file(instance)
    opt_nip, opt_qubo = exact_reference(
        exact_backend, di_g, src_node, tgt_node, B, curr_λ, max_exact_vars)

    row = {column: '' for column in CSV_COLUMNS}
    row.update({
        'instance': instance,
        'V': len(di_g.nodes()),
        'E': len(di_g.edges()),
        'B': B,
        'lambda': curr_λ,
        'maxflow': max_flow(di_g, src_node, tgt_node),
        'Opt_NIP': opt_nip,
        'Opt_QUBO': opt_qubo,
    })

    for solver in solvers:
        result = ni.qubo_verify_parameters(
            str(instance), di_g, curr_λ, src_node, tgt_node, B, solver=solver)
        nip_col, greedy_col, qubo_col = SOLVER_COLUMNS[solver]
        before = result['best_obj_before_greedy']
        # inf means not one of the 20 best samples stayed inside the budget.
        # Leave it blank: the plot reads a 0 in this column as "this solver
        # was not run" (it special-cases 0 to a zero-height bar), so writing
        # a sentinel number here would be read as data.
        row[nip_col] = '' if math.isinf(before) else before
        row[greedy_col] = result['best_obj_after_greedy']
        row[qubo_col] = result['best_energy']

    return row


def parse_case(spec):
    """Parse a --case argument of the form PATH:BUDGET:LAMBDA."""
    parts = spec.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected PATH:BUDGET:LAMBDA, got {spec!r}")
    path, budget, lam = parts
    try:
        return path, int(budget), float(lam)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"budget must be an int and lambda a float, got {spec!r}")


def parse_solvers(spec):
    if spec == 'none':
        return []
    solvers = [s.strip() for s in spec.split(',') if s.strip()]
    for solver in solvers:
        if solver not in SOLVER_COLUMNS:
            raise argparse.ArgumentTypeError(
                f"unknown solver {solver!r}, expected sa and/or qa")
    return solvers


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


def provenance_lines(args, solvers):
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
    ]
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
        '--solvers', type=parse_solvers, default='sa', metavar='sa[,qa]',
        help="annealing passes to run: 'sa', 'qa', 'sa,qa' or 'none' "
             '(default: sa)')
    parser.add_argument(
        '--max-exact-vars', type=int, default=DEFAULT_MAX_EXACT_VARS,
        help='refuse --exact brute past this many QUBO variables '
             f'(default: {DEFAULT_MAX_EXACT_VARS})')
    parser.add_argument(
        '--out', required=True, metavar='PATH', help='CSV to write')
    args = parser.parse_args(argv)

    solvers = args.solvers if isinstance(args.solvers, list) else \
        parse_solvers(args.solvers)

    rows = []
    for instance, B, curr_λ in args.cases:
        print(f'=== {instance} B={B} lambda={curr_λ} ===')
        rows.append(run_case(instance, B, curr_λ, args.exact, solvers,
                             args.max_exact_vars))

    write_csv(args.out, rows, provenance_lines(args, solvers))
    print(f'Wrote {len(rows)} row(s) to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

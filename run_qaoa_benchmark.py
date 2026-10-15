"""
Run the penalty-QAOA over a list of cases and depths and write one CSV row
per (case, depth), for the QAOA section of notebooks/analysis.ipynb (issues
#3, #16). Each row is qaoa_common.run_qaoa() on one instance and budget, with
lambda chosen from its exact window.

Columns
-------
instance, V, E, B, lambda   as in run_benchmark.py's CSV
qubits                      QUBO variables, one qubit each (V - 2 + 3E)
backend, reps               simulator and QAOA depth p
theta                       trained angles as JSON, [beta..., gamma...], for
                            the Hamiltonian scaled to |coefficients| <= 1
evaluations, train_time_sec COBYLA's expectation evaluations and wall time
expected_energy ...         the QaoaResult fields of the same names (see
  ... greedy                qaoa_common); flow_before_repair is empty when no
                            shot among the 20 lowest-energy ones respected
                            the budget
transfer_approx_ratio,      exact scores of the first case's trained angles,
transfer_p_optimal          at the same depth, applied unchanged to this case:
                            the parameter-transfer baseline for #16. Empty on
                            the first case's own rows.

Regenerating the committed CSV (needs requirements-qaoa.txt; about 15
minutes on a laptop CPU, almost all of it the 21-22 qubit cases):

    python run_qaoa_benchmark.py --reps 1,2,3 \\
        --case testcases/tiny_a.flow:1 --case testcases/tiny_b.flow:1 \\
        --case testcases/tiny_b.flow:2 --case testcases/tiny_c.flow:1 \\
        --case testcases/tiny_d.flow:1 --case testcases/tiny_d.flow:2 \\
        --out results/qaoa_tiny.csv

These are the cases of results/sample_results_sa.csv, so the notebook can
put QAOA next to simulated annealing and greedy on the same rows.
"""
import argparse
import csv
import datetime
import json
import math
import sys

import qaoa_common as qc
import run_benchmark as rb

CSV_COLUMNS = [
    'instance', 'V', 'E', 'B', 'lambda', 'qubits', 'backend', 'reps',
    'theta', 'evaluations', 'train_time_sec',
    'expected_energy', 'uniform_expected_energy', 'opt_qubo', 'approx_ratio',
    'p_ground', 'p_optimal', 'uniform_p_optimal',
    'shots', 'shot_p_optimal', 'best_sample_energy',
    'flow_before_repair', 'flow_after_repair', 'opt_nip', 'greedy',
    'transfer_approx_ratio', 'transfer_p_optimal',
]


def make_backend(name, problem, reps):
    """A QiskitBackend or CudaqBackend; imported here so each is optional."""
    if name == 'qiskit':
        import qaoa_qiskit
        return qaoa_qiskit.QiskitBackend(problem, reps)
    if name == 'cudaq':
        import qaoa_cudaq
        return qaoa_cudaq.CudaqBackend(problem, reps)
    raise ValueError(f'unknown backend: {name}')


def backend_versions(name):
    """The installed versions of the backend's framework, for provenance."""
    if name == 'qiskit':
        import qiskit
        import qiskit_aer
        return f'qiskit {qiskit.__version__}, qiskit-aer {qiskit_aer.__version__}'
    import cudaq
    return cudaq.__version__.split(' (')[0]


def result_row(problem, result):
    """One CSV row from a QaoaResult, with the transfer columns empty."""
    row = {column: '' for column in CSV_COLUMNS}
    row.update({
        'instance': problem.instance,
        'V': len(problem.di_g.nodes()),
        'E': len(problem.di_g.edges()),
        'qubits': problem.num_qubits,
        'lambda': problem.lam,
        'theta': json.dumps([round(t, 6) for t in result.theta]),
    })
    for column in CSV_COLUMNS:
        if row[column] == '' and hasattr(result, column):
            row[column] = rounded(getattr(result, column))
    if math.isinf(result.flow_before_repair):
        row['flow_before_repair'] = ''
    return row


def rounded(value):
    """Floats to 6 decimals for the CSV; far below anything it reports."""
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 6)
    return value


def run_cases(cases, depths, backend_name, shots, seed, maxiter):
    """
    Rows for every depth in `depths` and case (instance path, B) in `cases`.
    The first case is the transfer donor: its angles at each depth are
    scored, unchanged, on every other case at that depth.
    """
    problems = [qc.build_problem(instance, B) for instance, B in cases]
    rows = []
    for reps in depths:
        donor_theta = None
        for problem in problems:
            backend = make_backend(backend_name, problem, reps)
            result = qc.run_qaoa(backend, problem, reps, shots=shots,
                                 seed=seed, maxiter=maxiter)
            row = result_row(problem, result)
            if donor_theta is None:
                donor_theta = result.theta
            else:
                transfer = qc.score_probabilities(
                    problem, backend.probabilities(donor_theta))
                row['transfer_approx_ratio'] = rounded(transfer['approx_ratio'])
                row['transfer_p_optimal'] = rounded(transfer['p_optimal'])
            rows.append(row)
            print(f"{problem.instance} B={problem.B} p={reps}: approx ratio "
                  f"{result.approx_ratio:.4f}, P(optimal) "
                  f"{result.p_optimal:.4f} ({result.train_time_sec:.1f} s)",
                  flush=True)
    return rows


def parse_case(spec):
    """'path:B' -> (path, B)."""
    instance, _, budget = spec.rpartition(':')
    if not instance:
        raise argparse.ArgumentTypeError(f"expected instance:B, got {spec!r}")
    return instance, int(budget)


def provenance_lines(args, commit):
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds')
    donor = f'{args.cases[0][0]} B={args.cases[0][1]}'
    return [
        'Generated by run_qaoa_benchmark.py -- see its docstring.',
        f"command: {' '.join(sys.argv)}",
        f'repo commit: {commit}',
        f'generated (UTC): {generated}',
        f'backend: {args.backend} ({backend_versions(args.backend)})',
        f'depths: {", ".join(map(str, args.reps))}; shots: {args.shots}; '
        f'seed: {args.seed}; COBYLA maxiter: {args.maxiter}, from the '
        f'linear-ramp start only',
        'lambda = auto: the midpoint of the exact window, as in '
        'run_benchmark.py.',
        f'transfer donor: {donor} (its angles at each depth, unchanged)',
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--case', dest='cases', action='append',
                        type=parse_case, required=True,
                        help='instance:B; the first case is the transfer donor')
    parser.add_argument('--reps', type=lambda s: [int(p) for p in s.split(',')],
                        default=[1, 2, 3], help='comma-separated depths')
    parser.add_argument('--backend', choices=('qiskit', 'cudaq'),
                        default='qiskit')
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    # Read before anything is written: once --out exists inside the
    # checkout, it alone would mark the tree dirty.
    commit = rb.repo_commit()
    rows = run_cases(args.cases, args.reps, args.backend, args.shots,
                     args.seed, args.maxiter)
    with open(args.out, 'w', newline='') as handle:
        for line in provenance_lines(args, commit):
            handle.write(f'# {line}\n')
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())

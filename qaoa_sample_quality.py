"""
Expected best interdiction gap after N independent QAOA measurements.

Re-simulates the saved angles in results/qaoa_tiny.csv; no training or
greedy repair. A within-budget plan is scored by f/f* - 1, with f its
surviving max flow and f* the exact optimum. An over-budget plan receives
the empty plan's score (maxflow/f* - 1), so extra removals cannot improve
the score. The uniform initial state is the random-guessing baseline.

For sorted distinct gaps g_0 < ... < g_k and a single-shot gap G,

    E[min(G_1, ..., G_N)]
        = g_0 + sum_j (g_j - g_{j-1}) P(G >= g_j)**N.

This is an exact expectation under each ideal state, not a sampled best
result or a runtime comparison. The notebook reads the generated CSV and
averages per-case expectations, giving each of the six cases equal weight.

    python qaoa_sample_quality.py --out results/qaoa_sample_quality.csv

Needs requirements-qaoa.txt only to regenerate the data. The scoring
functions and the notebook do not need Qiskit or a Gurobi license.
"""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import qaoa_common as qc
import qaoa_outcomes as qo
import run_benchmark as rb

REPO = Path(__file__).resolve().parent
MEASUREMENTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4096)
FIELDS = ['case', 'instance', 'B', 'method', 'reps', 'qubits', 'measurements',
          'expected_best_gap', 'opt_nip', 'maxflow']


def plan_gaps(problem):
    """Gap of each x pattern; over-budget patterns score as the empty plan."""
    if problem.opt_nip <= 0:
        raise ValueError('relative gaps require a positive optimum')
    edges = problem.interdictable_edges
    flows = np.full(2 ** len(edges), float(problem.maxflow))
    for pattern in range(flows.size):
        if pattern.bit_count() <= problem.B:
            cut = [edge for e, edge in enumerate(edges) if (pattern >> e) & 1]
            flows[pattern] = rb.max_flow(problem.di_g, problem.src_node,
                                          problem.tgt_node, cut)
    return flows / problem.opt_nip - 1


def expected_best_gap(gaps, probabilities, measurements):
    """Expected minimum gap for each positive integer measurement count."""
    gaps = np.asarray(gaps, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    measurements = np.asarray(measurements)
    if (gaps.ndim != 1 or not gaps.size
            or probabilities.shape != gaps.shape
            or not np.isfinite(gaps).all() or (gaps < 0).any()):
        raise ValueError('gaps must be finite, nonnegative and match probabilities')
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-9)):
        raise ValueError('probabilities must be nonnegative and sum to one')
    if (measurements.ndim != 1 or not measurements.size
            or not np.issubdtype(measurements.dtype, np.integer)
            or (measurements < 1).any()):
        raise ValueError('measurement counts must be positive integers')
    levels, inverse = np.unique(gaps, return_inverse=True)
    weights = np.bincount(inverse, weights=probabilities / probabilities.sum(),
                          minlength=levels.size)
    # Upper tails avoid subtracting nearly equal powers when N is large.
    tails = np.clip(np.cumsum(weights[::-1])[::-1][1:], 0, 1)
    return levels[0] + (tails[:, None] ** measurements).T @ np.diff(levels)


def tiny_rows():
    """Score each stored tiny state, plus uniform guessing, at every N."""
    from qaoa_qiskit import QiskitBackend
    stored = pd.read_csv(qo.TINY_CSV, comment='#')
    rows = []
    for (instance, B), group in stored.groupby(['instance', 'B'], sort=False):
        problem = qc.build_problem(REPO / instance, int(B))
        if (not np.allclose(group['lambda'], problem.lam, rtol=0,
                            atol=qo.LAMBDA_TOL)
                or not np.allclose(group['opt_nip'], problem.opt_nip, rtol=0,
                                   atol=qo.LAMBDA_TOL)):
            raise ValueError(f'{instance} B={B}: rebuilt reference differs '
                             'from the stored QAOA problem')
        gaps = plan_gaps(problem)
        states = [(0, np.full(gaps.size, 1 / gaps.size))]
        for row in group.sort_values('reps').itertuples():
            probabilities = QiskitBackend(problem, int(row.reps)).probabilities(
                json.loads(row.theta))
            states.append((int(row.reps), qo.plan_probabilities(
                problem, probabilities)))
        for reps, probabilities in states:
            expected = expected_best_gap(gaps, probabilities, MEASUREMENTS)
            for count, gap in zip(MEASUREMENTS, expected):
                rows.append({
                    'case': f'{Path(instance).stem} B={B}', 'instance': instance,
                    'B': int(B), 'method': 'qaoa' if reps else 'uniform',
                    'reps': reps, 'qubits': problem.num_qubits,
                    'measurements': count, 'expected_best_gap': float(gap),
                    'opt_nip': problem.opt_nip, 'maxflow': problem.maxflow,
                })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True,
                        help='output CSV; refuses to overwrite')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    commit = rb.repo_commit()
    command = ' '.join(sys.argv[1:] if argv is None else argv)
    rows = tiny_rows()
    sources = [Path(__file__), qo.TINY_CSV,
               *[REPO / name for name in (
                   'qaoa_outcomes.py', 'qaoa_common.py', 'qaoa_qiskit.py',
                   'run_benchmark.py', 'network_interdiction.py')],
               *[REPO / name for name in sorted({r['instance'] for r in rows})]]
    with args.out.open('x', newline='') as stream:
        stream.write('# Generated by qaoa_sample_quality.py -- see its docstring.\n')
        stream.write(f'# command: qaoa_sample_quality.py {command}\n')
        stream.write(f'# repo commit: {commit}\n')
        stream.write('# exact statevector; stored angles; no training or repair\n')
        stream.write('# gap: f/f* - 1; over budget: maxflow/f* - 1\n')
        for package in ('numpy', 'qiskit', 'qiskit-aer'):
            stream.write(f'# {package}: {version(package)}\n')
        for path in sources:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stream.write(f'# sha256 {path.relative_to(REPO)}: {digest}\n')
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()

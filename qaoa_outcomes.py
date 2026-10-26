"""
Where QAOA's probability goes, over interdiction plans.

results/qaoa_tiny.csv and results/parameter_transfer.csv record P(optimal)
and energy scores for every trained angle set, but not what the rest of the
probability does. This script re-simulates each stored state exactly
(Qiskit Aer statevector) and splits its probability by the x bits, i.e. by
the interdiction plan a measurement proposes:

    optimal       at most B arcs cut, and the optimal flow survives
    suboptimal    1..B arcs cut, but more flow survives than the optimum
    empty         no arc cut: the full max flow survives
    over_budget   more than B arcs cut

The four columns sum to one. Depth-0 rows are the circuits' initial states:
the uniform superposition (method 'uniform') and, for the transfer cohort,
the regularized warm-start product state (method 'rws'). Each depth-p row
carries the stored P(optimal) it must reproduce (`stored_p_optimal`);
tests/test_qaoa_outcomes.py checks that it does.

    python qaoa_outcomes.py --out results/qaoa_outcomes.csv

Simulation needs qiskit and qiskit-aer (requirements-qaoa.txt), imported
only when needed, so plan_outcomes() and outcome_probabilities() work
without them. The transfer problems
are rebuilt with run_parameter_transfer.make_dataset() from the arguments
stored in results/parameter_transfer.json, and must match its graph records.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import qaoa_common as qc
import run_benchmark as rb

REPO = Path(__file__).resolve().parent
TINY_CSV = REPO / 'results' / 'qaoa_tiny.csv'
TRANSFER_CSV = REPO / 'results' / 'parameter_transfer.csv'
TRANSFER_JSON = REPO / 'results' / 'parameter_transfer.json'
OUTCOMES = ('optimal', 'suboptimal', 'empty', 'over_budget')
FIELDS = ['source', 'case', 'instance', 'B', 'method', 'reps', 'qubits',
          'x_bits', *[f'p_{name}' for name in OUTCOMES], 'stored_p_optimal']
LAMBDA_TOL = 1e-9


def plan_outcomes(problem):
    """
    The outcome of every x pattern (bit e = arc e of interdictable_edges
    is cut), as an array of indices into OUTCOMES.
    """
    edges = problem.interdictable_edges
    labels = np.empty(2 ** len(edges), dtype=np.int64)
    for pattern in range(2 ** len(edges)):
        cut = [edges[e] for e in range(len(edges)) if (pattern >> e) & 1]
        if len(cut) > problem.B:
            labels[pattern] = OUTCOMES.index('over_budget')
        elif rb.max_flow(problem.di_g, problem.src_node, problem.tgt_node,
                         cut) == problem.opt_nip:
            labels[pattern] = OUTCOMES.index('optimal')
        elif not cut:
            labels[pattern] = OUTCOMES.index('empty')
        else:
            labels[pattern] = OUTCOMES.index('suboptimal')
    return labels


def plan_probabilities(problem, probabilities):
    """Marginalize a little-endian state over its interdiction (x) bits."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (2 ** problem.num_qubits,):
        raise ValueError(f'expected {2 ** problem.num_qubits} probabilities, '
                         f'got shape {probabilities.shape}')
    index = np.arange(probabilities.size, dtype=np.int64)
    pattern = np.zeros(index.shape, dtype=np.int64)
    for e, qubit in enumerate(problem.x_qubits):
        pattern |= ((index >> qubit) & 1) << e
    return np.bincount(pattern, weights=probabilities,
                       minlength=2 ** len(problem.x_qubits))


def outcome_probabilities(problem, probabilities, labels=None):
    """
    Split a little-endian probability vector over OUTCOMES by its x bits.
    Returns {'p_optimal': ..., 'p_suboptimal': ..., ...}.
    """
    if labels is None:
        labels = plan_outcomes(problem)
    per_plan = plan_probabilities(problem, probabilities)
    shares = np.bincount(labels, weights=per_plan, minlength=len(OUTCOMES))
    return {f'p_{name}': float(share) for name, share in zip(OUTCOMES, shares)}


def initial_state(problem, warm_start_angles=None):
    """Probabilities of the circuit's initial state (all angles zero)."""
    from qaoa_qiskit import QiskitBackend
    backend = QiskitBackend(problem, 1, warm_start_angles=warm_start_angles)
    return backend.probabilities(np.zeros(2))


def tiny_rows():
    """One row per stored qaoa_tiny.csv state, plus each case's uniform state."""
    from qaoa_qiskit import QiskitBackend
    stored = pd.read_csv(TINY_CSV, comment='#')
    rows = []
    for (instance, B), group in stored.groupby(['instance', 'B'], sort=False):
        problem = qc.build_problem(instance, int(B))
        lam = group['lambda'].iloc[0]
        if abs(problem.lam - lam) > LAMBDA_TOL:
            raise ValueError(f'{instance} B={B}: rebuilt lambda {problem.lam} '
                             f'differs from the stored {lam}')
        labels = plan_outcomes(problem)
        case = f'{Path(instance).stem} B={B}'
        base = {'source': 'tiny', 'case': case, 'instance': instance, 'B': B,
                'qubits': problem.num_qubits, 'x_bits': len(problem.x_qubits)}
        rows.append({**base, 'method': 'uniform', 'reps': 0,
                     **outcome_probabilities(problem, initial_state(problem),
                                             labels),
                     'stored_p_optimal': group['uniform_p_optimal'].iloc[0]})
        for _, row in group.sort_values('reps').iterrows():
            backend = QiskitBackend(problem, int(row['reps']))
            probabilities = backend.probabilities(json.loads(row['theta']))
            rows.append({**base, 'method': 'qaoa', 'reps': int(row['reps']),
                         **outcome_probabilities(problem, probabilities,
                                                 labels),
                         'stored_p_optimal': row['p_optimal']})
    return rows


def transfer_problems():
    """The held-out test problems, rebuilt and checked against the JSON."""
    import run_parameter_transfer as rpt
    record = json.loads(TRANSFER_JSON.read_text())
    args = record['arguments']
    problems, graphs = rpt.make_dataset(args['train_count'],
                                        args['test_count'],
                                        args['test_nodes'], args['seed'])
    if json.loads(json.dumps(graphs, default=float)) != record['graphs']:
        raise ValueError('rebuilt transfer graphs differ from '
                         f'{TRANSFER_JSON.name}')
    return {p.instance: p for p in problems['test']}, args['strength'], args['seed']


def transfer_rows():
    """One row per stored parameter_transfer.csv state, plus initial states."""
    from qaoa_qiskit import QiskitBackend
    from qaoa_warm_start import regularized_warm_start
    stored = pd.read_csv(TRANSFER_CSV, comment='#')
    problems, strength, seed = transfer_problems()
    rows = []
    for group, rows_for_group in stored.groupby('group', sort=False):
        problem = problems[group]
        lam = rows_for_group['lambda'].iloc[0]
        if abs(problem.lam - lam) > LAMBDA_TOL:
            raise ValueError(f'{group}: rebuilt lambda {problem.lam} differs '
                             f'from the stored {lam}')
        labels = plan_outcomes(problem)
        warm = regularized_warm_start(problem, strength, seed)
        base = {'source': 'transfer', 'case': group, 'instance': group,
                'B': problem.B, 'qubits': problem.num_qubits,
                'x_bits': len(problem.x_qubits)}
        for method, angles in (('uniform', None), ('rws', warm)):
            rows.append({**base, 'method': method, 'reps': 0,
                         **outcome_probabilities(
                             problem, initial_state(problem, angles), labels),
                         'stored_p_optimal': ''})
        for _, row in rows_for_group.iterrows():
            backend = QiskitBackend(
                problem, int(row['reps']),
                warm_start_angles=warm if row['method'] == 'rws' else None)
            probabilities = backend.probabilities(json.loads(row['theta']))
            rows.append({**base, 'method': row['method'],
                         'reps': int(row['reps']),
                         **outcome_probabilities(problem, probabilities,
                                                 labels),
                         'stored_p_optimal': row['p_optimal']})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True,
                        help='output CSV; refuses to overwrite')
    args = parser.parse_args(argv)
    # Read the commit before creating the output, which would mark it dirty.
    commit = rb.repo_commit()
    command = ' '.join(sys.argv[1:] if argv is None else argv)
    rows = tiny_rows() + transfer_rows()
    with args.out.open('x', newline='') as stream:
        stream.write('# Generated by qaoa_outcomes.py -- see its docstring.\n')
        stream.write(f'# command: qaoa_outcomes.py {command}\n')
        stream.write(f'# repo commit: {commit}\n')
        stream.write(f'# inputs: {TINY_CSV.name}, {TRANSFER_CSV.name}, '
                     f'{TRANSFER_JSON.name} (exact statevector)\n')
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()

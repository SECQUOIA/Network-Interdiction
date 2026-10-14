"""
Measure the decomposition (issue #14) against the exact optimum on the
instances and budgets behind the D-Wave slides: the decomposition-versus-
classical comparison of #14 and the proposal's Q3-4 milestone.

Cases are the rows of results/slide_instances_sa.csv, which already hold
Gurobi's optimum and greedy for every (instance, B). Each case is
partitioned two ways:

    pieces4   4 pieces (partition_graph()), the setting of #21's table
    fit       the fewest pieces whose QUBOs fit --fit-qubits qubits (156 by
              default, the size of an IBM Heron device; partition_to_fit()),
              labelled fit156 in the CSV

and every budget rule (mincut, local, marginal) is solved in every mode
(independent, sequential, iterated). Each piece is solved exactly
(gurobi_piece_solver()), so the only error left is the decomposition's: the
split, and how the pieces' cuts combine. With --refine, a 'marginal+refine'
row adds refine_split() from the marginal split in iterated mode, for the
4-piece setting only, since it re-solves the decomposition once per ordered
pair of pieces per move.

Columns
-------
instance, B, setting      the case and the partition setting
pieces, max_piece_qubits  piece count and the largest piece QUBO
rule, mode                budget rule and solve mode
budgets                   the split, as JSON, one entry per piece
flow                      max flow left by the combined cuts
ratio                     (maxflow - flow) / (maxflow - optimum): 1 is
                          optimal, as in the rest of the repository
optimum, greedy, maxflow  from the slide CSV
time_sec                  wall time of the split and the solve; the
                          partition is shared and timed separately
partition_sec             wall time of that partition

When no partition fits -- a piece whose surroundings cannot be contracted
keeps most of the graph's nodes, so on 125.planarflow no number of pieces
gets under 156 qubits -- each case gets one row with rule 'no fit' and the
rest of the run carries on.

Rows are appended as they finish, so a long run keeps what it has done.
Needs a Gurobi license.

    python run_decomposition_benchmark.py --refine \\
        --out results/decomposition_benchmark.csv
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time

import pandas as pd

import decomposition as dc
import network_interdiction as ni
import run_benchmark as rb

SLIDE_CSV = os.path.join('results', 'slide_instances_sa.csv')
SETTINGS = ('pieces4', 'fit')
IBM_QUBITS = 156
COLUMNS = ['instance', 'B', 'setting', 'pieces', 'max_piece_qubits', 'rule',
           'mode', 'budgets', 'flow', 'ratio', 'optimum', 'greedy', 'maxflow',
           'time_sec', 'partition_sec']


def partition(setting, di_g, src, tgt, fit_qubits=IBM_QUBITS, seed=0):
    """
    (pieces, largest piece QUBO) for a setting. Raises ValueError from
    partition_to_fit() when no partition fits `fit_qubits`.
    """
    if setting == 'pieces4':
        pieces = dc.partition_graph(di_g, 4, seed=seed)
    elif setting == 'fit':
        pieces, _ = dc.partition_to_fit(di_g, src, tgt, fit_qubits, seed=seed)
    else:
        raise ValueError(f'unknown setting {setting!r}')
    sizes = [dc.piece_qubo_size(dc.piece_subproblem(di_g, src, tgt, pieces, i))
             for i in range(len(pieces))]
    return pieces, max(sizes)


def case_rows(di_g, src, tgt, B, pieces, refine):
    """(rule, mode, budgets, flow, seconds) for every variant of one case."""
    solve = dc.gurobi_piece_solver
    rows = []
    for rule in dc.SPLIT_STRATEGIES:
        start = time.perf_counter()
        budgets = dc.split_budget(di_g, src, tgt, B, pieces, rule, solve)
        split_time = time.perf_counter() - start
        for mode in dc.SOLVE_MODES:
            start = time.perf_counter()
            _, flow = dc.solve_decomposed(di_g, src, tgt, pieces, budgets,
                                          solve, mode)
            rows.append((rule, mode, budgets, flow,
                         split_time + time.perf_counter() - start))
        if rule == 'marginal' and refine:
            start = time.perf_counter()
            refined, _, flow = dc.refine_split(di_g, src, tgt, pieces, budgets,
                                               solve, mode='iterated')
            rows.append(('marginal+refine', 'iterated', refined, flow,
                         split_time + time.perf_counter() - start))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cases', default=SLIDE_CSV,
                        help='CSV with instance, B, Opt_NIP, Greedy, maxflow')
    parser.add_argument('--settings', default=','.join(SETTINGS))
    parser.add_argument('--refine', action='store_true',
                        help='add refine_split() rows (4-piece setting)')
    parser.add_argument('--fit-qubits', type=int, default=IBM_QUBITS,
                        help='qubit limit of the fit setting (default 156)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    settings = args.settings.split(',')

    commit = rb.repo_commit()
    cases = pd.read_csv(args.cases, comment='#')
    with open(args.out, 'w', newline='') as handle:
        handle.write('# Generated by run_decomposition_benchmark.py -- see '
                     'its docstring.\n')
        handle.write(f"# command: {' '.join(sys.argv)}\n")
        handle.write(f'# repo commit: {commit}\n')
        handle.write('# per-piece solver: gurobi_piece_solver (exact); '
                     'partition seed 0\n')
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        handle.flush()
        for instance, group in cases.groupby('instance', sort=False):
            di_g, src, tgt = ni.read_flow_file(instance)
            for setting in settings:
                label = (setting if setting != 'fit'
                         else f'fit{args.fit_qubits}')
                start = time.perf_counter()
                try:
                    pieces, largest = partition(setting, di_g, src, tgt,
                                                args.fit_qubits)
                except ValueError as error:
                    for case in group.itertuples():
                        writer.writerow({
                            'instance': instance, 'B': case.B,
                            'setting': label, 'pieces': 0, 'rule': 'no fit',
                            'optimum': case.Opt_NIP, 'greedy': case.Greedy,
                            'maxflow': case.maxflow,
                            'partition_sec': round(time.perf_counter() - start, 3)})
                    handle.flush()
                    print(f'{instance} {label}: {error}', flush=True)
                    continue
                partition_time = time.perf_counter() - start
                for case in group.itertuples():
                    with contextlib.redirect_stdout(io.StringIO()):
                        rows = case_rows(di_g, src, tgt, case.B, pieces,
                                         args.refine and setting == 'pieces4')
                    reduction = case.maxflow - case.Opt_NIP
                    for rule, mode, budgets, flow, seconds in rows:
                        writer.writerow({
                            'instance': instance, 'B': case.B,
                            'setting': label, 'pieces': len(pieces),
                            'max_piece_qubits': largest, 'rule': rule,
                            'mode': mode, 'budgets': json.dumps(budgets),
                            'flow': flow,
                            'ratio': round((case.maxflow - flow) / reduction, 6),
                            'optimum': case.Opt_NIP, 'greedy': case.Greedy,
                            'maxflow': case.maxflow,
                            'time_sec': round(seconds, 3),
                            'partition_sec': round(partition_time, 3)})
                    handle.flush()
                    best = min(row[3] for row in rows)
                    print(f'{instance} B={case.B} {label} ({len(pieces)} '
                          f'pieces, <= {largest} qubits): best {best}, '
                          f'optimum {case.Opt_NIP}, greedy {case.Greedy}',
                          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

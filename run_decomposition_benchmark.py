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

and every budget rule (mincut, local, marginal, multiplier) is solved in every mode
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
interdicted_edges          actual combined cut, as JSON
allocation_status, surrogate_value, surrogate_lower_bound
                          multiplier allocation diagnostics ONLY; the lower
                          bound is not a bound on global NIP flow
allocation_curves, multiplier_history
                          measured curves and every shared-price update

--rules and --modes restrict the comparison. --max-terminals caps exact
contractions at 0..12 boundary terminals; --multiplier-iterations caps the
price updates. All curve measurement work is included in time_sec. The
split is measured once and charged in full to each mode using that split.

When no partition fits -- a piece whose surroundings cannot be contracted
keeps most of the graph's nodes, so on 125.planarflow no number of pieces
gets under 156 qubits -- each case gets one row with rule 'no fit' and the
rest of the run carries on.

Rows are appended as they finish, so a long run keeps what it has done.
Every invocation starts a fresh file and refuses to overwrite an existing
output. All piece models use eight Gurobi threads; partitioning and exact
contraction are separate, serial NetworkX operations. Needs a Gurobi license.

    python run_decomposition_benchmark.py --refine \\
        --out results/decomposition_benchmark.csv
"""
import argparse
import contextlib
import csv
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
           'time_sec', 'partition_sec', 'interdicted_edges', 'allocation_status',
           'surrogate_value', 'surrogate_lower_bound', 'allocation_curves',
           'multiplier_history']


def partition(setting, di_g, src, tgt, fit_qubits=IBM_QUBITS, seed=0,
              max_terminals=dc.DEFAULT_MAX_TERMINALS):
    """
    (pieces, largest piece QUBO) for a setting. Raises ValueError from
    partition_to_fit() when no partition fits `fit_qubits`.
    """
    if setting == 'pieces4':
        pieces = dc.partition_graph(di_g, 4, seed=seed)
    elif setting == 'fit':
        pieces, _ = dc.partition_to_fit(di_g, src, tgt, fit_qubits, seed=seed,
                                        max_terminals=max_terminals)
    else:
        raise ValueError(f'unknown setting {setting!r}')
    sizes = [dc.piece_qubo_size(dc.piece_subproblem(di_g, src, tgt, pieces, i,
                                                  max_terminals=max_terminals))
             for i in range(len(pieces))]
    return pieces, max(sizes)


def case_rows(di_g, src, tgt, B, pieces, refine, *, rules=dc.SPLIT_STRATEGIES,
              modes=dc.SOLVE_MODES, max_terminals=dc.DEFAULT_MAX_TERMINALS,
              multiplier_iterations=100):
    """Yield each completed variant with its actual cut and allocation diagnostics."""
    solve = dc.gurobi_piece_solver
    for rule in rules:
        start = time.perf_counter()
        allocation = {}
        if rule == 'multiplier':
            allocation = dc.common_multiplier_split(
                di_g, src, tgt, B, pieces, solve, max_terminals=max_terminals,
                max_iterations=multiplier_iterations)
            budgets = allocation['budgets']
        else:
            budgets = dc.split_budget(di_g, src, tgt, B, pieces, rule, solve,
                                       max_terminals=max_terminals)
        split_time = time.perf_counter() - start
        for mode in modes:
            start = time.perf_counter()
            cut, flow = dc.solve_decomposed(di_g, src, tgt, pieces, budgets,
                                            solve, mode, max_terminals=max_terminals)
            yield (rule, mode, budgets, flow, split_time + time.perf_counter() - start,
                   cut, allocation)
        if rule == 'marginal' and refine:
            start = time.perf_counter()
            refined, cut, flow = dc.refine_split(di_g, src, tgt, pieces, budgets,
                                                 solve, mode='iterated',
                                                 max_terminals=max_terminals)
            yield ('marginal+refine', 'iterated', refined, flow,
                   split_time + time.perf_counter() - start, cut, {})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cases', default=SLIDE_CSV,
                        help='CSV with instance, B, Opt_NIP, Greedy, maxflow')
    parser.add_argument('--settings', default=','.join(SETTINGS))
    parser.add_argument('--rules', default=','.join(dc.SPLIT_STRATEGIES))
    parser.add_argument('--modes', default=','.join(dc.SOLVE_MODES))
    parser.add_argument('--max-terminals', type=int, default=dc.DEFAULT_MAX_TERMINALS,
                        help='cap terminals in each exact contraction (0..12)')
    parser.add_argument('--multiplier-iterations', type=int, default=100)
    parser.add_argument('--refine', action='store_true',
                        help='add refine_split() rows (4-piece setting)')
    parser.add_argument('--fit-qubits', type=int, default=IBM_QUBITS,
                        help='qubit limit of the fit setting (default 156)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    settings = args.settings.split(',')
    rules, modes = args.rules.split(','), args.modes.split(',')
    for label, selected, supported in [('settings', settings, SETTINGS),
                                      ('rules', rules, dc.SPLIT_STRATEGIES),
                                      ('modes', modes, dc.SOLVE_MODES)]:
        if any(value not in supported for value in selected) or len(set(selected)) != len(selected):
            parser.error(f'--{label} must select distinct values from {supported}')
    if not 0 <= args.max_terminals <= dc.DEFAULT_MAX_TERMINALS:
        parser.error(f'--max-terminals must be between 0 and {dc.DEFAULT_MAX_TERMINALS}')
    if 'multiplier' in rules and args.multiplier_iterations < 1:
        parser.error('--multiplier-iterations must be positive')

    commit = rb.repo_commit()
    cases = pd.read_csv(args.cases, comment='#')
    with open(args.out, 'x', newline='') as handle:
        handle.write('# Generated by run_decomposition_benchmark.py -- see '
                     'its docstring.\n')
        handle.write(f"# command: {' '.join(sys.argv)}\n")
        handle.write(f'# repo commit: {commit}\n')
        handle.write('# per-piece solver: gurobi_piece_solver (exact); '
                     'partition seed 0\n')
        handle.write(f'# Gurobi Threads: {ni.DEFAULT_GUROBI_THREADS}; '
                     'fresh run, no resumed rows\n')
        handle.write(f'# rules: {args.rules}; modes: {args.modes}; '
                     f'max_terminals: {args.max_terminals}; '
                     f'multiplier_iterations: {args.multiplier_iterations}\n')
        handle.write('# multiplier bounds apply only to the separable allocation '
                     'surrogate, not global NIP; time_sec includes all curve solves; '
                     'actual combined cuts are scored on the original graph\n')
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        handle.flush()
        for instance, group in cases.groupby('instance', sort=False):
            di_g, src, tgt = ni.read_flow_file(instance)
            for setting in settings:
                label = (setting if setting != 'fit'
                         else f'fit{args.fit_qubits}')
                print(f'{instance} {label}: partitioning and contracting '
                      '(NetworkX)', flush=True)
                start = time.perf_counter()
                try:
                    pieces, largest = partition(setting, di_g, src, tgt,
                                                args.fit_qubits,
                                                max_terminals=args.max_terminals)
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
                    print(f'{instance} B={case.B} {label}: solving pieces '
                          f'(Gurobi Threads={ni.DEFAULT_GUROBI_THREADS})',
                          flush=True)
                    reduction = case.maxflow - case.Opt_NIP
                    best = float('inf')
                    # Consume the generator under the output redirect and flush
                    # every completed variant before starting the next solve.
                    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                        for rule, mode, budgets, flow, seconds, cut, allocation in case_rows(
                                di_g, src, tgt, case.B, pieces,
                                args.refine and setting == 'pieces4', rules=rules, modes=modes,
                                max_terminals=args.max_terminals,
                                multiplier_iterations=args.multiplier_iterations):
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
                                'partition_sec': round(partition_time, 3),
                                'interdicted_edges': json.dumps(cut),
                                'allocation_status': allocation.get('status', ''),
                                'surrogate_value': allocation.get('surrogate_value', ''),
                                'surrogate_lower_bound': allocation.get('surrogate_lower_bound', ''),
                                'allocation_curves': json.dumps(allocation['curves']) if allocation else '',
                                'multiplier_history': json.dumps(allocation['history']) if allocation else '',
                            })
                            handle.flush()
                            best = min(best, flow)
                    print(f'{instance} B={case.B} {label} ({len(pieces)} '
                          f'pieces, <= {largest} qubits): best {best}, '
                          f'optimum {case.Opt_NIP}, greedy {case.Greedy}',
                          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

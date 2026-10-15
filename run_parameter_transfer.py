"""
Reproducible held-out comparison for issue #16 (requirements-transfer.txt).

Train on seeded small two-route flow networks, freeze the donor bank/GAT
and RWS schedule, then evaluate unseen capacity draws at the requested
size. All budgets of a graph belong to one split. Default test graphs have
one more vertex than training graphs. This is a small controlled transfer
experiment, not evidence about hard interdiction instances or advantage.

    python run_parameter_transfer.py --out results/parameter_transfer.csv

The CSV is accompanied by a JSON record of every graph and trained angle,
and one GAT checkpoint per depth. Refuses to overwrite an existing result.
"""
import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
import time

import networkx as nx
import numpy as np

import qaoa_common as qc
from qaoa_qiskit import QiskitBackend
from qaoa_transfer import AngleDonor, IBMTransfer
from qaoa_warm_start import regularized_warm_start
import run_benchmark as rb


def two_route_graph(n, rng):
    """Two directed source/sink routes, random capacities, optional cross arc."""
    if n < 4:
        raise ValueError('two-route graphs need at least four vertices')
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    middle = list(range(1, n - 1))
    split = len(middle) // 2
    for route in (middle[:split], middle[split:]):
        path = [0] + route + [n - 1]
        for u, v in zip(path, path[1:]):
            graph.add_edge(u, v, weight=int(rng.integers(1, 21)))
    if rng.random() < 0.5:
        graph.add_edge(middle[0], middle[split], weight=int(rng.integers(1, 21)))
    return graph


def make_dataset(train_count, test_count, test_nodes=5, seed=0):
    """Disjoint graphs with a valid B=1 window; return problems plus provenance."""
    rng = np.random.default_rng(seed)
    seen, records, problems = set(), [], {'train': [], 'test': []}
    for split, count, n in (('train', train_count, 4), ('test', test_count, test_nodes)):
        attempts = 0
        while len(problems[split]) < count:
            attempts += 1
            if attempts > 100 * count:
                raise RuntimeError('could not generate enough distinct valid windows')
            graph = two_route_graph(n, rng)
            arcs = [(u, v, graph[u][v]['weight']) for u, v in sorted(graph.edges)]
            signature = (n, tuple(arcs))
            if signature in seen:
                continue
            curve = rb.optimum_curve('brute', graph, 0, n - 1, 1)
            if rb.choose_lambda(*rb.lambda_window(curve, 1)) is None:
                continue
            group = f'{split}-{len(problems[split]):03d}'
            problem = qc.problem_from_graph(graph, 0, n - 1, 1, instance=group)
            problems[split].append(problem)
            records.append({'group': group, 'split': split, 'nodes': n,
                            'arcs': arcs, 'src': 0, 'tgt': n - 1,
                            'budget': 1, 'lambda': problem.lam})
            seen.add(signature)
    return problems, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/parameter_transfer.csv'))
    parser.add_argument('--train-count', type=int, default=12)
    parser.add_argument('--test-count', type=int, default=6)
    parser.add_argument('--test-nodes', type=int, default=5)
    parser.add_argument('--reps', default='1,2')
    parser.add_argument('--maxiter', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--strength', type=float, default=0.6)
    args = parser.parse_args(argv)
    reps_list = [int(value) for value in args.reps.split(',')]
    if min(args.train_count, args.test_count, args.maxiter, args.epochs,
           args.shots, *reps_list) <= 0:
        parser.error('counts, depths and iteration limits must be positive')
    artifacts = [args.out, args.out.with_suffix('.json')]
    artifacts += [args.out.with_suffix(f'.p{reps}.gat.json') for reps in reps_list]
    if any(path.exists() for path in artifacts):
        parser.error('output already exists; choose a new --out path')

    # Heavy optional dependencies are loaded only by this experiment.
    import torch
    from qaoa_gat import GATTransfer
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    problems, graph_records = make_dataset(
        args.train_count, args.test_count, args.test_nodes, args.seed)
    record = {'schema': 1, 'complete': False, 'commit': rb.repo_commit(),
              'versions': {name: version(name) for name in (
                  'numpy', 'scipy', 'networkx', 'qiskit', 'qiskit-aer',
                  'torch', 'torch-geometric', 'qaoa_training_pipeline')},
              'arguments': {key: str(val) if isinstance(val, Path) else val
                            for key, val in vars(args).items()},
              'graphs': graph_records, 'training': {}}
    fields = ['group', 'reps', 'method', 'donor_group', 'qubits', 'lambda',
              'opt_nip', 'greedy', 'expected_energy', 'approx_ratio',
              'gap_closed', 'p_optimal', 'p_feasible', 'shot_mean_ratio',
              'flow_after_repair', 'target_evaluations', 'setup_sec',
              'selection_sec', 'optimization_sec', 'sample_sec', 'theta']
    with args.out.open('x', newline='') as stream:
        stream.write('# generator: run_parameter_transfer.py\n')
        stream.write(f'# source: {record["commit"]}\n')
        stream.write(f'# arguments: {json.dumps(record["arguments"], sort_keys=True)}\n')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for reps in reps_list:
            started = time.perf_counter()
            donors = []
            for problem in problems['train']:
                backend = QiskitBackend(problem, reps)
                theta, _, evaluations = qc.optimize_angles(
                    backend.normalized_expectation, reps,
                    seed=args.seed, maxiter=args.maxiter)
                donors.append(AngleDonor(problem, theta, problem.instance))
                print(f'p={reps} trained {problem.instance}: {evaluations} evaluations', flush=True)
            donor_sec = time.perf_counter() - started
            started = time.perf_counter()
            gat, history = GATTransfer.train(donors, QiskitBackend,
                                             epochs=args.epochs, seed=args.seed)
            gat_sec = time.perf_counter() - started
            gat.save(args.out.with_suffix(f'.p{reps}.gat.json'))
            ibm = IBMTransfer(donors)
            started = time.perf_counter()
            warm_backends = [QiskitBackend(
                p, reps, warm_start_angles=regularized_warm_start(
                    p, args.strength, args.seed)) for p in problems['train']]
            # Fixed across all held-out graphs; only the classical initial
            # product state is recomputed on each target.
            rws_theta, _, rws_evaluations = qc.optimize_angles(
                lambda theta: np.mean([
                    backend.normalized_expectation(theta) / backend.problem.num_qubits
                    for backend in warm_backends]),
                reps, seed=args.seed, maxiter=args.maxiter)
            rws_sec = time.perf_counter() - started
            record['training'][str(reps)] = {
                'donors': [{'group': d.group, 'theta': d.theta.tolist()} for d in donors],
                'donor_training_sec': donor_sec, 'gat_training_sec': gat_sec,
                'gat_loss': history, 'rws_theta': rws_theta.tolist(),
                'rws_training_sec': rws_sec, 'rws_evaluations': rws_evaluations}
            args.out.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
            for problem in problems['test']:
                for method in ('optimized', 'fixed', 'ibm', 'gat', 'rws'):
                    setup = time.perf_counter()
                    warm = (regularized_warm_start(problem, args.strength, args.seed)
                            if method == 'rws' else None)
                    backend = QiskitBackend(problem, reps, warm_start_angles=warm)
                    setup_sec = time.perf_counter() - setup
                    start = time.perf_counter()
                    donor_group, theta = '', None
                    if method in ('ibm', 'gat'):
                        theta, index = (ibm if method == 'ibm' else gat).select(problem)
                        donor_group = donors[index].group
                    elif method == 'fixed':
                        theta, donor_group = donors[0].theta, donors[0].group
                    elif method == 'rws':
                        theta, donor_group = rws_theta, 'pooled-training'
                    selection_sec = time.perf_counter() - start
                    result = qc.run_qaoa(backend, problem, reps, theta=theta,
                                         maxiter=args.maxiter, shots=args.shots,
                                         seed=args.seed)
                    row = {key: getattr(result, key) for key in (
                        'opt_nip', 'greedy', 'expected_energy', 'approx_ratio',
                        'gap_closed', 'p_optimal', 'p_feasible', 'shot_mean_ratio',
                        'flow_after_repair')}
                    row.update(group=problem.instance, reps=reps, method=method,
                               donor_group=donor_group, qubits=problem.num_qubits,
                               **{'lambda': problem.lam},
                               target_evaluations=result.evaluations,
                               setup_sec=setup_sec, selection_sec=selection_sec,
                               optimization_sec=result.train_time_sec,
                               sample_sec=result.sample_time_sec,
                               theta=json.dumps(result.theta))
                    writer.writerow(row)
                    stream.flush()
                print(f'p={reps} scored {problem.instance}', flush=True)
    record['complete'] = True
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

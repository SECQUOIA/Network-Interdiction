"""Bounded software comparison of penalty QAOA, fixed-weight XY QAOA, and SA.

All methods use the same original NIP references and qaoa_common shot/repair
scorer. Raw energies belong to different Hamiltonians and are not quality
ratios across methods. No device service or hardware backend is used.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
from numbers import Integral
from pathlib import Path
import subprocess
import time

import gurobipy as gp
import neal

import network_interdiction as ni
import qaoa_common as qc
import run_benchmark as rb
from solve_qubo_via_qaoa import solve_bqm_qaoa, validate_options

MAX_EDGES = 7
MAX_QUBITS = 24
SOURCE_FILES = ('run_xy_comparison.py', 'qaoa_xy.py', 'qaoa_qiskit.py',
                'qaoa_common.py', 'solve_qubo_via_qaoa.py',
                'network_interdiction.py', 'run_benchmark.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def references(path, budget):
    """Guard all enumeration/compilation/solver work before dispatch."""
    graph, source, sink = ni.read_flow_file(path)
    edges = sorted(graph.edges())
    if not edges or len(edges) > MAX_EDGES or 3*len(edges)+len(graph)-2 > MAX_QUBITS:
        raise ValueError('comparison limited to 1..7 arcs and at most 24 compiled qubits')
    if isinstance(budget, bool) or not isinstance(budget, Integral) or not 0 <= budget <= len(edges):
        raise ValueError('budget must be between zero and the number of arcs')
    if any(data['weight'] < 0 for _, _, data in graph.edges(data=True)):
        raise ValueError('comparison requires nonnegative capacities')
    if source == sink or source not in graph or sink not in graph:
        raise ValueError('source and sink must be distinct graph nodes')
    if sum(data['weight'] for _, _, data in graph.edges(data=True)) > 2**30:
        raise ValueError('comparison capacities exceed the validated integer range')
    start = time.perf_counter()
    # Exactly 2**m subset evaluations, with m<=7 established above.
    flows_by_size = [[rb.max_flow(graph, source, sink, cut)
                      for cut in itertools.combinations(edges, k)] for k in range(len(edges)+1)]
    curve = list(map(min, flows_by_size))
    curve_seconds = time.perf_counter()-start
    model, variables = ni._build_nip_milp(graph, source, sink, budget)
    with model:
        model.Params.TimeLimit = 20
        model.Params.MIPGap = 0
        start = time.perf_counter()
        model.optimize()
        milp_seconds = time.perf_counter()-start
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f'NIP reference incomplete: status={model.Status}')
        selected = [edge for edge in edges if variables[edge].X > .5]
        objective, bound = float(model.ObjVal), float(model.ObjBound)
    if abs(objective-curve[budget]) > 1e-6 \
            or rb.max_flow(graph, source, sink, selected) != curve[budget]:
        raise AssertionError('Gurobi and independent exhaustive interdiction reference differ')
    lo, hi = rb.lambda_window(curve, budget)
    lam = rb.choose_lambda(lo, hi)
    beta = max(max(data['weight'] for _, _, data in graph.edges(data=True)), lam or 0) + 1
    return graph, source, sink, {
        'curve': curve, 'curve_seconds': curve_seconds,
        'lambda_lo': lo if math.isfinite(lo) else None,
        'lambda_hi': hi if math.isfinite(hi) else None, 'lambda': lam, 'beta': beta,
        'nip_optimum': curve[budget], 'milp_objective': objective, 'milp_bound': bound,
        'milp_seconds': milp_seconds, 'milp_edges': selected,
        'first_B_arcs_flow': rb.max_flow(graph, source, sink, edges[:budget]),
        'uniform_fixed_weight_p_optimal': sum(v == curve[budget] for v in flows_by_size[budget])
            / len(flows_by_size[budget]),
        'uniform_all_bits_p_optimal': sum(v == curve[budget] for values in flows_by_size[:budget+1]
                                          for v in values) / (1 << len(edges))}


def score_and_retain(path, graph, source, sink, budget, lam, ref, samples, bqm):
    """Reuse the existing raw-shot and repair semantics, retaining every count."""
    variables = list(samples.variables)
    h, J, offset = qc.ising_from_bqm(bqm, variables)
    problem = qc.NIPQaoaProblem(str(path), graph, source, sink, budget, lam, None,
                               bqm, variables, h, J, offset, ref['nip_optimum'])
    counts = {}
    for row in samples.record:
        bits = tuple(map(int, row.sample))
        counts[bits] = counts.get(bits, 0) + int(row.num_occurrences)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        scores = qc.score_samples(problem, counts)
    score_seconds = time.perf_counter()-start
    if not math.isfinite(scores['flow_before_repair']):
        scores['flow_before_repair'] = None
    exactly_budget = sum(count for bits, count in counts.items()
                         if sum(bits[q] for q in problem.x_qubits) == budget)
    scores['p_exact_budget'] = exactly_budget/sum(counts.values())
    scores['score_and_repair_seconds'] = score_seconds
    return {'scores': scores, 'variables': variables,
            'counts': {''.join(map(str, bits)): count for bits, count in counts.items()},
            'count_bit_order': 'characters follow variables, qubit zero first'}


def run_case(path, budget, *, reps=1, shots=1024, maxiter=20, seed=0, checkpoint=None):
    validate_options(reps=reps, shots=shots, maxiter=maxiter, seed=seed, max_qubits=MAX_QUBITS)
    if reps > 3 or shots > 10000 or maxiter > 100:
        raise ValueError('comparison caps: p<=3, shots<=10000, maxiter<=100')
    graph, source, sink, ref = references(path, budget)
    result = {'instance': str(path), 'input_sha256': digest(path), 'budget': budget,
              'references': ref, 'methods': [], 'status': 'running'}
    if checkpoint:
        checkpoint(result)
    for mode in ('sa', 'penalty', 'xy_basis', 'xy_dicke'):
        if mode in ('sa', 'penalty') and ref['lambda'] is None:
            result['methods'].append({'mode': mode, 'status': 'not_applicable',
                                      'reason': 'empty strict linear penalty window'})
            if checkpoint:
                checkpoint(result)
            continue
        lam = ref['lambda'] if mode in ('sa', 'penalty') else 0
        start = time.perf_counter()
        bqm = ni.nip_to_qubo(budget, graph, source, sink, lam, ref['beta'])
        if len(bqm) > MAX_QUBITS:
            raise ValueError('compiled BQM exceeds the comparison limit')
        build_seconds = time.perf_counter()-start
        start = time.perf_counter()
        if mode == 'sa':
            samples = neal.SimulatedAnnealingSampler().sample(
                bqm, num_reads=shots, num_sweeps=1000, seed=seed)
            metadata = {'backend': 'dwave-neal', 'reads': shots, 'sweeps': 1000, 'seed': seed}
        else:
            options = {} if mode == 'penalty' else {
                'fixed_weight_variables': [f'x_{i}' for i in range(len(graph.edges()))],
                'hamming_weight': budget, 'initial_state': mode.removeprefix('xy_')}
            samples = solve_bqm_qaoa(bqm, reps=reps, shots=shots, maxiter=maxiter,
                                    seed=seed, max_qubits=MAX_QUBITS, **options)
            metadata = samples.info['qaoa_metadata']
        sample_seconds = time.perf_counter()-start
        row = score_and_retain(path, graph, source, sink, budget, lam, ref, samples, bqm)
        if mode.startswith('xy_') and row['scores']['p_exact_budget'] != 1:
            raise AssertionError('XY samples leaked out of the fixed-budget sector')
        row.update(mode=mode, status='completed', cardinality_penalty=lam,
                   dual_penalty=ref['beta'], bqm_build_seconds=build_seconds,
                   sampling_pipeline_seconds=sample_seconds, metadata=metadata)
        result['methods'].append(row)
        if checkpoint:
            checkpoint(result)
    result['status'] = 'completed'
    if checkpoint:
        checkpoint(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, metavar='PATH:B')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--maxiter', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= len(args.case) <= 4 or args.out.exists():
        parser.error('provide 1..4 cases and a new output path')
    cases = [(p, int(b)) for p, b in (case.rsplit(':', 1) for case in args.case)]
    for path, _ in cases:
        Path(path).read_bytes()
    data = {'schema': 1, 'producer_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {p: digest(p) for p in SOURCE_FILES},
        'versions': {p: importlib.metadata.version(p) for p in ('qiskit', 'qiskit-aer',
            'gurobipy', 'dwave-neal', 'networkx', 'dimod', 'numpy', 'pyqubo')},
        'controls': {'reps': args.reps, 'shots': args.shots, 'maxiter': args.maxiter,
                     'seed': args.seed, 'gurobi_threads': ni.DEFAULT_GUROBI_THREADS,
                     'max_qubits': MAX_QUBITS, 'max_edges': MAX_EDGES,
                     'milp_seconds_limit': 20}, 'rows': [], 'failures': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(row):
        if not data['rows'] or data['rows'][-1] is not row:
            data['rows'].append(row)
        args.out.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

    for path, budget in cases:
        try:
            run_case(path, budget, reps=args.reps, shots=args.shots,
                     maxiter=args.maxiter, seed=args.seed, checkpoint=checkpoint)
        except Exception as error:
            data['failures'].append({'instance': path, 'budget': budget,
                                     'error': f'{type(error).__name__}: {error}'})
            raise
        finally:
            args.out.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

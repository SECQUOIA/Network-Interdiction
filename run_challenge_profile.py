"""Bounded MILP/Benders difficulty screen; no QPU or unbounded exact reference.

Each solver gets the same wall-time allowance (construction included), eight
threads and zero requested MIP gap. NetworkX/model construction are cooperative;
an external process deadline remains appropriate. Retain a checkpoint after each
solver. Classifications concern these two implementations and this machine only.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import gurobipy as gp
import networkx as nx

import benders
import challenge_instances as ci
import classical_baselines as cb
import network_interdiction as ni

SOURCES = ['run_challenge_profile.py', 'challenge_instances.py', 'benders.py',
           'classical_baselines.py', 'network_interdiction.py']


def validate(graph, source, sink, budget, allowed, time_limit):
    cb._validate(graph, source, sink, budget)
    if (not math.isfinite(time_limit) or not 0 < time_limit <= 60
            or isinstance(time_limit, bool)):
        raise ValueError('time limit must be positive and at most 60 seconds')
    if len(graph) > 1000 or graph.number_of_edges() > 4000:
        raise ValueError('profile input exceeds 1000 nodes or 4000 arcs')
    if any(type(d['weight']) is not int for _, _, d in graph.edges(data=True)):
        raise ValueError('integer capacities required')
    if sum(d['weight'] for _, _, d in graph.edges(data=True)) > 2**50:
        raise ValueError('capacity total exceeds profiling precision guard')
    if allowed - set(graph.edges()) or budget > len(allowed):
        raise ValueError('invalid allowed set or budget')


def terminal_flags(cut, source, sink):
    incident = [source in e or sink in e for e in cut]
    return {'any_terminal_incident': any(incident),
            'all_terminal_incident': bool(incident) and all(incident),
            'terminal_incident_count': sum(incident)}


def milp_profile(graph, source, sink, budget, allowed, *, time_limit=20):
    validate(graph, source, sink, budget, allowed, time_limit)
    start = time.perf_counter()
    original = nx.maximum_flow_value(graph, source, sink, capacity='weight')
    result = {'flow': original, 'interdicted_edges': [], 'lower_bound': 0.,
              'status': 'time_limit', 'node_count': 0., 'solver_time_sec': 0.,
              'trajectory': [], 'incumbents': []}
    model, x = ni._build_nip_milp(graph, source, sink, budget, allowed)
    try:
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 0
        model.Params.Seed = 0
        edges = sorted(graph.edges())
        last = -1.

        def callback(m, where):
            nonlocal last
            if where not in (gp.GRB.Callback.MIP, gp.GRB.Callback.MIPSOL):
                return
            elapsed = float(m.cbGet(gp.GRB.Callback.RUNTIME))
            if where == gp.GRB.Callback.MIP and elapsed - last >= .25:
                bound = float(m.cbGet(gp.GRB.Callback.MIP_OBJBND))
                incumbent = float(m.cbGet(gp.GRB.Callback.MIP_OBJBST))
                result['trajectory'].append({
                    'solver_time_sec': elapsed,
                    'lower_bound': max(0., bound) if abs(bound) < gp.GRB.INFINITY else 0.,
                    'milp_incumbent_objective': incumbent if abs(incumbent) < gp.GRB.INFINITY else None,
                    'node_count': float(m.cbGet(gp.GRB.Callback.MIP_NODCNT))})
                last = elapsed
            elif where == gp.GRB.Callback.MIPSOL:
                values = m.cbGetSolution([x[e] for e in edges])
                result['incumbents'].append({'solver_time_sec': elapsed,
                                             'interdicted_edges': [e for e, v in zip(edges, values) if v > .5]})

        remaining = time_limit - (time.perf_counter() - start)
        if remaining > 0:
            model.Params.TimeLimit = remaining
            model.optimize(callback)
            if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED):
                raise RuntimeError(f'unexpected MILP status {model.Status}')
            result['solver_time_sec'] = float(model.Runtime)
            result['node_count'] = float(model.NodeCount)
            if abs(model.ObjBound) < gp.GRB.INFINITY:
                result['lower_bound'] = max(0., float(model.ObjBound))
            if model.SolCount:
                result['incumbents'].append({'solver_time_sec': float(model.Runtime),
                                             'interdicted_edges': [e for e in edges if x[e].X > .5]})
            result['status'] = {gp.GRB.OPTIMAL: 'optimal', gp.GRB.TIME_LIMIT: 'time_limit',
                                gp.GRB.INTERRUPTED: 'interrupted'}[model.Status]
        result['trajectory'].append({'solver_time_sec': result['solver_time_sec'],
                                     'lower_bound': result['lower_bound'],
                                     'milp_incumbent_objective': float(model.ObjVal) if remaining > 0 and model.SolCount else None,
                                     'node_count': result['node_count']})
        result['runtime_sec'] = time.perf_counter() - start
    finally:
        model.dispose()
    # Independent scoring is outside solver timing and never slows its callback.
    before = time.perf_counter()
    for item in result['incumbents']:
        cut = item['interdicted_edges']
        if len(cut) > budget or not set(cut) <= allowed:
            raise RuntimeError('invalid MILP interdiction witness')
        residual = graph.copy()
        residual.remove_edges_from(cut)
        item['flow'] = nx.maximum_flow_value(residual, source, sink, capacity='weight')
        if item['flow'] < result['flow']:
            result['flow'], result['interdicted_edges'] = item['flow'], cut
    result['audit_time_sec'] = time.perf_counter() - before
    if result['lower_bound'] > result['flow'] + 1e-7:
        raise RuntimeError('MILP bound exceeds independently scored feasible flow')
    if result['status'] == 'optimal' and abs(result['flow'] - result['lower_bound']) > 1e-7:
        raise RuntimeError('optimal MILP status without a closed audited gap')
    result['relative_gap'] = cb.relative_gap(result['flow'], result['lower_bound'])
    result.update(terminal_flags(result['interdicted_edges'], source, sink))
    return result


def classify(milp, bend, allowed, source, sink, *, strain_seconds=5):
    """Conservative screen; an interior optimum witness need not be unique.

    When every allowed arc is interior, even an unknown optimum is interior by
    construction. Otherwise a completed optimum witness is required. Any terminal
    arc in either solver's optimal witness excludes the case. Time alone is never
    called intrinsic hardness or an optimum certificate.
    """
    exact = [r for r in (milp, bend) if r['status'] == 'optimal']
    if any(r['any_terminal_incident'] for r in exact):
        return 'excluded_terminal_optimum_witness'
    interior = all(source not in e and sink not in e for e in allowed)
    if not interior and not exact:
        return 'unclassified_optimal_terminal_location_unknown'
    if all(r['runtime_sec'] >= strain_seconds for r in (milp, bend)):
        return 'hard_for_both_measured_solvers'
    return 'not_hard_for_both_measured_solvers'


def profile_case(case, graph, source, sink, budget, allowed, time_limit):
    validate(graph, source, sink, budget, allowed, time_limit)
    start = time.perf_counter()
    # Count the actual compiled linear BQM, including eliminated variables.
    # Coefficients 1 and 2 are dimension probes only, not solution guarantees.
    bqm = ni.nip_to_qubo(budget, graph, source, sink, 1, 2, interdictable=allowed)
    return {'name': case['name'], 'family': case['family'], 'B': budget,
            'file': case['file'], 'sha256': case['sha256'],
            'nodes': len(graph), 'arcs': graph.number_of_edges(),
            'allowed_arcs': len(allowed), 'qubits': bqm.num_variables,
            'planar_undirected': nx.check_planarity(graph.to_undirected())[0],
            'source_out_degree': graph.out_degree(source),
            'sink_in_degree': graph.in_degree(sink),
            'couplings': bqm.num_interactions,
            'qubo_compile_sec': time.perf_counter() - start,
            'source': source, 'sink': sink,
            'allowed_only_interior': all(source not in e and sink not in e for e in allowed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ci.DATA / 'manifest.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, default=20)
    parser.add_argument('--names', help='optional comma-separated subset for a separate pilot')
    args = parser.parse_args(argv)
    if not math.isfinite(args.time_limit) or not 0 < args.time_limit <= 60:
        parser.error('time limit must be in (0,60]')
    cases = ci.load_cases(args.manifest)
    if args.names:
        names = set(args.names.split(','))
        if names - {c[0]['name'] for c in cases}:
            parser.error('unknown case name')
        cases = [c for c in cases if c[0]['name'] in names]
    for _, graph, source, sink, budget, allowed in cases:
        validate(graph, source, sink, budget, allowed, args.time_limit)
    root = Path(__file__).resolve().parent
    artifact = {'schema': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_hashes': {p: ci.digest(root / p) for p in SOURCES},
        'manifest_sha256': ci.digest(args.manifest), 'versions': {
            p: importlib.metadata.version(p) for p in ['gurobipy', 'networkx', 'numpy', 'pyqubo', 'dimod']},
        'python': platform.python_version(), 'machine': platform.platform(),
        'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')), 'unknown') if Path('/proc/cpuinfo').exists() else platform.processor(),
        'thread_environment': {name: os.environ.get(name) for name in
                               ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'settings': {'time_limit_per_solver_sec': args.time_limit, 'threads': 8,
                     'strain_seconds': 5, 'repetitions': 1, 'seed': 0,
                     'max_cases': 24, 'max_nodes': 1000, 'max_arcs': 4000,
                     'benders_max_iterations': 1000},
        'attempts': [{'attempt': 1, 'status': 'running', 'completed_solver_calls': 0}], 'cases': []}
    # Exclusive creation prevents loss of a previous attempt. Every checkpoint
    # precedes the next expensive call; a process kill leaves completed work.
    with args.out.open('x') as handle:
        def save():
            handle.seek(0)
            json.dump(artifact, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.truncate()
            handle.flush()
        save()
        try:
            for case, graph, source, sink, budget, allowed in cases:
                print(f'{case["name"]} B={budget}', flush=True)
                row = profile_case(case, graph, source, sink, budget, allowed, args.time_limit)
                artifact['cases'].append(row)
                save()
                row['milp'] = milp_profile(graph, source, sink, budget, allowed, time_limit=args.time_limit)
                artifact['attempts'][0]['completed_solver_calls'] += 1
                save()
                row['benders'] = benders.solve(graph, source, sink, budget,
                                                time_limit=args.time_limit, interdictable=allowed)
                row['benders'].update(terminal_flags(row['benders']['interdicted_edges'], source, sink))
                row['benders']['relative_gap'] = cb.relative_gap(row['benders']['flow'], row['benders']['lower_bound'])
                artifact['attempts'][0]['completed_solver_calls'] += 1
                row['classification'] = classify(row['milp'], row['benders'], allowed, source, sink)
                save()
                print(f'  {row["classification"]}: MILP {row["milp"]["flow"]} '
                      f'[{row["milp"]["lower_bound"]}] / Benders {row["benders"]["flow"]} '
                      f'[{row["benders"]["lower_bound"]}]', flush=True)
            artifact['attempts'][0]['status'] = 'completed'
        except Exception as error:
            artifact['attempts'][0].update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save()


if __name__ == '__main__':
    main()

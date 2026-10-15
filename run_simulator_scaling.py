"""Measured statevector/MPS limits with fixed-budget and penalty QAOA.

This is a bounded simulator comparison, never a claim of a universal maximum
or quantum advantage. All MPS outputs are approximate and their raw budget
feasibility is measured, including possible XY truncation leakage.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import challenge_instances as ci
import network_interdiction as ni
import run_matched_benchmark as matched

ROOT = Path(__file__).resolve().parent
CASES = [('testcases/tiny_a.flow', 1), ('testcases/tiny_b.flow', 1),
         ('testcases/mid/knn_6_2_s0.flow', 1), ('testcases/mid/erdos_8_0.45_s0.flow', 1),
         ('testcases/mid/planar_13_s0.flow', 1), ('testcases/mid/erdos_13_0.25_s0.flow', 1),
         ('testcases/challenge/cover_n30_s0.flow', 192)]
METHODS = ['reference', 'sa', 'penalty_statevector', 'xy_statevector', 'penalty_mps', 'xy_mps']


def inputs(path, budget):
    if (path, budget) not in CASES:
        raise ValueError('unknown scaling case')
    graph, source, sink = ni.read_flow_file(ROOT/path)
    allowed = set(graph.edges())
    if path.startswith('testcases/challenge/'):
        record = next(c for c in ci.load_cases() if c[0]['name'] == 'cover_n30_s0')
        _, graph, source, sink, B, allowed = record
        if B != budget:
            raise ValueError('budget mismatch')
    return graph, source, sink, allowed


def worker(path, budget, method):
    import bounded_benchmarks as bb
    from certified_penalties import certified_squared_qubo
    from solve_qubo_via_qaoa import solve_bqm_qaoa
    graph, source, sink, allowed = inputs(path, budget)
    if method == 'reference':
        if path.startswith('testcases/challenge/'):
            ref = matched.cases_and_references()['cover_n30_s0',budget]
            return dict(status='reused_archived_reference', result=ref['archived']['milp'],
                        source_commit=ref['reference_source_commit'], artifact=ref['reference'],
                        artifact_sha256=ref['reference_sha256'])
        from run_challenge_profile import milp_profile
        result = milp_profile(graph,source,sink,budget,allowed,time_limit=20)
        return dict(status='completed', result=result)
    if method == 'sa':
        return bb.anneal(graph,source,sink,budget,allowed)
    mixer, simulation = method.split('_')
    if mixer not in ('penalty', 'xy') or simulation not in ('statevector','mps'):
        raise ValueError('unknown sampler')
    start = time.perf_counter()
    bqm, certificate = certified_squared_qubo(budget,graph,source,sink,allowed)
    if mixer == 'xy':
        # The register enforces sum(x)=B in exact arithmetic; beta remains the
        # SAME certified value as the penalty comparison. Approximate MPS
        # leakage invalidates that exact-state guarantee and is measured below.
        bqm = ni.nip_to_qubo(budget,graph,source,sink,0,certificate.dual,allowed)
    details = dict(qubits=len(bqm), couplings=bqm.num_interactions,
                   certificate=asdict(certificate), cardinality_penalty=0 if mixer=='xy' else certificate.cardinality,
                   compile_time_sec=time.perf_counter()-start, raw_samples=[])
    cap = 24 if simulation == 'statevector' else 256
    if not 1 <= len(bqm) <= cap:
        return dict(status='qubit_guard_refused', runtime_sec=time.perf_counter()-start, max_qubits=cap, **details)
    work = bb.FlowWork(graph,source,sink,budget,allowed,seconds=50)
    edges = sorted(graph.edges())
    options = dict(method=simulation,max_qubits=cap,order='rcm')
    if simulation == 'mps':
        options['max_bond_dimension'] = 8
    if mixer == 'xy':
        options.update(fixed_weight_variables=[f'x_{i}' for i,e in enumerate(edges) if e in allowed],
                       hamming_weight=budget,initial_state='basis')
    # Baseline for the XY basis initialization; no optimization/refinement.
    initial_cut = sorted(allowed)[:budget]
    details['basis_initial_cut'] = initial_cut
    details['basis_initial_flow'] = work.evaluate(initial_cut)
    # Do not include the initial-cut baseline in the method's feasible result.
    work.best_cut, work.best_flow = [], work.cache[()]
    samples = solve_bqm_qaoa(bqm,reps=1,shots=256,maxiter=8,seed=0,**options)
    variables = list(bqm.variables)
    counts = samples.aggregate()
    details['observed_counts'] = [dict(bits=''.join(str(int(row.sample[v])) for v in variables),
                                           occurrences=int(row.num_occurrences),energy=float(row.energy))
                                  for row in counts.data()]
    details['raw_exact_budget_shots'] = sum(int(row.num_occurrences) for row in counts.data()
                                           if sum(row.sample.get(f'x_{i}',0) for i,e in enumerate(edges) if e in allowed)==budget)
    details['raw_at_most_budget_shots'] = sum(int(row.num_occurrences) for row in counts.data()
                                             if sum(row.sample.get(f'x_{i}',0) for i,e in enumerate(edges) if e in allowed)<=budget)
    details['metadata'] = samples.info['qaoa_metadata']
    variables, records = bb.sample_records(samples,bqm,graph,budget,work)
    details.update(variables=variables, raw_samples=records)
    try:
        for row in records:
            bb.repair(work,row['interdicted_edges'])
        result = work.result(**details)
    except bb.WorkLimit as error:
        result = work.result(str(error),**details)
    result['runtime_sec'] = time.perf_counter()-start
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',nargs=3,metavar=('PATH','B','METHOD'))
    args = parser.parse_args(argv)
    if args.worker:
        resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
        path,B,method = args.worker
        result = worker(path,int(B),method)
        with args.out.open('x') as handle:
            json.dump(result,handle,allow_nan=False)
        return
    for path,B in CASES:
        inputs(path,B)
    matched.cases_and_references()
    artifact = dict(schema=1, **matched.provenance(),
                    settings=dict(reps=1, shots=256, maxiter=8, seed=0, order='rcm',
                                  mps_bond_dimension=8,mps_truncation_threshold=1e-16,
                                  statevector_qubit_cap=24,mps_qubit_cap=256,outer_seconds=60,
                                  memory_bytes=8*1024**3,repetitions=1,
                                  meaningful='completed sampler with >=1 selected raw budget-feasible cut improving empty-cut flow'),
                    attempts=[dict(attempt=1,status='running',completed_methods=0)], cases=[])
    with args.out.open('x') as handle:
        def save():
            handle.seek(0);json.dump(artifact,handle,indent=2,allow_nan=False)
            handle.write('\n');handle.truncate();handle.flush()
        save()
        try:
            for path,B in CASES:
                row=dict(file=path,B=B,sha256=ci.digest(ROOT/path),methods={})
                artifact['cases'].append(row);save()
                for method in METHODS:
                    print(path,B,method,flush=True)
                    row['methods'][method]={'status':'dispatched'};save()
                    result=matched.process_call([path,str(B),method],timeout=60,script='run_simulator_scaling.py')
                    row['methods'][method]=result
                    artifact['attempts'][0]['completed_methods']+=1;save()
                    print(result['status'],result.get('flow'),flush=True)
            artifact['attempts'][0]['status']='completed'
        except Exception as error:
            artifact['attempts'][0].update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save()


if __name__ == '__main__':
    main()

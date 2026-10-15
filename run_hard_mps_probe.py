"""Separate bounded MPS probe beyond the initial ladder's 256-qubit guard.

Only the archived hard covering case is accepted. Two methods, one attempt
apiece, retain intermediate stages before expensive native calls. No hardware.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import bounded_benchmarks as bb
import challenge_instances as ci
from certified_penalties import certified_squared_qubo
import network_interdiction as ni
import run_matched_benchmark as matched

ROOT=Path(__file__).resolve().parent
METHODS=('penalty','xy')
OUTER_SECONDS=600


def checkpoint(path, value):
    temporary=path.with_suffix('.checkpoint')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def inputs():
    return next(c for c in ci.load_cases() if c[0]['name']=='cover_n30_s0')


def worker(method,out):
    from solve_qubo_via_qaoa import solve_bqm_qaoa
    if method not in METHODS:
        raise ValueError('unknown hard probe method')
    _,graph,source,sink,B,allowed=inputs()
    started=time.perf_counter()
    bqm,certificate=certified_squared_qubo(B,graph,source,sink,allowed)
    if method=='xy':
        bqm=ni.nip_to_qubo(B,graph,source,sink,0,certificate.dual,allowed)
    details=dict(status='running_sampler',method=method,qubits=len(bqm),couplings=bqm.num_interactions,
                 certificate=asdict(certificate),compile_time_sec=time.perf_counter()-started,
                 variables=list(bqm.variables),raw_samples=[],observed_counts=[])
    checkpoint(out,details)
    options={}
    if method=='xy':
        options=dict(fixed_weight_variables=[f'x_{i}' for i,e in enumerate(sorted(graph.edges())) if e in allowed],
                     hamming_weight=B,initial_state='basis')
    samples=solve_bqm_qaoa(bqm,reps=1,shots=256,maxiter=4,seed=0,method='mps',
                          max_qubits=1200,max_bond_dimension=8,order='rcm',extended_mps=True,**options)
    variables=list(bqm.variables)
    details.update(status='sampler_completed',metadata=samples.info['qaoa_metadata'],
                   sampler_wall_sec=time.perf_counter()-started,
                   observed_counts=[dict(bits=''.join(str(int(r.sample[v])) for v in variables),
                                         energy=float(r.energy),occurrences=int(r.num_occurrences))
                                    for r in samples.aggregate().data()])
    checkpoint(out,details)
    # Scoring gets its own 20-second allowance, under the enclosing 600-second
    # method cap. It cannot replace expensive optimization with uncounted work.
    work=bb.FlowWork(graph,source,sink,B,allowed,seconds=20)
    _,records=bb.sample_records(samples,bqm,graph,B,work)
    details.update(status='raw_scored',raw_samples=records)
    checkpoint(out,details)
    status='completed'
    try:
        for row in records:
            bb.repair(work,row['interdicted_edges'])
    except bb.WorkLimit as error:
        status=str(error)
    result=work.result(status)
    result['scoring_repair_sec']=result.pop('runtime_sec')
    details.update(result,runtime_sec=time.perf_counter()-started)
    checkpoint(out,details)


def run_method(method):
    with tempfile.TemporaryDirectory(prefix='nip-hard-mps-') as directory:
        path=Path(directory)/'worker.json'
        start=time.perf_counter()
        with (Path(directory)/'stderr.log').open('w') as log:
            try:
                process=subprocess.run([sys.executable,str(ROOT/'run_hard_mps_probe.py'),
                                        '--worker',method,'--out',str(path)],
                                       stdout=log,stderr=log,timeout=OUTER_SECONDS)
                outcome='returned' if process.returncode==0 else 'worker_failed'
                code=process.returncode
            except subprocess.TimeoutExpired:
                outcome='outer_time_limit';code=None
        result=json.loads(path.read_text()) if path.exists() else {}
        if outcome!='returned':
            result.update(last_completed_stage=result.get('status'),status=outcome,returncode=code)
        result.update(process_wall_sec=time.perf_counter()-start,outer_seconds=OUTER_SECONDS)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',choices=METHODS)
    args=parser.parse_args(argv)
    if args.worker:
        resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
        worker(args.worker,args.out)
        return
    ref=matched.cases_and_references()['cover_n30_s0',192]
    artifact=dict(schema=1,**matched.provenance(),
                  settings=dict(methods=METHODS,reps=1,maxiter=4,shots=256,seed=0,order='rcm',
                                max_qubits=1200,max_interactions=50000,bond_dimension=8,
                                outer_seconds=OUTER_SECONDS,memory_bytes=8*1024**3,
                                scoring_repair_seconds=20,repetitions=1),
                  reference=dict(artifact=ref['reference'],sha256=ref['reference_sha256'],
                                 source_commit=ref['reference_source_commit'],result=ref['archived']['milp']),
                  case=dict(name='cover_n30_s0',B=192,sha256=ref['case']['sha256']),
                  attempts=[dict(attempt=1,status='running',completed_methods=0)],methods={})
    # Reserve the path exclusively; checkpoints atomically replace our own file.
    with args.out.open('x'):
        pass
    checkpoint(args.out,artifact)
    try:
        for method in METHODS:
            print(method,flush=True)
            artifact['methods'][method]={'status':'dispatched'};checkpoint(args.out,artifact)
            artifact['methods'][method]=run_method(method)
            artifact['attempts'][0]['completed_methods']+=1;checkpoint(args.out,artifact)
            print(artifact['methods'][method]['status'],flush=True)
        artifact['attempts'][0]['status']='completed'
    except Exception as error:
        artifact['attempts'][0].update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        checkpoint(args.out,artifact)


if __name__=='__main__':
    main()

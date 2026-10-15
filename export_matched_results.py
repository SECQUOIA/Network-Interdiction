"""Independently audit retained witnesses and export plotting-compatible tables.

No optimization or hardware calls. Exact squared-BQM ground energy follows from
the certified reduction and the independently checked original MILP optimum.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import networkx as nx

import bounded_benchmarks as bb
import challenge_instances as ci
from certified_penalties import certified_squared_qubo
import classical_baselines as cb
import run_benchmark as rb
import run_matched_benchmark as matched

ROOT = Path(__file__).resolve().parent
REFUSED = {'qubit_guard_refused', 'curve_work_exceeds_allowance', 'worker_failed', 'outer_time_limit'}


def verify_producer(artifact):
    """Bind every executable source hash to its archived producing commit."""
    for path, expected in artifact['source_hashes'].items():
        data = subprocess.check_output(['git','show',f'{artifact["source_commit"]}:{path}'],cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'uncommitted producer: {path}')


def cut_flow(graph, source, sink, allowed, budget, cut, *, require_feasible=True):
    arcs = list(map(tuple,cut))
    if len(set(arcs)) != len(arcs) or not set(arcs) <= allowed:
        raise ValueError('invalid witness arcs')
    if require_feasible and len(arcs) > budget:
        raise ValueError('infeasible budget witness')
    residual = graph.copy();residual.remove_edges_from(arcs)
    return nx.maximum_flow_value(residual,source,sink,capacity='weight')


def audit_samples(result, bqm, graph, source, sink, allowed, budget):
    variables = result.get('variables', [])
    if set(variables) != set(bqm.variables) or len(variables) != len(bqm):
        raise ValueError('sample variable mismatch')
    for row in result['raw_samples']:
        if len(row['bits']) != len(variables) or set(row['bits']) - {'0','1'}:
            raise ValueError('invalid sample bits')
        sample = dict(zip(variables,map(int,row['bits'])))
        cut = [e for i,e in enumerate(sorted(graph.edges())) if sample.get(f'x_{i}',0)]
        if cut != list(map(tuple,row['interdicted_edges'])) or bqm.energy(sample) != row['energy']:
            raise ValueError('sample decode or energy mismatch')
        if row['budget_feasible'] != (len(cut)<=budget) or row['exact_budget'] != (len(cut)==budget):
            raise ValueError('raw feasibility mismatch')
        if row['flow'] is not None and row['flow'] != cut_flow(graph,source,sink,allowed,budget,cut,require_feasible=False):
            raise ValueError('raw flow mismatch')
        if type(row['occurrences']) is not int or row['occurrences'] <= 0:
            raise ValueError('invalid multiplicity')


def audit_case(row, ref):
    loaded = next(c for c in ci.load_cases(ROOT/ref['manifest']) if (c[0]['name'],c[4])==(row['name'],row['B']))
    case,graph,source,sink,B,allowed = loaded
    if (row['sha256'] != case['sha256'] or row['reference_sha256'] != ref['reference_sha256'] or
            row['reference_source_commit'] != ref['reference_source_commit']):
        raise ValueError('stale matched reference')
    exact = ref['archived']['milp']
    if exact['status'] != 'optimal' or exact['flow'] != cut_flow(graph,source,sink,allowed,B,exact['interdicted_edges']):
        raise ValueError('invalid exact reference')
    bqm, certificate = certified_squared_qubo(B,graph,source,sink,allowed)
    for name,result in row['methods'].items():
        if name == 'fit':
            continue
        if 'flow' in result:
            if result['flow'] != cut_flow(graph,source,sink,allowed,B,result['interdicted_edges']):
                raise ValueError('method flow mismatch')
            if result['flow'] < exact['flow'] or result['max_flow_calls'] > result['max_flow_call_limit']:
                raise ValueError('invalid bound or work count')
        if 'lower_bound' in result and result['lower_bound'] > exact['flow']:
            raise ValueError('invalid LP lower bound')
        if 'reconstructed_cut' in result:
            if result['reconstructed_flow'] != cut_flow(graph,source,sink,allowed,B,result['reconstructed_cut']):
                raise ValueError('reconstruction mismatch')
        if 'child_calls' in result and len(result['child_calls']) > result['max_child_calls']:
            raise ValueError('child work limit exceeded')
        if result.get('allocation'):
            curves = result['curves'];allocation=result['allocation']
            optimum = min(sum(curves[i][k] for i,k in enumerate(ks))
                          for ks in itertools.product(*(range(len(c)) for c in curves)) if sum(ks)<=B)
            value = sum(c[k] for c,k in zip(curves,allocation['budgets']))
            if not allocation['surrogate_lower_bound'] <= optimum <= value <= allocation['surrogate_value']:
                raise ValueError('invalid surrogate bound')
        if name == 'sa' and result.get('completed_reads',0):
            if result['certificate'] != asdict(certificate):
                raise ValueError('penalty certificate mismatch')
            audit_samples(result,bqm,graph,source,sink,allowed,B)
            if sum(r['occurrences'] for r in result['raw_samples']) > result['completed_reads']:
                raise ValueError('selected multiplicity exceeds sampler work')
    fit = row['methods']['fit']
    if fit['status']=='completed':
        import decomposition as dc
        from profile_embedding import interaction_graph, topology, audit_embedding
        pieces=list(map(set,fit['pieces']));target=topology('P16')
        for i, measured in enumerate(fit['rows']):
            model=bqm
            if i:
                sub=bb.restricted_subproblem(graph,source,sink,pieces,i-1,allowed)
                model,_=certified_squared_qubo(measured['B'],sub.graph,sub.src,sub.tgt,sub.interdictable)
            if len(model)!=measured['qubits'] or model.num_interactions!=measured['couplings']:
                raise ValueError('fit model mismatch')
            embedding=measured['embedding']
            if embedding['status']=='embedded':
                check=audit_embedding(interaction_graph(model),target,
                                      {r['variable']:r['chain'] for r in embedding['embedding']})
                if any(check[key]!=embedding[key] for key in check):
                    raise ValueError('embedding certificate mismatch')
        if fit['all_pieces_embedded'] != all(r['embedding']['status']=='embedded' for r in fit['rows'][1:]):
            raise ValueError('piece-fit aggregation mismatch')
    return loaded,bqm,certificate


def export_row(row,ref,loaded,bqm,certificate):
    case,g,s,t,B,allowed=loaded
    methods=row['methods'];exact=ref['archived']['milp'];bend=ref['archived']['benders']
    output={key:'' for key in rb.CSV_COLUMNS}
    output.update(instance=str(Path(ref['manifest']).parent/case['file']),V=len(g),E=g.number_of_edges(),B=B,
                  qubits=len(bqm),family=case['family'],lambda_cardinality=certificate.cardinality,
                  dual_penalty=certificate.dual,budget_penalty='squared',
                  maxflow=cut_flow(g,s,t,allowed,B,[]),Opt_NIP=exact['flow'],
                  Opt_QUBO=exact['flow']-certificate.normalized_offset,
                  reference_source_commit=ref['reference_source_commit'],reference_artifact=ref['reference'],
                  Benders=bend['flow'],Benders_lower_bound=bend['lower_bound'],Benders_status=bend['status'],
                  Benders_time_sec=bend['runtime_sec'],Gurobi_total_time_sec=exact['runtime_sec'])
    for method,column in [('greedy','Greedy'),('lp','LP_Round_Repair'),('sa','SA_NIP_Greedy'),*rb.DECOMPOSITION_COLUMNS.items()]:
        result=methods[method]
        output[column+'_status']=result['status']
        if result['status'] not in REFUSED and 'flow' in result:
            output[column]=result['flow']
        output[column+'_total_time_sec']=result.get('runtime_sec','')
    lp=methods['lp']
    if 'lower_bound' in lp:
        output.update(LP_lower_bound=lp['lower_bound'],LP_objective=lp['lp_objective'],
                      LP_time_sec=lp['lp_time_sec'],LP_variant=lp['variant'],
                      LP_Round_Repair_gap=cb.relative_gap(lp['flow'],lp['lower_bound']))
    sa=methods['sa'];raw=sa.get('raw_samples',[])
    feasible=[r['flow'] for r in raw if r['budget_feasible'] and r['flow'] is not None]
    output.update(SA_NIP=min(feasible) if feasible else '',SA_QUBO=min((r['energy'] for r in raw),default=''),
                  SA_raw_feasible_count=sum(r['budget_feasible'] for r in raw),SA_raw_samples_examined=len(raw),
                  SA_raw_samples=json.dumps(raw,separators=(',',':')),SA_time_sec=sa.get('sample_time_sec',''),
                  SA_total_time_sec=sa.get('runtime_sec',''),SA_completed_reads=sa.get('completed_reads',''))
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/matched_scaling.json')
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args(argv)
    artifact=json.loads(args.input.read_text());verify_producer(artifact)
    refs=matched.cases_and_references();groups={}
    if [(r['name'],r['B']) for r in artifact['cases']] != matched.SELECTION:
        raise ValueError('incomplete campaign')
    for row in artifact['cases']:
        ref=refs[row['name'],row['B']]
        loaded,bqm,certificate=audit_case(row,ref)
        groups.setdefault(loaded[0]['family'],[]).append(export_row(row,ref,loaded,bqm,certificate))
    args.outdir.mkdir(parents=True,exist_ok=True)
    for family,rows in groups.items():
        columns=list(dict.fromkeys(key for row in rows for key in row))
        with (args.outdir/f'{family}.csv').open('x',newline='') as handle:
            handle.write(f'# Derived from {args.input.name} sha256={ci.digest(args.input)}; source={artifact["source_commit"]}\n')
            handle.write('# Original MILP/Benders timings retained; bounded methods include setup/audits; hardware columns intentionally empty.\n')
            handle.write('# Data terms and attribution: testcases/challenge/README.md; SocioPatterns derivatives retain CC BY-NC-SA 3.0.\n')
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    print(f'Audited and exported {len(artifact["cases"])} cases in {len(groups)} families')


def audit_simulators(artifact):
    """Audit all observed bitstrings without rerunning a stochastic optimizer."""
    import run_simulator_scaling as scaling
    import network_interdiction as ni
    summary=[]
    if [(r['file'],r['B']) for r in artifact['cases']] != scaling.CASES:
        raise ValueError('incomplete simulator campaign')
    for case in artifact['cases']:
        path,B=case['file'],case['B']
        if case['sha256']!=ci.digest(ROOT/path):
            raise ValueError('stale simulator input')
        g,s,t,allowed=scaling.inputs(path,B)
        reference=case['methods']['reference']
        if 'result' not in reference:
            raise ValueError('missing simulator reference')
        exact=reference['result']
        if exact['status']!='optimal' or exact['flow']!=cut_flow(g,s,t,allowed,B,exact['interdicted_edges']):
            raise ValueError('invalid simulator reference')
        empty=cut_flow(g,s,t,allowed,B,[])
        penalty,certificate=certified_squared_qubo(B,g,s,t,allowed)
        for method,result in case['methods'].items():
            if method=='reference':
                continue
            if 'flow' in result:
                if result['flow'] < exact['flow'] or result['flow']!=cut_flow(g,s,t,allowed,B,result['interdicted_edges']):
                    raise ValueError('invalid simulator feasible result')
            if result['status'] in REFUSED:
                summary.append(dict(file=path,method=method,status=result['status'],meaningful=False))
                continue
            if not result.get('variables'):
                continue
            bqm=penalty if not method.startswith('xy_') else ni.nip_to_qubo(B,g,s,t,0,certificate.dual,allowed)
            audit_samples(result,bqm,g,s,t,allowed,B)
            if method=='sa':
                continue
            exact_shots=feasible_shots=total=0
            variables=result['variables'];edges=sorted(g.edges())
            for observed in result['observed_counts']:
                if len(observed['bits'])!=len(variables) or set(observed['bits'])-{'0','1'}:
                    raise ValueError('invalid observed bits')
                sample=dict(zip(variables,map(int,observed['bits'])))
                if bqm.energy(sample)!=observed['energy']:
                    raise ValueError('observed energy mismatch')
                n=sum(sample.get(f'x_{i}',0) for i,e in enumerate(edges) if e in allowed)
                count=observed['occurrences']
                if type(count) is not int or count<=0:
                    raise ValueError('invalid observed multiplicity')
                total+=count;exact_shots+=count*(n==B);feasible_shots+=count*(n<=B)
            if total!=256 or exact_shots!=result['raw_exact_budget_shots'] or feasible_shots!=result['raw_at_most_budget_shots']:
                raise ValueError('shot feasibility mismatch')
            if result['basis_initial_flow']!=cut_flow(g,s,t,allowed,B,result['basis_initial_cut']):
                raise ValueError('initialization baseline mismatch')
            meaningful=any(r['budget_feasible'] and r['flow'] is not None and r['flow']<empty for r in result['raw_samples'])
            summary.append(dict(file=path,method=method,qubits=len(bqm),status=result['status'],
                                meaningful=meaningful,exact_budget_shots=exact_shots,
                                raw_feasible_shots=feasible_shots))
    return summary


if __name__=='__main__':
    main()

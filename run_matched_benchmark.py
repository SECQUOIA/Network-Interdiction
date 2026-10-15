"""Bounded, checkpointed software comparisons with archived exact references.

No hardware calls. Every method runs in a separate capped process. Run from a
committed source tree; source hashes and original reference producers are kept.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import challenge_instances as ci

ROOT = Path(__file__).resolve().parent
SELECTION = [('ema', 3), ('school4a', 3), ('ieee14', 2),
             ('50.planarflow', 6), ('50_0.15.erdosflow', 6), ('50_5.2dnnflow', 6),
             ('cover_n12_s0', 20), ('cover_n30_s0', 192), ('150.planarflow', 6)]
METHODS = ['greedy', 'lp', 'sa'] + [f'{rule}_{mode}' for rule in
           ('mincut', 'local', 'marginal', 'multiplier') for mode in ('sequential', 'iterated')] + ['fit']
PAIRS = [('testcases/challenge/manifest.json', 'results/challenge_profile.json'),
         ('testcases/challenge/legacy_manifest.json', 'results/challenge_legacy_profile.json')]


def cases_and_references():
    """Bind archived references to exact inputs, allowed sets and budgets."""
    result = {}
    for manifest_name, reference_name in PAIRS:
        manifest, reference = ROOT / manifest_name, ROOT / reference_name
        artifact = json.loads(reference.read_text())
        if artifact['manifest_sha256'] != ci.digest(manifest):
            raise ValueError('stale reference manifest')
        if artifact['attempts'][-1]['status'] != 'completed':
            raise ValueError('incomplete reference campaign')
        indexed = {(row['name'], row['B']): row for row in artifact['cases']}
        for case, graph, source, sink, budget, allowed in ci.load_cases(manifest):
            key = case['name'], budget
            row = indexed[key]
            if (row['sha256'] != case['sha256'] or row['source'] != source or row['sink'] != sink
                    or row['allowed_arcs'] != len(allowed) or row['nodes'] != len(graph)
                    or row['arcs'] != graph.number_of_edges()):
                raise ValueError('stale reference instance')
            # The manifest hash binds the complete allowed set, not only its size.
            for method in ('milp', 'benders'):
                cut = set(map(tuple, row[method]['interdicted_edges']))
                if len(cut) > budget or not cut <= allowed:
                    raise ValueError('invalid archived witness')
            result[key] = dict(manifest=manifest_name, reference=reference_name,
                               reference_sha256=ci.digest(reference),
                               reference_source_commit=artifact['source_commit'],
                               archived=row, case=case)
    return result


def provenance():
    files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob('*.py'))
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_hashes={p: ci.digest(ROOT/p) for p in files},
                python=platform.python_version(), machine=platform.platform(),
                cpu_model=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                if line.startswith('model name')), 'unknown'),
                versions={p: importlib.metadata.version(p) for p in
                          ['gurobipy', 'scipy', 'numpy', 'networkx', 'dimod', 'pyqubo',
                           'dwave-samplers', 'minorminer', 'qiskit', 'qiskit-aer']},
                thread_environment={p: os.environ.get(p) for p in
                                    ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})


def _embedding_child(connection, source, target):
    from profile_embedding import embedding_attempt
    try:
        connection.send(embedding_attempt(source, target, seed=0, timeout=3))
    except Exception as error:
        connection.send(dict(status='embedding_failed', error=f'{type(error).__name__}: {error}'))
    finally:
        connection.close()


def hard_embedding_attempt(source, target, *, seconds=12):
    """Contain minorminer setup, which can exceed its cooperative timeout.

    This campaign already requires POSIX resource limits. Forking a single
    worker avoids serializing a dense interaction graph before the deadline.
    The pipe is drained before join, including large successful certificates.
    """
    import multiprocessing
    from profile_embedding import graph_digest
    if not 0 < seconds <= 30:
        raise ValueError('embedding outer limit must be in (0,30]')
    context=multiprocessing.get_context('fork')
    reader,writer=context.Pipe(duplex=False)
    process=context.Process(target=_embedding_child,args=(writer,source,target))
    start=time.perf_counter()
    process.start();writer.close()
    try:
        if reader.poll(seconds):
            try:
                result=reader.recv()
            except EOFError:
                result=dict(status='embedding_worker_failed')
        else:
            result=dict(status='embedding_outer_time_limit')
    finally:
        if process.is_alive():
            process.terminate()
        process.join();reader.close()
    result.update(process_wall_sec=time.perf_counter()-start,outer_seconds=seconds,
                  logical_variables=len(source),logical_interactions=source.number_of_edges(),
                  source_graph_sha256=graph_digest(source),target_graph_sha256=graph_digest(target))
    return result


def fit(graph, source, sink, budget, allowed):
    """Necessary logical guards plus actual ideal-P16 embedding attempts."""
    import bounded_benchmarks as bb
    import decomposition as dc
    from certified_penalties import certified_squared_qubo
    from profile_embedding import topology, interaction_graph
    from solve_qubo_via_qci import prepare_request
    start = time.perf_counter()
    pieces = dc.partition_graph(graph, min(4, len(graph)), seed=0)
    capacities = [sum(dc.arc_owners(graph, pieces)[e] == i for e in allowed) for i in range(len(pieces))]
    budgets = bb.capped_split(budget, capacities, capacities)
    target = topology('P16')
    rows = []
    inputs = [('whole', graph, source, sink, budget, allowed)]
    for i in range(len(pieces)):
        sub = bb.restricted_subproblem(graph, source, sink, pieces, i, allowed)
        inputs.append((f'piece_{i}', sub.graph, sub.src, sub.tgt, min(budgets[i],len(sub.interdictable)), sub.interdictable))
    for name, g, s, t, B, arcs in inputs:
        before = time.perf_counter()
        bqm, certificate = certified_squared_qubo(B, g, s, t, arcs)
        row = dict(name=name, nodes=len(g), arcs=g.number_of_edges(), B=B, allowed_arcs=len(arcs),
                   qubits=len(bqm), couplings=bqm.num_interactions, certificate=asdict(certificate),
                   compile_time_sec=time.perf_counter()-before,
                   statevector_size_guard=1 <= len(bqm) <= 24, mps_size_guard=1 <= len(bqm) <= 256)
        try:
            _, request = prepare_request(bqm)
            row['qci_schema'] = {'status': 'accepted_offline', 'optical_levels': sum(request['job_params']['num_levels'])}
        except ValueError as error:
            row['qci_schema'] = {'status': 'refused', 'reason': str(error)}
        row['embedding'] = hard_embedding_attempt(interaction_graph(bqm), target)
        rows.append(row)
    return dict(status='completed', runtime_sec=time.perf_counter()-start,
                pieces=[sorted(p) for p in pieces], budgets=budgets, rows=rows,
                max_piece_qubits=max(r['qubits'] for r in rows[1:]),
                all_pieces_embedded=all(r['embedding']['status']=='embedded' for r in rows[1:]))


def worker(manifest, name, budget, method):
    import bounded_benchmarks as bb
    cases = ci.load_cases(ROOT / manifest)
    matches = [c for c in cases if c[0]['name'] == name and c[4] == budget]
    if len(matches) != 1 or method not in METHODS:
        raise ValueError('unknown worker input')
    _, graph, source, sink, B, allowed = matches[0]
    if method == 'fit':
        return fit(graph, source, sink, B, allowed)
    if method in ('greedy', 'lp', 'sa'):
        return getattr(bb, 'anneal' if method == 'sa' else method)(graph, source, sink, B, allowed)
    rule, mode = method.split('_')
    return bb.decomposed(graph, source, sink, B, allowed, rule=rule, mode=mode)


def process_call(arguments, *, timeout=120, script='run_matched_benchmark.py'):
    """Return explicit failure status; never turn a timeout into a solved row."""
    with tempfile.TemporaryDirectory(prefix='nip-matched-') as directory:
        output = Path(directory)/'worker.json'
        log = Path(directory)/'worker.log'
        start = time.perf_counter()
        with log.open('w') as handle:
            try:
                process = subprocess.run([sys.executable, str(ROOT/script),
                                          '--worker', *arguments, '--out', str(output)],
                                         stdout=handle, stderr=handle, timeout=timeout)
            except subprocess.TimeoutExpired:
                return dict(status='outer_time_limit', process_wall_sec=time.perf_counter()-start,
                            outer_time_limit_sec=timeout)
        if process.returncode != 0:
            # Error diagnostics only; licensed solver stdout is not published.
            lines = log.read_text().splitlines()
            return dict(status='worker_failed', returncode=process.returncode,
                        process_wall_sec=time.perf_counter()-start,
                        error=lines[-1] if lines else 'no diagnostic')
        result = json.loads(output.read_text())
        result.update(process_wall_sec=time.perf_counter()-start, outer_time_limit_sec=timeout)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', nargs=4, metavar=('MANIFEST','NAME','B','METHOD'))
    parser.add_argument('--fit-only', choices=[name for name,_ in SELECTION],
                        help='separate retained fit attempt for one previously selected case')
    args = parser.parse_args(argv)
    if args.worker:
        # Hard process memory cap; native calls also remain subject to parent timeout.
        resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
        manifest,name,budget,method = args.worker
        result = worker(manifest, name, int(budget), method)
        with args.out.open('x') as handle:
            json.dump(result, handle, allow_nan=False)
        return
    refs = cases_and_references()  # All validation precedes optimization/output creation.
    selection = [key for key in SELECTION if not args.fit_only or key[0] == args.fit_only]
    methods = ['fit'] if args.fit_only else METHODS
    rows = []
    artifact = dict(schema=1, **provenance(),
                    settings=dict(selection=selection, methods=methods, seconds_per_method=20,
                                  max_flow_calls=5000, max_child_calls=128, child_seconds=5,
                                  parts=4, max_terminals=4, max_rounds=1, seed=0,
                                  reads=200, sweeps=1000, outer_seconds=120, memory_bytes=8*1024**3,
                                  embedding_seconds=3, embedding_outer_seconds=12, embedding_target='ideal P16', repetitions=1),
                    attempts=[dict(attempt=1,status='running',completed_methods=0)], cases=rows)
    with args.out.open('x') as handle:
        def save():
            handle.seek(0); json.dump(artifact, handle, indent=2, allow_nan=False)
            handle.write('\n'); handle.truncate(); handle.flush()
        save()
        try:
            for name,budget in selection:
                ref = refs[name,budget]
                row = dict(name=name,B=budget,manifest=ref['manifest'],
                           reference=ref['reference'],reference_sha256=ref['reference_sha256'],
                           reference_source_commit=ref['reference_source_commit'],
                           sha256=ref['case']['sha256'], methods={})
                rows.append(row); save()
                for method in methods:
                    print(name, budget, method, flush=True)
                    row['methods'][method] = dict(status='dispatched')
                    save()
                    result = process_call([ref['manifest'],name,str(budget),method])
                    row['methods'][method] = result
                    artifact['attempts'][0]['completed_methods'] += 1
                    save()
                    print(result['status'], result.get('flow',result.get('max_piece_qubits')), flush=True)
            artifact['attempts'][0]['status'] = 'completed'
        except Exception as error:
            artifact['attempts'][0].update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save()


if __name__ == '__main__':
    main()

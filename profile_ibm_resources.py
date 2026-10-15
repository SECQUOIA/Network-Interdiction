"""Offline penalty-QAOA routing on bundled IBM snapshots; no sampling/service.

Provide explicit path:budget:lambda cases. Results are fitting observations for
this bounded cohort and these historical snapshots, not current QPU capability.
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
import sys
import time

import network_interdiction as ni
from qaoa_ibm import prepare_circuit, validate_options
from run_benchmark import parse_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', type=parse_case, required=True)
    parser.add_argument('--snapshot', action='append', choices=['sherbrooke', 'torino'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-depth', type=int, default=10000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if any(lam == 'auto' for _, _, lam in args.case):
        parser.error('offline fitting requires an explicit lambda; no exact reference is solved')
    if any(budget < 0 or not math.isfinite(lam) or lam <= 0 for _, budget, lam in args.case):
        parser.error('budgets must be nonnegative and lambda finite and positive')
    try:
        validate_options(theta=[0.25, 0.5], seed_transpiler=args.seed, max_depth=args.max_depth)
    except ValueError as error:
        parser.error(str(error))
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino
    classes = {'sherbrooke': FakeSherbrooke, 'torino': FakeTorino}
    report = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                         ('profile_ibm_resources.py', 'qaoa_ibm.py', 'qaoa_common.py',
                          'qaoa_qiskit.py', 'solve_qubo_via_qaoa.py', 'network_interdiction.py',
                          'run_benchmark.py')},
        'argv': list(argv) if argv is not None else sys.argv[1:],
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('qiskit', 'qiskit-aer', 'qiskit-ibm-runtime', 'numpy', 'networkx', 'pyqubo')},
        'threads': {name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'scope': 'Offline p=1 routing only, beta=.25 gamma=.5; no optimization, '
                 'sampling, account, live calibration, or QPU. Width/depth fit is not solution quality.',
        'snapshots': {}, 'rows': [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.snapshot or ['sherbrooke', 'torino']:
        backend = classes[name]()
        report['snapshots'][name] = {
            'backend': backend.name, 'qubits': backend.num_qubits,
            'file_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(Path(backend.dirname).glob('*.json'))},
        }
        for instance, budget, lam in args.case:
            start = time.perf_counter()
            graph, source, sink = ni.read_flow_file(instance)
            if budget > len(graph.edges):
                raise ValueError('budget exceeds the number of edges')
            bqm = ni.nip_to_qubo(budget, graph, source, sink, lam, lam+1)
            row = dict(instance=str(instance), input_sha256=hashlib.sha256(Path(instance).read_bytes()).hexdigest(),
                       snapshot=name, budget=budget, cardinality_penalty=lam, dual_penalty=lam+1,
                       num_qubits=len(bqm), interactions=len(bqm.quadratic),
                       qpu_time_sec=None, execution_wall_time_sec=None)
            if len(bqm) > backend.num_qubits:
                row.update(status='too_wide', target_qubits=backend.num_qubits)
            else:
                circuit, _, meta = prepare_circuit(bqm, backend, theta=[0.25, 0.5],
                    seed_transpiler=args.seed, max_qubits=backend.num_qubits, max_depth=args.max_depth)
                # Audit the actual target instructions, including directionality.
                for item in circuit.data:
                    if item.operation.name != 'barrier' and not backend.target.instruction_supported(
                            operation_name=item.operation.name,
                            qargs=tuple(circuit.find_bit(q).index for q in item.qubits)):
                        raise RuntimeError('transpiled circuit contains an unsupported target instruction')
                row.update(status='transpiled', **meta)
            row['preparation_wall_time_sec'] = time.perf_counter()-start
            report['rows'].append(row)
            args.out.write_text(json.dumps(report, indent=2)+'\n')
            print(f"{name} {instance}: {row['num_qubits']} qubits, {row['status']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

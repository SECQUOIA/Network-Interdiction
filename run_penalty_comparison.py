"""Bounded exact comparison for certified squared-budget penalties; no hardware."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import gurobipy as gp
import networkx as nx

from certified_penalties import certified_squared_qubo
import network_interdiction as ni

MAX_VARIABLES = 160
MAX_EDGES = 50
TIME_LIMIT = 20
SOURCE_FILES = ('run_penalty_comparison.py', 'certified_penalties.py',
                'network_interdiction.py', 'run_benchmark.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optimal_status(model):
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f'exact reference did not finish: status={model.Status}')


def run_case(path, B):
    g, s, t = ni.read_flow_file(path)
    # The compiler allocates at most three variables per arc plus free pi.
    if len(g.edges()) > MAX_EDGES or 3*len(g.edges()) + len(g)-2 > MAX_VARIABLES:
        raise ValueError('case exceeds the bounded comparison size')
    start = time.perf_counter()
    bqm, cert = certified_squared_qubo(B, g, s, t)
    build_seconds = time.perf_counter() - start
    if bqm.num_variables > MAX_VARIABLES:
        raise ValueError('compiled model exceeds the bounded comparison size')
    # Comparing nonzero structure at identical coefficients isolates the
    # squared budget clique. The linear coefficients here are not tuned.
    linear = ni.nip_to_qubo(B, g, s, t, cert.cardinality, cert.dual)
    with ni._gurobi_model() as model:
        model.Params.TimeLimit = TIME_LIMIT
        model.Params.MIPGap = 0
        x = {v: model.addVar(vtype=gp.GRB.BINARY, name=v) for v in bqm.variables}
        model.setObjective(bqm.offset + gp.quicksum(c*x[v] for v, c in bqm.linear.items())
                           + gp.quicksum(c*x[u]*x[v] for (u, v), c in bqm.quadratic.items()))
        start = time.perf_counter()
        model.optimize()
        solve_seconds = time.perf_counter() - start
        optimal_status(model)
        sample = {v: int(round(var.X)) for v, var in x.items()}
        energy, bound = float(model.ObjVal), float(model.ObjBound)
    if abs(energy - bqm.energy(sample)) > 1e-6:
        raise AssertionError('solver energy and decoded binary energy differ')
    edges = sorted(g.edges())
    selected = [e for i, e in enumerate(edges) if sample[f'x_{i}']]
    pi = {v: 1 if v == s else 0 if v == t else sample[f'pi_{v}'] for v in g}
    q = sum(sample[f'w_{i}']*(pi[u]+pi[v]+sample[f'gamma_{i}']-1)
            + pi[v]*sample[f'gamma_{i}']-pi[v]-sample[f'gamma_{i}']+1
            for i, (u, v) in enumerate(edges))
    if len(selected) != B or q != 0:
        raise AssertionError('certified ground state violates an encoded constraint')
    residual = g.copy()
    residual.remove_edges_from(selected)
    flow = nx.maximum_flow_value(residual, s, t, capacity='weight')
    milp, _ = ni._build_nip_milp(g, s, t, B)
    with milp:
        milp.Params.TimeLimit = TIME_LIMIT
        milp.Params.MIPGap = 0
        start = time.perf_counter()
        milp.optimize()
        milp_seconds = time.perf_counter() - start
        optimal_status(milp)
        optimum, milp_bound = float(milp.ObjVal), float(milp.ObjBound)
    if abs(flow - optimum) > 1e-6 or abs(energy + cert.normalized_offset - optimum) > 1e-6:
        raise AssertionError('squared QUBO differs from the original NIP optimum')
    window = None
    if len(edges) <= 8:  # at most 256 max-flow evaluations before enumeration
        from run_benchmark import lambda_window
        curve = []
        for k in range(len(edges)+1):
            values = []
            for cut in itertools.combinations(edges, k):
                r = g.copy()
                r.remove_edges_from(cut)
                values.append(nx.maximum_flow_value(r, s, t, capacity='weight'))
            curve.append(min(values))
        lo, hi = lambda_window(curve, B)
        window = {'curve': curve, 'lo': lo if math.isfinite(lo) else None,
                  'hi': hi if math.isfinite(hi) else None, 'strict_window': lo < hi}
    return {'instance': str(path), 'input_sha256': digest(path), 'budget': B,
            'certificate': asdict(cert), 'variables': len(bqm),
            'linear_couplers': int(sum(c != 0 for c in linear.quadratic.values())),
            'squared_couplers': int(sum(c != 0 for c in bqm.quadratic.values())),
            'build_seconds': build_seconds, 'qubo_seconds': solve_seconds,
            'milp_seconds': milp_seconds, 'qubo_energy': energy, 'qubo_bound': bound,
            'nip_optimum': optimum, 'nip_bound': milp_bound, 'decoded_flow': flow,
            'interdicted_edges': selected, 'sample': sample, 'dual_violation': q,
            'linear_window': window}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, metavar='PATH:B')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= len(args.case) <= 12:
        parser.error('provide 1 to 12 cases')
    if args.out.exists():
        parser.error('output exists; use a new path to preserve earlier attempts')
    cases = [(p, int(b)) for p, b in (c.rsplit(':', 1) for c in args.case)]
    for path, _ in cases:
        Path(path).read_bytes()  # validate all inputs before writing anything
    payload = {'schema': 1, 'producer_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {p: digest(p) for p in SOURCE_FILES},
        'versions': {p: importlib.metadata.version(p) for p in
                     ('gurobipy', 'networkx', 'pyqubo', 'dimod', 'numpy')},
        'limits': {'max_cases': 12, 'max_variables': MAX_VARIABLES,
                   'max_edges': MAX_EDGES, 'seconds_per_solve': TIME_LIMIT,
                   'gurobi_threads': ni.DEFAULT_GUROBI_THREADS, 'mip_gap': 0},
        'rows': [], 'failures': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, B in cases:
        try:
            payload['rows'].append(run_case(path, B))
        except Exception as error:
            payload['failures'].append({'instance': path, 'budget': B,
                                        'error': f'{type(error).__name__}: {error}'})
            raise
        finally:
            args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

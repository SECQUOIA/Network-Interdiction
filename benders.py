"""Max-flow interdiction by a binary master and max-flow cuts (issue #15).

For any feasible flow f of value v in the original network, the surviving
flow F(x) satisfies F(x) >= v - sum_e f_e*x_e. Indeed, every s-t cut's
surviving capacity is at least its surviving forward flow, which is at
least v minus the flow removed over all arcs. Taking the minimum cut
proves the inequality. An optimal residual flow has f_e=0 on the current
interdictions, so its inequality is tight there. Complete master solves
and these cuts therefore converge on the finite binary interdiction set.

This is a max-flow cutting-plane/Benders implementation, not a reproduction
of Israeli and Wood's (2002) shortest-path algorithm or its supervalid cuts.
Capacities are nonnegative integers, as in the repository's .flow format,
so NetworkX flow/cut coefficients use integer arithmetic. Master bounds and
optimality are subject to Gurobi's numerical tolerances.
"""
import math
import numbers
import time

import gurobipy as gp
import networkx as nx

import classical_baselines as cb
import network_interdiction as ni


def flow_cut(graph, source, sink, interdicted_edges=()):
    """Return an integer flow value and arc coefficients, zero on removed arcs.

    The caller validates the integer-capacity network. This helper solves
    the actual max-flow subproblem; it does not invoke an interdiction MILP.
    """
    residual = graph.copy()
    residual.remove_edges_from(interdicted_edges)
    value, flow = nx.maximum_flow(residual, source, sink, capacity='weight')
    coefficients = {edge: flow.get(edge[0], {}).get(edge[1], 0)
                    for edge in sorted(graph.edges())}
    return value, coefficients


def solve(graph, source, sink, budget, *, time_limit=60.0,
          max_iterations=1000, record_cuts=False, interdictable=None):
    """Return a feasible cut, lower bound, history and honest stopping status.

    ``interdictable`` restricts removable arcs, as in the monolithic model.
    One wall-clock allowance covers construction, every master solve and
    max-flow subproblem. NetworkX calls are cooperative: a call in progress
    can overrun the deadline, after which no further master is dispatched.
    An empty interdiction supplies a feasible incumbent even if the master
    stops without a solution. `optimal` requires closing the bound gap;
    otherwise status is `time_limit`, `iteration_limit` or `interrupted`.
    `record_cuts` retains full generated coefficients for small audits only.
    """
    cb._validate(graph, source, sink, budget)
    if (not math.isfinite(time_limit) or time_limit <= 0
            or isinstance(time_limit, bool)):
        raise ValueError('time_limit must be positive and finite')
    if (not isinstance(max_iterations, numbers.Integral)
            or isinstance(max_iterations, bool) or max_iterations < 1):
        raise ValueError('max_iterations must be a positive integer')
    capacities = [data['weight'] for _, _, data in graph.edges(data=True)]
    if any(not isinstance(c, numbers.Integral) or isinstance(c, bool)
           for c in capacities):
        raise ValueError('Benders requires integer capacities (.flow format)')
    if sum(map(int, capacities)) > 2**53:
        raise ValueError('total capacity exceeds the exact integer export range')

    allowed = set(graph.edges()) if interdictable is None else set(interdictable)
    if allowed - set(graph.edges()):
        raise ValueError('interdictable arcs must belong to the graph')
    if budget > len(allowed):
        raise ValueError('budget exceeds the number of interdictable arcs')

    start = time.perf_counter()
    deadline = start + time_limit
    upper, initial = flow_cut(graph, source, sink)
    best_cut = []
    lower = 0.0
    history, cut_records = [], []
    cut_count = 0
    master_calls = 0
    master_time = 0.0
    node_count = 0.0
    subproblem_time = time.perf_counter() - start
    status = 'iteration_limit'

    def result():
        return {'flow': upper, 'interdicted_edges': best_cut,
                'lower_bound': lower, 'status': status,
                'iterations': master_calls, 'cut_count': cut_count,
                'runtime_sec': time.perf_counter() - start,
                'master_time_sec': master_time, 'node_count': node_count,
                'subproblem_time_sec': subproblem_time,
                'history': history, 'cuts': cut_records}

    if budget == 0 or upper == 0:
        lower, status = float(upper), 'optimal'
        return result()
    if time.perf_counter() >= deadline:
        status = 'time_limit'
        return result()

    edges = sorted(graph.edges())
    with ni._gurobi_model() as model:
        # Each master is solved to zero requested MIP gap when time permits.
        model.Params.Seed = 0
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 0
        x = model.addVars(edges, vtype=gp.GRB.BINARY, name='x')
        for edge in edges:
            if edge not in allowed:
                x[edge].UB = 0
        eta = model.addVar(lb=0, name='surviving_flow')
        model.addConstr(gp.quicksum(x[e] for e in edges) <= budget)
        model.setObjective(eta, gp.GRB.MINIMIZE)

        def add_cut(value, coefficients):
            nonlocal cut_count
            model.addConstr(eta >= value - gp.quicksum(
                int(coefficients[e]) * x[e] for e in edges if coefficients[e]))
            cut_count += 1
            if record_cuts:
                cut_records.append({'value': value,
                                    'coefficients': dict(coefficients)})

        add_cut(upper, initial)
        for iteration in range(1, max_iterations + 1):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                status = 'time_limit'
                break
            model.Params.TimeLimit = remaining
            before = time.perf_counter()
            master_calls += 1
            model.optimize()
            master_time += time.perf_counter() - before
            node_count += float(model.NodeCount)
            if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT,
                                     gp.GRB.INTERRUPTED):
                raise RuntimeError(f'Benders master failed: status {model.Status}')
            if math.isfinite(model.ObjBound):
                lower = max(lower, float(model.ObjBound))
            if model.Status == gp.GRB.INTERRUPTED or not model.SolCount:
                status = ('interrupted' if model.Status == gp.GRB.INTERRUPTED
                          else 'time_limit')
                break
            if time.perf_counter() >= deadline:
                status = 'time_limit'
                break
            current = [e for e in edges if x[e].X > 0.5]
            if len(current) > budget:
                raise RuntimeError('Benders master returned an over-budget cut')
            before = time.perf_counter()
            value, coefficients = flow_cut(graph, source, sink, current)
            subproblem_time += time.perf_counter() - before
            if value < upper:
                upper, best_cut = value, current
            # Do not conceal a material contradiction between bound and flow.
            if lower > upper + 1e-7:
                raise RuntimeError('Benders lower bound exceeds feasible flow')
            history.append({'iteration': iteration, 'lower_bound': lower,
                            'flow': value, 'best_flow': upper,
                            'time_sec': time.perf_counter() - start,
                            'master_status': model.Status, 'master_nodes': float(model.NodeCount)})
            if upper - lower <= 1e-7:
                status = 'optimal'
                break
            if time.perf_counter() >= deadline:
                status = 'time_limit'
                break
            add_cut(value, coefficients)
    return result()

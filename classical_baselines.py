"""LP round-and-repair baseline and objective-gap certificates (issue #15)."""
from fractions import Fraction
import math
import numbers
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import network_interdiction as ni


def _validate(graph, source, sink, budget):
    if not graph.is_directed() or graph.is_multigraph():
        raise ValueError('expected a simple directed flow network')
    if source == sink or source not in graph or sink not in graph:
        raise ValueError('source and sink must be distinct graph vertices')
    if (not isinstance(budget, numbers.Integral) or isinstance(budget, bool)
            or not 0 <= budget <= graph.number_of_edges()):
        raise ValueError('budget must be an integer between zero and |E|')
    for _, _, data in graph.edges(data=True):
        capacity = float(data['weight'])
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError('capacities must be finite and nonnegative')


def _lp_model(graph, source, sink, budget, interdictable=None):
    """Continuous min-cut/interdiction formulation, ordered as x, z, pi.

    min sum c_e z_e; pi_u-pi_v <= x_e+z_e; sum x_e <= B;
    pi_s=1, pi_t=0; all variables in [0,1]. With binary x and pi,
    z records each surviving forward cut arc. This is the projection of
    the McCormick linearization of (1-x)*gamma in the repository MILP.
    Relaxing every variable gives a lower bound, not an interdiction.
    """
    edges, nodes = sorted(graph.edges()), sorted(graph.nodes())
    m, n = len(edges), len(nodes)
    positions = {node: 2 * m + i for i, node in enumerate(nodes)}
    row, col, data = [], [], []
    for i, (u, v) in enumerate(edges):
        row.extend([i] * 4)
        col.extend([positions[u], positions[v], i, m + i])
        data.extend([1, -1, -1, -1])
    row.extend([m] * m)
    col.extend(range(m))
    data.extend([1] * m)
    matrix = coo_matrix((data, (row, col)), shape=(m + 1, 2 * m + n)).tocsr()
    cost = [0] * (2 * m + n)
    cost[m:2 * m] = [graph[u][v]['weight'] for u, v in edges]
    rhs = np.zeros(m + 1)
    rhs[-1] = budget
    bounds = [(0, 1)] * (2 * m + n)
    if interdictable is not None:
        allowed = set(interdictable)
        if not allowed <= set(edges) or budget > len(allowed):
            raise ValueError('invalid interdictable set or budget')
        for i, edge in enumerate(edges):
            if edge not in allowed:
                bounds[i] = (0, 0)
    bounds[positions[source]] = (1, 1)
    bounds[positions[sink]] = (0, 0)
    return edges, cost, matrix, rhs, bounds


def box_dual_bound(cost, matrix, rhs, bounds, marginals):
    """Certify a lower bound even when HiGHS duals have numerical residuals.

    For any y <= 0, b*y + min_{l<=z<=u} (c-A.T*y)*z is a lower
    bound on min c*z subject to A*z<=b. Project the reported inequality
    duals to y<=0 and evaluate this expression in exact rational arithmetic
    on the binary floating-point inputs. Round the final float downward.
    No assumption of exact primal feasibility or dual stationarity is used.
    """
    if not np.all(np.isfinite(marginals)):
        raise RuntimeError('LP returned non-finite dual multipliers')
    y = [Fraction(float(min(value, 0))) for value in marginals]
    reduced = [Fraction(value) for value in cost]
    sparse = matrix.tocoo()
    for i, j, a in zip(sparse.row, sparse.col, sparse.data):
        reduced[j] -= Fraction(a.item()) * y[i]
    bound = sum(Fraction(float(b)) * value for b, value in zip(rhs, y))
    bound += sum(r * Fraction(lo if r >= 0 else hi)
                 for r, (lo, hi) in zip(reduced, bounds))
    # The NIP objective is nonnegative independently of the LP certificate.
    bound = max(Fraction(0), bound)
    rounded = float(bound)
    if Fraction(rounded) > bound:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


def lp_relaxation(graph, source, sink, budget, *, interdictable=None, time_limit=None):
    """Return the fractional LP solution and rationally checked lower bound.

    Optional allowed arcs match the MILP; fixed x bounds are included in the
    certificate. A HiGHS time limit never becomes an optimal LP result. The
    optional limit applies to HiGHS; construction/certification are timed too.
    """
    _validate(graph, source, sink, budget)
    if time_limit is not None and (isinstance(time_limit, bool) or
            not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('LP time limit must be positive and finite')
    start = time.perf_counter()
    edges, cost, matrix, rhs, bounds = _lp_model(graph, source, sink, budget, interdictable)
    options = {} if time_limit is None else {'time_limit': float(time_limit)}
    result = linprog(cost, A_ub=matrix, b_ub=rhs, bounds=bounds, method='highs', options=options)
    if not result.success:
        raise RuntimeError(f'LP relaxation failed: {result.message}')
    if not np.all(np.isfinite(result.x)) or not math.isfinite(result.fun):
        raise RuntimeError('LP returned a non-finite solution')
    lower_bound = box_dual_bound(cost, matrix, rhs, bounds, result.ineqlin.marginals)
    return {'lower_bound': lower_bound, 'lp_objective': float(result.fun),
            'fractional_x': result.x[:len(edges)].tolist(),
            'lp_time_sec': time.perf_counter() - start}


def lp_round_repair(graph, source, sink, budget, *, interdictable=None):
    """Solve the continuous relaxation, round at 1/2, then greedily repair.

    Returned cuts use sorted arc order. LP time includes construction and
    certificate validation; repair time includes rounding and final scoring.
    """
    allowed = None if interdictable is None else set(interdictable)
    result = lp_relaxation(graph, source, sink, budget, interdictable=allowed)
    start = time.perf_counter()
    edges = sorted(graph.edges())
    rounded = [int(value >= 0.5) for value in result['fractional_x']]
    repair_options = {} if allowed is None else {'candidate_edges': allowed}
    cuts, flow = ni.improve_dwave_solution(graph, rounded, source, sink, budget,
                                          **repair_options)
    result.update(rounded_edges=[edge for edge, bit in zip(edges, rounded) if bit],
                  interdicted_edges=cuts, flow=float(flow),
                  repair_time_sec=time.perf_counter() - start)
    return result


def relative_gap(feasible_flow, lower_bound):
    """Upper bound on relative suboptimality (F-L)/F for minimization.

    Zero surviving flow is already optimal for nonnegative capacities.
    Reject materially inconsistent bounds instead of hiding them in a CSV.
    """
    if (not math.isfinite(feasible_flow) or not math.isfinite(lower_bound)
            or min(feasible_flow, lower_bound) < 0):
        raise ValueError('flow and lower bound must be finite and nonnegative')
    if lower_bound > feasible_flow and not math.isclose(
            lower_bound, feasible_flow, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('lower bound exceeds the feasible flow')
    return max(0.0, (feasible_flow - lower_bound) / feasible_flow) if feasible_flow else 0.0

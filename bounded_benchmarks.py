"""Shared bounded work and scoring for the matched software-only experiments."""
from dataclasses import replace
import math
import time

import networkx as nx

import classical_baselines as cb
import decomposition as dc
import network_interdiction as ni


class WorkLimit(RuntimeError):
    """A caller must retain its last independently scored feasible solution."""


class FlowWork:
    """One method's cooperative wall budget and counted/cached max-flow work.

    The initial empty-cut score is charged and always available as a feasible
    fallback. Every subsequent call checks the limit before copying a graph.
    A NetworkX call already running can overrun the deadline. Cached queries
    still check wall time but consume no additional max-flow allowance.
    """
    def __init__(self, graph, source, sink, budget, allowed, *, seconds=20, max_calls=5000):
        cb._validate(graph, source, sink, budget)
        if (not math.isfinite(seconds) or not 0 < seconds <= 120
                or isinstance(seconds, bool) or type(max_calls) is not int or not 1 <= max_calls <= 10000):
            raise ValueError('invalid bounded-flow resource allowance')
        self.allowed = set(allowed)
        if not self.allowed <= set(graph.edges()) or budget > len(self.allowed):
            raise ValueError('invalid allowed arcs or budget')
        self.graph, self.source, self.sink, self.budget = graph, source, sink, budget
        self.start = time.perf_counter()
        self.deadline = self.start + seconds
        self.max_calls, self.calls, self.requests = max_calls, 1, 1
        self.best_cut = []
        self.best_flow = nx.maximum_flow_value(graph, source, sink, capacity='weight')
        self.cache = {(): self.best_flow}

    def remaining(self):
        left = self.deadline - time.perf_counter()
        if left <= 0:
            raise WorkLimit('time_limit')
        return left

    def evaluate(self, cut):
        self.remaining()
        cut = tuple(sorted(tuple(edge) for edge in cut))
        if len(set(cut)) != len(cut) or not set(cut) <= self.allowed:
            raise ValueError('invalid interdiction witness')
        self.requests += 1
        if cut not in self.cache:
            if self.calls >= self.max_calls:
                raise WorkLimit('max_flow_call_limit')
            self.calls += 1
            residual = self.graph.copy()
            residual.remove_edges_from(cut)
            self.cache[cut] = nx.maximum_flow_value(residual, self.source, self.sink, capacity='weight')
        value = self.cache[cut]
        if len(cut) <= self.budget and (value, -len(cut), cut) < (
                self.best_flow, -len(self.best_cut), tuple(self.best_cut)):
            self.best_cut, self.best_flow = list(cut), value
        return value

    def result(self, status='completed', **details):
        return dict(flow=self.best_flow, interdicted_edges=self.best_cut, status=status,
                    runtime_sec=time.perf_counter() - self.start,
                    max_flow_calls=self.calls, max_flow_requests=self.requests,
                    max_flow_call_limit=self.max_calls, **details)


def repair(work, cut):
    """Use the repository's exact same greedy add/remove policy with a budget."""
    cut = set(map(tuple, cut))
    if not cut <= work.allowed:
        raise ValueError('repair starts with a forbidden arc')
    bits = [int(edge in cut) for edge in sorted(work.graph.edges())]
    return ni.improve_dwave_solution(work.graph, bits, work.source, work.sink, work.budget,
                                     candidate_edges=work.allowed, flow_oracle=work.evaluate)


def greedy(graph, source, sink, budget, allowed, *, seconds=20, max_calls=5000):
    work = FlowWork(graph, source, sink, budget, allowed, seconds=seconds, max_calls=max_calls)
    try:
        repair(work, [])
        return work.result()
    except WorkLimit as error:
        return work.result(str(error))


def lp(graph, source, sink, budget, allowed, *, seconds=20, max_calls=5000):
    work = FlowWork(graph, source, sink, budget, allowed, seconds=seconds, max_calls=max_calls)
    details = {'variant': 'half-threshold repair plus top-B candidate'}
    try:
        details.update(cb.lp_relaxation(graph, source, sink, budget, interdictable=allowed,
                                   time_limit=work.remaining()))
        edges = sorted(graph.edges())
        # A deterministic feasible fallback is available if rounding overshoots
        # and removal repair cannot finish. It is explicitly retained in output.
        ranking = sorted((i for i, edge in enumerate(edges) if edge in work.allowed),
                         key=lambda i: (-details['fractional_x'][i], edges[i]))
        fallback = [edges[i] for i in ranking[:budget]]
        work.evaluate(fallback)
        details['top_B_fallback'] = fallback
        rounded = [edge for edge, value in zip(edges, details['fractional_x']) if value >= .5]
        details['rounded_edges'] = rounded
        repair(work, rounded)
        return work.result(**details)
    except WorkLimit as error:
        return work.result(str(error), **details)


def restricted_subproblem(graph, source, sink, pieces, index, allowed, *, removed=(), max_terminals=4):
    """Contract only fixed surroundings, then retain globally allowed owned arcs."""
    sub = dc.piece_subproblem(graph, source, sink, pieces, index, removed, max_terminals)
    arcs = [edge for edge in sub.interdictable if sub.arc_map[edge] in allowed]
    return replace(sub, interdictable=arcs, arc_map={edge: sub.arc_map[edge] for edge in arcs})


def capped_split(budget, weights, capacities):
    """Proportional split with explicit capacities and deterministic redistribution."""
    if (type(budget) is not int or budget < 0 or not capacities or
            len(capacities) != len(weights) or any(type(c) is not int or c < 0 for c in capacities) or
            budget > sum(capacities) or any(not math.isfinite(w) or w < 0 for w in weights)):
        raise ValueError('invalid capped allocation')
    weights = list(weights) if sum(weights) else list(capacities)
    if not budget:
        return [0] * len(capacities)
    result = [min(c, b) for c, b in zip(capacities, dc.apportion(budget, weights))]
    # Add to the most underrepresented share per allocated unit; zero-weight
    # pieces receive residual budget only when positive-weight pieces are full.
    while sum(result) < budget:
        available = [i for i, c in enumerate(capacities) if result[i] < c]
        index = max(available, key=lambda i: (weights[i] / (result[i] + 1), -i))
        result[index] += 1
    return result


def decomposed(graph, source, sink, budget, allowed, *, rule='mincut', mode='sequential',
               seconds=20, max_child_calls=128, max_terminals=4, max_rounds=1):
    """Bound every piece solve with one shared allowance; retain partial progress.

    Fixed splits accept audited feasible children. Marginal/multiplier curves
    require optimal child solves and otherwise refuse allocation. Curve bounds
    are only surrogate bounds. Curve solves count toward the same allowance as
    the reconstruction, so many small solves cannot escape the work accounting.
    """
    import run_challenge_profile as rp
    if (rule not in dc.SPLIT_STRATEGIES or mode not in dc.SOLVE_MODES or
            type(max_child_calls) is not int or not 1 <= max_child_calls <= 256 or
            type(max_terminals) is not int or not 0 <= max_terminals <= 4 or
            type(max_rounds) is not int or not 0 <= max_rounds <= 2):
        raise ValueError('invalid bounded decomposition controls')
    work = FlowWork(graph, source, sink, budget, allowed, seconds=seconds)
    details = {'rule': rule, 'mode': mode, 'child_calls': [], 'max_child_calls': max_child_calls,
               'max_terminals': max_terminals, 'max_rounds': max_rounds,
               'curves': [], 'allocation': None, 'budgets': None}
    calls = details['child_calls']
    try:
        pieces = dc.partition_graph(graph, min(4, len(graph)), seed=0)
        details['pieces'] = [sorted(p) for p in pieces]
        owners = dc.arc_owners(graph, pieces)
        capacities = [sum(owners[e] == i for e in allowed) for i in range(len(pieces))]
        details['owned_allowed_arcs'] = capacities

        def child(sub, k, require_optimal=False):
            left = work.remaining()
            arcs = [edge for edge in sub.interdictable if sub.arc_map[edge] in work.allowed]
            k = min(k, len(arcs))
            if k == 0:
                return []
            if len(calls) >= max_child_calls:
                raise WorkLimit('child_call_limit')
            record = {'budget': k, 'nodes': len(sub.graph), 'arcs': sub.graph.number_of_edges(),
                      'allowed_arcs': len(arcs), 'time_limit_sec': min(5., left),
                      'status': 'dispatched'}
            calls.append(record)  # Reserve before dispatch, including failures.
            result = rp.milp_profile(sub.graph, sub.src, sub.tgt, k, set(arcs),
                                     time_limit=record['time_limit_sec'])
            record.update(result)
            chosen = [sub.arc_map[tuple(edge)] for edge in result['interdicted_edges']]
            if require_optimal and result['status'] != 'optimal':
                raise WorkLimit('curve_child_not_optimal')
            # Any one-piece cut is also a feasible original-network candidate.
            # Sequential/iterated combinations are scored by solve_decomposed.
            work.evaluate(chosen)
            return chosen

        if rule in ('marginal', 'multiplier'):
            reserved_curves = sum(min(budget, cap) for cap in capacities)
            details['planned_curve_calls_upper_bound'] = reserved_curves
            if reserved_curves + sum(cap > 0 for cap in capacities) > max_child_calls:
                raise WorkLimit('curve_work_exceeds_allowance')
            for i in range(len(pieces)):
                work.remaining()
                sub = restricted_subproblem(graph, source, sink, pieces, i, allowed,
                                             max_terminals=max_terminals)
                curve = [work.cache[()]]
                for k in range(1, min(budget, len(sub.interdictable)) + 1):
                    chosen = child(sub, k, require_optimal=True)
                    curve.append(work.evaluate(chosen))
                details['curves'].append(curve)
            if rule == 'multiplier':
                allocation = dc.allocate_budget_multiplier(details['curves'], budget, max_iterations=100)
                details['allocation'] = allocation
                budgets = allocation['budgets']
            else:
                budgets = dc.allocate_budget(details['curves'], budget)
        else:
            if rule == 'local':
                weights = dc.local_max_flows(graph, source, sink, pieces)
            else:
                _, (side, _) = nx.minimum_cut(graph, source, sink, capacity='weight')
                weights = [sum(owners[e] == i and e[0] in side and e[1] not in side
                               for e in allowed) for i in range(len(pieces))]
            details['weights'] = weights
            budgets = capped_split(budget, weights, capacities)
        details['budgets'] = budgets
        # Score each partial combination before a later child can hit the limit.
        # solve_decomposed's first sequential prefix is tracked by the wrapper.
        prefix = []
        dispatches = 0
        active = sum(k > 0 for k in budgets)

        def tracked_child(sub, k):
            nonlocal dispatches
            dispatches += 1
            chosen = child(sub, k)
            if mode != 'independent' and dispatches <= active:
                prefix.extend(chosen)
                work.evaluate(prefix)
            return chosen

        chosen, _ = dc.solve_decomposed(graph, source, sink, pieces, budgets, tracked_child,
                                        mode, max_terminals=max_terminals,
                                        max_rounds=max_rounds, flow_oracle=work.evaluate)
        details['reconstructed_cut'] = chosen
        details['reconstructed_flow'] = work.evaluate(chosen)
        details['child_oracles_all_optimal'] = all(c['status'] == 'optimal' for c in calls)
        status = ('completed' if details['child_oracles_all_optimal']
                  else 'completed_with_inexact_children')
        return work.result(status, **details)
    except WorkLimit as error:
        return work.result(str(error), **details)


def sample_records(samples, bqm, graph, budget, work, *, limit=20):
    """Score lowest-energy distinct observations before spending work on repair.

    Multiplicities refer to the full sampler output; this selection is not a
    probability estimate for all shots. Over-budget flows remain explicitly
    infeasible and cannot replace the feasible incumbent.
    """
    variables = list(bqm.variables)
    edges = sorted(graph.edges())
    records = []
    for row in samples.aggregate().data(sorted_by='energy'):
        if len(records) >= limit:
            break
        cut = [edge for i, edge in enumerate(edges) if row.sample.get(f'x_{i}', 0)]
        record = dict(bits=''.join(str(int(row.sample[v])) for v in variables),
                      energy=float(bqm.energy(row.sample)), occurrences=int(row.num_occurrences),
                      interdicted_edges=cut, budget_feasible=len(cut) <= budget,
                      exact_budget=len(cut) == budget, flow=None)
        records.append(record)
        try:
            record['flow'] = work.evaluate(cut)
        except WorkLimit:
            break
    return variables, records


def anneal(graph, source, sink, budget, allowed, *, seconds=20, max_calls=5000,
           reads=200, sweeps=1000, seed=0):
    """One certified squared BQM, seeded SA, then the shared bounded repair.

    Compilation counts against the method allowance. SA is interrupted between
    reads after half the remaining allowance, reserving time for scoring/repair.
    A read already running can overrun; the campaign adds an outer process cap.
    """
    from dataclasses import asdict
    from dwave.samplers import SimulatedAnnealingSampler
    from certified_penalties import certified_squared_qubo
    if (type(reads) is not int or not 1 <= reads <= 1000 or type(sweeps) is not int or
            not 1 <= sweeps <= 10000 or type(seed) is not int or not 0 <= seed < 2**31):
        raise ValueError('invalid SA resource controls')
    work = FlowWork(graph, source, sink, budget, allowed, seconds=seconds, max_calls=max_calls)
    details = dict(requested_reads=reads, sweeps=sweeps, seed=seed, completed_reads=0,
                   raw_samples=[], variables=[], certificate=None)
    try:
        start = time.perf_counter()
        bqm, certificate = certified_squared_qubo(budget, graph, source, sink, allowed)
        details.update(certificate=asdict(certificate), qubits=len(bqm), couplings=bqm.num_interactions,
                       compile_time_sec=time.perf_counter()-start)
        stop = time.perf_counter() + work.remaining()/2
        start = time.perf_counter()
        samples = SimulatedAnnealingSampler().sample(
            bqm, num_reads=reads, num_sweeps=sweeps, seed=seed,
            interrupt_function=lambda: time.perf_counter() >= stop)
        details.update(sample_time_sec=time.perf_counter()-start,
                       completed_reads=int(sum(samples.record.num_occurrences)))
        variables, records = sample_records(samples, bqm, graph, budget, work)
        details.update(variables=variables, raw_samples=records)
        for row in records:
            repair(work, row['interdicted_edges'])
        return work.result('completed' if details['completed_reads'] == reads else 'sampling_time_limit', **details)
    except WorkLimit as error:
        return work.result(str(error), **details)

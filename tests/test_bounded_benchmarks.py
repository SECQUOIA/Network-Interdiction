import itertools

import networkx as nx
import pytest

import bounded_benchmarks as bb
import challenge_instances as ci
import classical_baselines as cb
import network_interdiction as ni
from test_gurobi_trajectory import require_gurobi


def tiny():
    return ni.read_flow_file(ci.ROOT / 'testcases/tiny_a.flow')


def flow(g, s, t, removed):
    h = g.copy()
    h.remove_edges_from(removed)
    return nx.maximum_flow_value(h, s, t, capacity='weight')


def test_interrupted_repair_retains_new_feasible_improvement():
    g, s, t = tiny()
    result = bb.greedy(g, s, t, 1, set(g.edges()), max_calls=2)
    assert result['status'] == 'max_flow_call_limit'
    assert result['max_flow_calls'] == 2
    assert result['flow'] == flow(g, s, t, result['interdicted_edges'])
    assert result['flow'] < flow(g, s, t, [])
    assert len(result['interdicted_edges']) <= 1


def test_cache_is_counted_and_deadline_checked_even_for_cached_queries(monkeypatch):
    g, s, t = tiny()
    work = bb.FlowWork(g, s, t, 1, set(g.edges()), max_calls=2)
    cut = [sorted(g.edges())[0]]
    assert work.evaluate(cut) == work.evaluate(cut)
    assert work.calls == 2 and work.requests == 3
    monkeypatch.setattr(bb.time, 'perf_counter', lambda: work.deadline + 1)
    with pytest.raises(bb.WorkLimit, match='time_limit'):
        work.evaluate(cut)
    assert work.calls == 2


def test_repair_uses_same_policy_and_rejects_forbidden_start():
    g, s, t = tiny()
    allowed = {sorted(g.edges())[0]}
    result = bb.greedy(g, s, t, 1, allowed)
    assert result['status'] == 'completed'
    assert set(result['interdicted_edges']) == allowed
    work = bb.FlowWork(g, s, t, 1, allowed)
    with pytest.raises(ValueError, match='forbidden'):
        bb.repair(work, [sorted(g.edges())[-1]])


@pytest.mark.parametrize('budget', [0, 1, 3, 6])
def test_restricted_lp_bound_and_rounded_cut_against_exhaustive_oracle(budget):
    g, s, t, allowed, _ = ci.covering(4, 1, 0)
    optimum = min(flow(g, s, t, cut) for cut in itertools.combinations(allowed, budget))
    result = cb.lp_round_repair(g, s, t, budget, interdictable=allowed)
    assert result['lower_bound'] <= optimum <= result['flow']
    assert len(result['interdicted_edges']) <= budget
    assert set(result['interdicted_edges']) <= set(allowed)
    assert result['flow'] == flow(g, s, t, result['interdicted_edges'])
    for edge, x in zip(sorted(g.edges()), result['fractional_x']):
        if edge not in allowed:
            assert x == 0


@pytest.mark.parametrize('weights,capacities,B', [([0,0,0],[1,2,3],5),([10,0,0],[1,2,3],6),
                                                ([1,1],[0,4],3),([2,1],[3,3],0)])
def test_capped_allocation_spends_budget_without_exceeding_a_piece(weights, capacities, B):
    result = bb.capped_split(B, weights, capacities)
    assert sum(result) == B
    assert all(0 <= value <= cap for value,cap in zip(result,capacities))


def test_bounded_lp_preserves_feasible_cut_after_scoring_limit():
    g, s, t = tiny()
    result = bb.lp(g, s, t, 1, set(g.edges()), max_calls=2)
    assert result['flow'] == flow(g, s, t, result['interdicted_edges'])
    assert result['lower_bound'] <= result['flow']
    assert result['max_flow_calls'] == 2


@pytest.mark.parametrize('rule', ['mincut','local','marginal','multiplier'])
def test_decomposition_matches_allowed_arcs_and_bounded_work(rule):
    require_gurobi()
    g, s, t, allowed, _ = ci.covering(4, 1, 0)
    result = bb.decomposed(g, s, t, 3, set(allowed), rule=rule, mode='iterated')
    assert result['status'] == 'completed'
    assert set(result['interdicted_edges']) <= set(allowed)
    assert len(result['interdicted_edges']) <= 3
    assert result['flow'] == flow(g, s, t, result['interdicted_edges'])
    assert len(result['child_calls']) <= result['max_child_calls']
    assert all(r['status']=='optimal' for r in result['child_calls'])


def test_oversized_curve_allowance_refuses_before_child_dispatch(monkeypatch):
    g,s,t,allowed,_ = ci.covering(8,1,0)
    monkeypatch.setattr(ni, '_gurobi_model', lambda: pytest.fail('child dispatched'))
    result = bb.decomposed(g,s,t,20,set(allowed),rule='multiplier',max_child_calls=2)
    assert result['status'] == 'curve_work_exceeds_allowance'
    assert result['child_calls'] == []
    assert result['max_flow_calls'] == 1


def test_actual_sa_has_seeded_records_and_certified_penalties():
    g,s,t = tiny()
    first = bb.anneal(g,s,t,1,set(g.edges()),reads=16,sweeps=50,seed=7)
    second = bb.anneal(g,s,t,1,set(g.edges()),reads=16,sweeps=50,seed=7)
    assert first['completed_reads'] == second['completed_reads'] == 16
    assert first['raw_samples'] == second['raw_samples']
    assert first['certificate']['cardinality'] == sum(d['weight'] for _,_,d in g.edges(data=True))+1
    assert first['flow'] == flow(g,s,t,first['interdicted_edges'])
    assert first['flow'] >= 5
    assert sum(r['occurrences'] for r in first['raw_samples']) == 16
    for row in first['raw_samples']:
        assert row['flow'] == flow(g,s,t,row['interdicted_edges'])
        assert row['budget_feasible'] == (len(row['interdicted_edges']) <= 1)


def test_samples_are_scored_before_repair_and_preserve_occurrences():
    import dimod
    g,s,t = tiny()
    from certified_penalties import certified_squared_qubo
    bqm,_ = certified_squared_qubo(1,g,s,t)
    samples = dimod.SampleSet.from_samples_bqm([{v:0 for v in bqm.variables}],bqm,num_occurrences=[9])
    work = bb.FlowWork(g,s,t,1,set(g.edges()))
    variables, records = bb.sample_records(samples,bqm,g,1,work)
    assert len(variables) == len(bqm)
    assert records[0]['occurrences'] == 9
    assert records[0]['budget_feasible'] and not records[0]['exact_budget']
    assert records[0]['flow'] == flow(g,s,t,[])

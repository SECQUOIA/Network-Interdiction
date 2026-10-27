"""
Tests for Gurobi's incumbent trajectory (MIPSOL callback), the per-method
timing columns, and seeding simulated annealing, all used by the
time-versus-quality comparison in notebooks/analysis.ipynb.
"""
import json
import os

import gurobipy
import numpy as np
import pytest

import network_interdiction as ni
import run_benchmark as rb
import solve_qubo_via_dwave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny(name):
    return os.path.join(REPO_ROOT, 'testcases', f'{name}.flow')


def require_gurobi():
    """Skip when no Gurobi license is usable; the tiny models here fit the
    size-limited license pip installs, so these run in CI."""
    try:
        m = gurobipy.Model('license-check')
        m.setParam('OutputFlag', 0)
        m.optimize()
    except gurobipy.GurobiError as e:
        pytest.skip(f'no usable Gurobi license in this environment: {e}')


@pytest.mark.parametrize('name, B', [('tiny_b', 1), ('tiny_b', 2),
                                     ('tiny_c', 1), ('tiny_d', 2)])
def test_trajectory_ends_at_the_optimum(name, B):
    """
    Incumbents arrive in time order, every one is a real interdiction (its
    flow is between the optimum and the uninterdicted max flow), and the
    best of them is the brute-force optimum, found no later than Gurobi
    finishes.
    """
    require_gurobi()
    g, src, tgt = ni.read_flow_file(tiny(name))
    result = ni.solve_nip_gurobi_trajectory(g, src, tgt, B)
    trajectory = result['trajectory']
    optimum = rb.brute_force_nip_optimum(g, src, tgt, B)[0]

    assert trajectory
    times = [t for t, _ in trajectory]
    assert times == sorted(times)
    assert times[-1] <= result['runtime_sec'] + 1e-9
    flows = [flow for _, flow in trajectory]
    assert all(optimum <= flow <= rb.max_flow(g, src, tgt) for flow in flows)
    assert min(flows) == optimum == result['objective']


def test_final_point_is_the_true_flow_of_the_returned_cuts():
    """The trajectory reports surviving max flow, the metric every other
    method is scored by, not Gurobi's objective value."""
    require_gurobi()
    g, src, tgt = ni.read_flow_file(tiny('tiny_b'))
    result = ni.solve_nip_gurobi_trajectory(g, src, tgt, 1)
    assert result['trajectory'][-1][1] == rb.max_flow(
        g, src, tgt, result['interdicted_edges'])


def test_run_case_records_times_and_a_consistent_trajectory():
    require_gurobi()
    row = rb.run_case(tiny('tiny_b'), 2, 'auto', 'gurobi', solvers=['sa'],
                      baselines=['greedy', 'random', 'lp', 'lagrangian', 'benders'], sa_seed=0)
    optional = {'QCI_time_sec', 'QCI_repair_time_sec', 'QAOA_time_sec', 'QAOA_repair_time_sec'}
    for column in rb.TIME_COLUMNS:
        if column in optional:
            assert row[column] == ''
        else:
            assert isinstance(row[column], float) and row[column] >= 0, column
    assert row['Gurobi_total_time_sec'] >= row['Gurobi_time_sec']
    assert row['SA_total_time_sec'] >= row['SA_time_sec'] + row['SA_repair_time_sec']
    trajectory = json.loads(row[rb.TRAJECTORY_COLUMN])
    assert min(flow for _, flow in trajectory) == row['Opt_NIP']


def test_brute_backend_leaves_the_gurobi_columns_empty():
    row = rb.run_case(tiny('tiny_a'), 1, 'auto', 'brute', solvers=[],
                      baselines=['greedy'])
    assert row['Gurobi_time_sec'] == ''
    assert row['Gurobi_total_time_sec'] == ''
    assert row['SA_total_time_sec'] == ''
    assert row['Reference_time_sec'] >= 0
    assert row[rb.TRAJECTORY_COLUMN] == ''
    assert row['Greedy_time_sec'] >= 0
    for column in ('LP_time_sec', 'LP_repair_time_sec', 'Lagrangian_time_sec',
                   'Benders_time_sec'):
        assert row[column] == ''


def test_simulated_annealing_is_reproducible_from_its_seed():
    """
    The original slide runs could not be reproduced because SA was unseeded.
    Compares the raw samples, not the best result: on an instance this small
    every seed reaches the same optimum, so equal results would not show the
    seed was used.
    """
    g, src, tgt = ni.read_flow_file(tiny('tiny_d'))
    bqm = ni.nip_to_qubo(1, g, src, tgt, 8.0, 9.0)
    first = solve_qubo_via_dwave.solve_bqm_dwave(bqm, 'sa', seed=11)
    again = solve_qubo_via_dwave.solve_bqm_dwave(bqm, 'sa', seed=11)
    other = solve_qubo_via_dwave.solve_bqm_dwave(bqm, 'sa', seed=12)
    assert np.array_equal(first.record.sample, again.record.sample)
    assert not np.array_equal(first.record.sample, other.record.sample)


def test_qubo_verify_parameters_reports_its_times():
    g, src, tgt = ni.read_flow_file(tiny('tiny_d'))
    result = ni.qubo_verify_parameters('t', g, 8.0, src, tgt, 1, 'sa', seed=0)
    assert result['sample_time_sec'] > 0
    assert result['postprocess_time_sec'] > 0

"""
Tests for choosing the cardinality penalty lambda from its exact window
(issue #2): optimum_curve(), lambda_window(), choose_lambda(), and the
`auto` lambda path through run_case()/main() in run_benchmark.py.
"""
import os

import gurobipy
import pandas
import pytest

import network_interdiction as ni
import run_benchmark as rb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny(name):
    return os.path.join(REPO_ROOT, 'testcases', f'{name}.flow')


def require_gurobi():
    """Skip when no usable Gurobi license is present (as in CI)."""
    try:
        m = gurobipy.Model('license-check')
        m.setParam('OutputFlag', 0)
        m.optimize()
    except gurobipy.GurobiError as e:
        pytest.skip(f'no usable Gurobi license in this environment: {e}')


# Windows predicted by Proposition 3 for the six sample rows. These match
# the ranges found by sweeping lambda against dimod.ExactSolver when the
# sample was first generated (Table 1 of mfnip-qubo-research-notes).
SAMPLE_WINDOWS = [
    ('tiny_a', 1, (5.0, 10.0)),
    ('tiny_b', 1, (8.0, 12.0)),
    ('tiny_b', 2, (5.0, 8.0)),
    ('tiny_c', 1, (4.0, 7.0)),
    ('tiny_d', 1, (7.0, 9.0)),
    ('tiny_d', 2, (6.0, 7.0)),
]


@pytest.mark.parametrize('name, B, expected', SAMPLE_WINDOWS)
def test_window_matches_the_swept_ranges(name, B, expected):
    g, src, tgt = ni.read_flow_file(tiny(name))
    curve = rb.optimum_curve('brute', g, src, tgt, B)
    assert rb.lambda_window(curve, B) == expected


@pytest.mark.parametrize('name, B, expected', SAMPLE_WINDOWS)
def test_auto_lambda_is_interior_and_encodes_the_instance(name, B, expected):
    """
    The chosen lambda lies strictly inside the window, and the exact QUBO
    ground state at that lambda really is the optimal B-arc interdiction --
    exact_reference() raises otherwise, so this checks Proposition 3 against
    an exhaustive solve rather than against the formula it came from.
    """
    g, src, tgt = ni.read_flow_file(tiny(name))
    lo, hi = expected
    lam = rb.choose_lambda(lo, hi)
    assert lo < lam < hi
    opt_nip, _ = rb.exact_reference('brute', g, src, tgt, B, lam)
    assert opt_nip == rb.brute_force_nip_optimum(g, src, tgt, B)[0]


def test_the_old_slide_range_admits_a_lambda_that_overshoots_the_budget():
    """
    The slide deck bounded lambda by (min capacity, max capacity). On tiny_b
    at B=1 that admits lambda=6, where f(k) + 6k = 25, 19, 17, 18 for
    k = 0..3, so the ground state cuts two arcs. The window is (8, 12).
    """
    g, src, tgt = ni.read_flow_file(tiny('tiny_b'))
    capacities = [d['weight'] for _, _, d in g.edges(data=True)]
    assert min(capacities) < 6.0 < max(capacities)

    with pytest.raises(ValueError, match='does not encode this instance'):
        rb.exact_reference('brute', g, src, tgt, 1, 6.0)


def test_curve_stops_at_its_first_zero_past_the_budget():
    """tiny_b: f = 25, 13, 5, 0; nothing past the first zero can matter."""
    g, src, tgt = ni.read_flow_file(tiny('tiny_b'))
    assert rb.optimum_curve('brute', g, src, tgt, 1) == [25, 13, 5, 0]
    assert rb.optimum_curve('brute', g, src, tgt, 2) == [25, 13, 5, 0]


def test_window_uses_every_k_not_just_the_neighbours():
    """
    With f = 20, 19, 18, 0 at B=1 the binding constraint on the lower end
    comes from k=3, not k=2: (19 - 0)/2 = 9.5 > (19 - 18)/1 = 1. Checking
    only the neighbours of B would report (1, 1) -- and it is empty anyway,
    since the upper end is f(0) - f(1) = 1.
    """
    lo, hi = rb.lambda_window([20, 19, 18, 0], 1)
    assert (lo, hi) == (9.5, 1.0)
    assert rb.choose_lambda(lo, hi) is None


def test_empty_window_when_f_is_not_convex_at_the_budget():
    """f = 10, 9, 0 at B=1: one cut saves 1, the second saves 9."""
    lo, hi = rb.lambda_window([10, 9, 0], 1)
    assert (lo, hi) == (9.0, 1.0)
    assert rb.choose_lambda(lo, hi) is None


def write_two_equal_paths(path):
    """
    Two disjoint s-t paths of capacity 5. f = 10, 5, 0 is a straight line, so
    B=1 is not a vertex of its convex envelope: the window at B=1 is (5, 5),
    and at any lambda the QUBO ties or prefers 0 or 2 cuts.
    """
    path.write_text('4\n0 3\n0 1 5\n1 3 5\n0 2 5\n2 3 5\n')
    return str(path)


def test_empty_window_row_keeps_baselines_and_drops_the_qubo(tmp_path):
    instance = write_two_equal_paths(tmp_path / 'equal.flow')
    row = rb.run_case(instance, 1, 'auto', 'brute', solvers=['sa'],
                      baselines=['greedy'])

    assert (row['lambda_lo'], row['lambda_hi']) == (5.0, 5.0)
    assert row['Opt_NIP'] == 5
    assert row['Greedy'] == 5
    # No lambda, so no QUBO optimum and no solver scored against one.
    assert row['lambda'] == ''
    assert row['Opt_QUBO'] == ''
    for column in rb.SOLVER_COLUMNS['sa']:
        assert row[column] == ''


def test_cli_header_names_the_empty_window_rows(tmp_path):
    instance = write_two_equal_paths(tmp_path / 'equal.flow')
    out = tmp_path / 'auto.csv'
    rb.main(['--exact', 'brute', '--solvers', 'none',
             '--case', f"{tiny('tiny_b')}:1:auto",
             '--case', f'{instance}:1:auto', '--out', str(out)])

    text = out.read_text()
    assert '# lambda = auto:' in text
    assert f'empty lambda window (f not convex at B), baselines only: ' \
           f'{instance} B=1' in text

    frame = pandas.read_csv(out, comment='#')
    assert frame.loc[0, 'lambda'] == 10.0
    assert pandas.isna(frame.loc[1, 'lambda'])


def test_gurobi_curve_matches_brute_force():
    require_gurobi()
    g, src, tgt = ni.read_flow_file(tiny('tiny_b'))
    brute = rb.optimum_curve('brute', g, src, tgt, 2)
    gurobi = rb.optimum_curve('gurobi', g, src, tgt, 2)
    assert gurobi == brute


def test_a_slide_instance_has_an_empty_window():
    """
    Erdos-Renyi 90_0.1 at B=10 (a budget on slide 9 of S-T_NIP.pptx):
    f(9) - f(10) = 36 but f(10) - f(11) = 31 and (f(10) - f(12))/2 = 36.5,
    so the lower end 36.5 exceeds the upper end 36. No lambda makes the
    linear-penalty QUBO's ground state the optimum for that row.
    """
    require_gurobi()
    g, src, tgt = ni.read_flow_file(
        os.path.join(REPO_ROOT, 'testcases', '90_0.1.erdosflow'))
    curve = rb.optimum_curve('gurobi', g, src, tgt, 10)
    lo, hi = rb.lambda_window(curve, 10)
    assert (lo, hi) == (36.5, 36.0)
    assert rb.choose_lambda(lo, hi) is None

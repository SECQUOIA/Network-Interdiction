"""
Checks on results/slide_instances_sa.csv, the classical regeneration of the
D-Wave runs shown in S-T_NIP.pptx (issue #18): it covers exactly the slide
instances and budgets, and its exact and baseline columns are what
run_benchmark.py computes, not hand-edited numbers.
"""
import os

import gurobipy
import pandas
import pytest

import network_interdiction as ni
import run_benchmark as rb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDE_CSV = os.path.join(REPO_ROOT, 'results', 'slide_instances_sa.csv')

# (instance, budget) for every bar on slides 8-10. The x-axis label on the
# slides is the QUBO variable count V + 3E, which identifies the instance.
SLIDE_CASES = {
    ('testcases/50.planarflow', B) for B in (5, 6, 8, 10)} | {
    ('testcases/75.planarflow', B) for B in (5, 8, 10)} | {
    ('testcases/125.planarflow', B) for B in (6, 10)} | {
    ('testcases/50_0.15.erdosflow', B) for B in (5, 6, 8)} | {
    ('testcases/70_0.1.erdosflow', B) for B in (7, 8, 10)} | {
    ('testcases/90_0.1.erdosflow', B) for B in (8, 10, 12)} | {
    ('testcases/50_5.2dnnflow', B) for B in (4, 5, 6)} | {
    ('testcases/75_6.2dnnflow', B) for B in (6, 7, 8)} | {
    ('testcases/100_6.2dnnflow', B) for B in (6, 7, 8)}

SLIDE_SIZES = {
    'planarflow': {914, 1389, 2339},
    'erdosflow': {1124, 1438, 2370},
    '2dnnflow': {1070, 1845, 2398},
}


@pytest.fixture(scope='module')
def slides():
    return pandas.read_csv(SLIDE_CSV, comment='#')


def load(row):
    return ni.read_flow_file(os.path.join(REPO_ROOT, row['instance']))


def require_unrestricted_gurobi():
    """
    Skip unless Gurobi can solve a model larger than the size-limited
    license that ships with pip's gurobipy (as in CI). An empty-model check
    passes under that license, but the slide instances exceed it.
    """
    try:
        m = gurobipy.Model('license-capacity-check')
        m.setParam('OutputFlag', 0)
        x = m.addVars(5000, vtype=gurobipy.GRB.BINARY)
        m.setObjective(x.sum(), gurobipy.GRB.MAXIMIZE)
        m.optimize()
    except gurobipy.GurobiError as e:
        pytest.skip(f'no unrestricted Gurobi license in this environment: {e}')


def test_covers_exactly_the_slide_cases(slides):
    cases = set(zip(slides['instance'], slides['B']))
    assert cases == SLIDE_CASES
    assert len(slides) == len(SLIDE_CASES) == 27


def test_instances_have_the_sizes_printed_on_the_slides(slides):
    for _, row in slides.iterrows():
        family = row['instance'].rsplit('.', 1)[1]
        assert row['V'] + 3 * row['E'] in SLIDE_SIZES[family], row['instance']


def test_only_the_empty_window_row_lacks_a_qubo(slides):
    """
    Erdos-Renyi 90_0.1 at B=10 has no valid lambda (see
    test_lambda_window.py); every other row has one, strictly inside its
    window, and a QUBO optimum.
    """
    no_qubo = slides[slides['lambda'].isna()]
    assert list(zip(no_qubo['instance'], no_qubo['B'])) == [
        ('testcases/90_0.1.erdosflow', 10)]
    assert (no_qubo['lambda_lo'] > no_qubo['lambda_hi']).all()

    rest = slides.dropna(subset=['lambda'])
    assert ((rest['lambda_lo'] < rest['lambda'])
            & (rest['lambda'] < rest['lambda_hi'])).all()
    assert rest['Opt_QUBO'].notna().all()


@pytest.mark.parametrize('instance, B', [
    ('testcases/50_5.2dnnflow', 4),
    ('testcases/50.planarflow', 5),
])
def test_greedy_column_reproduces(slides, instance, B):
    """License-free spot check on the two cheapest rows."""
    row = slides[(slides['instance'] == instance) & (slides['B'] == B)].iloc[0]
    g, src, tgt = load(row)
    assert rb.greedy_from_scratch(g, src, tgt, B) == row['Greedy']
    assert rb.max_flow(g, src, tgt) == row['maxflow']


def test_exact_columns_and_windows_reproduce(slides):
    """Every row's optimum and lambda window, recomputed with Gurobi."""
    require_unrestricted_gurobi()
    for _, row in slides.iterrows():
        g, src, tgt = load(row)
        B = int(row['B'])
        curve = rb.optimum_curve('gurobi', g, src, tgt, B)
        assert curve[B] == row['Opt_NIP'], row['instance']
        lo, hi = rb.lambda_window(curve, B)
        assert lo == pytest.approx(row['lambda_lo']), row['instance']
        assert hi == pytest.approx(row['lambda_hi']), row['instance']


def test_every_row_has_a_trajectory_ending_at_its_optimum(slides):
    """
    Gurobi's recorded incumbents (MIPSOL callback) must reach the row's
    optimum, in time order, within Gurobi's reported runtime, and every
    method has a recorded time on every row it ran on.
    """
    import json
    for _, row in slides.iterrows():
        trajectory = json.loads(row['Gurobi_trajectory'])
        times = [t for t, _ in trajectory]
        assert times == sorted(times), row['instance']
        assert times[-1] <= row['Gurobi_time_sec'] + 1e-9, row['instance']
        assert min(flow for _, flow in trajectory) == row['Opt_NIP'], row['instance']
        assert row['Greedy_time_sec'] > 0 and row['Random_Greedy_time_sec'] > 0
        if pandas.notna(row['SA_NIP_Greedy']):
            assert row['SA_time_sec'] > 0 and row['SA_repair_time_sec'] > 0

"""
Runs notebooks/analysis.ipynb's code cells and checks what its figures and
tables show against numbers computed independently from the raw CSVs, so the
notebook cannot silently drift from the data it describes.
"""
import json
import os
from unittest import mock

import matplotlib
import networkx as nx
import numpy as np
import pandas
import pytest

matplotlib.use('Agg')
# Import pyplot before Figure.savefig is stubbed: pyplot wraps it on import.
import matplotlib.pyplot  # noqa: E402,F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO_ROOT, 'notebooks', 'analysis.ipynb')
RESULTS = os.path.join(REPO_ROOT, 'results')
SLIDE_CSV = os.path.join(RESULTS, 'slide_instances_sa.csv')
QAOA_CSV = os.path.join(RESULTS, 'qaoa_tiny.csv')
SAMPLE_CSV = os.path.join(RESULTS, 'sample_results_sa.csv')
SAVED_FIGURES = {'example_instances', 'problem_sizes', 'classical_gaps',
                 'time_vs_gap', 'qaoa_outcomes_depth', 'transfer_outcomes',
                 'mid_qaoa', 'decomposition'}


def read(name):
    return pandas.read_csv(os.path.join(RESULTS, name), comment='#')


@pytest.fixture(scope='module')
def run():
    """
    Execute every code cell in order, from the repo root, and return the
    resulting globals and the figure names saved. Needs no Jupyter kernel:
    IPython magics (``%...`` lines) are dropped, and Figure.savefig is
    stubbed so a test run does not rewrite the committed figures/.
    """
    with open(NOTEBOOK, encoding='utf-8') as handle:
        cells = json.load(handle)['cells']
    namespace = {}
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        with mock.patch('matplotlib.figure.Figure.savefig') as savefig:
            for cell in cells:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source']).splitlines()
                    code = '\n'.join(line for line in source
                                     if not line.lstrip().startswith('%'))
                    exec(code, namespace)
    finally:
        os.chdir(cwd)
    saved = {os.path.splitext(os.path.basename(str(call.args[0])))[0]
             for call in savefig.call_args_list}
    return namespace, saved


@pytest.fixture(scope='module')
def namespace(run):
    return run[0]


@pytest.fixture(scope='module')
def raw():
    return read('slide_instances_sa.csv')


@pytest.fixture(scope='module')
def legacy_tables(raw):
    """Keep regression coverage for diagnostic helpers removed from the main notebook."""
    import utils
    df = utils.load_results(SLIDE_CSV)
    qaoa = utils.load_qaoa_results(QAOA_CSV)
    return {
        'summary': utils.head_to_head_table(df),
        'cmp': utils.compare_with_slides(df),
        'qaoa': qaoa,
        'QAOA_TABLE_DEPTH': 3,
        'qaoa_table': utils.qaoa_vs_classical_table(
            qaoa, pandas.read_csv(SAMPLE_CSV, comment='#'), reps=3),
    }


# ── the notebook ────────────────────────────────────────────────────────────
def test_every_figure_is_saved_once(run):
    assert run[1] == SAVED_FIGURES


def test_raw_flow_columns_are_never_overwritten(namespace, raw):
    """
    Derived columns once reused the CSV's own column names, which overwrote
    the flows and turned every count comparing flows into nonsense.
    """
    large = namespace['large']
    for column in ('Greedy', 'SA_NIP', 'SA_NIP_Greedy', 'Random_Greedy', 'Opt_NIP'):
        pandas.testing.assert_series_equal(large[column], raw[column],
                                           check_names=False)


def test_classical_summary_counts(namespace, raw):
    summary = namespace['summary']
    for label, column in (('Greedy from scratch', 'Greedy'),
                          ('SA + greedy repair', 'SA_NIP_Greedy'),
                          ('Random start + repair', 'Random_Greedy')):
        ran = raw.dropna(subset=[column])
        assert summary.loc[label, 'cases'] == len(ran)
        assert summary.loc[label, 'optimal'] == (ran[column] == ran['Opt_NIP']).sum()
        assert summary.loc[label, 'median_gap'] == pytest.approx(
            (ran[column] / ran['Opt_NIP'] - 1).median(), abs=5e-4)
    assert summary.loc['Gurobi (exact MILP)', 'optimal'] == len(raw)


def test_heat_map_cells_are_the_gaps(namespace, raw):
    image = namespace['fig_gaps'].axes[0].images[0].get_array()
    for row, column in enumerate(('Greedy', 'SA_NIP_Greedy', 'Random_Greedy')):
        expected = 100 * (raw[column] / raw['Opt_NIP'] - 1)
        np.testing.assert_array_equal(image.mask[row], expected.isna())
        np.testing.assert_allclose(image.data[row][~expected.isna()],
                                   expected.dropna(), rtol=1e-12)


def test_time_plot_points(namespace, raw):
    collections = namespace['fig_time'].axes[0].collections
    gurobi, greedy = collections[0].get_offsets(), collections[2].get_offsets()
    np.testing.assert_allclose(gurobi[:, 0], raw['Gurobi_time_sec'])
    np.testing.assert_allclose(gurobi[:, 1], 0)
    np.testing.assert_allclose(greedy[:, 0], raw['Greedy_time_sec'])
    np.testing.assert_allclose(greedy[:, 1], 100 * (raw['Greedy'] / raw['Opt_NIP'] - 1))
    title = namespace['fig_time'].axes[0].get_title()
    assert f"{raw['Gurobi_time_sec'].max():.2f} s" in title


def test_example_instances_draw_optimal_cuts(namespace):
    """The red arcs are the stored cuts, and those cuts really are optimal."""
    import network_interdiction as ni
    import results_plots as rp
    mid = read('mid_instances.csv')
    for ax, (instance, B) in zip(namespace['fig_instances'].axes, rp.EXAMPLE_CASES):
        row = mid[(mid['instance'] == instance) & (mid['B'] == B)].iloc[0]
        cut = json.loads(row['optimal_cut'])
        red = [patch for patch in ax.patches
               if matplotlib.colors.to_hex(patch.get_edgecolor()) == rp.C_ALERT.lower()]
        assert len(red) == len(cut) <= B
        di_g, src, tgt = ni.read_flow_file(os.path.join(REPO_ROOT, instance))
        assert nx.maximum_flow_value(di_g, src, tgt, capacity='weight') == row['maxflow']
        di_g.remove_edges_from(map(tuple, cut))
        assert nx.maximum_flow_value(di_g, src, tgt, capacity='weight') == row['Opt_NIP']
        assert f"{row['maxflow']} → {row['Opt_NIP']}" in ax.get_title()


def test_qubit_count_formula_matches_the_built_qubos():
    """V - 2 + 3E, used for the large instances, matches every compiled QUBO."""
    import results_plots as rp
    mid, tiny = read('mid_instances.csv'), read('qaoa_tiny.csv')
    np.testing.assert_array_equal(rp.qubo_qubits(mid['V'], mid['E']), mid['qubits'])
    np.testing.assert_array_equal(rp.qubo_qubits(tiny['V'], tiny['E']), tiny['qubits'])


def test_problem_sizes_mark_the_fitted_pieces(namespace, raw):
    decomposition = read('decomposition_large.csv')
    fitted = decomposition[(decomposition['setting'] == 'fit156')
                           & decomposition['flow'].notna()]
    largest = fitted.groupby('instance')['max_piece_qubits'].max()
    ax = namespace['fig_sizes'].axes[0]
    hollow = [c.get_offsets()[0, 1] for c in ax.collections
              if len(c.get_facecolors()) and np.allclose(c.get_facecolors()[0], [1, 1, 1, 1])]
    assert sorted(hollow) == sorted(largest)
    assert max(hollow) <= 155


def test_qaoa_depth_lines_and_title(namespace):
    outcomes = read('qaoa_outcomes.csv')
    tiny = outcomes[outcomes['source'] == 'tiny']
    left = namespace['fig_depth'].axes[0]
    assert len(left.lines) == tiny['case'].nunique() == 6
    for line, (_case, sub) in zip(left.lines, tiny.groupby('case', sort=False)):
        sub = sub.sort_values('reps')
        np.testing.assert_array_equal(line.get_xdata(), sub['reps'])
        np.testing.assert_allclose(line.get_ydata(), sub['p_optimal'])
    mean = tiny.groupby('reps')[['p_optimal', 'p_empty']].mean()
    title = namespace['fig_depth']._suptitle.get_text()
    for value in (mean.loc[0, 'p_optimal'], mean.loc[3, 'p_optimal'],
                  mean.loc[0, 'p_empty'], mean.loc[3, 'p_empty']):
        assert f'{value:.2f}' in title


def test_transfer_bars_are_method_means(namespace):
    outcomes = read('qaoa_outcomes.csv')
    transfer = outcomes[outcomes['source'] == 'transfer']
    order = [('uniform', 0), ('fixed', None), ('ibm', None), ('gat', None),
             ('optimized', None), ('rws', 0), ('rws', None)]
    for ax, depth in zip(namespace['fig_transfer'].axes, (1, 2)):
        optimal, _suboptimal, empty, _over = ax.containers
        for bar_optimal, bar_empty, (method, reps) in zip(optimal, empty, order):
            sub = transfer[(transfer['method'] == method)
                           & (transfer['reps'] == (depth if reps is None else reps))]
            assert len(sub) == 6
            assert bar_optimal.get_height() == pytest.approx(sub['p_optimal'].mean())
            assert bar_empty.get_height() == pytest.approx(sub['p_empty'].mean())
        dots = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
        expected = np.concatenate([
            transfer[(transfer['method'] == method)
                     & (transfer['reps'] == (depth if reps is None else reps))]['p_optimal']
            for method, reps in order])
        np.testing.assert_allclose(np.sort(dots), np.sort(expected))


def test_mid_qaoa_points(namespace):
    qaoa_mid = read('qaoa_mid.csv')
    for ax, column in zip(namespace['fig_mid'].axes, ('p_feasible', 'shot_p_optimal')):
        points = np.concatenate([c.get_offsets() for c in ax.collections])
        expected = qaoa_mid[['qubits', column]].to_numpy(dtype=float)
        np.testing.assert_allclose(points[np.lexsort(points.T[::-1])],
                                   expected[np.lexsort(expected.T[::-1])])
    title = namespace['fig_mid']._suptitle.get_text()
    assert f"{qaoa_mid.loc[qaoa_mid['qubits'] > 100, 'p_feasible'].max():.1%}" in title


def test_mid_summary_counts(namespace):
    qaoa_mid = read('qaoa_mid.csv')
    summary = namespace['mid_summary']
    for (angles, reps), sub in qaoa_mid.groupby(['angles', 'reps']):
        assert summary.loc[(angles, reps), 'runs'] == len(sub)
        assert summary.loc[(angles, reps), 'optimal_after_repair'] == (
            sub['flow_after_repair'] == sub['opt_nip']).sum()
    # Repair is not QAOA's credit here: greedy alone is optimal on every case.
    mid = read('mid_instances.csv')
    assert (mid['Greedy'] == mid['Opt_NIP']).all()


def test_decomposition_medians_and_title(namespace, raw):
    decomposition = read('decomposition_large.csv')
    iterated = decomposition[(decomposition['mode'] == 'iterated')
                             & decomposition['flow'].notna()]
    greedy = 100 * (raw['Greedy'] / raw['Opt_NIP'] - 1).median()
    for ax, setting, rules in (
            (namespace['fig_decomposition'].axes[0], 'pieces4',
             ['mincut', 'local', 'marginal', 'marginal+refine']),
            (namespace['fig_decomposition'].axes[1], 'fit156',
             ['mincut', 'local', 'marginal'])):
        sub = iterated[iterated['setting'] == setting]
        expected = [100 * (sub.loc[sub['rule'] == rule, 'flow']
                           / sub.loc[sub['rule'] == rule, 'optimum'] - 1).median()
                    for rule in rules] + [greedy]
        np.testing.assert_allclose([line.get_ydata()[0] for line in ax.lines], expected)
    refined = iterated[iterated['rule'] == 'marginal+refine']
    reached = (refined['flow'] == refined['optimum']).sum()
    assert f'{reached}/{len(refined)}' in namespace['fig_decomposition']._suptitle.get_text()


# ── helpers kept from the earlier notebook ──────────────────────────────────
def test_head_to_head_counts_match_the_csv(legacy_tables, raw):
    with_qubo = raw.dropna(subset=['SA_NIP_Greedy'])
    expected = {
        'rows': len(with_qubo),
        'SA+repair better': int((with_qubo['SA_NIP_Greedy']
                                 < with_qubo['Greedy']).sum()),
        'tie': int((with_qubo['SA_NIP_Greedy'] == with_qubo['Greedy']).sum()),
        'Greedy better': int((with_qubo['SA_NIP_Greedy']
                              > with_qubo['Greedy']).sum()),
        'Greedy optimal': int((with_qubo['Greedy']
                               == with_qubo['Opt_NIP']).sum()),
        'SA+repair optimal': int((with_qubo['SA_NIP_Greedy']
                                  == with_qubo['Opt_NIP']).sum()),
    }
    reported = legacy_tables['summary'].loc['all']
    for key, value in expected.items():
        assert reported[key] == value, key


def test_slide_repaired_values_match_plain_greedy(legacy_tables):
    """
    The slides' repaired bars are plain greedy, to the ~0.01 precision the
    figures can be read at, and they are not our SA + repair.
    """
    cmp = legacy_tables['cmp']
    assert len(cmp) == 27
    assert cmp['gap_vs_greedy'].max() <= 0.01
    assert cmp['gap_vs_sa_repair'].max() > 0.1


def test_qaoa_table_joins_the_two_csvs_row_by_row(legacy_tables):
    """
    utils.qaoa_vs_classical_table puts results/qaoa_tiny.csv beside
    results/sample_results_sa.csv. Each cell is checked against the raw file
    it comes from, matched on (instance, B), so a join that shifted rows or
    columns cannot pass.
    """
    table = legacy_tables['qaoa_table']
    reps = legacy_tables['QAOA_TABLE_DEPTH']
    qaoa = pandas.read_csv(QAOA_CSV, comment='#')
    qaoa = qaoa[qaoa['reps'] == reps]
    sample = pandas.read_csv(SAMPLE_CSV, comment='#')
    assert len(table) == len(qaoa) == len(sample)
    for _, q in qaoa.iterrows():
        name = os.path.splitext(os.path.basename(q['instance']))[0]
        row = table.loc[f"{name} B={q['B']}"]
        s = sample[(sample['instance'] == q['instance'])
                   & (sample['B'] == q['B'])].iloc[0]
        assert row['qubits'] == q['qubits']
        assert row['QAOA + repair'] == q['flow_after_repair']
        assert row['P(optimal), QAOA'] == q['p_optimal']
        assert row['P(optimal), uniform'] == q['uniform_p_optimal']
        assert row['optimum'] == s['Opt_NIP']
        assert row['greedy'] == s['Greedy']
        assert row['SA + repair'] == s['SA_NIP_Greedy']


def test_qaoa_enrichment_is_relative_to_the_uniform_state(legacy_tables):
    qaoa = legacy_tables['qaoa']
    raw = pandas.read_csv(QAOA_CSV, comment='#')
    pandas.testing.assert_series_equal(
        qaoa['enrichment'], raw['p_optimal'] / raw['uniform_p_optimal'],
        check_names=False)
    donor = raw['transfer_p_optimal'].isna()
    assert (raw.loc[donor, 'instance'] == raw['instance'].iloc[0]).all()
    pandas.testing.assert_series_equal(
        qaoa.loc[~donor, 'transfer_enrichment'],
        raw.loc[~donor, 'transfer_p_optimal']
        / raw.loc[~donor, 'uniform_p_optimal'], check_names=False)


def test_original_objective_metric_definitions():
    from utils import objective_metrics
    measured = objective_metrics(pandas.Series([12., 10., 30., 0., np.nan]),
                                 pandas.Series([10., 10., 10., 0., 10.]))
    np.testing.assert_allclose(measured['approx_ratio_minus_1'],
                               [0.2, 0., 2., np.nan, np.nan], equal_nan=True)
    # A twofold degradation is 0%, not 50%: this is the reference
    # notebook's definition, not the reciprocal ratio or flow reduction.
    np.testing.assert_allclose(measured['performance_pct'],
                               [80., 100., -100., np.nan, np.nan], equal_nan=True)


def test_transfer_metrics_ignore_hamiltonian_scores(tmp_path):
    from utils import load_transfer_objectives
    path = tmp_path / 'transfer.csv'
    pandas.DataFrame({'flow_after_repair': [12., 10.], 'opt_nip': [10., 10.],
                      'expected_energy': [-1000., 1000.],
                      'gap_closed': [1., 0.], 'p_optimal': [0., 1.]}).to_csv(path, index=False)
    measured = load_transfer_objectives(path)
    np.testing.assert_allclose(measured['performance_pct'], [80., 100.])
    np.testing.assert_allclose(measured['approx_ratio_minus_1'], [0.2, 0.])

"""
Runs notebooks/analysis.ipynb's code cells and checks the
numbers it reports against counts computed independently from the raw CSV,
so the notebook cannot silently drift from the data it describes.
"""
import json
import os
from unittest import mock

import matplotlib
import numpy as np
import pandas
import pytest

matplotlib.use('Agg')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO_ROOT, 'notebooks', 'analysis.ipynb')
SLIDE_CSV = os.path.join(REPO_ROOT, 'results', 'slide_instances_sa.csv')
QAOA_CSV = os.path.join(REPO_ROOT, 'results', 'qaoa_tiny.csv')
SAMPLE_CSV = os.path.join(REPO_ROOT, 'results', 'sample_results_sa.csv')


@pytest.fixture(scope='module')
def namespace():
    """
    Execute every code cell in order, from the repo root, and return the
    resulting globals. Needs no Jupyter kernel: IPython magics (``%...``
    lines) are dropped, and plt.savefig is stubbed so a test run does not
    rewrite the committed figures/.
    """
    with open(NOTEBOOK, encoding='utf-8') as handle:
        cells = json.load(handle)['cells']
    namespace = {}
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        with mock.patch('matplotlib.pyplot.savefig'):
            for cell in cells:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source']).splitlines()
                    code = '\n'.join(line for line in source
                                      if not line.lstrip().startswith('%'))
                    exec(code, namespace)
    finally:
        os.chdir(cwd)
    return namespace


@pytest.fixture(scope='module')
def raw():
    return pandas.read_csv(SLIDE_CSV, comment='#')


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


def test_raw_flow_columns_are_never_overwritten(namespace, raw):
    """
    The derived ratio columns once reused the CSV's own column names, which
    overwrote the flows and turned every count comparing flows into nonsense.
    """
    df = namespace['df']
    for column in ('Greedy', 'SA_NIP', 'SA_NIP_Greedy', 'Random_Greedy'):
        pandas.testing.assert_series_equal(df[column], raw[column],
                                           check_names=False)


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
    The notebook's central claim: the slides' repaired bars are plain greedy,
    to the ~0.01 precision the figures can be read at, and they are not our
    SA + repair.
    """
    cmp = legacy_tables['cmp']
    assert len(cmp) == 27
    assert cmp['gap_vs_greedy'].max() <= 0.01
    assert cmp['gap_vs_sa_repair'].max() > 0.1


def test_qaoa_table_joins_the_two_csvs_row_by_row(legacy_tables):
    """
    The QAOA section's table puts results/qaoa_tiny.csv beside
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


def test_transfer_notebook_reports_repaired_solution_tie(namespace):
    data = namespace['transfer']
    assert len(data) == 60
    assert (data['flow_after_repair'] == data['opt_nip']).all()
    assert (data['performance_pct'] == 100).all()
    for ax in namespace['fig_transfer'].axes:
        np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata(), dtype=float),
                                   np.zeros(5))


def test_trajectory_uses_original_objective_gap(namespace, raw):
    row = raw[raw['instance'].str.endswith('100_6.2dnnflow') & (raw['B'] == 6)].iloc[0]
    history = json.loads(row['Gurobi_trajectory'])
    expected = np.minimum.accumulate([value for _, value in history]) / row['Opt_NIP'] - 1
    plotted = namespace['fig_trajectory'].axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(plotted, np.append(expected, expected[-1]))


def test_performance_distribution_uses_original_objectives(namespace, raw):
    # Verify every suboptimal scatter value, independently of the metric helper.
    gaps = []
    for column in ('Greedy', 'SA_NIP_Greedy', 'Random_Greedy'):
        valid = raw[(raw['Opt_NIP'] != 0) & raw[column].notna()]
        values = 100 * (valid[column] - valid['Opt_NIP']) / valid['Opt_NIP'].abs()
        gaps.extend(values[values > 0.001])
    plotted = []
    for collection in namespace['fig_performance'].axes[1].collections:
        plotted.extend(collection.get_offsets()[:, 1])
    np.testing.assert_allclose(sorted(plotted), sorted(gaps))

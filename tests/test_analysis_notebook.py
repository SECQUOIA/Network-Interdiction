"""
Runs notebooks/analysis.ipynb's code cells and checks the
numbers it reports against counts computed independently from the raw CSV,
so the notebook cannot silently drift from the data it describes.
"""
import json
import os
from unittest import mock

import matplotlib
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


def test_raw_flow_columns_are_never_overwritten(namespace, raw):
    """
    The derived ratio columns once reused the CSV's own column names, which
    overwrote the flows and turned every count comparing flows into nonsense.
    """
    df = namespace['df']
    for column in ('Greedy', 'SA_NIP', 'SA_NIP_Greedy', 'Random_Greedy'):
        pandas.testing.assert_series_equal(df[column], raw[column],
                                           check_names=False)


def test_head_to_head_counts_match_the_csv(namespace, raw):
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
    reported = namespace['summary'].loc['all']
    for key, value in expected.items():
        assert reported[key] == value, key


def test_slide_repaired_values_match_plain_greedy(namespace):
    """
    The notebook's central claim: the slides' repaired bars are plain greedy,
    to the ~0.01 precision the figures can be read at, and they are not our
    SA + repair.
    """
    cmp = namespace['cmp']
    assert len(cmp) == 27
    assert cmp['gap_vs_greedy'].max() <= 0.01
    assert cmp['gap_vs_sa_repair'].max() > 0.1


def test_qaoa_table_joins_the_two_csvs_row_by_row(namespace):
    """
    The QAOA section's table puts results/qaoa_tiny.csv beside
    results/sample_results_sa.csv. Each cell is checked against the raw file
    it comes from, matched on (instance, B), so a join that shifted rows or
    columns cannot pass.
    """
    table = namespace['qaoa_table']
    reps = namespace['QAOA_TABLE_DEPTH']
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


def test_qaoa_enrichment_is_relative_to_the_uniform_state(namespace):
    qaoa = namespace['qaoa']
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

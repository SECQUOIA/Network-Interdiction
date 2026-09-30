"""
Runs notebooks/analysis.ipynb's code cells and checks the
numbers it reports against counts computed independently from the raw CSV,
so the notebook cannot silently drift from the data it describes.
"""
import json
import os

import matplotlib
import pandas
import pytest

matplotlib.use('Agg')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO_ROOT, 'notebooks', 'analysis.ipynb')
SLIDE_CSV = os.path.join(REPO_ROOT, 'results', 'slide_instances_sa.csv')


@pytest.fixture(scope='module')
def namespace():
    """Execute every code cell in order, from the repo root, and return the
    resulting globals. Needs no Jupyter kernel."""
    with open(NOTEBOOK, encoding='utf-8') as handle:
        cells = json.load(handle)['cells']
    namespace = {}
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        for cell in cells:
            if cell['cell_type'] == 'code':
                exec(''.join(cell['source']), namespace)
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
    assert cmp['|slide repaired − our Greedy|'].max() <= 0.01
    assert cmp['|slide repaired − our SA+repair|'].max() > 0.1

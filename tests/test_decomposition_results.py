"""
results/decomposition_slide.csv: the eight-thread run of
run_decomposition_benchmark.py --refine on the 27 slide cases (source
1ce5438). Checks its numbers against results/slide_instances_sa.csv and
against themselves, since the cut sets are not stored.
"""
import json
import os

import numpy as np
import pandas
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECOMPOSITION_CSV = os.path.join(REPO_ROOT, 'results', 'decomposition_slide.csv')
SLIDE_CSV = os.path.join(REPO_ROOT, 'results', 'slide_instances_sa.csv')
RULES = {'mincut', 'local', 'marginal', 'marginal+refine', 'no fit'}


@pytest.fixture(scope='module')
def runs():
    return pandas.read_csv(DECOMPOSITION_CSV, comment='#')


@pytest.fixture(scope='module')
def solved(runs):
    return runs[runs['rule'] != 'no fit']


def test_header_names_the_clean_source_commit():
    with open(DECOMPOSITION_CSV, encoding='utf-8') as handle:
        header = [line for line in handle if line.startswith('#')]
    assert '# repo commit: 1ce5438c3478bf6da8b62abc3edd61a59f522870\n' in header
    assert not any('/mnt/' in line or 'rames102' in line for line in header)


def test_covers_the_27_slide_cases_with_their_references(runs):
    slide = pandas.read_csv(SLIDE_CSV, comment='#')
    cases = runs[['instance', 'B']].drop_duplicates()
    assert len(cases) == 27
    merged = runs.merge(slide[['instance', 'B', 'Opt_NIP', 'Greedy', 'maxflow']],
                        on=['instance', 'B'], how='left',
                        suffixes=('', '_slide'), validate='many_to_one')
    assert merged['Opt_NIP'].notna().all()
    assert (merged['optimum'] == merged['Opt_NIP']).all()
    assert (merged['greedy'] == merged['Greedy']).all()
    assert (merged['maxflow'] == merged['maxflow_slide']).all()


def test_rows_per_setting(runs):
    assert set(runs['rule']) == RULES
    assert set(runs['setting']) == {'pieces4', 'fit156'}
    # Only 125.planarflow has no partition that fits 155 qubits.
    no_fit = runs[runs['rule'] == 'no fit']
    assert set(no_fit['instance']) == {'testcases/125.planarflow'}
    assert no_fit['setting'].eq('fit156').all() and no_fit['flow'].isna().all()
    # Every case has one refined row, for the four-piece partition.
    refined = runs[runs['rule'] == 'marginal+refine']
    assert len(refined) == 27 and refined['setting'].eq('pieces4').all()


def test_flows_lie_between_the_optimum_and_the_max_flow(solved):
    assert (solved['flow'] >= solved['optimum']).all()
    assert (solved['flow'] <= solved['maxflow']).all()
    expected = ((solved['maxflow'] - solved['flow'])
                / (solved['maxflow'] - solved['optimum']))
    np.testing.assert_allclose(solved['ratio'], expected, atol=1e-6)


def test_piece_budgets_respect_the_total(solved):
    budgets = solved['budgets'].map(json.loads)
    assert (budgets.map(len) == solved['pieces']).all()
    assert (budgets.map(sum) <= solved['B']).all()
    assert budgets.map(lambda split: min(split) >= 0).all()


def test_fit156_pieces_fit_a_heron_device(solved):
    fitted = solved[solved['setting'] == 'fit156']
    assert (fitted['max_piece_qubits'] <= 155).all()
    assert (solved.loc[solved['setting'] == 'pieces4', 'pieces'] == 4).all()


def test_refinement_never_loses_to_its_starting_split(solved):
    """--refine starts from the marginal split and keeps the best it sees."""
    pieces4 = solved[solved['setting'] == 'pieces4']
    start = pieces4[(pieces4['rule'] == 'marginal')
                    & (pieces4['mode'] == 'iterated')].set_index(['instance', 'B'])
    refined = pieces4[pieces4['rule'] == 'marginal+refine'].set_index(['instance', 'B'])
    assert (refined['flow'] <= start.loc[refined.index, 'flow']).all()

"""
Tests for the results CSV pipeline (issue #6): run_benchmark.py produces it,
network_interdiction_plot.py consumes it, and the checked-in sample is really
what run_benchmark.py computed rather than hand-typed numbers.
"""
import os
import subprocess
import sys

import pandas
import pytest

import network_interdiction as ni
import network_interdiction_plot as plot
import run_benchmark as rb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO_ROOT, plot.DEFAULT_CSV)


def instance_path(row):
    """Absolute path of the instance a sample row was computed from."""
    return os.path.join(REPO_ROOT, row['instance'])


@pytest.fixture(scope='module')
def sample():
    return pandas.read_csv(SAMPLE_CSV, comment='#')


def test_sample_csv_carries_every_column_the_plot_reads(sample):
    """
    The writer and the reader agree on column names.

    This is the drift the issue was about: the plot read columns nothing in
    the repo produced, so a rename on either side has to fail here.
    """
    missing = [c for c in rb.PLOT_COLUMNS if c not in sample.columns]
    assert not missing, f"sample CSV is missing {missing}"


def test_sample_csv_exact_columns_are_reproducible(sample):
    """
    Recompute Opt_NIP, Opt_QUBO and maxflow for every checked-in row from the
    instance file it names, and require the committed numbers to match.

    Committed result files can be edited to look plausible, so this reruns
    the exact backend rather than reading the numbers back.
    """
    assert len(sample) > 0
    for _, row in sample.iterrows():
        di_g, src_node, tgt_node = ni.read_flow_file(instance_path(row))
        B = int(row['B'])
        curr_λ = float(row['lambda'])

        assert len(di_g.nodes()) == row['V']
        assert len(di_g.edges()) == row['E']
        assert rb.max_flow(di_g, src_node, tgt_node) == row['maxflow']

        opt_nip, opt_qubo = rb.exact_reference(
            'brute', di_g, src_node, tgt_node, B, curr_λ)
        assert opt_nip == row['Opt_NIP']
        assert opt_qubo == row['Opt_QUBO']


def test_sample_rows_leave_a_flow_reduction_to_normalise_by(sample):
    """
    Every ratio the plot draws is divided by (maxflow - Opt_NIP)/maxflow, so a
    row whose budget buys no reduction at all would make the whole figure NaN.
    """
    assert (sample['Opt_NIP'] < sample['maxflow']).all()


def test_sample_rows_avoid_the_zero_sentinel(sample):
    """
    The plot reads a 0 in SA_NIP/QA_NIP as "this solver produced nothing for
    this row" and draws a zero-height bar. A row whose genuine optimum is 0
    would therefore be indistinguishable from a missing measurement.
    """
    assert (sample['Opt_NIP'] > 0).all()


def test_brute_force_nip_optimum_matches_hand_computed_answer():
    """
    tiny_a.flow is the two-path fixture from test_network_interdiction.py:
    max flow 15, and with one interdicted edge the best achievable is 5, by
    cutting either edge of the capacity-10 path.
    """
    path = os.path.join(REPO_ROOT, 'testcases', 'tiny_a.flow')
    di_g, src_node, tgt_node = ni.read_flow_file(path)

    assert rb.max_flow(di_g, src_node, tgt_node) == 15
    obj, edges = rb.brute_force_nip_optimum(di_g, src_node, tgt_node, 1)
    assert obj == 5
    assert edges in ([(0, 1)], [(1, 3)])


def test_exact_reference_rejects_a_lambda_that_does_not_encode_the_instance():
    """
    Penalty coefficients are not free (issue #2): at lambda=20 on tiny_a the
    QUBO's ground state no longer encodes the NIP optimum, and a row built at
    that lambda would compare the annealers against the wrong target.
    """
    path = os.path.join(REPO_ROOT, 'testcases', 'tiny_a.flow')
    di_g, src_node, tgt_node = ni.read_flow_file(path)

    # The coefficient the sample row actually uses is accepted...
    rb.exact_reference('brute', di_g, src_node, tgt_node, 1, 6.0)

    # ...while one that breaks the encoding is refused rather than written out.
    with pytest.raises(ValueError, match='does not encode this instance'):
        rb.exact_reference('brute', di_g, src_node, tgt_node, 1, 20.0)


def test_brute_force_qubo_optimum_refuses_an_oversized_instance():
    """The 2**n enumeration is capped instead of exhausting memory."""
    path = os.path.join(REPO_ROOT, 'testcases', 'tiny_a.flow')
    di_g, src_node, tgt_node = ni.read_flow_file(path)

    with pytest.raises(ValueError, match='--max-exact-vars'):
        rb.brute_force_qubo_optimum(di_g, src_node, tgt_node, 1, 6.0,
                                    max_exact_vars=4)


def test_run_case_fills_the_sa_columns_and_leaves_qa_empty():
    """
    End-to-end for one row: the SA pass fills its three columns, the QA pass
    was not requested so its columns stay empty (not 0, which the plot reads
    as data), and on an instance this small SA reaches the exact optimum.
    """
    path = os.path.join(REPO_ROOT, 'testcases', 'tiny_a.flow')
    row = rb.run_case(path, B=1, curr_λ=6.0, exact_backend='brute',
                      solvers=['sa'])

    assert set(row) == set(rb.CSV_COLUMNS)
    assert row['Opt_NIP'] == 5
    assert row['SA_NIP'] == row['Opt_NIP']
    assert row['SA_NIP_Greedy'] == row['Opt_NIP']
    assert row['SA_QUBO'] == row['Opt_QUBO']
    for column in rb.SOLVER_COLUMNS['qa']:
        assert row[column] == ''


def test_available_solvers_skips_the_unrun_pass(sample):
    assert plot.available_solvers(sample) == ['SA']


def test_compute_series_scores_the_sample_against_the_optimum(sample):
    """
    SA matches the exact optimum on every sample row, so both ratios the plot
    draws must come out at exactly 1.0 -- anything else means the ratio
    arithmetic drifted.
    """
    _, flow_reduction, energy_ratio = plot.compute_series(sample, ['SA'])
    assert list(flow_reduction['SA']) == [1.0] * len(sample)
    assert list(flow_reduction['SA-Greedy']) == [1.0] * len(sample)
    assert list(energy_ratio['SA']) == [1.0] * len(sample)


def test_compute_series_labels_axes_with_integer_counts(sample):
    """
    #Variables (#Interdicted Edges), not '16.0 (1.0)'.

    Checked on a CSV carrying only the columns the plot itself reads, as well
    as on the sample: a row-wise read of the sample survives because its
    string `instance` column forces object dtype, while an all-numeric frame
    (any CSV predating that column) upcasts the counts to float.
    """
    numeric_only = sample[list(rb.PLOT_COLUMNS)]
    assert numeric_only.dtypes.map(pandas.api.types.is_numeric_dtype).all()

    for frame in (sample, numeric_only):
        x_labels, _, _ = plot.compute_series(frame, ['SA'])
        assert x_labels[0] == '16 (1)'
        assert len(x_labels) == len(frame)


def test_plot_script_runs_standalone(tmp_path):
    """
    The regression the issue reports: running the plotting script with no
    arguments must produce both figures, without a results CSV having to be
    supplied out of band.
    """
    env = dict(os.environ, MPLBACKEND='Agg')
    result = subprocess.run(
        [sys.executable, 'network_interdiction_plot.py',
         '--outdir', str(tmp_path)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    stem = os.path.splitext(os.path.basename(plot.DEFAULT_CSV))[0]
    for suffix in ('flow_reduction', 'qubo_energy'):
        produced = tmp_path / f'{stem}_{suffix}.pdf'
        assert produced.exists(), f"{produced} was not written"
        assert produced.stat().st_size > 0


def test_plot_script_reports_a_csv_it_cannot_use(tmp_path):
    """A CSV without the plot's columns fails with a pointer, not a KeyError."""
    unusable = tmp_path / 'unusable.csv'
    unusable.write_text('V,E,B\n4,4,1\n')

    env = dict(os.environ, MPLBACKEND='Agg')
    result = subprocess.run(
        [sys.executable, 'network_interdiction_plot.py',
         '--csv', str(unusable), '--outdir', str(tmp_path)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)

    assert result.returncode != 0
    assert 'missing required column' in result.stderr
    assert 'run_benchmark.py' in result.stderr

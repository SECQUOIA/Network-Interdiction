"""
Tests for the classical baseline passes in run_benchmark.py (issue #15):
greedy-from-scratch and random-start-plus-greedy-repair, the controls that
separate what an annealer sample contributes from what the greedy repair
recovers on its own.
"""
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pandas
import pytest

import network_interdiction as ni
import network_interdiction_plot as plot
import run_benchmark as rb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny(name):
    return os.path.join(REPO_ROOT, 'testcases', f'{name}.flow')


def build_greedy_trap():
    """
    A 5-node instance on which greedy-from-scratch is not optimal at B=2.

    Max flow is 12. The best single cuts are (0,3) and (1,4), both leaving 4;
    greedy takes (0,3) because ties go to the first arc in sorted order, and
    from there the best second cut, (0,1), leaves 2. The optimal pair is
    {(1,4), (2,4)}, leaving 1 -- it starts from the *other* tied arc. So this
    failure runs through a tie-break: it shows greedy is not an exact solver,
    not that greedy is generally far from optimal. Found by random search;
    it was the only such instance among ~3,400 random 5-6 node graphs, which
    is the point of issue #15 -- on small random instances greedy is nearly
    always optimal, so it is the bar an annealer has to clear.
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(5))
    for u, v, w in [(0, 1, 2), (0, 2, 2), (0, 3, 11), (1, 2, 3),
                    (1, 4, 12), (2, 4, 3), (3, 1, 7), (3, 4, 1)]:
        g.add_edge(u, v, weight=w)
    return g, 0, 4


def test_greedy_is_not_an_exact_solver_in_disguise():
    g, src, tgt = build_greedy_trap()

    assert rb.max_flow(g, src, tgt) == 12
    optimum, optimal_set = rb.brute_force_nip_optimum(g, src, tgt, 2)
    assert optimum == 1
    assert sorted(optimal_set) == [(1, 4), (2, 4)]

    assert rb.greedy_from_scratch(g, src, tgt, 2) == 2


def test_greedy_matches_the_optimum_where_the_greedy_order_is_right():
    """On tiny_b (three disjoint paths) greedy cuts the two widest paths."""
    g, src, tgt = ni.read_flow_file(tiny('tiny_b'))
    assert rb.greedy_from_scratch(g, src, tgt, 1) == 13
    assert rb.greedy_from_scratch(g, src, tgt, 2) == 5


def test_random_greedy_is_reproducible_from_its_seed():
    g, src, tgt = build_greedy_trap()
    first = rb.random_greedy(g, src, tgt, 2, np.random.default_rng(7))
    again = rb.random_greedy(g, src, tgt, 2, np.random.default_rng(7))
    assert first == again


def test_random_greedy_returns_a_budget_feasible_value():
    """
    Every repaired sample interdicts exactly B arcs, so its flow can never be
    below the B-arc optimum, and it is a real interdiction, so it can never
    exceed the uninterdicted max flow.
    """
    g, src, tgt = build_greedy_trap()
    for seed in range(5):
        value = rb.random_greedy(g, src, tgt, 2, np.random.default_rng(seed))
        assert 1 <= value <= 12


def test_random_greedy_draws_budget_sized_samples(monkeypatch):
    """
    The null model is "samples that know the budget and nothing else", so the
    starting interdictions should cut about B arcs, not about E/2 -- a uniform
    bitstring would hand all the work to the repair and measure the repair.
    """
    g, src, tgt = ni.read_flow_file(
        os.path.join(REPO_ROOT, 'testcases', '50.planarflow'))
    num_edges = len(g.edges())
    B = 5
    sizes = []

    def record(di_g, x, src_node, tgt_node, budget):
        sizes.append(int(np.sum(x)))
        return [], 0

    monkeypatch.setattr(ni, 'improve_dwave_solution', record)
    rb.random_greedy(g, src, tgt, B, np.random.default_rng(0),
                     num_samples=200)

    assert len(sizes) == 200
    # Binomial(E, B/E): mean B, far below E/2 for any E worth benchmarking.
    assert abs(np.mean(sizes) - B) < 1.0
    assert max(sizes) < num_edges / 4


def test_run_case_fills_only_the_requested_baselines():
    path = tiny('tiny_a')
    both = rb.run_case(path, 1, 6.0, 'brute', solvers=[],
                       baselines=['greedy', 'random'],
                       rng=np.random.default_rng(0))
    assert both['Greedy'] == 5
    assert both['Random_Greedy'] == 5

    neither = rb.run_case(path, 1, 6.0, 'brute', solvers=[])
    assert neither['Greedy'] == ''
    assert neither['Random_Greedy'] == ''


def test_cli_runs_both_baselines_by_default_and_records_the_seed(tmp_path):
    out = tmp_path / 'baselines.csv'
    rb.main(['--exact', 'brute', '--solvers', 'none',
             '--case', f"{tiny('tiny_b')}:2:6.0", '--out', str(out)])

    text = out.read_text()
    assert '# baselines: greedy, random (seed 0)' in text

    frame = pandas.read_csv(out, comment='#')
    assert frame.loc[0, 'Greedy'] == 5
    assert frame.loc[0, 'Random_Greedy'] >= frame.loc[0, 'Opt_NIP']


def test_sample_csv_baselines_are_drawn(tmp_path):
    sample = pandas.read_csv(os.path.join(REPO_ROOT, plot.DEFAULT_CSV),
                             comment='#')
    baselines = plot.available_baselines(sample)
    assert baselines == ['Greedy', 'Random-Greedy']

    _, flow_reduction, energy_ratio = plot.compute_series(
        sample, ['SA'], baselines)
    # Both baselines reach the exact optimum on every sample row, which is
    # the finding worth knowing about the sample: it cannot separate SA from
    # greedy.
    assert list(flow_reduction['Greedy']) == [1.0] * len(sample)
    assert list(flow_reduction['Random-Greedy']) == [1.0] * len(sample)
    assert 'Greedy' not in energy_ratio


def run_plot(csv_path, outdir):
    env = dict(os.environ, MPLBACKEND='Agg')
    return subprocess.run(
        [sys.executable, 'network_interdiction_plot.py',
         '--csv', str(csv_path), '--outdir', str(outdir)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def test_plot_still_accepts_a_csv_without_baseline_columns(tmp_path):
    """CSVs written before issue #15 have no Greedy/Random_Greedy columns."""
    sample = pandas.read_csv(os.path.join(REPO_ROOT, plot.DEFAULT_CSV),
                             comment='#')
    legacy = tmp_path / 'legacy.csv'
    sample.drop(columns=['Greedy', 'Random_Greedy']).to_csv(legacy,
                                                            index=False)

    result = run_plot(legacy, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'legacy_flow_reduction.pdf').exists()
    assert (tmp_path / 'legacy_qubo_energy.pdf').exists()


def test_plot_draws_a_baselines_only_csv_without_an_energy_figure(tmp_path):
    out = tmp_path / 'only.csv'
    rb.main(['--exact', 'brute', '--solvers', 'none',
             '--case', f"{tiny('tiny_a')}:1:6.0", '--out', str(out)])

    result = run_plot(out, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'only_flow_reduction.pdf').exists()
    assert not (tmp_path / 'only_qubo_energy.pdf').exists()

"""
Best-of-N quality uses the original feasible objective, not QUBO energy.
Checks include direct enumeration of measurement sequences, independent
flow calculations, and re-simulation of one saved state with Statevector.
"""
import importlib
import itertools
import json
import os
from types import SimpleNamespace
from unittest import mock

import networkx as nx
import numpy as np
import pandas
import pytest

import qaoa_common as qc
import qaoa_outcomes as qo
import qaoa_sample_quality as qs


@pytest.fixture(scope='module')
def quality():
    return pandas.read_csv(qs.REPO / 'results' / 'qaoa_sample_quality.csv',
                           comment='#')


def test_expected_best_matches_enumerated_measurement_sequences():
    gaps, probabilities = [0., 0.25, 2.], [0.1, 0.3, 0.6]
    expected = []
    for count in (1, 2, 3, 4):
        expected.append(sum(
            min(gaps[i] for i in sequence)
            * np.prod([probabilities[i] for i in sequence])
            for sequence in itertools.product(range(3), repeat=count)))
    np.testing.assert_allclose(
        qs.expected_best_gap(gaps, probabilities, [1, 2, 3, 4]), expected)


def test_equal_scores_and_large_measurement_counts():
    # Two distinct plans share gap 2. The best is bad only when every
    # measurement misses the optimal plan, with probability (1/2)**N.
    counts = np.array([1, 2, 10, 1000, 4096])
    np.testing.assert_allclose(
        qs.expected_best_gap([2., 0., 2.], [0.2, 0.5, 0.3], counts),
        2 * 0.5 ** counts, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(
        qs.expected_best_gap([1.5], [1.], counts), np.full(5, 1.5))


@pytest.mark.parametrize('gaps, probabilities, counts', [
    ([-0.1, 1.], [0.5, 0.5], [1]),
    ([0., np.nan], [0.5, 0.5], [1]),
    ([0., 1.], [1.], [1]),
    ([0., 1.], [0.2, 0.3], [1]),
    ([0., 1.], [-0.1, 1.1], [1]),
    ([0., 1.], [np.nan, 1.], [1]),
    ([0., 1.], [0.5, 0.5], [0]),
    ([0., 1.], [0.5, 0.5], [1.5]),
])
def test_invalid_scores_or_counts_are_rejected(gaps, probabilities, counts):
    with pytest.raises(ValueError):
        qs.expected_best_gap(gaps, probabilities, counts)


def test_x_marginal_sums_auxiliary_bits_in_the_right_order():
    problem = SimpleNamespace(num_qubits=3, x_qubits=[2, 0])
    probabilities = [0.1, 0.15, 0.2, 0.05, 0.1, 0.05, 0.25, 0.1]
    np.testing.assert_allclose(qo.plan_probabilities(problem, probabilities),
                               [0.3, 0.35, 0.2, 0.15])
    with pytest.raises(ValueError, match='expected 8 probabilities'):
        qo.plan_probabilities(problem, [1.])


def independent_gaps(instance, budget, optimum):
    """Enumerate cuts using NetworkX directly, with no production scorer."""
    import network_interdiction as ni
    graph, source, sink = ni.read_flow_file(qs.REPO / instance)
    edges = list(graph.edges)
    uncut_flow = nx.maximum_flow_value(graph, source, sink, capacity='weight')
    scores = []
    for pattern in range(2 ** len(edges)):
        cut = [edge for e, edge in enumerate(edges) if (pattern >> e) & 1]
        if len(cut) > budget:
            flow = uncut_flow
        else:
            remaining = graph.copy()
            remaining.remove_edges_from(cut)
            flow = nx.maximum_flow_value(remaining, source, sink,
                                         capacity='weight')
        scores.append(flow / optimum - 1)
    return np.array(scores)


def test_over_budget_plans_receive_the_empty_plan_score():
    problem = qc.build_problem('testcases/tiny_a.flow', 1)
    gaps = qs.plan_gaps(problem)
    np.testing.assert_allclose(gaps, independent_gaps(
        problem.instance, problem.B, problem.opt_nip))
    # Cutting every arc would stop all flow, but is illegal. It must have
    # the same (positive) gap as cutting nothing, rather than a negative gap.
    assert gaps[-1] == gaps[0] > 0
    only_illegal = np.zeros(gaps.size)
    only_illegal[-1] = 1
    np.testing.assert_allclose(
        qs.expected_best_gap(gaps, only_illegal, [1, 10, 4096]), gaps[0])


def test_zero_optimum_is_not_given_a_relative_gap():
    with pytest.raises(ValueError, match='positive optimum'):
        qs.plan_gaps(SimpleNamespace(opt_nip=0))


def test_saved_grid_covers_each_case_and_depth_once(quality):
    stored = pandas.read_csv(qo.TINY_CSV, comment='#')
    expected = {(r.instance, r.B, r.reps, count)
                for r in stored.itertuples() for count in qs.MEASUREMENTS}
    expected |= {(r.instance, r.B, 0, count)
                 for r in stored.itertuples() for count in qs.MEASUREMENTS}
    keys = list(zip(quality['instance'], quality['B'], quality['reps'],
                    quality['measurements']))
    assert len(keys) == len(set(keys)) == len(expected)
    assert set(keys) == expected
    for row in quality.itertuples():
        reference = stored[(stored['instance'] == row.instance)
                           & (stored['B'] == row.B)].iloc[0]
        assert row.opt_nip == reference['opt_nip']
        assert row.qubits == reference['qubits']
        assert row.method == ('qaoa' if row.reps else 'uniform')


def test_uniform_scores_match_all_single_shots_and_pairs(quality):
    uniform = quality[quality['reps'] == 0]
    for (instance, budget), group in uniform.groupby(['instance', 'B']):
        gaps = independent_gaps(instance, int(budget), group['opt_nip'].iloc[0])
        saved = group.set_index('measurements')['expected_best_gap']
        assert saved.loc[1] == pytest.approx(gaps.mean())
        assert saved.loc[2] == pytest.approx(
            np.minimum(gaps[:, None], gaps).mean())


def test_saved_gaps_are_bounded_and_improve_with_measurements(quality):
    for _, group in quality.groupby(['case', 'reps']):
        values = group.sort_values('measurements')['expected_best_gap']
        assert (values >= 0).all()
        assert (values <= group['maxflow'] / group['opt_nip'] - 1 + 1e-12).all()
        assert (np.diff(values) <= 1e-12).all()


def test_saved_trained_curve_matches_an_independent_statevector(quality):
    for dependency in ('qiskit', 'qiskit_aer'):
        if os.environ.get('REQUIRE_QAOA_BACKENDS'):
            importlib.import_module(dependency)
        else:
            pytest.importorskip(dependency)
    from qiskit.quantum_info import Statevector
    from qaoa_qiskit import QiskitBackend
    stored = pandas.read_csv(qo.TINY_CSV, comment='#')
    row = stored[(stored['instance'] == 'testcases/tiny_a.flow')
                 & (stored['B'] == 1) & (stored['reps'] == 2)].iloc[0]
    problem = qc.build_problem(row['instance'], 1)
    circuit = QiskitBackend(problem, 2).circuit.assign_parameters(
        json.loads(row['theta']))
    probabilities = Statevector.from_instruction(circuit).probabilities()
    weights = np.zeros(2 ** len(problem.x_qubits))
    for index, probability in enumerate(probabilities):
        bits = f'{index:0{problem.num_qubits}b}'[::-1]
        pattern = sum(int(bits[qubit]) * 2 ** e
                      for e, qubit in enumerate(problem.x_qubits))
        weights[pattern] += probability
    gaps = independent_gaps(row['instance'], 1, row['opt_nip'])
    levels = np.unique(gaps)
    cdf = np.array([weights[gaps <= gap].sum() for gap in levels])
    saved = quality[(quality['instance'] == row['instance'])
                    & (quality['B'] == 1) & (quality['reps'] == 2)]
    for point in saved.itertuples():
        # Alternative CDF calculation: the chance the minimum equals each
        # level, rather than the survival-tail identity in the scorer.
        best_cdf = 1 - np.clip(1 - cdf, 0, 1) ** point.measurements
        expected = levels @ np.diff(np.r_[0, best_cdf])
        assert point.expected_best_gap == pytest.approx(expected, abs=1e-9)


def test_cli_refuses_to_overwrite_before_simulation(tmp_path):
    path = tmp_path / 'existing.csv'
    path.write_text('keep this')
    with mock.patch.object(qs, 'tiny_rows', side_effect=AssertionError):
        with pytest.raises(FileExistsError):
            qs.main(['--out', str(path)])
    assert path.read_text() == 'keep this'

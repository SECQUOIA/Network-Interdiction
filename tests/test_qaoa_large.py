"""
Tests for running the penalty QAOA past the statevector limit: the
reverse Cuthill-McKee qubit order, matrix-product-state simulation, Gurobi
ground energies, and scoring from shots alone (issues #1, #17).

MPS and the reordering are checked against the exact statevector on
tiny_a, where both are available, so neither can drift unnoticed on the
mid-size instances, where only they are. Reuses the independent numpy
reference simulation from tests/test_qaoa.py.
"""
import math
import os

import numpy as np
import pytest

import instances
import network_interdiction as ni
import qaoa_common as qc
import run_benchmark as rb
from test_qaoa import (TINY_A, TRAINED_P1, backend_module, bits_for,
                       dimod_energies, reference_probabilities)
from test_decomposition_refinement import needs_gurobi


@pytest.fixture(scope='module')
def tiny_family():
    return qc.build_problem(TINY_A, 1)


@pytest.fixture(scope='module')
def tiny_rcm():
    return qc.build_problem(TINY_A, 1, order='rcm')


def bandwidth(problem):
    return max(abs(i - j) for i, j in problem.J)


# --- qubit order ---------------------------------------------------------------

def test_rcm_order_is_a_permutation_that_shortens_couplings():
    di_g, src, tgt = instances.make_instance('knn', 18, 2, 0)
    bqm = ni.nip_to_qubo(1, di_g, src, tgt, 5.0, 6.0)
    family, rcm = qc.variable_order(bqm), qc.rcm_order(bqm)
    assert sorted(rcm) == sorted(family)
    spans = []
    for order in (family, rcm):
        position = {label: i for i, label in enumerate(order)}
        spans.append(max(abs(position[u] - position[v])
                         for u, v in bqm.quadratic))
    assert spans[1] < spans[0] / 3


def test_rcm_problem_has_the_same_energies(tiny_family, tiny_rcm):
    """The same Hamiltonian, qubits permuted: equal energy for equal assignments."""
    assert tiny_rcm.variables != tiny_family.variables
    rng = np.random.default_rng(0)
    for _ in range(50):
        assignment = dict(zip(tiny_family.variables,
                              rng.integers(0, 2, tiny_family.num_qubits)))
        energies = [p.bits_energy([assignment[v] for v in p.variables])
                    for p in (tiny_family, tiny_rcm)]
        assert energies[0] == pytest.approx(energies[1])
    assert tiny_rcm.ground_energy == tiny_family.ground_energy


# --- matrix product state --------------------------------------------------------

@pytest.mark.parametrize('order', ['family', 'rcm'])
def test_uncapped_mps_matches_the_statevector(order, tiny_family):
    module = backend_module('qiskit')
    problem = qc.build_problem(TINY_A, 1, order=order)
    theta = np.random.default_rng(1).uniform(-1, 1, 4)
    mps = module.QiskitBackend(problem, 2, method='mps')
    phases = (dimod_energies(tiny_family) - tiny_family.offset) / tiny_family.scale
    expected = reference_probabilities(tiny_family, theta) @ phases
    assert mps.normalized_expectation(theta) == pytest.approx(expected,
                                                              abs=1e-9)


def test_the_bond_dimension_cap_reaches_the_simulator(tiny_rcm, tiny_family):
    module = backend_module('qiskit')
    theta = np.random.default_rng(1).uniform(-1, 1, 4)
    phases = (dimod_energies(tiny_family) - tiny_family.offset) / tiny_family.scale
    exact = reference_probabilities(tiny_family, theta) @ phases
    # In the rcm order tiny_a needs bond dimension 4 at p = 2: a cap of 4 is
    # still exact, a cap of 2 truncates and moves the value (by 0.056).
    at_four = module.QiskitBackend(tiny_rcm, 2, method='mps',
                                   max_bond_dimension=4)
    at_two = module.QiskitBackend(tiny_rcm, 2, method='mps',
                                  max_bond_dimension=2)
    assert at_four.normalized_expectation(theta) == pytest.approx(exact,
                                                                  abs=1e-9)
    assert 0.01 < abs(at_two.normalized_expectation(theta) - exact) < 0.5


def test_mps_shots_decode_in_the_rcm_order(tiny_rcm, tiny_family):
    module = backend_module('qiskit')
    backend = module.QiskitBackend(tiny_rcm, 1, method='mps')
    probabilities = reference_probabilities(tiny_family, TRAINED_P1)
    energies = dimod_energies(tiny_family)
    mean = probabilities @ energies
    shots = 4000
    stderr = math.sqrt(probabilities @ (energies - mean) ** 2 / shots)
    counts = backend.sample(TRAINED_P1, shots, seed=3)
    sample_mean = sum(tiny_rcm.bits_energy(bits) * count
                      for bits, count in counts.items()) / shots
    assert abs(sample_mean - mean) < 5 * stderr


def test_mps_backend_has_no_exact_probabilities(tiny_rcm):
    module = backend_module('qiskit')
    backend = module.QiskitBackend(tiny_rcm, 1, method='mps')
    assert not backend.exact_probabilities
    with pytest.raises(ValueError, match='statevector'):
        backend.probabilities(TRAINED_P1)
    with pytest.raises(ValueError, match='applies to method mps'):
        module.QiskitBackend(tiny_rcm, 1, max_bond_dimension=8)


# --- scores ------------------------------------------------------------------------

def test_shot_scores_count_feasibility_and_flow_ratio(tiny_family):
    # tiny_a, B = 1: max flow 15, optimum 5.
    optimal = bits_for(tiny_family, [1, 0, 0, 0])   # flow 5: ratio 1
    empty = bits_for(tiny_family, [0, 0, 0, 0])     # flow 15: ratio 0
    over = bits_for(tiny_family, [1, 1, 0, 0])      # over budget: scores 0
    scores = qc.score_samples(tiny_family, {optimal: 3, empty: 5, over: 2})
    assert scores['p_feasible'] == pytest.approx(8 / 10)
    assert scores['shot_mean_ratio'] == pytest.approx(3 / 10)


def test_gap_closed_runs_from_uniform_to_ground(tiny_family):
    assert qc.gap_closed(tiny_family, tiny_family.offset) == 0
    assert qc.gap_closed(tiny_family, tiny_family.ground_energy) == 1
    uniform = qc.uniform_scores(tiny_family)['uniform_expected_energy']
    assert qc.gap_closed(tiny_family, uniform) == pytest.approx(0)


@needs_gurobi
def test_gurobi_ground_energy_matches_enumeration(tiny_family):
    _, ground = rb.exact_reference('gurobi', tiny_family.di_g,
                                   tiny_family.src_node, tiny_family.tgt_node,
                                   1, tiny_family.lam)
    assert ground == pytest.approx(dimod_energies(tiny_family).min())


def test_given_angles_skip_training(tiny_family):
    module = backend_module('qiskit')
    backend = module.QiskitBackend(tiny_family, 1)
    result = qc.run_qaoa(backend, tiny_family, 1, shots=200, theta=TRAINED_P1)
    assert result.evaluations == 0
    assert result.theta == pytest.approx(TRAINED_P1)
    probabilities = reference_probabilities(tiny_family, TRAINED_P1)
    assert result.p_optimal == pytest.approx(
        probabilities[tiny_family.optimal_mask()].sum(), abs=1e-9)
    assert result.gap_closed == pytest.approx(qc.gap_closed(
        tiny_family, probabilities @ dimod_energies(tiny_family)), abs=1e-9)


# --- past the statevector limit -----------------------------------------------------

@needs_gurobi
def test_forty_qubit_problem_runs_on_mps_and_scores_from_shots():
    module = backend_module('qiskit')
    di_g, src, tgt = instances.make_instance('knn', 6, 2, 0)
    problem = qc.problem_from_graph(di_g, src, tgt, 1, instance='knn_6',
                                    max_qubits=200, exact_backend='gurobi',
                                    order='rcm')
    assert problem.num_qubits == 40 and not problem.exact
    with pytest.raises(ValueError, match='only problems of up to'):
        problem.energies()
    backend = module.QiskitBackend(problem, 1, method='mps')
    result = qc.run_qaoa(backend, problem, 1, shots=500,
                         theta=qc.linear_ramp(1))
    assert math.isnan(result.p_optimal) and math.isnan(result.approx_ratio)
    assert math.isnan(result.uniform_p_optimal)
    assert result.opt_qubo == problem.ground_energy
    assert 0 < result.gap_closed < 1
    assert result.gap_closed == pytest.approx(
        qc.gap_closed(problem, result.expected_energy))
    assert 0 <= result.shot_p_optimal <= result.p_feasible <= 1
    assert result.flow_after_repair >= problem.opt_nip


def test_command_line_runs_mps(capsys):
    module = backend_module('qiskit')
    assert module.main([TINY_A, '--budget', '1', '--reps', '1', '--method',
                        'mps', '--maxiter', '5', '--shots', '100']) == 0
    out = capsys.readouterr().out
    assert 'gap closed' in out and 'exact scores need' in out


@needs_gurobi
def test_mid_script_rows_for_donor_and_trained_angles(tmp_path):
    backend_module('qiskit')
    import json
    import pandas
    import run_qaoa_mid

    out = tmp_path / 'mid.csv'
    assert run_qaoa_mid.main([
        '--reps', '1', '--angles', 'donor,train', '--only', 'knn_6_2_s0.flow',
        '--shots', '200', '--train-maxiter', '10', '--out', str(out)]) == 0
    frame = pandas.read_csv(out, comment='#')
    assert list(frame.columns) == run_qaoa_mid.COLUMNS
    assert list(frame['angles']) == ['donor', 'train']
    donor = frame.iloc[0]
    assert json.loads(donor['theta']) == pytest.approx(
        run_qaoa_mid.donor_angles()[1], abs=1e-6)
    assert donor['evaluations'] == 0 and frame.iloc[1]['evaluations'] > 0
    for _, row in frame.iterrows():
        assert row['qubits'] == 40
        assert row['gap_closed'] == pytest.approx(
            (row['offset'] - row['expected_energy'])
            / (row['offset'] - row['ground_energy']), abs=1e-5)
        assert 0 <= row['shot_p_optimal'] <= row['p_feasible'] <= 1
        assert row['flow_after_repair'] >= row['opt_nip']


# --- the committed mid-size results -------------------------------------------------

MID_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'qaoa_mid.csv')
MANIFEST_CSV = os.path.join(os.path.dirname(MID_CSV), 'mid_instances.csv')


@pytest.fixture(scope='module')
def mid_results():
    import pandas
    return (pandas.read_csv(MID_CSV, comment='#'),
            pandas.read_csv(MANIFEST_CSV, comment='#'))


def test_mid_results_cover_every_manifest_row(mid_results):
    """Transferred angles on every row and depth; training up to 81 qubits."""
    frame, manifest = mid_results
    rows = set(zip(manifest['instance'], manifest['B']))
    for reps in (1, 2, 3):
        donor = frame[(frame['angles'] == 'donor') & (frame['reps'] == reps)]
        assert set(zip(donor['instance'], donor['B'])) == rows
        trained = frame[(frame['angles'] == 'train') & (frame['reps'] == reps)]
        small = manifest[manifest['qubits'] <= 81]
        assert set(zip(trained['instance'], trained['B'])) == set(
            zip(small['instance'], small['B']))
    assert len(frame) == 3 * len(manifest) + 3 * int((manifest['qubits'] <= 81).sum())


def test_mid_results_agree_with_the_manifest_and_donor(mid_results):
    import json
    import run_qaoa_mid

    frame, manifest = mid_results
    donor = run_qaoa_mid.donor_angles()
    merged = frame.merge(manifest, on=['instance', 'B'], suffixes=('', '_m'))
    assert len(merged) == len(frame)
    assert (merged['qubits'] == merged['qubits_m']).all()
    assert (merged['opt_nip'] == merged['Opt_NIP']).all()
    assert (merged['greedy'] == merged['Greedy']).all()
    for _, row in frame[frame['angles'] == 'donor'].iterrows():
        assert json.loads(row['theta']) == pytest.approx(donor[row['reps']], abs=1e-6)
        assert row['evaluations'] == 0


def test_mid_results_scores_are_self_consistent(mid_results):
    frame, _ = mid_results
    gap = (frame['offset'] - frame['expected_energy']) / (
        frame['offset'] - frame['ground_energy'])
    assert (gap - frame['gap_closed']).abs().max() < 1e-5
    assert ((frame['gap_closed'] > 0) & (frame['gap_closed'] < 1)).all()
    assert (frame['shot_p_optimal'] <= frame['p_feasible']).all()
    assert (frame['flow_after_repair'] >= frame['opt_nip']).all()
    assert (frame['bond_dimension'] == 32).all()


def test_mid_offsets_rebuild_without_gurobi(mid_results):
    """offset depends only on the QUBO, rebuilt here from the manifest's lambda."""
    frame, manifest = mid_results
    for _, row in manifest.iterrows():
        di_g, src, tgt = ni.read_flow_file(os.path.join(
            os.path.dirname(os.path.dirname(MID_CSV)), row['instance']))
        bqm = ni.nip_to_qubo(int(row['B']), di_g, src, tgt, row['lambda'],
                             row['lambda'] + 1)
        _, _, offset = qc.ising_from_bqm(bqm, qc.rcm_order(bqm))
        recorded = frame[(frame['instance'] == row['instance'])
                         & (frame['B'] == row['B'])]['offset']
        assert recorded.to_numpy() == pytest.approx(offset, abs=1e-5)

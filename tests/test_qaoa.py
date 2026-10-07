"""
Tests for the penalty-QAOA (issues #3, #16): the Hamiltonian, decoding and
scoring in qaoa_common, and the Qiskit and CUDA-Q backends.

Each backend is compared with an independent numpy simulation of the same
QAOA (reference_probabilities()), whose cost phases come from dimod's
energies for the BQM rather than from qaoa_common's Ising conversion. A wrong
angle factor, gate sign or bit order in either backend therefore fails here
instead of showing up as a slightly worse QAOA result. Backend tests skip
when that backend is not installed (requirements-qaoa.txt), except under
REQUIRE_QAOA_BACKENDS, which CI's qaoa job sets after installing both.

The fixture is testcases/tiny_a.flow at B = 1 (14 qubits): arcs, in x order,
(0,1) cap 10, (0,2) cap 5, (1,3) cap 10, (2,3) cap 5, source 0, sink 3. Max
flow 15; cutting either 10-arc leaves 5, the optimum, so exactly two of the
16 x patterns are optimal interdictions.
"""
import importlib
import json
import math
import os

import dimod
import numpy as np
import pandas
import pytest

import qaoa_common as qc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_A = os.path.join(REPO_ROOT, 'testcases', 'tiny_a.flow')

# Angles trained once for tiny_a, B = 1, p = 1 (theta = [beta, gamma]),
# pinned so the shot tests do not depend on the optimizer. They concentrate
# the state: <H_C> is about -14.8 against 2.25 for the uniform state.
TRAINED_P1 = [0.439, -0.607]


@pytest.fixture(scope='module')
def problem():
    return qc.build_problem(TINY_A, 1)


def all_bits(num_qubits):
    """Every assignment as rows of per-qubit bits, little-endian row order."""
    index = np.arange(2 ** num_qubits)
    return ((index[:, None] >> np.arange(num_qubits)) & 1).astype(np.int8)


def dimod_energies(problem):
    """QUBO energy of every basis state, straight from dimod."""
    return problem.bqm.energies(
        (all_bits(problem.num_qubits), problem.variables))


def reference_probabilities(problem, theta, energies=None):
    """
    Outcome probabilities of the QAOA state, by direct statevector
    evolution: |+>^n, then per layer the diagonal phase
    exp(-i gamma (E - offset) / scale) and exp(-i beta X) on every qubit.
    E comes from dimod unless `energies` is given (dimod's array is too
    large to build at 21 qubits; problem.energies() is checked against it on
    tiny_a by test_hamiltonian_equals_the_qubo_energy_on_every_state).
    """
    n = problem.num_qubits
    reps = len(theta) // 2
    if energies is None:
        energies = dimod_energies(problem)
    phases = (energies - problem.offset) / problem.scale
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for layer in range(reps):
        beta, gamma = theta[layer], theta[reps + layer]
        psi = psi * np.exp(-1j * gamma * phases)
        cos, sin = np.cos(beta), -1j * np.sin(beta)
        for qubit in range(n):
            view = psi.reshape(2 ** (n - qubit - 1), 2, 2 ** qubit)
            zero, one = view[:, 0, :].copy(), view[:, 1, :].copy()
            view[:, 0, :] = cos * zero + sin * one
            view[:, 1, :] = sin * zero + cos * one
    return np.abs(psi) ** 2


# --- qaoa_common --------------------------------------------------------------

def test_hamiltonian_equals_the_qubo_energy_on_every_state(problem):
    assert problem.num_qubits == 14
    np.testing.assert_allclose(problem.energies(), dimod_energies(problem),
                               rtol=0, atol=1e-9)


def test_qubits_follow_variable_families(problem):
    assert problem.variables[:4] == ['x_0', 'x_1', 'x_2', 'x_3']
    assert problem.variables[4:8] == [f'gamma_{i}' for i in range(4)]
    assert problem.variables[8:12] == [f'w_{i}' for i in range(4)]
    assert problem.variables[12:] == ['pi_1', 'pi_2']
    assert problem.edge_list == [(0, 1), (0, 2), (1, 3), (2, 3)]


def test_auto_lambda_and_ground_state(problem):
    # f = [15, 5, 0]: window (5 - 0, 15 - 5) = (5, 10), midpoint 7.5.
    assert problem.lambda_window == (5.0, 10.0)
    assert problem.lam == 7.5
    assert problem.opt_nip == 5
    exact = dimod.ExactSolver().sample(problem.bqm).first
    assert problem.energies().min() == pytest.approx(exact.energy)


def test_optimal_mask_marks_exactly_the_two_optimal_cuts(problem):
    mask = problem.optimal_mask()
    x_pattern = np.arange(2 ** problem.num_qubits) & 0b1111
    # x_0 alone cuts (0,1), x_2 alone cuts (1,3): both leave 5.
    assert sorted(set(x_pattern[mask])) == [0b0001, 0b0100]
    assert qc.uniform_scores(problem)['uniform_p_optimal'] == 2 / 16


def test_uniform_state_scores(problem):
    scores = qc.uniform_scores(problem)
    assert scores['uniform_expected_energy'] == pytest.approx(problem.offset)
    assert scores['uniform_expected_energy'] == pytest.approx(
        dimod_energies(problem).mean())


def test_a_window_without_interior_is_refused():
    # f = [15, 5, 0, 0, 0]: at B = 3 both ends of the window are 0.
    with pytest.raises(ValueError, match='no interior'):
        qc.build_problem(TINY_A, 3)


def test_a_lambda_outside_the_window_is_refused():
    with pytest.raises(ValueError, match='outside the window'):
        qc.build_problem(TINY_A, 1, lam=1.0)


def bits_for(problem, x):
    """Per-qubit bits with the given x bits and every other variable 0."""
    bits = [0] * problem.num_qubits
    for qubit, value in zip(problem.x_qubits, x):
        bits[qubit] = value
    return tuple(bits)


def test_score_samples_decodes_repairs_and_counts(problem):
    optimal = bits_for(problem, [1, 0, 0, 0])      # cuts (0,1): flow 5
    empty = bits_for(problem, [0, 0, 0, 0])        # no cut: flow 15
    over = bits_for(problem, [1, 1, 0, 0])         # two cuts, over budget
    scores = qc.score_samples(problem, {optimal: 3, empty: 5, over: 2})
    assert scores['flow_before_repair'] == 5
    assert scores['flow_after_repair'] == 5
    assert scores['shot_p_optimal'] == pytest.approx(3 / 10)
    assert scores['best_sample_energy'] == pytest.approx(min(
        problem.bits_energy(bits) for bits in (optimal, empty, over)))

    # Without the optimal sample only the empty one is in budget (15); the
    # repair still reaches 5 from either of the others.
    scores = qc.score_samples(problem, {empty: 5, over: 2})
    assert scores['flow_before_repair'] == 15
    assert scores['flow_after_repair'] == 5
    assert scores['shot_p_optimal'] == 0


# --- backends ----------------------------------------------------------------

BACKEND_DEPENDENCIES = {'qiskit': ('qiskit', 'qiskit_aer'), 'cudaq': ('cudaq',)}


def backend_module(name):
    """
    qaoa_qiskit or qaoa_cudaq, skipping the test when the framework is not
    installed -- unless REQUIRE_QAOA_BACKENDS is set (CI's qaoa job), where a
    missing framework must fail rather than leave the job green untested.
    """
    for dependency in BACKEND_DEPENDENCIES[name]:
        if os.environ.get('REQUIRE_QAOA_BACKENDS'):
            importlib.import_module(dependency)
        else:
            pytest.importorskip(dependency)
    return importlib.import_module(f'qaoa_{name}')


def make_backend(name, problem, reps):
    module = backend_module(name)
    if name == 'qiskit':
        return module.QiskitBackend(problem, reps)
    return module.CudaqBackend(problem, reps)


BACKENDS = ['qiskit', 'cudaq']


@pytest.mark.parametrize('name', BACKENDS)
def test_expectation_matches_reference(name, problem):
    theta = np.random.default_rng(1).uniform(-1, 1, 4)
    backend = make_backend(name, problem, reps=2)
    phases = (dimod_energies(problem) - problem.offset) / problem.scale
    expected = reference_probabilities(problem, theta) @ phases
    assert backend.normalized_expectation(theta) == pytest.approx(
        expected, abs=1e-9)


@pytest.mark.parametrize('name', BACKENDS)
def test_probabilities_match_reference(name, problem):
    theta = np.random.default_rng(2).uniform(-1, 1, 6)
    backend = make_backend(name, problem, reps=3)
    np.testing.assert_allclose(backend.probabilities(theta),
                               reference_probabilities(problem, theta),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', BACKENDS)
def test_samples_decode_in_qubit_order(name, problem):
    # At the trained angles the state is far from uniform, so decoding the
    # shots with the wrong bit order would move their mean energy by about
    # 17, over a hundred standard errors; the right order lands within a few.
    backend = make_backend(name, problem, reps=1)
    probabilities = reference_probabilities(problem, TRAINED_P1)
    energies = dimod_energies(problem)
    mean = probabilities @ energies
    shots = 4000
    stderr = math.sqrt(probabilities @ (energies - mean) ** 2 / shots)

    counts = backend.sample(TRAINED_P1, shots, seed=3)
    assert sum(counts.values()) == shots
    sample_mean = sum(problem.bits_energy(bits) * count
                      for bits, count in counts.items()) / shots
    assert abs(sample_mean - mean) < 5 * stderr

    p_optimal = probabilities[problem.optimal_mask()].sum()
    shot_p = qc.score_samples(problem, counts)['shot_p_optimal']
    assert abs(shot_p - p_optimal) < 5 * math.sqrt(
        p_optimal * (1 - p_optimal) / shots)


@pytest.mark.parametrize('name', BACKENDS)
def test_sampling_is_reproducible_with_a_seed(name, problem):
    backend = make_backend(name, problem, reps=1)
    assert backend.sample(TRAINED_P1, 500, seed=11) == \
        backend.sample(TRAINED_P1, 500, seed=11)


@pytest.mark.parametrize('name', BACKENDS)
def test_training_beats_the_uniform_state(name, problem):
    backend = make_backend(name, problem, reps=1)
    result = qc.run_qaoa(backend, problem, reps=1, shots=1000, seed=0,
                         maxiter=100)
    assert result.backend == name
    assert len(result.theta) == 2
    # The exact scores are those of the reference state at the trained angles.
    probabilities = reference_probabilities(problem, result.theta)
    assert result.expected_energy == pytest.approx(
        probabilities @ dimod_energies(problem), abs=1e-9)
    assert result.p_optimal == pytest.approx(
        probabilities[problem.optimal_mask()].sum(), abs=1e-12)
    assert result.expected_energy < result.uniform_expected_energy - 10
    assert result.p_optimal > 1.5 * result.uniform_p_optimal
    assert result.opt_nip == 5
    assert result.flow_after_repair == 5


def test_backends_prepare_the_same_state(problem):
    qiskit_backend = make_backend('qiskit', problem, reps=3)
    cudaq_backend = make_backend('cudaq', problem, reps=3)
    theta = np.random.default_rng(4).uniform(-1, 1, 6)
    assert qiskit_backend.normalized_expectation(theta) == pytest.approx(
        cudaq_backend.normalized_expectation(theta), abs=1e-9)
    np.testing.assert_allclose(qiskit_backend.probabilities(theta),
                               cudaq_backend.probabilities(theta),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', BACKENDS)
def test_command_line_writes_json(name, tmp_path, capsys):
    module = backend_module(name)
    out = tmp_path / 'result.json'
    # Few iterations: this checks the command line, not the training.
    status = module.main([TINY_A, '--budget', '1', '--reps', '1',
                          '--shots', '200', '--maxiter', '5',
                          '--json', str(out)])
    assert status == 0
    assert f'{name} QAOA, p=1' in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result['backend'] == name
    assert result['num_qubits'] == 14
    assert result['opt_nip'] == 5
    assert result['lam'] == 7.5


def test_benchmark_rows_and_transfer_columns(tmp_path):
    backend_module('qiskit')
    import run_qaoa_benchmark

    out = tmp_path / 'qaoa.csv'
    assert run_qaoa_benchmark.main([
        '--reps', '1', '--case', f'{TINY_A}:1', '--case', f'{TINY_A}:2',
        '--maxiter', '20', '--shots', '200', '--out', str(out)]) == 0
    assert '# transfer donor: ' in out.read_text()
    frame = pandas.read_csv(out, comment='#')
    assert list(frame.columns) == run_qaoa_benchmark.CSV_COLUMNS
    assert list(frame['B']) == [1, 2]
    donor, other = frame.iloc[0], frame.iloc[1]
    assert math.isnan(donor['transfer_p_optimal'])

    # The trained columns score this case's own angles, the transfer columns
    # the donor's, both on this case. theta is stored to 6 decimals.
    problem = qc.build_problem(TINY_A, 2)
    own = reference_probabilities(problem, json.loads(other['theta']))
    donated = reference_probabilities(problem, json.loads(donor['theta']))
    mask = problem.optimal_mask()
    assert other['p_optimal'] == pytest.approx(own[mask].sum(), abs=1e-5)
    assert other['transfer_p_optimal'] == pytest.approx(donated[mask].sum(),
                                                        abs=1e-5)
    assert other['transfer_p_optimal'] != pytest.approx(other['p_optimal'],
                                                        abs=1e-3)


QAOA_CSV = os.path.join(REPO_ROOT, 'results', 'qaoa_tiny.csv')
SAMPLE_CSV = os.path.join(REPO_ROOT, 'results', 'sample_results_sa.csv')


@pytest.fixture(scope='module')
def committed():
    """results/qaoa_tiny.csv and a rebuilt problem for each of its cases."""
    frame = pandas.read_csv(QAOA_CSV, comment='#')
    problems = {(instance, B): qc.build_problem(os.path.join(REPO_ROOT, instance), B)
                for instance, B in dict.fromkeys(zip(frame['instance'], frame['B']))}
    return frame, problems


def test_committed_csv_covers_the_sample_cases_at_every_depth(committed):
    frame, _ = committed
    sample = pandas.read_csv(SAMPLE_CSV, comment='#')
    cases = set(zip(sample['instance'], sample['B']))
    assert set(zip(frame['instance'], frame['B'])) == cases
    for reps in (1, 2, 3):
        at_depth = frame[frame['reps'] == reps]
        assert set(zip(at_depth['instance'], at_depth['B'])) == cases
    assert len(frame) == 3 * len(cases)


def test_committed_csv_matches_rebuilt_problems(committed):
    """
    Everything that does not depend on the QAOA run is recomputed from the
    instance: lambda, qubit count, optimum, ground energy, the uniform
    state's scores and greedy. The trained scores must be consistent.
    """
    frame, problems = committed
    for _, row in frame.iterrows():
        problem = problems[(row['instance'], row['B'])]
        assert row['lambda'] == problem.lam
        assert row['qubits'] == problem.num_qubits
        assert row['opt_nip'] == problem.opt_nip
        assert row['opt_qubo'] == pytest.approx(problem.energies().min())
        uniform = qc.uniform_scores(problem)
        assert row['uniform_p_optimal'] == pytest.approx(
            uniform['uniform_p_optimal'], abs=1e-6)
        assert row['uniform_expected_energy'] == pytest.approx(
            uniform['uniform_expected_energy'], abs=1e-6)
        assert 0 <= row['p_ground'] <= row['p_optimal'] <= 1
        assert 0 <= row['approx_ratio'] <= 1
        assert row['flow_after_repair'] >= problem.opt_nip
        assert len(json.loads(row['theta'])) == 2 * row['reps']


def test_committed_scores_follow_from_the_stored_angles(committed):
    """
    Recompute the exact scores from the stored angles with the reference
    simulation, so the CSV's numbers are those of its angles and not of some
    other run: every tiny_a row (14 qubits) and one 21-qubit row's trained
    and transferred scores. theta is stored to 6 decimals.
    """
    frame, problems = committed
    donor = frame[frame['transfer_p_optimal'].isna()]
    checks = [(row, 'own') for _, row in frame[frame['qubits'] == 14].iterrows()]
    other = frame[(frame['qubits'] == 21) & (frame['reps'] == 1)].iloc[0]
    checks += [(other, 'own'), (other, 'transfer')]
    for row, which in checks:
        problem = problems[(row['instance'], row['B'])]
        theta = row['theta'] if which == 'own' else \
            donor[donor['reps'] == row['reps']].iloc[0]['theta']
        probabilities = reference_probabilities(problem, json.loads(theta),
                                                problem.energies())
        p_optimal = probabilities[problem.optimal_mask()].sum()
        if which == 'own':
            assert row['p_optimal'] == pytest.approx(p_optimal, abs=1e-4)
            assert row['expected_energy'] == pytest.approx(
                probabilities @ problem.energies(), abs=1e-3)
        else:
            assert row['transfer_p_optimal'] == pytest.approx(p_optimal,
                                                              abs=1e-4)


@pytest.mark.parametrize('name', BACKENDS)
def test_command_line_reports_an_empty_window(name, capsys):
    module = backend_module(name)
    assert module.main([TINY_A, '--budget', '3']) == 2
    assert 'no interior' in capsys.readouterr().err

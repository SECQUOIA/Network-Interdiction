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


def test_capped_mps_stays_close_on_tiny_a(tiny_rcm, tiny_family):
    module = backend_module('qiskit')
    theta = np.random.default_rng(1).uniform(-1, 1, 4)
    capped = module.QiskitBackend(tiny_rcm, 2, method='mps',
                                  max_bond_dimension=4)
    phases = (dimod_energies(tiny_family) - tiny_family.offset) / tiny_family.scale
    exact = reference_probabilities(tiny_family, theta) @ phases
    # A cap of 4 is far below what tiny_a needs, so the value moves, but it
    # remains an estimate of the same quantity.
    assert capped.normalized_expectation(theta) == pytest.approx(exact,
                                                                 abs=0.5)


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

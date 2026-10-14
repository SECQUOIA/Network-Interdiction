"""Transfer contracts, an independent warm-start simulation, and real adapters."""
import importlib
import json
import os

import numpy as np
import pytest

import decomposition as dec
import qaoa_common as qc
from qaoa_transfer import (AngleDonor, IBMTransfer, NearestFlowMatcher,
                           flow_features, transferred_piece_solver, validate_donors)
from qaoa_warm_start import regularized_warm_start, warm_start_objective
from test_qaoa import TINY_A, dimod_energies
from test_decomposition_refinement import needs_gurobi


def optional_module(name):
    if os.environ.get('REQUIRE_TRANSFER_BACKENDS'):
        return importlib.import_module(name)
    return pytest.importorskip(name)


@pytest.fixture(scope='module')
def problem():
    return qc.build_problem(TINY_A, 1)


@pytest.fixture(scope='module')
def restricted(problem):
    return qc.problem_from_graph(
        problem.di_g, 0, 3, 1, interdictable=[(0, 1), (0, 2)])


def test_restricted_problem_ground_and_decoding(restricted):
    # Only source arcs may be cut: f=[15,5,0], lambda=7.5, two -8.5 constants.
    assert restricted.lambda_window == (5, 10)
    assert restricted.opt_nip == 5
    assert restricted.ground_energy == pytest.approx(-12)
    assert 'x_2' not in restricted.variables and 'x_3' not in restricted.variables
    bits = [int(v == 'x_0') for v in restricted.variables]
    assert restricted.interdicted_edges(bits) == [(0, 1)]
    assert restricted.x_vector(bits).tolist() == [1, 0, 0, 0]
    mask = restricted.optimal_mask()
    assert mask.mean() == 0.25
    assert np.all(restricted.energies()[mask] >= -12 - 1e-9)


def test_restricted_reference_above_enumeration_limit(problem, monkeypatch):
    # Exercise the analytic reference on the same independently enumerable model.
    monkeypatch.setattr(qc, 'EXACT_QUBITS', 2)
    piece = qc.problem_from_graph(
        problem.di_g, 0, 3, 1, interdictable=[(0, 1), (0, 2)])
    assert piece.ground_energy == pytest.approx(-12)


def test_repair_cannot_cut_a_fixed_high_capacity_arc(problem):
    # The unconstrained greedy winner is a 10-capacity arc. Only the two
    # 5-capacity arcs are owned, so repairing an empty cut must leave flow 10.
    piece = qc.problem_from_graph(
        problem.di_g, 0, 3, 1, interdictable=[(0, 2), (2, 3)])
    result = qc.score_samples(piece, {(0,) * piece.num_qubits: 10})
    assert result['flow_after_repair'] == 10
    assert len(result['interdicted_edges']) == 1
    assert set(result['interdicted_edges']) <= {(0, 2), (2, 3)}


@needs_gurobi
def test_restricted_gurobi_reference_agrees_with_enumeration(problem, restricted):
    actual = qc.problem_from_graph(
        problem.di_g, 0, 3, 1, interdictable=[(0, 1), (0, 2)],
        exact_backend='gurobi')
    assert actual.lambda_window == pytest.approx(restricted.lambda_window)
    assert actual.ground_energy == pytest.approx(-12)


def test_zero_budget_sample_score_is_finite():
    zero = qc.build_problem(TINY_A, 0)
    result = qc.score_samples(zero, {(0,) * zero.num_qubits: 10})
    assert result['shot_mean_ratio'] == 1
    assert result['flow_after_repair'] == 15


def test_flow_features_have_direction_capacity_budget_and_ownership(problem, restricted):
    nodes, edges, attrs, globals_ = flow_features(problem)
    assert nodes.shape == (4, 6)
    assert edges.shape == (2, 8)
    assert nodes[0, 0] == nodes[3, 1] == 1
    assert attrs[:4].tolist() == [[1, 1, 1], [1, -1, 1],
                                  [0.5, 1, 1], [0.5, -1, 1]]
    assert globals_[2:6] == pytest.approx([0.25, 0.75, 0.1, 1])
    _, _, restricted_attrs, restricted_globals = flow_features(restricted)
    assert restricted_attrs[-1, -1] == 0
    assert restricted_globals[2] == 0.5
    assert restricted_globals[5] == 0.5
    other = qc.build_problem(TINY_A, 2)
    assert not np.allclose(globals_, flow_features(other)[3])
    graph = problem.di_g.copy()
    graph[0][2]['weight'] = 4
    changed = qc.problem_from_graph(graph, 0, 3, 1)
    assert not np.allclose(attrs, flow_features(changed)[2])


def test_nearest_matching_uses_distance_and_deterministic_ties():
    matcher = NearestFlowMatcher([[1, 10], [3, 10]])
    assert matcher((2.9, 10, -1), {(1, 10, 0), (3, 10, 1)}) == (3, 10, 1)
    assert matcher((1, 10, -1), {(1, 10, 1), (1, 10, 0)}) == (1, 10, 0)


def test_donor_depth_and_finiteness_are_checked(problem):
    donor = AngleDonor(problem, [0.2, -0.3], 'a')
    assert validate_donors([donor]) == 1
    with pytest.raises(ValueError, match='same depth'):
        validate_donors([donor, AngleDonor(problem, [0] * 4, 'b')])
    with pytest.raises(ValueError, match='finite'):
        validate_donors([AngleDonor(problem, [np.nan, 0], 'a')])


def test_ibm_transfers_the_known_schedule_and_keeps_duplicate_keys(problem):
    optional_module('qaoa_training_pipeline')
    donors = [AngleDonor(problem, [0.23, -0.41], 'first'),
              AngleDonor(problem, [0.7, -0.1], 'second')]
    transfer = IBMTransfer(donors)
    assert len(transfer.data) == 2
    angles, index = transfer.select(problem)
    assert index == 0
    assert angles == pytest.approx([0.23, -0.41])


def test_ibm_rejects_a_mismatched_cost_operator(problem):
    optional_module('qaoa_training_pipeline')
    from qaoa_transfer import FlowFeatureExtractor
    from qaoa_qiskit import cost_operator
    extractor = FlowFeatureExtractor(problem, 1)
    with pytest.raises(ValueError, match='different NIP'):
        extractor(2 * cost_operator(problem), 1)
    with pytest.raises(ValueError, match='different NIP'):
        extractor(cost_operator(problem), 2)


def test_warm_objective_matches_product_state_energy_and_finite_difference(restricted):
    z = np.linspace(-0.6, 0.7, restricted.num_qubits)
    strength = 0.4
    value, gradient = warm_start_objective(restricted, z, strength)
    states = np.arange(2 ** restricted.num_qubits)
    probability = np.ones(len(states))
    for i, coordinate in enumerate(z):
        probability *= np.where((states >> i) & 1, (1 - coordinate) / 2,
                                 (1 + coordinate) / 2)
    expected = ((probability @ dimod_energies(restricted) - restricted.offset)
                / restricted.scale + strength * (z @ z))
    assert value == pytest.approx(expected)
    for i in range(len(z)):
        direction = np.zeros(len(z))
        direction[i] = 1e-6
        difference = (warm_start_objective(restricted, z + direction, strength)[0]
                      - warm_start_objective(restricted, z - direction, strength)[0]) / 2e-6
        assert gradient[i] == pytest.approx(difference, abs=1e-7)


def test_regularizer_moves_the_state_toward_uniform(restricted):
    low = regularized_warm_start(restricted, strength=0.1)
    high = regularized_warm_start(restricted, strength=100)
    assert np.linalg.norm(np.cos(high)) < np.linalg.norm(np.cos(low)) / 10
    assert warm_start_objective(restricted, np.cos(low), 0.1)[0] <= 0


def warm_reference(problem, initial_angles, theta):
    """Direct product amplitudes and 2x2 tilted-mixer matrices; no Qiskit gates."""
    n = problem.num_qubits
    states = np.arange(2 ** n)
    state = np.ones(2 ** n, dtype=complex)
    for i, angle in enumerate(initial_angles):
        state *= np.where((states >> i) & 1, np.sin(angle / 2), np.cos(angle / 2))
    energies = (dimod_energies(problem) - problem.offset) / problem.scale
    reps = len(theta) // 2
    for beta, gamma in zip(theta[:reps], theta[reps:]):
        state *= np.exp(-1j * gamma * energies)
        for i, angle in enumerate(initial_angles):
            z = 1 - 2 * ((states >> i) & 1)
            state = ((np.cos(beta) - 1j * np.sin(beta) * np.cos(angle) * z) * state
                     - 1j * np.sin(beta) * np.sin(angle) * state[states ^ (1 << i)])
    return np.abs(state) ** 2


def test_warm_circuit_matches_independent_two_layer_simulation(restricted):
    optional_module('qiskit_aer')
    from qaoa_qiskit import QiskitBackend
    angles = np.linspace(0.2, 2.9, restricted.num_qubits)
    theta = [0.17, 0.39, -0.32, -0.57]
    backend = QiskitBackend(restricted, 2, warm_start_angles=angles)
    expected = warm_reference(restricted, angles, theta)
    assert backend.probabilities(theta) == pytest.approx(expected, abs=1e-11)
    energy = (expected @ dimod_energies(restricted) - restricted.offset) / restricted.scale
    assert backend.normalized_expectation(theta) == pytest.approx(energy, abs=1e-10)


def test_uniform_warm_start_reduces_to_standard_qaoa(restricted):
    optional_module('qiskit_aer')
    from qaoa_qiskit import QiskitBackend
    theta = [0.22, -0.41]
    warm = QiskitBackend(restricted, 1,
                         warm_start_angles=np.full(restricted.num_qubits, np.pi / 2))
    standard = QiskitBackend(restricted, 1)
    assert warm.probabilities(theta) == pytest.approx(standard.probabilities(theta), abs=1e-11)


def test_gat_is_node_permutation_invariant_and_restores_weights(problem, tmp_path):
    torch = optional_module('torch')
    optional_module('torch_geometric')
    from torch_geometric.data import Batch
    from qaoa_gat import FlowGAT, GATTransfer, graph_data
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = FlowGAT(2).eval()
    data = graph_data(problem)
    perm = torch.tensor([2, 0, 3, 1])
    changed = data.clone()
    changed.x[perm] = data.x
    changed.edge_index = perm[data.edge_index]
    with torch.no_grad():
        original = model(Batch.from_data_list([data]))
        reordered = model(Batch.from_data_list([changed]))
    assert reordered.numpy() == pytest.approx(original.numpy(), abs=1e-7)
    donors = [AngleDonor(problem, [0.2, -0.3], 'a'),
              AngleDonor(problem, [0.4, -0.5], 'b')]
    transfer = GATTransfer(donors, model, [0, 1])
    path = tmp_path / 'gat.json'
    transfer.save(path)
    restored = GATTransfer.load(path, donors)
    assert restored.select(problem)[0] == pytest.approx(transfer.select(problem)[0])
    with torch.no_grad():
        assert restored.model(Batch.from_data_list([data])).numpy() == pytest.approx(
            original.numpy(), abs=1e-8)


def test_gat_training_runs_on_real_circuit_energies(restricted):
    torch = optional_module('torch')
    optional_module('torch_geometric')
    optional_module('qiskit_aer')
    from qaoa_gat import GATTransfer
    from qaoa_qiskit import QiskitBackend
    torch.set_num_threads(1)
    donors = [AngleDonor(restricted, [0.1, -0.2], 'a'),
              AngleDonor(restricted, [0.3, -0.4], 'b')]
    transfer, history = GATTransfer.train(donors, QiskitBackend, epochs=30)
    assert history[-1] < history[0] / 2
    best = min(range(2), key=lambda i: QiskitBackend(
        restricted, 1).normalized_expectation(donors[i].theta))
    assert transfer.select(restricted)[1] == best


def test_transfer_reaches_the_decomposition_callback(problem):
    optional_module('qaoa_training_pipeline')
    optional_module('qiskit_aer')
    transfer = IBMTransfer([AngleDonor(problem, [0.3, -0.4], 'training')])
    solver = transferred_piece_solver(transfer, exact_backend='brute', shots=128)
    pieces = [{0, 1}, {2, 3}]
    cuts, flow = dec.solve_decomposed(problem.di_g, 0, 3, pieces, [1, 0], solver)
    assert flow == 5
    assert len(cuts) == 1 and cuts[0] in {(0, 1), (1, 3)}
    sub = dec.piece_subproblem(problem.di_g, 0, 3, pieces, 0)
    assert dec.piece_qubo_size(sub) < problem.num_qubits
    assert solver(sub, 0) == []


def test_gat_logits_use_budget_and_capacity_features(problem):
    torch = optional_module('torch')
    optional_module('torch_geometric')
    from torch_geometric.data import Batch
    from qaoa_gat import FlowGAT, graph_data
    torch.manual_seed(9)
    torch.set_num_threads(1)
    model = FlowGAT(3).eval()
    original = graph_data(problem)
    budget = original.clone()
    budget.graph_features[0, 2] += 0.5
    capacity = original.clone()
    capacity.edge_attr[0, 0] *= 0.1
    with torch.no_grad():
        logits = [model(Batch.from_data_list([data])).numpy()
                  for data in (original, budget, capacity)]
    assert not np.allclose(logits[0], logits[1], atol=1e-8)
    assert not np.allclose(logits[0], logits[2], atol=1e-8)


def test_generated_training_and_test_graphs_are_disjoint():
    optional_module('qiskit_aer')
    from run_parameter_transfer import make_dataset
    _, records = make_dataset(3, 2, test_nodes=4, seed=5)
    train = {tuple(map(tuple, row['arcs'])) for row in records if row['split'] == 'train'}
    test = {tuple(map(tuple, row['arcs'])) for row in records if row['split'] == 'test'}
    assert len(train) == 3 and len(test) == 2
    assert train.isdisjoint(test)


def test_saved_donors_round_trip_and_reject_held_out_data(problem, tmp_path):
    from qaoa_transfer import load_donors
    record = {'schema': 1, 'graphs': [{
        'group': 'a', 'split': 'train', 'nodes': 4, 'src': 0, 'tgt': 3,
        'budget': 1, 'lambda': 7.5,
        'arcs': [(u, v, problem.di_g[u][v]['weight']) for u, v in problem.edge_list]}],
        'training': {'1': {'donors': [{'group': 'a', 'theta': [0.2, -0.3]}]}}}
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(record))
    donors = load_donors(path, 1)
    assert donors[0].theta == [0.2, -0.3]
    assert flow_features(donors[0].problem)[3] == pytest.approx(flow_features(problem)[3])
    record['graphs'][0]['split'] = 'test'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='held-out'):
        load_donors(path, 1)

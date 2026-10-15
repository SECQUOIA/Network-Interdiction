"""Check transfer results against their saved graphs and donor schedules."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import network_interdiction as ni
import qaoa_common as qc
from test_parameter_transfer import optional_module


ROOT = Path(__file__).resolve().parents[1]
METHODS = {'optimized', 'fixed', 'ibm', 'gat', 'rws'}


@pytest.fixture(scope='module')
def record():
    return json.loads((ROOT / 'results/parameter_transfer.json').read_text())


@pytest.fixture(scope='module')
def results():
    return pd.read_csv(ROOT / 'results/parameter_transfer.csv', comment='#')


def graph_from_record(row):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(row['nodes']))
    graph.add_weighted_edges_from(row['arcs'])
    return graph


def test_complete_graph_disjoint_comparison(record, results):
    assert record['complete'] is True
    training = {row['group'] for row in record['graphs'] if row['split'] == 'train'}
    testing = {row['group'] for row in record['graphs'] if row['split'] == 'test'}
    assert len(training) == 12 and len(testing) == 6
    assert training.isdisjoint(testing)
    signatures = [tuple(map(tuple, row['arcs'])) for row in record['graphs']]
    assert len(set(signatures)) == 18
    expected = {(group, depth, method) for group in testing
                for depth in (1, 2) for method in METHODS}
    assert len(results) == 60
    assert set(zip(results.group, results.reps, results.method)) == expected
    assert record['commit'] in (ROOT / 'results/parameter_transfer.csv').read_text()
    assert not record['commit'].endswith('-dirty')


def test_angles_come_from_training_and_transfers_do_not_optimize_targets(record, results):
    for row in results.itertuples():
        bank = record['training'][str(row.reps)]
        donors = {donor['group']: donor['theta'] for donor in bank['donors']}
        theta = json.loads(row.theta)
        assert len(theta) == 2 * row.reps and np.isfinite(theta).all()
        if row.method == 'optimized':
            assert row.target_evaluations > 0
        else:
            assert row.target_evaluations == 0
            if row.method == 'rws':
                assert theta == bank['rws_theta']
            else:
                assert row.donor_group in donors
                assert theta == donors[row.donor_group]
                if row.method == 'fixed':
                    assert row.donor_group == bank['donors'][0]['group']


def test_flows_and_energy_gaps_match_the_saved_weighted_networks(record, results):
    for data in record['graphs']:
        if data['split'] != 'test':
            continue
        graph = graph_from_record(data)
        source, sink, lam = data['src'], data['tgt'], data['lambda']
        maxflow = nx.maximum_flow_value(graph, source, sink, capacity='weight')
        # Every committed case has B=1: independently enumerate every arc
        # deletion. Greedy also reaches this optimum by definition.
        assert data['budget'] == 1
        flows = []
        for edge in graph.edges:
            remaining = graph.copy()
            remaining.remove_edge(*edge)
            flows.append(nx.maximum_flow_value(remaining, source, sink, capacity='weight'))
        optimum = min(flows)
        bqm = ni.nip_to_qubo(1, graph, source, sink, lam, lam + 1)
        uniform = bqm.offset + sum(bqm.linear.values()) / 2 + sum(bqm.quadratic.values()) / 4
        ground = optimum - (lam + 1) * len(graph.edges)
        group = results[results.group == data['group']]
        assert np.all(group.qubits == len(bqm.variables))
        assert np.all(group['lambda'] == lam)
        assert np.all(group.opt_nip == optimum)
        assert np.all(group.greedy == optimum)
        assert np.all(group.flow_after_repair >= optimum)
        assert np.all(group.flow_after_repair <= maxflow)
        expected = (uniform - group.expected_energy) / (uniform - ground)
        assert group.gap_closed.to_numpy() == pytest.approx(expected.to_numpy())
        for column in ('approx_ratio', 'p_optimal', 'p_feasible', 'shot_mean_ratio'):
            assert group[column].between(0, 1).all()


def test_saved_schedules_reproduce_one_held_out_energy_per_method(record, results):
    optional_module('qiskit_aer')
    from qaoa_qiskit import QiskitBackend
    from qaoa_warm_start import regularized_warm_start
    data = next(row for row in record['graphs'] if row['group'] == 'test-000')
    problem = qc.problem_from_graph(graph_from_record(data), data['src'], data['tgt'],
                                    data['budget'], lam=data['lambda'])
    standard = QiskitBackend(problem, 1)
    warm = QiskitBackend(problem, 1, warm_start_angles=regularized_warm_start(
        problem, record['arguments']['strength'], record['arguments']['seed']))
    for row in results[(results.group == 'test-000') & (results.reps == 1)].itertuples():
        backend = warm if row.method == 'rws' else standard
        expected = problem.offset + problem.scale * backend.normalized_expectation(json.loads(row.theta))
        assert row.expected_energy == pytest.approx(expected, abs=1e-6)


def test_saved_gat_model_solves_a_contracted_piece():
    torch = optional_module('torch')
    optional_module('torch_geometric')
    optional_module('qiskit_aer')
    import decomposition as dec
    from qaoa_gat import GATTransfer
    from qaoa_transfer import load_donors, transferred_piece_solver
    torch.set_num_threads(1)
    donors = load_donors(ROOT / 'results/parameter_transfer.json', 1)
    transfer = GATTransfer.load(ROOT / 'results/parameter_transfer.p1.gat.json', donors)
    problem = qc.build_problem(ROOT / 'testcases/tiny_a.flow', 1)
    solver = transferred_piece_solver(transfer, exact_backend='brute', shots=128)
    cuts, flow = dec.solve_decomposed(problem.di_g, 0, 3, [{0, 1}, {2, 3}], [1, 0], solver)
    assert flow == 5
    assert len(cuts) == 1 and cuts[0] in {(0, 1), (1, 3)}


@pytest.mark.parametrize('method', ['ibm', 'gat'])
def test_saved_models_reproduce_the_reported_donor_choices(method, record, results):
    optional_module('qiskit_aer')
    from qaoa_transfer import IBMTransfer, load_donors
    if method == 'ibm':
        optional_module('qaoa_training_pipeline')
    else:
        torch = optional_module('torch')
        optional_module('torch_geometric')
        from qaoa_gat import GATTransfer
        torch.set_num_threads(1)
    for reps in (1, 2):
        donors = load_donors(ROOT / 'results/parameter_transfer.json', reps)
        transfer = (IBMTransfer(donors) if method == 'ibm' else GATTransfer.load(
            ROOT / f'results/parameter_transfer.p{reps}.gat.json', donors))
        for data in record['graphs']:
            if data['split'] != 'test':
                continue
            problem = qc.problem_from_graph(
                graph_from_record(data), data['src'], data['tgt'],
                data['budget'], lam=data['lambda'])
            theta, index = transfer.select(problem)
            row = results[(results.group == data['group']) & (results.reps == reps)
                          & (results.method == method)].iloc[0]
            assert row.donor_group == donors[index].group
            assert theta == pytest.approx(json.loads(row.theta))

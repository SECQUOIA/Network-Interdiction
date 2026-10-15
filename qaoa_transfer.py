"""
Flow-network features and IBM parameter transfer for penalty QAOA (#16).

Angles always have qaoa_common's [beta..., gamma...] order and refer to
(H_C - offset) / scale. Donor banks contain a single circuit depth. A
donor's group identifies its graph; split groups before building banks so
budgets of a held-out graph cannot leak into training.

The flow graph, not the QUBO graph, supplies features. Each directed arc
has a forward and reverse message with capacity, direction and whether it
can be interdicted. Source/sink, degrees and capacity sums are node
features. Graph features retain size, budget, lambda and capacity scale.
The latter matters because the dual penalty is lambda + 1.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import networkx as nx
import numpy as np

import qaoa_common as qc


@dataclass
class AngleDonor:
    """One training graph/budget and its optimized normalized-Hamiltonian angles."""
    problem: qc.NIPQaoaProblem
    theta: object
    group: str


def validate_donors(donors):
    """Return the common depth, rejecting mixed conventions or invalid angles."""
    if not donors:
        raise ValueError('parameter transfer needs at least one donor')
    size = len(donors[0].theta)
    if size == 0 or size % 2:
        raise ValueError('donor angles need beta and gamma for every layer')
    for donor in donors:
        theta = np.asarray(donor.theta, dtype=float)
        if theta.shape != (size,) or not np.all(np.isfinite(theta)):
            raise ValueError('donor angles must be finite and have the same depth')
        if not donor.group:
            raise ValueError('donors need a graph group for held-out evaluation')
    return size // 2


def load_donors(path, reps):
    """Restore the training bank saved by run_parameter_transfer.py."""
    record = json.loads(Path(path).read_text())
    if record['schema'] != 1:
        raise ValueError('unsupported transfer experiment schema')
    graphs = {row['group']: row for row in record['graphs']}
    donors = []
    for row in record['training'][str(reps)]['donors']:
        data = graphs[row['group']]
        if data['split'] != 'train':
            raise ValueError('a donor may not come from the held-out split')
        graph = nx.DiGraph()
        graph.add_nodes_from(range(data['nodes']))
        graph.add_weighted_edges_from(data['arcs'])
        problem = qc.problem_from_graph(
            graph, data['src'], data['tgt'], data['budget'], lam=data['lambda'],
            instance=row['group'])
        donors.append(AngleDonor(problem, row['theta'], row['group']))
    if validate_donors(donors) != reps:
        raise ValueError('saved donor angles have the wrong depth')
    return donors


def flow_features(problem):
    """(node features, message endpoints, edge features, graph features) arrays."""
    graph = problem.di_g
    nodes, edges = sorted(graph.nodes), problem.edge_list
    index = {node: i for i, node in enumerate(nodes)}
    n, m = len(nodes), len(edges)
    capacities = np.array([graph[u][v]['weight'] for u, v in edges], dtype=float)
    cap = max(float(capacities.max()), 1.0)
    allowed = set(problem.interdictable_edges)
    node_features = []
    for node in nodes:
        incoming = sum(graph[u][node]['weight'] for u in graph.predecessors(node))
        outgoing = sum(graph[node][v]['weight'] for v in graph.successors(node))
        node_features.append([
            float(node == problem.src_node), float(node == problem.tgt_node),
            graph.in_degree(node) / max(n - 1, 1),
            graph.out_degree(node) / max(n - 1, 1),
            incoming / (cap * m), outgoing / (cap * m)])
    endpoints, edge_features = [], []
    for (u, v), capacity in zip(edges, capacities):
        for a, b, direction in ((u, v, 1), (v, u, -1)):
            endpoints.append([index[a], index[b]])
            edge_features.append([capacity / cap, direction,
                                  float((u, v) in allowed)])
    global_features = [
        np.log1p(n), np.log1p(m), problem.B / max(len(allowed), 1),
        problem.lam / cap, 1 / cap, len(allowed) / m,
        capacities.mean() / cap, capacities.std() / cap,
        node_features[index[problem.src_node]][5],
        node_features[index[problem.tgt_node]][4]]
    return (np.asarray(node_features), np.asarray(endpoints).T,
            np.asarray(edge_features), np.asarray(global_features))


class FlowFeatureExtractor:
    """
    IBM cost-operator adapter bound to one NIP graph and its variable semantics.

    SparsePauliOp alone cannot recover source/sink or which arcs are fixed.
    Bind the original problem and reject a different operator or depth.
    """
    def __init__(self, problem, reps):
        from qaoa_qiskit import cost_operator
        self.operator = cost_operator(problem)
        self.features = tuple(flow_features(problem)[3])
        self.reps = reps

    def __call__(self, cost_op, qaoa_depth):
        if qaoa_depth != self.reps or not self.operator.equiv(cost_op):
            raise ValueError('feature extractor is bound to a different NIP problem')
        return self.features + (-1,)

    def to_config(self):
        return {'features': list(self.features), 'reps': self.reps}


class NearestFlowMatcher:
    """Euclidean nearest donor; feature standard deviations use training data only."""
    def __init__(self, features):
        scale = np.std(features, axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.0)

    def __call__(self, features, set_of_features):
        # Final key component is donor index: keep duplicate feature vectors
        # and break ties deterministically instead of overwriting a donor.
        target = np.asarray(features[:-1])
        return min(set_of_features, key=lambda key: (
            float(np.sum(((np.asarray(key[:-1]) - target) / self.scale) ** 2)),
            key[-1]))

    def to_config(self):
        return {'scale': self.scale.tolist()}


class IBMTransfer:
    """Nearest flow-feature schedules through IBM's actual TransferTrainer."""
    def __init__(self, donors):
        self.donors = list(donors)
        self.reps = validate_donors(self.donors)
        features = [flow_features(donor.problem)[3] for donor in self.donors]
        self.matcher = NearestFlowMatcher(features)
        self.data = {
            tuple(feature) + (i,): {
                'qaoa_angles': np.asarray(donor.theta).tolist(),
                'metadata': {'donor': i, 'group': donor.group}}
            for i, (feature, donor) in enumerate(zip(features, self.donors))}

    def select(self, problem):
        """(angles, donor index), without target circuit optimization."""
        from qaoa_training_pipeline.training.transfer_trainer import TransferTrainer
        from qaoa_training_pipeline.training.data_loading import TrivialDataLoader
        from qaoa_qiskit import cost_operator

        trainer = TransferTrainer(
            TrivialDataLoader(self.data), FlowFeatureExtractor(problem, self.reps),
            feature_matcher=self.matcher, reps=self.reps)
        result = trainer.provide_params(cost_operator(problem))
        return (np.asarray(result['optimized_qaoa_angles']),
                int(result['metadata']['donor']))


def transferred_piece_solver(transfer, max_qubits=156, shots=4096, seed=0,
                             exact_backend='gurobi', max_bond_dimension=32):
    """
    solve_decomposed() callback: build the restricted QUBO, select a schedule,
    sample it with MPS, repair only owned arcs, and return original-network arcs.

    Lambda is validated for each piece. An empty window raises ValueError;
    using a different graph's lambda would silently solve the wrong problem.
    Budget zero needs no QAOA. Exact references used to set lambda remain a
    classical cost; this callback makes no end-to-end scaling claim.
    """
    from qaoa_qiskit import QiskitBackend

    def solve(sub, budget):
        budget = min(budget, len(sub.interdictable))
        if budget == 0:
            return []
        problem = qc.problem_from_graph(
            sub.graph, sub.src, sub.tgt, budget, order='rcm',
            max_qubits=max_qubits, exact_backend=exact_backend,
            interdictable=sub.interdictable)
        theta, _ = transfer.select(problem)
        backend = QiskitBackend(problem, transfer.reps, method='mps',
                                max_bond_dimension=max_bond_dimension)
        result = qc.score_samples(problem, backend.sample(theta, shots, seed))
        return [sub.arc_map[arc] for arc in result['interdicted_edges']]

    return solve

"""
GAT donor classification for NIP parameter transfer (issue #16).

Adapted from jxsoortha/QAOA-Parameter-Transfer-via-GAT, Apache-2.0,
commit ab8434ffcb2cfcac16d136d8677f327752d9ca8d,
pennylane-MIS/qaoa_pytorch_model.py and 4-gan_stratified.py.
The upstream license is in third_party/gat/LICENSE.

Changes: flow features replace node2vec; capacity/direction/ownership are
edge features; pooled features include B and lambda. Three GATv2 layers,
mean pooling, dropout and cross-entropy donor classification retain the
MIS training path. Training-only transfer-energy profiles are clustered;
each cluster contributes a medoid schedule. Labels select the best medoid
on each training graph. No target circuit evaluation is used by select().
"""
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from qaoa_transfer import flow_features, validate_donors


def graph_data(problem):
    """One directed flow network in PyTorch Geometric format."""
    nodes, edges, edge_features, globals_ = flow_features(problem)
    return Data(x=torch.tensor(nodes, dtype=torch.float32),
                edge_index=torch.tensor(edges, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32),
                graph_features=torch.tensor(globals_[None], dtype=torch.float32))


class FlowGAT(nn.Module):
    """Three GATv2 message layers and graph-level donor logits."""
    def __init__(self, num_donors, hidden=32):
        super().__init__()
        self.convs = nn.ModuleList([
            GATv2Conv(6, hidden, edge_dim=3),
            GATv2Conv(hidden, hidden, edge_dim=3),
            GATv2Conv(hidden, hidden, edge_dim=3)])
        self.layers = nn.Sequential(
            nn.Linear(hidden + 10, hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden, num_donors))

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr).relu()
            x = F.dropout(x, p=0.2, training=self.training)
        pooled = global_mean_pool(x, data.batch)
        return self.layers(torch.cat([pooled, data.graph_features], dim=1))


def representative_donors(energies, num_clusters):
    """
    Medoids of transfer-energy profiles; rows are targets, columns donors.
    Dividing by qubit count prevents large training graphs dominating distance.
    The caller supplies per-qubit normalized energies from training graphs only.
    """
    profiles = np.asarray(energies).T
    if num_clusters >= len(profiles):
        return np.arange(len(profiles))
    labels = cut_tree(linkage(profiles, method='ward'),
                      n_clusters=num_clusters).ravel()
    representatives = []
    for label in sorted(set(labels)):
        members = np.flatnonzero(labels == label)
        distances = cdist(profiles[members], profiles[members])
        representatives.append(int(members[np.argmin(distances.sum(axis=1))]))
    return np.array(representatives)


class GATTransfer:
    """A trained classifier and its fixed donor schedules at one depth."""
    def __init__(self, donors, model, representative_indices, hidden=32):
        self.donors = list(donors)
        self.reps = validate_donors(self.donors)
        self.model = model.eval()
        self.representative_indices = np.asarray(representative_indices, dtype=int)
        self.hidden = hidden

    @classmethod
    def train(cls, donors, make_backend, num_clusters=4, epochs=200,
              seed=0, hidden=32):
        """Train on donors only; return (transfer model, loss history)."""
        reps = validate_donors(donors)
        if epochs <= 0 or num_clusters <= 0:
            raise ValueError('epochs and num_clusters must be positive')
        energies = np.empty((len(donors), len(donors)))
        for i, target in enumerate(donors):
            backend = make_backend(target.problem, reps)
            for j, donor in enumerate(donors):
                energies[i, j] = (backend.normalized_expectation(donor.theta)
                                  / target.problem.num_qubits)
        indices = representative_donors(energies, num_clusters)
        labels = torch.tensor(np.argmin(energies[:, indices], axis=1),
                               dtype=torch.long)
        batch = Batch.from_data_list([graph_data(donor.problem) for donor in donors])
        # Keep model initialization/dropout reproducible without resetting a
        # caller's global RNG state.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = FlowGAT(len(indices), hidden)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            history = []
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch), labels)
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach()))
        return cls(donors, model, indices, hidden), history

    def select(self, problem):
        """(angles, donor index); inference does not inspect target optima."""
        batch = Batch.from_data_list([graph_data(problem)])
        with torch.no_grad():
            label = int(self.model(batch).argmax(dim=1).item())
        index = int(self.representative_indices[label])
        return np.asarray(self.donors[index].theta).copy(), index

    def save(self, path):
        """JSON weights and donor identities; load with the same ordered bank."""
        data = {'schema': 1, 'reps': self.reps, 'hidden': self.hidden,
                'groups': [donor.group for donor in self.donors],
                'angles': [np.asarray(donor.theta).tolist() for donor in self.donors],
                'representatives': self.representative_indices.tolist(),
                'weights': {key: value.tolist()
                            for key, value in self.model.state_dict().items()}}
        Path(path).write_text(json.dumps(data, sort_keys=True) + '\n')

    @classmethod
    def load(cls, path, donors):
        """Restore without executing pickle code; reject a different donor bank."""
        data = json.loads(Path(path).read_text())
        if (data['schema'] != 1 or data['reps'] != validate_donors(donors)
                or data['groups'] != [donor.group for donor in donors]
                or data['angles'] != [np.asarray(donor.theta).tolist() for donor in donors]):
            raise ValueError('checkpoint does not match the donor bank')
        model = FlowGAT(len(data['representatives']), data['hidden'])
        model.load_state_dict({key: torch.tensor(value)
                               for key, value in data['weights'].items()})
        return cls(donors, model, data['representatives'], data['hidden'])

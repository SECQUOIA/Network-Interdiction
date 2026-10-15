"""Conservative exact penalties for the optional squared-budget NIP model.

Integer capacities keep the unit constraint gap exact. The floating-point
encoding is accepted only within a conservative integer arithmetic envelope.
The original linear-penalty builder and its window remain unchanged by default.
"""
from dataclasses import dataclass
from numbers import Integral

import network_interdiction as ni


@dataclass(frozen=True)
class PenaltyCertificate:
    objective_upper_bound: int
    cardinality: int
    dual: int
    interdictable_count: int
    normalized_offset: int


def squared_penalty_coefficients(B, graph, interdictable=None):
    """Return λ=β=1+Σu for nonnegative integer capacities and 0≤B≤|A_I|.

    For every binary assignment E+β|A_I|=C+λ(k-B)²+βQ, with 0≤C≤U
    and integer Q≥0. If k!=B or Q>0 its normalized energy exceeds U;
    every encoded feasible assignment has energy C≤U. This covers joint
    violations. An original feasible solution has an auxiliary completion
    with Q=0; other auxiliary choices are not called encoded feasible.

    The at-most-B NIP has the same optimum because any smaller interdiction
    set can be padded within A_I without increasing flow. This does not
    assert that every optimal interdiction set has exactly B arcs.
    """
    if not graph.is_directed() or graph.is_multigraph():
        raise ValueError('requires a simple directed graph')
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in graph):
        raise ValueError('nodes must be consecutive integers starting at zero')
    if set(graph) != set(range(len(graph))):
        raise ValueError('nodes must be consecutive integers starting at zero')
    allowed = set(graph.edges()) if interdictable is None else set(interdictable)
    if not allowed <= set(graph.edges()):
        raise ValueError('interdictable arcs not in the graph')
    m = len(allowed)
    if isinstance(B, bool) or not isinstance(B, Integral) or not 0 <= B <= m:
        raise ValueError('budget must be an integer between zero and the interdictable count')
    capacities = [data.get('weight') for _, _, data in graph.edges(data=True)]
    if any(isinstance(u, bool) or not isinstance(u, Integral) or u < 0
           for u in capacities):
        raise ValueError('capacities must be nonnegative integers')
    U = sum(map(int, capacities))
    penalty = U + 1
    # Bound absolute expanded coefficient mass, including intermediates of
    # the unsimplified dual polynomial and normalization. Staying below 2^52
    # leaves margin inside exact binary64 integer arithmetic (2^53).
    mass_bound = 4 * U + penalty * ((m + int(B)) ** 2 + 16 * m)
    if mass_bound > 2 ** 52:
        raise ValueError('certified integer encoding exceeds the binary64 safety bound')
    return PenaltyCertificate(U, penalty, penalty, m, penalty * m)


def certified_squared_qubo(B, graph, src_node, tgt_node, interdictable=None):
    """Build the squared model and return (BQM, coefficient certificate)."""
    allowed = None if interdictable is None else set(interdictable)
    certificate = squared_penalty_coefficients(B, graph, allowed)
    if src_node not in graph or tgt_node not in graph or src_node == tgt_node:
        raise ValueError('source and target must be distinct graph nodes')
    bqm = ni.nip_to_qubo(B, graph, src_node, tgt_node, certificate.cardinality,
                        certificate.dual, allowed, budget_penalty='squared')
    return bqm, certificate

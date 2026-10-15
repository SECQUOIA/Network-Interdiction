"""Actual Aer circuits checked against independent binary-basis evolution."""
import itertools
import math
import os
from types import SimpleNamespace

import dimod
import numpy as np
import pytest

if os.getenv('REQUIRE_QAOA_BACKENDS'):
    import qiskit
    import qiskit_aer
else:
    pytest.importorskip('qiskit')
    pytest.importorskip('qiskit_aer')

import qaoa_common as qc
from qaoa_qiskit import QiskitBackend
from qaoa_xy import FixedWeightMixer, fixed_weight_ansatz
from solve_qubo_via_qaoa import solve_bqm_qaoa


def small_problem(n):
    variables = [f'v{i}' for i in range(n)]
    bqm = dimod.BinaryQuadraticModel({v: (i+1)*.3 for i, v in enumerate(variables)},
                                    {(variables[0], variables[-1]): -.7,
                                     (variables[1], variables[2]): .4}, 1.3, dimod.BINARY)
    h, J, offset = qc.ising_from_bqm(bqm, variables)
    scale = max(max(abs(h)), max(map(abs, J.values())))
    p = SimpleNamespace(h=h, J=J, offset=offset, scale=scale, num_qubits=n)
    bits = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int8)[:, ::-1]
    energies = bqm.energies((bits, variables))
    return p, bqm, bits, energies


def independent_probabilities(p, bits, energies, register, weight, initial, pairs, theta):
    # Enumerate only the small test Hilbert spaces (at most 64 states).
    assert p.num_qubits <= 6
    mask = bits[:, list(register)].sum(axis=1) == weight
    if initial == 'basis':
        target = np.array([int(i < weight) for i in range(len(register))])
        mask = (bits[:, list(register)] == target).all(axis=1)
    state = mask.astype(complex) / math.sqrt(mask.sum())
    reps = len(theta)//2
    for layer in range(reps):
        beta, gamma = theta[layer], theta[reps+layer]
        state *= np.exp(-1j*gamma*(energies-p.offset)/p.scale)
        for i, j in pairs:
            for a in range(len(state)):
                if (a >> i)&1 == 0 and (a >> j)&1 == 1:
                    b = a ^ (1 << i) ^ (1 << j)
                    u, v = state[a], state[b]
                    state[a] = np.cos(2*beta)*u - 1j*np.sin(2*beta)*v
                    state[b] = -1j*np.sin(2*beta)*u + np.cos(2*beta)*v
        for i in set(range(p.num_qubits))-set(register):
            for a in range(len(state)):
                if (a >> i)&1 == 0:
                    b = a ^ (1 << i)
                    u, v = state[a], state[b]
                    state[a] = np.cos(beta)*u - 1j*np.sin(beta)*v
                    state[b] = -1j*np.sin(beta)*u + np.cos(beta)*v
    return abs(state)**2


@pytest.mark.parametrize('register,pairs,n', [
    ((2, 0), [(2, 0)], 3),
    ((3, 0, 2), [(3, 0), (0, 2), (2, 3)], 5),
    ((4, 1, 5, 0), [(4, 1), (5, 0), (1, 5), (0, 4)], 6),
])
@pytest.mark.parametrize('initial', ['basis', 'dicke'])
def test_actual_circuit_matches_independent_ordered_xy_evolution(register, pairs, n, initial):
    p, _, bits, energies = small_problem(n)
    theta = [.37, -.19, .28, .51]
    backend = QiskitBackend(p, 2, fixed_weight=FixedWeightMixer(register, 1, initial))
    probabilities = backend.probabilities(theta)
    reference = independent_probabilities(p, bits, energies, register, 1, initial, pairs, theta)
    np.testing.assert_allclose(probabilities, reference, atol=1e-10, rtol=0)
    assert probabilities[bits[:, list(register)].sum(axis=1) != 1].sum() < 1e-20
    assert backend.normalized_expectation(theta)*p.scale+p.offset == pytest.approx(
        float(probabilities@energies), abs=1e-9)
    counts = backend.sample(theta, 1024, seed=7)
    assert sum(counts.values()) == 1024
    assert all(sum(sample[q] for q in register) == 1 for sample in counts)


@pytest.mark.parametrize('weight', [0, 2, 3])
@pytest.mark.parametrize('initial', ['basis', 'dicke'])
def test_boundary_and_multi_excitation_sectors_are_preserved(weight, initial):
    p, _, bits, energies = small_problem(5)
    register = (3, 0, 2)
    backend = QiskitBackend(p, 2, fixed_weight=FixedWeightMixer(register, weight, initial))
    theta = [.6, -.4, .3, .9]
    probs = backend.probabilities(theta)
    reference = independent_probabilities(p, bits, energies, register, weight, initial,
                                         [(3, 0), (0, 2), (2, 3)], theta)
    np.testing.assert_allclose(probs, reference, atol=1e-10, rtol=0)
    assert probs[bits[:, list(register)].sum(axis=1) != weight].sum() < 1e-20


def test_dense_dicke_limit_is_checked_before_allocation(monkeypatch):
    p = SimpleNamespace(num_qubits=14)
    monkeypatch.setattr(np, 'zeros', lambda *a, **kw: pytest.fail('allocated dense state'))
    with pytest.raises(ValueError, match='limited to 12'):
        fixed_weight_ansatz(p, 1, FixedWeightMixer(tuple(range(13)), 4, 'dicke'))


@pytest.mark.parametrize('mixer', [FixedWeightMixer((), 0), FixedWeightMixer((0, 0), 1),
    FixedWeightMixer((0, 7), 1), FixedWeightMixer((True, 1), 1),
    FixedWeightMixer((0, 1), 1.5), FixedWeightMixer((0, 1), -1),
    FixedWeightMixer((0, 1), 3), FixedWeightMixer((0, 1), 1, 'invalid')])
def test_invalid_registers_are_refused(mixer):
    with pytest.raises(ValueError):
        mixer.validate(4)


def test_product_warm_start_cannot_break_fixed_weight_contract():
    p, _, _, _ = small_problem(3)
    with pytest.raises(ValueError, match='cannot use product'):
        QiskitBackend(p, 1, warm_start_angles=[.5]*3,
                      fixed_weight=FixedWeightMixer((0, 2), 1))


def test_existing_uncapped_mps_method_runs_the_same_xy_circuit():
    p, _, _, _ = small_problem(5)
    mixer = FixedWeightMixer((3, 0, 2), 2, 'dicke')
    exact = QiskitBackend(p, 2, fixed_weight=mixer)
    mps = QiskitBackend(p, 2, method='mps', fixed_weight=mixer)
    theta = [.4, .2, -.3, .7]
    assert mps.normalized_expectation(theta) == pytest.approx(
        exact.normalized_expectation(theta), abs=1e-9)
    assert all(sum(bits[q] for q in mixer.qubits) == 2
               for bits in mps.sample(theta, 128, seed=8))


def test_shared_sampler_keeps_original_energies_and_register_labels():
    _, bqm, _, _ = small_problem(5)
    result = solve_bqm_qaoa(bqm, reps=1, shots=256, maxiter=6, seed=3,
                            fixed_weight_variables=['v3', 'v0', 'v2'],
                            hamming_weight=2, initial_state='dicke')
    assert sum(result.record.num_occurrences) == 256
    for row in result.data(['sample', 'energy']):
        assert sum(row.sample[v] for v in ('v3', 'v0', 'v2')) == 2
        assert row.energy == pytest.approx(bqm.energy(row.sample))
    meta = result.info['qaoa_metadata']
    assert meta['fixed_weight_variables'] == ['v3', 'v0', 'v2']
    assert meta['hamming_weight'] == 2 and meta['initial_state'] == 'dicke'


def test_dual_penalty_must_allow_fixed_x_repair():
    import networkx as nx
    import network_interdiction as ni
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 7)])
    # At B=0 the mixer never permits the old x=1 repair. beta>lambda=0
    # is insufficient; beta>capacity fixes gamma without changing x.
    for beta, expected_gamma in [(1, 0), (8, 1)]:
        bqm = ni.nip_to_qubo(0, graph, 0, 1, 0, beta)
        assert len(bqm) == 3
        candidates = [row for row in dimod.ExactSolver().sample(bqm).data()
                      if row.sample['x_0'] == 0]
        best = min(candidates, key=lambda row: row.energy)
        assert best.sample['gamma_0'] == expected_gamma
        if beta == 8:
            assert best.energy + beta == 7


def test_comparison_guard_precedes_exact_work(tmp_path, monkeypatch):
    import run_xy_comparison as comparison
    path = tmp_path / 'large.flow'
    path.write_text('10\n0 9\n'+''.join(f'{i} {i+1} 1\n' for i in range(9)))
    monkeypatch.setattr(comparison.rb, 'max_flow', lambda *a: pytest.fail('reference dispatched'))
    with pytest.raises(ValueError, match='limited to'):
        comparison.references(path, 1)


def test_comparison_retains_completed_method_before_later_failure(tmp_path, monkeypatch):
    import json
    import gurobipy as gp
    import run_xy_comparison as comparison
    path = tmp_path / 'one.flow'
    path.write_text('2\n0 1\n0 1 7\n')
    retained = []
    def fail_sampler(*args, **kwargs):
        raise RuntimeError('deliberate sampler failure')
    monkeypatch.setattr(comparison, 'solve_bqm_qaoa', fail_sampler)
    try:
        with pytest.raises(RuntimeError, match='deliberate sampler failure'):
            comparison.run_case(path, 0, shots=16, maxiter=4,
                                checkpoint=lambda row: retained.append(json.loads(json.dumps(row))))
    except gp.GurobiError as error:
        pytest.skip(f'Gurobi unavailable: {error}')
    assert retained[-1]['methods'][0]['mode'] == 'sa'
    assert sum(retained[-1]['methods'][0]['counts'].values()) == 16
    assert retained[-1]['status'] == 'running'

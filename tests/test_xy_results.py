"""Backend-free audit of every retained shot and the original NIP objective."""
import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

import network_interdiction as ni

ROOT = Path(__file__).resolve().parents[1]


def surviving(graph, source, sink, edges):
    residual = graph.copy()
    residual.remove_edges_from(edges)
    return nx.minimum_cut_value(residual, source, sink, capacity='weight')


def test_all_retained_counts_and_scores_against_independent_flow_audit():
    data = json.loads((ROOT / 'results/xy_comparison.json').read_text())
    assert data['failures'] == [] and len(data['rows']) == 3
    assert {(r['instance'],r['budget']) for r in data['rows']} == {
        ('testcases/tiny_a.flow', 1), ('testcases/tiny_b.flow', 2),
        ('testcases/equal_paths.flow', 1)}
    for row in data['rows']:
        assert row['status'] == 'completed'
        path = ROOT / row['instance']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['input_sha256']
        graph, source, sink = ni.read_flow_file(path)
        edges = sorted(graph.edges())
        assert len(edges) <= 7  # before independent subset enumeration
        flows = {tuple(i for i in range(len(edges)) if (mask >> i)&1):
                 surviving(graph, source, sink,
                           [edge for i, edge in enumerate(edges) if (mask >> i)&1])
                 for mask in range(1 << len(edges))}
        B = row['budget']
        opt = min(flow for cut, flow in flows.items() if len(cut) <= B)
        ref = row['references']
        assert opt == ref['nip_optimum']
        assert ref['milp_objective'] == pytest.approx(opt, abs=1e-6)
        assert ref['milp_bound'] == pytest.approx(opt, abs=1e-6)
        assert ref['first_B_arcs_flow'] == flows[tuple(range(B))]
        assert ref['uniform_fixed_weight_p_optimal'] == pytest.approx(
            sum(flow == opt for cut, flow in flows.items() if len(cut) == B)
            / sum(len(cut) == B for cut in flows))
        assert ref['uniform_all_bits_p_optimal'] == pytest.approx(
            sum(flow == opt and len(cut) <= B for cut, flow in flows.items()) / len(flows))
        assert [m['mode'] for m in row['methods']] == ['sa','penalty','xy_basis','xy_dicke']
        for method in row['methods']:
            if method['status'] == 'not_applicable':
                assert row['instance'].endswith('equal_paths.flow')
                assert method['mode'] in ('sa','penalty') and ref['lambda'] is None
                continue
            assert method['status'] == 'completed'
            variables, counts = method['variables'], method['counts']
            bqm = ni.nip_to_qubo(B, graph, source, sink,
                                 method['cardinality_penalty'], method['dual_penalty'])
            assert set(variables) == set(bqm.variables)
            assert all(len(bits) == len(variables) and set(bits) <= {'0','1'} for bits in counts)
            assert all(isinstance(count, int) and count > 0 for count in counts.values())
            total = sum(counts.values())
            assert total == data['controls']['shots'] == 1024
            optimum = feasible = exact_budget = 0
            ratio_sum = 0
            energies = {}
            raw_flows = {}
            for bits, count in counts.items():
                sample = dict(zip(variables, map(int, bits)))
                cut = tuple(i for i in range(len(edges)) if sample[f'x_{i}'])
                flow = flows[cut]
                energies[bits] = bqm.energy(sample)
                raw_flows[bits] = flow if len(cut) <= B else None
                exact_budget += count * (len(cut) == B)
                if len(cut) <= B:
                    feasible += count
                    optimum += count * (flow == opt)
                    ratio_sum += count * ((flows[()]-flow)/(flows[()]-opt)
                                           if flows[()] != opt else float(flow == opt))
            score = method['scores']
            assert score['p_feasible'] == feasible/total
            assert score['p_exact_budget'] == exact_budget/total
            assert score['shot_p_optimal'] == optimum/total
            assert score['shot_mean_ratio'] == pytest.approx(ratio_sum/total, abs=1e-12)
            assert score['best_sample_energy'] == min(energies.values())
            selected = sorted(counts, key=lambda bits: (energies[bits], bits))[:20]
            pre = [raw_flows[bits] for bits in selected if raw_flows[bits] is not None]
            assert score['flow_before_repair'] == (min(pre) if pre else None)
            repaired = [tuple(edge) for edge in score['interdicted_edges']]
            assert len(repaired) == B and set(repaired) <= set(edges)
            assert surviving(graph, source, sink, repaired) == score['flow_after_repair'] >= opt
            if method['mode'].startswith('xy_'):
                assert exact_budget == total
            if method['mode'] != 'sa':
                metadata = method['metadata']
                assert 1 <= metadata['evaluations'] <= data['controls']['maxiter']
                assert len(metadata['theta']) == 2*data['controls']['reps']
                assert metadata['circuit_depth'] > 0 and metadata['two_qubit_gates'] > 0


def test_saved_campaign_manifest():
    manifest = json.loads((ROOT/'results/xy_comparison.manifest.json').read_text())
    assert manifest['result_sha256'] == hashlib.sha256(
        (ROOT/'results/xy_comparison.json').read_bytes()).hexdigest()
    assert manifest['completed_sampler_methods'] == 10
    assert manifest['inapplicable_linear_methods'] == 2
    assert manifest['failed_attempts'] == manifest['hardware_runs'] == 0

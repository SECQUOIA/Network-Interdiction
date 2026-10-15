"""Audit the frozen offline fitting cohort without a simulator or IBM account."""
import hashlib
import json
from pathlib import Path

import network_interdiction as ni

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/ibm_snapshot_resources.json'


def test_offline_resource_provenance_and_counts():
    report = json.loads(DATA.read_text())
    manifest = json.loads((ROOT/'results/ibm_snapshot_resources.manifest.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest['report_sha256']
    assert report['source_commit'] == manifest['source_commit']
    assert len(report['rows']) == 28
    assert report['packages']['qiskit-ibm-runtime'] == '0.50.0'
    for row in report['rows']:
        path = ROOT/row['instance']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['input_sha256']
        graph, _, _ = ni.read_flow_file(path)
        assert row['num_qubits'] == len(graph)-2+3*len(graph.edges)
        assert 0 <= row['budget'] <= len(graph.edges)
        assert row['cardinality_penalty'] == 10 and row['dual_penalty'] == 11
        assert row['qpu_time_sec'] is None and row['execution_wall_time_sec'] is None
        target = report['snapshots'][row['snapshot']]['qubits']
        if row['status'] == 'too_wide':
            assert row['num_qubits'] > target
            assert 'transpile_time_sec' not in row
        else:
            assert row['status'] == 'transpiled'
            assert row['num_qubits'] <= row['active_physical_qubits'] <= target
            assert len(row['variables']) == len(set(row['variables'])) == row['num_qubits']
            assert len(set(row['final_layout'])) == row['num_qubits']
            assert all(0 <= q < target for q in row['final_layout'])
            assert row['circuit_register_qubits'] == target
            assert row['operation_counts']['measure'] == row['num_qubits']
            assert row['two_qubit_gates'] == sum(row['operation_counts'].get(gate, 0)
                                               for gate in ('ecr', 'cz'))
            assert 0 < row['circuit_depth'] <= row['max_depth'] == 10000
            assert 0 <= row['transpile_time_sec'] <= row['preparation_wall_time_sec']


def test_largest_fitting_case_and_explicit_refusals():
    report = json.loads(DATA.read_text())
    for snapshot in report['snapshots']:
        rows = [row for row in report['rows'] if row['snapshot'] == snapshot]
        assert len({r['instance'] for r in rows}) == 14
        fitted = [r for r in rows if r['status'] == 'transpiled']
        assert len(fitted) == 11
        largest = max(fitted, key=lambda r: r['num_qubits'])
        assert largest['instance'] == 'testcases/mid/planar_13_s0.flow'
        assert largest['num_qubits'] == 116
        refused = {r['instance']: r['num_qubits'] for r in rows if r['status'] == 'too_wide'}
        assert refused == {'testcases/mid/erdos_13_0.25_s0.flow': 149,
                           'testcases/mid/knn_18_2_s0.flow': 142,
                           'testcases/50.planarflow': 912}

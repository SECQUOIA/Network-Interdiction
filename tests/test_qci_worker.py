"""Real qci-client schema/serialization, with all HTTP calls forbidden."""
import importlib.util
import io
import json
import os

import pytest

if importlib.util.find_spec('qci_client') is None:
    if os.environ.get('REQUIRE_QCI_SDK') == '1':
        raise ImportError('QCI SDK is required in this lane')
    pytest.skip('QCI SDK has a separate environment and CI lane', allow_module_level=True)

from qci_client import QciClient
from qci_client.optimization.data_converter import data_to_json
import requests

import qci_worker


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(requests.Session, 'request',
                        lambda *a, **k: pytest.fail('HTTP must not run in offline SDK tests'))


def request():
    return {'file': {'file_name': 'fixture', 'file_config': {'polynomial': {
        'num_variables': 3, 'min_degree': 1, 'max_degree': 2,
        'data': [{'idx': [0, 1], 'val': -3}, {'idx': [1, 2], 'val': -5}]}}},
        'job_params': {'device_type': 'dirac-3', 'num_levels': [2, 2, 2],
                       'num_samples': 20, 'relaxation_schedule': 1}}


@pytest.mark.parametrize('terminal', ['COMPLETED', 'ERRORED', 'CANCELLED'])
def test_actual_sdk_schema_and_worker_polling_stop_on_each_terminal(monkeypatch, terminal):
    client = QciClient(api_token='offline-placeholder', url='https://invalid.example', timeout=1)
    captured, events, sleeps = [], [], []
    def upload(*, file):
        # Exercise the installed SDK's real converter, normally called by upload_file.
        assert data_to_json(file=file) == file
        captured.append('upload')
        return {'file_id': 'offline-file'}
    def submit(*, job_body):
        captured.append(job_body)
        assert job_body == {'job_submission': {
            'problem_config': {'qudit_hamiltonian_optimization': {'polynomial_file_id': 'offline-file'}},
            'device_config': {'dirac-3_qudit': {'num_samples': 20,
                'num_levels': [2, 2, 2], 'relaxation_schedule': 1}}}}
        return {'job_id': 'offline-job'}
    statuses = iter(['QUEUED', 'RUNNING', terminal])
    monkeypatch.setattr(client, 'upload_file', upload)
    monkeypatch.setattr(client, 'submit_job', submit)
    monkeypatch.setattr(client, 'get_job_status', lambda **k: {'status': next(statuses)})
    monkeypatch.setattr(client, 'get_job_results', lambda **k: {'status': terminal,
        'results': {'solutions': [[0, 1, 0]], 'counts': [20], 'energies': [0]}
                   if terminal == 'COMPLETED' else None,
        'job_info': {'job_result': {'device_usage_s': 2}}})
    qci_worker.run_request(request(), client, events.append, sleep=sleeps.append)
    assert captured[0] == 'upload' and len(captured) == 2
    assert sleeps == [1, 1]
    assert events[0] == {'event': 'submitted', 'job_id': 'offline-job'}
    assert events[-1]['status'] == terminal and events[-1]['client_version'] == '5.0.3'
    assert events[-1]['device_usage_sec'] == 2


def test_worker_main_sanitizes_sdk_exceptions(monkeypatch, capsys):
    import qci_client
    def failed_client(**kwargs):
        raise RuntimeError('credential-secret-placeholder')
    monkeypatch.setattr(qci_client, 'QciClient', failed_client)
    monkeypatch.setattr(qci_worker.sys, 'stdin', io.StringIO(json.dumps(request())))
    assert qci_worker.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {'event': 'error', 'error_type': 'RuntimeError'}
    assert 'credential-secret-placeholder' not in output

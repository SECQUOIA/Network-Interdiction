"""QCI transport contracts and NIP scoring, with no hardware submissions."""
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import dimod
import pytest

import solve_qubo_via_qci as qci
import run_benchmark as rb

ROOT = Path(__file__).resolve().parents[1]


def model():
    return dimod.BinaryQuadraticModel({'gamma_2': -3, ('x', 7): 2, 9: 0},
                                     {('gamma_2', ('x', 7)): -5, (9, ('x', 7)): 1.5},
                                     11, dimod.BINARY)


def response(solutions=None, counts=None):
    return {'event': 'result', 'status': 'COMPLETED', 'job_id': 'offline-job',
            'client_version': '5.0.3', 'device_usage_sec': 0,
            'results': {'solutions': solutions or [[0, 0, 0], [1, 1, 0]],
                        'counts': counts or [3, 17], 'energies': [999, 999]}}


def test_polynomial_matches_every_binary_assignment_including_mixed_labels():
    bqm = model()
    variables, request = qci.prepare_request(bqm)
    poly = request['file']['file_config']['polynomial']
    assert poly['num_variables'] == 3
    assert request['job_params']['num_levels'] == [2, 2, 2]
    for bits in itertools.product((0, 1), repeat=3):
        energy = bqm.offset
        for term in poly['data']:
            value = term['val']
            for index in term['idx']:
                value *= bits[index-1] if index else 1
            energy += value
        assert energy == bqm.energy(dict(zip(variables, bits)))
    assert json.loads(json.dumps(request)) == request


def test_linear_only_polynomial_keeps_unused_variable_dimensions():
    variables, request = qci.prepare_request(dimod.BQM({'a': 2, 'b': 0}, {}, 1, 'BINARY'))
    assert len(variables) == 2
    assert request['file']['file_config']['polynomial'] == {
        'num_variables': 2, 'min_degree': 1, 'max_degree': 1,
        'data': [{'idx': [1], 'val': 2.0}]}


@pytest.mark.parametrize('kwargs', [{'num_samples': 0}, {'num_samples': 101},
    {'num_samples': True}, {'relaxation_schedule': 5}, {'relaxation_schedule': 1.0}])
def test_invalid_device_controls_fail_before_launch(monkeypatch, kwargs):
    monkeypatch.setattr(qci.subprocess, 'run', lambda *a, **k: pytest.fail('worker launched'))
    with pytest.raises(ValueError):
        qci.solve_bqm_qci(model(), **kwargs)


@pytest.mark.parametrize('bqm', [dimod.BQM({}, {}, 0, 'BINARY'),
    dimod.BQM({'a': 0}, {}, 4, 'BINARY'), dimod.BQM({'a': 1}, {}, 0, 'SPIN'),
    dimod.BQM({'a': float('nan')}, {}, 0, 'BINARY'),
    dimod.BQM({i: 1 for i in range(475)}, {}, 0, 'BINARY')])
def test_invalid_or_oversized_models_are_not_submitted(bqm):
    with pytest.raises(ValueError):
        qci.prepare_request(bqm)


def test_recomputed_energy_restores_offset_and_occurrences():
    bqm = model()
    variables, _ = qci.prepare_request(bqm)
    samples = qci.decode_response(bqm, variables, response(), 20)
    assert list(samples.record.num_occurrences) == [3, 17]
    assert list(samples.record.energy) == [11, 5]
    assert samples.info['qci_device_usage_sec'] == 0
    assert samples.info['qci_job_id'] == 'offline-job'


@pytest.mark.parametrize('solutions,counts', [([[0, 1]], [20]), ([[0, 1, 0.4]], [20]),
    ([[0, 0, 0]], [-1]), ([[0, 0, 0]], [True]), ([[0, 0, 0]], [19]),
    ([[0, 0, 0]], [20.0]), ([[0, 0, 0]], [1, 19])])
def test_malformed_responses_are_not_silently_rounded_or_truncated(solutions, counts):
    with pytest.raises(ValueError):
        qci.decode_response(model(), list(model().variables), response(solutions, counts), 20)


def test_missing_device_time_remains_unmeasured():
    result = response()
    result.pop('device_usage_sec')
    assert qci.decode_response(model(), list(model().variables), result, 20).info['qci_device_usage_sec'] is None


@pytest.mark.parametrize('status', ['ERRORED', 'CANCELLED', 'RUNNING'])
def test_noncompleted_jobs_do_not_become_samples(status):
    result = response()
    result['status'] = status
    with pytest.raises(RuntimeError, match=status):
        qci.decode_response(model(), list(model().variables), result, 20)


def test_worker_exchange_does_not_serialize_credentials(monkeypatch):
    monkeypatch.setenv('QCI_TOKEN', 'secret-placeholder')
    monkeypatch.setenv('QCI_PYTHON', '/isolated/python')
    calls = []
    def launch(command, **options):
        calls.append((command, options))
        assert command[0] == '/isolated/python'
        assert 'secret-placeholder' not in options['input']
        assert options['timeout'] == 3
        return SimpleNamespace(returncode=0, stdout=json.dumps(response())+'\n')
    monkeypatch.setattr(qci.subprocess, 'run', launch)
    samples = qci.solve_bqm_qci(model(), timeout_sec=3)
    assert len(calls) == 1 and len(samples) == 2


def test_timeout_retains_job_receipt_and_does_not_retry(monkeypatch):
    monkeypatch.setenv('QCI_TOKEN', 'placeholder')
    calls = []
    def timeout(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 0.1,
            output=b'{"event":"submitted","job_id":"offline-job"}\n',
            stderr=b'secret-placeholder')
    monkeypatch.setattr(qci.subprocess, 'run', timeout)
    with pytest.raises(TimeoutError, match='offline-job') as error:
        qci.solve_bqm_qci(model(), python_executable='unused', timeout_sec=0.1)
    assert len(calls) == 1 and 'secret-placeholder' not in str(error.value)


def test_configuration_absence_does_not_start_a_worker(monkeypatch):
    monkeypatch.delenv('QCI_PYTHON', raising=False)
    monkeypatch.delenv('QCI_TOKEN', raising=False)
    monkeypatch.setattr(qci.subprocess, 'run', lambda *a, **k: pytest.fail('worker launched'))
    with pytest.raises(ValueError, match='QCI_PYTHON'):
        qci.solve_bqm_qci(model())
    with pytest.raises(ValueError, match='QCI_TOKEN'):
        qci.solve_bqm_qci(model(), python_executable='unused')


def test_qci_pass_uses_shared_reference_scoring_and_optional_columns(monkeypatch):
    observed = []
    def offline_sample(bqm, **options):
        observed.append(options)
        sample = dimod.ExactSolver().sample(bqm).first.sample
        return dimod.SampleSet.from_samples_bqm([sample], bqm, num_occurrences=[20],
            info={'qci_job_id': 'offline-job', 'qci_device': 'dirac-3',
                  'qci_device_usage_sec': 0, 'qci_client_version': '5.0.3'})
    monkeypatch.setattr(qci, 'solve_bqm_qci', offline_sample)
    row = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 6, 'brute', ['qci'],
                      qci_options={'num_samples': 20})
    assert observed == [{'num_samples': 20}]
    assert row['Opt_NIP'] == row['QCI_NIP'] == row['QCI_NIP_Greedy'] == 5
    assert row['Opt_QUBO'] == row['QCI_QUBO']
    assert row['QCI_raw_feasible_count'] == row['QCI_raw_samples_examined'] == 1
    assert json.loads(row['QCI_raw_samples'])[0]['num_occurrences'] == 20
    assert row['QCI_job_id'] == 'offline-job' and row['QCI_device_usage_sec'] == 0
    assert all(row[c] == '' for solver in ('sa', 'qa') for c in rb.SOLVER_COLUMNS[solver])
    assert row['QCI_time_sec'] >= 0 and row['QCI_repair_time_sec'] >= 0
    blank = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 6, 'brute', [])
    assert all(value == '' for key, value in blank.items() if key.startswith('QCI_'))


def test_qci_cli_limits_fail_before_reading_an_instance(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, 'run_case', lambda *a, **k: pytest.fail('case dispatched'))
    with pytest.raises(SystemExit) as error:
        rb.main(['--case', 'unused.flow:1:auto', '--solvers', 'qci',
                 '--qci-samples', '101', '--out', str(tmp_path / 'unused.csv')])
    assert error.value.code == 2


def test_empty_lambda_window_leaves_qci_unrun(monkeypatch, tmp_path):
    from test_lambda_window import write_two_equal_paths
    path = tmp_path / 'equal.flow'
    write_two_equal_paths(path)
    monkeypatch.setattr(qci, 'solve_bqm_qci', lambda *a, **k: pytest.fail('hardware dispatched'))
    row = rb.run_case(str(path), 1, 'auto', 'brute', ['qci'])
    assert row['Opt_NIP'] == 5 and row['lambda'] == ''
    assert all(value == '' for key, value in row.items() if key.startswith('QCI_'))

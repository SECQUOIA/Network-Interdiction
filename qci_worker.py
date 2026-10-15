"""SDK-only subprocess for solve_qubo_via_qci; no imports from the core project.

stdin is one polynomial/job request. stdout emits a submission receipt followed
by a result, both JSON lines. Exceptions expose their type, never SDK messages
that might contain credentials. A killed worker does not cancel remote jobs.
"""
import contextlib
import importlib.metadata
import json
import os
import sys
import time


def run_request(request, client, emit, sleep=time.sleep):
    """Submit exactly once and poll all terminal states using the actual SDK."""
    from qci_client import JobStatus, JOB_STATUSES_FINAL
    # Validate the job configuration before the upload (the SDK builder is local).
    client.build_job_body(job_type='sample-hamiltonian-integer',
                          polynomial_file_id='validation-only',
                          job_params=request['job_params'])
    uploaded = client.upload_file(file=request['file'])
    job_body = client.build_job_body(job_type='sample-hamiltonian-integer',
                                     polynomial_file_id=uploaded['file_id'],
                                     job_params=request['job_params'])
    submitted = client.submit_job(job_body=job_body)
    job_id = submitted['job_id']
    emit({'event': 'submitted', 'job_id': job_id})
    while True:
        status = JobStatus(client.get_job_status(job_id=job_id)['status'])
        if status in JOB_STATUSES_FINAL:
            break
        sleep(1)
    response = client.get_job_results(job_id=job_id)
    result = {'event': 'result', 'job_id': job_id,
              'status': response['status'], 'results': response.get('results'),
              'device_usage_sec': response.get('job_info', {}).get('job_result', {}).get('device_usage_s'),
              'client_version': importlib.metadata.version('qci-client')}
    emit(result)


def main():
    output = sys.stdout
    def emit(message):
        output.write(json.dumps(message, allow_nan=False) + '\n')
        output.flush()
    try:
        # Keep SDK output out of the JSON channel; parent suppresses stderr too.
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            from qci_client import QciClient
            if importlib.metadata.version('qci-client') != '5.0.3':
                raise RuntimeError('install the pinned requirements-qci.txt')
            request = json.load(sys.stdin)
            client = QciClient(timeout=30)
            run_request(request, client, emit)
    except Exception as error:
        emit({'event': 'error', 'error_type': type(error).__name__})
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

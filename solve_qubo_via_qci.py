"""Binary BQM sampling on Dirac-3 through an isolated qci-client worker.

The SDK requires NetworkX <3; this repository requires >=3.2. Install
requirements-qci.txt into a separate environment and set QCI_PYTHON to its
Python executable. QCI_TOKEN stays in the environment, never in JSON or argv.
Calling solve_bqm_qci submits one hardware job; there is no automatic resubmission.
"""
import json
import math
import numbers
import os
from pathlib import Path
import subprocess

import dimod

# The current cloud guide documents 949 optical levels, two per binary
# variable. This is a conservative submission guard, not an empirical fit.
MAX_BINARY_VARIABLES = 474


def prepare_request(bqm, *, num_samples=20, relaxation_schedule=1):
    """Return (local variable order, JSON request) without SDK or network access.

    Dirac-3's integer mode with num_levels=[2,...,2] preserves the binary
    domain. Polynomial indices are one-based, padded with zero for linear
    terms; each quadratic bias appears once. The constant offset stays local.
    """
    if bqm.vartype is not dimod.BINARY:
        raise ValueError('QCI requires a BINARY BQM')
    if not 1 <= len(bqm) <= MAX_BINARY_VARIABLES:
        raise ValueError(f'QCI accepts 1..{MAX_BINARY_VARIABLES} binary variables here')
    if (isinstance(num_samples, bool) or not isinstance(num_samples, numbers.Integral)
            or not 1 <= num_samples <= 100):
        raise ValueError('QCI num_samples must be an integer in 1..100')
    if (isinstance(relaxation_schedule, bool)
            or not isinstance(relaxation_schedule, numbers.Integral)
            or relaxation_schedule not in (1, 2, 3, 4)):
        raise ValueError('QCI relaxation_schedule must be an integer in 1..4')
    if not all(math.isfinite(value) for value in
               [bqm.offset, *bqm.linear.values(), *bqm.quadratic.values()]):
        raise ValueError('QCI coefficients and offset must be finite')
    variables = list(bqm.variables)
    positions = {v: i+1 for i, v in enumerate(variables)}
    terms = [([0, positions[v]], float(bias)) for v, bias in bqm.linear.items() if bias]
    terms += [(sorted([positions[u], positions[v]]), float(bias))
              for (u, v), bias in bqm.quadratic.items() if bias]
    if not terms:
        raise ValueError('QCI submission requires a nonconstant objective')
    degree = max(2 if idx[0] else 1 for idx, _ in terms)
    data = [{'idx': idx[-degree:], 'val': bias} for idx, bias in terms]
    polynomial = {'num_variables': len(variables),
                  'min_degree': min(sum(i != 0 for i in row['idx']) for row in data),
                  'max_degree': degree, 'data': data}
    return variables, {
        'file': {'file_name': 'network-interdiction-bqm',
                 'file_config': {'polynomial': polynomial}},
        'job_params': {'device_type': 'dirac-3', 'num_levels': [2]*len(variables),
                       'num_samples': int(num_samples),
                       'relaxation_schedule': int(relaxation_schedule)},
    }


def decode_response(bqm, variables, response, expected_samples):
    """Validate the binary result and recompute energies in original BQM units."""
    if response.get('status') != 'COMPLETED':
        raise RuntimeError(f"QCI job did not complete: {response.get('status')!r}")
    result = response.get('results') or {}
    solutions, counts = result.get('solutions', []), result.get('counts', [])
    if not solutions or len(solutions) != len(counts):
        raise ValueError('QCI returned empty or mismatched solutions/counts')
    if any(len(row) != len(variables) or any(value not in (0, 1) for value in row)
           for row in solutions):
        raise ValueError('QCI returned a nonbinary or wrongly sized solution')
    if any(isinstance(count, bool) or not isinstance(count, numbers.Integral)
           or count <= 0 for count in counts) or sum(counts) != expected_samples:
        raise ValueError('QCI occurrence counts do not match requested samples')
    usage = response.get('device_usage_sec')
    if usage is not None and (isinstance(usage, bool) or not isinstance(usage, numbers.Real)
                              or not math.isfinite(usage) or usage < 0):
        raise ValueError('QCI device usage must be finite and nonnegative')
    # device_usage_s is rounded billable usage, not precise QPU execution time.
    info = {'qci_job_id': response.get('job_id'), 'qci_device': 'dirac-3',
            'qci_device_usage_sec': usage, 'qci_client_version': response.get('client_version')}
    return dimod.SampleSet.from_samples_bqm(
        (solutions, variables), bqm, num_occurrences=counts, info=info)


def _messages(output):
    if isinstance(output, bytes):
        output = output.decode(errors='replace')
    messages = []
    for line in (output or '').splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            messages.append(value)
    return messages


def solve_bqm_qci(bqm, *, python_executable=None, num_samples=20,
                  relaxation_schedule=1, timeout_sec=300):
    """Submit once and return a SampleSet; timeout does not cancel the remote job.

    Wall time includes worker startup, authentication, upload, queue and result
    retrieval. A timeout/error reports the job ID when received; check that job
    before deciding whether to submit again. Device usage is in SampleSet.info.
    """
    variables, request = prepare_request(bqm, num_samples=num_samples,
                                         relaxation_schedule=relaxation_schedule)
    if (isinstance(timeout_sec, bool) or not isinstance(timeout_sec, numbers.Real)
            or not math.isfinite(timeout_sec) or timeout_sec <= 0):
        raise ValueError('QCI timeout_sec must be finite and positive')
    executable = python_executable or os.environ.get('QCI_PYTHON')
    if not executable:
        raise ValueError('Set QCI_PYTHON to the separate requirements-qci.txt environment')
    if not os.environ.get('QCI_TOKEN'):
        raise ValueError('Set QCI_TOKEN in the environment before requesting a QCI job')
    worker = Path(__file__).with_name('qci_worker.py')
    try:
        process = subprocess.run([str(executable), str(worker)],
            input=json.dumps(request, allow_nan=False), text=True,
            capture_output=True, timeout=float(timeout_sec), check=False)
    except subprocess.TimeoutExpired as error:
        messages = _messages(error.stdout)
        job = next((m['job_id'] for m in messages if m.get('job_id')), 'unknown')
        raise TimeoutError(f'QCI wait timed out; job ID {job}. The remote job may '
                           'still run; inspect it before resubmitting.') from None
    messages = _messages(process.stdout)
    if process.returncode or not messages or messages[-1].get('event') != 'result':
        job = next((m['job_id'] for m in messages if m.get('job_id')), 'unknown')
        raise RuntimeError(f'QCI worker failed; job ID {job}. Check SDK setup, credentials '
                           'and remote job state before resubmitting.')
    try:
        return decode_response(bqm, variables, messages[-1], int(num_samples))
    except (ValueError, RuntimeError, TypeError, KeyError) as error:
        job = next((m['job_id'] for m in messages if m.get('job_id')), 'unknown')
        # Do not expose arbitrary response values or a third-party exception
        # message. The job receipt is enough to retrieve the detailed diagnosis.
        raise RuntimeError(f'QCI job ID {job}: unsuccessful or invalid result '
                           f'({type(error).__name__}). Inspect the remote job '
                           'before resubmitting.') from None

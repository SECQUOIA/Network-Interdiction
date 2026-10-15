"""A killed/crashed native worker must retain its last usable checkpoint."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import run_hard_mps_probe as probe


@pytest.mark.parametrize('cause',['timeout','crash'])
def test_native_failure_preserves_completed_sampling_checkpoint(monkeypatch,cause):
    saved={'status':'sampler_completed','observed_counts':[{'bits':'010','occurrences':256}],
           'qubits':1095,'couplings':25000}
    def run(command,**kwargs):
        Path(command[-1]).write_text(json.dumps(saved))
        if cause=='timeout':
            raise subprocess.TimeoutExpired(command,kwargs['timeout'])
        return SimpleNamespace(returncode=-11)
    monkeypatch.setattr(probe.subprocess,'run',run)
    result=probe.run_method('penalty')
    assert result['status']==('outer_time_limit' if cause=='timeout' else 'worker_failed')
    assert result['last_completed_stage']=='sampler_completed'
    assert result['observed_counts']==saved['observed_counts']
    assert result['qubits']==1095

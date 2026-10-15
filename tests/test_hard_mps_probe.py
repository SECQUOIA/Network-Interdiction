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


def test_worker_refuses_existing_output_before_dispatch(tmp_path,monkeypatch):
    path=tmp_path/'retained.json'
    path.write_text('retain this attempt')
    monkeypatch.setattr(probe,'worker',lambda *a:pytest.fail('worker dispatched'))
    with pytest.raises(FileExistsError):
        probe.main(['--worker','xy','--out',str(path)])
    assert path.read_text()=='retain this attempt'


def test_early_native_crash_with_empty_reservation_is_recorded(monkeypatch):
    def run(command,**kwargs):
        Path(command[-1]).touch()
        return SimpleNamespace(returncode=-11)
    monkeypatch.setattr(probe.subprocess,'run',run)
    result=probe.run_method('penalty')
    assert result['status']=='worker_failed'
    assert result['last_completed_stage'] is None
    assert result['returncode']==-11


def test_retained_hard_probe_matches_original_flows_and_sample_energies():
    import export_matched_results as exported
    root=Path(__file__).resolve().parents[1]
    artifact=json.loads((root/'results/hard_mps_probe.json').read_text())
    summary=exported.audit_hard_probe(artifact)
    assert summary==json.loads((root/'results/hard_mps_audit.json').read_text())
    assert artifact['attempts']==[dict(attempt=1,status='completed',completed_methods=2)]
    assert summary['optimum']==8
    assert summary['basis_initial_flow']==summary['methods']['xy']['flow']==10
    assert summary['methods']['xy']['exact_budget_shots']==241
    assert summary['methods']['xy']['at_most_budget_shots']==254
    assert summary['methods']['penalty']['shots']==0
    assert artifact['methods']['xy']['metadata']['evaluations']==4

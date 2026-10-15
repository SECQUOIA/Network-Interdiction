"""Outer deadlines must stop native workers' descendants as well as parents."""
import os
from pathlib import Path
import signal
import time

import pytest

import run_hard_mps_probe as probe
import run_matched_benchmark as matched


def running(pid):
    """An adopted zombie is terminated, even before the host reaps its record."""
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != 'Z'


@pytest.mark.parametrize('launcher', ['matched', 'hard_probe'])
def test_outer_timeout_stops_descendants_and_preserves_checkpoint(tmp_path, monkeypatch, launcher):
    pid_file = tmp_path / 'descendant.pid'
    script = tmp_path / 'run_hard_mps_probe.py'
    script.write_text(
        'from pathlib import Path\n'
        'import json, subprocess, sys, time\n'
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f'Path({str(pid_file)!r}).write_text(str(child.pid))\n'
        "Path(sys.argv[sys.argv.index('--out') + 1]).write_text(json.dumps({'status': 'running_sampler'}))\n"
        'time.sleep(60)\n'
    )
    try:
        if launcher == 'matched':
            result = matched.process_call([], timeout=2, script=script)
        else:
            monkeypatch.setattr(probe, 'ROOT', tmp_path)
            monkeypatch.setattr(probe, 'OUTER_SECONDS', 2)
            result = probe.run_method('penalty')
            assert result['last_completed_stage'] == 'running_sampler'
        assert result['status'] == 'outer_time_limit'
        assert pid_file.exists(), 'the descendant must start before the timeout'
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 1
        while running(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not running(pid), 'outer timeout left its descendant running'
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text())
            if running(pid):
                os.kill(pid, signal.SIGKILL)

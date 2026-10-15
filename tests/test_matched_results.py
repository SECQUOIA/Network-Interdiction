"""Recompute retained witnesses, sampler energies and embedding certificates."""
import json
from pathlib import Path

import pytest

import export_matched_results as exported
import run_matched_benchmark as matched

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def artifact():
    return json.loads((ROOT/'results/matched_scaling.json').read_text())


@pytest.mark.parametrize('key',matched.SELECTION)
def test_matched_witnesses_and_surrogate_bounds(artifact,key):
    row=next(r for r in artifact['cases'] if (r['name'],r['B'])==key)
    ref=matched.cases_and_references()[key]
    loaded,bqm,certificate=exported.audit_case(row,ref)
    assert set(row['methods'])==set(matched.METHODS)
    assert all(result['status']!='dispatched' for result in row['methods'].values())
    output=exported.export_row(row,ref,loaded,bqm,certificate)
    if key==('cover_n30_s0',192):
        assert output['Opt_NIP']==8
        assert output['Decomp_multiplier_iterated']==''
        assert output['Decomp_multiplier_iterated_status']=='curve_work_exceeds_allowance'
    assert all(output[c]=='' for c in ['QA_NIP(advantage1)','QCI_NIP','QAOA_NIP'])


def test_matched_attempt_ledger_is_complete(artifact):
    assert artifact['attempts']==[dict(attempt=1,status='completed',completed_methods=108)]
    assert len(artifact['cases'])==9


def test_simulator_shots_budgets_energies_and_feasible_results():
    artifact=json.loads((ROOT/'results/simulator_scaling.json').read_text())
    summary=exported.audit_simulators(artifact)
    assert artifact['attempts']==[dict(attempt=1,status='completed',completed_methods=42)]
    hard=[r for r in summary if r['file'].endswith('cover_n30_s0.flow')]
    assert len(hard)==4
    assert all(r['status']=='qubit_guard_refused' and not r['meaningful'] for r in hard)

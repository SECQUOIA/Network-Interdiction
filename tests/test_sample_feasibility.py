"""Raw feasibility must survive QA descent, aggregation and best-20 selection."""
import json
from pathlib import Path
from types import SimpleNamespace

import dimod
import pytest

import network_interdiction as ni
import run_benchmark as rb
import solve_qubo_via_dwave as dwave
from test_lambda_window import write_two_equal_paths

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return ni.read_flow_file(ROOT / 'testcases/tiny_a.flow')


def samples(cuts, energies=None, occurrences=None):
    graph, source, sink = fixture()
    bqm = ni.nip_to_qubo(1, graph, source, sink, 6, 7)
    values = []
    for indices in cuts:
        value = dict.fromkeys(bqm.variables, 0)
        value.update({f'x_{index}': 1 for index in indices})
        values.append(value)
    return dimod.SampleSet.from_samples(values, dimod.BINARY,
                                       energy=energies or list(range(len(cuts))),
                                       num_occurrences=occurrences)


def run(monkeypatch, raw, processed=None, solver='sa'):
    processed = raw if processed is None else processed
    def sample(bqm, annealer_type, seed=None, *, return_raw=False):
        return (processed, raw) if return_raw else processed
    monkeypatch.setattr(dwave, 'solve_bqm_dwave', sample)
    graph, source, sink = fixture()
    return ni.qubo_verify_parameters('audit', graph, 6, source, sink, 1, solver)


def test_count_uses_energy_ordered_records_and_preserves_occurrences(monkeypatch):
    result = run(monkeypatch, samples([[], [0], [0, 1]], [3, 2, 1], [2, 50, 3]))
    assert result['raw_feasible_count'] == 2
    assert result['raw_samples_examined'] == 3
    records = result['raw_samples']
    assert [r['energy'] for r in records] == [1, 2, 3]
    assert [r['num_occurrences'] for r in records] == [3, 50, 2]
    assert [r['budget_feasible'] for r in records] == [False, True, True]
    assert result['raw_samples'] == result['pre_greedy_samples']
    graph, source, sink = fixture()
    for record in records:
        assert rb.max_flow(graph, source, sink, record['interdicted_edges']) == record['flow']
    assert result['best_obj_before_greedy'] == 5


def test_only_the_best_twenty_records_contribute(monkeypatch):
    result = run(monkeypatch, samples([[0, 1]] * 20 + [[], [0]]))
    assert result['raw_samples_examined'] == 20
    assert result['raw_feasible_count'] == 0
    assert result['best_obj_before_greedy'] == float('inf')
    assert len(result['raw_samples']) == 20


def test_qa_raw_count_precedes_classical_descent(monkeypatch):
    raw = samples([[0, 1], [0, 1, 2]])
    processed = samples([[], [0]])
    result = run(monkeypatch, raw, processed, solver='qa')
    assert result['raw_feasible_count'] == 0
    assert result['pre_greedy_feasible_count'] == 2
    assert result['raw_samples_examined'] == result['pre_greedy_samples_examined'] == 2
    assert result['best_obj_before_greedy'] == 5
    assert result['raw_samples'] != result['pre_greedy_samples']
    row = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 6, 'brute', ['qa'])
    assert row['QA_raw_feasible_count'] == 0
    assert row['QA_pre_greedy_feasible_count'] == 2
    assert row['QA_NIP(advantage1)'] == 5
    assert len(json.loads(row['QA_raw_samples'])) == row['QA_raw_samples_examined'] == 2


def test_qa_adapter_preserves_raw_samples_without_changing_default_return(monkeypatch):
    raw, processed = samples([[0, 1]]), samples([[0]])
    calls = []
    def qpu_sample(bqm, **kwargs):
        calls.append(kwargs)
        return raw
    def descent(bqm, initial_states):
        assert initial_states is raw
        return processed
    monkeypatch.setattr(dwave, 'DWaveSampler', lambda **kwargs: object())
    monkeypatch.setattr(dwave, 'EmbeddingComposite', lambda qpu: SimpleNamespace(sample=qpu_sample))
    monkeypatch.setattr(dwave, 'SteepestDescentSolver', lambda: SimpleNamespace(sample=descent))
    bqm = dimod.BinaryQuadraticModel({'x': 1}, {}, 0, dimod.BINARY)
    assert dwave.solve_bqm_dwave(bqm, 'qa') is processed
    assert dwave.solve_bqm_dwave(bqm, 'qa', return_raw=True) == (processed, raw)
    assert calls == [{'num_reads': 2500, 'annealing_time': 50}] * 2


def test_empty_samples_fail_clearly_before_repair(monkeypatch):
    empty = samples([[]]).slice(0)
    monkeypatch.setattr(ni, 'improve_dwave_solution', lambda *a: pytest.fail('repair dispatched'))
    with pytest.raises(ValueError, match='empty'):
        run(monkeypatch, empty)


def test_qa_descent_preserves_original_records_with_real_postprocessor(monkeypatch):
    raw = samples([[0, 1], []])
    before = raw.record.copy()
    monkeypatch.setattr(dwave, 'DWaveSampler', lambda **kwargs: object())
    monkeypatch.setattr(dwave, 'EmbeddingComposite', lambda qpu:
                        SimpleNamespace(sample=lambda *args, **kwargs: raw))
    graph, source, sink = fixture()
    bqm = ni.nip_to_qubo(1, graph, source, sink, 6, 7)
    processed, original = dwave.solve_bqm_dwave(bqm, 'qa', return_raw=True)
    assert original is raw and processed is not raw
    import numpy as np
    assert np.array_equal(raw.record, before)


def test_empty_lambda_window_does_not_dispatch_or_fabricate_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(ni, 'qubo_verify_parameters', lambda *a, **k: pytest.fail('sample dispatched'))
    instance = write_two_equal_paths(tmp_path / 'equal.flow')
    row = rb.run_case(instance, 1, 'auto', 'brute', ['sa'])
    assert row['lambda'] == ''
    assert all(row[column] == '' for column in rb.SAMPLE_COLUMNS)


def test_csv_counts_and_json_survive_roundtrip_and_unrun_qa_stays_empty(tmp_path):
    path = tmp_path / 'sample.csv'
    assert rb.main(['--exact', 'brute', '--solvers', 'sa', '--baselines', 'none',
                    '--case', f'{ROOT / "testcases/tiny_a.flow"}:1:6',
                    '--out', str(path)]) == 0
    import pandas as pd
    row = pd.read_csv(path, comment='#').iloc[0]
    records = json.loads(row.SA_raw_samples)
    assert row.SA_raw_samples_examined == len(records) == 20
    assert row.SA_raw_feasible_count == sum(record['budget_feasible'] for record in records)
    assert row.SA_raw_feasible_count == row.SA_pre_greedy_feasible_count
    assert json.loads(row.SA_pre_greedy_samples) == records
    assert all(pd.isna(row[column]) for column in rb.SAMPLE_COLUMNS if column.startswith('QA_'))
    graph, source, sink = fixture()
    for record in records:
        assert rb.max_flow(graph, source, sink, record['interdicted_edges']) == record['flow']


def test_no_sampler_leaves_all_diagnostics_empty():
    row = rb.run_case(str(ROOT / 'testcases/tiny_a.flow'), 1, 6, 'brute', [])
    assert all(row[column] == '' for column in rb.SAMPLE_COLUMNS)

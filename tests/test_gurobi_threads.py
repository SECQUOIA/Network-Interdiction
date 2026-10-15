"""Verify the thread budget at optimize(), including nested reference solves."""
from pathlib import Path

import gurobipy
import pytest

import decomposition as dc
import network_interdiction as ni
import run_benchmark as rb
import run_decomposition_benchmark as rdb
from test_gurobi_trajectory import require_gurobi


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('solver', ['exact', 'penalty', 'relaxation',
                                   'piece', 'benchmark', 'cli'])
def test_all_gurobi_solves_use_eight_threads(solver, monkeypatch, tmp_path):
    require_gurobi()
    observed = []

    class RecordingModel(gurobipy.Model):
        def optimize(self, *args, **kwargs):
            observed.append(self.Params.Threads)
            return super().optimize(*args, **kwargs)

    monkeypatch.setattr(ni, 'Model', RecordingModel)
    path = ROOT / 'testcases/tiny_a.flow'
    graph, source, sink = ni.read_flow_file(path)
    if solver == 'exact':
        objective, _ = ni.solve_nip_gurobi(graph, source, sink, 1)
        assert objective == pytest.approx(5)
    elif solver == 'penalty':
        _, objective, _ = ni.gurobi_verify_parameters(graph, source, sink, 1, 6, 7)
        assert objective == 5
    elif solver == 'relaxation':
        ni.solve_nip_gurobi_relaxation(graph, source, sink, 1, 5, 10)
    elif solver == 'piece':
        sub = dc.piece_subproblem(graph, source, sink, [{0, 1}, {2, 3}], 0)
        cuts = dc.gurobi_piece_solver(sub, 1)
        assert rb.max_flow(graph, source, sink, cuts) == 5
    elif solver == 'benchmark':
        row = rb.run_case(str(path), 1, 'auto', 'gurobi', solvers=[])
        assert row['Opt_NIP'] == pytest.approx(5)
        assert row['Gurobi_time_sec'] >= 0
    else:
        output = tmp_path / 'reference.csv'
        rb.main(['--case', f'{path}:1:auto', '--solvers', 'none',
                 '--baselines', 'none', '--out', str(output)])
        assert 'Gurobi trajectory solve threads: 8' in output.read_text()
    assert observed and set(observed) == {8}


def test_decomposition_refuses_to_overwrite_previous_results(tmp_path):
    output = tmp_path / 'benchmark.csv'
    original = b'previous benchmark evidence\n'
    output.write_bytes(original)
    with pytest.raises(FileExistsError):
        rdb.main(['--cases', str(ROOT / 'results/slide_instances_sa.csv'),
                  '--out', str(output)])
    assert output.read_bytes() == original

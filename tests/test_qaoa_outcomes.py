"""
results/qaoa_outcomes.csv: every stored QAOA state's probability, split
over the interdiction plans its x bits propose (qaoa_outcomes.py).

The data checks need only pandas. The re-simulation checks need Qiskit and
skip without it, unless REQUIRE_QAOA_BACKENDS is set (CI's qaoa job).
"""
import importlib
import json
import os

import numpy as np
import pandas
import pytest

import qaoa_common as qc
import qaoa_outcomes as qo

OUTCOME_COLUMNS = [f'p_{name}' for name in qo.OUTCOMES]


@pytest.fixture(scope='module')
def outcomes():
    return pandas.read_csv(qo.REPO / 'results' / 'qaoa_outcomes.csv',
                           comment='#')


def require_qiskit():
    for dependency in ('qiskit', 'qiskit_aer'):
        if os.environ.get('REQUIRE_QAOA_BACKENDS'):
            importlib.import_module(dependency)
        else:
            pytest.importorskip(dependency)


def test_header_names_a_clean_commit():
    header = [line for line in (qo.REPO / 'results' / 'qaoa_outcomes.csv')
              .read_text().splitlines() if line.startswith('#')]
    commit = [line for line in header if line.startswith('# repo commit:')]
    assert len(commit) == 1 and not commit[0].endswith('-dirty')


def test_outcomes_are_a_probability_split(outcomes):
    values = outcomes[OUTCOME_COLUMNS]
    assert ((values >= -1e-12) & (values <= 1 + 1e-12)).all().all()
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-9)


def test_every_stored_state_appears_once(outcomes):
    tiny = pandas.read_csv(qo.TINY_CSV, comment='#')
    transfer = pandas.read_csv(qo.TRANSFER_CSV, comment='#')
    expected = {(f"{os.path.splitext(os.path.basename(r.instance))[0]} B={r.B}",
                 'qaoa', r.reps) for r in tiny.itertuples()}
    expected |= {(r.group, r.method, r.reps) for r in transfer.itertuples()}
    stored = outcomes[outcomes['reps'] > 0]
    keys = list(zip(stored['case'], stored['method'], stored['reps']))
    assert len(keys) == len(set(keys)) == len(expected)
    assert set(keys) == expected


def test_stored_p_optimal_is_reproduced(outcomes):
    checked = outcomes.dropna(subset=['stored_p_optimal'])
    # 18 tiny states, their 6 uniform states, 60 transfer states.
    assert len(checked) == 84
    np.testing.assert_allclose(checked['p_optimal'],
                               checked['stored_p_optimal'], atol=1e-6)


def test_uniform_rows_count_plans(outcomes):
    """
    The uniform state puts 2**-E on every x pattern, so its split is a plan
    count -- computed here by classifying plans, with no simulation.
    """
    uniform = outcomes[(outcomes['source'] == 'tiny')
                       & (outcomes['method'] == 'uniform')]
    assert len(uniform) == 6
    for row in uniform.itertuples():
        problem = qc.build_problem(row.instance, int(row.B))
        counts = np.bincount(qo.plan_outcomes(problem),
                             minlength=len(qo.OUTCOMES)) / 2 ** row.x_bits
        np.testing.assert_allclose(
            [getattr(row, column) for column in OUTCOME_COLUMNS], counts,
            atol=1e-12)


def test_point_masses_land_in_their_outcome():
    problem = qc.build_problem('testcases/tiny_a.flow', 1)
    labels = qo.plan_outcomes(problem)

    def basis_state(pattern):
        index = 0
        for e, qubit in enumerate(problem.x_qubits):
            index |= ((pattern >> e) & 1) << qubit
        probabilities = np.zeros(2 ** problem.num_qubits)
        probabilities[index] = 1
        return probabilities

    all_cut = 2 ** len(problem.x_qubits) - 1
    optimal = int(np.flatnonzero(labels == qo.OUTCOMES.index('optimal'))[0])
    for pattern, outcome in ((0, 'empty'), (all_cut, 'over_budget'),
                             (optimal, 'optimal')):
        split = qo.outcome_probabilities(problem, basis_state(pattern), labels)
        assert split[f'p_{outcome}'] == 1, (pattern, outcome)
    # The optimal plan really is optimal and within budget.
    edges = problem.interdictable_edges
    cut = [edges[e] for e in range(len(edges)) if (optimal >> e) & 1]
    assert 0 < len(cut) <= problem.B
    assert qo.rb.max_flow(problem.di_g, problem.src_node, problem.tgt_node,
                          cut) == problem.opt_nip


def test_a_tiny_row_resimulates(outcomes):
    require_qiskit()
    from qaoa_qiskit import QiskitBackend
    stored = pandas.read_csv(qo.TINY_CSV, comment='#')
    row = stored[(stored['instance'] == 'testcases/tiny_a.flow')
                 & (stored['B'] == 1) & (stored['reps'] == 2)].iloc[0]
    problem = qc.build_problem(row['instance'], 1)
    split = qo.outcome_probabilities(problem, QiskitBackend(problem, 2)
                                     .probabilities(json.loads(row['theta'])))
    saved = outcomes[(outcomes['case'] == 'tiny_a B=1')
                     & (outcomes['method'] == 'qaoa')
                     & (outcomes['reps'] == 2)].iloc[0]
    for column in OUTCOME_COLUMNS:
        assert split[column] == pytest.approx(saved[column], abs=1e-9)


def test_a_warm_start_row_resimulates(outcomes):
    """The RWS rows need the rebuilt problem and its warm-start angles."""
    require_qiskit()
    from qaoa_qiskit import QiskitBackend
    from qaoa_warm_start import regularized_warm_start
    problems, strength, seed = qo.transfer_problems()
    problem = problems['test-000']
    warm = regularized_warm_start(problem, strength, seed)
    stored = pandas.read_csv(qo.TRANSFER_CSV, comment='#')
    row = stored[(stored['group'] == 'test-000') & (stored['method'] == 'rws')
                 & (stored['reps'] == 1)].iloc[0]
    split = qo.outcome_probabilities(
        problem, QiskitBackend(problem, 1, warm_start_angles=warm)
        .probabilities(json.loads(row['theta'])))
    saved = outcomes[(outcomes['case'] == 'test-000')
                     & (outcomes['method'] == 'rws')
                     & (outcomes['reps'] == 1)].iloc[0]
    for column in OUTCOME_COLUMNS:
        assert split[column] == pytest.approx(saved[column], abs=1e-9)

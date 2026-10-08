"""
Tests for instances.py: the mid-size instance generator and the committed
testcases/mid/ files and results/mid_instances.csv manifest (issues #1,
#17).

The manifest's optima came from Gurobi; here they are checked without it,
by brute force for up to two cuts and by recomputing the max flow each
recorded optimal cut leaves.
"""
import filecmp
import itertools
import json
import os

import networkx as nx
import pandas
import pytest

import instances
import network_interdiction as ni
import run_benchmark as rb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, 'results', 'mid_instances.csv')

# The QUBO sizes the catalog was chosen for (instances.CATALOG's comment).
CATALOG_QUBITS = [39, 90, 116, 40, 78, 149, 40, 81, 142]


@pytest.fixture(scope='module')
def manifest():
    return pandas.read_csv(MANIFEST, comment='#')


@pytest.mark.parametrize('entry', instances.CATALOG)
def test_instances_are_valid_flow_networks(entry):
    family, n, parameter, seed = entry
    di_g, src, tgt = instances.make_instance(family, n, parameter, seed)
    undirected = di_g.to_undirected()
    assert di_g.number_of_nodes() == n
    assert nx.is_strongly_connected(di_g)
    assert all(15 <= w <= 99 for _, _, w in di_g.edges(data='weight'))
    if family == 'planar':
        assert nx.check_planarity(undirected)[0]
    if family == 'knn':
        assert min(dict(undirected.degree()).values()) >= parameter
    # The source and sink have the most arc-disjoint paths of any pair.
    best = max(nx.edge_connectivity(di_g, u, v)
               for u, v in itertools.permutations(di_g.nodes(), 2))
    assert nx.edge_connectivity(di_g, src, tgt) == best


def test_instances_are_reproducible_and_seeded():
    first = instances.make_instance('erdos', 8, 0.45, 0)
    again = instances.make_instance('erdos', 8, 0.45, 0)
    other = instances.make_instance('erdos', 8, 0.45, 1)
    assert sorted(first[0].edges(data='weight')) == \
        sorted(again[0].edges(data='weight'))
    assert first[1:] == again[1:]
    assert sorted(first[0].edges(data='weight')) != \
        sorted(other[0].edges(data='weight'))


def test_catalog_spans_the_sizes_it_was_chosen_for():
    sizes = [instances.qubits(instances.make_instance(*entry)[0])
             for entry in instances.CATALOG]
    assert sizes == CATALOG_QUBITS
    assert max(sizes) <= instances.IBM_QUBITS


@pytest.mark.parametrize('curve, expected', [
    # B = 1: window (6, 10); B = 2: (4, 6); f(3) = 0 ends the scan.
    ([20, 10, 4, 0], [1, 2]),
    # B = 1: both ends 5, a tie, so no lambda works; f(2) = 0.
    ([10, 5, 0], []),
    # B = 1: lo = max(4, 3.5) = 4 > hi = 3, B above the convex envelope.
    ([10, 7, 3, 0], [2]),
])
def test_interesting_budgets_need_a_window_and_flow_left(curve, expected):
    assert instances.interesting_budgets(curve) == expected


# --- the committed files ----------------------------------------------------------

def test_committed_flow_files_regenerate_byte_for_byte(tmp_path):
    for family, n, parameter, seed in instances.CATALOG:
        name = instances.instance_name(family, n, parameter, seed)
        regenerated = tmp_path / name
        instances.write_flow_file(str(regenerated),
                                  *instances.make_instance(family, n, parameter, seed))
        assert filecmp.cmp(regenerated, os.path.join(REPO_ROOT, 'testcases',
                                                     'mid', name), shallow=False)


def test_manifest_covers_every_instance_with_its_sizes(manifest):
    names = {instances.instance_name(*entry) for entry in instances.CATALOG}
    assert {os.path.basename(path) for path in manifest['instance']} == names
    for _, row in manifest.iterrows():
        di_g, src, tgt = ni.read_flow_file(os.path.join(REPO_ROOT, row['instance']))
        assert (row['V'], row['E']) == (di_g.number_of_nodes(), di_g.number_of_edges())
        assert row['qubits'] == instances.qubits(di_g) <= instances.IBM_QUBITS
        assert row['maxflow'] == rb.max_flow(di_g, src, tgt)


def test_manifest_optima_check_out_without_gurobi(manifest):
    for _, row in manifest.iterrows():
        di_g, src, tgt = ni.read_flow_file(os.path.join(REPO_ROOT, row['instance']))
        curve = json.loads(row['f_curve'])
        B = row['B']
        # Brute force for the first two budgets.
        for k in range(min(3, len(curve))):
            assert curve[k] == rb.brute_force_nip_optimum(di_g, src, tgt, k)[0]
        # The recorded optimal cut is within budget and leaves the optimum.
        cut = [tuple(arc) for arc in json.loads(row['optimal_cut'])]
        assert len(cut) <= B
        assert rb.max_flow(di_g, src, tgt, cut) == row['Opt_NIP'] == curve[B]
        assert row['cut_touches_st'] == any(src in arc or tgt in arc for arc in cut)
        assert row['Greedy'] == rb.greedy_from_scratch(di_g, src, tgt, B)
        # The window and its midpoint follow from the curve.
        lo, hi = rb.lambda_window(curve, B)
        assert (row['lambda_lo'], row['lambda_hi']) == (lo, hi)
        assert row['lambda'] == rb.choose_lambda(lo, hi)
        assert B in instances.interesting_budgets(curve, max_budget=B)

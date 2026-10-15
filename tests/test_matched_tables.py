import pandas as pd
import pytest

import network_interdiction_plot as plot


def test_modern_raw_zero_flow_is_a_feasible_optimum_not_missing():
    frame=pd.DataFrame([dict(V=4,E=4,B=1,qubits=14,Opt_NIP=0,Opt_QUBO=-10,maxflow=5,
                             SA_NIP=0,SA_NIP_Greedy=0,SA_QUBO=-10,SA_raw_feasible_count=1)])
    labels,flows,_=plot.compute_series(frame,['SA'])
    assert flows['SA'][0]==1
    assert labels==['14 (1)']
    # Legacy CSVs retain the zero-sentinel convention.
    _,legacy,_=plot.compute_series(frame.drop(columns=['SA_raw_feasible_count']),['SA'])
    assert legacy['SA'][0]==0


def test_missing_decomposition_result_is_not_a_zero_flow_optimum():
    frame=pd.DataFrame([dict(V=4,E=4,B=1,Opt_NIP=1,Opt_QUBO=-10,maxflow=5,
                             Decomp_mincut_sequential=2,Decomp_multiplier_iterated=float('nan'))])
    labels=plot.available_baselines(frame)
    assert 'mincut-sequential' in labels
    assert 'multiplier-iterated' not in labels
    _,flows,_=plot.compute_series(frame,[],labels)
    assert flows['mincut-sequential'][0]==pytest.approx(.75)

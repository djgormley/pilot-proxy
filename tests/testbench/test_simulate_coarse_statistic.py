from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import math
import numpy as np
import pytest
from scipy.stats import f,ncf

TOOLS=Path(__file__).resolve().parents[2]/"tools"


def load():
    spec=spec_from_file_location("coarse_gnu",TOOLS/"simulate_coarse_statistic.py")
    module=module_from_spec(spec);spec.loader.exec_module(module);return module


def test_float64_accumulation_preserves_terms_lost_by_float32_sequential_sum():
    m=load();a=np.ones(m.P,dtype=np.float32);a[0]=2**24
    b=np.ones(m.P,dtype=np.float32);c=np.full(m.P,2,dtype=np.float32)
    actual=m.aggregate_float64([a,b,c])
    expected_a=2**24+m.P-1
    assert actual[0,0]==expected_a
    assert actual[0,1]==m.P and actual[0,2]==2*m.P
    assert actual[0,3]==2*expected_a/(3*m.P)
    assert float(np.sum(a,dtype=np.float32))!=expected_a


def test_power_sums_are_formed_before_ratio_and_match_fsum():
    m=load();rng=np.random.default_rng(321)
    data=[rng.uniform(0,5,size=m.P*2).astype(np.float32) for _ in range(3)]
    actual=m.aggregate_float64(data)
    for row in range(2):
        expected=[math.fsum(v[row*m.P:(row+1)*m.P]) for v in data]
        assert np.all(np.abs(actual[row,:3]-expected)<=4*np.spacing(expected))
        assert actual[row,3]==pytest.approx(2*expected[0]/(expected[1]+expected[2]),rel=1e-15)


@pytest.mark.parametrize("gamma",[0,.02])
def test_closed_moments_and_degrees_match_exact_f_law(gamma):
    m=load();moments=m.theoretical_moments(gamma)
    df1,df2=2*m.P,4*m.P
    assert (df1,df2)==(524288,1048576)
    law=f(df1,df2) if gamma==0 else ncf(df1,df2,2*m.P*gamma)
    assert moments["Q_mean"]==pytest.approx(law.mean(),rel=2e-15)
    assert moments["Q_variance"]==pytest.approx(law.var(),rel=1e-9)
    assert moments["noncentrality"]==2*m.P*gamma


def test_unique_explicit_state_branch_seeds():
    m=load();seeds=[seed for value in m.STATES.values() for seed in value["seeds"]]
    assert len(seeds)==len(set(seeds))==6
    assert all(0<seed<2**31 for seed in seeds)

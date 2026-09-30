import numpy as np
import pytest

from pilot_proxy.capture.lag_moments import frame_moments, lag_moments, summarize_lags


def direct(x, lags, support):
    records = []
    for lag in lags:
        row = []
        for col in range(x.shape[1]):
            pairs = [(x[t+lag,col], x[t,col]) for t in range(len(x)-lag)
                     if support[t+lag,col] and support[t,col]
                     and np.isfinite(x[t+lag,col]) and np.isfinite(x[t,col])]
            row.append([len(pairs), sum(a for a,b in pairs), sum(b for a,b in pairs),
                        sum(abs(a)**2 for a,b in pairs), sum(abs(b)**2 for a,b in pairs),
                        sum(a*b.conjugate() for a,b in pairs)])
        records.append(row)
    return np.array(records)


def test_exact_direct_support_and_orientation():
    x = np.array([[1+2j,3j],[2-1j,4],[5j,6j],[7,2-3j],[2+4j,1]],complex)
    mask = np.array([[1,1],[1,0],[0,1],[1,1],[1,1]],bool)
    x[3,1] = np.nan
    actual = lag_moments(x,[0,1,3],mask)
    expected = direct(x,[0,1,3],mask)
    for j,key in enumerate(actual):
        np.testing.assert_allclose(actual[key],expected[:,:,j])


def test_positive_lag_is_late_conjugate_early():
    x=np.exp(1j*.37*np.arange(100))[:,None]
    result=summarize_lags(lag_moments(x,[0,1,8]))
    np.testing.assert_allclose(result['normalized_raw'][:,0],np.exp(1j*.37*np.array([0,1,8])))


def test_centering_removes_constant_offset():
    x=np.array([1+2j,4-1j,3+3j,-1j,8],complex)[:,None]
    one=summarize_lags(lag_moments(x,[0,1,3]))
    two=summarize_lags(lag_moments(x+10-7j,[0,1,3]))
    np.testing.assert_allclose(one['centered'],two['centered'],atol=1e-12)
    np.testing.assert_allclose(one['normalized_centered'],two['normalized_centered'],atol=1e-12)
    assert not np.allclose(one['raw'],two['raw'])


def test_cross_boundary_and_tail_preserved():
    x=np.arange(11,dtype=float)[:,None].astype(complex)
    m=lag_moments(x,[0,1,4])
    assert m['count'][:,0].tolist()==[11,10,7]
    assert m['cross'][1,0]==sum(t*(t-1) for t in range(1,11))
    f=frame_moments(x,4)
    assert f['blocks'].tolist()==[[0,4],[4,8],[8,11]]
    assert f['count'][:,0].tolist()==[4,4,3]
    assert f['complete'].tolist()==[True,True,False]
    assert f['sum'].sum()==x.sum()


def test_undefined_zero_constant_and_empty_support():
    x=np.column_stack([np.zeros(8),np.ones(8),np.arange(8)]).astype(complex)
    mask=np.ones_like(x,dtype=bool); mask[:,2]=False
    s=summarize_lags(lag_moments(x,[0,3],mask))
    assert np.isnan(s['normalized_centered']).all()
    assert np.isnan(s['normalized_raw'][:,[0,2]]).all()
    np.testing.assert_array_equal(s['normalized_raw'][:,1],1)


@pytest.mark.parametrize('lags', [[-1],[4],[1.,2.],[],[1,1]])
def test_bad_lags(lags):
    with pytest.raises(ValueError): lag_moments(np.ones((4,1),complex),lags)


@pytest.mark.parametrize('n',[0,-1,True,1.5])
def test_bad_frame_sizes(n):
    with pytest.raises(ValueError): frame_moments(np.ones((4,1),complex),n)


def test_no_mask_time_compression():
    x=np.arange(5)[:,None].astype(complex); mask=np.ones_like(x,bool); mask[2]=False
    m=lag_moments(x,[1],mask)
    assert m['count'][0,0]==2
    assert m['cross'][0,0]==12

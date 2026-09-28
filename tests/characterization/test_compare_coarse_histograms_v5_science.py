"""Scientific regression boundaries for the frozen descriptive histogram analysis.

These fixtures are synthetic and do not refit or certify the CANFAR populations.
"""
import numpy as np
import pytest
from scipy import stats

from pilot_proxy.characterization import histograms as m



@pytest.mark.parametrize("gamma",[1e-7,.02,1.,100.])
def test_known_noncentral_medians_recover_their_nonnegative_signal_parameter(gamma):
    # Independently choose the model then ask the inverse path to recover its
    # parameter. This catches wrong aggregate degrees or a factor-of-two lambda.
    expected=stats.ncf(524288,1048576,524288*gamma)
    median=float(expected.ppf(.5))
    law,recovered,status,error=m.median_matched(median)
    assert status=="median_matched"
    assert recovered==pytest.approx(gamma,rel=2e-6,abs=1e-10)
    assert abs(error)<1e-7
    assert float(law.cdf(median))==pytest.approx(.5,abs=1e-7)


def test_below_central_median_is_unmatched_not_a_fitted_zero_signal():
    central_median=float(stats.f(524288,1048576).ppf(.5))
    q=np.linspace(central_median-.008,central_median-.004,61)
    result=m.measure(q)
    assert result["model_status"]=="below_null_median_no_nonnegative_match"
    assert result["median_equivalent_gamma"] is None
    assert result["median_equivalent_lambda"] is None
    assert result["model_gamma_for_reference"]==0
    assert result["model_lambda_for_reference"]==0
    assert result["median_cdf_error"]<0
    assert result["model_std"]==pytest.approx(stats.f(524288,1048576).std(),rel=1e-12)
    assert result["supported_30_frames"] is True


def test_exact_central_median_is_a_legitimate_zero_signal_match():
    result=m.measure(np.full(31,m.NULL_MEDIAN))
    assert result["model_status"]=="median_matched"
    assert result["median_equivalent_gamma"]==0
    assert result["median_equivalent_lambda"]==0
    assert result["median_cdf_error"]==0
    assert result["central68_width_ratio"]==0


def test_just_below_null_does_not_silently_cross_parameter_boundary():
    median=np.nextafter(m.NULL_MEDIAN,-np.inf)
    _,gamma,status,_=m.median_matched(median)
    assert gamma==0 and status=="below_null_median_no_nonnegative_match"


def test_exact_bin_mass_has_full_normalization_and_survives_upper_tail_cdf_roundoff():
    # F(2,4) has exact survival S(r)=4/(r+2)^2. At r>=1e10, CDFs
    # lose the interval probabilities through roundoff, though the masses remain positive.
    edges=np.array([0.,.1,1.,10.,1000.,1e8,1e10,1e12,np.inf])
    survival=4/(edges+2)**2
    expected=survival[:-1]-survival[1:]
    actual=m.bin_masses(stats.f(2,4),edges)
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=1e-17)
    assert actual.sum()==pytest.approx(1.,abs=2e-15)
    assert actual[-2]>0 and actual[-1]>0
    cdf_masses=np.diff(stats.f.cdf(edges,2,4))[-2:]
    assert not np.allclose(cdf_masses,expected[-2:],rtol=1e-3,atol=0)
    assert actual[-2]==pytest.approx(expected[-2],rel=2e-13,abs=0)
    assert actual[-1]==pytest.approx(expected[-1],rel=2e-13,abs=0)


@pytest.mark.parametrize("gamma",[0.,.02,25.])
def test_actual_coarse_law_bin_masses_preserve_truncated_probability(gamma):
    law=stats.f(524288,1048576) if gamma==0 else stats.ncf(524288,1048576,524288*gamma)
    # Equal-probability bins give an independent inverse-CDF check on
    # exact displayed-bin masses; their total deliberately remains0.98.
    probabilities=np.linspace(.01,.99,41)
    edges=law.ppf(probabilities)
    mass=m.bin_masses(law,edges)
    np.testing.assert_allclose(mass,np.diff(probabilities),rtol=1e-5,atol=2e-8)
    assert mass.sum()==pytest.approx(.98,abs=1e-7)


def test_unequal_group_population_variance_and_pooled_degrees_are_exact():
    q=np.array([1.,3.,4.,4.,10.,20.])
    group=np.array([11,11,22,22,22,33])
    # Group means2,6,20 with counts2,3,1; grand mean7. Within SS=26,
    # between SS=222,total SS=248. Equal weighting of groups is wrong here.
    result=m.variance_parts(q,group)
    assert result["total_population_variance"]==pytest.approx(248/6)
    assert result["within_population_variance"]==pytest.approx(26/6)
    assert result["between_population_variance"]==pytest.approx(222/6)
    assert result["between_variance_fraction"]==pytest.approx(222/248)
    assert result["within_group_pooled_sd"]==pytest.approx(np.sqrt(26/3))
    assert result["iid_equal_variance_expected_between_fraction"]==pytest.approx(2/5)
    shifted=m.variance_parts(q+1e8,group)
    assert shifted["total_population_variance"]==result["total_population_variance"]
    assert shifted["within_group_pooled_sd"]==result["within_group_pooled_sd"]


def test_singleton_groups_do_not_claim_an_estimable_within_group_sd():
    result=m.variance_parts(np.array([1.,2.,3.]),np.array([10,20,30]))
    assert result["within_group_pooled_sd"] is None
    assert result["within_population_variance"]==0
    assert result["between_variance_fraction"]==1


def test_nonfinite_or_empty_populations_are_refused():
    for q in (np.array([]),np.array([1.,np.nan]),np.array([1.,np.inf])):
        with pytest.raises(ValueError,match="nonempty finite"):
            m.measure(q)


def test_render_unmatched_current_era_uses_boundary_label_and_all_frame_normalization(tmp_path):
    # A below-null current era with one zero: zero remains in the denominator,
    # is counted explicitly outside the log panel and cannot become a matched fit.
    q=np.r_[np.linspace(.97,.99,60),0.]
    row=m.measure(q)
    row.update(channel=14,era_id=1,is_current=True,first_month="2020-01",last_month="2020-02",state="synthetic test fixture",n_acquisitions=7)
    class Recorder:
        figure=None
        def savefig(self,figure,**kwargs):self.figure=figure
    atlas=Recorder();output=tmp_path/"below-null-current"
    result=m.channel_page(14,{}, {"Q":q,"current_histogram_eligible":np.ones(q.size,bool)},[row],[],output,atlas)
    labels=[text.get_text() for legend in atlas.figure.legends for text in legend.get_texts()]
    assert labels==["CANFAR frames","Central boundary reference (unmatched)","Noise-only model"]
    assert output.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    assert output.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert result["full_log_nonpositive_frames"]==1
    assert result["full_log_counts"].sum()==60
    assert result["zoom_excluded_frames"]==1
    assert len(result["zoom_segments"])==1 and result["zoom_eta"] is None
    seg=result["zoom_segments"][0]
    area=np.sum(seg["observed_density"]*np.diff(seg["edges"]))
    assert area==pytest.approx(60/61)
    assert row["median_equivalent_gamma"] is None
    # Boundary curves are the same central law; labels retain their different roles.
    np.testing.assert_array_equal(seg["model_bin_mass"],seg["noise_bin_mass"])


def test_zoom_hugs_both_models_and_eta_and_splits_only_distant_features():
    sd=m.NULL_SD
    # near: one segment holding the noise model, the matched model and eta, bins at most half a model sd wide
    (edges,)=m.zoom_segments(1.03,sd,eta=1.015)
    assert edges[0]<m.NULL_MEDIAN-8*m.NULL_SD and edges[-1]>1.03+8*sd and edges[0]<1.015<edges[-1]
    assert np.diff(edges).max()<=.5*sd+1e-15
    # far: the noise model gets its own segment; eta beside the matched model shares the matched segment
    null_seg,model_seg=m.zoom_segments(100.,.14,eta=110.)
    assert null_seg[0]<m.NULL_MEDIAN<null_seg[-1] and null_seg[-1]<2
    assert model_seg[0]<100-8*.14 and model_seg[-1]>110 and model_seg[0]>50
    # without eta the rule still holds both models
    assert len(m.zoom_segments(1.2,.003))==1 and len(m.zoom_segments(25.,.04))==2

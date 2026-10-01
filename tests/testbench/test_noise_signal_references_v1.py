from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(name):
    spec = spec_from_file_location(name, TOOLS / f"{name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import json
import pytest


def core():
    return load_tool("analyze_noise_signal_references_v1")


def test_fractional_absolute_frequency_still_gives_orthogonal_equal_norm_weights():
    m=core();w=m.weights()
    np.testing.assert_allclose(w.conj()@w.T,128*np.eye(3),atol=4e-13)
    assert m.geometry()["normalized_gram_max_off_diagonal_magnitude"]<4e-15
    assert m.geometry()["central_F_mean"]==2
    assert m.geometry()["null_ratio_of_expected_powers"]==pytest.approx(1)


def test_positive_tone_projection_sign_and_single_row_matches_cpu_reference():
    m=core()
    from pilot_proxy.detector_reference import coarse_power_ratio_cpu_reference
    rng=np.random.default_rng(617)
    iq=rng.normal(size=128*7)+1j*rng.normal(size=128*7)
    z=m.projections(iq);r,status=m.ratios(z)
    assert status["positive_denominator"]==7
    for index in range(7):
        expected,_=coarse_power_ratio_cpu_reference(iq[index*128:(index+1)*128][None,:],m.weights())
        assert r[index]==pytest.approx(expected)
    tone=np.exp(2j*np.pi*m.TARGET*np.arange(128*7)/m.FS)
    p=np.abs(m.projections(tone))**2
    np.testing.assert_allclose(p[:,0],128**2,rtol=1e-14)
    assert np.max(p[:,1:])<1e-24


def test_zero_denominators_are_not_converted_to_valid_zero_ratios():
    m=core();values,status=m.ratios(np.array([[0,0,0],[1,0,0],[0,1,1]],complex))
    assert np.isnan(values[0]) and np.isposinf(values[1]) and values[2]==0
    assert status=={"positive_denominator":1,"infinite_ratio_positive_numerator_zero_denominator":1,"undefined_both_zero":1}
    with pytest.raises(ValueError,match="finite"):
        m.ecdf(values,np.array([1.]))


@pytest.mark.parametrize("lam",[0.,2.,20.])
def test_matched_raw_gaussian_simulation_matches_independent_analytic_cdf(lam):
    m=core();z=m.simulation(40000,lam,5467+int(lam));values,status=m.ratios(z)
    assert status["positive_denominator"]==40000
    q=np.geomspace(.01,500,1000);d=q+2
    exact=np.exp(-lam/d)*(q*(q+4)/d**2+lam*q*q/d**3)
    assert np.max(np.abs(m.ecdf(values,q)-exact))<.014
    if lam==0:
        p=np.abs(z)**2
        np.testing.assert_allclose(p.mean(axis=0),128,rtol=.025)
        # Realizations are not forcibly orthogonalized or normalized to exact energy.
        assert np.std(p[:,0])>100


def test_power_calibration_uses_separate_txzero_power_not_mean_ratio():
    m=core();captures=[]
    for _ in range(2):
        for mode,p in (("noise",[1,1,1]),("txzero",[2,2,2]),("tone",[12,2,2])):
            captures.append({"entry":{"mode":mode},"diagnostics":{"mean_branch_power":p}})
    result=m.power_calibration(captures)
    assert result["lambda"]==10
    assert result["noise_baseline_lambda_diagnostic_only"]==22
    assert result["tone_over_txzero_branch_power"]==[6,1,1]


def manifest_value(m,tmp_path):
    return {"schema":"noise-signal-reference-inputs-v1","analysis_source_sha256":m.sha(m.__file__),
            "config":m.CONFIG,"assigned_utc":"2026-09-09T00:00:00+00:00",
            "captures":[{"id":f"capture-{index}-{mode}","mode":mode,
                "role":"calibration" if index<2 else "comparison","directory":str(tmp_path/f"capture-{index}-{mode}")}
                for index in range(5) for mode in m.MODES]}


@pytest.mark.parametrize("change",["duplicate","role","source"])
def test_manifest_refuses_reused_capture_or_retrospective_reassignment(tmp_path,change):
    m=core();value=manifest_value(m,tmp_path)
    if change=="duplicate":value["captures"][1]["directory"]=value["captures"][0]["directory"]
    if change=="role":value["captures"][0]["role"]="comparison"
    if change=="source":value["analysis_source_sha256"]="0"*64
    path=tmp_path/"manifest.json";path.write_text(json.dumps(value))
    with pytest.raises(ValueError):m.load_manifest(path)


def test_process_calibration_never_opens_comparison_files(tmp_path,monkeypatch):
    m=core();manifest=manifest_value(m,tmp_path);opened=[]
    def read(entry):
        assert entry["role"]=="calibration"
        opened.append(entry["id"])
        directory=Path(entry["directory"]);directory.mkdir()
        (directory/"launch.json").write_text(json.dumps({"started_utc":"2026-09-09T01:00:00+00:00"}))
        iq=np.random.default_rng(22).normal(size=128*100)+1j*np.random.default_rng(23).normal(size=128*100)
        return iq.astype(np.complex64),{}, {"completed_utc":"2026-09-09T01:01:00+00:00"}
    monkeypatch.setattr(m,"read_capture",read)
    output=tmp_path/"output";output.mkdir()
    captures,_=m.process_role(manifest,"calibration",output)
    assert len(opened)==6 and len(captures)==6
    assert m.power_calibration(captures)["lambda"]==0

from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import os
import subprocess

import numpy as np
import pytest

TOOLS=Path(__file__).resolve().parents[2]/"tools"


def load(name):
    spec=spec_from_file_location(name,TOOLS/f"{name}.py");module=module_from_spec(spec);spec.loader.exec_module(module)
    return module


def command(*args):
    env={**os.environ,"PYTHONNOUSERSITE":"1"}
    return subprocess.run(["/usr/bin/python3",str(TOOLS/"generate_gnuradio_reference.py"),*map(str,args)],env=env,text=True,capture_output=True,check=True)


def test_actual_gnuradio_reproducibility_gaussian_normalization_and_source_sum(tmp_path):
    probe=subprocess.run(["/usr/bin/python3","-c","from gnuradio import gr; assert gr.version()=='3.10.9.2'"],capture_output=True,env={**os.environ,"PYTHONNOUSERSITE":"1"})
    if probe.returncode:pytest.skip("system GNU Radio3.10.9.2 unavailable")
    for name,lam,seed in (("noise",0,5161),("repeat",0,5161),("tone",20,5162)):
        directory=tmp_path/name
        command("prepare","--output",directory,"--mode","tone" if lam else "noise","--seed",seed,"--lambda",lam,"--samples",128000)
        command("generate","--output",directory)
    first=np.fromfile(tmp_path/"noise"/"iq.cfile",dtype="<c8")
    second=np.fromfile(tmp_path/"repeat"/"iq.cfile",dtype="<c8")
    assert np.array_equal(first,second)
    assert np.mean(np.abs(first.astype(np.complex128))**2)==pytest.approx(1,rel=.02)
    assert np.var(first.real)==pytest.approx(.5,rel=.025)
    assert np.var(first.imag)==pytest.approx(.5,rel=.025)
    noise=np.fromfile(tmp_path/"tone"/"noise.cfile",dtype="<c8")
    signal=np.fromfile(tmp_path/"tone"/"signal.cfile",dtype="<c8")
    summed=np.fromfile(tmp_path/"tone"/"iq.cfile",dtype="<c8")
    assert np.array_equal(summed,noise+signal)
    assert not np.array_equal(first,noise)
    assert np.mean(np.abs(signal.astype(np.complex128))**2)==pytest.approx(20/256,rel=2e-5)
    core=load("analyze_noise_signal_references")
    signal_projection=core.projections(signal)
    measured_lambda=2*np.mean(np.abs(signal_projection[:,0])**2)/128
    assert measured_lambda==pytest.approx(20,rel=3e-5)
    assert np.mean(np.abs(signal_projection[:,1:])**2)<1e-7
    # Exclusive receipts prevent regeneration from overwriting a recorded run.
    with pytest.raises(subprocess.CalledProcessError):command("generate","--output",tmp_path/"tone")


def test_density_per_dex_uses_all_samples_and_refuses_zero_or_invalid():
    module=load("plot_gnuradio_sdr_references")
    density,outside=module.density_per_dex(np.array([.001,.1,1.,10.,1000.]),np.array([-1.,0.,1.]))
    assert outside==2
    np.testing.assert_allclose(density,[.2,.4])
    for bad in ([0,1],[np.inf,1],[np.nan,1]):
        with pytest.raises(ValueError):module.density_per_dex(np.array(bad),np.array([-1.,0.,1.]))


def test_per_dex_jacobian_integrates_to_covered_cdf_probability():
    module=load("plot_gnuradio_sdr_references")
    x=np.linspace(-4,5,50000);r=10**x
    for lam in (0.,20.):
        model=module.core.distribution(lam)
        mass=np.trapezoid(np.log(10)*r*model.pdf(r),x)
        assert mass==pytest.approx(model.cdf(r[-1])-model.cdf(r[0]),rel=2e-8)

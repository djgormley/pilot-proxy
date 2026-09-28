"""Contamination residuals: the second half of the tolerance cost.

Two things this suite has to establish. First, that adding the term changes
nothing when it is absent; every published number in ``out/`` predates it and
must survive byte-for-byte. Second, that with it present the forecast stops
being monotone in the masked fraction, which is the whole point: only then is a
detector threshold something you can optimise rather than assume.
"""
import numpy as np
import pytest

from pilot_proxy.characterization import residual_chain as residual
from pilot_proxy.products.reader import FRAME_SECONDS as CHIME_FRAME_SECONDS


# ----------------------------------------------------------------------
# Backward compatibility: no residuals -> identical behavior
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# The coupling itself
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# The budget
# ----------------------------------------------------------------------

def test_n_coh_from_correlation_time():
    n = residual.coherent_frames(3600.0)
    assert n == pytest.approx(3600.0 / CHIME_FRAME_SECONDS, rel=1e-9)
    assert 10 * np.log10(n) == pytest.approx(49.3, abs=0.1)
    # never below 1: a residual cannot average down faster than thermal noise
    assert residual.coherent_frames(1e-6) == 1.0
    # never above one sidereal day: longer is m = 0 and already removed
    assert residual.coherent_frames(1e9) == pytest.approx(
        residual.MAX_TAU_C_SECONDS / CHIME_FRAME_SECONDS)
    with pytest.raises(ValueError):
        residual.coherent_frames(1.0, frame_seconds=0.0)


# ----------------------------------------------------------------------
# The correlation-time estimator
# ----------------------------------------------------------------------

SID = 86164.0905


def _write_product(path, shelf_db, unit_of_frame, unit_t0, channel=35,
                   rejected=None):
    """Minimal survey product with the keys the residual module reads."""
    n = shelf_db.size
    if rejected is None:
        rejected = (shelf_db > np.percentile(shelf_db, 10)).astype(np.uint8)
    np.savez(
        path,
        valid=np.ones((n, 1), dtype=np.uint8),
        reject_mask=rejected.reshape(n, 1).astype(np.uint8),
        snr_shelf_db=shelf_db.reshape(n, 1),
        frame_unit_index=unit_of_frame.astype(np.int32),
        unit_time0_ctime=unit_t0,
        physical_channel=np.array([channel], dtype=np.int32),
        freq_id=np.array([521], dtype=np.int64),
        chime_frequency_hz=np.array([596.48e6]),
    )
    return path


def _epoch_product(tmp_path, name, on_first: bool):
    """One frame per unit: an on epoch at -10 dB and an off epoch at -40 dB.

    ``on_first=True`` is a sign-off channel (on through 2020-12, off from
    2021-01); ``on_first=False`` is the time-mirrored sign-on channel.
    """
    import datetime as _dt
    rng = np.random.default_rng(7)
    n = 300
    on_shelf = -10.0 + 0.1 * rng.standard_normal(n)
    off_shelf = -40.0 + 0.1 * rng.standard_normal(n)
    t_on = _dt.datetime(2020, 6, 1, tzinfo=_dt.timezone.utc).timestamp() \
        + np.arange(n) * 3600.0
    t_off = _dt.datetime(2021, 6, 1, tzinfo=_dt.timezone.utc).timestamp() \
        + np.arange(n) * 3600.0
    if on_first:
        shelf = np.concatenate([on_shelf, off_shelf])
        t0 = np.concatenate([t_on, t_off])
        rejected = np.r_[np.ones(n), np.zeros(n)].astype(np.uint8)
    else:
        shelf = np.concatenate([off_shelf, on_shelf])
        t0 = np.concatenate([t_off - 2 * 366 * 86400.0, t_on])
        rejected = np.r_[np.zeros(n), np.ones(n)].astype(np.uint8)
    return _write_product(tmp_path / name, shelf,
                          np.arange(2 * n), t0, rejected=rejected)


def test_off_epoch_specs_are_mutually_exclusive(tmp_path):
    p = _epoch_product(tmp_path, "signoff.npz", on_first=True)
    with pytest.raises(ValueError, match="at most one"):
        residual.shelf_statistics(p, off_through="2020-12", off_from="2021-01")
    with pytest.raises(ValueError, match="at most one"):
        residual.correlation_time(p, off_through="2020-12", off_from="2021-01")


def _stationary_product(tmp_path, tau_true, n_days=400, per_day=6,
                        frames_per_unit=6, seed=1):
    """Shelf = constant + slow day offset + AR(1) intra-day term + noise."""
    rng = np.random.default_rng(seed)
    shelf, uof, t0 = [], [], []
    u = 0
    for d in range(n_days):
        day_start = d * SID + 3000.0
        day_off = 0.02 * rng.standard_normal()
        # acquisitions spread over ~5 h, AR(1) correlated with time constant tau
        times = np.sort(rng.uniform(0.0, 5 * 3600.0, per_day))
        x, prev, prev_t = [], rng.standard_normal(), times[0]
        for tt in times:
            a = np.exp(-(tt - prev_t) / tau_true)
            prev = a * prev + np.sqrt(max(1 - a * a, 0.0)) * rng.standard_normal()
            prev_t = tt
            x.append(prev)
        for tt, xi in zip(times, x):
            t0.append(day_start + tt)
            lin = 1.0 * (1.0 + day_off + 0.06 * xi)
            frames = lin * (1.0 + 0.004 * rng.standard_normal(frames_per_unit))
            shelf.append(10.0 * np.log10(np.maximum(frames, 1e-12)))
            uof.append(np.full(frames_per_unit, u))
            u += 1
    return _write_product(tmp_path / "stationary.npz",
                          np.concatenate(shelf), np.concatenate(uof),
                          np.array(t0))


def _episodic_product(tmp_path, n_days=400, per_day=6, frames_per_unit=6,
                      seed=2):
    """Quiet baseline punctuated by rare strong bursts, ch34/ch36's shape."""
    rng = np.random.default_rng(seed)
    shelf, uof, t0 = [], [], []
    u = 0
    for d in range(n_days):
        day_start = d * SID + 3000.0
        times = np.sort(rng.uniform(0.0, 5 * 3600.0, per_day))
        for tt in times:
            t0.append(day_start + tt)
            lin = 1e-4 * (1.0 + 0.3 * rng.standard_normal())
            if rng.random() < 0.25:                 # burst, broad amplitude
                lin *= 10.0 ** rng.uniform(0.5, 4.5)
            frames = np.abs(lin * (1.0 + 0.1 * rng.standard_normal(frames_per_unit)))
            shelf.append(10.0 * np.log10(np.maximum(frames, 1e-18)))
            uof.append(np.full(frames_per_unit, u))
            u += 1
    return _write_product(tmp_path / "episodic.npz",
                          np.concatenate(shelf), np.concatenate(uof),
                          np.array(t0), channel=34)


def test_correlation_time_recovers_a_known_timescale(tmp_path):
    path = _stationary_product(tmp_path, tau_true=2400.0)
    ct = residual.correlation_time(path, n_boot=60)
    assert ct.is_measured, ct.reason
    assert 1200.0 < ct.tau_c < 4800.0, ct.tau_c
    assert ct.tau_lo <= ct.tau_c <= ct.tau_hi
    assert ct.trim_spread < 2.0 and ct.surviving_spread < 2.0


def test_correlation_time_orders_two_known_timescales(tmp_path):
    slow = residual.correlation_time(
        _stationary_product(tmp_path / "a", tau_true=3600.0, seed=11)
        if (tmp_path / "a").mkdir() or True else None, n_boot=40)
    fast = residual.correlation_time(
        _stationary_product(tmp_path / "b", tau_true=600.0, seed=12)
        if (tmp_path / "b").mkdir() or True else None, n_boot=40)
    assert slow.is_measured and fast.is_measured
    assert slow.tau_c > fast.tau_c


def test_correlation_time_refuses_an_episodic_shelf(tmp_path):
    ct = residual.correlation_time(_episodic_product(tmp_path), n_boot=40)
    assert not ct.is_measured
    assert any(k in ct.reason for k in ("episodic", "artefact", "cut",
                                       "unresolved"))
    # and the refusal hands back the conservative cap rather than a guess
    assert ct.tau_for_budget == residual.MAX_TAU_C_SECONDS
    assert np.isnan(ct.tau_c)


def test_noise_correction_stops_sparse_units_faking_a_short_tau(tmp_path):
    """Unit-mean noise inflates D at every lag; uncorrected it looks fast."""
    path = _stationary_product(tmp_path, tau_true=2400.0, frames_per_unit=2,
                               seed=7)
    ct = residual.correlation_time(path, n_boot=40)
    assert ct.is_measured, ct.reason
    assert ct.tau_c > 900.0, "noise correction failed: tau collapsed"


def test_bootstrap_must_preserve_within_day_ordering(tmp_path):
    """A day-block resample keeps each day's sequence; the interval is finite."""
    ct = residual.correlation_time(
        _stationary_product(tmp_path, tau_true=2400.0), n_boot=80)
    assert np.isfinite(ct.tau_lo) and np.isfinite(ct.tau_hi)
    assert ct.tau_hi > ct.tau_lo


def _fast_stationary_product(tmp_path, seed=21):
    """Stationary shelf that decorrelates faster than the acquisition cadence.

    The two failure modes of the estimator point opposite ways, and this is the
    favourable one: a short tau_c is an upper bound worth ~24 dB against the
    sidereal-day cap, so it must not be discarded like an episodic shelf.
    """
    rng = np.random.default_rng(seed)
    shelf, uof, t0 = [], [], []
    u = 0
    for d in range(400):
        day_start = d * SID + 3000.0
        day_off = 0.02 * rng.standard_normal()
        times = np.sort(rng.uniform(0.0, 5 * 3600.0, 6))
        for tt in times:                       # independent between acquisitions
            t0.append(day_start + tt)
            lin = 1.0 * (1.0 + day_off + 0.06 * rng.standard_normal())
            frames = lin * (1.0 + 0.004 * rng.standard_normal(6))
            shelf.append(10.0 * np.log10(np.maximum(frames, 1e-12)))
            uof.append(np.full(6, u))
            u += 1
    return _write_product(tmp_path / "fast.npz", np.concatenate(shelf),
                          np.concatenate(uof), np.array(t0))


def test_fast_stationary_shelf_is_bounded_not_refused(tmp_path):
    ct = residual.correlation_time(_fast_stationary_product(tmp_path), n_boot=40)
    assert ct.quality == "bounded_above", (ct.quality, ct.reason)
    assert ct.is_usable and not ct.is_measured
    assert ct.tau_c == pytest.approx(residual.STRUCTURE_LAG_EDGES[1])
    # and the bound is worth using: far below the cap it would otherwise get
    assert ct.tau_for_budget < residual.MAX_TAU_C_SECONDS / 100


def test_stationarity_is_checked_before_resolution(tmp_path):
    """An episodic shelf must refuse, never claim a favourable short bound."""
    ct = residual.correlation_time(_episodic_product(tmp_path), n_boot=40)
    assert ct.quality == "refused"
    assert ct.surviving_spread > 2.0 or ct.trim_spread > 2.0
    assert ct.tau_for_budget == residual.MAX_TAU_C_SECONDS


# ----------------------------------------------------------------------
# Is masking worth it?
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# The four-way policy comparison
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# One booking, one floor discipline (the ch35/ch32 reconciliation)
# ----------------------------------------------------------------------

def test_surviving_components_refusal_removes_split_credit(tmp_path):
    """A refused tau_c invalidates the variance split, not just the timescale."""
    p = _episodic_product(tmp_path)
    st = residual.shelf_statistics(p)
    ct = residual.correlation_time(p, n_boot=40)
    assert not ct.is_usable
    cap_n = residual.coherent_frames(residual.MAX_TAU_C_SECONDS)
    assert residual.surviving_components(st, ct) == ((1.0, cap_n),)
    # an explicit timescale narrows the cap but cannot restore the credit
    assert residual.surviving_components(st, ct, tau_intraday=3600.0) \
        == ((1.0, residual.coherent_frames(3600.0)),)


def test_surviving_components_usable_books_the_split(tmp_path):
    p = _stationary_product(tmp_path, tau_true=2400.0)
    st = residual.shelf_statistics(p)
    ct = residual.correlation_time(p, n_boot=60)
    assert ct.is_usable
    comps = residual.surviving_components(st, ct)
    assert len(comps) == 2
    assert comps[0][0] == pytest.approx(st.intraday_fraction)
    assert comps[0][1] == pytest.approx(
        residual.coherent_frames(ct.tau_c))
    assert comps[1] == (pytest.approx(st.fast_fraction), 1.0)

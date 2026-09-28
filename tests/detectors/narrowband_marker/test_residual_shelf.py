"""Contamination residuals: the second half of the tolerance cost.

Two things this suite has to establish. First, that adding the term changes
nothing when it is absent; every published number in ``out/`` predates it and
must survive byte-for-byte. Second, that with it present the forecast stops
being monotone in the masked fraction, which is the whole point: only then is a
detector threshold something you can optimise rather than assume.
"""
import numpy as np
import pytest

from pilot_proxy.detectors.narrowband_marker import shelf as residual


# ----------------------------------------------------------------------
# Backward compatibility: no residuals -> identical behavior
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# The coupling itself
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# The budget
# ----------------------------------------------------------------------

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


def test_off_from_mirrors_off_through(tmp_path):
    """A sign-off channel calibrates exactly like its time-mirrored sign-on."""
    signoff = _epoch_product(tmp_path, "signoff.npz", on_first=True)
    signon = _epoch_product(tmp_path, "signon.npz", on_first=False)
    a = residual.shelf_statistics(signoff, off_from="2021-01")
    b = residual.shelf_statistics(signon, off_through="2020-05")
    assert a.floor_db == pytest.approx(b.floor_db, abs=0.05)
    assert a.floor_db == pytest.approx(-40.0, abs=0.5)
    assert a.on_shelf_db == pytest.approx(-10.0, abs=0.5)
    # without the epoch, the off sample is empty (every on frame is rejected,
    # every off frame kept) and the floor comes from the kept frames instead
    c = residual.shelf_statistics(signoff)
    assert c.floor_db == pytest.approx(-40.0, abs=0.5)


# ----------------------------------------------------------------------
# Is masking worth it?
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# The four-way policy comparison
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# One booking, one floor discipline (the ch35/ch32 reconciliation)
# ----------------------------------------------------------------------

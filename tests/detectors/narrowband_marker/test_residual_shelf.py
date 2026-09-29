"""Contamination residuals: the second half of the tolerance cost.

Two things this suite has to establish. First, that adding the term changes
nothing when it is absent; every published number in ``out/`` predates it and
must survive byte-for-byte. Second, that with it present the forecast stops
being monotone in the masked fraction, which is the whole point: only then is a
detector threshold something you can optimise rather than assume.
"""
import math

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
    """Minimal v5 survey product with the keys the residual module reads.

    Under the v5 contract a kept frame has no positive pilot excess, so it
    carries no shelf. To carry every shelf value asked for, this fixture
    rejects every frame (``rejected`` is not used). The shelf calibration
    offset is 0 dB, so a frame's shelf is its pilot excess in dB, carried by
    exact integer power terms.
    """
    import json
    from pilot_proxy.products import reader as v5
    n = shelf_db.size
    rejected = np.ones(n, dtype=bool)
    half = np.uint64(1 << 40)
    lower = np.full((n, 1), half, dtype=np.uint64)
    reference = lower + lower
    ratio_wanted = np.where(rejected, 1.0 + 10.0 ** (shelf_db / 10.0), 1.0)
    target = np.rint(ratio_wanted * float(reference[0, 0])).astype(np.uint64).reshape(n, 1)
    target[~rejected, 0] = reference[~rejected, 0]
    ratio = target.astype(np.float64) / reference.astype(np.float64)
    excess = ratio - 1.0
    excess_db = np.full((n, 1), np.nan)
    excess_db[excess > 0.0] = 10.0 * np.log10(excess[excess > 0.0])
    mask = np.array([int(t) > int(r) for t, r in zip(target[:, 0], reference[:, 0])], dtype=np.uint8)
    assert np.array_equal(mask.astype(bool), rejected), "a rejected frame needs a positive excess"
    decision = {"active_decision": {"implementation": "host_exact_integer_comparison",
                                    "method": "coarse_normalized_positive_excess", "output_field": "reject_mask"},
                "fine_candidate_decision": {"active": False, "method": "fine_order_statistic_cfar"},
                "fine_measurement": {"method": v5.FINE_MEASUREMENT_METHOD,
                                     "role": "measurement_only_no_scan_time_decision",
                                     "terms_field": "fine_power_u64"}}
    detector = {"schema_version": "pilotproxy_detector_contract_v1", "num_weight_terms": 3,
                "power_accumulator": "uint64", "power_accumulator_bits": 64, "threshold_mode": "none",
                "per_frequency_threshold": False, "valid_rule": v5.VALID_RULE, "mask_rule": v5.MASK_RULE,
                "equivalent_mask_rule": "R_coarse > R_null; R_null = 2*target_norm_sq/reference_norm_sum_sq"}
    np.savez(
        path,
        schema_name=np.asarray(v5.PRODUCT_SCHEMA_NAME),
        schema_revision=np.asarray(v5.PRODUCT_SCHEMA_REVISION, dtype=np.int64),
        schema_version=np.asarray(v5.PRODUCT_SCHEMA_TOKEN),
        source_event_key_schema_version=np.asarray(v5.SOURCE_EVENT_KEY_SCHEMA),
        decision_contract_json=np.asarray(json.dumps(decision)),
        detector_contract_json=np.asarray(json.dumps(detector)),
        mask_rule=np.asarray(v5.MASK_RULE),
        frame_index=np.arange(n, dtype=np.int64),
        valid=np.ones((n, 1), dtype=np.uint8),
        reject_mask=mask.reshape(n, 1),
        p_target_u64=target,
        p_ref_lower_u64=lower,
        p_ref_upper_u64=lower.copy(),
        p_ref_sum_u64=reference,
        target_norm_sq=np.array([1], dtype=np.int64),
        reference_norm_sum_sq=np.array([1], dtype=np.int64),
        coarse_power_ratio=2.0 * ratio,
        normalized_coarse_power_ratio_db=10.0 * np.log10(ratio),
        normalized_pilot_excess=excess,
        pilot_excess_db=excess_db,
        estimated_data_shelf_snr_db=excess_db + 0.0,
        pilot_below_data_db=np.float64(0.0),
        bin_enbw_hz=np.float64(1.0),
        dtv_bandwidth_hz=np.float64(1.0),
        pilot_capture_efficiency=np.float64(1.0),
        pilot_in_band=np.array([1], dtype=np.uint8),
        frame_unit_index=unit_of_frame.astype(np.int32),
        unit_time0_ctime=np.asarray(unit_t0, dtype=np.float64),
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
    # without the epoch, the off sample is the kept frames; a v5 kept frame
    # carries no shelf (and this fixture keeps none), so there is no floor:
    # NaN with no off frames, as the releases carry for a band with no dated
    # off epoch
    c = residual.shelf_statistics(signoff)
    assert math.isnan(c.floor_db) and c.n_off_frames == 0


# ----------------------------------------------------------------------
# Is masking worth it?
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# The four-way policy comparison
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# One booking, one floor discipline (the ch35/ch32 reconciliation)
# ----------------------------------------------------------------------

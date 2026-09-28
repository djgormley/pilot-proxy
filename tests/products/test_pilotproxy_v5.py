"""Current product adaptation and archive acceptance tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.products.acceptance import (
    ArchiveAcceptanceError, validate_archive_products)
from pilot_proxy.products.reader import MASK_RULE

from pilot_proxy.config.project import default_project


class products:  # noqa: N801 -- stands in for the retired RFIsher product registry
    freq_id = staticmethod(lambda ch: default_project().target_freq_id(str(ch)))


FIXTURE = Path(__file__).parent / "data" / "pilotproxy_v5_fixture.json"


def _fixture_spec() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _decision_contract() -> dict:
    return {
        "active_decision": {
            "implementation": "host_exact_integer_comparison",
            "method": "coarse_normalized_positive_excess",
            "output_field": "reject_mask",
        },
        "fine_candidate_decision": {
            "active": False,
            "calibration_status": "pending_campaign",
            "method": "fine_order_statistic_cfar",
        },
        "fine_measurement": {
            "method": "exact_fine_power_terms",
            "role": "measurement_only_no_scan_time_decision",
            "terms_field": "fine_power_u64",
        },
    }


def _detector_contract(spec: dict) -> dict:
    return {
        "schema_version": "pilotproxy_detector_contract_v1",
        "detector_window_samples": 128,
        "skipped_guard_bins": 1,
        "reference_offset_bins": 2,
        "num_weight_terms": 3,
        "sample_bits_per_component": 4,
        "input_format": "complex_int4_packed_int8",
        "power_accumulator": "uint64",
        "power_accumulator_bits": 64,
        "combine_mode": "all_rows_summed_before_ratio",
        "weight_coordinate_system": "post_spectral_sense_normalization",
        "input_coordinate_system": "post_spectral_sense_normalized",
        "input_preprocessing": {
            "time_reverse_detector_windows_before_kernel": True,
        },
        "mask_source": "normalized_positive_excess_decision",
        "threshold_mode": "none",
        "per_frequency_threshold": False,
        "valid_rule": "p_ref_sum != 0",
        "mask_rule": MASK_RULE,
        "equivalent_mask_rule": (
            "R_coarse > R_null; R_null = "
            "2*target_norm_sq/reference_norm_sum_sq"
        ),
        "fine_reduction": {
            "pad_factor": spec["fine_pad_factor"],
            "guard_fine_bins": spec["fine_guard_fine_bins"],
            "p_fa": spec["fine_p_fa"],
            "designated_bins": spec["fine_designated_bins"],
            "census_excluded_bins": spec["fine_census_excluded_bins"],
        },
    }


def _frame_coordinates(frames: int, units: int):
    frame_unit = np.floor(
        np.arange(frames, dtype=np.float64) * units / frames
    ).astype(np.int32)
    frame_in_unit = np.empty(frames, dtype=np.int32)
    for unit in range(units):
        rows = np.flatnonzero(frame_unit == unit)
        frame_in_unit[rows] = np.arange(rows.size, dtype=np.int32)
    return frame_unit, frame_in_unit


def _unit_times(spec: dict, units: int) -> np.ndarray:
    split = units // 2
    values = []
    for unit in range(units):
        late = unit >= split
        index = unit - split if late else unit
        values.append(
            spec["first_time0_ctime"]
            + (spec["late_era_offset_seconds"] if late else 0.0)
            + (index // 3) * 86164.0905
            + (index % 3) * 3600.0
        )
    return np.asarray(values, dtype=np.float64)


def _write_product(path: Path, channel: int, *, frames=None, units=None) -> Path:
    spec = _fixture_spec()
    frame_count = int(spec["frames"] if frames is None else frames)
    unit_count = int(spec["units"] if units is None else units)
    unit_count = min(unit_count, frame_count)
    freq_id = products.freq_id(channel)
    frame_unit, frame_in_unit = _frame_coordinates(frame_count, unit_count)

    lower = np.full((frame_count, 1), spec["p_ref_lower"], dtype=np.uint64)
    upper = np.full((frame_count, 1), spec["p_ref_upper"], dtype=np.uint64)
    reference = lower + upper
    cycle = np.asarray(spec["p_target_cycle"], dtype=np.uint64)
    target = np.resize(cycle, frame_count).reshape(frame_count, 1)
    target_norm = int(spec["target_norm_sq"])
    reference_norm = int(spec["reference_norm_sum_sq"])
    ratio = (target.astype(np.float64) * reference_norm
             / (reference.astype(np.float64) * target_norm))
    excess = ratio - 1.0
    coarse = 2.0 * target.astype(np.float64) / reference.astype(np.float64)
    ratio_db = 10.0 * np.log10(ratio)
    excess_db = np.full((frame_count, 1), np.nan, dtype=np.float64)
    positive = excess > 0.0
    excess_db[positive] = 10.0 * np.log10(excess[positive])
    offset = (
        spec["pilot_below_data_db"]
        - 10.0 * np.log10(spec["dtv_bandwidth_hz"] / spec["bin_enbw_hz"])
        - 10.0 * np.log10(spec["pilot_capture_efficiency"])
    )
    shelf = excess_db + offset
    rejected = np.fromiter(
        (int(num) * reference_norm > target_norm * int(den)
         for num, den in zip(target[:, 0], reference[:, 0])),
        dtype=np.uint8,
        count=frame_count,
    ).reshape(frame_count, 1)

    fine = np.full(
        (frame_count, 3, spec["fine_num_bins"]), 20, dtype=np.uint64)
    bins = np.arange(spec["fine_num_bins"], dtype=np.uint64)
    for frame in range(frame_count):
        fine[frame, 0] += bins
        fine[frame, 1] += bins % 7
        fine[frame, 2] += bins % 11
    fine[:, 0, spec["fine_designated_bins"]] = 10

    source_keys = np.asarray([
        "local:" + json.dumps([
            "fixture.event", f"{channel}-{unit}",
            f"baseband_{channel}_{unit}.h5",
        ], separators=(",", ":"))
        for unit in range(unit_count)
    ])
    unit_keys = np.asarray([
        "local:" + json.dumps([
            "fixture.event", f"{channel}-{unit}",
            f"baseband_{channel}_{unit}_{freq_id}.h5",
        ], separators=(",", ":"))
        for unit in range(unit_count)
    ])
    unit_times = _unit_times(spec, unit_count)
    sample_total = 2 * spec["nfft"] * spec["num_input_streams"]
    kernel = spec["kernel_sha256"]
    source_digest = "e" * 64
    detector_version = (
        "pilot-proxy/fixture "
        f"source={source_digest} kernel=2.3.0 kernel_sha256={kernel} "
        "pilotproxy_per_pilot_product_v5 K=128"
    )
    chime_frequency = 800e6 - freq_id * spec["sample_rate_hz"]
    zeros_u64 = np.zeros((frame_count, 1), dtype=np.uint64)
    np.savez(
        path,
        schema_name=np.asarray(spec["schema_name"]),
        schema_revision=np.asarray(spec["schema_revision"], dtype=np.int64),
        schema_version=np.asarray(spec["schema_version"]),
        source_event_key_schema_version=np.asarray(
            spec["source_event_key_schema_version"]),
        decision_contract_json=np.asarray(json.dumps(_decision_contract())),
        detector_contract_json=np.asarray(json.dumps(_detector_contract(spec))),
        detector_version=np.asarray(detector_version),
        mask_rule=np.asarray(MASK_RULE),
        reference_placement_json=np.asarray(json.dumps({
            "physical_channel": channel,
            "reference_offset_bins": 2,
        })),
        physical_channel=np.asarray([channel], dtype=np.int32),
        freq_id=np.asarray([freq_id], dtype=np.int64),
        pilot_frequency_hz=np.asarray(
            [470.309441e6 + 6e6 * (channel - 14)], dtype=np.float64),
        chime_frequency_hz=np.asarray([chime_frequency], dtype=np.float64),
        pilot_in_band=np.asarray([1], dtype=np.uint8),
        frame_index=np.arange(frame_count, dtype=np.int64),
        frame_unit_index=frame_unit,
        frame_in_unit=frame_in_unit,
        p_target_u64=target,
        p_ref_lower_u64=lower,
        p_ref_upper_u64=upper,
        p_ref_sum_u64=reference,
        target_norm_sq=np.asarray([target_norm], dtype=np.int64),
        reference_norm_sum_sq=np.asarray([reference_norm], dtype=np.int64),
        valid=np.ones((frame_count, 1), dtype=np.uint8),
        reject_mask=rejected,
        coarse_power_ratio=coarse,
        normalized_coarse_power_ratio_db=ratio_db,
        normalized_pilot_excess=excess,
        pilot_excess_db=excess_db,
        estimated_data_shelf_snr_db=shelf,
        pilot_below_data_db=np.asarray(
            spec["pilot_below_data_db"], dtype=np.float64),
        bin_enbw_hz=np.asarray(spec["bin_enbw_hz"], dtype=np.float64),
        dtv_bandwidth_hz=np.asarray(
            spec["dtv_bandwidth_hz"], dtype=np.float64),
        pilot_capture_efficiency=np.asarray(
            spec["pilot_capture_efficiency"], dtype=np.float64),
        fine_status=np.asarray("enabled"),
        fine_num_bins=np.asarray(spec["fine_num_bins"], dtype=np.int64),
        fine_power_u64=fine,
        fine_pad_factor=np.asarray(spec["fine_pad_factor"], dtype=np.int64),
        fine_guard_fine_bins=np.asarray(
            spec["fine_guard_fine_bins"], dtype=np.int64),
        fine_p_fa=np.asarray(spec["fine_p_fa"], dtype=np.float64),
        fine_designated_bins=np.asarray(
            spec["fine_designated_bins"], dtype=np.int64),
        fine_census_excluded_bins=np.asarray(
            spec["fine_census_excluded_bins"], dtype=np.int64),
        source_event_keys=source_keys,
        unit_keys=unit_keys,
        unit_order=unit_keys,
        archive_version=np.asarray(["fixture-v1"] * unit_count),
        unit_git_version_tag=np.asarray(["fixture"] * unit_count),
        unit_input_map_sha256=np.asarray([
            hashlib.sha256(f"input-{channel}-{unit}".encode()).hexdigest()
            for unit in range(unit_count)
        ]),
        unit_collection_server=np.asarray(["local"] * unit_count),
        unit_scope=np.asarray(["local"] * unit_count),
        unit_event_id=np.arange(unit_count, dtype=np.int64) + channel * 1000,
        unit_time0_fpga=np.arange(unit_count, dtype=np.uint64) + 1000,
        unit_time0_ctime=unit_times,
        unit_delta_time=np.full(
            unit_count, 1.0 / spec["sample_rate_hz"], dtype=np.float64),
        sample_rate_hz=np.asarray(spec["sample_rate_hz"], dtype=np.float64),
        nfft=np.asarray(spec["nfft"], dtype=np.int64),
        num_input_streams=np.asarray(
            spec["num_input_streams"], dtype=np.int64),
        detector_window_samples=np.asarray(
            spec["detector_window_samples"], dtype=np.int64),
        sense=np.asarray(spec["sense"], dtype=np.int64),
        rational_overflow_count=np.asarray(0, dtype=np.uint64),
        railed_sample_count=zeros_u64,
        fill_sample_count=zeros_u64,
        railed_sample_total=np.full(
            (frame_count, 1), sample_total, dtype=np.uint64),
        weights_hash=np.asarray(
            hashlib.sha256(f"weights-{channel}".encode()).hexdigest()),
        weight_bank_sha256=np.asarray(spec["weight_bank_sha256"]),
        weight_manifest_sha256=np.asarray(spec["weight_manifest_sha256"]),
        weight_coefficients_sha256=np.asarray(
            spec["weight_coefficients_sha256"]),
    )
    return path


def _replace(path: Path, **changes) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True)
                  for name in archive.files}
    values.update(changes)
    np.savez(path, **values)


def _cohort(tmp_path: Path) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for channel in range(14, 37):
        frames = 120 if channel == 14 else 6
        units = 12 if channel == 14 else 2
        paths.append(_write_product(
            tmp_path / f"{products.freq_id(channel)}.npz",
            channel, frames=frames, units=units))
    return paths


def test_archive_gate_refuses_missing_duplicate_and_mixed_products(tmp_path):
    paths = _cohort(tmp_path)
    with pytest.raises(ArchiveAcceptanceError, match="exactly 23"):
        validate_archive_products(paths[:-1])
    with pytest.raises(ArchiveAcceptanceError, match="duplicate paths"):
        validate_archive_products(paths[:-1] + [paths[0]])

    _replace(paths[-1], fine_status=np.asarray("disabled_by_option"))
    with pytest.raises(ArchiveAcceptanceError, match="fine measurement"):
        validate_archive_products(paths)


def test_archive_gate_refuses_incomplete_units_and_mixed_identity(tmp_path):
    paths = _cohort(tmp_path)
    with np.load(paths[-1], allow_pickle=False) as product:
        frame_unit = np.array(product["frame_unit_index"], copy=True)
    frame_unit[:] = 0
    _replace(paths[-1], frame_unit_index=frame_unit)
    with pytest.raises(ArchiveAcceptanceError, match="cover every unit"):
        validate_archive_products(paths)

    paths = _cohort(tmp_path / "fresh")
    with np.load(paths[-1], allow_pickle=False) as product:
        detector = json.loads(str(product["detector_contract_json"].item()))
    detector["sample_bits_per_component"] = 8
    _replace(paths[-1], detector_contract_json=np.asarray(json.dumps(detector)))
    with pytest.raises(ArchiveAcceptanceError, match="detector contract differs"):
        validate_archive_products(paths)

    paths = _cohort(tmp_path / "mixed")
    _replace(paths[-1], weight_bank_sha256=np.asarray("f" * 64))
    with pytest.raises(ArchiveAcceptanceError, match="mixed cohort identities"):
        validate_archive_products(paths)


def test_archive_gate_refuses_wrong_pilot_and_frame_coordinates(tmp_path):
    paths = _cohort(tmp_path / "pilot")
    with np.load(paths[-1], allow_pickle=False) as product:
        pilot_frequency = np.array(product["pilot_frequency_hz"], copy=True)
    pilot_frequency[0] += 1000.0
    _replace(paths[-1], pilot_frequency_hz=pilot_frequency)
    with pytest.raises(ArchiveAcceptanceError, match="pilot_frequency_hz"):
        validate_archive_products(paths)

    paths = _cohort(tmp_path / "order")
    with np.load(paths[-1], allow_pickle=False) as product:
        frame_unit = np.array(product["frame_unit_index"], copy=True)
    _replace(paths[-1], frame_unit_index=frame_unit[::-1])
    with pytest.raises(ArchiveAcceptanceError, match="nondecreasing"):
        validate_archive_products(paths)

    paths = _cohort(tmp_path / "gaps")
    with np.load(paths[-1], allow_pickle=False) as product:
        frame_in_unit = np.array(product["frame_in_unit"], copy=True)
    _replace(paths[-1], frame_in_unit=frame_in_unit + 1)
    with pytest.raises(ArchiveAcceptanceError, match="contiguous from zero"):
        validate_archive_products(paths)


def test_archive_gate_allows_designated_windows_and_unordered_unit_keys(tmp_path):
    paths = _cohort(tmp_path)
    with np.load(paths[-1], allow_pickle=False) as product:
        detector = json.loads(str(product["detector_contract_json"].item()))
        designated = np.array(product["fine_designated_bins"], copy=True)
        unit_keys = np.array(product["unit_keys"], copy=True)
    designated = (designated + 7) % 256
    detector["fine_reduction"]["designated_bins"] = designated.tolist()
    _replace(
        paths[-1],
        fine_designated_bins=designated,
        detector_contract_json=np.asarray(json.dumps(detector)),
        unit_keys=unit_keys[::-1],
    )
    report = validate_archive_products(paths)
    assert len(report.products) == 23


def test_archive_gate_checks_source_frequency_and_sample_accounting(tmp_path):
    paths = _cohort(tmp_path / "source")
    with np.load(paths[-1], allow_pickle=False) as product:
        source_keys = np.array(product["source_event_keys"], copy=True)
    source_keys[0] = source_keys[0].replace(".h5", "_wrong.h5")
    _replace(paths[-1], source_event_keys=source_keys)
    with pytest.raises(ArchiveAcceptanceError, match="source_event_keys"):
        validate_archive_products(paths)

    paths = _cohort(tmp_path / "frequency")
    with np.load(paths[-1], allow_pickle=False) as product:
        frequency = np.array(product["chime_frequency_hz"], copy=True)
    frequency[0] += 1.0
    _replace(paths[-1], chime_frequency_hz=frequency)
    with pytest.raises(ArchiveAcceptanceError, match="chime_frequency_hz"):
        validate_archive_products(paths)

    paths = _cohort(tmp_path / "samples")
    with np.load(paths[-1], allow_pickle=False) as product:
        total = np.array(product["railed_sample_total"], copy=True)
    _replace(
        paths[-1],
        railed_sample_count=(total * np.uint64(3)) // np.uint64(4),
        fill_sample_count=total // np.uint64(2),
    )
    with pytest.raises(ArchiveAcceptanceError, match="sample-quality count"):
        validate_archive_products(paths)


def test_archive_gate_requires_complete_nonlocal_receiver_identity(tmp_path):
    paths = _cohort(tmp_path)
    with np.load(paths[-1], allow_pickle=False) as product:
        scopes = np.array(product["unit_scope"], copy=True)
        archive_versions = np.array(product["archive_version"], copy=True)
    scopes[:] = "triggered"
    archive_versions[0] = ""
    _replace(
        paths[-1],
        unit_scope=scopes,
        archive_version=archive_versions,
    )
    with pytest.raises(ArchiveAcceptanceError, match="receiver provenance"):
        validate_archive_products(paths)


def test_archive_gate_reports_units_without_a_recorded_sample_interval(tmp_path):
    """An untimed acquisition is a v5 state, not a corruption.

    The producing contract writes NaN into ``unit_delta_time`` for a unit whose
    acquisition metadata never carried a sample interval, and 470 of the 167,728
    units in the September 2026 rebuild are in that state. The gate has to
    accept them --- refusing would refuse the archive the run of record used ---
    while carrying their count, because those frames have no frame time.
    """
    paths = _cohort(tmp_path)
    with np.load(paths[-1], allow_pickle=False) as product:
        delta = np.array(product["unit_delta_time"], copy=True)
        frame_unit = np.array(product["frame_unit_index"], copy=True)
    delta[0] = np.nan
    _replace(paths[-1], unit_delta_time=delta)

    report = validate_archive_products(paths)
    untimed = [item for item in report.products if item.untimed_units]
    assert len(untimed) == 1
    assert untimed[0].untimed_units == 1
    assert untimed[0].untimed_frames == int(np.count_nonzero(frame_unit == 0))
    assert report.untimed_units == 1
    assert report.untimed_frames == untimed[0].untimed_frames
    assert "untimed units: 1" in report.summary()


def test_archive_gate_refuses_untimed_products_and_wrong_intervals(tmp_path):
    """Silence about timing is admissible per unit, never per product.

    A product with no recorded interval anywhere has no exposure and no time
    axis for an era split, and a recorded interval that disagrees with
    ``sample_rate_hz`` contradicts the frame-time definition downstream, so both
    still refuse.
    """
    paths = _cohort(tmp_path / "untimed")
    with np.load(paths[-1], allow_pickle=False) as product:
        delta = np.array(product["unit_delta_time"], copy=True)
    _replace(paths[-1], unit_delta_time=np.full_like(delta, np.nan))
    with pytest.raises(ArchiveAcceptanceError,
                       match="no unit records a sampling interval"):
        validate_archive_products(paths)

    paths = _cohort(tmp_path / "disagree")
    with np.load(paths[-1], allow_pickle=False) as product:
        delta = np.array(product["unit_delta_time"], copy=True)
    delta[0] = np.nan
    delta[1] = float(delta[1]) * 2.0
    _replace(paths[-1], unit_delta_time=delta)
    with pytest.raises(ArchiveAcceptanceError,
                       match="unit sampling intervals are inconsistent"):
        validate_archive_products(paths)

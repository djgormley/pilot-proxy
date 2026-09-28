"""Acceptance gate for a complete archive cohort of per-pilot products.

The cohort is one product per screened band of the project's frequency plan,
each at the band's target freq_id; the detector and instrument dimensions come
from the project profile. ``pilot-proxy products check-archive DIR`` runs it.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from pilot_proxy.config.instrument import nyquist_sign
from pilot_proxy.config.project import default_project

from .reader import (FINE_BINS, PRODUCT_SCHEMA_TOKEN, ProductContractError,
                     _array, _float_scalar, _integer_scalar, _string_scalar,
                     _string_vector, open_product)

_PROJECT = default_project()
_INSTRUMENT = _PROJECT.instrument
EXPECTED_BANDS = _PROJECT.frequency_plan.bands("screened")
EXPECTED_CHANNELS = tuple(int(band.label) for band in EXPECTED_BANDS)
EXPECTED_NFFT = _PROJECT.detector_config.nfft
EXPECTED_SAMPLE_RATE_HZ = _INSTRUMENT.sample_rate_hz
EXPECTED_INPUT_STREAMS = _INSTRUMENT.n_inputs
EXPECTED_DETECTOR_WINDOW = _PROJECT.detector_config.detector_window
EXPECTED_SENSE = nyquist_sign(_INSTRUMENT.nyquist_zone)
EXPECTED_FINE_PAD_FACTOR = 2
EXPECTED_FINE_GUARD_BINS = 1
EXPECTED_FINE_P_FA = 0.001
_KERNEL_PATTERN = re.compile(r"(?:^| )kernel_sha256=([0-9a-f]{64})(?: |$)")


class ArchiveAcceptanceError(ValueError):
    """Raised when a cohort is unsafe to use downstream."""


@dataclass(frozen=True)
class ArchiveProductAccounting:
    path: Path
    physical_channel: int
    freq_id: int
    frame_count: int
    valid_frames: int
    invalid_frames: int
    rejected_frames: int
    unit_count: int
    frames_by_unit: tuple[int, ...]
    valid_by_unit: tuple[int, ...]
    untimed_units: int
    untimed_frames: int
    weights_hash: str


@dataclass(frozen=True)
class ArchiveAcceptanceReport:
    products: tuple[ArchiveProductAccounting, ...]
    detector_version: str
    kernel_sha256: str
    detector_contract_sha256: str
    decision_contract_sha256: str
    weight_bank_sha256: str
    weight_manifest_sha256: str
    weight_coefficients_sha256: str

    @property
    def frame_count(self) -> int:
        return sum(item.frame_count for item in self.products)

    @property
    def valid_frames(self) -> int:
        return sum(item.valid_frames for item in self.products)

    @property
    def invalid_frames(self) -> int:
        return sum(item.invalid_frames for item in self.products)

    @property
    def unit_count(self) -> int:
        return sum(item.unit_count for item in self.products)

    @property
    def untimed_units(self) -> int:
        """Units whose acquisition recorded no sample interval.

        Their frames have no frame time, so a calendar or exposure statement
        made from this cohort has to name them as its missing denominator
        rather than average over them silently.
        """
        return sum(item.untimed_units for item in self.products)

    @property
    def untimed_frames(self) -> int:
        return sum(item.untimed_frames for item in self.products)

    def summary(self) -> str:
        lines = [
            "archive cohort accepted",
            f"  channels: {len(self.products)}",
            f"  units: {self.unit_count}",
            f"  frames: {self.frame_count}",
            f"  valid: {self.valid_frames}",
            f"  invalid: {self.invalid_frames}",
            f"  untimed units: {self.untimed_units} "
            f"(frames without a frame time: {self.untimed_frames})",
            f"  kernel: {self.kernel_sha256}",
            f"  weight bank: {self.weight_bank_sha256}",
            f"  weight manifest: {self.weight_manifest_sha256}",
        ]
        lines.extend(
            f"  ch{item.physical_channel}: freq_id={item.freq_id}, "
            f"units={item.unit_count}, frames={item.frame_count}, "
            f"valid={item.valid_frames}, invalid={item.invalid_frames}, "
            f"untimed_units={item.untimed_units}"
            for item in self.products
        )
        return "\n".join(lines)



def _digest(product: Mapping, name: str) -> str:
    value = _string_scalar(product, name)
    if (len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ArchiveAcceptanceError(
            f"{name!r} must be a lowercase SHA-256 digest"
        )
    return value


def _json_record(product: Mapping, name: str) -> tuple[dict, str]:
    raw = _string_scalar(product, name)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ArchiveAcceptanceError(f"{name!r} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ArchiveAcceptanceError(f"{name!r} must contain an object")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    return value, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_sha256(value: dict) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _common_detector_sha256(detector: dict) -> str:
    common = copy.deepcopy(detector)
    reduction = common.get("fine_reduction")
    if not isinstance(reduction, dict) or "designated_bins" not in reduction:
        raise ArchiveAcceptanceError(
            "detector contract lacks fine designated-bin coordinates"
        )
    del reduction["designated_bins"]
    return _json_sha256(common)


def _source_event_key(unit_key: str, freq_id: int) -> str:
    namespace, separator, payload_text = unit_key.partition(":")
    if not separator or not namespace:
        raise ArchiveAcceptanceError("unit_order contains a malformed identity")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ArchiveAcceptanceError(
            "unit_order contains a malformed identity") from exc
    if (not isinstance(payload, list) or not payload
            or not isinstance(payload[-1], str)):
        raise ArchiveAcceptanceError("unit_order contains a malformed identity")
    filename = payload[-1]
    marker = f"_{freq_id}"
    dot = filename.rfind(".")
    stem = filename if dot < 0 else filename[:dot]
    suffix = "" if dot < 0 else filename[dot:]
    if not stem.endswith(marker):
        raise ArchiveAcceptanceError(
            "unit_order filename lacks its product frequency suffix"
        )
    payload[-1] = stem[:-len(marker)] + suffix
    return namespace + ":" + json.dumps(
        payload, separators=(",", ":"), ensure_ascii=True)


def _fine_identity(product: Mapping, detector: dict,
                   frame_count: int) -> dict[str, object]:
    if _string_scalar(product, "fine_status") != "enabled":
        raise ArchiveAcceptanceError("fine measurement is not enabled")
    if _integer_scalar(product, "fine_num_bins") != FINE_BINS:
        raise ArchiveAcceptanceError(f"fine measurement must have {FINE_BINS} bins")
    fine = _array(product, "fine_power_u64", np.uint64,
                  (frame_count, 3, FINE_BINS))
    valid = _array(product, "valid", np.uint8, (frame_count, 1))[:, 0].astype(bool)
    has_reference = np.any(
        (fine[:, 1, :] > 0) | (fine[:, 2, :] > 0), axis=1)
    if np.any(valid & ~has_reference):
        row = int(np.flatnonzero(
            valid & ~has_reference)[0])
        raise ArchiveAcceptanceError(
            f"enabled fine terms have no positive reference at frame {row}"
        )

    pad_factor = _integer_scalar(product, "fine_pad_factor")
    guard = _integer_scalar(product, "fine_guard_fine_bins")
    p_fa = _float_scalar(product, "fine_p_fa", positive=True)
    if (pad_factor != EXPECTED_FINE_PAD_FACTOR
            or guard != EXPECTED_FINE_GUARD_BINS
            or p_fa != EXPECTED_FINE_P_FA):
        raise ArchiveAcceptanceError(
            "fine measurement settings differ from the archive contract")
    designated = np.asarray(product.get("fine_designated_bins"))
    excluded = np.asarray(product.get("fine_census_excluded_bins"))
    for name, values in (("fine_designated_bins", designated),
                         ("fine_census_excluded_bins", excluded)):
        if values.dtype != np.dtype(np.int64) or values.ndim != 1:
            raise ArchiveAcceptanceError(f"{name!r} must be an int64 vector")
        if (np.any(values < 0) or np.any(values >= FINE_BINS)
                or np.unique(values).size != values.size):
            raise ArchiveAcceptanceError(f"{name!r} contains invalid bins")
    if designated.size == 0:
        raise ArchiveAcceptanceError("fine designated-bin set is empty")
    if np.intersect1d(designated, excluded).size:
        raise ArchiveAcceptanceError("fine designated and excluded bins overlap")

    reduction = detector.get("fine_reduction")
    if not isinstance(reduction, dict):
        raise ArchiveAcceptanceError("detector contract lacks fine reduction")
    expected = {
        "pad_factor": pad_factor,
        "guard_fine_bins": guard,
        "p_fa": p_fa,
        "designated_bins": designated.tolist(),
        "census_excluded_bins": excluded.tolist(),
    }
    for name, value in expected.items():
        if reduction.get(name) != value:
            raise ArchiveAcceptanceError(
                f"fine field {name!r} disagrees with the detector contract"
            )
    return {
        "fine_num_bins": FINE_BINS,
        "fine_pad_factor": pad_factor,
        "fine_guard_fine_bins": guard,
        "fine_p_fa": p_fa,
        "fine_census_excluded_bins": tuple(int(value) for value in excluded),
    }


def _unit_accounting(product: Mapping, frame_count: int, freq_id: int,
                     valid: np.ndarray) -> tuple[int, tuple[int, ...],
                                                  tuple[int, ...], int, int]:
    source_values = np.asarray(product.get("source_event_keys"))
    if source_values.ndim != 1 or source_values.dtype.kind not in {"U", "S"}:
        raise ArchiveAcceptanceError("source_event_keys must be a string vector")
    unit_count = int(source_values.size)
    if unit_count == 0:
        raise ArchiveAcceptanceError("source-event inventory is empty")
    source_keys = _string_vector(product, "source_event_keys", unit_count)
    unit_keys = _string_vector(product, "unit_keys", unit_count)
    unit_order = _string_vector(product, "unit_order", unit_count)
    if (any(not value for value in source_keys + unit_keys + unit_order)
            or len(set(source_keys)) != unit_count
            or len(set(unit_keys)) != unit_count
            or len(set(unit_order)) != unit_count):
        raise ArchiveAcceptanceError("source and unit identities must be unique")
    if set(unit_keys) != set(unit_order):
        raise ArchiveAcceptanceError("unit_keys and unit_order do not agree")
    expected_source_keys = tuple(
        _source_event_key(value, freq_id) for value in unit_order)
    if source_keys != expected_source_keys:
        raise ArchiveAcceptanceError(
            "source_event_keys do not match the ordered unit identities"
        )

    provenance = {
        name: _string_vector(product, name, unit_count)
        for name in ("archive_version", "unit_git_version_tag",
                     "unit_input_map_sha256", "unit_collection_server",
                     "unit_scope")
    }
    if any(not value.strip() for value in provenance["unit_scope"]):
        raise ArchiveAcceptanceError("unit scope is incomplete")
    for index, scope in enumerate(provenance["unit_scope"]):
        input_digest = provenance["unit_input_map_sha256"][index]
        if input_digest and (len(input_digest) != 64
                             or any(char not in "0123456789abcdef"
                                    for char in input_digest)):
            raise ArchiveAcceptanceError("unit input-map digest is malformed")
        receiver_identity = (
            provenance["archive_version"][index],
            provenance["unit_git_version_tag"][index],
            input_digest,
            provenance["unit_collection_server"][index],
        )
        if (scope != "local"
                and any(not value.strip() for value in receiver_identity)):
            raise ArchiveAcceptanceError(
                "nonlocal units require complete receiver provenance"
            )

    unit_event_id = _array(product, "unit_event_id", np.int64, (unit_count,))
    _array(product, "unit_time0_fpga", np.uint64, (unit_count,))
    time0 = _array(product, "unit_time0_ctime", np.float64, (unit_count,))
    delta = _array(product, "unit_delta_time", np.float64, (unit_count,))
    if (np.any(unit_event_id < -1) or not np.isfinite(time0).all()
            or any(scope != "local" and event_id < 0
                   for scope, event_id in zip(
                       provenance["unit_scope"], unit_event_id))):
        raise ArchiveAcceptanceError("unit event or time provenance is invalid")
    sample_rate = _float_scalar(product, "sample_rate_hz", positive=True)
    # A unit whose acquisition never recorded a sample interval carries NaN
    # here, and that is a product state the v5 contract admits rather than a
    # corruption: the producer's own check (pilot_proxy.product_contract, the
    # ``allow_nan=True`` unit_delta_time array) constrains only the recorded
    # entries, and the rest of this package reads the field the same way ---
    # the score bundle requires finite timing only of the frames a caller
    # actually selects, and products.reader.Product.frame_time returns
    # NaN for such frames so that a calendar statistic must state its
    # denominator. Demanding every unit be timed would refuse the whole
    # September rebuild over 470 of its 167,728 units. What the cohort does
    # need is that the intervals that *were* recorded agree with the declared
    # sample rate, and that a product is not entirely untimed --- an untimed
    # product has no exposure or time axis for any downstream era split.
    recorded = np.isfinite(delta)
    if not recorded.any():
        raise ArchiveAcceptanceError("no unit records a sampling interval")
    if (np.any(delta[recorded] <= 0.0)
            or not np.allclose(delta[recorded], 1.0 / sample_rate,
                               rtol=1e-12, atol=0.0)):
        raise ArchiveAcceptanceError("unit sampling intervals are inconsistent")

    frame_unit = _array(product, "frame_unit_index", np.int32, (frame_count,))
    frame_in_unit = _array(product, "frame_in_unit", np.int32, (frame_count,))
    if (np.any(frame_unit < 0) or np.any(frame_unit >= unit_count)
            or np.any(frame_in_unit < 0)):
        raise ArchiveAcceptanceError("frame unit coordinates are out of range")
    if np.any(np.diff(frame_unit) < 0):
        raise ArchiveAcceptanceError("frame unit indices are not nondecreasing")
    if set(frame_unit.tolist()) != set(range(unit_count)):
        raise ArchiveAcceptanceError("frame coordinates do not cover every unit")
    if len(set(zip(frame_unit.tolist(), frame_in_unit.tolist()))) != frame_count:
        raise ArchiveAcceptanceError("frame unit coordinates are not unique")
    if any(not np.array_equal(
            frame_in_unit[frame_unit == index],
            np.arange(np.count_nonzero(frame_unit == index), dtype=np.int32))
            for index in range(unit_count)):
        raise ArchiveAcceptanceError(
            "frame_in_unit is not contiguous from zero for every unit")
    frames_by_unit = tuple(
        int(np.count_nonzero(frame_unit == index)) for index in range(unit_count)
    )
    valid_by_unit = tuple(
        int(np.count_nonzero(valid & (frame_unit == index)))
        for index in range(unit_count)
    )
    if sum(frames_by_unit) != frame_count or sum(valid_by_unit) != int(valid.sum()):
        raise ArchiveAcceptanceError("unit frame accounting does not close")
    # Accepting untimed units silently would hand downstream code a frame-time
    # vector with unannounced NaNs, so the report carries their count: any
    # calendar statement built on this cohort has to quote this denominator.
    untimed_units = int(np.count_nonzero(~recorded))
    untimed_frames = int(np.isin(frame_unit, np.flatnonzero(~recorded)).sum())
    return (unit_count, frames_by_unit, valid_by_unit,
            untimed_units, untimed_frames)


def _sample_accounting(product: Mapping, frame_count: int) -> dict[str, int]:
    nfft = _integer_scalar(product, "nfft")
    streams = _integer_scalar(product, "num_input_streams")
    window = _integer_scalar(product, "detector_window_samples")
    sense = _integer_scalar(product, "sense")
    if (nfft != EXPECTED_NFFT or streams != EXPECTED_INPUT_STREAMS
            or window != EXPECTED_DETECTOR_WINDOW
            or sense != EXPECTED_SENSE):
        raise ArchiveAcceptanceError(
            "detector dimensions differ from the archive contract")
    if _integer_scalar(product, "rational_overflow_count", np.uint64) != 0:
        raise ArchiveAcceptanceError("exact rational comparison overflowed")
    expected_total = 2 * nfft * streams
    total = _array(product, "railed_sample_total", np.uint64,
                   (frame_count, 1))[:, 0]
    railed = _array(product, "railed_sample_count", np.uint64,
                    (frame_count, 1))[:, 0]
    fill = _array(product, "fill_sample_count", np.uint64,
                  (frame_count, 1))[:, 0]
    if np.any(total.astype(object) != expected_total):
        raise ArchiveAcceptanceError("railed-sample denominator does not close")
    combined = railed.astype(object) + fill.astype(object)
    if (np.any(railed > total) or np.any(fill > total)
            or np.any(combined > total.astype(object))):
        raise ArchiveAcceptanceError("sample-quality count exceeds its denominator")
    sample_rate = _float_scalar(product, "sample_rate_hz", positive=True)
    if sample_rate != EXPECTED_SAMPLE_RATE_HZ:
        raise ArchiveAcceptanceError(
            "sample rate differs from the archive contract")
    return {
        "nfft": nfft,
        "num_input_streams": streams,
        "detector_window_samples": window,
        "sense": sense,
        "sample_rate_hz": sample_rate,
    }


def _product_record(path: Path) -> tuple[ArchiveProductAccounting, dict]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArchiveAcceptanceError(f"{path.name} is not a safe NPZ product") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ArchiveAcceptanceError(f"{path.name} is not an NPZ product")
    try:
        with loaded as product:
            view = open_product(product)
            if view.schema != PRODUCT_SCHEMA_TOKEN:
                raise ArchiveAcceptanceError("archive cohort requires current v5 products")
            frame_count = int(view.valid.size)
            detector, _ = _json_record(
                product, "detector_contract_json")
            required_detector_values = {
                "detector_window_samples": EXPECTED_DETECTOR_WINDOW,
                "skipped_guard_bins": 1,
                "reference_offset_bins": 2,
                "num_weight_terms": 3,
                "sample_bits_per_component": 4,
                "input_format": "complex_int4_packed_int8",
                "power_accumulator": "uint64",
                "power_accumulator_bits": 64,
                "combine_mode": "all_rows_summed_before_ratio",
                "weight_coordinate_system": (
                    "post_spectral_sense_normalization"),
                "input_coordinate_system": (
                    "post_spectral_sense_normalized"),
                "mask_source": "normalized_positive_excess_decision",
                "valid_rule": "p_ref_sum != 0",
                "mask_rule": (
                    "valid && (p_target * reference_norm_sum_sq > "
                    "target_norm_sq * p_ref_sum)"),
                "equivalent_mask_rule": (
                    "R_coarse > R_null; R_null = "
                    "2*target_norm_sq/reference_norm_sum_sq"),
                "per_frequency_threshold": False,
                "threshold_mode": "none",
            }
            differences = [
                name for name, value in required_detector_values.items()
                if detector.get(name) != value
            ]
            if differences:
                raise ArchiveAcceptanceError(
                    "detector contract differs for " + ", ".join(differences))
            preprocessing = detector.get("input_preprocessing")
            if preprocessing != {
                    "time_reverse_detector_windows_before_kernel": True}:
                raise ArchiveAcceptanceError(
                    "detector input preprocessing differs")
            detector_sha = _common_detector_sha256(detector)
            _, decision_sha = _json_record(product, "decision_contract_json")
            fine_identity = _fine_identity(product, detector, frame_count)
            (unit_count, frames_by_unit, valid_by_unit,
             untimed_units, untimed_frames) = _unit_accounting(
                product, frame_count, view.freq_id, view.valid)
            sample_identity = _sample_accounting(product, frame_count)
            expected_frequency = _INSTRUMENT.hz_of_freq_id(view.freq_id)
            if not math.isclose(
                    view.chime_frequency_hz, expected_frequency,
                    rel_tol=0.0, abs_tol=1e-6):
                raise ArchiveAcceptanceError(
                    "chime_frequency_hz disagrees with freq_id and sample_rate_hz"
                )
            pilot_frequency = float(_array(
                product, "pilot_frequency_hz", np.float64, (1,))[0])
            expected_pilot = _PROJECT.marker_hz(str(view.physical_channel))
            if (not math.isfinite(pilot_frequency)
                    or abs(pilot_frequency - expected_pilot) >= 1.0):
                raise ArchiveAcceptanceError(
                    "pilot_frequency_hz disagrees with the physical channel")
            if abs(pilot_frequency - view.chime_frequency_hz) \
                    > sample_identity["sample_rate_hz"] / 2.0:
                raise ArchiveAcceptanceError(
                    "pilot_frequency_hz lies outside the coarse channel")

            detector_version = _string_scalar(product, "detector_version")
            kernel_match = _KERNEL_PATTERN.search(detector_version)
            if kernel_match is None:
                raise ArchiveAcceptanceError(
                    "detector_version lacks a kernel SHA-256 identity"
                )
            _json_record(product, "reference_placement_json")
            identity = {
                "detector_version": detector_version,
                "kernel_sha256": kernel_match.group(1),
                "detector_contract_sha256": detector_sha,
                "decision_contract_sha256": decision_sha,
                "weight_bank_sha256": _digest(product, "weight_bank_sha256"),
                "weight_manifest_sha256": _digest(
                    product, "weight_manifest_sha256"),
                "weight_coefficients_sha256": _digest(
                    product, "weight_coefficients_sha256"),
                "pilot_below_data_db": _float_scalar(
                    product, "pilot_below_data_db"),
                "bin_enbw_hz": _float_scalar(
                    product, "bin_enbw_hz", positive=True),
                "dtv_bandwidth_hz": _float_scalar(
                    product, "dtv_bandwidth_hz", positive=True),
                "pilot_capture_efficiency": _float_scalar(
                    product, "pilot_capture_efficiency", positive=True),
                **fine_identity,
                **sample_identity,
            }
            accounting = ArchiveProductAccounting(
                path=path,
                physical_channel=view.physical_channel,
                freq_id=view.freq_id,
                frame_count=frame_count,
                valid_frames=int(view.valid.sum()),
                invalid_frames=int((~view.valid).sum()),
                rejected_frames=int(view.rejected.sum()),
                unit_count=unit_count,
                frames_by_unit=frames_by_unit,
                valid_by_unit=valid_by_unit,
                untimed_units=untimed_units,
                untimed_frames=untimed_frames,
                weights_hash=_digest(product, "weights_hash"),
            )
            return accounting, identity
    except ArchiveAcceptanceError:
        raise
    except ProductContractError as exc:
        raise ArchiveAcceptanceError(str(exc)) from exc
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ArchiveAcceptanceError("product contract could not be read") from exc


def validate_archive_products(paths: Sequence[str | Path]
                              ) -> ArchiveAcceptanceReport:
    """Validate one complete, non-overlapping cohort: one product per screened band."""
    resolved = tuple(Path(path).expanduser().resolve() for path in paths)
    if len(resolved) != len(EXPECTED_CHANNELS):
        raise ArchiveAcceptanceError(
            f"archive cohort must contain exactly {len(EXPECTED_CHANNELS)} "
            f"products; got {len(resolved)}"
        )
    if len(set(resolved)) != len(resolved):
        raise ArchiveAcceptanceError("archive cohort contains duplicate paths")
    missing_files = [str(path) for path in resolved if not path.is_file()]
    if missing_files:
        raise ArchiveAcceptanceError(
            "archive product paths do not exist: " + ", ".join(missing_files)
        )

    records = []
    identities = []
    for path in resolved:
        try:
            accounting, identity = _product_record(path)
        except ArchiveAcceptanceError as exc:
            raise ArchiveAcceptanceError(f"{path.name}: {exc}") from exc
        records.append(accounting)
        identities.append(identity)

    channels = [item.physical_channel for item in records]
    frequencies = [item.freq_id for item in records]
    if len(set(channels)) != len(channels):
        raise ArchiveAcceptanceError("archive cohort contains duplicate channels")
    if len(set(frequencies)) != len(frequencies):
        raise ArchiveAcceptanceError("archive cohort contains duplicate frequency IDs")
    weight_rows = [item.weights_hash for item in records]
    if len(set(weight_rows)) != len(weight_rows):
        raise ArchiveAcceptanceError("archive cohort contains duplicate weight rows")
    if set(channels) != set(EXPECTED_CHANNELS):
        missing = sorted(set(EXPECTED_CHANNELS) - set(channels))
        extra = sorted(set(channels) - set(EXPECTED_CHANNELS))
        raise ArchiveAcceptanceError(
            f"physical-channel coverage is incomplete; missing={missing}, extra={extra}"
        )
    expected_freq = {int(band.label): _PROJECT.target_freq_id(band)
                     for band in EXPECTED_BANDS}
    for item in records:
        if item.freq_id != expected_freq[item.physical_channel]:
            raise ArchiveAcceptanceError(
                f"ch{item.physical_channel} carries freq_id {item.freq_id}; "
                f"expected {expected_freq[item.physical_channel]}"
            )

    reference = identities[0]
    for item, identity in zip(records[1:], identities[1:]):
        differences = sorted(
            name for name, value in reference.items()
            if identity.get(name) != value
        )
        if differences:
            raise ArchiveAcceptanceError(
                f"ch{item.physical_channel} has mixed cohort identities: "
                + ", ".join(differences)
            )
    ordered = tuple(sorted(records, key=lambda item: item.physical_channel))
    return ArchiveAcceptanceReport(
        products=ordered,
        detector_version=str(reference["detector_version"]),
        kernel_sha256=str(reference["kernel_sha256"]),
        detector_contract_sha256=str(reference["detector_contract_sha256"]),
        decision_contract_sha256=str(reference["decision_contract_sha256"]),
        weight_bank_sha256=str(reference["weight_bank_sha256"]),
        weight_manifest_sha256=str(reference["weight_manifest_sha256"]),
        weight_coefficients_sha256=str(reference["weight_coefficients_sha256"]),
    )


def _input_paths(values: Sequence[str]) -> tuple[Path, ...]:
    paths = tuple(Path(value).expanduser() for value in values)
    directories = [path for path in paths if path.is_dir()]
    if directories:
        if len(paths) != 1:
            raise ArchiveAcceptanceError(
                "pass either one product directory or explicit product paths"
            )
        paths = tuple(sorted(directories[0].glob("*.npz")))
    return paths


def report_json(report: ArchiveAcceptanceReport) -> str:
    """Every field of the report as canonical JSON (``dataclasses.asdict``)."""
    import dataclasses

    def enc(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)
    return json.dumps(dataclasses.asdict(report), indent=1, sort_keys=True, default=enc) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pilot-proxy products check-archive",
        description="Validate a complete archive cohort of current per-pilot products.")
    parser.add_argument(
        "products", nargs="+",
        help=f"one directory containing products, or exactly {len(EXPECTED_CHANNELS)} NPZ paths")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write every field of the report as JSON to this file")
    args = parser.parse_args(argv)
    try:
        report = validate_archive_products(_input_paths(args.products))
    except ArchiveAcceptanceError as exc:
        parser.error(str(exc))
    print(report.summary())
    if args.json is not None:
        args.json.write_text(report_json(report), encoding="utf-8")
    return 0


__all__ = [
    "EXPECTED_CHANNELS", "ArchiveAcceptanceError",
    "ArchiveProductAccounting", "ArchiveAcceptanceReport",
    "validate_archive_products", "report_json", "main",
]

"""The narrowband-marker detector's per-frame scores: exact Q16 keep
boundaries per background rank (the OS-CFAR statistic), and the candidate
grids built from them.

For one frame, the fine statistic ``T[f] = 2 S_0[f] / (S_1[f] + S_2[f])`` of
the marker bins ``D`` and of the bulk ``B`` is ranked in exact rational
arithmetic; for every one-based background rank ``rho`` the frame's *keep
boundary* is the smallest Q16 multiplier ``eta_q16`` at which
``max_D T <= (eta_q16 / 2^16) T_(rho)`` (:func:`required_eta_q16_by_rank`).
A frame is kept when the multiplier is at least its boundary; a frame that no
deployable multiplier keeps carries :data:`ALWAYS_MASKED_Q16`. The arithmetic
is release-authoritative and must equal the detector's own fine decision
(``pilot_proxy.fine_decision.fine_required_multiplier_q16``).

A :class:`ScoreBundle` holds the boundaries of a selected frame cohort with
their frame times and acquisitions, identified by a content digest over a
canonical manifest (``source_id``); :meth:`ScoreBundle.prepare_threshold_family`
attaches residuals and enters the prepared family
(:func:`pilot_proxy.characterization.stability.prepare_family`). The manifest
pins the producing code (this module and ``characterization/surface.py``) and
the package version; a bundle made by other code is refused on load.

The candidate grids: every rank supported by all accepted frames
(:func:`candidate_rho_values`), and every exact change point of the keep
decision plus Q16 one (:func:`candidate_eta_q16`).
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile

import numpy as np

from pilot_proxy import __version__
from pilot_proxy.config.project import default_project
from pilot_proxy.characterization.surface import (ALWAYS_MASKED_Q16, MAX_MULTIPLIER_Q16,
                                                  Q16_SCALE)
from pilot_proxy.products.reader import (FINE_BINS, PRODUCT_SCHEMA_NAME,
                                         PRODUCT_SCHEMA_REVISION, PRODUCT_SCHEMA_TOKEN,
                                         SOURCE_EVENT_KEY_SCHEMA)


SCORE_BUNDLE_SCHEMA = "residual_score_bundle_v2"
_PRODUCER_SCHEMA = "residual_score_producer_v2"
_PRODUCER_FILES = {
    "src/pilot_proxy/detectors/narrowband_marker/scores.py": Path(__file__).resolve(),
    "src/pilot_proxy/characterization/surface.py": Path(__file__).resolve().parents[2]
    / "characterization" / "surface.py",
}

_ARCHIVE_FIELDS = frozenset({
    "manifest_json", "source_row_index", "frame_index", "frame_time",
    "acquisition_index", "exposure_seconds", "rho",
    "required_multiplier_q16", "always_masked_bits",
})
_ARRAY_DTYPES = {
    "source_row_index": np.dtype("<i8"),
    "frame_index": np.dtype("<i8"),
    "frame_time": np.dtype("<f8"),
    "acquisition_index": np.dtype("<i8"),
    "exposure_seconds": np.dtype("<f8"),
    "rho": np.dtype("<i8"),
    "required_multiplier_q16": np.dtype("<u8"),
    "always_masked_bits": np.dtype("u1"),
}


class ScoreRefused(RuntimeError):
    """Raised when a score bundle cannot be established exactly."""


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _producer_record() -> dict[str, object]:
    files = {
        label: _sha256_file(path)
        for label, path in sorted(_PRODUCER_FILES.items())
    }
    digest = hashlib.sha256(b"residual-score-producer-v2\0")
    for label, file_sha256 in files.items():
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256))
    return {
        "schema_version": _PRODUCER_SCHEMA,
        "package": "pilot-proxy",
        "package_version": __version__,
        "source_files": files,
        "source_sha256": digest.hexdigest(),
    }


def _array_sha256(name: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(b"residual-score-array-v2\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_record(name: str, values: np.ndarray) -> dict[str, object]:
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": _array_sha256(name, values),
    }


def _string_scalar(product, field: str) -> str:
    if field not in product:
        raise ScoreRefused(f"source product is missing {field!r}")
    values = np.asarray(product[field])
    if values.shape != () or values.dtype.kind not in {"U", "S"}:
        raise ScoreRefused(f"source field {field!r} must be a string scalar")
    result = str(values.item())
    if not result:
        raise ScoreRefused(f"source field {field!r} must not be empty")
    return result


def _integer_scalar(product, field: str, dtype, *, minimum=None,
                    maximum=None) -> int:
    if field not in product:
        raise ScoreRefused(f"source product is missing {field!r}")
    values = np.asarray(product[field])
    expected = np.dtype(dtype)
    if values.shape != () or values.dtype != expected:
        raise ScoreRefused(
            f"source field {field!r} must be a {expected} scalar")
    result = int(values.item())
    if minimum is not None and result < minimum:
        raise ScoreRefused(f"source field {field!r} is below {minimum}")
    if maximum is not None and result > maximum:
        raise ScoreRefused(f"source field {field!r} exceeds {maximum}")
    return result


def _float_scalar(product, field: str, *, positive=False) -> float:
    if field not in product:
        raise ScoreRefused(f"source product is missing {field!r}")
    values = np.asarray(product[field])
    if values.shape != () or values.dtype != np.dtype(np.float64):
        raise ScoreRefused(
            f"source field {field!r} must be a float64 scalar")
    result = float(values.item())
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ScoreRefused(f"source field {field!r} must be {qualifier}")
    return result


def _exact_array(product, field: str, dtype, shape) -> np.ndarray:
    if field not in product:
        raise ScoreRefused(f"source product is missing {field!r}")
    values = np.asarray(product[field])
    expected = np.dtype(dtype)
    if values.dtype != expected or values.shape != shape:
        raise ScoreRefused(
            f"source field {field!r} must have dtype {expected} and shape "
            f"{shape}, got {values.dtype} and {values.shape}")
    return values


def _digest_scalar(product, field: str) -> str:
    value = _string_scalar(product, field)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ScoreRefused(
            f"source field {field!r} must be a lowercase SHA-256 digest")
    return value


def _json_scalar(product, field: str) -> tuple[str, dict]:
    raw = _string_scalar(product, field)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScoreRefused(f"source field {field!r} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ScoreRefused(f"source field {field!r} must contain an object")
    return raw, value


def _string_vector(product, field: str, size: int) -> tuple[str, ...]:
    if field not in product:
        raise ScoreRefused(f"source product is missing {field!r}")
    values = np.asarray(product[field])
    if (values.ndim != 1 or values.size != size
            or values.dtype.kind not in {"U", "S"}):
        raise ScoreRefused(
            f"source field {field!r} must be a string vector of length {size}")
    return tuple(str(value) for value in values.tolist())


def _declared_bins(anchor_bin, designated_half_width) -> np.ndarray:
    for value, name in ((anchor_bin, "anchor_bin"),
                        (designated_half_width, "designated_half_width")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ScoreRefused(f"{name} must be an integer")
    anchor = int(anchor_bin)
    width = int(designated_half_width)
    if not 0 <= anchor < FINE_BINS:
        raise ScoreRefused("anchor_bin must be in [0, 256)")
    if not 0 <= width < FINE_BINS // 2:
        raise ScoreRefused("designated_half_width must be in [0, 128)")
    return np.asarray([(anchor + offset) % FINE_BINS
                       for offset in range(-width, width + 1)], dtype="<i8")


def _boolean_mask(values, size: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype(bool) or result.shape != (size,):
        raise ScoreRefused(
            f"{name} must be an explicit boolean vector of length {size}")
    return np.ascontiguousarray(result)


def _pack_bulk_mask(mask: np.ndarray) -> list[str]:
    words = []
    for start in range(0, FINE_BINS, 64):
        word = 0
        for offset in np.flatnonzero(mask[start:start + 64]):
            word |= 1 << int(offset)
        words.append(f"{word:016x}")
    return words


def _rational_order(left, right) -> int:
    left_num, left_den, left_bin = left
    right_num, right_den, right_bin = right
    first = left_num * right_den
    second = right_num * left_den
    if first < second:
        return -1
    if first > second:
        return 1
    return left_bin - right_bin


def required_eta_q16_by_rank(
        fine_power_u64,
        *,
        anchor_bin: int,
        designated_half_width: int,
        bulk_mask,
) -> tuple[int, ...]:
    """Return every exact one-based-rho Q16 boundary for one frame."""
    powers = np.asarray(fine_power_u64)
    if powers.dtype != np.dtype(np.uint64) or powers.shape != (3, FINE_BINS):
        raise ScoreRefused(
            "fine_power_u64 must have uint64 dtype and shape (3, 256)")
    designated = _declared_bins(anchor_bin, designated_half_width)
    bulk = _boolean_mask(bulk_mask, FINE_BINS, "bulk_mask")
    if not bulk.any():
        raise ScoreRefused("bulk_mask must select at least one bin")
    if bulk[designated].any():
        raise ScoreRefused("bulk_mask must exclude every designated bin")

    numerator = [2 * int(value) for value in powers[0]]
    denominator = [int(lower) + int(upper)
                   for lower, upper in zip(powers[1], powers[2])]
    ranked = [(numerator[bin_index], denominator[bin_index], bin_index)
              for bin_index in np.flatnonzero(bulk)
              if denominator[bin_index] > 0]
    if not ranked:
        raise ScoreRefused(
            "declared bulk has no bin with a positive reference denominator")
    ranked.sort(key=cmp_to_key(_rational_order))

    designated_values = [
        (numerator[int(bin_index)], denominator[int(bin_index)])
        for bin_index in designated
        if denominator[int(bin_index)] > 0 and numerator[int(bin_index)] > 0
    ]
    designated_maximum = None
    for value in designated_values:
        if (designated_maximum is None
                or value[0] * designated_maximum[1]
                > designated_maximum[0] * value[1]):
            designated_maximum = value
    boundaries = []
    for rank_num, rank_den, _ in ranked:
        if designated_maximum is None:
            boundaries.append(1)
            continue
        if rank_num == 0:
            boundaries.append(ALWAYS_MASKED_Q16)
            continue
        designated_num, designated_den = designated_maximum
        numerator_product = designated_num * Q16_SCALE * rank_den
        denominator_product = rank_num * designated_den
        boundary = ((numerator_product + denominator_product - 1)
                    // denominator_product)
        boundaries.append(
            ALWAYS_MASKED_Q16 if boundary > MAX_MULTIPLIER_Q16
            else max(1, boundary))
    return tuple(boundaries)


def _validate_source_product(product, selected_frames, *, source_path: Path):
    name = _string_scalar(product, "schema_name")
    revision = _integer_scalar(product, "schema_revision", np.int64)
    token = _string_scalar(product, "schema_version")
    if (name != PRODUCT_SCHEMA_NAME or revision != PRODUCT_SCHEMA_REVISION
            or token != PRODUCT_SCHEMA_TOKEN):
        raise ScoreRefused(
            f"unsupported source schema {name!r} revision {revision} "
            f"token {token!r}; expected {PRODUCT_SCHEMA_TOKEN!r}")
    if _string_scalar(product, "source_event_key_schema_version") \
            != SOURCE_EVENT_KEY_SCHEMA:
        raise ScoreRefused("unsupported source-event identity schema")

    decision_raw, decision = _json_scalar(product, "decision_contract_json")
    measurement = decision.get("fine_measurement")
    candidate = decision.get("fine_candidate_decision")
    if (not isinstance(measurement, dict)
            or measurement.get("method") != "exact_fine_power_terms"
            or measurement.get("role") != "measurement_only_no_scan_time_decision"
            or measurement.get("terms_field") != "fine_power_u64"
            or not isinstance(candidate, dict)
            or candidate.get("method") != "fine_order_statistic_cfar"
            or candidate.get("active") is not False):
        raise ScoreRefused(
            "source decision contract does not describe inactive exact fine "
            "measurements")
    detector_raw, detector = _json_scalar(product, "detector_contract_json")
    if (detector.get("schema_version") != "pilotproxy_detector_contract_v1"
            or detector.get("num_weight_terms") != 3
            or detector.get("power_accumulator") != "uint64"):
        raise ScoreRefused("source detector contract is incompatible")

    if _string_scalar(product, "fine_status") != "enabled":
        raise ScoreRefused("source fine measurement is not enabled")
    pilot_in_band = _exact_array(product, "pilot_in_band", np.uint8, (1,))
    if int(pilot_in_band[0]) != 1:
        raise ScoreRefused("source pilot is not in the measured band")
    bins = _integer_scalar(product, "fine_num_bins", np.int64,
                           minimum=FINE_BINS, maximum=FINE_BINS)
    frame_index = np.asarray(product["frame_index"])
    if frame_index.dtype != np.dtype(np.int64) or frame_index.ndim != 1:
        raise ScoreRefused(
            "source frame_index must be a one-dimensional int64 array")
    frame_count = int(frame_index.size)
    if frame_count == 0 or not np.array_equal(
            frame_index, np.arange(frame_count, dtype=np.int64)):
        raise ScoreRefused(
            "source frame_index must be non-empty, contiguous, and zero-based")
    selected = _boolean_mask(selected_frames, frame_count, "selected_frames")
    if not selected.any():
        raise ScoreRefused("selected_frames must retain at least one frame")

    fine = _exact_array(
        product, "fine_power_u64", np.uint64, (frame_count, 3, bins))
    exact_powers = {
        field: _exact_array(product, field, np.uint64, (frame_count, 1))
        for field in ("p_target_u64", "p_ref_sum_u64", "p_ref_lower_u64",
                      "p_ref_upper_u64")
    }
    for row, (lower, upper, total) in enumerate(zip(
            exact_powers["p_ref_lower_u64"].reshape(-1),
            exact_powers["p_ref_upper_u64"].reshape(-1),
            exact_powers["p_ref_sum_u64"].reshape(-1))):
        combined = int(lower) + int(upper)
        if combined > MAX_MULTIPLIER_Q16 or combined != int(total):
            raise ScoreRefused(
                f"source exact coarse reference terms disagree at frame {row}")

    valid = _exact_array(product, "valid", np.uint8, (frame_count, 1))
    if np.any(valid > 1):
        raise ScoreRefused("source valid flags must be zero or one")
    selected_rows = np.flatnonzero(selected).astype("<i8")
    if not np.all(valid[selected_rows, 0] == 1):
        raise ScoreRefused("selected_frames includes an invalid source frame")

    if "source_event_keys" not in product:
        raise ScoreRefused("source product is missing 'source_event_keys'")
    source_key_values = np.asarray(product["source_event_keys"])
    if (source_key_values.ndim != 1
            or source_key_values.dtype.kind not in {"U", "S"}):
        raise ScoreRefused("source_event_keys must be a string vector")
    unit_count = int(source_key_values.size)
    source_keys = _string_vector(product, "source_event_keys", unit_count)
    if (unit_count == 0 or any(not value for value in source_keys)
            or len(set(source_keys)) != unit_count):
        raise ScoreRefused("source_event_keys must be non-empty and unique")
    unit_order = _string_vector(product, "unit_order", unit_count)
    unit_keys = _string_vector(product, "unit_keys", unit_count)
    if (any(not value for value in unit_order + unit_keys)
            or len(set(unit_order)) != unit_count
            or len(set(unit_keys)) != unit_count
            or set(unit_order) != set(unit_keys)):
        raise ScoreRefused("source unit identities are incomplete")
    unit_provenance = {
        field: _string_vector(product, field, unit_count)
        for field in ("archive_version", "unit_git_version_tag",
                      "unit_input_map_sha256", "unit_collection_server",
                      "unit_scope")
    }
    if any(not value.strip() for value in unit_provenance["unit_scope"]):
        raise ScoreRefused("source unit_scope values must not be empty")
    for value in unit_provenance["unit_input_map_sha256"]:
        if value and (len(value) != 64
                      or any(ch not in "0123456789abcdef" for ch in value)):
            raise ScoreRefused(
                "source unit_input_map_sha256 values must be empty or "
                "lowercase SHA-256 digests")
    for index, scope in enumerate(unit_provenance["unit_scope"]):
        if (scope != "local"
                and (not unit_provenance["unit_git_version_tag"][index].strip()
                     or not unit_provenance["unit_input_map_sha256"][index])):
            raise ScoreRefused(
                "nonlocal source units require git-version and input-map "
                "provenance")
    unit_event_id = _exact_array(
        product, "unit_event_id", np.int64, (unit_count,))
    unit_time0_fpga = _exact_array(
        product, "unit_time0_fpga", np.uint64, (unit_count,))
    if np.any(unit_event_id < -1):
        raise ScoreRefused("source unit_event_id values are invalid")
    frame_unit = _exact_array(
        product, "frame_unit_index", np.int32, (frame_count,))
    frame_in_unit = _exact_array(
        product, "frame_in_unit", np.int32, (frame_count,))
    if (np.any(frame_unit < 0) or np.any(frame_unit >= unit_count)
            or np.any(frame_in_unit < 0)):
        raise ScoreRefused("source frame coordinates are out of range")
    if set(frame_unit.tolist()) != set(range(unit_count)):
        raise ScoreRefused("source frame coordinates do not cover every unit")
    coordinates = zip(frame_unit.tolist(), frame_in_unit.tolist())
    if len(set(coordinates)) != frame_count:
        raise ScoreRefused("source frame coordinates are not unique")

    time0 = _exact_array(product, "unit_time0_ctime", np.float64,
                         (unit_count,))
    delta = _exact_array(product, "unit_delta_time", np.float64,
                         (unit_count,))
    sample_rate = _float_scalar(product, "sample_rate_hz", positive=True)
    nfft = _integer_scalar(product, "nfft", np.int64, minimum=1)
    finite_delta = delta[np.isfinite(delta)]
    if (finite_delta.size and (np.any(finite_delta <= 0.0)
            or not np.allclose(finite_delta, 1.0 / sample_rate,
                               rtol=1e-12, atol=0.0))):
        raise ScoreRefused("source frame sampling metadata is inconsistent")
    selected_units = frame_unit[selected_rows]
    selected_time0 = time0[selected_units]
    selected_delta = delta[selected_units]
    if (not np.isfinite(selected_time0).all()
            or not np.isfinite(selected_delta).all()
            or np.any(selected_delta <= 0.0)):
        raise ScoreRefused("selected frames lack finite timing metadata")
    exposure = np.asarray(nfft * selected_delta, dtype="<f8")
    if not np.equal(exposure, exposure[0]).all():
        raise ScoreRefused("selected frames do not have equal exposure")
    frame_time = np.asarray(
        selected_time0 + frame_in_unit[selected_rows] * exposure, dtype="<f8")
    if not np.isfinite(frame_time).all():
        raise ScoreRefused("selected frame times are not finite")

    physical_channel = _exact_array(product, "physical_channel", np.int32, (1,))
    freq_id = _exact_array(product, "freq_id", np.int64, (1,))
    if not 2 <= int(physical_channel[0]) <= 69:
        raise ScoreRefused("source physical channel is out of range")
    if not 0 <= int(freq_id[0]) <= 1023:
        raise ScoreRefused("source frequency index is out of range")
    detector_version = _string_scalar(product, "detector_version")
    source_digests = {
        field: _digest_scalar(product, field)
        for field in ("weights_hash", "weight_bank_sha256",
                      "weight_manifest_sha256")
    }
    if "weight_coefficients_sha256" in product:
        source_digests["weight_coefficients_sha256"] = _digest_scalar(
            product, "weight_coefficients_sha256")

    metadata = {
        "product_file": source_path.name,
        "product_sha256": _sha256_file(source_path),
        "schema_name": name,
        "schema_revision": revision,
        "schema_version": token,
        "physical_channel": int(physical_channel[0]),
        "freq_id": int(freq_id[0]),
        "detector_version": detector_version,
        "decision_contract_sha256": _sha256_text(decision_raw),
        "detector_contract_sha256": _sha256_text(detector_raw),
        "source_event_keys_sha256": _sha256_text(
            _canonical_json(source_keys)),
        "acquisition_provenance_sha256": _sha256_text(_canonical_json([
            {
                "source_event_key": source_keys[index],
                "unit_order": unit_order[index],
                "unit_event_id": int(unit_event_id[index]),
                "unit_time0_fpga": int(unit_time0_fpga[index]),
                **{field: values[index]
                   for field, values in unit_provenance.items()},
            }
            for index in range(unit_count)
        ])),
        **source_digests,
    }
    return {
        "metadata": metadata,
        "selected": selected,
        "selected_rows": selected_rows,
        "fine": fine,
        "frame_index": frame_index,
        "frame_time": frame_time,
        "acquisition_index": np.asarray(selected_units, dtype="<i8"),
        "exposure": exposure,
    }


def _score_selected_frames(source, *, anchor_bin, designated_half_width,
                           bulk_mask):
    selected_rows = source["selected_rows"]
    bulk_count = int(np.count_nonzero(bulk_mask))
    values = np.zeros((selected_rows.size, bulk_count), dtype="<u8")
    always = np.zeros((selected_rows.size, bulk_count), dtype=bool)
    common_count = bulk_count
    for out_row, source_row in enumerate(selected_rows):
        boundaries = required_eta_q16_by_rank(
            source["fine"][source_row],
            anchor_bin=anchor_bin,
            designated_half_width=designated_half_width,
            bulk_mask=bulk_mask)
        common_count = min(common_count, len(boundaries))
        for column, boundary in enumerate(boundaries):
            if boundary == ALWAYS_MASKED_Q16:
                always[out_row, column] = True
            else:
                values[out_row, column] = boundary
    if common_count <= 0:
        raise ScoreRefused("selected frames have no common supported rank")
    return (np.ascontiguousarray(values[:, :common_count]),
            np.ascontiguousarray(always[:, :common_count]))


def _bundle_arrays(source, values, always):
    return {
        "source_row_index": np.asarray(source["selected_rows"], dtype="<i8"),
        "frame_index": np.asarray(
            source["frame_index"][source["selected_rows"]], dtype="<i8"),
        "frame_time": np.asarray(source["frame_time"], dtype="<f8"),
        "acquisition_index": np.asarray(
            source["acquisition_index"], dtype="<i8"),
        "exposure_seconds": np.asarray(source["exposure"], dtype="<f8"),
        "rho": np.arange(1, values.shape[1] + 1, dtype="<i8"),
        "required_multiplier_q16": np.asarray(values, dtype="<u8"),
        "always_masked_bits": np.packbits(
            always, axis=1, bitorder="little").astype("u1", copy=False),
    }


def _manifest(source, arrays, *, anchor_bin, designated_half_width,
              designated, bulk_mask):
    selected_bytes = np.packbits(
        source["selected"], bitorder="little").astype("u1", copy=False)
    core = {
        "schema_version": SCORE_BUNDLE_SCHEMA,
        "producer": _producer_record(),
        "source": source["metadata"],
        "selection": {
            "source_frame_count": int(source["selected"].size),
            "selected_frame_count": int(source["selected_rows"].size),
            "selected_mask_sha256": _array_sha256(
                "selected_frame_bits", selected_bytes),
        },
        "calibration": {
            "anchor_bin": int(anchor_bin),
            "designated_half_width": int(designated_half_width),
            "designated_bins": [int(value) for value in designated],
            "bulk_mask_u64_hex": _pack_bulk_mask(bulk_mask),
            "bulk_mask_sha256": _array_sha256("bulk_mask", bulk_mask),
        },
        "score": {
            "method": "exact_fine_order_statistic_q16_keep_boundary",
            "rho_indexing": "one_based",
            "supported_rho_count": int(arrays["rho"].size),
            "multiplier_scale": Q16_SCALE,
            "always_masked_value": str(ALWAYS_MASKED_Q16),
            "sentinel_encoding": "zero_q16_plus_little_endian_rank_bitset",
        },
        "arrays": {
            name: _array_record(name, values)
            for name, values in arrays.items()
        },
    }
    content_sha256 = _sha256_text(_canonical_json(core))
    return {**core, "content_sha256": content_sha256}


class _RequirementColumn(Sequence[int]):
    def __init__(self, values: np.ndarray, always: np.ndarray):
        self._values = values
        self._always = always

    def __len__(self) -> int:
        return int(self._values.size)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[position]
                         for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(index)
        if bool(self._always[position]):
            return ALWAYS_MASKED_Q16
        return int(self._values[position])

    def __iter__(self) -> Iterator[int]:
        for value, is_always in zip(self._values, self._always):
            yield ALWAYS_MASKED_Q16 if bool(is_always) else int(value)


@dataclass(frozen=True)
class ScoreBundle:
    """Safe exact-Q16 interchange bundle for one selected frame cohort."""

    manifest_json: str
    source_row_index: np.ndarray
    frame_index: np.ndarray
    frame_time: np.ndarray
    acquisition_index: np.ndarray
    exposure_seconds: np.ndarray
    rho: np.ndarray
    required_multiplier_q16: np.ndarray
    always_masked: np.ndarray

    def __post_init__(self):
        try:
            manifest = json.loads(self.manifest_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ScoreRefused("bundle manifest is invalid JSON") from exc
        if not isinstance(manifest, dict):
            raise ScoreRefused("bundle manifest must contain an object")
        if manifest.get("schema_version") != SCORE_BUNDLE_SCHEMA:
            raise ScoreRefused("bundle schema is unsupported")
        if self.manifest_json != _canonical_json(manifest):
            raise ScoreRefused("bundle manifest is not canonical JSON")

        vector_size = np.asarray(self.source_row_index).size
        vectors = ("source_row_index", "frame_index", "frame_time",
                   "acquisition_index", "exposure_seconds")
        for name in vectors:
            values = np.asarray(getattr(self, name))
            if values.dtype != _ARRAY_DTYPES[name] or values.shape != (vector_size,):
                raise ScoreRefused(f"bundle array {name!r} is malformed")
            object.__setattr__(self, name, values)
        rho = np.asarray(self.rho)
        values = np.asarray(self.required_multiplier_q16)
        always = np.asarray(self.always_masked)
        if (rho.dtype != _ARRAY_DTYPES["rho"] or rho.ndim != 1
                or rho.size == 0
                or not np.array_equal(rho, np.arange(1, rho.size + 1))):
            raise ScoreRefused("bundle rho must be contiguous and one-based")
        if (values.dtype != _ARRAY_DTYPES["required_multiplier_q16"]
                or values.shape != (vector_size, rho.size)):
            raise ScoreRefused("bundle Q16 array is malformed")
        if always.dtype != np.dtype(bool) or always.shape != values.shape:
            raise ScoreRefused("bundle sentinel mask is malformed")
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "required_multiplier_q16", values)
        object.__setattr__(self, "always_masked", always)
        if vector_size == 0:
            raise ScoreRefused("bundle frame cohort is empty")
        if np.any(always & (values != 0)) or np.any(~always & (values == 0)):
            raise ScoreRefused("bundle Q16 and sentinel arrays disagree")
        if (not np.isfinite(self.frame_time).all()
                or not np.isfinite(self.exposure_seconds).all()
                or np.any(self.exposure_seconds <= 0.0)
                or not np.equal(self.exposure_seconds,
                                self.exposure_seconds[0]).all()):
            raise ScoreRefused("bundle frame metadata is invalid")
        _validate_manifest(manifest, _serializable_arrays(self))
        for name in (*vectors, "rho", "required_multiplier_q16",
                     "always_masked"):
            values = np.asarray(getattr(self, name))
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @property
    def manifest(self) -> dict:
        return json.loads(self.manifest_json)

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])

    @property
    def source_id(self) -> str:
        return f"sha256:{self.content_sha256}"

    @property
    def frame_count(self) -> int:
        return int(self.frame_index.size)

    @property
    def supported_rho_count(self) -> int:
        return int(self.rho.size)

    def requirements_by_rho(self) -> Mapping[int, Sequence[int]]:
        return {
            int(rho): _RequirementColumn(
                self.required_multiplier_q16[:, column],
                self.always_masked[:, column])
            for column, rho in enumerate(self.rho)
        }

    def prepare_threshold_family(self, systematic_residuals, *,
                                 variance_residuals=None, **options):
        """Attach calibrated residuals and enter strict preparation."""
        reserved = {"frame_times", "acquisition_ids", "exposure_seconds",
                    "source_id"}
        overlap = reserved.intersection(options)
        if overlap:
            raise ScoreRefused(
                "stored bundle metadata cannot be overridden: "
                + ", ".join(sorted(overlap)))
        from pilot_proxy.characterization.stability import prepare_family

        return prepare_family(
            self.requirements_by_rho(), systematic_residuals,
            frame_times=self.frame_time,
            acquisition_ids=self.acquisition_index,
            exposure_seconds=self.exposure_seconds,
            source_id=self.source_id,
            variance_residuals=variance_residuals,
            **options)

    def save(self, path, *, overwrite: bool = False) -> Path:
        """Write the safe bundle atomically."""
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = _serializable_arrays(self)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w+b", prefix=f".{target.name}.", suffix=".tmp",
                    dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, manifest_json=np.asarray(self.manifest_json),
                         **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, target)
            else:
                try:
                    os.link(temporary, target)
                except FileExistsError as exc:
                    raise ScoreRefused(
                        "output already exists; explicit overwrite is required") \
                        from exc
                temporary.unlink()
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return target


def _serializable_arrays(bundle: ScoreBundle) -> dict[str, np.ndarray]:
    packed = np.packbits(
        bundle.always_masked, axis=1, bitorder="little").astype("u1", copy=False)
    return {
        "source_row_index": bundle.source_row_index,
        "frame_index": bundle.frame_index,
        "frame_time": bundle.frame_time,
        "acquisition_index": bundle.acquisition_index,
        "exposure_seconds": bundle.exposure_seconds,
        "rho": bundle.rho,
        "required_multiplier_q16": bundle.required_multiplier_q16,
        "always_masked_bits": packed,
    }


def _validate_manifest(manifest, arrays):
    if not isinstance(manifest, dict):
        raise ScoreRefused("bundle manifest must contain an object")
    if manifest.get("schema_version") != SCORE_BUNDLE_SCHEMA:
        raise ScoreRefused("bundle schema is unsupported")
    content = manifest.get("content_sha256")
    if (not isinstance(content, str) or len(content) != 64
            or any(ch not in "0123456789abcdef" for ch in content)):
        raise ScoreRefused("bundle content identity is invalid")
    core = dict(manifest)
    core.pop("content_sha256", None)
    if _sha256_text(_canonical_json(core)) != content:
        raise ScoreRefused(
            "bundle content identity does not match its manifest")
    if manifest.get("producer") != _producer_record():
        raise ScoreRefused(
            "bundle producer provenance does not match the current scorer")
    records = manifest.get("arrays")
    if not isinstance(records, dict) or set(records) != set(arrays):
        raise ScoreRefused("bundle array manifest is incomplete")
    for name, values in arrays.items():
        if values.dtype != _ARRAY_DTYPES[name]:
            raise ScoreRefused(f"bundle array {name!r} has the wrong dtype")
        if records[name] != _array_record(name, values):
            raise ScoreRefused(
                f"bundle array {name!r} failed its identity check")

    source = manifest.get("source")
    required_source = {
        "product_sha256", "schema_name", "schema_revision", "schema_version",
        "physical_channel", "freq_id", "detector_version",
        "decision_contract_sha256", "detector_contract_sha256",
        "source_event_keys_sha256", "acquisition_provenance_sha256",
        "weights_hash", "weight_bank_sha256", "weight_manifest_sha256",
    }
    if not isinstance(source, dict) or not required_source.issubset(source):
        raise ScoreRefused("bundle source provenance is incomplete")
    if (source["schema_name"] != PRODUCT_SCHEMA_NAME
            or source["schema_revision"] != PRODUCT_SCHEMA_REVISION
            or source["schema_version"] != PRODUCT_SCHEMA_TOKEN):
        raise ScoreRefused("bundle source schema provenance is invalid")
    digest_fields = required_source.difference({
        "schema_name", "schema_revision", "schema_version",
        "physical_channel", "freq_id", "detector_version",
    })
    for field in digest_fields:
        value = source[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)):
            raise ScoreRefused(
                f"bundle source identity {field!r} is invalid")

    selection = manifest.get("selection")
    frame_count = int(arrays["source_row_index"].size)
    if (not isinstance(selection, dict)
            or selection.get("selected_frame_count") != frame_count
            or not isinstance(selection.get("source_frame_count"), int)
            or selection["source_frame_count"] < frame_count):
        raise ScoreRefused("bundle selection provenance is invalid")
    selected_rows = arrays["source_row_index"]
    if (np.any(selected_rows < 0)
            or (selected_rows.size and selected_rows[-1]
                >= selection["source_frame_count"])
            or not np.array_equal(selected_rows, arrays["frame_index"])
            or np.any(np.diff(selected_rows) <= 0)):
        raise ScoreRefused("bundle selected source rows are invalid")
    selected_mask = np.zeros(selection["source_frame_count"], dtype=bool)
    selected_mask[selected_rows] = True
    selected_bits = np.packbits(
        selected_mask, bitorder="little").astype("u1", copy=False)
    if selection.get("selected_mask_sha256") != _array_sha256(
            "selected_frame_bits", selected_bits):
        raise ScoreRefused("bundle selected-frame identity is invalid")

    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ScoreRefused("bundle calibration provenance is incomplete")
    designated = _declared_bins(
        calibration.get("anchor_bin"),
        calibration.get("designated_half_width"))
    if calibration.get("designated_bins") != designated.tolist():
        raise ScoreRefused("bundle designated-bin provenance is invalid")
    words = calibration.get("bulk_mask_u64_hex")
    try:
        if (not isinstance(words, list) or len(words) != 4
                or any(not isinstance(word, str) or len(word) != 16
                       for word in words)):
            raise ValueError
        integers = [int(word, 16) for word in words]
    except ValueError as exc:
        raise ScoreRefused("bundle bulk-mask provenance is invalid") from exc
    bulk = np.asarray([
        bool((integers[bin_index // 64] >> (bin_index % 64)) & 1)
        for bin_index in range(FINE_BINS)
    ], dtype=bool)
    if (not bulk.any() or bulk[designated].any()
            or calibration.get("bulk_mask_sha256")
            != _array_sha256("bulk_mask", bulk)):
        raise ScoreRefused("bundle bulk-mask provenance is invalid")

    score = manifest.get("score")
    if (not isinstance(score, dict)
            or score.get("method")
            != "exact_fine_order_statistic_q16_keep_boundary"
            or score.get("rho_indexing") != "one_based"
            or score.get("supported_rho_count") != int(arrays["rho"].size)
            or score.get("multiplier_scale") != Q16_SCALE
            or score.get("always_masked_value") != str(ALWAYS_MASKED_Q16)
            or score.get("sentinel_encoding")
            != "zero_q16_plus_little_endian_rank_bitset"):
        raise ScoreRefused("bundle score provenance is invalid")


def build_score_bundle(
        product_path,
        selected_frames,
        *,
        anchor_bin: int,
        designated_half_width: int,
        bulk_mask,
) -> ScoreBundle:
    """Build an exact bundle from a current v5 per-pilot product."""
    source_path = Path(product_path).expanduser().resolve()
    if not source_path.is_file():
        raise ScoreRefused(f"source product does not exist: {source_path}")
    designated = _declared_bins(anchor_bin, designated_half_width)
    bulk = _boolean_mask(bulk_mask, FINE_BINS, "bulk_mask")
    if not bulk.any():
        raise ScoreRefused("bulk_mask must select at least one bin")
    if bulk[designated].any():
        raise ScoreRefused("bulk_mask must exclude every designated bin")
    try:
        loaded = np.load(source_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ScoreRefused(
            "source product is not a safe NumPy archive") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ScoreRefused("source product must be an NPZ archive")
    try:
        with loaded as product:
            source = _validate_source_product(
                product, selected_frames, source_path=source_path)
            values, always = _score_selected_frames(
                source, anchor_bin=anchor_bin,
                designated_half_width=designated_half_width,
                bulk_mask=bulk)
    except ScoreRefused:
        raise
    except (KeyError, OSError, ValueError) as exc:
        raise ScoreRefused(
            "source product could not be read under the v5 contract") from exc

    arrays = _bundle_arrays(source, values, always)
    manifest = _manifest(
        source, arrays, anchor_bin=anchor_bin,
        designated_half_width=designated_half_width,
        designated=designated, bulk_mask=bulk)
    return ScoreBundle(
        manifest_json=_canonical_json(manifest),
        source_row_index=arrays["source_row_index"],
        frame_index=arrays["frame_index"],
        frame_time=arrays["frame_time"],
        acquisition_index=arrays["acquisition_index"],
        exposure_seconds=arrays["exposure_seconds"],
        rho=arrays["rho"],
        required_multiplier_q16=arrays["required_multiplier_q16"],
        always_masked=always)


def load_score_bundle(path) -> ScoreBundle:
    """Load and authenticate a safe exact-Q16 interchange bundle."""
    source = Path(path).expanduser()
    try:
        loaded = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ScoreRefused("score bundle is not a safe NumPy archive") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ScoreRefused("score bundle must be an NPZ archive")
    try:
        with loaded as archive:
            if set(archive.files) != _ARCHIVE_FIELDS:
                raise ScoreRefused(
                    "score bundle fields do not match the schema")
            raw_manifest = np.asarray(archive["manifest_json"])
            if (raw_manifest.shape != ()
                    or raw_manifest.dtype.kind not in {"U", "S"}):
                raise ScoreRefused(
                    "score bundle manifest is not a string scalar")
            manifest_json = str(raw_manifest.item())
            try:
                manifest = json.loads(manifest_json)
            except json.JSONDecodeError as exc:
                raise ScoreRefused(
                    "score bundle manifest is invalid JSON") from exc
            if manifest_json != _canonical_json(manifest):
                raise ScoreRefused(
                    "score bundle manifest is not canonical JSON")
            arrays = {
                name: np.ascontiguousarray(archive[name])
                for name in _ARRAY_DTYPES
            }
    except ScoreRefused:
        raise
    except (KeyError, OSError, ValueError) as exc:
        raise ScoreRefused("score bundle could not be read") from exc
    _validate_manifest(manifest, arrays)

    rho_count = int(arrays["rho"].size)
    frame_count = int(arrays["source_row_index"].size)
    expected_packed = (frame_count, (rho_count + 7) // 8)
    if arrays["always_masked_bits"].shape != expected_packed:
        raise ScoreRefused("score bundle sentinel bitset has the wrong shape")
    unpacked = np.unpackbits(
        arrays["always_masked_bits"], axis=1, count=rho_count,
        bitorder="little").astype(bool, copy=False)
    repacked = np.packbits(unpacked, axis=1, bitorder="little")
    if not np.array_equal(repacked, arrays["always_masked_bits"]):
        raise ScoreRefused("score bundle sentinel bitset has nonzero padding")
    return ScoreBundle(
        manifest_json=manifest_json,
        source_row_index=arrays["source_row_index"],
        frame_index=arrays["frame_index"],
        frame_time=arrays["frame_time"],
        acquisition_index=arrays["acquisition_index"],
        exposure_seconds=arrays["exposure_seconds"],
        rho=arrays["rho"],
        required_multiplier_q16=arrays["required_multiplier_q16"],
        always_masked=unpacked)


# ---------------------------------------------------------------- candidate grids
_REGISTER = default_project().register


def rank_from_index(cfar_rank: int) -> int:
    """Convert the detector's zero-based rank to one-based rho."""
    if isinstance(cfar_rank, bool) or not isinstance(cfar_rank, Integral):
        raise TypeError("cfar_rank must be an integer")
    rank = int(cfar_rank)
    if rank < 0:
        raise ValueError("cfar_rank must be non-negative")
    return rank + 1


def candidate_rho_values(minimum_valid_bulk_count: int) -> tuple[int, ...]:
    """Return every one-based rank supported by all accepted frames."""
    if (isinstance(minimum_valid_bulk_count, bool)
            or not isinstance(minimum_valid_bulk_count, Integral)):
        raise TypeError("minimum_valid_bulk_count must be an integer")
    size = int(minimum_valid_bulk_count)
    if size <= 0:
        raise ValueError("minimum_valid_bulk_count must be positive")
    return tuple(range(1, size + 1))


def candidate_eta_q16(
        required_multiplier_q16,
) -> tuple[int, ...]:
    """Return the exact deployable empirical decision staircase."""
    if isinstance(required_multiplier_q16, (str, bytes)):
        raise TypeError("required_multiplier_q16 must be a sequence of integers")
    try:
        values = tuple(required_multiplier_q16)
    except TypeError as exc:
        raise TypeError(
            "required_multiplier_q16 must be a sequence of integers") from exc
    if not values:
        raise ValueError("required_multiplier_q16 must not be empty")
    floor = int(_REGISTER.value("preparation.minimum_multiplier_q16"))
    candidates = {floor}
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise TypeError(
                "required_multiplier_q16 must contain only integers")
        value = int(raw)
        if not 1 <= value <= ALWAYS_MASKED_Q16:
            raise ValueError(
                "required_multiplier_q16 contains an invalid decision boundary")
        if value <= MAX_MULTIPLIER_Q16:
            candidates.add(value)
    return tuple(sorted(candidates))


__all__ = ["SCORE_BUNDLE_SCHEMA", "ScoreBundle", "ScoreRefused", "build_score_bundle",
           "candidate_eta_q16", "candidate_rho_values", "load_score_bundle", "rank_from_index",
           "required_eta_q16_by_rank"]

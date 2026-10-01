# coding=utf-8
"""Validate CHIME run products for shape and metadata consistency."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

import numpy as np

from pilot_proxy.detector_contract import (
    CHIME_RUN_CONFIG_SCHEMA_TOKEN,
    CHIME_STATS_SCHEMA_TOKEN,
    NORMALIZED_POSITIVE_EXCESS_MASK_SOURCE,
    validate_detector_contract,
)
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.provenance import file_sha256

from .hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    COMPLEX_FLOAT,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
    REAL_IMAG_LAST_AXIS,
    STRUCTURED_COMPLEX,
    UNKNOWN_ENCODING,
)
from .products import (
    CHIME_COMBINE_CANONICAL_RELATIVE_PATHS,
    CHIME_COMBINE_GENERATION_MANIFEST_FILENAME,
    CHIME_COMBINE_GENERATION_MANIFEST_SCHEMA,
    CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME,
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_INPUT_MANIFEST_SCHEMA_TOKEN,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
    SCAN_INPUT_MANIFEST_SCHEMA_TOKEN,
)
from .reductions import CHIME_REDUCTIONS_10S_FILENAME


class NpzLike(Protocol):
    files: list[str]

    def __getitem__(self, key: str) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


def _add_error(errors: list[dict[str, str]], check: str, message: str) -> None:
    errors.append({"severity": "error", "check": str(check), "message": str(message)})


def _load_json(path: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _add_error(errors, f"required_file.{path.name}", f"missing {path}")
        return {}
    except json.JSONDecodeError as exc:
        _add_error(errors, f"json.{path.name}", f"invalid JSON: {exc}")
        return {}
    if not isinstance(data, dict):
        _add_error(errors, f"json.{path.name}", "top-level JSON value is not an object")
        return {}
    return data


def _load_npz(path: Path, errors: list[dict[str, str]]) -> NpzLike | None:
    try:
        return cast(NpzLike, np.load(path))
    except FileNotFoundError:
        _add_error(errors, f"required_file.{path.name}", f"missing {path}")
    except Exception as exc:  # noqa: BLE001 - validator should report file problems.
        _add_error(errors, f"npz.{path.name}", f"could not load NPZ: {exc}")
    return None


def _coerce_int_metadata(value: object, *, field: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes)):
        return int(value)
    raise TypeError(f"{field} must be an integer-compatible value, got {value!r}")


def _require_arrays(
    npz: NpzLike,
    *,
    filename: str,
    names: Sequence[str],
    errors: list[dict[str, str]],
) -> None:
    files = set(npz.files)
    for name in names:
        if name not in files:
            _add_error(errors, f"{filename}.{name}", "required array is missing")


def _check_shape(
    *,
    actual: tuple[int, ...],
    expected: tuple[int, ...],
    check: str,
    errors: list[dict[str, str]],
) -> None:
    if tuple(actual) != tuple(expected):
        _add_error(errors, check, f"shape {actual!r} does not match {expected!r}")


def _check_json_schema(
    *,
    payload: dict[str, Any],
    filename: str,
    expected: str,
    errors: list[dict[str, str]],
) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        _add_error(
            errors,
            f"{filename}.schema_version",
            f"schema_version {actual!r} does not match {expected!r}",
        )


def _validate_exact_fields(
    payload: dict[str, Any],
    *,
    required: frozenset[str],
    check: str,
    errors: list[dict[str, str]],
) -> bool:
    keys = {str(key) for key in payload}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        _add_error(errors, check, f"missing required fields: {missing}")
    if unknown:
        _add_error(errors, check, f"unknown fields: {unknown}")
    return not missing and not unknown


def _manifest_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _manifest_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, TypeError, ValueError):
        return False


_CHIME_DATASET_FIELDS = frozenset(
    {
        "physical_channel",
        "pilot_frequency_hz",
        "coarse_channel_center_hz",
        "freq_id",
        "dataset_path",
        "time_axis",
        "stream_axis",
        "complex_axis",
        "sample_encoding",
        "num_input_streams",
        "total_time_samples",
        "segments",
    }
)
_CHIME_SEGMENT_FIELDS = frozenset(
    {
        "path",
        "num_time_samples",
        "shape",
        "dtype",
        "freq_id",
        "coarse_channel_center_hz",
        "sample_encoding",
    }
)
_CHIME_SAMPLE_ENCODINGS = frozenset(
    {
        CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
        COMPLEX_FLOAT,
        STRUCTURED_COMPLEX,
        REAL_IMAG_LAST_AXIS,
        UNKNOWN_ENCODING,
    }
)


@dataclass(frozen=True)
class _InputManifestIdentity:
    physical_channels: tuple[int, ...]
    pilot_frequency_hz: tuple[float, ...] | None = None
    coarse_channel_center_hz: tuple[float | None, ...] | None = None


def _validate_input_manifest(
    payload: dict[str, Any],
    *,
    errors: list[dict[str, str]],
) -> _InputManifestIdentity | None:
    """Validate one of the two current, intentionally distinct manifests."""
    token = payload.get("schema_version")
    if token == CHIME_INPUT_MANIFEST_SCHEMA_TOKEN:
        if not _validate_exact_fields(
            payload,
            required=frozenset(
                {"schema_version", "input_dir", "absolute_time_used", "datasets"}
            ),
            check="input_manifest.fields",
            errors=errors,
        ):
            return None
        if not isinstance(payload["input_dir"], str) or not payload["input_dir"]:
            _add_error(
                errors,
                "input_manifest.input_dir",
                "input_dir must be a non-empty string",
            )
        if not isinstance(payload["absolute_time_used"], bool):
            _add_error(
                errors,
                "input_manifest.absolute_time_used",
                "absolute_time_used must be boolean",
            )
        rows = payload["datasets"]
        if not isinstance(rows, list) or not rows:
            _add_error(
                errors,
                "input_manifest.datasets",
                "datasets must be a non-empty list",
            )
            return None
        channels: list[int] = []
        pilot_frequencies: list[float] = []
        coarse_centers: list[float | None] = []
        for index, row in enumerate(rows):
            check = f"input_manifest.datasets[{index}]"
            if not isinstance(row, dict):
                _add_error(errors, check, "dataset entry must be an object")
                continue
            if not _validate_exact_fields(
                row,
                required=_CHIME_DATASET_FIELDS,
                check=f"{check}.fields",
                errors=errors,
            ):
                continue
            if not _manifest_int(row["physical_channel"]):
                _add_error(
                    errors,
                    f"{check}.physical_channel",
                    "physical_channel must be an integer",
                )
            else:
                channels.append(int(row["physical_channel"]))
            if not _manifest_real(row["pilot_frequency_hz"]) or float(
                row["pilot_frequency_hz"]
            ) <= 0.0:
                _add_error(
                    errors,
                    f"{check}.pilot_frequency_hz",
                    "pilot_frequency_hz must be a positive finite number",
                )
            else:
                pilot_frequencies.append(float(row["pilot_frequency_hz"]))
            coarse_center = row["coarse_channel_center_hz"]
            if coarse_center is not None and not _manifest_real(coarse_center):
                _add_error(
                    errors,
                    f"{check}.coarse_channel_center_hz",
                    "coarse_channel_center_hz must be null or a finite number",
                )
            else:
                coarse_centers.append(
                    None if coarse_center is None else float(coarse_center)
                )
            if row["freq_id"] is not None and not _manifest_int(row["freq_id"]):
                _add_error(
                    errors,
                    f"{check}.freq_id",
                    "freq_id must be null or an integer",
                )
            if not isinstance(row["dataset_path"], str) or not row["dataset_path"]:
                _add_error(
                    errors,
                    f"{check}.dataset_path",
                    "dataset_path must be a non-empty string",
                )
            axes: dict[str, int | None] = {}
            for field, optional in (
                ("time_axis", False),
                ("stream_axis", False),
                ("complex_axis", True),
            ):
                value = row[field]
                if optional and value is None:
                    axes[field] = None
                elif not _manifest_int(value) or int(value) < 0:
                    _add_error(
                        errors,
                        f"{check}.{field}",
                        f"{field} must be "
                        + ("null or " if optional else "")
                        + "a non-negative integer",
                    )
                    axes[field] = None
                else:
                    axes[field] = int(value)
            populated_axes = [value for value in axes.values() if value is not None]
            if len(set(populated_axes)) != len(populated_axes):
                _add_error(
                    errors,
                    f"{check}.axes",
                    "time_axis, stream_axis, and complex_axis must be distinct",
                )
            if (
                not isinstance(row["sample_encoding"], str)
                or row["sample_encoding"] not in _CHIME_SAMPLE_ENCODINGS
            ):
                _add_error(
                    errors,
                    f"{check}.sample_encoding",
                    "sample_encoding is not emitted by the current HDF5 reader",
                )
            if (
                not _manifest_int(row["num_input_streams"])
                or int(row["num_input_streams"]) <= 0
            ):
                _add_error(
                    errors,
                    f"{check}.num_input_streams",
                    "num_input_streams must be a positive integer",
                )
            if (
                not _manifest_int(row["total_time_samples"])
                or int(row["total_time_samples"]) <= 0
            ):
                _add_error(
                    errors,
                    f"{check}.total_time_samples",
                    "total_time_samples must be a positive integer",
                )

            segments = row["segments"]
            if not isinstance(segments, list) or not segments:
                _add_error(
                    errors,
                    f"{check}.segments",
                    "segments must be a non-empty list",
                )
                continue
            segment_lengths: list[int] = []
            for segment_index, segment in enumerate(segments):
                segment_check = f"{check}.segments[{segment_index}]"
                if not isinstance(segment, dict):
                    _add_error(
                        errors,
                        segment_check,
                        "segment entry must be an object",
                    )
                    continue
                if not _validate_exact_fields(
                    segment,
                    required=_CHIME_SEGMENT_FIELDS,
                    check=f"{segment_check}.fields",
                    errors=errors,
                ):
                    continue
                if not isinstance(segment["path"], str) or not segment["path"]:
                    _add_error(
                        errors,
                        f"{segment_check}.path",
                        "path must be a non-empty string",
                    )
                segment_length = segment["num_time_samples"]
                if not _manifest_int(segment_length) or int(segment_length) <= 0:
                    _add_error(
                        errors,
                        f"{segment_check}.num_time_samples",
                        "num_time_samples must be a positive integer",
                    )
                else:
                    segment_lengths.append(int(segment_length))
                shape = segment["shape"]
                shape_values: list[int] | None
                if (
                    not isinstance(shape, list)
                    or not shape
                    or any(not _manifest_int(value) or int(value) <= 0 for value in shape)
                ):
                    _add_error(
                        errors,
                        f"{segment_check}.shape",
                        "shape must be a non-empty list of positive integers",
                    )
                    shape_values = None
                else:
                    shape_values = [int(value) for value in shape]
                if not isinstance(segment["dtype"], str) or not segment["dtype"]:
                    _add_error(
                        errors,
                        f"{segment_check}.dtype",
                        "dtype must be a non-empty string",
                    )
                if segment["freq_id"] is not None and not _manifest_int(
                    segment["freq_id"]
                ):
                    _add_error(
                        errors,
                        f"{segment_check}.freq_id",
                        "freq_id must be null or an integer",
                    )
                segment_center = segment["coarse_channel_center_hz"]
                if segment_center is not None and not _manifest_real(segment_center):
                    _add_error(
                        errors,
                        f"{segment_check}.coarse_channel_center_hz",
                        "coarse_channel_center_hz must be null or a finite number",
                    )
                if (
                    not isinstance(segment["sample_encoding"], str)
                    or segment["sample_encoding"] not in _CHIME_SAMPLE_ENCODINGS
                ):
                    _add_error(
                        errors,
                        f"{segment_check}.sample_encoding",
                        "sample_encoding is not emitted by the current HDF5 reader",
                    )
                if shape_values is not None:
                    for field, axis in axes.items():
                        if axis is not None and axis >= len(shape_values):
                            _add_error(
                                errors,
                                f"{segment_check}.shape",
                                f"{field}={axis} is outside shape rank "
                                f"{len(shape_values)}",
                            )
                    time_axis = axes["time_axis"]
                    if (
                        time_axis is not None
                        and time_axis < len(shape_values)
                        and _manifest_int(segment_length)
                        and shape_values[time_axis] != int(segment_length)
                    ):
                        _add_error(
                            errors,
                            f"{segment_check}.shape",
                            "shape at time_axis does not match num_time_samples",
                        )
                    stream_axis = axes["stream_axis"]
                    if (
                        stream_axis is not None
                        and stream_axis < len(shape_values)
                        and _manifest_int(row["num_input_streams"])
                        and shape_values[stream_axis]
                        != int(row["num_input_streams"])
                    ):
                        _add_error(
                            errors,
                            f"{segment_check}.shape",
                            "shape at stream_axis does not match num_input_streams",
                        )
            if (
                len(segment_lengths) == len(segments)
                and _manifest_int(row["total_time_samples"])
                and sum(segment_lengths) != int(row["total_time_samples"])
            ):
                _add_error(
                    errors,
                    f"{check}.total_time_samples",
                    "total_time_samples does not equal the sum of segment lengths",
                )
        if len(set(channels)) != len(channels):
            _add_error(
                errors,
                "input_manifest.datasets.physical_channel",
                "datasets contain duplicate physical channels",
            )
        if not (
            len(channels)
            == len(pilot_frequencies)
            == len(coarse_centers)
            == len(rows)
        ):
            return None
        return _InputManifestIdentity(
            physical_channels=tuple(channels),
            pilot_frequency_hz=tuple(pilot_frequencies),
            coarse_channel_center_hz=tuple(coarse_centers),
        )

    if token == SCAN_INPUT_MANIFEST_SCHEMA_TOKEN:
        if not _validate_exact_fields(
            payload,
            required=frozenset(
                {"schema_version", "source", "physical_channels", "input_files"}
            ),
            check="input_manifest.fields",
            errors=errors,
        ):
            return None
        if payload["source"] != "chime-scan":
            _add_error(
                errors,
                "input_manifest.source",
                "source must be 'chime-scan'",
            )
        channels_raw = payload["physical_channels"]
        if (
            not isinstance(channels_raw, list)
            or not channels_raw
            or any(not _manifest_int(value) for value in channels_raw)
        ):
            _add_error(
                errors,
                "input_manifest.physical_channels",
                "physical_channels must be a non-empty integer list",
            )
            channels = None
        else:
            channels = [int(value) for value in channels_raw]
            if len(set(channels)) != len(channels):
                _add_error(
                    errors,
                    "input_manifest.physical_channels",
                    "physical_channels contains duplicates",
                )
        input_files = payload["input_files"]
        if not isinstance(input_files, list) or any(
            not isinstance(value, str) or not value for value in input_files
        ):
            _add_error(
                errors,
                "input_manifest.input_files",
                "input_files must be a list of non-empty strings",
            )
        return (
            None
            if channels is None
            else _InputManifestIdentity(physical_channels=tuple(channels))
        )

    _add_error(
        errors,
        "input_manifest.schema_version",
        f"schema_version {token!r} is not one of the current manifest schemas: "
        f"{CHIME_INPUT_MANIFEST_SCHEMA_TOKEN!r}, "
        f"{SCAN_INPUT_MANIFEST_SCHEMA_TOKEN!r}",
    )
    return None


def _is_binary_array(values: np.ndarray) -> bool:
    arr = np.asarray(values)
    return bool(np.all((arr == 0) | (arr == 1)))


def _is_normalized_positive_excess_run(
    *,
    run_config: dict[str, Any],
    stats: dict[str, Any],
) -> bool:
    policy = stats.get("mask_policy") or run_config.get("mask_policy")
    contract = stats.get("detector_contract") or run_config.get("detector_contract")
    return bool(
        (
            isinstance(policy, dict)
            and str(policy.get("mask_source")) == NORMALIZED_POSITIVE_EXCESS_MASK_SOURCE
        )
        or (
            isinstance(contract, dict)
            and str(contract.get("mask_source")) == NORMALIZED_POSITIVE_EXCESS_MASK_SOURCE
        )
    )


def _validate_detector_contract(
    *,
    run_config: dict[str, Any],
    stats: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    run_contract = run_config.get("detector_contract")
    stats_contract = stats.get("detector_contract")
    if not isinstance(run_contract, dict):
        _add_error(
            errors,
            "run_config.detector_contract",
            "detector_contract is missing or not an object",
        )
        return
    run_contract_typed: dict[str, Any] = dict(run_contract)
    if not isinstance(stats_contract, dict):
        _add_error(
            errors,
            "stats.detector_contract",
            "detector_contract is missing or not an object",
        )
        return
    stats_contract_typed: dict[str, Any] = dict(stats_contract)
    if run_contract_typed != stats_contract_typed:
        _add_error(
            errors,
            "detector_contract.consistency",
            "run_config and stats detector_contract objects differ",
        )

    for label, contract in (
        ("run_config", run_contract_typed),
        ("stats", stats_contract_typed),
    ):
        try:
            validate_detector_contract(contract)
        except (TypeError, ValueError) as exc:
            _add_error(
                errors,
                f"{label}.detector_contract.current_schema",
                str(exc),
            )

    contract_window = run_contract_typed.get("detector_window_samples")
    if contract_window is not None:
        try:
            expected_window = _coerce_int_metadata(
                contract_window,
                field="detector_window_samples",
            )
        except (TypeError, ValueError) as exc:
            _add_error(
                errors,
                "detector_contract.detector_window_samples",
                str(exc),
            )
            expected_window = None
        for label, payload in (
            ("run_config", run_config),
            ("stats", stats),
        ):
            payload_window = payload.get("detector_window_samples")
            if payload_window is None or expected_window is None:
                continue
            try:
                payload_window_int = _coerce_int_metadata(
                    payload_window,
                    field=f"{label}.detector_window_samples",
                )
            except (TypeError, ValueError) as exc:
                _add_error(
                    errors,
                    f"{label}.detector_window_samples",
                    str(exc),
                )
                continue
            if payload_window_int != expected_window:
                _add_error(
                    errors,
                    f"{label}.detector_window_samples",
                    "detector_window_samples does not match detector_contract: "
                    f"{payload_window!r} != {expected_window}",
                )


def _validate_detector(
    detector: NpzLike,
    *,
    run_config: dict[str, Any],
    stats: dict[str, Any],
    errors: list[dict[str, str]],
) -> tuple[int, int] | None:
    required = [
        "physical_channel",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "frame_index",
        "p_target_u64",
        "p_ref_sum_u64",
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "mask",
        "valid",
    ]
    _require_arrays(
        detector,
        filename=CHIME_DETECTOR_OUTPUTS_FILENAME,
        names=required,
        errors=errors,
    )
    if any(name not in detector.files for name in required):
        return None

    num_frames = int(np.asarray(detector["frame_index"]).size)
    num_pilots = int(np.asarray(detector["physical_channel"]).size)
    frame_pilot_shape = (num_frames, num_pilots)

    for name in [
        "p_target_u64",
        "p_ref_sum_u64",
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "mask",
        "valid",
    ]:
        _check_shape(
            actual=tuple(np.asarray(detector[name]).shape),
            expected=frame_pilot_shape,
            check=f"detector.{name}.shape",
            errors=errors,
        )
    for name in [
        "pilot_frequency_hz",
        "chime_frequency_hz",
    ]:
        _check_shape(
            actual=tuple(np.asarray(detector[name]).shape),
            expected=(num_pilots,),
            check=f"detector.{name}.shape",
            errors=errors,
        )

    mask = np.asarray(detector["mask"])
    valid = np.asarray(detector["valid"])
    if not _is_binary_array(mask):
        _add_error(errors, "detector.mask.binary", "mask contains values outside 0/1")
    if not _is_binary_array(valid):
        _add_error(errors, "detector.valid.binary", "valid contains values outside 0/1")
    if np.any((mask != 0) & (valid == 0)):
        _add_error(errors, "detector.mask.invalid", "invalid frames are masked")
    if np.any((np.asarray(detector["p_ref_sum_u64"]) > 0) != (valid != 0)):
        _add_error(
            errors,
            "detector.valid.denominator",
            "valid array does not match p_ref_sum_u64 > 0",
        )
    if _is_normalized_positive_excess_run(
        run_config=run_config,
        stats=stats,
    ):
        p_target = np.asarray(detector["p_target_u64"], dtype=np.uint64)
        p_ref_sum = np.asarray(detector["p_ref_sum_u64"], dtype=np.uint64)
        required_norms = {"target_norm_sq", "reference_norm_sum_sq"}
        missing_norms = sorted(required_norms.difference(detector.files))
        if missing_norms:
            _add_error(
                errors,
                "detector.mask.norms_missing",
                "the current normalized mask rule requires arrays: "
                + ", ".join(missing_norms),
            )
        else:
            target_norm = np.asarray(detector["target_norm_sq"]).reshape(-1)
            reference_norm = np.asarray(
                detector["reference_norm_sum_sq"]
            ).reshape(-1)
            p_target_object = p_target.astype(object)
            p_ref_object = p_ref_sum.astype(object)
            target_norm_object = target_norm.astype(object)[np.newaxis, :]
            reference_norm_object = reference_norm.astype(object)[np.newaxis, :]
            expected_mask = (
                (valid != 0)
                & (p_ref_sum != 0)
                & np.asarray(
                    p_target_object * reference_norm_object
                    > target_norm_object * p_ref_object,
                    dtype=bool,
                )
            )
            if np.any((mask != 0) != expected_mask):
                _add_error(
                    errors,
                    "detector.mask.normalized_positive_excess_rule",
                    "mask does not match valid && (p_target * "
                    "reference_norm_sum_sq > target_norm_sq * p_ref_sum)",
                )

    if stats:
        for key, expected in [
            ("num_frames", num_frames),
            ("num_pilots", num_pilots),
        ]:
            if key in stats and int(stats[key]) != int(expected):
                _add_error(
                    errors,
                    f"stats.{key}",
                    f"stats {key}={stats[key]!r} does not match detector {expected}",
                )
        overflow = stats.get("rational_overflow_count_by_pilot")
        if (
            isinstance(overflow, Iterable)
            and not isinstance(overflow, (str, bytes, dict))
            and any(int(value) != 0 for value in overflow)
        ):
            _add_error(
                errors,
                "stats.rational_overflow_count_by_pilot",
                f"nonzero rational overflow counts: {overflow!r}",
            )

    return frame_pilot_shape


def _validate_spectrogram_cache(
    cache: NpzLike,
    detector: NpzLike,
    *,
    shape: tuple[int, int],
    errors: list[dict[str, str]],
) -> None:
    required = [
        "baseband_power_linear",
        "baseband_power_db",
        "mask",
        "valid",
        "physical_channel",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "frame_index",
        "relative_time_s",
    ]
    _require_arrays(
        cache,
        filename=CHIME_SPECTROGRAM_CACHE_FILENAME,
        names=required,
        errors=errors,
    )
    if any(name not in cache.files for name in required):
        return

    for name in ["baseband_power_linear", "baseband_power_db", "mask", "valid"]:
        _check_shape(
            actual=tuple(np.asarray(cache[name]).shape),
            expected=shape,
            check=f"spectrogram.{name}.shape",
            errors=errors,
        )
    for name in ["physical_channel", "pilot_frequency_hz", "chime_frequency_hz"]:
        if not np.array_equal(
            np.asarray(cache[name]),
            np.asarray(detector[name]),
            equal_nan=True,
        ):
            _add_error(
                errors,
                f"spectrogram.{name}",
                f"{name} does not match detector output",
            )
    if not np.array_equal(np.asarray(cache["frame_index"]), np.asarray(detector["frame_index"])):
        _add_error(errors, "spectrogram.frame_index", "frame_index does not match")
    if not np.array_equal(np.asarray(cache["mask"]), np.asarray(detector["mask"])):
        _add_error(errors, "spectrogram.mask", "mask does not match detector output")
    if not np.array_equal(np.asarray(cache["valid"]), np.asarray(detector["valid"])):
        _add_error(errors, "spectrogram.valid", "valid does not match detector output")
    _check_shape(
        actual=tuple(np.asarray(cache["relative_time_s"]).shape),
        expected=(shape[0],),
        check="spectrogram.relative_time_s.shape",
        errors=errors,
    )


def _validate_reductions(
    reductions: NpzLike,
    detector: NpzLike,
    *,
    shape: tuple[int, int],
    errors: list[dict[str, str]],
) -> None:
    required = [
        "chunk_index",
        "chunk_start_frame",
        "chunk_stop_frame",
        "input_power_mean",
        "cleaned_power_mean",
        "valid_count",
        "invalid_count",
        "masked_count_valid",
        "unmasked_count_valid",
        "mask_fraction_valid",
        "mask_fraction_total",
    ]
    _require_arrays(
        reductions,
        filename=CHIME_REDUCTIONS_10S_FILENAME,
        names=required,
        errors=errors,
    )
    if any(name not in reductions.files for name in required):
        return

    num_chunks = int(np.asarray(reductions["chunk_index"]).size)
    num_pilots = int(shape[1])
    for name in ["chunk_start_frame", "chunk_stop_frame"]:
        _check_shape(
            actual=tuple(np.asarray(reductions[name]).shape),
            expected=(num_chunks,),
            check=f"reductions.{name}.shape",
            errors=errors,
        )
    for name in [
        "input_power_mean",
        "cleaned_power_mean",
        "valid_count",
        "invalid_count",
        "masked_count_valid",
        "unmasked_count_valid",
        "mask_fraction_valid",
        "mask_fraction_total",
    ]:
        _check_shape(
            actual=tuple(np.asarray(reductions[name]).shape),
            expected=(num_chunks, num_pilots),
            check=f"reductions.{name}.shape",
            errors=errors,
        )

    valid = np.asarray(detector["valid"]) != 0
    mask = np.asarray(detector["mask"]) != 0
    if np.any(np.sum(np.asarray(reductions["valid_count"]), axis=0) != np.sum(valid, axis=0)):
        _add_error(errors, "reductions.valid_count", "valid_count does not sum to detector valid frames")
    if np.any(
        np.sum(np.asarray(reductions["masked_count_valid"]), axis=0)
        != np.sum(mask & valid, axis=0)
    ):
        _add_error(
            errors,
            "reductions.masked_count_valid",
            "masked_count_valid does not sum to detector masked valid frames",
        )


def _entry_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _safe_generation_file(run: Path, relative: Path) -> Path | None:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    root = run.absolute()
    resolved_root = root.resolve(strict=False)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return None
    try:
        candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _read_generation_snapshot(
    run: Path,
    *,
    phase: str,
    errors: list[dict[str, str]],
    validate_files: bool,
) -> bytes | None:
    marker = run / CHIME_COMBINE_GENERATION_MANIFEST_FILENAME
    if marker.is_symlink():
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            "combine generation manifest may not be a symlink",
        )
        return b"<symlink>"
    try:
        raw = marker.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            f"could not read combine generation manifest: {exc}",
        )
        return b"<unreadable>"
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            f"invalid combine generation manifest JSON: {exc}",
        )
        return raw
    if not isinstance(payload, dict):
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            "combine generation manifest must be a JSON object",
        )
        return raw
    if payload.get("schema_version") != CHIME_COMBINE_GENERATION_MANIFEST_SCHEMA:
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            "combine generation manifest has an unsupported schema",
        )
    generation_id = payload.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or len(generation_id) != 32
        or any(character not in "0123456789abcdef" for character in generation_id)
    ):
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            "combine generation_id must be 32 lowercase hexadecimal characters",
        )
    if payload.get("state") not in {"committed", "recovered"}:
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            "combine generation state must be 'committed' or 'recovered'",
        )
    files = payload.get("files")
    if not isinstance(files, dict):
        _add_error(
            errors,
            f"publish.generation_manifest.{phase}",
            "combine generation files must be an object",
        )
        return raw

    declared: set[str] = set()
    for name, expected_digest in files.items():
        if (
            not isinstance(name, str)
            or name not in CHIME_COMBINE_CANONICAL_RELATIVE_PATHS
            or name == CHIME_COMBINE_GENERATION_MANIFEST_FILENAME
        ):
            _add_error(
                errors,
                f"publish.generation_manifest.{phase}",
                f"generation manifest contains a non-canonical path: {name!r}",
            )
            continue
        declared.add(name)
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_digest
            )
        ):
            _add_error(
                errors,
                f"publish.generation_manifest.{phase}",
                f"invalid SHA-256 for canonical path {name!r}",
            )
            continue
        candidate = _safe_generation_file(run, Path(name))
        if candidate is None:
            _add_error(
                errors,
                f"publish.generation_manifest.{phase}",
                f"canonical path is symlinked or escapes the run directory: {name}",
            )
            continue
        if validate_files:
            digest = file_sha256(candidate)
            if digest != expected_digest:
                _add_error(
                    errors,
                    "publish.generation_hash",
                    f"canonical file {name!r} does not match its generation hash",
                )

    if validate_files:
        actual: set[str] = set()
        for name in CHIME_COMBINE_CANONICAL_RELATIVE_PATHS:
            if name == CHIME_COMBINE_GENERATION_MANIFEST_FILENAME:
                continue
            candidate = _safe_generation_file(run, Path(name))
            if candidate is None:
                if _entry_exists(run / name):
                    _add_error(
                        errors,
                        "publish.generation_manifest.before",
                        f"canonical path is symlinked or escapes the run directory: {name}",
                    )
                continue
            if candidate.is_file():
                actual.add(name)
        if declared != actual:
            _add_error(
                errors,
                "publish.generation_manifest.before",
                "generation manifest file inventory differs from canonical files: "
                f"declared={sorted(declared)!r}, actual={sorted(actual)!r}",
            )
    return raw


def validate_products(
    *,
    run_dir: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    errors: list[dict[str, str]] = []

    publish_journal = run / CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME
    journal_before = _entry_exists(publish_journal)
    generation_before = _read_generation_snapshot(
        run, phase="before", errors=errors, validate_files=True
    )

    run_config = _load_json(run / "run_config.json", errors)
    input_manifest = _load_json(run / "input_manifest.json", errors)
    stats = _load_json(run / "stats.json", errors)
    manifest_identity = (
        _validate_input_manifest(input_manifest, errors=errors)
        if input_manifest
        else None
    )
    if run_config:
        _check_json_schema(
            payload=run_config,
            filename="run_config.json",
            expected=CHIME_RUN_CONFIG_SCHEMA_TOKEN,
            errors=errors,
        )
    if stats:
        _check_json_schema(
            payload=stats,
            filename="stats.json",
            expected=CHIME_STATS_SCHEMA_TOKEN,
            errors=errors,
        )
    if run_config and stats:
        _validate_detector_contract(
            run_config=run_config,
            stats=stats,
            errors=errors,
        )

    detector = _load_npz(run / CHIME_DETECTOR_OUTPUTS_FILENAME, errors)
    cache = _load_npz(run / CHIME_SPECTROGRAM_CACHE_FILENAME, errors)
    reductions = _load_npz(run / CHIME_REDUCTIONS_10S_FILENAME, errors)

    try:
        if detector is not None:
            shape = _validate_detector(
                detector,
                run_config=run_config,
                stats=stats,
                errors=errors,
            )
            if shape is not None and manifest_identity is not None:
                detector_channels = [
                    int(value)
                    for value in np.asarray(detector["physical_channel"]).reshape(-1)
                ]
                manifest_channels = list(manifest_identity.physical_channels)
                if manifest_channels != detector_channels:
                    _add_error(
                        errors,
                        "input_manifest.physical_channels",
                        "manifest physical channels do not match detector output: "
                        f"{manifest_channels!r} != {detector_channels!r}",
                    )
                if manifest_identity.pilot_frequency_hz is not None:
                    manifest_pilots = np.asarray(
                        manifest_identity.pilot_frequency_hz,
                        dtype=np.float64,
                    )
                    detector_pilots = np.asarray(
                        detector["pilot_frequency_hz"],
                        dtype=np.float64,
                    ).reshape(-1)
                    if not np.array_equal(manifest_pilots, detector_pilots):
                        _add_error(
                            errors,
                            "input_manifest.pilot_frequency_hz",
                            "manifest pilot frequencies do not match detector output",
                        )
                if manifest_identity.coarse_channel_center_hz is not None:
                    manifest_centers = np.asarray(
                        [
                            np.nan if value is None else value
                            for value in manifest_identity.coarse_channel_center_hz
                        ],
                        dtype=np.float64,
                    )
                    detector_centers = np.asarray(
                        detector["chime_frequency_hz"],
                        dtype=np.float64,
                    ).reshape(-1)
                    if not np.array_equal(
                        manifest_centers,
                        detector_centers,
                        equal_nan=True,
                    ):
                        _add_error(
                            errors,
                            "input_manifest.coarse_channel_center_hz",
                            "manifest coarse-channel centers do not match detector "
                            "output (null corresponds to detector NaN)",
                        )
            if shape is not None and cache is not None:
                _validate_spectrogram_cache(
                    cache,
                    detector,
                    shape=shape,
                    errors=errors,
                )
            if shape is not None and reductions is not None:
                _validate_reductions(
                    reductions,
                    detector,
                    shape=shape,
                    errors=errors,
                )
    finally:
        for item in [detector, cache, reductions]:
            if item is not None:
                close = getattr(item, "close", None)
                if callable(close):
                    close()

    generation_after = _read_generation_snapshot(
        run, phase="after", errors=errors, validate_files=False
    )
    # This must be the final filesystem observation. A publisher creates the
    # durable journal before changing canonical bytes, so a false result here
    # is the validator's stable linearization point; checking it before the
    # generation marker leaves a window in which publication can begin unseen.
    journal_after = _entry_exists(publish_journal)
    if journal_before or journal_after:
        _add_error(
            errors,
            "publish.transaction",
            "a chime-combine publication overlapped validation or is awaiting "
            "rollback; rerun chime-combine before consuming canonical products",
        )
    if generation_before != generation_after:
        _add_error(
            errors,
            "publish.generation_stability",
            "canonical generation identity changed while products were being "
            "validated; retry validation on a stable run directory",
        )

    report = {
        "schema_version": "pilotproxy_chime_product_validation_v1",
        "run_dir": str(run),
        "valid": len(errors) == 0,
        "num_errors": int(len(errors)),
        "errors": errors,
    }
    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_strict(Path(output_json), report, indent=2, sort_keys=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CHIME run products for internal consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_products(
        run_dir=args.run_dir,
        output_json=args.output_json,
    )
    print("valid, num_errors, run_dir", flush=True)
    print(
        f"{bool(report['valid'])}, {int(report['num_errors'])}, {report['run_dir']}",
        flush=True,
    )
    for error in report["errors"]:
        print(f"ERROR {error['check']}: {error['message']}", flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

# coding=utf-8
"""Export compact runtime weight bundles for future deployment paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.atsc_channels import validate_uhf_physical_channel
from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
    build_detector_contract,
    detector_contract_sha256,
    input_coordinate_system_for_weight_coordinate,
    normalize_weight_coordinate_system,
    null_power_ratio_from_weight_norms,
    validate_detector_contract,
    weight_term_norms_sq,
)
from pilot_proxy.fine_decision import FINE_DECISION_VERSION
from pilot_proxy.fine_reduction import (
    CFAR_DEFAULT_GUARD_FINE_BINS,
    FINE_PAD_FACTOR,
)
from pilot_proxy.integration import (
    load_detector_core_profile,
    load_receiver_profile,
    parse_physical_channel_selection,
)
from pilot_proxy.integration.receiver_profile import receiver_profile_hash
from pilot_proxy.integration.stream_layout import validate_integration_compatibility
from pilot_proxy.integration.weight_generation import (
    generate_weight_table_from_receiver_profile,
    profile_requires_window_time_reversal,
)
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.provenance import file_sha256
from pilot_proxy.schema_identity import schema_token

RUNTIME_WEIGHT_MANIFEST_SCHEMA_NAME = "pilotproxy_runtime_weights_manifest"
RUNTIME_WEIGHT_MANIFEST_SCHEMA_REVISION = 1
RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN = schema_token(
    RUNTIME_WEIGHT_MANIFEST_SCHEMA_NAME, RUNTIME_WEIGHT_MANIFEST_SCHEMA_REVISION
)
RUNTIME_PILOT_PROFILES_SCHEMA_NAME = "pilotproxy_runtime_pilot_profiles"
RUNTIME_PILOT_PROFILES_SCHEMA_REVISION = 1
RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN = schema_token(
    RUNTIME_PILOT_PROFILES_SCHEMA_NAME, RUNTIME_PILOT_PROFILES_SCHEMA_REVISION
)
RUNTIME_BUNDLE_VALIDATION_SCHEMA_NAME = "pilotproxy_runtime_bundle_validation"
RUNTIME_BUNDLE_VALIDATION_SCHEMA_REVISION = 1
RUNTIME_BUNDLE_VALIDATION_SCHEMA_TOKEN = schema_token(
    RUNTIME_BUNDLE_VALIDATION_SCHEMA_NAME, RUNTIME_BUNDLE_VALIDATION_SCHEMA_REVISION
)
DEFAULT_WEIGHTS_FILENAME = "weights.bin"
DEFAULT_WEIGHTS_MANIFEST_FILENAME = "weights.manifest.json"
DEFAULT_DETECTOR_CONTRACT_FILENAME = "detector_contract.json"
DEFAULT_PILOT_PROFILES_FILENAME = "pilot_profiles.json"
DEFAULT_SHA256SUMS_FILENAME = "sha256sums.txt"
REQUIRED_RUNTIME_BUNDLE_FILES = (
    DEFAULT_DETECTOR_CONTRACT_FILENAME,
    DEFAULT_PILOT_PROFILES_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    DEFAULT_WEIGHTS_MANIFEST_FILENAME,
    DEFAULT_SHA256SUMS_FILENAME,
)
CURRENT_PILOT_PROFILES_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "weight_coordinate_system",
        "input_coordinate_system",
        "input_preprocessing",
        "receiver_profile_id",
        "receiver_channel_id_namespace",
        "detector_contract_sha256",
        "weights_sha256",
        "profiles",
    }
)
CURRENT_WEIGHTS_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "weight_format",
        "weight_coordinate_system",
        "input_coordinate_system",
        "input_preprocessing",
        "detector_contract_sha256",
        "receiver_profile_path",
        "receiver_profile_hash",
        "detector_core_profile_path",
        "physical_channels",
        "num_profiles",
        "weight_profile_shape",
        "weight_profile_dtype",
        "weight_profile_nbytes",
        "weights_sha256",
        "target_reference_layout",
    }
)
CURRENT_RUNTIME_PROFILE_REQUIRED_FIELDS = frozenset(
    {
        "physical_channel",
        "receiver_channel_id",
        "receiver_channel_id_namespace",
        "chime_channel_id",
        "chord_channel_id",
        "pilot_frequency_hz",
        "coarse_channel_center_hz",
        "coarse_channel_index",
        "weight_bank_index",
        "weight_bank_offset_bytes",
        "weight_bank_nbytes",
        "target_norm_sq",
        "reference_norm_sum_sq",
        "null_power_ratio",
        "positive_excess_half_threshold_num",
        "positive_excess_half_threshold_den",
        "reference_placement_status",
        "placement_warnings",
        "fine_calibration",
    }
)


def _reference_placement_summary_from_layouts(
    *,
    core: Any,
    layouts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    adaptive = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("adaptive_reference_placement", False))
    ]
    dc_shifted = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("dc_reference_shifted", False))
    ]
    edge_wrapped = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("edge_reference_wrapped", False))
    ]
    skipped_guard = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("forbidden_tone_in_skipped_guard", False))
    ]
    statuses = sorted({str(row.get("reference_placement_status", "")) for row in layouts})
    status = statuses[0] if len(statuses) == 1 else "mixed:" + ";".join(statuses)
    return {
        "reference_offset_bins": int(core.reference_offset_bins),
        "skipped_guard_bins": int(core.skipped_guard_bins),
        "reference_placement_status": status,
        "num_channels_with_adaptive_reference": int(len(adaptive)),
        "channels_with_adaptive_reference": adaptive,
        "num_dc_shifted_references": int(len(dc_shifted)),
        "channels_with_dc_shifted_reference": dc_shifted,
        "num_edge_wrapped_references": int(len(edge_wrapped)),
        "channels_with_edge_wrapped_reference": edge_wrapped,
        "num_forbidden_tone_in_skipped_guard": int(len(skipped_guard)),
        "channels_with_forbidden_tone_in_skipped_guard": skipped_guard,
        "forbidden_tone_policy": {
            "forbidden_tone": "physical_data_dc",
        },
    }


def _write_sha256sums(output_dir: Path, filenames: Sequence[str]) -> Path:
    path = output_dir / DEFAULT_SHA256SUMS_FILENAME
    lines = []
    for name in filenames:
        digest = file_sha256(output_dir / name)
        lines.append(f"{digest}  {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _add_error(errors: list[dict[str, str]], check: str, message: str) -> None:
    errors.append({"severity": "error", "check": str(check), "message": str(message)})


def _validate_required_fields(
    payload: dict[str, Any],
    *,
    required: frozenset[str],
    label: str,
    errors: list[dict[str, str]],
) -> None:
    missing = sorted(required - set(payload))
    if missing:
        _add_error(
            errors,
            f"{label}.required_fields",
            f"missing required current-schema fields: {missing}",
        )


def _load_json_object(path: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
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


def _read_sha256sums(path: Path, errors: list[dict[str, str]]) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _add_error(errors, "required_file.sha256sums.txt", f"missing {path}")
        return {}
    out: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            _add_error(
                errors,
                "sha256sums.format",
                f"invalid line {line_number}: {line!r}",
            )
            continue
        digest, filename = parts
        out[filename.strip()] = digest.strip()
    return out


def _validate_expected_sha256sums(
    *,
    bundle_dir: Path,
    expected: dict[str, str],
    errors: list[dict[str, str]],
) -> None:
    for filename in REQUIRED_RUNTIME_BUNDLE_FILES:
        if filename == DEFAULT_SHA256SUMS_FILENAME:
            continue
        path = bundle_dir / filename
        if not path.exists():
            continue
        expected_digest = expected.get(filename)
        if expected_digest is None:
            _add_error(
                errors,
                f"sha256sums.{filename}",
                f"{filename} is missing from sha256sums.txt",
            )
            continue
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            _add_error(
                errors,
                f"sha256sums.{filename}",
                f"sha256 mismatch for {filename}: {actual_digest} != {expected_digest}",
            )


def _validate_weight_profile_geometry(
    *,
    detector_contract: dict[str, Any],
    weights_manifest: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    """Check manifest weight-profile geometry against the detector contract.

    The window length is a per-receiver quantity, so a bundle whose manifest
    shape disagrees with its own detector contract mixes two geometries and
    must be rejected.
    """
    shape = weights_manifest.get("weight_profile_shape")
    if not (
        isinstance(shape, list)
        and len(shape) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in shape
        )
    ):
        _add_error(
            errors,
            "weights_manifest.weight_profile_shape",
            "weight_profile_shape must contain two integers",
        )
        return
    for field, shape_index, description in (
        ("num_weight_terms", 0, "terms"),
        ("detector_window_samples", 1, "window length"),
    ):
        contract_value = detector_contract.get(field)
        if contract_value is None:
            continue
        try:
            expected_value = _coerce_int_metadata(
                contract_value,
                field=f"detector_contract.{field}",
            )
        except (TypeError, ValueError) as exc:
            _add_error(errors, f"detector_contract.{field}", str(exc))
            continue
        if int(shape[shape_index]) != expected_value:
            _add_error(
                errors,
                "weights_manifest.weight_profile_shape",
                f"weight_profile_shape {description} {shape[shape_index]} "
                f"does not match detector_contract {field} {expected_value}",
            )
    profile_nbytes = weights_manifest.get("weight_profile_nbytes")
    if isinstance(profile_nbytes, bool) or not isinstance(profile_nbytes, int):
        _add_error(
            errors,
            "weights_manifest.weight_profile_nbytes",
            "weight_profile_nbytes must be an integer",
        )
    elif int(shape[0]) * int(shape[1]) != profile_nbytes:
        _add_error(
            errors,
            "weights_manifest.weight_profile_nbytes",
            f"weight_profile_nbytes {profile_nbytes} does not match "
            f"weight_profile_shape {shape[0]}x{shape[1]} int8 profiles",
        )


def _validate_runtime_profile_offsets(
    *,
    bundle_dir: Path,
    profiles: list[Any],
    expected_profile_nbytes: int,
    errors: list[dict[str, str]],
) -> None:
    weights_path = bundle_dir / DEFAULT_WEIGHTS_FILENAME
    weights_size = weights_path.stat().st_size if weights_path.exists() else 0
    ranges: list[tuple[int, int, int]] = []
    for index, row in enumerate(profiles):
        if not isinstance(row, dict):
            _add_error(
                errors,
                "pilot_profiles.profiles",
                f"profile row {index} is not an object",
            )
            continue
        missing = sorted(CURRENT_RUNTIME_PROFILE_REQUIRED_FIELDS - set(row))
        if missing:
            _add_error(
                errors,
                "pilot_profiles.profile_required_fields",
                f"profile row {index} is missing current-schema fields: {missing}",
            )
        try:
            offset = _coerce_int_metadata(
                row.get("weight_bank_offset_bytes"),
                field=f"profiles[{index}].weight_bank_offset_bytes",
            )
            nbytes = _coerce_int_metadata(
                row.get("weight_bank_nbytes"),
                field=f"profiles[{index}].weight_bank_nbytes",
            )
        except (KeyError, TypeError, ValueError) as exc:
            _add_error(
                errors,
                "pilot_profiles.profile_offsets",
                f"profile row {index} has invalid offset/size fields: {exc}",
            )
            continue
        if expected_profile_nbytes > 0 and offset % expected_profile_nbytes != 0:
            _add_error(
                errors,
                "pilot_profiles.profile_offset_alignment",
                f"profile row {index} offset {offset} is not aligned to "
                f"{expected_profile_nbytes} bytes",
            )
        if 0 < expected_profile_nbytes != nbytes:
            _add_error(
                errors,
                "pilot_profiles.profile_nbytes",
                f"profile row {index} nbytes {nbytes} does not match "
                f"{expected_profile_nbytes}",
            )
        if offset < 0 or nbytes <= 0 or offset + nbytes > weights_size:
            _add_error(
                errors,
                "pilot_profiles.profile_bounds",
                f"profile row {index} range [{offset}, {offset + nbytes}) "
                f"is outside weights.bin size {weights_size}",
            )
        if not str(row.get("reference_placement_status", "")).strip():
            _add_error(
                errors,
                "pilot_profiles.reference_placement_status",
                f"profile row {index} is missing reference_placement_status",
            )
        norm_fields = {
            "target_norm_sq",
            "reference_norm_sum_sq",
            "null_power_ratio",
            "positive_excess_half_threshold_num",
            "positive_excess_half_threshold_den",
        }
        if norm_fields <= set(row):
            if 0 <= offset and offset + nbytes <= weights_size and nbytes > 0:
                raw = weights_path.read_bytes()[offset : offset + nbytes]
                try:
                    packed = np.frombuffer(raw, dtype=np.int8).reshape(3, -1)
                except ValueError as exc:
                    _add_error(
                        errors,
                        "pilot_profiles.weight_shape",
                        f"profile row {index} weight bytes are invalid: {exc}",
                    )
                    continue
                got_nt, got_nl, got_nu = weight_term_norms_sq(packed)
                got_nrs = int(got_nl + got_nu)
                declared = {
                    "target_norm_sq": got_nt,
                    "reference_norm_sum_sq": got_nrs,
                    "positive_excess_half_threshold_num": got_nt,
                    "positive_excess_half_threshold_den": got_nrs,
                }
                for key, expected_value in declared.items():
                    try:
                        matches = _coerce_int_metadata(
                            row[key],
                            field=f"profiles[{index}].{key}",
                        ) == int(expected_value)
                    except (TypeError, ValueError):
                        matches = False
                    if not matches:
                        _add_error(
                            errors,
                            f"pilot_profiles.{key}",
                            f"profile row {index} declares {key}="
                            f"{row[key]!r} but the bundled weights give "
                            f"{expected_value}",
                        )
                if got_nrs > 0:
                    expected_null_power_ratio = null_power_ratio_from_weight_norms(got_nt, got_nrs)
                    try:
                        null_ratio_matches = (
                            abs(
                                float(row["null_power_ratio"])
                                - expected_null_power_ratio
                            )
                            <= 1e-12
                        )
                    except (TypeError, ValueError):
                        null_ratio_matches = False
                    if not null_ratio_matches:
                        _add_error(
                            errors,
                            "pilot_profiles.null_power_ratio",
                            f"profile row {index} declares null_power_ratio={row['null_power_ratio']!r} "
                            f"but the bundled weights give {expected_null_power_ratio!r}",
                        )
        ranges.append((offset, offset + nbytes, index))
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if previous[1] > current[0]:
            _add_error(
                errors,
                "pilot_profiles.profile_overlap",
                f"profile row {previous[2]} overlaps profile row {current[2]}",
            )


def _validate_manifest_profile_bindings(
    *,
    profiles: list[Any],
    weights_manifest: dict[str, Any],
    expected_profile_nbytes: int,
    errors: list[dict[str, str]],
) -> None:
    """Bind manifest channel/layout order to weight-bank profile rows."""
    manifest_channels = weights_manifest.get("physical_channels")
    if not isinstance(manifest_channels, list):
        _add_error(
            errors,
            "weights_manifest.physical_channels",
            "physical_channels must be a list",
        )
        return
    target_layouts = weights_manifest.get("target_reference_layout")
    if not isinstance(target_layouts, list):
        _add_error(
            errors,
            "weights_manifest.target_reference_layout",
            "target_reference_layout must be a list",
        )
        return
    if len(manifest_channels) != len(profiles):
        _add_error(
            errors,
            "weights_manifest.physical_channels",
            "physical_channels length does not match profile count",
        )
    if len(target_layouts) != len(manifest_channels):
        _add_error(
            errors,
            "weights_manifest.target_reference_layout",
            "target_reference_layout length does not match physical_channels",
        )

    parsed_manifest_channels: list[int] = []
    for index, value in enumerate(manifest_channels):
        try:
            parsed_manifest_channels.append(
                _coerce_int_metadata(value, field=f"physical_channels[{index}]")
            )
        except (TypeError, ValueError) as exc:
            _add_error(errors, "weights_manifest.physical_channels", str(exc))
    if len(parsed_manifest_channels) == len(manifest_channels) and len(
        set(parsed_manifest_channels)
    ) != len(parsed_manifest_channels):
        _add_error(
            errors,
            "weights_manifest.physical_channels",
            "physical_channels contains duplicates",
        )

    count = min(len(profiles), len(manifest_channels), len(target_layouts))
    for index in range(count):
        profile = profiles[index]
        layout = target_layouts[index]
        if not isinstance(profile, dict) or not isinstance(layout, dict):
            if not isinstance(layout, dict):
                _add_error(
                    errors,
                    "weights_manifest.target_reference_layout",
                    f"layout row {index} is not an object",
                )
            continue
        try:
            manifest_channel = _coerce_int_metadata(
                manifest_channels[index],
                field=f"physical_channels[{index}]",
            )
            profile_channel = _coerce_int_metadata(
                profile.get("physical_channel"),
                field=f"profiles[{index}].physical_channel",
            )
            layout_channel = _coerce_int_metadata(
                layout.get("physical_channel"),
                field=f"target_reference_layout[{index}].physical_channel",
            )
        except (TypeError, ValueError) as exc:
            _add_error(errors, "runtime_bundle.channel_binding", str(exc))
        else:
            if not (
                manifest_channel == profile_channel == layout_channel
            ):
                _add_error(
                    errors,
                    "runtime_bundle.channel_binding",
                    f"row {index} channel identities disagree: manifest="
                    f"{manifest_channel}, profile={profile_channel}, "
                    f"layout={layout_channel}",
                )

        integer_bindings = (
            ("coarse_channel_index", "coarse_channel_index"),
            ("target_norm_sq", "target_norm_sq"),
            ("reference_norm_sum_sq", "reference_norm_sum_sq"),
        )
        for profile_field, layout_field in integer_bindings:
            try:
                profile_value = _coerce_int_metadata(
                    profile.get(profile_field),
                    field=f"profiles[{index}].{profile_field}",
                )
                layout_value = _coerce_int_metadata(
                    layout.get(layout_field),
                    field=f"target_reference_layout[{index}].{layout_field}",
                )
            except (TypeError, ValueError) as exc:
                _add_error(errors, "runtime_bundle.layout_binding", str(exc))
                continue
            if profile_value != layout_value:
                _add_error(
                    errors,
                    "runtime_bundle.layout_binding",
                    f"row {index} {profile_field} disagrees: profile="
                    f"{profile_value}, layout={layout_value}",
                )

        float_bindings = (
            ("pilot_frequency_hz", "dtv_pilot_hz"),
            ("coarse_channel_center_hz", "coarse_channel_center_hz"),
            ("null_power_ratio", "null_power_ratio"),
        )
        for profile_field, layout_field in float_bindings:
            try:
                profile_value = float(profile[profile_field])
                layout_value = float(layout[layout_field])
            except (KeyError, TypeError, ValueError) as exc:
                _add_error(
                    errors,
                    "runtime_bundle.layout_binding",
                    f"row {index} has invalid {profile_field}/{layout_field}: {exc}",
                )
                continue
            if not (
                math.isfinite(profile_value)
                and math.isfinite(layout_value)
                and math.isclose(profile_value, layout_value, rel_tol=0.0, abs_tol=1e-12)
            ):
                _add_error(
                    errors,
                    "runtime_bundle.layout_binding",
                    f"row {index} {profile_field} disagrees: profile="
                    f"{profile_value!r}, layout={layout_value!r}",
                )

        try:
            bank_index = _coerce_int_metadata(
                profile.get("weight_bank_index"),
                field=f"profiles[{index}].weight_bank_index",
            )
        except (TypeError, ValueError) as exc:
            _add_error(errors, "pilot_profiles.weight_bank_index", str(exc))
        else:
            if bank_index != index:
                _add_error(
                    errors,
                    "pilot_profiles.weight_bank_index",
                    f"profile row {index} declares weight_bank_index="
                    f"{bank_index}; expected {index}",
                )

        try:
            offset = _coerce_int_metadata(
                profile.get("weight_bank_offset_bytes"),
                field=f"profiles[{index}].weight_bank_offset_bytes",
            )
        except (TypeError, ValueError) as exc:
            _add_error(
                errors,
                "pilot_profiles.weight_bank_offset_order",
                str(exc),
            )
        else:
            if expected_profile_nbytes > 0:
                expected_offset = index * expected_profile_nbytes
                if offset != expected_offset:
                    _add_error(
                        errors,
                        "pilot_profiles.weight_bank_offset_order",
                        f"profile row {index} offset {offset} does not match "
                        f"ordered weight-bank offset {expected_offset}",
                    )


FINE_CALIBRATION_STATUS_PENDING = "pending_campaign"
FINE_CALIBRATION_STATUS_CALIBRATED = "calibrated"
FINE_CALIBRATION_DEFAULT_HALF_WIDTH = 2  # survey window convention
FINE_CALIBRATION_NUM_BINS = 256
FINE_CALIBRATION_GUARD_PADDED_BINS = (
    CFAR_DEFAULT_GUARD_FINE_BINS * FINE_PAD_FACTOR
)


def _validate_fine_calibration_provenance(
    *,
    provenance: object,
    status: str,
    field: str,
    errors: list[dict[str, str]],
) -> None:
    provenance_field = f"{field}.provenance"
    if not isinstance(provenance, dict):
        _add_error(errors, provenance_field, "provenance must be an object")
        return
    required = {"epochs", "null_quantile", "source_product_sha256"}
    missing = sorted(required - set(provenance))
    if missing:
        _add_error(
            errors,
            f"{provenance_field}.required_fields",
            f"missing required fields: {missing}",
        )
    unknown = sorted(set(provenance) - required - {"note"})
    if unknown:
        _add_error(
            errors,
            f"{provenance_field}.unknown_fields",
            f"unknown fields: {unknown}",
        )

    epochs = provenance.get("epochs")
    source_hashes = provenance.get("source_product_sha256")
    null_quantile = provenance.get("null_quantile")
    if not isinstance(epochs, list) or any(
        not isinstance(epoch, str) or not epoch.strip() for epoch in epochs
    ):
        _add_error(
            errors,
            f"{provenance_field}.epochs",
            "epochs must be a list of non-empty strings",
        )
    if not isinstance(source_hashes, list) or any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
        for digest in source_hashes
    ):
        _add_error(
            errors,
            f"{provenance_field}.source_product_sha256",
            "source_product_sha256 must be a list of 64-character hex digests",
        )
    note = provenance.get("note")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        _add_error(
            errors,
            f"{provenance_field}.note",
            "note must be a non-empty string when present",
        )

    if status == FINE_CALIBRATION_STATUS_PENDING:
        if epochs != [] or source_hashes != [] or null_quantile is not None:
            _add_error(
                errors,
                provenance_field,
                "pending calibration provenance must have empty epochs and "
                "source hashes with a null quantile",
            )
        return

    if not epochs:
        _add_error(
            errors,
            f"{provenance_field}.epochs",
            "calibrated provenance must name at least one epoch",
        )
    if not source_hashes:
        _add_error(
            errors,
            f"{provenance_field}.source_product_sha256",
            "calibrated provenance must include at least one source-product hash",
        )
    if isinstance(null_quantile, bool) or not isinstance(null_quantile, (int, float)):
        _add_error(
            errors,
            f"{provenance_field}.null_quantile",
            "calibrated null_quantile must be numeric",
        )
    elif not math.isfinite(float(null_quantile)) or not 0.0 < float(null_quantile) < 1.0:
        _add_error(
            errors,
            f"{provenance_field}.null_quantile",
            "calibrated null_quantile must be finite and strictly between 0 and 1",
        )


def _default_fine_calibration_block() -> dict[str, Any]:
    """Per-channel fine decision calibration, exported pending-campaign.

    The deployed kernel (core 2.3.0, ``FStat_Compute_FusedFineMask_U64``)
    consumes these values as arguments: the decision arithmetic is frozen
    (``fine_decision_v1``), the operating point is bundle data. The
    calibration campaign replaces the null fields and flips ``status`` to
    ``calibrated`` with provenance; until then a consumer must treat the
    channel's fine mask path as not deployable.
    """
    return {
        "status": FINE_CALIBRATION_STATUS_PENDING,
        "decision_version": FINE_DECISION_VERSION,
        "anchor_bin": None,
        "designated_half_width": FINE_CALIBRATION_DEFAULT_HALF_WIDTH,
        "bulk_mask_words_hex": None,
        "cfar_rank": None,
        "cfar_multiplier_q16": None,
        "provenance": {
            "epochs": [],
            "null_quantile": None,
            "source_product_sha256": [],
            "note": (
                "pending calibration campaign: anchors from survey "
                "on-epochs; rank/multiplier from the verified-null "
                "quantile program (docs/DESIGN_DECISIONS.md)"
            ),
        },
    }


def _fine_calibration_int(
    *,
    block: dict[str, Any],
    key: str,
    field: str,
    errors: list[dict[str, str]],
) -> int | None:
    """Return one calibrated integer field without coercing JSON values."""
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _add_error(
            errors,
            f"{field}.{key}",
            f"{key} must be an integer",
        )
        return None
    return value


def _fine_calibration_mask_words(
    *,
    value: object,
    field: str,
    errors: list[dict[str, str]],
) -> list[int] | None:
    """Parse the four JSON string words that encode the 256-bit bulk mask."""
    if not isinstance(value, list):
        _add_error(
            errors,
            field,
            "bulk_mask_words_hex must be a list of four hex strings",
        )
        return None
    if len(value) != 4:
        _add_error(errors, field, "must contain exactly 4 hex words")
        return None
    if any(not isinstance(word, str) for word in value):
        _add_error(errors, field, "every mask word must be a hex string")
        return None
    try:
        words = [int(word, 16) for word in value]
    except ValueError as exc:
        _add_error(errors, field, f"bad hex: {exc}")
        return None
    if any(not 0 <= word < (1 << 64) for word in words):
        _add_error(errors, field, "words must fit in uint64")
        return None
    return words


def _validate_fine_calibration(
    *,
    profiles: list[Any],
    errors: list[dict[str, str]],
) -> None:
    """Validate required per-profile ``fine_calibration`` blocks.

    A block must declare the frozen decision version and a known status; a
    ``calibrated`` block must carry complete, range-checked values whose
    bulk mask is consistent with the designated set (designated bins and
    their guard cannot be bulk bins) and deep enough for the rank.
    """
    for index, row in enumerate(profiles):
        if not isinstance(row, dict):
            continue
        if "fine_calibration" not in row:
            _add_error(
                errors,
                f"pilot_profiles.fine_calibration[{index}]",
                "missing required fine_calibration block",
            )
            continue
        block = row["fine_calibration"]
        field = f"pilot_profiles.fine_calibration[{index}]"
        if not isinstance(block, dict):
            _add_error(errors, field, "fine_calibration is not an object")
            continue
        required_fields = {
            "status",
            "decision_version",
            "anchor_bin",
            "designated_half_width",
            "bulk_mask_words_hex",
            "cfar_rank",
            "cfar_multiplier_q16",
            "provenance",
        }
        missing_fields = sorted(required_fields - set(block))
        if missing_fields:
            _add_error(
                errors,
                f"{field}.required_fields",
                f"missing required fields: {missing_fields}",
            )
        if block.get("decision_version") != FINE_DECISION_VERSION:
            _add_error(
                errors,
                f"{field}.decision_version",
                f"decision_version {block.get('decision_version')!r} does "
                f"not match {FINE_DECISION_VERSION!r}",
            )
        status = block.get("status")
        if status not in (
            FINE_CALIBRATION_STATUS_PENDING,
            FINE_CALIBRATION_STATUS_CALIBRATED,
        ):
            _add_error(
                errors, f"{field}.status", f"unknown status {status!r}"
            )
            continue
        _validate_fine_calibration_provenance(
            provenance=block.get("provenance"),
            status=status,
            field=field,
            errors=errors,
        )
        if status == FINE_CALIBRATION_STATUS_PENDING:
            continue
        anchor = _fine_calibration_int(
            block=block,
            key="anchor_bin",
            field=field,
            errors=errors,
        )
        half_width = _fine_calibration_int(
            block=block,
            key="designated_half_width",
            field=field,
            errors=errors,
        )
        rank = _fine_calibration_int(
            block=block,
            key="cfar_rank",
            field=field,
            errors=errors,
        )
        mult = _fine_calibration_int(
            block=block,
            key="cfar_multiplier_q16",
            field=field,
            errors=errors,
        )
        words = _fine_calibration_mask_words(
            value=block.get("bulk_mask_words_hex"),
            field=f"{field}.bulk_mask_words_hex",
            errors=errors,
        )
        if None in (anchor, half_width, rank, mult) or words is None:
            continue
        if not 0 <= anchor < FINE_CALIBRATION_NUM_BINS:
            _add_error(errors, f"{field}.anchor_bin", f"out of range: {anchor}")
        if not 0 <= half_width < FINE_CALIBRATION_NUM_BINS // 2:
            _add_error(
                errors,
                f"{field}.designated_half_width",
                f"out of range: {half_width}",
            )
        if not 0 < mult < (1 << 64):
            _add_error(
                errors,
                f"{field}.cfar_multiplier_q16",
                f"must be in uint64 range [1, {(1 << 64) - 1}]: {mult}",
            )
        if 0 <= anchor < FINE_CALIBRATION_NUM_BINS:
            popcount = sum(bin(w).count("1") for w in words)
            if not 0 <= rank < popcount:
                _add_error(
                    errors,
                    f"{field}.cfar_rank",
                    f"rank {rank} is not below the bulk population "
                    f"{popcount}",
                )
            if 0 <= half_width < FINE_CALIBRATION_NUM_BINS // 2:
                exclusion_half_width = (
                    half_width + FINE_CALIBRATION_GUARD_PADDED_BINS
                )
                for k in range(-exclusion_half_width, exclusion_half_width + 1):
                    b = (anchor + k) % FINE_CALIBRATION_NUM_BINS
                    if (words[b >> 6] >> (b & 63)) & 1:
                        _add_error(
                            errors,
                            f"{field}.bulk_mask_words_hex",
                            f"designated/guard bin {b} is set in the bulk mask",
                        )
                        break


def _coerce_int_metadata(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    return value


def _validate_receiver_namespace_channel_ids(
    *,
    profiles: list[Any],
    field: str,
    errors: list[dict[str, str]],
) -> None:
    """Check one receiver channel-id field for integer type and uniqueness."""
    seen: set[int] = set()
    for index, row in enumerate(profiles):
        if not isinstance(row, dict):
            continue
        value = row.get(field)
        if value is None:
            continue
        try:
            channel_id = _coerce_int_metadata(
                value,
                field=field,
            )
        except (TypeError, ValueError):
            _add_error(
                errors,
                f"pilot_profiles.{field}",
                f"invalid {field} {value!r} at profile row {index}",
            )
            continue
        if channel_id in seen:
            _add_error(
                errors,
                f"pilot_profiles.{field}",
                f"duplicate {field} {channel_id} at profile row {index}",
            )
        seen.add(channel_id)


RECEIVER_CHANNEL_ID_FIELDS = (
    "chime_channel_id",
    "chord_channel_id",
    "receiver_channel_id",
)
# Receiver channel-id namespaces the exporter knows how to alias into a
# telescope-specific field. The namespace string is declared by the receiver
# profile (metadata.channel_id_map.namespace).
CHANNEL_ID_NAMESPACE_FIELD_ALIASES = {
    "chime_freq_id": "chime_channel_id",
    "chord_freq_id": "chord_channel_id",
}


def _validate_receiver_channel_ids(
    *,
    profiles: list[Any],
    receiver_channel_id_namespace: object,
    errors: list[dict[str, str]],
) -> None:
    namespace_is_valid = receiver_channel_id_namespace is None or (
        isinstance(receiver_channel_id_namespace, str)
        and bool(receiver_channel_id_namespace.strip())
    )
    if not namespace_is_valid:
        _add_error(
            errors,
            "pilot_profiles.receiver_channel_id_namespace",
            "receiver_channel_id_namespace must be null or a non-empty string",
        )
    for field in RECEIVER_CHANNEL_ID_FIELDS:
        _validate_receiver_namespace_channel_ids(
            profiles=profiles,
            field=field,
            errors=errors,
        )
    for index, row in enumerate(profiles):
        if not isinstance(row, dict):
            continue
        row_namespace = row.get("receiver_channel_id_namespace")
        if row_namespace != receiver_channel_id_namespace:
            _add_error(
                errors,
                "pilot_profiles.receiver_channel_id_namespace",
                f"profile row {index} namespace {row_namespace!r} does not match "
                f"the top-level namespace {receiver_channel_id_namespace!r}",
            )

        if receiver_channel_id_namespace is None:
            populated = [
                field
                for field in RECEIVER_CHANNEL_ID_FIELDS
                if row.get(field) is not None
            ]
            if populated:
                _add_error(
                    errors,
                    "pilot_profiles.receiver_channel_binding",
                    f"profile row {index} has null receiver namespace but "
                    f"populates channel-id fields {populated}",
                )
            continue

        if receiver_channel_id_namespace == "chord_freq_id":
            try:
                coarse_channel_index = _coerce_int_metadata(
                    row.get("coarse_channel_index"),
                    field=f"profiles[{index}].coarse_channel_index",
                )
                receiver_channel_id = _coerce_int_metadata(
                    row.get("receiver_channel_id"),
                    field=f"profiles[{index}].receiver_channel_id",
                )
                chord_channel_id = _coerce_int_metadata(
                    row.get("chord_channel_id"),
                    field=f"profiles[{index}].chord_channel_id",
                )
            except (TypeError, ValueError) as exc:
                _add_error(
                    errors,
                    "pilot_profiles.chord_channel_binding",
                    str(exc),
                )
            else:
                if not (
                    receiver_channel_id
                    == chord_channel_id
                    == coarse_channel_index
                ):
                    _add_error(
                        errors,
                        "pilot_profiles.chord_channel_binding",
                        f"profile row {index} requires receiver_channel_id and "
                        "chord_channel_id to equal coarse_channel_index; got "
                        f"receiver={receiver_channel_id}, chord={chord_channel_id}, "
                        f"coarse={coarse_channel_index}",
                    )
            if row.get("chime_channel_id") is not None:
                _add_error(
                    errors,
                    "pilot_profiles.chord_channel_binding",
                    f"profile row {index} in chord_freq_id namespace must have "
                    "a null chime_channel_id",
                )

        receiver_id = row.get("receiver_channel_id")
        if receiver_id is None:
            continue
        for alias_field in ("chime_channel_id", "chord_channel_id"):
            alias_value = row.get(alias_field)
            if alias_value is None:
                continue
            try:
                same = _coerce_int_metadata(
                    alias_value, field=alias_field
                ) == _coerce_int_metadata(
                    receiver_id, field="receiver_channel_id"
                )
            except (TypeError, ValueError):
                continue
            if not same:
                _add_error(
                    errors,
                    f"pilot_profiles.{alias_field}",
                    f"{alias_field} {alias_value!r} does not match "
                    f"receiver_channel_id {receiver_id!r} at profile row "
                    f"{index}",
                )


def _resolve_profile_channel_id_map(profile: Any) -> tuple[str, int] | None:
    """Return (namespace, offset_from_coarse_channel_index) when declared.

    A receiver profile opts into runtime-bundle channel-id population by
    declaring ``metadata.channel_id_map`` with a ``namespace`` string and an
    integer ``offset_from_coarse_channel_index`` such that::

        receiver_channel_id = coarse_channel_index + offset

    For the CHORD/kotekan integration the namespace is ``chord_freq_id`` with
    offset 0 (kotekan ``chordMetadata.coarse_freq`` entries are
    ``CHORDTelescope::to_freq_id`` values, which equal the profile's
    coarse-channel index in the first Nyquist zone). Profiles without the
    declaration keep the legacy behavior: channel-id fields exported as null
    pending a verified metadata mapping.
    """
    metadata = getattr(profile, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    channel_id_map = metadata.get("channel_id_map")
    if not isinstance(channel_id_map, dict):
        return None
    namespace = channel_id_map.get("namespace")
    offset = channel_id_map.get("offset_from_coarse_channel_index")
    if namespace is None or offset is None:
        raise ValueError(
            "receiver profile metadata.channel_id_map requires both "
            "'namespace' and integer 'offset_from_coarse_channel_index'; "
            f"got {channel_id_map!r}"
        )
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError(
            "receiver profile metadata.channel_id_map.namespace must be a "
            "non-empty string"
        )
    try:
        parsed_offset = _coerce_int_metadata(
            offset,
            field="metadata.channel_id_map.offset_from_coarse_channel_index",
        )
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    return namespace, parsed_offset


def _validate_bundle_coordinate_systems(
    *,
    detector_contract: dict[str, Any],
    pilot_profiles: dict[str, Any],
    weights_manifest: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    values = {
        "detector_contract": detector_contract.get("weight_coordinate_system"),
        "pilot_profiles": pilot_profiles.get("weight_coordinate_system"),
        "weights_manifest": weights_manifest.get("weight_coordinate_system"),
    }
    normalized: dict[str, str] = {}
    for label, value in values.items():
        if value is None:
            _add_error(
                errors,
                f"{label}.weight_coordinate_system",
                "missing weight_coordinate_system",
            )
            continue
        try:
            normalized[label] = normalize_weight_coordinate_system(value)
        except ValueError as exc:
            _add_error(errors, f"{label}.weight_coordinate_system", str(exc))
    coordinate_system = next(iter(normalized.values())) if normalized else None
    if len(set(normalized.values())) > 1:
        _add_error(
            errors,
            "weight_coordinate_system.consistency",
            "detector_contract, pilot_profiles, and weights_manifest disagree: "
            f"{values}",
        )

    input_coordinate_values = {
        "detector_contract": detector_contract.get("input_coordinate_system"),
        "pilot_profiles": pilot_profiles.get("input_coordinate_system"),
        "weights_manifest": weights_manifest.get("input_coordinate_system"),
    }
    present_input_coordinates: dict[str, str] = {}
    for label, value in input_coordinate_values.items():
        if value is None:
            _add_error(
                errors,
                f"{label}.input_coordinate_system",
                "missing input_coordinate_system",
            )
        else:
            present_input_coordinates[label] = str(value)
    if len(set(present_input_coordinates.values())) > 1:
        _add_error(
            errors,
            "input_coordinate_system.consistency",
            "detector_contract, pilot_profiles, and weights_manifest disagree: "
            f"{input_coordinate_values}",
        )
    if coordinate_system is not None:
        expected_coordinate = input_coordinate_system_for_weight_coordinate(
            coordinate_system
        )
        for label, value in present_input_coordinates.items():
            if value != expected_coordinate:
                _add_error(
                    errors,
                    f"{label}.input_coordinate_system",
                    f"input_coordinate_system {value!r} does not match "
                    f"{expected_coordinate!r} for weight_coordinate_system "
                    f"{coordinate_system!r}",
                )

    preprocessing_values = {
        "detector_contract": detector_contract.get("input_preprocessing"),
        "pilot_profiles": pilot_profiles.get("input_preprocessing"),
        "weights_manifest": weights_manifest.get("input_preprocessing"),
    }
    time_reverse_values: dict[str, bool] = {}
    for label, preprocessing in preprocessing_values.items():
        if not isinstance(preprocessing, dict):
            _add_error(
                errors,
                f"{label}.input_preprocessing",
                "input_preprocessing is missing or not an object",
            )
            continue
        if "time_reverse_detector_windows_before_kernel" not in preprocessing:
            _add_error(
                errors,
                f"{label}.input_preprocessing.time_reverse",
                "missing time_reverse_detector_windows_before_kernel",
            )
            continue
        time_reverse_values[label] = bool(
            preprocessing["time_reverse_detector_windows_before_kernel"]
        )
    if len(set(time_reverse_values.values())) > 1:
        _add_error(
            errors,
            "input_preprocessing.time_reverse.consistency",
            "detector_contract, pilot_profiles, and weights_manifest disagree: "
            f"{time_reverse_values}",
        )
    if (
        coordinate_system == WEIGHT_COORDINATE_RAW_INPUT
        and any(time_reverse_values.values())
    ):
        _add_error(
            errors,
            "input_preprocessing.time_reverse",
            "raw input-coordinate weights must not request detector-window "
            "time reversal before the kernel",
        )


def validate_runtime_weight_bundle(
    *,
    bundle_dir: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Validate a runtime weight bundle for deployment use."""
    bundle = Path(bundle_dir)
    errors: list[dict[str, str]] = []
    for filename in REQUIRED_RUNTIME_BUNDLE_FILES:
        if not (bundle / filename).exists():
            _add_error(
                errors,
                f"required_file.{filename}",
                f"missing {bundle / filename}",
            )

    detector_contract = _load_json_object(
        bundle / DEFAULT_DETECTOR_CONTRACT_FILENAME,
        errors,
    )
    pilot_profiles = _load_json_object(
        bundle / DEFAULT_PILOT_PROFILES_FILENAME,
        errors,
    )
    weights_manifest = _load_json_object(
        bundle / DEFAULT_WEIGHTS_MANIFEST_FILENAME,
        errors,
    )
    expected_sums = _read_sha256sums(bundle / DEFAULT_SHA256SUMS_FILENAME, errors)
    _validate_expected_sha256sums(
        bundle_dir=bundle,
        expected=expected_sums,
        errors=errors,
    )

    _validate_required_fields(
        pilot_profiles,
        required=CURRENT_PILOT_PROFILES_REQUIRED_FIELDS,
        label="pilot_profiles",
        errors=errors,
    )
    _validate_required_fields(
        weights_manifest,
        required=CURRENT_WEIGHTS_MANIFEST_REQUIRED_FIELDS,
        label="weights_manifest",
        errors=errors,
    )
    try:
        validate_detector_contract(detector_contract)
    except (TypeError, ValueError) as exc:
        _add_error(errors, "detector_contract.current_schema", str(exc))

    if pilot_profiles.get("schema_version") != RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN:
        _add_error(
            errors,
            "pilot_profiles.schema_version",
            f"schema_version {pilot_profiles.get('schema_version')!r} does not "
            f"match {RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN!r}",
        )
    if (
        weights_manifest.get("schema_version")
        != RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN
    ):
        _add_error(
            errors,
            "weights_manifest.schema_version",
            f"schema_version {weights_manifest.get('schema_version')!r} does not "
            f"match {RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN!r}",
        )
    _validate_weight_profile_geometry(
        detector_contract=detector_contract,
        weights_manifest=weights_manifest,
        errors=errors,
    )

    contract_digest = detector_contract_sha256(detector_contract)
    weights_path = bundle / DEFAULT_WEIGHTS_FILENAME
    weights_digest = file_sha256(weights_path) if weights_path.exists() else ""
    for label, payload in (
        ("pilot_profiles", pilot_profiles),
        ("weights_manifest", weights_manifest),
    ):
        if payload.get("detector_contract_sha256") != contract_digest:
            _add_error(
                errors,
                f"{label}.detector_contract_sha256",
                f"detector_contract_sha256 does not match detector_contract.json "
                f"for {label}",
            )
        if payload.get("weights_sha256") != weights_digest:
            _add_error(
                errors,
                f"{label}.weights_sha256",
                f"weights_sha256 does not match weights.bin for {label}",
            )

    profiles_raw = pilot_profiles.get("profiles", [])
    profiles = profiles_raw if isinstance(profiles_raw, list) else []
    if not isinstance(profiles_raw, list):
        _add_error(errors, "pilot_profiles.profiles", "profiles is not a list")
    elif not profiles:
        _add_error(
            errors,
            "pilot_profiles.profiles",
            "current runtime bundles must contain at least one profile",
        )
    manifest_num_profiles = weights_manifest.get("num_profiles")
    if manifest_num_profiles is not None:
        try:
            manifest_num_profiles_int = _coerce_int_metadata(
                manifest_num_profiles,
                field="num_profiles",
            )
        except (TypeError, ValueError) as exc:
            _add_error(errors, "weights_manifest.num_profiles", str(exc))
            manifest_num_profiles_int = None
        if (
            manifest_num_profiles_int is not None
            and manifest_num_profiles_int != len(profiles)
        ):
            _add_error(
                errors,
                "weights_manifest.num_profiles",
                f"num_profiles {manifest_num_profiles!r} does not match "
                f"profile count {len(profiles)}",
            )
    try:
        expected_profile_nbytes = _coerce_int_metadata(
            weights_manifest.get("weight_profile_nbytes", 0),
            field="weight_profile_nbytes",
        )
    except (TypeError, ValueError):
        expected_profile_nbytes = 0
        _add_error(
            errors,
            "weights_manifest.weight_profile_nbytes",
            "weight_profile_nbytes must be an integer",
        )
    _validate_manifest_profile_bindings(
        profiles=profiles,
        weights_manifest=weights_manifest,
        expected_profile_nbytes=expected_profile_nbytes,
        errors=errors,
    )
    _validate_runtime_profile_offsets(
        bundle_dir=bundle,
        profiles=profiles,
        expected_profile_nbytes=expected_profile_nbytes,
        errors=errors,
    )
    _validate_receiver_channel_ids(
        profiles=profiles,
        receiver_channel_id_namespace=pilot_profiles.get(
            "receiver_channel_id_namespace"
        ),
        errors=errors,
    )
    _validate_fine_calibration(profiles=profiles, errors=errors)
    _validate_bundle_coordinate_systems(
        detector_contract=detector_contract,
        pilot_profiles=pilot_profiles,
        weights_manifest=weights_manifest,
        errors=errors,
    )

    report = {
        "schema_version": RUNTIME_BUNDLE_VALIDATION_SCHEMA_TOKEN,
        "bundle_dir": str(bundle),
        "valid": len(errors) == 0,
        "num_errors": int(len(errors)),
        "errors": errors,
    }
    if output_json is not None:
        write_json_strict(Path(output_json), report, indent=2, sort_keys=True)
    return report


def export_runtime_weight_bundle(
    *,
    receiver_profile_path: Path,
    detector_core_profile_path: Path,
    physical_channels: Sequence[int],
    weight_coordinate_system: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write a compact runtime detector bundle for selected DTV pilots."""
    profile = load_receiver_profile(receiver_profile_path)
    core = load_detector_core_profile(detector_core_profile_path)
    validate_integration_compatibility(profile=profile, detector_core=core)
    # The receiver profile owns the deployed window length (K); the core
    # declares which lengths the compiled kernel family supports.
    core = core.with_detector_window_samples(int(profile.detector_window_samples))
    coordinate_system = normalize_weight_coordinate_system(weight_coordinate_system)
    channels = [
        validate_uhf_physical_channel(channel)
        for channel in physical_channels
    ]
    if not channels:
        raise ValueError("at least one physical channel must be selected")

    table, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=channels,
        weight_coordinate_system=coordinate_system,
    )
    layout_by_channel = {int(row["physical_channel"]): row for row in layouts}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    profile_nbytes = int(core.num_weight_terms) * int(core.detector_window_samples)
    profile_nbytes *= np.dtype(np.int8).itemsize
    weights_path = output / DEFAULT_WEIGHTS_FILENAME
    channel_id_map = _resolve_profile_channel_id_map(profile)
    channel_id_namespace = channel_id_map[0] if channel_id_map else None
    channel_id_offset = channel_id_map[1] if channel_id_map else 0
    channel_id_alias_field = (
        CHANNEL_ID_NAMESPACE_FIELD_ALIASES.get(channel_id_namespace)
        if channel_id_namespace is not None
        else None
    )
    profile_rows: list[dict[str, Any]] = []
    weight_chunks: list[bytes] = []
    offset = 0
    for index, channel in enumerate(channels):
        layout = layout_by_channel[int(channel)]
        coarse_index = int(layout["coarse_channel_index"])
        weights = np.ascontiguousarray(table[coarse_index], dtype=np.int8)
        payload = weights.tobytes(order="C")
        if len(payload) != profile_nbytes:
            raise ValueError(
                f"unexpected weight profile size for channel {channel}: "
                f"{len(payload)} != {profile_nbytes}"
            )
        weight_chunks.append(payload)
        row_nt, row_nl, row_nu = weight_term_norms_sq(weights)
        row_nrs = int(row_nl + row_nu)
        receiver_channel_id = (
            coarse_index + channel_id_offset if channel_id_map else None
        )
        channel_id_fields: dict[str, Any] = {
            # Telescope channel identifiers for first-frame profile selection
            # in a receiver pipeline stage (for example the kotekan
            # cudaPilotProxyDetector). Populated only when the receiver
            # profile declares metadata.channel_id_map; otherwise null,
            # pending a verified metadata mapping.
            "receiver_channel_id": receiver_channel_id,
            "receiver_channel_id_namespace": channel_id_namespace,
            "chime_channel_id": None,
            "chord_channel_id": None,
        }
        if channel_id_alias_field is not None and receiver_channel_id is not None:
            channel_id_fields[channel_id_alias_field] = receiver_channel_id
        profile_rows.append(
            {
                "physical_channel": int(channel),
                **channel_id_fields,
                "pilot_frequency_hz": float(layout["dtv_pilot_hz"]),
                "coarse_channel_center_hz": float(
                    layout["coarse_channel_center_hz"]
                ),
                "coarse_channel_index": coarse_index,
                "weight_bank_index": int(index),
                "weight_bank_offset_bytes": int(offset),
                "weight_bank_nbytes": int(profile_nbytes),
                # Exact integer weight-norm zero-point for this channel. The
                # kernel's rational half-threshold half_num:half_den = nt:(nl+nu)
                # sets the mask threshold at F = null_power_ratio (the norm-corrected
                # positive-excess rule) with zero kernel changes.
                "target_norm_sq": int(row_nt),
                "reference_norm_sum_sq": row_nrs,
                "null_power_ratio": float(null_power_ratio_from_weight_norms(row_nt, row_nrs)),
                "positive_excess_half_threshold_num": int(row_nt),
                "positive_excess_half_threshold_den": row_nrs,
                "reference_placement_status": str(
                    layout["reference_placement_status"]
                ),
                "placement_warnings": str(layout["placement_warnings"]),
                # Fine decision calibration (kernel core 2.3.0): frozen
                # arithmetic, data-driven operating point; exported
                # pending until the campaign supplies measured values.
                "fine_calibration": _default_fine_calibration_block(),
            }
        )
        offset += profile_nbytes
    weights_path.write_bytes(b"".join(weight_chunks))

    reference_placement_summary = _reference_placement_summary_from_layouts(
        core=core,
        layouts=layouts,
    )
    detector_contract = build_detector_contract(
        detector_window_samples=int(core.detector_window_samples),
        skipped_guard_bins=int(core.skipped_guard_bins),
        reference_offset_bins=int(core.reference_offset_bins),
        num_weight_terms=int(core.num_weight_terms),
        sample_bits_per_component=int(core.sample_bits_per_component),
        input_format=core.input_format,
        power_accumulator=core.power_accumulator,
        weight_coordinate_system=coordinate_system,
        input_coordinate_system=input_coordinate_system_for_weight_coordinate(
            coordinate_system
        ),
        time_reverse_detector_windows_before_kernel=(
            profile_requires_window_time_reversal(profile, coordinate_system)
        ),
        reference_placement_summary=reference_placement_summary,
    )
    contract_path = output / DEFAULT_DETECTOR_CONTRACT_FILENAME
    write_json_strict(contract_path, detector_contract, indent=2, sort_keys=True)
    contract_digest = detector_contract_sha256(detector_contract)
    weights_digest = file_sha256(weights_path)

    weights_manifest = {
        "schema_version": RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN,
        "weight_format": "concatenated_int8_weight_profiles_v1",
        "weight_coordinate_system": coordinate_system,
        "input_coordinate_system": detector_contract["input_coordinate_system"],
        "input_preprocessing": detector_contract["input_preprocessing"],
        "detector_contract_sha256": contract_digest,
        "receiver_profile_path": str(receiver_profile_path),
        "receiver_profile_hash": receiver_profile_hash(profile),
        "detector_core_profile_path": str(detector_core_profile_path),
        "physical_channels": channels,
        "num_profiles": int(len(channels)),
        "weight_profile_shape": [
            int(core.num_weight_terms),
            int(core.detector_window_samples),
        ],
        "weight_profile_dtype": "int8",
        "weight_profile_nbytes": int(profile_nbytes),
        "weights_sha256": weights_digest,
        "target_reference_layout": layouts,
    }
    weights_manifest_path = output / DEFAULT_WEIGHTS_MANIFEST_FILENAME
    write_json_strict(weights_manifest_path, weights_manifest, indent=2, sort_keys=True)

    pilot_profiles = {
        "schema_version": RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN,
        "weight_coordinate_system": coordinate_system,
        "input_coordinate_system": detector_contract["input_coordinate_system"],
        "input_preprocessing": detector_contract["input_preprocessing"],
        "receiver_profile_id": profile.receiver_profile_id,
        "receiver_channel_id_namespace": channel_id_namespace,
        "detector_contract_sha256": contract_digest,
        "weights_sha256": weights_digest,
        "profiles": profile_rows,
    }
    pilot_profiles_path = output / DEFAULT_PILOT_PROFILES_FILENAME
    write_json_strict(pilot_profiles_path, pilot_profiles, indent=2, sort_keys=True)

    sha256sums_path = _write_sha256sums(
        output,
        [
            DEFAULT_DETECTOR_CONTRACT_FILENAME,
            DEFAULT_PILOT_PROFILES_FILENAME,
            DEFAULT_WEIGHTS_FILENAME,
            DEFAULT_WEIGHTS_MANIFEST_FILENAME,
        ],
    )
    return {
        "detector_contract": contract_path,
        "pilot_profiles": pilot_profiles_path,
        "weights": weights_path,
        "weights_manifest": weights_manifest_path,
        "sha256sums": sha256sums_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a compact runtime detector weight bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--receiver-profile", type=Path, required=True)
    parser.add_argument("--detector-core-profile", type=Path, required=True)
    parser.add_argument(
        "--weight-coordinate-system",
        choices=[
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            WEIGHT_COORDINATE_RAW_INPUT,
        ],
        required=True,
    )
    parser.add_argument("--physical-channel", type=int, action="append", default=None)
    parser.add_argument("--physical-channel-range", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    physical_channels = parse_physical_channel_selection(
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
    )
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=args.receiver_profile,
        detector_core_profile_path=args.detector_core_profile,
        physical_channels=physical_channels,
        weight_coordinate_system=args.weight_coordinate_system,
        output_dir=args.output_dir,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

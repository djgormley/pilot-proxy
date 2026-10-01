# coding=utf-8
"""Chunked CHIME real-data runner for the locked PilotProxy detector."""

from __future__ import annotations

import argparse
import operator
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from pilot_proxy.detect import detect_packed_detector_input
from pilot_proxy.detector_contract import (
    CHIME_RUN_CONFIG_SCHEMA_TOKEN,
    CHIME_STATS_SCHEMA_TOKEN,
    COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    build_detector_contract,
    input_coordinate_system_for_weight_coordinate,
    normalize_weight_coordinate_system,
    normalized_positive_excess,
    normalized_positive_excess_policy,
    null_power_ratio_from_weight_norms,
    weight_term_norms_sq,
)
from pilot_proxy.detector_geometry import spectral_sense_requires_time_reversal
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.dtv_units import (
    DETECTOR_WINDOW_SAMPLES,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    normalized_pilot_excess_to_db,
    pilot_excess_db_to_data_shelf_snr_db,
    power_terms_to_coarse_power_ratio,
    power_terms_to_normalized_coarse_power_ratio_db,
    power_terms_to_normalized_pilot_excess,
)
from pilot_proxy.integration import (
    DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
    DEFAULT_CHIME_STREAM_MAP,
    layout_uint64_bound_check,
    load_receiver_profile,
    load_stream_map,
    parse_physical_channel_selection,
    receiver_frequency_to_channel,
)
from pilot_proxy.integration.packing import estimate_complex_scale
from pilot_proxy.integration.stream_layout import validate_integration_compatibility
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH
from pilot_proxy.provenance import file_sha256, sidecar_manifest_path

from .frame_adapter import pack_chime_block_for_detector
from .hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
    ChimePilotDataset,
    discover_chime_pilot_datasets,
    read_complex_window,
)
from .products import (
    ensure_run_dirs,
    write_detector_outputs,
    write_input_manifest,
    write_mask_summary,
    write_run_config,
    write_spectrogram_cache,
    write_spectrum_table,
    write_stats,
)
from .reductions import write_reductions_npz
from .segmented_input import available_frames, iter_frame_chunks

DEFAULT_FRAMES_PER_CHUNK = 1
DEFAULT_FRAME_SIZE_SAMPLES = 16_384
DEFAULT_CALIBRATION_SECONDS = 2.0
DetectorFn = Callable[..., dict[str, Any]]


class WeightBankLike(Protocol):
    path: Any
    manifest: Mapping[str, Any]



def _exact_detection_u64(value: object, *, field: str) -> int:
    """Accept exact backend uint64-domain values without coercive casts."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"detector field {field!r} must not be boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"detector field {field!r} must be an exact unsigned integer"
        ) from exc
    if not 0 <= result < (1 << 64):
        raise ValueError(f"detector field {field!r} is outside uint64")
    return int(result)


def _coerce_int_metadata(value: object, *, field: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes)):
        return int(value)
    raise TypeError(f"{field} must be an integer-compatible value, got {value!r}")


def _mapping_int_value(mapping: Mapping[str, Any], key: str, default: int) -> int:
    value = mapping.get(key)
    if value is None:
        return default
    return _coerce_int_metadata(value, field=key)


def _detector_contract_for_run(
    *,
    detector_window_samples: int,
    kernel_specs: Mapping[str, Any] | None,
    reference_placement_summary: Mapping[str, Any] | None,
    weight_coordinate_system: str,
    time_reverse_detector_windows_before_kernel: bool,
) -> dict[str, Any]:
    kernel = dict(kernel_specs or {})
    placement = dict(reference_placement_summary or {})
    reference_offset_bins = _mapping_int_value(
        placement,
        "reference_offset_bins",
        _mapping_int_value(kernel, "reference_offset_bins", 2),
    )
    skipped_guard_bins = _mapping_int_value(
        placement,
        "skipped_guard_bins",
        max(0, reference_offset_bins - 1),
    )
    return build_detector_contract(
        detector_window_samples=_mapping_int_value(
            kernel,
            "detector_window_samples",
            detector_window_samples,
        ),
        skipped_guard_bins=skipped_guard_bins,
        reference_offset_bins=reference_offset_bins,
        num_weight_terms=_mapping_int_value(kernel, "num_weight_terms", 3),
        sample_bits_per_component=_mapping_int_value(
            kernel,
            "sample_bits_per_component",
            4,
        ),
        input_format="complex_int4_packed_int8",
        power_accumulator="uint64",
        power_accumulator_bits=64,
        combine_mode=COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
        weight_coordinate_system=weight_coordinate_system,
        input_coordinate_system=input_coordinate_system_for_weight_coordinate(
            weight_coordinate_system
        ),
        time_reverse_detector_windows_before_kernel=(
            time_reverse_detector_windows_before_kernel
        ),
        reference_placement_summary=(
            dict(reference_placement_summary)
            if reference_placement_summary is not None
            else None
        ),
    )


def _kernel_detector_window_samples(kernel: Any) -> int | None:
    specs = getattr(kernel, "specs", None)
    if specs is None:
        return None
    value = getattr(specs, "detector_window_samples", None)
    if value is None:
        value = getattr(specs, "K", None)
    if value is None:
        return None
    return _coerce_int_metadata(value, field="detector_window_samples")


def _kernel_num_weight_terms(kernel: Any) -> int | None:
    specs = getattr(kernel, "specs", None)
    if specs is None:
        return None
    value = getattr(specs, "num_weight_terms", None)
    if value is None:
        value = getattr(specs, "N", None)
    if value is None:
        return None
    return _coerce_int_metadata(value, field="num_weight_terms")


def _validate_detector_window_contract(
    *,
    requested_detector_window_samples: int,
    selected_physical_channels: Sequence[int],
    kernel: Any,
    weight_bank: DetectorWeightBank | None,
    weights_by_channel: Mapping[int, np.ndarray] | None,
) -> int:
    requested = int(requested_detector_window_samples)
    kernel_window = _kernel_detector_window_samples(kernel)
    if kernel_window is not None and requested != int(kernel_window):
        raise ValueError(
            "detector_window_samples does not match kernel specs: "
            f"requested={requested}, kernel={int(kernel_window)}"
        )
    effective = requested if kernel_window is None else int(kernel_window)
    if weight_bank is not None and effective != int(weight_bank.K):
        raise ValueError(
            "detector_window_samples does not match weight bank: "
            f"requested={requested}, weights={int(weight_bank.K)}"
        )
    if weights_by_channel is not None:
        expected_terms = _kernel_num_weight_terms(kernel)
        for channel in selected_physical_channels:
            weights = np.asarray(weights_by_channel[int(channel)])
            if weights.ndim != 2:
                raise ValueError(
                    "caller-supplied CHIME weights must have shape "
                    f"(num_weight_terms, detector_window_samples); "
                    f"channel {int(channel)} has shape {weights.shape}"
                )
            if int(weights.shape[1]) != effective:
                raise ValueError(
                    "detector_window_samples does not match caller-supplied "
                    f"weights for physical channel {int(channel)}: "
                    f"requested={requested}, weights={int(weights.shape[1])}"
                )
            if expected_terms is not None and int(weights.shape[0]) != expected_terms:
                raise ValueError(
                    "num_weight_terms does not match caller-supplied weights "
                    f"for physical channel {int(channel)}: "
                    f"kernel={int(expected_terms)}, weights={int(weights.shape[0])}"
                )
    return effective


def _select_datasets(
    datasets: Mapping[int, ChimePilotDataset],
    *,
    physical_channels: Sequence[int] | None,
    physical_channel_range: str | None,
) -> list[ChimePilotDataset]:
    discovered = sorted(int(channel) for channel in datasets)
    if physical_channels or physical_channel_range:
        selected_channels = parse_physical_channel_selection(
            physical_channels=list(physical_channels or []),
            physical_channel_range=physical_channel_range,
        )
    else:
        selected_channels = discovered
    missing = [channel for channel in selected_channels if int(channel) not in datasets]
    if missing:
        raise ValueError(
            "requested physical channels were not discovered in the input: "
            + ", ".join(str(channel) for channel in missing)
        )
    return [datasets[int(channel)] for channel in selected_channels]


def _common_num_frames(
    datasets: Sequence[ChimePilotDataset],
    *,
    frame_size_samples: int,
    max_frames: int | None,
) -> int:
    frames = [
        available_frames(dataset, frame_size_samples=int(frame_size_samples))
        for dataset in datasets
    ]
    if not frames:
        raise ValueError("no CHIME pilot datasets selected")
    total = min(frames)
    if max_frames is not None:
        total = min(total, int(max_frames))
    if total <= 0:
        raise ValueError("selected CHIME datasets do not contain one full frame")
    return int(total)


def _kernel_version_string(kernel: Any) -> str | None:
    version = getattr(kernel, "version", None)
    if version is None:
        return None
    if hasattr(version, "as_string"):
        return str(version.as_string())
    return str(version)


def _run_file_provenance(
    *,
    receiver_profile_path: Path,
    stream_map_path: Path | None,
    weights_path: Path,
    lib_path: Path,
    input_manifest_path: Path | None = None,
    caller_supplied_weights: bool = False,
) -> dict[str, Any]:
    weights_manifest_path = sidecar_manifest_path(weights_path)
    return {
        "receiver_profile_sha256": file_sha256(receiver_profile_path),
        "stream_map_sha256": file_sha256(stream_map_path),
        "weights_sha256": (
            None if caller_supplied_weights else file_sha256(weights_path)
        ),
        "weight_manifest_path": (
            None
            if caller_supplied_weights or weights_manifest_path is None
            else str(weights_manifest_path)
        ),
        "weight_manifest_sha256": (
            None
            if caller_supplied_weights or weights_manifest_path is None
            else file_sha256(weights_manifest_path)
        ),
        "kernel_library_sha256": file_sha256(lib_path),
        "input_manifest_sha256": file_sha256(input_manifest_path),
    }


def _calibration_scale_for_dataset(
    dataset: ChimePilotDataset,
    *,
    frame_size_samples: int,
    calibration_seconds: float,
    bits_per_component: int,
    clip_sigma: float,
) -> float | None:
    if dataset.sample_encoding in {
        CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
    }:
        return None
    calibration_samples = min(
        int(dataset.total_time_samples),
        max(
            int(frame_size_samples),
            int(round(float(calibration_seconds) * 390_625.0)),
        ),
    )
    block = read_complex_window(
        dataset,
        start_sample=0,
        stop_sample=int(calibration_samples),
    )
    return estimate_complex_scale(
        block,
        bits_per_component=int(bits_per_component),
        clip_sigma=float(clip_sigma),
    )


def _weights_for_channel(
    *,
    physical_channel: int,
    weight_bank: DetectorWeightBank | None,
    weights_by_channel: Mapping[int, np.ndarray] | None,
) -> tuple[np.ndarray, bool]:
    if weights_by_channel is not None:
        weights = np.asarray(weights_by_channel[int(physical_channel)], dtype=np.int8)
        return np.ascontiguousarray(weights), True
    if weight_bank is None:
        raise ValueError("weight_bank is required when weights_by_channel is omitted")
    weights, valid = weight_bank.get_weights_for_physical_channel(int(physical_channel))
    if weights is None or not valid:
        raise ValueError(
            f"no valid detector weights for physical channel {physical_channel}"
        )
    return weights, bool(valid)


def _manifest_receiver_spectral_sense(manifest: Mapping[str, Any]) -> str | None:
    receiver_profile = manifest.get("receiver_profile")
    if not isinstance(receiver_profile, Mapping):
        return None
    channelizer = receiver_profile.get("channelizer")
    if isinstance(channelizer, Mapping):
        frequency_axis = channelizer.get("frequency_axis")
        if isinstance(frequency_axis, Mapping) and frequency_axis.get("spectral_sense"):
            return str(frequency_axis["spectral_sense"]).strip().lower()
        if channelizer.get("spectral_sense"):
            return str(channelizer["spectral_sense"]).strip().lower()
    if receiver_profile.get("spectral_sense"):
        return str(receiver_profile["spectral_sense"]).strip().lower()
    return None


def _weight_coordinate_metadata(
    *,
    weight_bank: WeightBankLike | None,
    input_spectral_sense: str,
) -> dict[str, Any]:
    input_requires_time_reversal = spectral_sense_requires_time_reversal(
        input_spectral_sense
    )
    if weight_bank is None:
        return {
            "expected_weight_coordinate_system": WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            "effective_weight_coordinate_system": "caller_supplied_weights",
            "input_spectral_sense": str(input_spectral_sense),
            "input_requires_time_reversal": bool(input_requires_time_reversal),
            "validated": False,
        }

    manifest = getattr(weight_bank, "manifest", {}) or {}
    declared = manifest.get("weight_coordinate_system")
    if declared is None:
        raise ValueError(
            "The current weight manifest requires weight_coordinate_system. "
            f"Regenerate {getattr(weight_bank, 'path', '<unknown>')} with an "
            "explicit weight coordinate convention."
        )
    coordinate_system = normalize_weight_coordinate_system(declared)
    manifest_sense = _manifest_receiver_spectral_sense(manifest)

    metadata = {
        "expected_weight_coordinate_system": WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        "effective_weight_coordinate_system": coordinate_system,
        "declared_weight_coordinate_system": coordinate_system,
        "manifest_receiver_spectral_sense": manifest_sense,
        "input_spectral_sense": str(input_spectral_sense),
        "input_requires_time_reversal": bool(input_requires_time_reversal),
        "weights_path": str(getattr(weight_bank, "path", "")),
        "validated": True,
    }
    if (
        input_requires_time_reversal
        and coordinate_system != WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    ):
        raise ValueError(
            "CHIME input is time-reversed into the normal detector coordinate, "
            "so the runner requires detector-coordinate weights. Use the shipped "
            "reference weights or a manifest declaring "
            f"{WEIGHT_COORDINATE_POST_SPECTRAL_SENSE!r}; got {coordinate_system!r} "
            "from "
            f"{getattr(weight_bank, 'path', '<unknown>')}."
        )
    preprocessing = manifest.get("input_preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError(
            "The current weight manifest requires input_preprocessing."
        )
    if "time_reverse_detector_windows_before_kernel" not in preprocessing:
        raise ValueError(
            "Weight manifest input_preprocessing requires "
            "time_reverse_detector_windows_before_kernel."
        )
    declared_time_reverse = bool(
        preprocessing["time_reverse_detector_windows_before_kernel"]
    )
    if declared_time_reverse != bool(input_requires_time_reversal):
        raise ValueError(
            "Weight manifest preprocessing does not match the runtime input "
            "spectral sense: declared time_reverse_detector_windows_before_kernel="
            f"{declared_time_reverse}, runtime requires {input_requires_time_reversal}."
        )
    return metadata


def _format_unique_status(values: Sequence[str]) -> str:
    unique = sorted({str(value) for value in values if str(value)})
    if not unique:
        return "unknown"
    return unique[0] if len(unique) == 1 else f"mixed:{';'.join(unique)}"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _reference_placement_summary(
    weight_manifest: Mapping[str, Any] | None,
    physical_channel: Sequence[int] | np.ndarray,
) -> dict[str, Any] | None:
    if not isinstance(weight_manifest, Mapping):
        return None
    layouts_raw = weight_manifest.get("target_reference_layout")
    if not isinstance(layouts_raw, (list, tuple)):
        return None
    selected_channels = {
        int(value) for value in np.asarray(physical_channel, dtype=np.int64).ravel()
    }
    layouts: list[Mapping[str, Any]] = []
    for row in layouts_raw:
        if not isinstance(row, Mapping):
            continue
        channel = _optional_int(row.get("physical_channel"))
        if channel is None or channel not in selected_channels:
            continue
        layouts.append(row)
    if not layouts:
        return None

    adaptive_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if str(layout.get("reference_placement_status", "unknown")) != "nominal"
    ]
    dc_shifted_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if bool(layout.get("dc_reference_shifted", False))
    ]
    edge_wrapped_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if bool(layout.get("edge_reference_wrapped", False))
    ]
    skipped_guard_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if bool(layout.get("forbidden_tone_in_skipped_guard", False))
    ]
    num_dc_shifted_references = sum(
        int(bool(layout.get("lower_reference_dc_shifted", False)))
        + int(bool(layout.get("upper_reference_dc_shifted", False)))
        for layout in layouts
    )
    num_edge_wrapped_references = sum(
        int(bool(layout.get("lower_reference_edge_wrapped", False)))
        + int(bool(layout.get("upper_reference_edge_wrapped", False)))
        for layout in layouts
    )
    status_values = [
        str(layout.get("reference_placement_status", "unknown"))
        for layout in layouts
    ]
    kernel_spec = weight_manifest.get("kernel_spec")
    if not isinstance(kernel_spec, Mapping):
        kernel_spec = {}
    reference_offset_bins = int(kernel_spec.get("reference_offset_bins", 0))
    skipped_guard_bins = int(
        kernel_spec.get("skipped_guard_bins", max(0, reference_offset_bins - 1))
    )
    return {
        "reference_offset_bins": int(reference_offset_bins),
        "skipped_guard_bins": int(skipped_guard_bins),
        "reference_placement_status": _format_unique_status(status_values),
        "num_channels_with_adaptive_reference": int(len(adaptive_channels)),
        "channels_with_adaptive_reference": adaptive_channels,
        "num_dc_shifted_references": int(num_dc_shifted_references),
        "channels_with_dc_shifted_reference": dc_shifted_channels,
        "num_edge_wrapped_references": int(num_edge_wrapped_references),
        "channels_with_edge_wrapped_reference": edge_wrapped_channels,
        "num_forbidden_tone_in_skipped_guard": int(len(skipped_guard_channels)),
        "channels_with_forbidden_tone_in_skipped_guard": skipped_guard_channels,
        "forbidden_tone_policy": weight_manifest.get("forbidden_tone_policy"),
        "by_channel": [
            {
                "physical_channel": int(layout["physical_channel"]),
                "reference_placement_status": str(
                    layout.get("reference_placement_status", "unknown")
                ),
                "placement_warnings": str(layout.get("placement_warnings", "")),
                "lower_reference_offset_bins": int(
                    layout.get("lower_reference_offset_bins", 0)
                ),
                "upper_reference_offset_bins": int(
                    layout.get("upper_reference_offset_bins", 0)
                ),
                "lower_reference_relative_to_target_hz": layout.get(
                    "lower_reference_relative_to_target_hz"
                ),
                "upper_reference_relative_to_target_hz": layout.get(
                    "upper_reference_relative_to_target_hz"
                ),
                "edge_reference_wrapped": bool(
                    layout.get("edge_reference_wrapped", False)
                ),
                "dc_reference_shifted": bool(
                    layout.get("dc_reference_shifted", False)
                ),
                "forbidden_tone_in_skipped_guard": bool(
                    layout.get("forbidden_tone_in_skipped_guard", False)
                ),
            }
            for layout in layouts
        ],
    }


def _append_detection_rows(
    *,
    detection: dict[str, Any],
    output_frame_start: int,
    pilot_index: int,
    p_target_u64: np.ndarray,
    p_ref_sum_u64: np.ndarray,
    coarse_power_ratio: np.ndarray,
    normalized_coarse_power_ratio_db: np.ndarray,
    pilot_excess_db: np.ndarray,
    estimated_data_shelf_snr_db: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    normalized_pilot_excess: np.ndarray,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    dtv_bandwidth_hz: float,
    pilot_capture_efficiency: float,
) -> None:
    rows = list(detection["results"])
    for local_index, row in enumerate(rows):
        frame = int(output_frame_start) + local_index
        num = _exact_detection_u64(
            row.get("p_target_u64", 0), field="p_target_u64"
        )
        den = _exact_detection_u64(
            row.get("p_ref_sum_u64", 0), field="p_ref_sum_u64"
        )
        p_target_u64[frame, pilot_index] = np.uint64(num)
        p_ref_sum_u64[frame, pilot_index] = np.uint64(den)
        coarse_power_ratio[frame, pilot_index] = float(
            power_terms_to_coarse_power_ratio(num, den)
        )
        normalized_coarse_power_ratio_db[frame, pilot_index] = float(
            power_terms_to_normalized_coarse_power_ratio_db(
                num,
                den,
                target_norm_sq=target_norm_sq,
                reference_norm_sum_sq=reference_norm_sum_sq,
            )
        )
        normalized_excess = float(
            power_terms_to_normalized_pilot_excess(
                num,
                den,
                target_norm_sq=target_norm_sq,
                reference_norm_sum_sq=reference_norm_sum_sq,
            )
        )
        normalized_pilot_excess[frame, pilot_index] = normalized_excess
        pilot_excess = float(normalized_pilot_excess_to_db(normalized_excess))
        pilot_excess_db[frame, pilot_index] = pilot_excess
        estimated_data_shelf_snr_db[frame, pilot_index] = float(
            pilot_excess_db_to_data_shelf_snr_db(
                pilot_excess,
                pilot_below_data_db=float(pilot_below_data_db),
                bin_enbw_hz=float(bin_enbw_hz),
                dtv_bandwidth_hz=float(dtv_bandwidth_hz),
                pilot_capture_efficiency=float(pilot_capture_efficiency),
            )
        )
        mask[frame, pilot_index] = normalized_positive_excess(
            num,
            den,
            target_norm_sq=target_norm_sq,
            reference_norm_sum_sq=reference_norm_sum_sq,
        )
        valid[frame, pilot_index] = 1 if den > 0 else 0


def _coarse_center_hz(dataset, receiver_profile) -> float:
    """Coarse-channel center frequency for a dataset (kept after offset-module removal)."""
    if dataset.coarse_channel_center_hz is not None:
        return float(dataset.coarse_channel_center_hz)
    selection = receiver_frequency_to_channel(
        float(dataset.pilot_frequency_hz),
        receiver_profile,
    )
    return float(
        receiver_profile.coarse_channel_center_hz(selection.coarse_channel_index)
    )


def run_chime_analysis(
    *,
    input_dir: Path,
    output_dir: Path,
    receiver_profile_path: Path = DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
    stream_map_path: Path | None = DEFAULT_CHIME_STREAM_MAP,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    lib_path: Path = DEFAULT_LIB_PATH,
    dataset_path: str | None = None,
    filename_pattern: str = "*.h5",
    physical_channels: Sequence[int] | None = None,
    physical_channel_range: str | None = None,
    frame_size_samples: int = DEFAULT_FRAME_SIZE_SAMPLES,
    detector_window_samples: int = DETECTOR_WINDOW_SAMPLES,
    frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK,
    max_frames: int | None = None,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    bin_enbw_hz: float = EFFECTIVE_BIN_BW_HZ,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
    calibration_seconds: float = DEFAULT_CALIBRATION_SECONDS,
    plot: bool = False,
    kernel: Any | None = None,
    detector_fn: DetectorFn = detect_packed_detector_input,
    weights_by_channel: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Path]:
    """Run the CHIME adapter without loading all samples into memory."""
    run_dir = Path(output_dir)
    ensure_run_dirs(run_dir)

    receiver_profile = load_receiver_profile(receiver_profile_path)
    channelized_sample_rate_hz = float(receiver_profile.sample_rate_hz) / float(
        receiver_profile.channelizer.fft_size
    )
    stream_map = None if stream_map_path is None else load_stream_map(stream_map_path)
    validate_integration_compatibility(
        profile=receiver_profile,
        stream_map=stream_map,
        require_stream_map=True,
    )
    discovered = discover_chime_pilot_datasets(
        Path(input_dir),
        dataset_path=dataset_path,
        filename_pattern=filename_pattern,
    )
    selected = _select_datasets(
        discovered,
        physical_channels=physical_channels,
        physical_channel_range=physical_channel_range,
    )
    if selected:
        expected = int(selected[0].num_input_streams)
        for dataset in selected:
            observed = int(dataset.num_input_streams)
            label = f"physical channel {int(dataset.physical_channel)}"
            if observed != expected:
                raise ValueError(
                    "selected CHIME datasets use inconsistent input-stream "
                    f"counts: {label} has {observed}, expected {expected}."
                )
            if stream_map is not None and int(stream_map.num_streams) != observed:
                raise ValueError(
                    "stream map num_streams does not match CHIME input streams "
                    f"for {label}: {stream_map.num_streams} != {observed}"
                )
            if int(receiver_profile.num_input_streams) != observed:
                raise ValueError(
                    "receiver profile num_input_streams does not match CHIME "
                    f"data for {label}: {receiver_profile.num_input_streams} "
                    f"!= {observed}"
                )
    physical_channel = np.asarray(
        [dataset.physical_channel for dataset in selected], dtype=np.int32
    )
    kernel_obj = kernel if kernel is not None else FStatKernel(lib_path)
    weight_bank = (
        None
        if weights_by_channel is not None
        else DetectorWeightBank(
            explicit_path=weights_path,
            expected_kernel=getattr(kernel_obj, "specs", None),
        )
    )
    detector_window_samples = _validate_detector_window_contract(
        requested_detector_window_samples=int(detector_window_samples),
        selected_physical_channels=[int(value) for value in physical_channel],
        kernel=kernel_obj,
        weight_bank=weight_bank,
        weights_by_channel=weights_by_channel,
    )
    if int(frame_size_samples) % int(detector_window_samples) != 0:
        raise ValueError(
            "frame_size_samples must be divisible by detector_window_samples"
        )

    num_frames = _common_num_frames(
        selected,
        frame_size_samples=int(frame_size_samples),
        max_frames=max_frames,
    )
    pilot_frequency_hz = np.asarray(
        [dataset.pilot_frequency_hz for dataset in selected],
        dtype=np.float64,
    )
    coarse_channel_center_hz = np.asarray(
        [_coarse_center_hz(dataset, receiver_profile) for dataset in selected],
        dtype=np.float64,
    )
    chime_frequency_hz = coarse_channel_center_hz.copy()
    layout_check = layout_uint64_bound_check(
        frame_size_samples=int(frame_size_samples),
        detector_window_samples=int(detector_window_samples),
        num_input_streams=int(selected[0].num_input_streams),
        num_selected_channels=1,
    )
    weight_coordinate = _weight_coordinate_metadata(
        weight_bank=weight_bank,
        input_spectral_sense=receiver_profile.spectral_sense,
    )
    reference_placement_summary = _reference_placement_summary(
        None if weight_bank is None else getattr(weight_bank, "manifest", None),
        physical_channel,
    )

    frame_index = np.arange(num_frames, dtype=np.int64)
    shape = (int(num_frames), int(len(selected)))
    p_target_u64 = np.zeros(shape, dtype=np.uint64)
    p_ref_sum_u64 = np.zeros(shape, dtype=np.uint64)
    coarse_power_ratio = np.full(shape, np.nan, dtype=np.float64)
    normalized_coarse_power_ratio_db = np.full(shape, np.nan, dtype=np.float64)
    pilot_excess_db = np.full(shape, np.nan, dtype=np.float64)
    estimated_data_shelf_snr_db = np.full(shape, np.nan, dtype=np.float64)
    normalized_pilot_excess = np.full(shape, np.nan, dtype=np.float64)
    mask = np.zeros(shape, dtype=np.uint8)
    valid = np.zeros(shape, dtype=np.uint8)
    baseband_power_linear = np.full(shape, np.nan, dtype=np.float64)

    quantization_by_pilot: list[dict[str, Any]] = []
    selected_channel_by_pilot: list[int] = []
    overflow_count_by_pilot = np.zeros(len(selected), dtype=np.uint64)
    target_norm_sq_by_pilot = np.zeros(len(selected), dtype=np.int64)
    reference_norm_sum_sq_by_pilot = np.zeros(len(selected), dtype=np.int64)
    null_power_ratio_by_channel = np.full(len(selected), np.nan, dtype=np.float64)

    for pilot_index, dataset in enumerate(selected):
        channel = int(dataset.physical_channel)
        weights, weights_valid = _weights_for_channel(
            physical_channel=channel,
            weight_bank=weight_bank,
            weights_by_channel=weights_by_channel,
        )
        # Exact integer squared norms of the three weight terms; these set the
        # H0 zero-point null_power_ratio = 2*nt/(nl+nu) that the norm-corrected mask (and the
        # corrected pilot excess) divide out. Computed from the weights actually
        # used, so caller-supplied test weights are handled identically.
        _nt, _nl, _nu = weight_term_norms_sq(weights)
        target_norm_sq_by_pilot[pilot_index] = int(_nt)
        reference_norm_sum_sq_by_pilot[pilot_index] = int(_nl + _nu)
        if (_nl + _nu) > 0:
            null_power_ratio_by_channel[pilot_index] = null_power_ratio_from_weight_norms(_nt, _nl + _nu)
        selection = receiver_frequency_to_channel(
            float(dataset.pilot_frequency_hz),
            receiver_profile,
        )
        selected_channel_by_pilot.append(int(selection.coarse_channel_index))
        scale = _calibration_scale_for_dataset(
            dataset,
            frame_size_samples=int(frame_size_samples),
            calibration_seconds=float(calibration_seconds),
            bits_per_component=int(getattr(receiver_profile, "bits_per_component", 4)),
            clip_sigma=float(getattr(receiver_profile, "clip_sigma_default", 3.0)),
        )
        first_quantization: dict[str, Any] | None = None
        for chunk in iter_frame_chunks(
            dataset,
            frame_size_samples=int(frame_size_samples),
            frames_per_chunk=int(frames_per_chunk),
            max_frames=int(num_frames),
        ):
            block = read_complex_window(
                dataset,
                start_sample=chunk.start_sample,
                stop_sample=chunk.stop_sample,
            )
            packed = pack_chime_block_for_detector(
                block,
                frame_size_samples=int(frame_size_samples),
                detector_window_samples=int(detector_window_samples),
                spectral_sense=receiver_profile.spectral_sense,
                frames_in_chunk=int(chunk.frames_in_chunk),
                sample_encoding=dataset.sample_encoding,
                selected_coarse_channel=int(selection.coarse_channel_index),
                physical_channel=channel,
                bits_per_component=int(
                    getattr(receiver_profile, "bits_per_component", 4)
                ),
                scale=scale,
                clip_sigma=float(getattr(receiver_profile, "clip_sigma_default", 3.0)),
            )
            if first_quantization is None:
                quantization = dict(packed.quantization)
                if scale is not None:
                    quantization["calibration_scale"] = float(scale)
                first_quantization = quantization
            baseband_power_linear[
                chunk.start_frame : chunk.start_frame + chunk.frames_in_chunk,
                pilot_index,
            ] = packed.baseband_power_linear
            detection = detector_fn(
                packed=packed.packed,
                weights=weights,
                kernel=kernel_obj,
            )
            overflow = _exact_detection_u64(
                detection.get("rational_overflow_count", 0),
                field="rational_overflow_count",
            )
            accumulated_overflow = (
                int(overflow_count_by_pilot[pilot_index]) + overflow
            )
            if accumulated_overflow >= (1 << 64):
                raise ValueError(
                    "accumulated rational_overflow_count exceeds uint64"
                )
            overflow_count_by_pilot[pilot_index] = np.uint64(
                accumulated_overflow
            )
            _append_detection_rows(
                detection=detection,
                output_frame_start=int(chunk.start_frame),
                pilot_index=int(pilot_index),
                p_target_u64=p_target_u64,
                p_ref_sum_u64=p_ref_sum_u64,
                coarse_power_ratio=coarse_power_ratio,
                normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
                pilot_excess_db=pilot_excess_db,
                estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
                mask=mask,
                valid=valid,
                normalized_pilot_excess=normalized_pilot_excess,
                target_norm_sq=int(target_norm_sq_by_pilot[pilot_index]),
                reference_norm_sum_sq=int(
                    reference_norm_sum_sq_by_pilot[pilot_index]
                ),
                pilot_below_data_db=float(pilot_below_data_db),
                bin_enbw_hz=float(bin_enbw_hz),
                dtv_bandwidth_hz=float(dtv_bandwidth_hz),
                pilot_capture_efficiency=float(pilot_capture_efficiency),
            )
        if not weights_valid:
            valid[:, pilot_index] = 0
        quantization_by_pilot.append(first_quantization or {})

    valid = ((valid != 0) & (p_ref_sum_u64 != 0)).astype(np.uint8)
    # _append_detection_rows derives the declared norm-corrected rule from the
    # exact stored marginals rather than trusting a backend mask bit. Fold in
    # the final validity here, including the weights-validity policy.
    mask = ((valid != 0) & (mask != 0)).astype(np.uint8)

    input_manifest_path = write_input_manifest(
        run_dir,
        datasets=selected,
        input_dir=Path(input_dir),
    )
    provenance = _run_file_provenance(
        receiver_profile_path=Path(receiver_profile_path),
        stream_map_path=None if stream_map_path is None else Path(stream_map_path),
        weights_path=Path(weights_path),
        lib_path=Path(lib_path),
        input_manifest_path=input_manifest_path,
        caller_supplied_weights=weights_by_channel is not None,
    )
    kernel_specs = getattr(kernel_obj, "specs", None)
    kernel_specs_serializer = getattr(kernel_specs, "as_descriptive_dict", None)
    kernel_specs_dict: dict[str, Any] | None = None
    if callable(kernel_specs_serializer):
        raw_kernel_specs = kernel_specs_serializer()
        if isinstance(raw_kernel_specs, MappingABC):
            kernel_specs_dict = dict(raw_kernel_specs)
    contract_weight_coordinate = str(
        weight_coordinate["effective_weight_coordinate_system"]
    )
    if contract_weight_coordinate == "caller_supplied_weights":
        contract_weight_coordinate = str(
            weight_coordinate["expected_weight_coordinate_system"]
        )
    detector_contract = _detector_contract_for_run(
        detector_window_samples=int(detector_window_samples),
        kernel_specs=kernel_specs_dict,
        reference_placement_summary=reference_placement_summary,
        weight_coordinate_system=contract_weight_coordinate,
        time_reverse_detector_windows_before_kernel=bool(
            weight_coordinate["input_requires_time_reversal"]
        ),
    )
    run_config_payload: dict[str, Any] = {
        "schema_version": CHIME_RUN_CONFIG_SCHEMA_TOKEN,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "receiver_profile": str(receiver_profile_path),
        "stream_map": None if stream_map_path is None else str(stream_map_path),
        "weights_path": str(weights_path),
        "lib_path": str(lib_path),
        "dataset_path": dataset_path,
        "filename_pattern": filename_pattern,
        "physical_channels": [int(value) for value in physical_channel],
        "pilot_frequency_hz": [float(value) for value in pilot_frequency_hz],
        "chime_frequency_hz": [float(value) for value in chime_frequency_hz],
        "frame_size_samples": int(frame_size_samples),
        "detector_window_samples": int(detector_window_samples),
        "frames_per_chunk": int(frames_per_chunk),
        "max_frames": None if max_frames is None else int(max_frames),
        "absolute_time_used": False,
        "weight_coordinate": weight_coordinate,
        "mask_policy": normalized_positive_excess_policy(),
        "detector_contract": detector_contract,
        "provenance": provenance,
        **provenance,
    }
    if reference_placement_summary is not None:
        run_config_payload["reference_placement_summary"] = (
            reference_placement_summary
        )
    run_config_path = write_run_config(run_dir, run_config_payload)
    stats_payload: dict[str, Any] = {
        "schema_version": CHIME_STATS_SCHEMA_TOKEN,
        "absolute_time_used": False,
        "time_axis": "contiguous_file_order_frame_index",
        "num_input_streams": int(selected[0].num_input_streams),
        "frame_size_samples": int(frame_size_samples),
        "detector_window_samples": int(detector_window_samples),
        "windows_per_stream": int(frame_size_samples)
        // int(detector_window_samples),
        "detector_rows_per_frame": int(layout_check["detector_rows_per_frame"]),
        "combine_mode": "all_rows_summed_before_ratio",
        "coarse_power_ratio_definition": (
            "R_coarse = 2 * sum(P_target) / "
            "(sum(P_ref_lower) + sum(P_ref_upper))"
        ),
        "num_frames": int(num_frames),
        "num_pilots": int(len(selected)),
        "layout_check": layout_check,
        "weight_coordinate": weight_coordinate,
        "sample_encoding_by_pilot": [
            {
                "physical_channel": int(dataset.physical_channel),
                "sample_encoding": dataset.sample_encoding,
                "dataset_path": dataset.dataset_path,
                "dtype": dataset.segments[0].dtype,
                "shape": list(dataset.segments[0].shape),
                "time_axis": int(dataset.time_axis),
                "stream_axis": int(dataset.stream_axis),
            }
            for dataset in selected
        ],
        "selected_coarse_channel_by_pilot": selected_channel_by_pilot,
        "chime_frequency_hz_by_pilot": [float(value) for value in chime_frequency_hz],
        "pilot_frequency_hz_by_pilot": [float(value) for value in pilot_frequency_hz],
        "quantization_by_pilot": quantization_by_pilot,
        "rational_overflow_count_by_pilot": [
            int(value) for value in overflow_count_by_pilot
        ],
        "target_norm_sq_by_pilot": [int(v) for v in target_norm_sq_by_pilot],
        "reference_norm_sum_sq_by_pilot": [int(v) for v in reference_norm_sum_sq_by_pilot],
        "null_power_ratio_by_channel": [float(v) for v in null_power_ratio_by_channel],
        "kernel_version": _kernel_version_string(kernel_obj),
        "kernel_specs": kernel_specs_dict,
        "mask_policy": normalized_positive_excess_policy(),
        "detector_contract": detector_contract,
        "provenance": provenance,
        **provenance,
    }
    if reference_placement_summary is not None:
        stats_payload["reference_placement_summary"] = reference_placement_summary
    stats_path = write_stats(run_dir, stats_payload)
    detector_outputs_path = write_detector_outputs(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        p_target_u64=p_target_u64,
        p_ref_sum_u64=p_ref_sum_u64,
        coarse_power_ratio=coarse_power_ratio,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        pilot_excess_db=pilot_excess_db,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        mask=mask,
        valid=valid,
        target_norm_sq=target_norm_sq_by_pilot,
        reference_norm_sum_sq=reference_norm_sum_sq_by_pilot,
        null_power_ratio=null_power_ratio_by_channel,
        normalized_pilot_excess=normalized_pilot_excess,
    )
    spectrogram_cache_path = write_spectrogram_cache(
        run_dir,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        frame_size_samples=int(frame_size_samples),
        sample_rate_hz=channelized_sample_rate_hz,
    )
    reductions_10s_path = write_reductions_npz(
        run_dir,
        frame_index=frame_index,
        frame_size_samples=int(frame_size_samples),
        sample_rate_hz=channelized_sample_rate_hz,
        chunk_seconds=10.0,
        coarse_power_ratio=coarse_power_ratio,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
    )
    mask_summary_path = write_mask_summary(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        mask=mask,
        valid=valid,
    )
    spectrum_table_path = write_spectrum_table(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
    )

    plot_outputs: list[Path] = []
    if plot:
        from .plots import clean_known_figures, generate_chime_plots

        clean_known_figures(run_dir)
        plot_outputs = generate_chime_plots(run_dir)

    outputs = {
        "run_config": run_config_path,
        "input_manifest": input_manifest_path,
        "stats": stats_path,
        "detector_outputs": detector_outputs_path,
        "spectrogram_cache": spectrogram_cache_path,
        "reductions_10s": reductions_10s_path,
        **{f"plot_{index}": path for index, path in enumerate(plot_outputs)},
    }
    if mask_summary_path is not None:
        outputs["mask_summary"] = mask_summary_path
    if spectrum_table_path is not None:
        outputs["spectrum_table"] = spectrum_table_path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the chunked CHIME real-data adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--receiver-profile", type=Path, default=DEFAULT_CHIME_DTV_RECEIVER_PROFILE
    )
    parser.add_argument("--stream-map", type=Path, default=DEFAULT_CHIME_STREAM_MAP)
    parser.add_argument(
        "--weights-path", dest="weights_path", type=Path, default=DEFAULT_WEIGHTS_PATH
    )
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--filename-pattern", default="*.h5")
    parser.add_argument("--physical-channel", type=int, action="append", default=None)
    parser.add_argument("--physical-channel-range", default=None)
    parser.add_argument(
        "--frame-size-samples", type=int, default=DEFAULT_FRAME_SIZE_SAMPLES
    )
    parser.add_argument(
        "--frames-per-chunk", type=int, default=DEFAULT_FRAMES_PER_CHUNK
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--pilot-below-data-db", type=float, default=PILOT_BELOW_DATA_DB
    )
    parser.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    parser.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    parser.add_argument(
        "--pilot-capture-efficiency", type=float, default=PILOT_CAPTURE_EFFICIENCY
    )
    parser.add_argument(
        "--calibration-seconds", type=float, default=DEFAULT_CALIBRATION_SECONDS
    )
    parser.add_argument("--plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_chime_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        receiver_profile_path=args.receiver_profile,
        stream_map_path=args.stream_map,
        weights_path=args.weights_path,
        lib_path=args.lib_path,
        dataset_path=args.dataset_path,
        filename_pattern=args.filename_pattern,
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
        frame_size_samples=args.frame_size_samples,
        frames_per_chunk=args.frames_per_chunk,
        max_frames=args.max_frames,
        pilot_below_data_db=args.pilot_below_data_db,
        bin_enbw_hz=args.bin_enbw_hz,
        dtv_bandwidth_hz=args.dtv_bandwidth_hz,
        pilot_capture_efficiency=args.pilot_capture_efficiency,
        calibration_seconds=args.calibration_seconds,
        plot=args.plot,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

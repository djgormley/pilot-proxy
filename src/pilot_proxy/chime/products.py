# coding=utf-8
"""Output product writers for CHIME real-data runs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.atomic_io import (
    create_temporary_sibling,
    fsync_directory,
)
from pilot_proxy.config.project import default_project
from pilot_proxy.detector_contract import NORMALIZED_POSITIVE_EXCESS_MASK_RULE
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.schema_identity import schema_token

from .hdf5_input import ChimePilotDataset, dataset_manifest

CHIME_DETECTOR_OUTPUTS_FILENAME = "chime_detector_outputs.npz"
CHIME_SPECTROGRAM_CACHE_FILENAME = "chime_spectrogram_cache.npz"
CHIME_INTEGRATED_SPECTRA_FILENAME = "chime_integrated_spectra.npz"
CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME = ".pilotproxy-combine-publish.json"
CHIME_COMBINE_PUBLISH_LOCK_FILENAME = ".pilotproxy-combine-publish.lock"
CHIME_COMBINE_GENERATION_MANIFEST_FILENAME = (
    ".pilotproxy-combine-generation.json"
)
CHIME_COMBINE_GENERATION_MANIFEST_SCHEMA = (
    "pilotproxy_combine_generation_v1"
)
CHIME_COMBINE_CANONICAL_RELATIVE_PATHS = frozenset(
    {
        CHIME_DETECTOR_OUTPUTS_FILENAME,
        CHIME_SPECTROGRAM_CACHE_FILENAME,
        CHIME_INTEGRATED_SPECTRA_FILENAME,
        "chime_reductions_10s.npz",
        "chime_frame_identity.npz",
        "run_config.json",
        "stats.json",
        "input_manifest.json",
        "tables/mask_summary_by_pilot.csv",
        CHIME_COMBINE_GENERATION_MANIFEST_FILENAME,
    }
)
CHIME_INPUT_MANIFEST_SCHEMA_NAME = "pilotproxy_chime_input_manifest"
CHIME_INPUT_MANIFEST_SCHEMA_REVISION = 1
CHIME_INPUT_MANIFEST_SCHEMA_TOKEN = schema_token(
    CHIME_INPUT_MANIFEST_SCHEMA_NAME,
    CHIME_INPUT_MANIFEST_SCHEMA_REVISION,
)
SCAN_INPUT_MANIFEST_SCHEMA_NAME = "pilotproxy_scan_input_manifest"
SCAN_INPUT_MANIFEST_SCHEMA_REVISION = 1
SCAN_INPUT_MANIFEST_SCHEMA_TOKEN = schema_token(
    SCAN_INPUT_MANIFEST_SCHEMA_NAME,
    SCAN_INPUT_MANIFEST_SCHEMA_REVISION,
)
# The channel sample rate of the project's instrument (390625 Hz on CHIME).
SAMPLE_RATE_HZ = default_project().instrument.sample_rate_hz


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> Path:
    """Write and CRC-read an NPZ beside its destination before atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = create_temporary_sibling(
        destination, suffix=".tmp.npz"
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        # Force every member through NumPy/zip CRC validation before the file
        # can replace a prior canonical product.
        with np.load(temporary, allow_pickle=False) as archive:
            for name in archive.files:
                np.asarray(archive[name])
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def ensure_run_dirs(run_dir: Path) -> tuple[Path, Path]:
    run = Path(run_dir)
    tables = run / "tables"
    figures = run / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def power_to_db(power: np.ndarray) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    valid = values > 0.0
    out[valid] = 10.0 * np.log10(values[valid])
    return out


def _valid_array_like(values: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    if valid is None:
        return np.ones_like(np.asarray(values), dtype=bool)
    flags = np.asarray(valid) != 0
    if flags.shape != np.asarray(values).shape:
        raise ValueError("valid must have the same shape as the frame product")
    return flags


def _optional_frequency_array(
    values: Sequence[float] | np.ndarray | None,
    *,
    shape_like: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return frequency metadata without inventing a different coordinate.

    A nominal pilot frequency is not a substitute for an unknown receiver
    coarse-channel centre.  Missing metadata is represented by NaN and remains
    visibly unavailable to downstream validation and plotting.
    """
    reference = np.asarray(shape_like)
    if values is None:
        return np.full(reference.shape, np.nan, dtype=np.float64)
    out = np.asarray(values, dtype=np.float64)
    if out.shape != reference.shape:
        raise ValueError("frequency metadata must match the pilot-frequency shape")
    return out


def mean_where(
    values: np.ndarray, include: np.ndarray, *, axis: int = 0
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    flags = np.asarray(include, dtype=bool)
    if arr.shape != flags.shape:
        raise ValueError("values and include must have the same shape")
    numerator = np.sum(np.where(flags, arr, 0.0), axis=axis)
    denominator = np.sum(flags, axis=axis)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def valid_mask_counts(
    mask: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Return per-pilot valid/mask counts and fractions.

    The valid mask fraction uses valid frames as the denominator. The total mask
    fraction uses all stored frames as the denominator.
    """
    flags = np.asarray(mask) != 0
    valid_flags = _valid_array_like(flags, valid)
    if flags.shape != valid_flags.shape:
        raise ValueError("mask and valid must have the same shape")
    valid_count = np.sum(valid_flags, axis=0).astype(np.int64)
    total_count = np.full(valid_count.shape, flags.shape[0], dtype=np.int64)
    masked_count_valid = np.sum(flags & valid_flags, axis=0).astype(np.int64)
    unmasked_count_valid = np.sum((~flags) & valid_flags, axis=0).astype(np.int64)
    invalid_count = total_count - valid_count
    mask_fraction_valid = np.full(valid_count.shape, np.nan, dtype=np.float64)
    np.divide(
        masked_count_valid,
        valid_count,
        out=mask_fraction_valid,
        where=valid_count > 0,
    )
    mask_fraction_total = np.full(total_count.shape, np.nan, dtype=np.float64)
    np.divide(
        np.sum(flags, axis=0),
        total_count,
        out=mask_fraction_total,
        where=total_count > 0,
    )
    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count.astype(np.int64),
        "masked_count_valid": masked_count_valid,
        "unmasked_count_valid": unmasked_count_valid,
        "mask_fraction_valid": mask_fraction_valid,
        "mask_fraction_total": mask_fraction_total,
        "total_count": total_count,
    }


def relative_time_seconds(
    frame_index: np.ndarray,
    *,
    frame_size_samples: int,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    frames = np.asarray(frame_index, dtype=np.float64)
    return np.asarray(frames * (
        float(frame_size_samples) / float(sample_rate_hz)
    ))


def write_detector_outputs(
    run_dir: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray | None = None,
    frame_index: np.ndarray,
    p_target_u64: np.ndarray,
    p_ref_sum_u64: np.ndarray,
    coarse_power_ratio: np.ndarray,
    normalized_coarse_power_ratio_db: np.ndarray,
    pilot_excess_db: np.ndarray,
    estimated_data_shelf_snr_db: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    target_norm_sq: np.ndarray,
    reference_norm_sum_sq: np.ndarray,
    null_power_ratio: np.ndarray,
    normalized_pilot_excess: np.ndarray,
) -> Path:
    path = Path(run_dir) / CHIME_DETECTOR_OUTPUTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = dict(
        physical_channel=np.asarray(physical_channel, dtype=np.int32),
        pilot_frequency_hz=np.asarray(pilot_frequency_hz, dtype=np.float64),
        chime_frequency_hz=_optional_frequency_array(
            chime_frequency_hz,
            shape_like=pilot_frequency_hz,
        ),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        p_target_u64=np.asarray(p_target_u64, dtype=np.uint64),
        p_ref_sum_u64=np.asarray(p_ref_sum_u64, dtype=np.uint64),
        coarse_power_ratio=np.asarray(coarse_power_ratio, dtype=np.float64),
        normalized_coarse_power_ratio_db=np.asarray(normalized_coarse_power_ratio_db, dtype=np.float64),
        pilot_excess_db=np.asarray(pilot_excess_db, dtype=np.float64),
        estimated_data_shelf_snr_db=np.asarray(estimated_data_shelf_snr_db, dtype=np.float64),
        mask=np.asarray(mask, dtype=np.uint8),
        valid=np.asarray(valid, dtype=np.uint8),
        target_norm_sq=np.asarray(target_norm_sq, dtype=np.int64),
        reference_norm_sum_sq=np.asarray(reference_norm_sum_sq, dtype=np.int64),
        null_power_ratio=np.asarray(null_power_ratio, dtype=np.float64),
        normalized_pilot_excess=np.asarray(
            normalized_pilot_excess, dtype=np.float64
        ),
    )
    return atomic_savez_compressed(path, **arrays)


def write_spectrogram_cache(
    run_dir: Path,
    *,
    baseband_power_linear: np.ndarray,
    mask: np.ndarray,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray | None = None,
    frame_index: np.ndarray,
    frame_size_samples: int,
    sample_rate_hz: float,
    valid: np.ndarray | None = None,
) -> Path:
    path = Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_array = _valid_array_like(baseband_power_linear, valid)
    return atomic_savez_compressed(
        path,
        baseband_power_linear=np.asarray(baseband_power_linear, dtype=np.float64),
        baseband_power_db=power_to_db(baseband_power_linear),
        mask=np.asarray(mask, dtype=np.uint8),
        valid=np.asarray(valid_array, dtype=np.uint8),
        physical_channel=np.asarray(physical_channel, dtype=np.int32),
        pilot_frequency_hz=np.asarray(pilot_frequency_hz, dtype=np.float64),
        chime_frequency_hz=_optional_frequency_array(
            chime_frequency_hz,
            shape_like=pilot_frequency_hz,
        ),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        relative_time_s=relative_time_seconds(
            np.asarray(frame_index),
            frame_size_samples=int(frame_size_samples),
            sample_rate_hz=float(sample_rate_hz),
        ),
    )


def write_integrated_spectra(
    run_dir: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray,
    integrated_spectrum_before_mask: np.ndarray,
    integrated_spectrum_after_mask: np.ndarray,
    masked_fraction_by_channel: np.ndarray,
    sample_rate_hz: float,
    nfft: int,
    freq_id: np.ndarray | None = None,
) -> Path:
    """Per-pilot integrated power spectra stacked along the pilot axis.

    ``integrated_spectrum_*`` are ``[n_pilots, nfft]`` (rectangular-window |FFT|^2
    summed over feeds, accumulated over frames): ``before`` over every valid frame,
    ``after`` over kept (not-rejected) frames, so ``before - after`` is the
    spectrum the positive-excess mask removed. Bin ``k`` maps to baseband frequency
    ``((k + nfft//2) % nfft - nfft//2) * sample_rate_hz / nfft``.

    Reporting-only convenience: the authoritative per-channel copy lives in each
    ``<freq_id>.npz``; this is the 23-up stack so a report need open one file.
    """
    path = Path(run_dir) / CHIME_INTEGRATED_SPECTRA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        physical_channel=np.asarray(physical_channel, dtype=np.int32),
        pilot_frequency_hz=np.asarray(pilot_frequency_hz, dtype=np.float64),
        chime_frequency_hz=np.asarray(chime_frequency_hz, dtype=np.float64),
        integrated_spectrum_before_mask=np.asarray(
            integrated_spectrum_before_mask, dtype=np.float64),
        integrated_spectrum_after_mask=np.asarray(
            integrated_spectrum_after_mask, dtype=np.float64),
        masked_fraction_by_channel=np.asarray(
            masked_fraction_by_channel, dtype=np.float64),
        sample_rate_hz=np.asarray(float(sample_rate_hz), dtype=np.float64),
        nfft=np.asarray(int(nfft), dtype=np.int64),
        schema_version=np.asarray("pilotproxy_chime_integrated_spectra_v1"),
    )
    if freq_id is not None:
        payload["freq_id"] = np.asarray(freq_id, dtype=np.int64)
    return atomic_savez_compressed(path, **payload)


def spectrum_before_after(
    baseband_power_linear: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(baseband_power_linear, dtype=np.float64)
    flags = np.asarray(mask, dtype=np.uint8)
    if values.shape != flags.shape:
        raise ValueError("baseband_power_linear and mask must have the same shape")
    valid_flags = _valid_array_like(values, valid)
    before = mean_where(values, valid_flags, axis=0)
    after = mean_where(values, valid_flags & (flags == 0), axis=0)
    return power_to_db(before), power_to_db(np.asarray(after, dtype=np.float64))


def write_spectrum_table(
    run_dir: Path,
    *,
    physical_channel: Sequence[int],
    pilot_frequency_hz: Sequence[float],
    chime_frequency_hz: Sequence[float] | None = None,
    baseband_power_linear: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray | None = None,
) -> Path:
    before_db, after_db = spectrum_before_after(baseband_power_linear, mask, valid)
    counts = valid_mask_counts(mask, valid)
    coarse_centres = _optional_frequency_array(
        chime_frequency_hz,
        shape_like=pilot_frequency_hz,
    )
    path = Path(run_dir) / "tables" / "spectrum_before_after.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "physical_channel",
                "pilot_frequency_hz",
                "chime_frequency_hz",
                "before_mask_power_db",
                "after_mask_power_db",
                "mask_fraction",
                "valid_count",
                "invalid_count",
                "masked_count_valid",
                "unmasked_count_valid",
                "mask_fraction_valid",
                "mask_fraction_total",
            ],
        )
        writer.writeheader()
        for index, channel in enumerate(physical_channel):
            writer.writerow(
                {
                    "physical_channel": int(channel),
                    "pilot_frequency_hz": float(pilot_frequency_hz[index]),
                    "chime_frequency_hz": float(coarse_centres[index]),
                    "before_mask_power_db": float(before_db[index]),
                    "after_mask_power_db": float(after_db[index]),
                    "mask_fraction": float(counts["mask_fraction_valid"][index]),
                    "valid_count": int(counts["valid_count"][index]),
                    "invalid_count": int(counts["invalid_count"][index]),
                    "masked_count_valid": int(counts["masked_count_valid"][index]),
                    "unmasked_count_valid": int(counts["unmasked_count_valid"][index]),
                    "mask_fraction_valid": float(counts["mask_fraction_valid"][index]),
                    "mask_fraction_total": float(counts["mask_fraction_total"][index]),
                }
            )
    return path


def write_mask_summary(
    run_dir: Path,
    *,
    physical_channel: Sequence[int],
    pilot_frequency_hz: Sequence[float],
    chime_frequency_hz: Sequence[float] | None = None,
    mask: np.ndarray,
    valid: np.ndarray | None = None,
    mask_source: str = "normalized_positive_excess_decision",
    mask_rule: str = NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    valid_rule: str = "p_ref_sum != 0",
) -> Path:
    counts = valid_mask_counts(mask, valid)
    coarse_centres = _optional_frequency_array(
        chime_frequency_hz,
        shape_like=pilot_frequency_hz,
    )
    path = Path(run_dir) / "tables" / "mask_summary_by_pilot.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "physical_channel",
                "pilot_frequency_hz",
                "chime_frequency_hz",
                "mask_source",
                "valid_rule",
                "mask_rule",
                "valid_count",
                "invalid_count",
                "masked_count_valid",
                "unmasked_count_valid",
                "mask_fraction_valid",
                "mask_fraction_total",
            ],
        )
        writer.writeheader()
        for index, channel in enumerate(physical_channel):
            writer.writerow(
                {
                    "physical_channel": int(channel),
                    "pilot_frequency_hz": float(pilot_frequency_hz[index]),
                    "chime_frequency_hz": float(coarse_centres[index]),
                    "mask_source": str(mask_source),
                    "valid_rule": str(valid_rule),
                    "mask_rule": str(mask_rule),
                    "valid_count": int(counts["valid_count"][index]),
                    "invalid_count": int(counts["invalid_count"][index]),
                    "masked_count_valid": int(counts["masked_count_valid"][index]),
                    "unmasked_count_valid": int(counts["unmasked_count_valid"][index]),
                    "mask_fraction_valid": float(counts["mask_fraction_valid"][index]),
                    "mask_fraction_total": float(counts["mask_fraction_total"][index]),
                }
            )
    return path


def write_input_manifest(
    run_dir: Path,
    *,
    datasets: Sequence[ChimePilotDataset],
    input_dir: Path,
) -> Path:
    path = Path(run_dir) / "input_manifest.json"
    payload = {
        "schema_version": CHIME_INPUT_MANIFEST_SCHEMA_TOKEN,
        "input_dir": str(input_dir),
        "absolute_time_used": False,
        "datasets": [dataset_manifest(dataset) for dataset in datasets],
    }
    write_json_strict(path, payload, indent=2, sort_keys=True)
    return path


def write_stats(run_dir: Path, stats: dict[str, Any]) -> Path:
    path = Path(run_dir) / "stats.json"
    write_json_strict(path, stats, indent=2, sort_keys=True)
    return path


def write_run_config(run_dir: Path, config: dict[str, Any]) -> Path:
    path = Path(run_dir) / "run_config.json"
    write_json_strict(path, config, indent=2, sort_keys=True)
    return path


__all__ = [
    "atomic_savez_compressed",
    "CHIME_DETECTOR_OUTPUTS_FILENAME",
    "CHIME_COMBINE_CANONICAL_RELATIVE_PATHS",
    "CHIME_COMBINE_GENERATION_MANIFEST_FILENAME",
    "CHIME_COMBINE_GENERATION_MANIFEST_SCHEMA",
    "CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME",
    "CHIME_COMBINE_PUBLISH_LOCK_FILENAME",
    "CHIME_INPUT_MANIFEST_SCHEMA_NAME",
    "CHIME_INPUT_MANIFEST_SCHEMA_REVISION",
    "CHIME_INPUT_MANIFEST_SCHEMA_TOKEN",
    "SCAN_INPUT_MANIFEST_SCHEMA_NAME",
    "SCAN_INPUT_MANIFEST_SCHEMA_REVISION",
    "SCAN_INPUT_MANIFEST_SCHEMA_TOKEN",
    "CHIME_SPECTROGRAM_CACHE_FILENAME",
    "SAMPLE_RATE_HZ",
    "ensure_run_dirs",
    "mean_where",
    "power_to_db",
    "relative_time_seconds",
    "spectrum_before_after",
    "valid_mask_counts",
    "write_detector_outputs",
    "write_input_manifest",
    "write_mask_summary",
    "write_run_config",
    "write_spectrogram_cache",
    "write_spectrum_table",
    "write_stats",
]

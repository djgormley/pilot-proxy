#!/usr/bin/env python3
# coding=utf-8
"""Run fixed-point DTV pilot detection on packed detector input."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .atsc_channels import physical_channel_to_pilot_hz
from .detector_contract import (
    NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    normalized_positive_excess,
    null_power_ratio_from_weight_norms,
    weight_term_norms_sq,
)
from .detector_reference import (
    REFERENCE_LOWER_TERM_INDEX,
    REFERENCE_TARGET_TERM_INDEX,
    REFERENCE_UPPER_TERM_INDEX,
    REFERENCE_WEIGHT_TERMS,
)
from .detector_weights import DetectorWeightBank
from .dtv_units import (
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    normalized_pilot_excess_to_db,
    pilot_excess_db_to_data_shelf_snr_db,
    power_terms_to_coarse_power_ratio,
    power_terms_to_normalized_coarse_power_ratio_db,
    power_terms_to_normalized_pilot_excess,
    power_terms_to_raw_pilot_excess,
)
from .fine_decision import FINE_BINS
from .fxfft import N_IN as FINE_WINDOWS_PER_STREAM
from .json_utils import write_json_strict
from .kernel import FStatKernel
from .paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH
from .result_schema import (
    COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    RESULT_SCHEMA_TOKEN,
    result_schema_object,
)

DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ = 10.0
HZ_PER_MHZ = 1.0e6
DEFAULT_DETECTOR_INPUT_DIR = Path("generated/detector_input")
DEFAULT_DETECTOR_MATRIX_FILENAME = "detector_matrix_i4.npy"
DEFAULT_DETECTOR_BLOCKS_FILENAME = "detector_blocks_i4.npy"
DEFAULT_CLEAN_ATSC_IQ_PATH = Path("generated/atsc/atsc_8vsb_complex64.cfile")
DEFAULT_QUANTIZE_HINT = (
    "python -m pilot_proxy.testbench.quantize "
    "--input-iq generated/atsc/atsc_8vsb_complex64.cfile "
    "--output-dir generated/detector_input "
    "--physical-channel 14"
)


class KernelSpecsLike(Protocol):
    K: int
    N: int


class KernelLike(Protocol):
    specs: KernelSpecsLike


def _missing_detector_input_message(path: Path) -> str:
    default_matrix = DEFAULT_DETECTOR_INPUT_DIR / DEFAULT_DETECTOR_MATRIX_FILENAME
    default_blocks = DEFAULT_DETECTOR_INPUT_DIR / DEFAULT_DETECTOR_BLOCKS_FILENAME
    details = [
        f"Input detector matrix does not exist: {path}",
        "",
        "Generate packed detector input before running detect.",
    ]
    if default_blocks.exists() and path == default_matrix:
        details.extend(
            [
                "",
                f"A batched detector-input file exists: {default_blocks}",
                f"You can pass --input-detector-matrix {default_blocks}",
            ]
        )
    elif DEFAULT_CLEAN_ATSC_IQ_PATH.exists():
        details.extend(
            [
                "",
                "A clean ATSC IQ file exists, so this usually means the quantize "
                "step has not been run:",
                f"  {DEFAULT_QUANTIZE_HINT}",
            ]
        )
    return "\n".join(details)


def _is_default_detector_matrix_path(path: Path) -> bool:
    return Path(path) == DEFAULT_DETECTOR_INPUT_DIR / DEFAULT_DETECTOR_MATRIX_FILENAME


def _resolve_detector_input_path(path: Path) -> Path:
    """Resolve the common single-block path to batched output when needed."""
    requested = Path(path)
    if requested.exists():
        return requested
    default_blocks = DEFAULT_DETECTOR_INPUT_DIR / DEFAULT_DETECTOR_BLOCKS_FILENAME
    if _is_default_detector_matrix_path(requested) and default_blocks.exists():
        return default_blocks
    if not requested.exists():
        raise SystemExit(_missing_detector_input_message(path))
    return requested


def _load_detector_input(path: Path) -> tuple[np.ndarray, int, int]:
    path = _resolve_detector_input_path(path)
    packed = np.load(path)
    if packed.ndim == 2:
        batch = 1
        rows, samples = packed.shape
        return np.ascontiguousarray(packed), batch, int(rows)
    if packed.ndim == 3:
        batch, rows, samples = packed.shape
        del samples
        return np.ascontiguousarray(packed), int(batch), int(rows)
    raise SystemExit(
        "input detector matrix must have shape (rows, detector_window_samples) "
        "or (batch, rows, detector_window_samples)."
    )


def _validate_kernel_inputs(
    *,
    packed: np.ndarray,
    weights: np.ndarray,
    kernel: KernelLike,
) -> None:
    """Validate packed detector input before handing pointers to CUDA."""
    if packed.ndim not in (2, 3):
        raise ValueError(
            "packed detector input must have shape "
            "(rows, detector_window_samples) or "
            "(batch, rows, detector_window_samples)."
        )
    if packed.dtype != np.dtype(np.int8):
        raise ValueError(
            "packed detector input must be dtype int8 for the locked 4+4 bit "
            f"format; got {packed.dtype}."
        )
    if packed.shape[-1] != int(kernel.specs.K):
        raise ValueError(
            "packed detector input has wrong detector-window length: "
            f"got {packed.shape[-1]}, expected {int(kernel.specs.K)}."
        )

    expected_weights_shape = (int(kernel.specs.N), int(kernel.specs.K))
    if tuple(weights.shape) != expected_weights_shape:
        raise ValueError(
            "weights have wrong shape: "
            f"got {tuple(weights.shape)}, expected {expected_weights_shape}."
        )
    if weights.dtype != np.dtype(np.int8):
        raise ValueError(
            "weights must be dtype int8 for the locked 4+4 bit format; "
            f"got {weights.dtype}."
        )


def detect_packed_detector_input(
    *,
    packed: np.ndarray,
    weights: np.ndarray,
    kernel: FStatKernel,
    emit_row_projections: bool = False,
) -> dict[str, Any]:
    """Run fixed-point detection and return exact power ratios.

    The positive-excess mask is norm-corrected: it compares against the H0
    zero-point ``null_power_ratio = 2*target_norm_sq/reference_norm_sum_sq`` implied by the
    supplied int4 weights (exactly, in integers), not against ``F > 1``.
    """
    _validate_kernel_inputs(packed=packed, weights=weights, kernel=kernel)
    if not getattr(kernel, "_has_powers_u64", False):
        raise RuntimeError(
            "Kernel library does not expose FStat_Compute_Powers_U64; "
            "rebuild libfstatistic.so with the current CUDA sources."
        )

    target_norm_sq, ref_lower_norm_sq, ref_upper_norm_sq = weight_term_norms_sq(
        weights
    )
    reference_norm_sum_sq = int(ref_lower_norm_sq + ref_upper_norm_sq)
    if target_norm_sq <= 0 or reference_norm_sum_sq <= 0:
        raise ValueError(
            "weights have a zero-power term (target_norm_sq="
            f"{target_norm_sq}, reference_norm_sum_sq={reference_norm_sum_sq}); "
            "an all-zero steering vector cannot form a detector."
        )
    null_power_ratio = null_power_ratio_from_weight_norms(target_norm_sq, reference_norm_sum_sq)

    import cupy as cp

    packed = np.ascontiguousarray(packed)
    if packed.ndim == 2:
        batch = 1
        rows = int(packed.shape[0])
        d_in = cp.asarray(packed)
        d_out = cp.zeros(batch, dtype=cp.float32)
        handle = kernel.create_raw(rows, d_in.data.ptr, d_out.data.ptr)
    elif packed.ndim == 3:
        batch = int(packed.shape[0])
        rows = int(packed.shape[1])
        d_in = cp.asarray(packed)
        d_out = cp.zeros(batch, dtype=cp.float32)
        handle = kernel.create_raw_batch(rows, batch, d_in.data.ptr, d_out.data.ptr)
    else:
        raise ValueError("packed must be a 2D or 3D detector matrix array.")

    d_powers = cp.zeros((batch, REFERENCE_WEIGHT_TERMS), dtype=cp.uint64)
    d_matched_filter_row_projections = None
    d_fine_powers_u64 = None
    if emit_row_projections:
        if not kernel.supports_row_projections():
            raise RuntimeError(
                "row sums requested but the kernel library does not expose "
                "FStat_Compute_RowSums_I32; rebuild libfstatistic.so from the "
                "v2 CUDA sources."
            )
        d_matched_filter_row_projections = cp.zeros(
            batch * REFERENCE_WEIGHT_TERMS * rows * 2, dtype=cp.int32
        )
        # The exact fine terms are only obtainable here, while the row sums are
        # still on the device. Recovering them later from the float ratio is
        # impossible, so emit them whenever the library can.
        # fxfft256 is frozen at 128 windows per stream, and this entry point
        # does not see the caller's frame size, so it can only derive the
        # stream count as rows // 128. Refuse to guess when that is not exact:
        # a wrong stream count would emit quietly meaningless powers.
        if (
            kernel.supports_fine_powers()
            and rows % FINE_WINDOWS_PER_STREAM == 0
        ):
            d_fine_powers_u64 = cp.zeros(
                batch * REFERENCE_WEIGHT_TERMS * FINE_BINS, dtype=cp.uint64
            )
    try:
        kernel.compute_powers_u64(handle, weights.ctypes.data, d_powers.data.ptr)
        if d_matched_filter_row_projections is not None:
            kernel.compute_row_projections_i32(
                handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
            )
            if d_fine_powers_u64 is not None:
                kernel.compute_fine_powers_u64(
                    d_matched_filter_row_projections.data.ptr,
                    rows // FINE_WINDOWS_PER_STREAM,
                    FINE_WINDOWS_PER_STREAM,
                    batch,
                    d_fine_powers_u64.data.ptr,
                )
        cp.cuda.Device().synchronize()
        powers = cp.asnumpy(d_powers).astype(np.uint64, copy=False)
    finally:
        kernel.destroy(handle)

    fine_powers_u64 = None
    if d_fine_powers_u64 is not None:
        fine_powers_u64 = cp.asnumpy(d_fine_powers_u64).astype(
            np.uint64, copy=False
        ).reshape(batch, REFERENCE_WEIGHT_TERMS, FINE_BINS)

    matched_filter_row_projections = None
    if d_matched_filter_row_projections is not None:
        matched_filter_row_projections = d_matched_filter_row_projections.reshape(batch, REFERENCE_WEIGHT_TERMS, rows, 2)
        # Exact all-bin marginal identity against the deployed u64 powers.
        # Any mismatch means stage-1 output is corrupt; fail immediately.
        from .fine_reduction import (
            check_coarse_power_marginal_identity,
            exact_coarse_power_by_term,
        )

        for b_idx in range(batch):
            marginal = exact_coarse_power_by_term(
                matched_filter_row_projections[b_idx], num_weight_terms=REFERENCE_WEIGHT_TERMS, xp=cp
            )
            check_coarse_power_marginal_identity(marginal, powers[b_idx])

    rows_out: list[dict[str, Any]] = []
    for idx in range(batch):
        p_target = int(powers[idx, REFERENCE_TARGET_TERM_INDEX])
        p_ref_lower = int(powers[idx, REFERENCE_LOWER_TERM_INDEX])
        p_ref_upper = int(powers[idx, REFERENCE_UPPER_TERM_INDEX])
        p_ref_sum = int(p_ref_lower + p_ref_upper)
        coarse_power_ratio = float(
            power_terms_to_coarse_power_ratio(p_target, p_ref_sum)
        )
        normalized_coarse_power_ratio_db = float(
            power_terms_to_normalized_coarse_power_ratio_db(
                p_target,
                p_ref_sum,
                target_norm_sq=target_norm_sq,
                reference_norm_sum_sq=reference_norm_sum_sq,
            )
        )
        raw_pilot_excess = float(
            power_terms_to_raw_pilot_excess(p_target, p_ref_sum)
        )
        normalized_pilot_excess = float(
            power_terms_to_normalized_pilot_excess(
                p_target,
                p_ref_sum,
                target_norm_sq=target_norm_sq,
                reference_norm_sum_sq=reference_norm_sum_sq,
            )
        )
        pilot_excess_db = float(
            normalized_pilot_excess_to_db(normalized_pilot_excess)
        )
        positive_excess = normalized_positive_excess(
            p_target,
            p_ref_sum,
            target_norm_sq=target_norm_sq,
            reference_norm_sum_sq=reference_norm_sum_sq,
        )
        rows_out.append(
            {
                "block_index": int(idx),
                "mask": positive_excess,
                "normalized_positive_excess_mask": positive_excess,
                "p_target_u64": p_target,
                "p_ref_lower_u64": p_ref_lower,
                "p_ref_upper_u64": p_ref_upper,
                "p_ref_sum_u64": p_ref_sum,
                "coarse_power_ratio": coarse_power_ratio,
                "normalized_coarse_power_ratio_db": normalized_coarse_power_ratio_db,
                "raw_pilot_excess": raw_pilot_excess,
                "normalized_pilot_excess": float(normalized_pilot_excess),
                "pilot_excess_db": pilot_excess_db,
            }
        )

    return {
        "batch": int(batch),
        "detector_rows_per_block": int(rows),
        "matched_filter_row_projections": matched_filter_row_projections,
        "fine_powers_u64": fine_powers_u64,
        "mask_source": "normalized_positive_excess_decision",
        "mask_rule": NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
        "target_norm_sq": int(target_norm_sq),
        "reference_norm_sum_sq": int(reference_norm_sum_sq),
        "null_power_ratio": float(null_power_ratio),
        "results": rows_out,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-point DTV pilot detection on packed detector input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-detector-matrix",
        type=Path,
        default=DEFAULT_DETECTOR_INPUT_DIR / DEFAULT_DETECTOR_MATRIX_FILENAME,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("generated/detections/detect.json"),
    )
    parser.add_argument("--physical-channel", type=int, default=None,
                        help="ATSC physical channel of the packed matrix. Default: "
                             "read from the metadata.json sidecar quantize wrote "
                             "next to the matrix; an explicit value must agree "
                             "with it. Without a sidecar, one of --physical-channel/"
                             "--dtv-pilot-mhz is required.")
    parser.add_argument("--dtv-pilot-mhz", type=float, default=None)
    parser.add_argument(
        "--pilot-frequency-tolerance-hz",
        type=float,
        default=DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    )
    parser.add_argument(
        "--pilot-below-data-db",
        dest="pilot_below_data_db",
        type=float,
        default=PILOT_BELOW_DATA_DB,
        help="Positive dB offset: ATSC pilot power below average data-shelf power.",
    )
    parser.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    parser.add_argument(
        "--pilot-capture-efficiency",
        type=float,
        default=PILOT_CAPTURE_EFFICIENCY,
    )
    parser.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    parser.add_argument(
        "--frame-size-samples",
        type=int,
        default=None,
        help=(
            "Frame size per input stream for result metadata. If omitted, "
            "detect assumes one input stream and derives frame size from rows."
        ),
    )
    parser.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=1,
        help="Number of input streams flattened into the detector rows.",
    )
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    return parser


def _sidecar_pilot_hz(input_path: Path) -> tuple[float | None, Path]:
    """The pilot frequency quantize recorded in the matrix's ``metadata.json``
    sidecar, or None when the sidecar is absent, unreadable, or lacks a finite
    ``dtv_pilot_hz``."""
    sidecar = Path(input_path).parent / "metadata.json"
    try:
        with open(sidecar) as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return None, sidecar
    value = meta.get("dtv_pilot_hz") if isinstance(meta, dict) else None
    try:
        hz = float(value)
    except (TypeError, ValueError):
        return None, sidecar
    if not math.isfinite(hz) or hz <= 0:
        return None, sidecar
    return hz, sidecar


def _resolve_pilot_request(
    physical_channel: int | None,
    dtv_pilot_mhz: float | None,
    input_path: Path,
    tolerance_hz: float,
) -> tuple[int | None, float | None]:
    """Resolve the pilot identity for a packed detector matrix.

    The sidecar records what quantize actually packed, so it is authoritative:
    with no flags the pilot comes from the sidecar; an explicit flag must agree
    with it (a mismatched channel would run the detector against the wrong
    weight row); with neither a sidecar nor a flag, refuse rather than guess."""
    try:
        tolerance = float(tolerance_hz)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "detect: --pilot-frequency-tolerance-hz must be finite and "
            "nonnegative"
        ) from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise SystemExit(
            "detect: --pilot-frequency-tolerance-hz must be finite and "
            "nonnegative"
        )
    if physical_channel is not None and dtv_pilot_mhz is not None:
        raise SystemExit("Use either --physical-channel or --dtv-pilot-mhz, not both.")
    sidecar_hz, sidecar_path = _sidecar_pilot_hz(input_path)
    if physical_channel is None and dtv_pilot_mhz is None:
        if sidecar_hz is None:
            raise SystemExit(
                "detect: no --physical-channel/--dtv-pilot-mhz given and no readable "
                f"metadata.json sidecar next to {input_path} (quantize writes one "
                "with dtv_pilot_hz). Pass the pilot identity explicitly."
            )
        print(
            f"Pilot frequency from sidecar {sidecar_path}: "
            f"{sidecar_hz / HZ_PER_MHZ:.6f} MHz"
        )
        return None, sidecar_hz / HZ_PER_MHZ
    if sidecar_hz is not None:
        requested_hz = (
            physical_channel_to_pilot_hz(physical_channel)
            if physical_channel is not None
            else float(dtv_pilot_mhz) * HZ_PER_MHZ
        )
        if abs(requested_hz - sidecar_hz) > tolerance:
            raise SystemExit(
                f"detect: requested pilot {requested_hz / HZ_PER_MHZ:.6f} MHz "
                f"disagrees with the matrix sidecar {sidecar_path} "
                f"({sidecar_hz / HZ_PER_MHZ:.6f} MHz). The sidecar records what "
                f"quantize packed; fix the flag or requantize."
            )
    return physical_channel, dtv_pilot_mhz


def _resolve_layout_metadata(
    *,
    rows: int,
    detector_window_samples: int,
    frame_size_samples: int | None,
    num_input_streams: int,
) -> dict[str, int | str]:
    streams = int(num_input_streams)
    window = int(detector_window_samples)
    if streams <= 0:
        raise SystemExit("--num-input-streams must be positive.")
    if frame_size_samples is None:
        if int(rows) % streams != 0:
            raise SystemExit(
                "Cannot derive frame_size_samples: detector rows are not divisible "
                f"by num_input_streams ({rows} rows, {streams} streams)."
            )
        frame = int(rows) // streams * window
    else:
        frame = int(frame_size_samples)
    if frame <= 0:
        raise SystemExit("--frame-size-samples must be positive.")
    if frame % window != 0:
        raise SystemExit("--frame-size-samples must be a multiple of kernel K.")
    expected_rows = streams * (frame // window)
    if expected_rows != int(rows):
        raise SystemExit(
            "Packed detector matrix row count does not match layout metadata: "
            f"rows={rows}, expected={expected_rows} from "
            f"frame_size_samples={frame}, num_input_streams={streams}, K={window}."
        )
    return {
        "frame_size_samples": int(frame),
        "num_input_streams": int(streams),
        "detector_window_samples": int(window),
        "windows_per_stream": int(frame // window),
        "detector_rows_per_frame": int(expected_rows),
        "detector_rows_per_block": int(expected_rows),
        "combine_mode": COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_input_path = _resolve_detector_input_path(args.input_detector_matrix)
    if resolved_input_path != args.input_detector_matrix:
        print(f"Using detector input {resolved_input_path}")
    args.physical_channel, args.dtv_pilot_mhz = _resolve_pilot_request(
        args.physical_channel,
        args.dtv_pilot_mhz,
        Path(resolved_input_path),
        float(args.pilot_frequency_tolerance_hz),
    )
    packed, batch, rows = _load_detector_input(resolved_input_path)
    kernel = FStatKernel(args.lib_path)
    weights_bank = DetectorWeightBank(
        explicit_path=args.weights_path,
        expected_kernel=kernel.specs,
    )
    if args.physical_channel is not None:
        dtv_pilot_mhz = (
            physical_channel_to_pilot_hz(args.physical_channel) / HZ_PER_MHZ
        )
        selected_weight_layout = weights_bank.layout_for_physical_channel(
            int(args.physical_channel),
            tolerance_hz=float(args.pilot_frequency_tolerance_hz),
        )
        weights, valid = weights_bank.get_weights_for_physical_channel(
            int(args.physical_channel),
            tolerance_hz=float(args.pilot_frequency_tolerance_hz),
        )
    else:
        dtv_pilot_mhz = float(args.dtv_pilot_mhz)
        selected_weight_layout = weights_bank.layout_for_pilot_frequency(
            dtv_pilot_mhz,
            tolerance_hz=float(args.pilot_frequency_tolerance_hz),
        )
        weights, valid = weights_bank.get_weights_for_pilot_frequency(
            dtv_pilot_mhz,
            tolerance_hz=float(args.pilot_frequency_tolerance_hz),
        )
    if weights is None or not valid:
        raise SystemExit(f"No valid detector weights for pilot {dtv_pilot_mhz:.6f} MHz.")

    layout = _resolve_layout_metadata(
        rows=int(rows),
        detector_window_samples=int(kernel.specs.K),
        frame_size_samples=args.frame_size_samples,
        num_input_streams=int(args.num_input_streams),
    )
    detection = detect_packed_detector_input(
        packed=packed,
        weights=weights,
        kernel=kernel,
    )
    detection_results = cast(list[dict[str, Any]], detection["results"])
    for row in detection_results:
        row["estimated_data_shelf_snr_db"] = float(
            pilot_excess_db_to_data_shelf_snr_db(
                row["pilot_excess_db"],
                pilot_below_data_db=float(args.pilot_below_data_db),
                bin_enbw_hz=float(args.bin_enbw_hz),
                dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
                pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            )
        )

    payload = {
        "schema_version": "pilotproxy_detection_result_v1",
        "result_schema_version": RESULT_SCHEMA_TOKEN,
        "input_detector_matrix": str(args.input_detector_matrix),
        "resolved_input_detector_matrix": str(resolved_input_path),
        "dtv_pilot_mhz": float(dtv_pilot_mhz),
        "physical_channel": None
        if args.physical_channel is None
        else int(args.physical_channel),
        "selected_weight_layout": selected_weight_layout,
        "result_schema": result_schema_object(
            frame_size_samples=int(layout["frame_size_samples"]),
            num_input_streams=int(layout["num_input_streams"]),
            detector_window_samples=int(layout["detector_window_samples"]),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            bin_enbw_hz=float(args.bin_enbw_hz),
            pilot_below_data_db=float(args.pilot_below_data_db),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            threshold=None,
            reference_offset_bins=int(weights_bank.reference_offset_bins),
            num_selected_channels=1,
        ),
        "layout": layout,
        "kernel_version": kernel.version.as_string(),
        "kernel_specs": kernel.specs.as_descriptive_dict(),
        **detection,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_strict(args.output_json, payload, indent=2)
    print(f"Wrote {args.output_json}")
    positive_excess_set = sum(
        int(row["normalized_positive_excess_mask"]) for row in detection_results
    )
    coarse_power_ratio_values = np.asarray(
        [float(row["coarse_power_ratio"]) for row in detection_results],
        dtype=np.float64,
    )
    snr_shelf_values = np.asarray(
        [float(row["estimated_data_shelf_snr_db"]) for row in detection_results],
        dtype=np.float64,
    )
    coarse_power_ratio_mean = float(np.nanmean(coarse_power_ratio_values))
    estimated_data_shelf_snr_db_mean = float(np.nanmean(snr_shelf_values))
    print(
        "blocks, positive_excess_set, coarse_power_ratio_mean, "
        "estimated_data_shelf_snr_db_mean"
    )
    print(
        f"{int(cast(int, detection['batch']))}, "
        f"{int(positive_excess_set)}, "
        f"{coarse_power_ratio_mean:.9g}, "
        f"{estimated_data_shelf_snr_db_mean:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

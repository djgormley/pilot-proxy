#!/usr/bin/env python3
# coding=utf-8
"""Convert a GNU Radio ATSC IQ file into reference-channelizer detector input."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz  # noqa: E402
from pilot_proxy.integration import QUANTIZATION_SCALE_MODE_GLOBAL  # noqa: E402
from pilot_proxy.integration.packing import (  # noqa: E402
    estimate_complex_scale,
    pack_channelized_streams_for_detector,
)
from pilot_proxy.json_utils import write_json_strict  # noqa: E402
from pilot_proxy.paths import GENERATED_DIR, resolve_user_path  # noqa: E402
from pilot_proxy.reference_channelizer import (  # noqa: E402
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
    ReferenceChannelizerSpec,
    apply_reference_archive_phase_convention,
    channelize_real_blocks_to_reference_channels,
    complex_envelope_to_real_adc_blocks,
    nearest_reference_channel_index,
    reference_channel_frequencies_hz,
    sinc_hamming_pfb_response,
)

GNU_RADIO_ATSC_SYMBOL_RATE_HZ = 4_500_000.0 / 286.0 * 684.0
ATSC_CHANNEL_WIDTH_HZ = 6.0e6
ATSC_PILOT_OFFSET_HZ = 309_441.0
DEFAULT_DTV_PILOT_HZ = 470_309_441.0
HZ_PER_MHZ = 1.0e6
HALF_SCALE = 2.0
LOCKED_DETECTOR_WINDOW_SAMPLES = 128
LOCKED_BITS_PER_COMPONENT = 4
DEFAULT_FRAME_SIZE_SAMPLES = 16_384
DEFAULT_NUM_BLOCKS = 1
DEFAULT_NUM_INPUT_STREAMS = 1
DEFAULT_CHANNEL_HALF_WIDTH = 0
DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_OUTPUT_CHUNK_SAMPLES = 1024
DEFAULT_ADC_CHUNK_SAMPLES = 1_048_576


def _read_complex64(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.complex64)
    if data.size == 0:
        raise SystemExit(f"Input IQ file is empty: {path}")
    return np.ascontiguousarray(data)


def _parse_channel_indices(values: list[int] | None) -> list[int]:
    if not values:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _resolve_rf_center_hz(args: argparse.Namespace) -> float:
    if args.rf_center_mhz is not None:
        return float(args.rf_center_mhz) * HZ_PER_MHZ
    pilot_hz = float(args.dtv_pilot_mhz) * HZ_PER_MHZ
    return pilot_hz + (
        ATSC_CHANNEL_WIDTH_HZ / HALF_SCALE - float(args.atsc_pilot_offset_hz)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_strict(path, payload, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a GNU Radio ATSC complex64 .cfile, model the reference 4-tap "
            "2048-point PFB, select coarse channels, and write signed 4+4 bit "
            "detector input."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-iq",
        type=Path,
        required=True,
        help="GNU Radio ATSC complex64 IQ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GENERATED_DIR / "detector_input",
    )
    parser.add_argument(
        "--iq-sample-rate-hz",
        type=float,
        default=GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    )
    parser.add_argument(
        "--adc-sample-rate-hz",
        type=float,
        default=REFERENCE_ADC_SAMPLE_RATE_HZ,
    )
    parser.add_argument(
        "--band-lower-mhz",
        type=float,
        default=REFERENCE_BAND_LOWER_HZ / HZ_PER_MHZ,
    )
    parser.add_argument(
        "--dtv-pilot-mhz",
        type=float,
        default=DEFAULT_DTV_PILOT_HZ / HZ_PER_MHZ,
        help="Absolute DTV pilot frequency used for default channel selection.",
    )
    parser.add_argument("--physical-channel", type=int, default=None)
    parser.add_argument(
        "--rf-center-mhz",
        type=float,
        default=None,
        help=(
            "Absolute RF center frequency represented by 0 Hz in the ATSC IQ. "
            "Defaults to dtv_pilot + 3 MHz - atsc_pilot_offset."
        ),
    )
    parser.add_argument(
        "--atsc-pilot-offset-hz",
        type=float,
        default=ATSC_PILOT_OFFSET_HZ,
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        action="append",
        default=None,
        help="Reference coarse channel index to extract. Repeat to extract multiple.",
    )
    parser.add_argument(
        "--channel-half-width",
        type=int,
        default=DEFAULT_CHANNEL_HALF_WIDTH,
        help="Also extract this many neighboring channels around the selected channel.",
    )
    parser.add_argument(
        "--frame-size-samples",
        dest="frame_size_samples",
        type=int,
        default=DEFAULT_FRAME_SIZE_SAMPLES,
        help="Frame size, in channelized samples, to pack per detector block.",
    )
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=DEFAULT_NUM_INPUT_STREAMS,
        help=(
            "Number of independent input streams/feeds to combine into one "
            "detector decision. With one input IQ file, the selected "
            "channelized stream is replicated into this many stream slots."
        ),
    )
    parser.add_argument(
        "--experimental-detector-window-samples",
        dest="detector_window_samples",
        type=int,
        default=LOCKED_DETECTOR_WINDOW_SAMPLES,
        help="Advanced: select a detector window supported by the current kernel.",
    )
    parser.add_argument(
        "--spectral-sense",
        choices=["normal", "inverted"],
        default="normal",
        help="Detector-window spectral-sense correction to apply before packing.",
    )
    parser.add_argument(
        "--reference-archive-phase",
        dest="reference_archive_phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the conj/(-1)^n archive phase convention before packing. "
            "Its half-rate factor displaces the pilot by half a coarse "
            "channel relative to the shipped weight layouts; leave this off "
            "except to diagnose coordinate conventions."
        ),
    )
    parser.add_argument(
        "--experimental-bits",
        dest="bits",
        type=int,
        default=LOCKED_BITS_PER_COMPONENT,
        help="Advanced: the current kernel requires the locked 4+4-bit format.",
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--clip-sigma", type=float, default=DEFAULT_CLIP_SIGMA)
    parser.add_argument(
        "--save-channelized",
        action="store_true",
        help="Also save selected complex64 channelized streams before int4 packing.",
    )
    parser.add_argument(
        "--output-chunk-samples",
        type=int,
        default=DEFAULT_OUTPUT_CHUNK_SAMPLES,
        help="PFB output rows to FFT per chunk.",
    )
    parser.add_argument(
        "--adc-chunk-samples",
        type=int,
        default=DEFAULT_ADC_CHUNK_SAMPLES,
        help="ADC samples to synthesize per interpolation chunk.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    caller_cwd = Path.cwd()
    args.input_iq = resolve_user_path(args.input_iq, relative_to=caller_cwd)
    args.output_dir = resolve_user_path(args.output_dir, relative_to=caller_cwd)

    if args.bits != LOCKED_BITS_PER_COMPONENT:
        raise SystemExit("This converter is intended for the locked 4+4 bit format.")
    if args.detector_window_samples != LOCKED_DETECTOR_WINDOW_SAMPLES:
        raise SystemExit(
            "This converter is intended for the locked 128-sample detector "
            "window used by the shipped kernel and weights."
        )
    if args.physical_channel is not None:
        args.dtv_pilot_mhz = physical_channel_to_pilot_hz(
            int(args.physical_channel)
        ) / HZ_PER_MHZ
    if args.frame_size_samples <= 0 or args.num_blocks <= 0:
        raise SystemExit("--frame-size-samples and --num-blocks must be positive.")
    if args.num_input_streams <= 0:
        raise SystemExit("--num-input-streams must be positive.")
    if args.frame_size_samples % args.detector_window_samples != 0:
        raise SystemExit(
            "--frame-size-samples must be an integer multiple of the locked "
            "128-sample detector window."
        )
    if args.channel_half_width < 0:
        raise SystemExit("--channel-half-width must be non-negative.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    iq = _read_complex64(args.input_iq)
    band_lower_hz = float(args.band_lower_mhz) * HZ_PER_MHZ
    rf_center_hz = _resolve_rf_center_hz(args)
    spec = ReferenceChannelizerSpec(
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        band_lower_hz=band_lower_hz,
    )
    channel_freqs = reference_channel_frequencies_hz(spec)

    base_channels = _parse_channel_indices(args.channel_index)
    if not base_channels:
        base_channels = [
            nearest_reference_channel_index(float(args.dtv_pilot_mhz) * HZ_PER_MHZ, spec)
        ]
    if args.channel_half_width:
        expanded: list[int] = []
        for idx in base_channels:
            lo = max(0, idx - int(args.channel_half_width))
            hi = min(spec.num_channels - 1, idx + int(args.channel_half_width))
            expanded.extend(range(lo, hi + 1))
        base_channels = sorted(set(expanded))

    n_output = int(args.frame_size_samples) * int(args.num_blocks)
    n_blocks = n_output + REFERENCE_PFB_TAPS - 1
    raw_blocks = complex_envelope_to_real_adc_blocks(
        iq,
        iq_sample_rate_hz=float(args.iq_sample_rate_hz),
        rf_center_hz=rf_center_hz,
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        band_lower_hz=band_lower_hz,
        n_blocks=n_blocks,
        block_size=REFERENCE_PFB_FFT_SIZE,
        chunk_samples=int(args.adc_chunk_samples),
    )
    response = sinc_hamming_pfb_response(REFERENCE_PFB_TAPS, REFERENCE_PFB_FFT_SIZE)
    channel_streams = channelize_real_blocks_to_reference_channels(
        raw_blocks,
        channel_indices=base_channels,
        response=response,
        spec=spec,
        output_chunk_samples=int(args.output_chunk_samples),
    )
    if args.reference_archive_phase:
        channel_streams = apply_reference_archive_phase_convention(channel_streams)
    feed_channel_streams = np.repeat(
        channel_streams[np.newaxis, :, :],
        int(args.num_input_streams),
        axis=0,
    )
    if args.scale is not None:
        scale = float(args.scale)
    else:
        try:
            scale = estimate_complex_scale(
                feed_channel_streams,
                bits_per_component=args.bits,
                clip_sigma=args.clip_sigma,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    packed_input = pack_channelized_streams_for_detector(
        feed_channel_streams,
        frame_size_samples=int(args.frame_size_samples),
        detector_window_samples=int(args.detector_window_samples),
        spectral_sense=str(args.spectral_sense),
        quantization_scale_mode=QUANTIZATION_SCALE_MODE_GLOBAL,
        clip_sigma=float(args.clip_sigma),
        bits_per_component=int(args.bits),
        num_blocks=int(args.num_blocks),
        scale=scale,
        selected_channel_indices=[int(idx) for idx in base_channels],
        physical_channel=(
            None if args.physical_channel is None else int(args.physical_channel)
        ),
    )
    streams = packed_input.flattened_streams
    stream_map = packed_input.stream_map
    input_layout = packed_input.input_layout
    packed_blocks = np.ascontiguousarray(packed_input.packed)

    packed_path = output_dir / "detector_blocks_i4.npy"
    np.save(packed_path, packed_blocks)
    single_block_path = None
    if packed_blocks.shape[0] == 1:
        single_block_path = output_dir / "detector_matrix_i4.npy"
        np.save(single_block_path, packed_blocks[0])

    channelized_path = None
    if args.save_channelized:
        channelized_path = output_dir / "channelized_streams_complex64.npy"
        np.save(channelized_path, streams)

    selected_freqs = channel_freqs[np.asarray(base_channels, dtype=np.int64)]
    metadata = {
        "schema_version": "pilotproxy_atsc_detector_input_v1",
        "input_iq": str(args.input_iq),
        "iq_sample_rate_hz": float(args.iq_sample_rate_hz),
        "adc_sample_rate_hz": float(args.adc_sample_rate_hz),
        "band_lower_hz": float(band_lower_hz),
        "rf_center_hz": float(rf_center_hz),
        "dtv_pilot_hz": float(args.dtv_pilot_mhz) * HZ_PER_MHZ,
        "atsc_pilot_offset_hz": float(args.atsc_pilot_offset_hz),
        "pfb_taps": REFERENCE_PFB_TAPS,
        "pfb_fft_size": REFERENCE_PFB_FFT_SIZE,
        "pfb_response": "sinc_hamming",
        "selected_channel_indices": [int(idx) for idx in base_channels],
        "selected_channel_frequency_hz": [float(freq) for freq in selected_freqs],
        "selected_channel_rfft_bins": [int(idx) + 1 for idx in base_channels],
        "input_layout": input_layout,
        "stream_map": stream_map,
        "input_stream_model": "replicated_single_input_iq",
        "num_blocks": int(args.num_blocks),
        "spectral_sense": str(args.spectral_sense),
        "reference_archive_phase": bool(args.reference_archive_phase),
        "bits_per_component": int(args.bits),
        "packed_format": (
            "signed two's-complement int4 complex in int8: "
            "high nibble real, low nibble imaginary"
        ),
        "quantization_scale": float(scale),
        "quantization": packed_input.quantization,
        "clip_sigma": float(args.clip_sigma),
        "detector_blocks_shape": [int(dim) for dim in packed_blocks.shape],
        "detector_blocks_path": str(packed_path),
        "detector_matrix_path": (
            None if single_block_path is None else str(single_block_path)
        ),
        "channelized_streams_path": None
        if channelized_path is None
        else str(channelized_path),
    }
    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, metadata)

    print(f"Wrote {packed_path}")
    if single_block_path is not None:
        print(f"Wrote {single_block_path}")
    if channelized_path is not None:
        print(f"Wrote {channelized_path}")
    print(f"Wrote {metadata_path}")
    print(f"detector_blocks_shape={packed_blocks.shape}")
    print(f"selected_channel_indices={base_channels}")
    print(
        "selected_channel_frequency_mhz="
        f"{[round(float(freq) / HZ_PER_MHZ, 6) for freq in selected_freqs]}"
    )
    print(f"quantization_scale={scale:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

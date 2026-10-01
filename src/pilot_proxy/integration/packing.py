# coding=utf-8
"""Canonical adapter from receiver streams to packed detector rows."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pilot_proxy.detector_constants import (
    DEFAULT_DETECTOR_WINDOW_SAMPLES,
    LOCKED_SAMPLE_BITS_PER_COMPONENT as DEFAULT_BITS_PER_COMPONENT,
)
from pilot_proxy.detector_geometry import (
    DetectorFrameLayout,
    apply_spectral_sense_to_detector_matrix,
    build_stream_map,
    flatten_feed_channel_streams,
    stream_time_block_to_detector_matrix,
)
from pilot_proxy.detector_reference import quantize_complex_numpy

from .schemas import (
    COMBINE_MODE_COMBINED_STREAMS,
    COMBINE_MODE_PER_STREAM_DIAGNOSTIC,
    QUANTIZATION_SCALE_MODE_GLOBAL,
    QUANTIZATION_SCALE_MODE_PER_STREAM,
    QUANTIZATION_SCALE_MODE_PROVIDED,
)
from .stream_layout import quantization_metadata

DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_BLOCK_STEP_MULTIPLIER = 1


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive.")
    return result


@dataclass(frozen=True)
class PackedDetectorInput:
    """Packed rows plus metadata explaining how receiver streams were flattened."""

    packed: np.ndarray
    input_layout: dict[str, Any]
    stream_map: list[dict[str, Any]]
    quantization: dict[str, Any]
    flattened_streams: np.ndarray


def estimate_complex_scale(
    values: np.ndarray,
    *,
    bits_per_component: int,
    clip_sigma: float,
) -> float:
    """Estimate the shared complex quantization scale used by all adapters.

    The detector quantizes real and imaginary components onto the same signed
    integer grid.  Prefer the real-component spread, then fall back to the
    imaginary and magnitude spreads for degenerate synthetic inputs.
    """
    arr = np.asarray(values)
    sigma = float(np.std(arr.real))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(arr.imag))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(np.abs(arr)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("could not estimate a positive quantization scale.")
    max_int = (1 << (int(bits_per_component) - 1)) - 1
    return float(max_int) / (float(clip_sigma) * sigma)


def _clip_fraction(
    values: np.ndarray,
    *,
    scale: float,
    bits_per_component: int,
) -> float:
    max_int = (1 << (int(bits_per_component) - 1)) - 1
    scaled = np.asarray(values) * float(scale)
    clipped = (
        (scaled.real > max_int)
        | (scaled.real < -max_int)
        | (scaled.imag > max_int)
        | (scaled.imag < -max_int)
    )
    return float(np.mean(clipped))


def _resolve_scale_by_stream(
    streams: np.ndarray,
    *,
    mode: str,
    bits_per_component: int,
    clip_sigma: float,
    scale: float | None,
    scale_by_stream: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    num_streams = int(streams.shape[0])
    if mode == QUANTIZATION_SCALE_MODE_GLOBAL:
        global_scale = (
            float(scale)
            if scale is not None
            else estimate_complex_scale(
                streams,
                bits_per_component=bits_per_component,
                clip_sigma=clip_sigma,
            )
        )
        return np.full(num_streams, global_scale, dtype=np.float64)
    if mode == QUANTIZATION_SCALE_MODE_PER_STREAM:
        return np.asarray(
            [
                estimate_complex_scale(
                    streams[index],
                    bits_per_component=bits_per_component,
                    clip_sigma=clip_sigma,
                )
                for index in range(num_streams)
            ],
            dtype=np.float64,
        )
    if mode == QUANTIZATION_SCALE_MODE_PROVIDED:
        if scale_by_stream is None:
            if scale is None:
                raise ValueError(
                    "provided scale mode requires scale or scale_by_stream."
                )
            return np.full(num_streams, float(scale), dtype=np.float64)
        scales = np.asarray(scale_by_stream, dtype=np.float64)
        if scales.shape != (num_streams,):
            raise ValueError(
                "scale_by_stream must have shape "
                f"({num_streams},); got {scales.shape}."
            )
        return scales
    raise ValueError(f"unsupported quantization scale mode: {mode!r}")


def _quantize_rows_by_stream(
    detector_matrix: np.ndarray,
    *,
    windows_per_stream: int,
    scale_by_stream: np.ndarray,
    bits_per_component: int,
) -> np.ndarray:
    if np.all(scale_by_stream == scale_by_stream[0]):
        return quantize_complex_numpy(
            detector_matrix,
            bits_per_component,
            float(scale_by_stream[0]),
        )
    first_stop = int(windows_per_stream)
    first_rows = quantize_complex_numpy(
        detector_matrix[:first_stop],
        bits_per_component,
        float(scale_by_stream[0]),
    )
    packed = np.empty(detector_matrix.shape, dtype=first_rows.dtype)
    packed[:first_stop] = first_rows
    for stream_index, stream_scale in enumerate(scale_by_stream[1:], start=1):
        start = stream_index * int(windows_per_stream)
        stop = start + int(windows_per_stream)
        packed[start:stop] = quantize_complex_numpy(
            detector_matrix[start:stop],
            bits_per_component,
            float(stream_scale),
        )
    return np.ascontiguousarray(packed)


def _default_stream_map(
    *,
    num_input_streams: int,
    num_selected_channels: int,
    selected_channel_indices: Sequence[int] | None,
    physical_channel: int | None,
) -> list[dict[str, Any]]:
    if selected_channel_indices is None:
        channels = list(range(int(num_selected_channels)))
    else:
        channels = [int(index) for index in selected_channel_indices]
    return [
        dict(item)
        for item in build_stream_map(
            num_feeds=int(num_input_streams),
            selected_channel_indices=channels,
            physical_channel=physical_channel,
        )
    ]


def pack_channelized_streams_for_detector(
    feed_channel_streams: np.ndarray,
    *,
    frame_size_samples: int,
    detector_window_samples: int = DEFAULT_DETECTOR_WINDOW_SAMPLES,
    spectral_sense: str = "normal",
    quantization_scale_mode: str = QUANTIZATION_SCALE_MODE_GLOBAL,
    clip_sigma: float = DEFAULT_CLIP_SIGMA,
    combine_mode: str = COMBINE_MODE_COMBINED_STREAMS,
    bits_per_component: int = DEFAULT_BITS_PER_COMPONENT,
    num_blocks: int = 1,
    block_step_samples: int | None = None,
    scale: float | None = None,
    scale_by_stream: Sequence[float] | np.ndarray | None = None,
    selected_channel_indices: Sequence[int] | None = None,
    physical_channel: int | None = None,
    stream_map: list[dict[str, Any]] | None = None,
) -> PackedDetectorInput:
    """Pack feed/channel/time streams into detector rows.

    Combined mode returns one detector row matrix per batch. Per-stream
    diagnostic mode returns a separate row matrix for each input stream.
    """
    frame_size_samples = _positive_integer(frame_size_samples, "frame_size_samples")
    detector_window_samples = _positive_integer(detector_window_samples, "detector_window_samples")
    num_blocks = _positive_integer(num_blocks, "num_blocks")
    if block_step_samples is not None:
        block_step_samples = _positive_integer(block_step_samples, "block_step_samples")
    arr = np.asarray(feed_channel_streams)
    if arr.ndim != 3:
        raise ValueError(
            "feed_channel_streams must have shape "
            "(num_input_streams, num_selected_channels, time)."
        )
    num_input_streams, num_selected_channels, total_samples = arr.shape
    if total_samples < int(frame_size_samples):
        raise ValueError(
            "feed_channel_streams does not contain one full frame: "
            f"{total_samples} < {int(frame_size_samples)}."
        )
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be positive.")
    step = (
        int(frame_size_samples) * DEFAULT_BLOCK_STEP_MULTIPLIER
        if block_step_samples is None
        else int(block_step_samples)
    )
    if step <= 0:
        raise ValueError("block_step_samples must be positive.")

    layout = DetectorFrameLayout(
        frame_size_samples=int(frame_size_samples),
        detector_window_samples=int(detector_window_samples),
        num_input_streams=int(num_input_streams),
        num_selected_channels=int(num_selected_channels),
        combine_mode=combine_mode,
    )
    flattened = flatten_feed_channel_streams(arr)
    scales = _resolve_scale_by_stream(
        flattened,
        mode=quantization_scale_mode,
        bits_per_component=int(bits_per_component),
        clip_sigma=float(clip_sigma),
        scale=scale,
        scale_by_stream=scale_by_stream,
    )

    combined_blocks: list[np.ndarray] = []
    diagnostic_blocks: list[np.ndarray] = []
    for block_index in range(int(num_blocks)):
        start = block_index * step
        stop = start + int(frame_size_samples)
        if stop > flattened.shape[1]:
            raise ValueError(
                "not enough time samples to build requested detector blocks: "
                f"block_index={block_index}, stop={stop}, "
                f"available={flattened.shape[1]}"
            )
        detector_matrix = stream_time_block_to_detector_matrix(
            flattened[:, start:stop],
            detector_window_samples=int(detector_window_samples),
        )
        detector_matrix = apply_spectral_sense_to_detector_matrix(
            detector_matrix,
            spectral_sense=spectral_sense,
        )
        packed_matrix = _quantize_rows_by_stream(
            detector_matrix,
            windows_per_stream=int(layout.windows_per_stream),
            scale_by_stream=scales,
            bits_per_component=int(bits_per_component),
        )
        if combine_mode == COMBINE_MODE_PER_STREAM_DIAGNOSTIC:
            diagnostic_blocks.extend(
                [
                    packed_matrix[
                        stream_index
                        * int(layout.windows_per_stream) : (stream_index + 1)
                        * int(layout.windows_per_stream)
                    ]
                    for stream_index in range(int(layout.num_streams))
                ]
            )
        else:
            combined_blocks.append(packed_matrix)

    if combine_mode == COMBINE_MODE_PER_STREAM_DIAGNOSTIC:
        packed = np.ascontiguousarray(np.stack(diagnostic_blocks, axis=0))
    else:
        packed = np.ascontiguousarray(np.stack(combined_blocks, axis=0))

    clip_fraction = np.asarray(
        [
            _clip_fraction(
                flattened[index],
                scale=float(scales[index]),
                bits_per_component=int(bits_per_component),
            )
            for index in range(int(layout.num_streams))
        ],
        dtype=np.float64,
    )
    return PackedDetectorInput(
        packed=packed,
        input_layout=layout.to_dict(),
        stream_map=(
            [dict(item) for item in stream_map]
            if stream_map is not None
            else _default_stream_map(
                num_input_streams=int(num_input_streams),
                num_selected_channels=int(num_selected_channels),
                selected_channel_indices=selected_channel_indices,
                physical_channel=physical_channel,
            )
        ),
        quantization=quantization_metadata(
            mode=quantization_scale_mode,
            bits_per_component=int(bits_per_component),
            clip_sigma=float(clip_sigma),
            scale_by_stream=scales,
            clip_fraction_by_stream=clip_fraction,
        ),
        flattened_streams=flattened,
    )


__all__ = [
    "PackedDetectorInput",
    "estimate_complex_scale",
    "pack_channelized_streams_for_detector",
]

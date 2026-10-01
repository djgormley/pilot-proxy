# coding=utf-8
"""Detector input layout and block-to-kernel geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pilot_proxy.schema_identity import schema_token

SPECTRAL_SENSE_NORMAL = "normal"
SPECTRAL_SENSE_INVERTED = "inverted"
SUPPORTED_SPECTRAL_SENSE = frozenset({SPECTRAL_SENSE_NORMAL, SPECTRAL_SENSE_INVERTED})
STREAM_LAYOUT_SCHEMA_NAME = "pilotproxy_stream_layout"
STREAM_LAYOUT_SCHEMA_REVISION = 1
STREAM_LAYOUT_SCHEMA_TOKEN = schema_token(
    STREAM_LAYOUT_SCHEMA_NAME, STREAM_LAYOUT_SCHEMA_REVISION
)
COMBINE_MODE_COMBINED_STREAMS = "incoherent_power_sum_over_streams"
COMBINE_MODE_PER_STREAM_DIAGNOSTIC = "per_stream_diagnostic"
SUPPORTED_COMBINE_MODES = frozenset(
    {
        COMBINE_MODE_COMBINED_STREAMS,
        COMBINE_MODE_PER_STREAM_DIAGNOSTIC,
    }
)


def normalize_spectral_sense(value: Any | None) -> str:
    """Normalize detector input spectral-sense metadata."""
    if value is None:
        return SPECTRAL_SENSE_NORMAL
    text = str(value).strip().lower()
    if text not in SUPPORTED_SPECTRAL_SENSE:
        raise ValueError(
            f"spectral_sense must be one of {sorted(SUPPORTED_SPECTRAL_SENSE)}, "
            f"got {value!r}"
        )
    return text


def spectral_sense_requires_time_reversal(value: Any | None) -> bool:
    """Return whether detector-window samples must be reversed for this sense."""
    return normalize_spectral_sense(value) == SPECTRAL_SENSE_INVERTED


# Half-width of the default designated window in padded fine bins. The
# predicted line position is exact up to the station's own carrier offset;
# the survey measured offsets up to ~340 Hz (ch35), so +/-30 bins of
# 11.92 Hz (+/-358 Hz) covers the observed population with margin.
DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS = 30


def predicted_pilot_fine_bin(
    *,
    pilot_rf_hz: float,
    coarse_center_hz: float,
    sample_rate_hz: float,
    detector_window_samples: int,
    nfft: int,
    spectral_sense: Any | None,
    pad_factor: int = 2,
) -> int:
    """Fine (envelope) bin where the pilot line is predicted to land.

    The weights target the coarse detector bin nearest the pilot, so the
    carrier appears in the window-axis (envelope) spectrum at the grid
    quantization residual -- not at zero envelope offset. Designating fine
    bin 0 therefore watches an empty bin and leaves the actual line inside
    the CFAR null bulk; this predicts the bin from the channel geometry
    alone. The inverted-sense mapping is survey-verified: all six strong
    2018--2026 carriers land within ~30 bins of the prediction, the
    difference being each station's own carrier offset, which the
    designated window's half-width absorbs.
    """
    windows = int(nfft) // int(detector_window_samples)
    fine_bins = int(pad_factor) * windows
    coarse_bin_hz = float(sample_rate_hz) / float(detector_window_samples)
    fine_bin_hz = coarse_bin_hz / float(fine_bins)
    sense_sign = (-1.0 if spectral_sense_requires_time_reversal(spectral_sense)
                  else 1.0)
    eff_hz = sense_sign * (float(pilot_rf_hz) - float(coarse_center_hz))
    resid_hz = eff_hz - round(eff_hz / coarse_bin_hz) * coarse_bin_hz
    return int(round(resid_hz / fine_bin_hz)) % fine_bins


def predicted_fine_designated_bins(
    predicted_bin: int, half_width: int, fine_bins: int
) -> list[int]:
    """Designated window ``predicted_bin +/- half_width``, modulo the grid."""
    if not 0 <= int(half_width) < int(fine_bins) // 2:
        raise ValueError("half_width must be in [0, fine_bins/2).")
    return [(int(predicted_bin) + i) % int(fine_bins)
            for i in range(-int(half_width), int(half_width) + 1)]


@dataclass(frozen=True)
class DetectorFrameLayout:
    """Canonical receiver-frame geometry for one detector decision."""

    frame_size_samples: int
    detector_window_samples: int
    num_input_streams: int
    num_selected_channels: int = 1
    combine_mode: str = COMBINE_MODE_COMBINED_STREAMS

    def __post_init__(self) -> None:
        frame_size = int(self.frame_size_samples)
        window = int(self.detector_window_samples)
        inputs = int(self.num_input_streams)
        channels = int(self.num_selected_channels)
        combine_mode = str(self.combine_mode)
        if frame_size <= 0:
            raise ValueError("frame_size_samples must be positive.")
        if window <= 0:
            raise ValueError("detector_window_samples must be positive.")
        if inputs <= 0:
            raise ValueError("num_input_streams must be positive.")
        if channels <= 0:
            raise ValueError("num_selected_channels must be positive.")
        if frame_size % window != 0:
            raise ValueError(
                "frame_size_samples must be an integer multiple of "
                "detector_window_samples: "
                f"frame_size_samples={frame_size}, "
                f"detector_window_samples={window}"
            )
        if combine_mode not in SUPPORTED_COMBINE_MODES:
            raise ValueError(f"unsupported combine_mode: {combine_mode!r}")
        object.__setattr__(self, "frame_size_samples", frame_size)
        object.__setattr__(self, "detector_window_samples", window)
        object.__setattr__(self, "num_input_streams", inputs)
        object.__setattr__(self, "num_selected_channels", channels)
        object.__setattr__(self, "combine_mode", combine_mode)

    @property
    def windows_per_stream(self) -> int:
        return self.frame_size_samples // self.detector_window_samples

    @property
    def num_streams(self) -> int:
        return self.num_input_streams * self.num_selected_channels

    @property
    def detector_rows_per_frame(self) -> int:
        return self.num_streams * self.windows_per_stream

    @property
    def samples_per_result(self) -> int:
        return self.frame_size_samples * self.num_streams

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STREAM_LAYOUT_SCHEMA_TOKEN,
            "frame_size_samples": self.frame_size_samples,
            "detector_window_samples": self.detector_window_samples,
            "windows_per_stream": self.windows_per_stream,
            "num_input_streams": self.num_input_streams,
            "num_selected_channels": self.num_selected_channels,
            "num_streams": self.num_streams,
            "detector_rows_per_frame": self.detector_rows_per_frame,
            "samples_per_result": self.samples_per_result,
            "combine_mode": self.combine_mode,
        }


def flatten_feed_channel_streams(feed_channel_streams: np.ndarray) -> np.ndarray:
    """Flatten feed/channel/time data into stream/time data.

    The stream order is feed-major, then selected-channel-major.
    """
    arr = np.asarray(feed_channel_streams)
    if arr.ndim != 3:
        raise ValueError(
            "feed_channel_streams must have shape (feed, channel, time); "
            f"got shape={arr.shape}"
        )
    num_feeds, num_selected_channels, samples = arr.shape
    if num_feeds <= 0 or num_selected_channels <= 0 or samples <= 0:
        raise ValueError(
            "feed_channel_streams axes must all be non-empty; "
            f"got shape={arr.shape}"
        )
    return np.ascontiguousarray(arr.reshape(num_feeds * num_selected_channels, samples))


def build_stream_map(
    *,
    num_feeds: int,
    selected_channel_indices: list[int],
    physical_channel: int | None = None,
) -> list[dict[str, int | None]]:
    """Build metadata for feed/channel streams flattened into kernel rows."""
    feeds = int(num_feeds)
    if feeds <= 0:
        raise ValueError("num_feeds must be positive.")
    if not selected_channel_indices:
        raise ValueError("selected_channel_indices must not be empty.")

    stream_map: list[dict[str, int | None]] = []
    stream_index = 0
    for feed_index in range(feeds):
        for channel_index in selected_channel_indices:
            stream_map.append(
                {
                    "stream_index": int(stream_index),
                    "feed_index": int(feed_index),
                    "selected_channel_index": int(channel_index),
                    "physical_channel": (
                        None if physical_channel is None else int(physical_channel)
                    ),
                }
            )
            stream_index += 1
    return stream_map


def block_time_stream_to_detector_matrix(
    block: np.ndarray,
    *,
    detector_window_samples: int,
    time_axis: int = 0,
    stream_axis: int = 1,
) -> np.ndarray:
    """Convert a time-by-stream block into row-major detector windows."""
    arr = np.asarray(block)
    if arr.ndim != 2:
        raise ValueError(
            f"block must be 2D with axes (time, stream); got shape={arr.shape}"
        )
    time_axis = int(time_axis)
    stream_axis = int(stream_axis)
    if time_axis < 0:
        time_axis += arr.ndim
    if stream_axis < 0:
        stream_axis += arr.ndim
    if time_axis == stream_axis:
        raise ValueError("time_axis and stream_axis must be different.")
    if time_axis not in (0, 1) or stream_axis not in (0, 1):
        raise ValueError("time_axis and stream_axis must identify the two axes.")

    time_stream = np.moveaxis(arr, (time_axis, stream_axis), (0, 1))
    layout = DetectorFrameLayout(
        frame_size_samples=int(time_stream.shape[0]),
        detector_window_samples=int(detector_window_samples),
        num_input_streams=int(time_stream.shape[1]),
    )
    windows = time_stream.reshape(
        layout.windows_per_stream,
        layout.detector_window_samples,
        layout.num_streams,
    )
    detector_matrix = np.transpose(windows, (2, 0, 1)).reshape(
        layout.detector_rows_per_frame,
        layout.detector_window_samples,
    )
    return np.ascontiguousarray(detector_matrix)


def stream_time_block_to_detector_matrix(
    streams: np.ndarray,
    *,
    detector_window_samples: int,
) -> np.ndarray:
    """Convert stream-by-time data into row-major detector windows."""
    return block_time_stream_to_detector_matrix(
        streams,
        detector_window_samples=detector_window_samples,
        time_axis=1,
        stream_axis=0,
    )


def stack_stream_time_blocks(
    streams: np.ndarray,
    *,
    detector_window_samples: int,
    frame_size_samples: int,
    block_step_samples: int,
    num_blocks: int,
) -> np.ndarray:
    """Build a batch of detector matrices from stream-major time series."""
    streams = np.asarray(streams)
    if streams.ndim != 2:
        raise ValueError(
            "streams must be 2D with shape (stream, time); "
            f"got shape={streams.shape}"
        )
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1.")
    if block_step_samples <= 0:
        raise ValueError("block_step_samples must be positive.")

    blocks = []
    for block_index in range(int(num_blocks)):
        start = block_index * int(block_step_samples)
        stop = start + int(frame_size_samples)
        if stop > streams.shape[1]:
            raise ValueError(
                "not enough time samples to build requested blocks: "
                f"block_index={block_index}, stop={stop}, available={streams.shape[1]}"
            )
        blocks.append(
            stream_time_block_to_detector_matrix(
                streams[:, start:stop],
                detector_window_samples=detector_window_samples,
            )
        )
    return np.ascontiguousarray(np.stack(blocks))


def apply_spectral_sense_to_detector_matrix(
    detector_matrix: np.ndarray,
    *,
    spectral_sense: str,
) -> np.ndarray:
    """Apply spectral-sense correction to a detector-matrix view."""
    sense = normalize_spectral_sense(spectral_sense)
    arr = np.asarray(detector_matrix)
    if spectral_sense_requires_time_reversal(sense):
        return np.ascontiguousarray(np.flip(arr, axis=-1))
    return np.ascontiguousarray(arr)


__all__ = [
    "COMBINE_MODE_COMBINED_STREAMS",
    "COMBINE_MODE_PER_STREAM_DIAGNOSTIC",
    "DetectorFrameLayout",
    "SPECTRAL_SENSE_INVERTED",
    "SPECTRAL_SENSE_NORMAL",
    "STREAM_LAYOUT_SCHEMA_NAME",
    "STREAM_LAYOUT_SCHEMA_REVISION",
    "STREAM_LAYOUT_SCHEMA_TOKEN",
    "SUPPORTED_COMBINE_MODES",
    "SUPPORTED_SPECTRAL_SENSE",
    "DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS",
    "apply_spectral_sense_to_detector_matrix",
    "block_time_stream_to_detector_matrix",
    "build_stream_map",
    "flatten_feed_channel_streams",
    "normalize_spectral_sense",
    "predicted_fine_designated_bins",
    "predicted_pilot_fine_bin",
    "spectral_sense_requires_time_reversal",
    "stack_stream_time_blocks",
    "stream_time_block_to_detector_matrix",
]

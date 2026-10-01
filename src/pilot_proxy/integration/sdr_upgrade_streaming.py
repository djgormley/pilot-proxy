"""Bounded-state, CPU-only companion to the frozen single-record SDR adapter.

Transport indices must be explicit. A discontinuity starts a new independently
trimmed segment: no missing sample is fabricated and no frame spans a gap.
"""
from __future__ import annotations

from dataclasses import asdict
from numbers import Integral

import numpy as np
from scipy.signal import upfirdn

from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
    quantize_complex_numpy,
)
from pilot_proxy.integration.sdr_upgrade_adapter import (
    DOWN,
    FIR_TAPS,
    FRAME_SAMPLES,
    INPUT_RATE_HZ,
    OUTPUT_RATE_HZ,
    UP,
    DigitalAdapterConfig,
    K,
    L,
    projector_weights,
    resampler_coefficients,
)

INPUT_BLOCK_SAMPLES = 8192
# One FIR support plus the alignment back to a multiple of DOWN input samples.
MAX_HISTORY_SAMPLES = (FIR_TAPS - 1 + UP - 1) // UP + DOWN - 1
FIRST_OUTPUT_INDEX = (FIR_TAPS - 1 + DOWN - 1) // DOWN
HALF_FILTER = (FIR_TAPS - 1) // 2


def _index(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be an explicit nonnegative integer.")
    return int(value)


class StreamingDigitalAdapter:
    """Preserve the batch adapter's arithmetic across arbitrary input chunks.

    ``push`` consumes its entire chunk synchronously and returns complete frames.
    Returned frames belong to the caller; memory retained by this object is
    bounded independently of recording duration and caller chunk size. Caller
    chunks and the returned list naturally need memory proportional to their
    own sizes. ``finish`` returns any complete final frames and a discard receipt.

    ``gap`` declares a known positive sample loss. ``reset`` declares a new
    segment with an explicit source index (including when loss is unknown).
    Both drain valid outputs, discard the incomplete frame, and reset mixer
    phase, resampler lattice and frame origin. Neither reconstructs continuity.
    """

    def __init__(self, config: DigitalAdapterConfig, *, first_input_index: int = 0):
        if not isinstance(config, DigitalAdapterConfig):
            raise TypeError("config must be a DigitalAdapterConfig.")
        self.config = config
        self._h = resampler_coefficients() * UP
        _, self._weights = projector_weights(config)
        self._pending_input = np.empty(INPUT_BLOCK_SAMPLES, dtype=np.complex128)
        self._pending_frame = np.empty(FRAME_SAMPLES, dtype=np.complex128)
        self._segment = -1
        self._start(_index(first_input_index, "first_input_index"), "initial_segment")

    def _start(self, first_input_index: int, reason: str) -> None:
        self._segment += 1
        self._origin = first_input_index
        self._start_reason = reason
        self._accepted = self._processed = self._frame_count = 0
        self._input_count = self._frame_fill = 0
        self._history = np.empty(0, dtype=np.complex128)
        self._history_origin = 0
        self._next_output = FIRST_OUTPUT_INDEX
        self._closed = False

    @property
    def next_input_index(self) -> int:
        return self._origin + self._accepted

    @property
    def state(self) -> dict:
        """Expose bounded buffer occupancy and continuity without sample data."""
        return {
            "segment_id": self._segment, "closed": self._closed,
            "segment_start_reason": self._start_reason,
            "first_input_index": self._origin,
            "next_input_index": self.next_input_index,
            "accepted_input_samples": self._accepted,
            "processed_input_samples": self._processed,
            "pending_input_samples": self._input_count,
            "mixed_history_samples": int(self._history.size),
            "pending_frame_samples": self._frame_fill,
            "emitted_frames": self._frame_count,
            "next_untrimmed_output_index": self._next_output,
            "buffer_capacities_samples": {
                "pending_input": INPUT_BLOCK_SAMPLES,
                "mixed_history": MAX_HISTORY_SAMPLES,
                "pending_frame": FRAME_SAMPLES,
            },
        }

    def push(self, iq: np.ndarray, *, input_start_index: int) -> list[dict]:
        """Consume a contiguous complex chunk, refusing implicit gaps/overlap.

        Empty complex chunks at the expected index are valid no-ops. Invalid
        chunks are rejected before modifying state. Single-threaded ownership
        is required; this class does not serialize concurrent callers.
        """
        if self._closed:
            raise RuntimeError("Segment is finished; explicitly reset before push.")
        start = _index(input_start_index, "input_start_index")
        if start != self.next_input_index:
            raise ValueError("Noncontiguous input index; declare gap or reset explicitly.")
        x = np.asarray(iq)
        if x.ndim != 1 or not np.iscomplexobj(x):
            raise ValueError("iq must be a one-dimensional complex array.")
        # Fixed-size validation temporaries, including for very large chunks.
        for offset in range(0, x.size, INPUT_BLOCK_SAMPLES):
            if not np.all(np.isfinite(x[offset:offset + INPUT_BLOCK_SAMPLES])):
                raise ValueError("iq must contain only finite samples.")
        if self._accepted + x.size > 2**53 - 1:
            raise ValueError("Segment exceeds the exact float64 mixer-index range.")
        frames = []
        offset = 0
        while offset < x.size:
            take = min(x.size - offset, INPUT_BLOCK_SAMPLES - self._input_count)
            self._pending_input[self._input_count:self._input_count + take] = x[offset:offset + take]
            self._input_count += take
            self._accepted += take
            offset += take
            if self._input_count == INPUT_BLOCK_SAMPLES:
                frames.extend(self._drain_input())
        return frames

    def _drain_input(self) -> list[dict]:
        if not self._input_count:
            return []
        n = self._input_count
        indices = np.arange(self._processed, self._processed + n)
        phase = -2j * np.pi * self.config.mixer_hz * indices / INPUT_RATE_HZ
        mixed = self._pending_input[:n] * np.exp(phase)
        work = np.concatenate((self._history, mixed))
        origin = self._history_origin
        assert origin % DOWN == 0
        local = upfirdn(self._h, work, up=UP, down=DOWN)
        self._processed += n
        stop = ((self._processed - 1) * UP) // DOWN + 1
        offset = origin * UP // DOWN
        valid = local[self._next_output - offset:stop - offset]
        # Short initial segments can have no fully supported output at all.
        if stop < self._next_output:
            valid = local[:0]
        frames = self._consume_outputs(valid)
        self._next_output = max(self._next_output, stop)
        earliest = max(0, (self._next_output * DOWN - (FIR_TAPS - 1) + UP - 1) // UP)
        keep_origin = earliest // DOWN * DOWN
        keep_origin = min(keep_origin, self._processed // DOWN * DOWN)
        self._history = work[keep_origin - origin:].copy()
        self._history_origin = keep_origin
        assert self._history.size <= MAX_HISTORY_SAMPLES
        self._input_count = 0
        return frames

    def _consume_outputs(self, samples: np.ndarray) -> list[dict]:
        frames = []
        offset = 0
        while offset < samples.size:
            take = min(samples.size - offset, FRAME_SAMPLES - self._frame_fill)
            self._pending_frame[self._frame_fill:self._frame_fill + take] = samples[offset:offset + take]
            self._frame_fill += take
            offset += take
            if self._frame_fill == FRAME_SAMPLES:
                frames.append(self._emit_frame())
                self._frame_fill = 0
                self._frame_count += 1
        return frames

    def _emit_frame(self) -> dict:
        y = self._pending_frame.reshape(L, K).copy()
        packed = quantize_complex_numpy(y, 4, self.config.sample_scale)
        projections = matched_filter_row_projections_cpu_reference_packed(packed, self._weights, 4)
        powers = (projections.astype(np.float64) ** 2).sum(axis=(1, 2))
        denominator = powers[1] + powers[2]
        ratio = 2 * powers[0] / denominator if denominator > 0 else (np.inf if powers[0] > 0 else np.nan)
        j = FIRST_OUTPUT_INDEX + self._frame_count * FRAME_SAMPLES
        time_numerator = self._origin * UP + j * DOWN - HALF_FILTER
        relative_start = (FIRST_OUTPUT_INDEX * DOWN - HALF_FILTER) / (INPUT_RATE_HZ * UP)
        relative_start += self._frame_count * FRAME_SAMPLES / OUTPUT_RATE_HZ
        rounded_r = np.rint(y.real * self.config.sample_scale)
        rounded_i = np.rint(y.imag * self.config.sample_scale)
        scaled = np.concatenate((y.real.ravel(), y.imag.ravel())) * self.config.sample_scale
        active = scaled[abs(scaled) < 7.5]
        half_step_distance = float(np.min(abs(active - np.floor(active) - .5))) if active.size else None
        return {
            "segment_id": self._segment, "frame_index_in_segment": self._frame_count,
            "segment_first_input_index": self._origin,
            "first_untrimmed_output_index": j,
            "frame_start_seconds_from_segment_start": relative_start,
            "sample_center_time_numerator": time_numerator,
            "sample_center_time_denominator_hz": INPUT_RATE_HZ * UP,
            "frame_start_seconds_from_source_index_zero": time_numerator / (INPUT_RATE_HZ * UP),
            "frame_size_samples": FRAME_SAMPLES, "K": K, "L": L,
            "resampled_frame": y, "packed_frame": packed,
            "projections_i32": projections, "term_power_sums": powers,
            "coarse_ratio": float(ratio),
            "ratio_status": "finite" if denominator > 0 else ("infinite" if powers[0] > 0 else "undefined_both_zero"),
            "saturated_component_count": int(np.count_nonzero(abs(rounded_r) > 7) + np.count_nonzero(abs(rounded_i) > 7)),
            "minimum_unclipped_quantizer_half_step_distance": half_step_distance,
        }

    def finish(self, *, reason: str = "end_of_record") -> dict:
        """Drain complete frames, discard unsupported edges/tail, then close.

        Finishing twice is an error, preventing duplicate boundary receipts.
        No FIR ring-down padding and no partial detector frame is emitted.
        """
        if self._closed:
            raise RuntimeError("Segment is already finished.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A nonempty boundary reason is required.")
        frames = self._drain_input()
        supported = max(0, self._next_output - FIRST_OUTPUT_INDEX)
        receipt = {
            "schema": "sdr-upgrade-streaming-segment-v1", "reason": reason,
            "segment_start_reason": self._start_reason,
            "config": asdict(self.config), "segment_id": self._segment,
            "first_input_index": self._origin, "next_input_index": self.next_input_index,
            "input_samples": self._accepted, "frame_count": self._frame_count,
            "full_support_output_samples": supported,
            "discarded_incomplete_frame_samples": self._frame_fill,
            "boundary_policy": "no FIR padding; discard partial frame; no cross-segment sample or phase continuity",
            "mixer_phase_origin": "zero at each explicitly declared segment start",
            "time_domain": "digital sample-index lattice only; no hardware timestamp or physical RF calibration",
        }
        self._frame_fill = 0
        self._history = np.empty(0, dtype=np.complex128)
        self._closed = True
        return {"frames": frames, "receipt": receipt}

    def gap(self, missing_input_samples: int) -> dict:
        """Declare a positive known transport loss and begin after the gap."""
        missing = _index(missing_input_samples, "missing_input_samples")
        if missing == 0:
            raise ValueError("A gap must contain at least one missing input sample.")
        next_origin = self.next_input_index + missing
        boundary = self.finish(reason="known_transport_gap")
        boundary["receipt"]["missing_input_samples_to_next_segment"] = missing
        self._start(next_origin, "after_known_transport_gap")
        return boundary

    def reset(self, *, first_input_index: int, reason: str) -> dict | None:
        """Explicitly start a new segment, even if lost sample count is unknown.

        The supplied index labels the new source timeline; a reset may rewind
        that timeline. Segment IDs distinguish it from previous data. Reset
        does not infer lost samples from the difference between two indices.
        """
        origin = _index(first_input_index, "first_input_index")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A nonempty reset reason is required.")
        boundary = None if self._closed else self.finish(reason=reason)
        self._start(origin, reason)
        return boundary

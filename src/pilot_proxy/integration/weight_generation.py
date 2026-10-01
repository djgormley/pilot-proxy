# coding=utf-8
"""Generate packed detector weight ROMs from receiver profiles."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np

from pilot_proxy.atsc_channels import (
    parse_physical_channel_spec,
    physical_channel_to_pilot_hz,
    validate_uhf_physical_channel,
)
from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
    input_coordinate_system_for_weight_coordinate,
    normalize_weight_coordinate_system,
    null_power_ratio_from_weight_norms,
    weight_term_norms_sq,
)
from pilot_proxy.detector_geometry import SPECTRAL_SENSE_NORMAL
from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.detector_weights import (
    CRC32_UNSIGNED_MASK,
    CRC_SIZE,
    HEADER_FIXED_FMT,
    HEADER_FIXED_SIZE,
    WEIGHT_MAGIC,
    WEIGHT_VERSION,
)

from .detector_core import DetectorCoreProfile
from .receiver_profile import (
    ReceiverProfile,
    receiver_frequency_to_channel,
    receiver_profile_hash,
)
from .stream_layout import validate_integration_compatibility

DEFAULT_DOPPLER_TOL_HZ = math.nan
HZ_PER_MHZ = 1.0e6
WEIGHT_TERM_TARGET = 0
WEIGHT_TERM_REF_LOWER = 1
WEIGHT_TERM_REF_UPPER = 2
REFERENCE_SELECTION_SCORE_NOMINAL = 0.6
REFERENCE_SELECTION_SCORE_ADAPTIVE = 0.5
REFERENCE_SELECTION_METHOD = "adaptive_circular_reference_placement"
REFERENCE_PLACEMENT_STATUS_NOMINAL = "nominal"
REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED = "edge_wrapped"
REFERENCE_PLACEMENT_STATUS_DC_SHIFTED = "dc_shifted"
REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED_AND_DC_SHIFTED = (
    "edge_wrapped_and_dc_shifted"
)
REFERENCE_PLACEMENT_REASON_NOMINAL = "nominal"
REFERENCE_PLACEMENT_REASON_EDGE_WRAPPED = "edge_reference_wrapped"
REFERENCE_PLACEMENT_REASON_DC_SHIFTED = "dc_reference_shifted_away"
BASEBAND_FRAME_MODE_EXPLICIT = "explicit_profile_frame"
# ``edge_wrapped`` is a bookkeeping event: the requested reference frequency
# left the normalized interval [0, 1) and was reduced modulo 1. Because the
# K-tap projection is periodic in normalized frequency, the wrapped weights
# are bit-identical to the unwrapped ones; what the flag records is that the
# reference crossed the baseband-frame ORIGIN (nu = 0). What that origin is
# physically depends on the receiver profile:
#   channel_center_normalized = 0.0 -> nu = 0 is the coarse-channel center
#                                      (the forbidden DC tone);
#   channel_center_normalized = 0.5 -> nu = 0 is the coarse-channel edge.
# The physical coarse-channel edge (+-fs/2 from the channel center) is
# reported separately by the ``*_crosses_channel_edge`` fields below.
FRAME_ORIGIN_DESCRIPTION_GENERIC = "the baseband-frame origin (nu = 0)"
FRAME_ORIGIN_DESCRIPTION_CENTER = (
    "the baseband-frame origin nu = 0 (the coarse-channel center / DC under "
    "this profile, not the channel edge)"
)
FRAME_ORIGIN_DESCRIPTION_EDGE = (
    "the baseband-frame origin nu = 0 (the coarse-channel edge under this "
    "profile)"
)


def frame_origin_description(frame_center_normalized: float) -> str:
    """Say what nu = 0 is physically for a profile's frame convention."""
    center = float(frame_center_normalized) % 1.0
    if math.isclose(center, 0.0, abs_tol=1e-12) or math.isclose(
        center, 1.0, abs_tol=1e-12
    ):
        return FRAME_ORIGIN_DESCRIPTION_CENTER
    if math.isclose(center, 0.5, abs_tol=1e-12):
        return FRAME_ORIGIN_DESCRIPTION_EDGE
    return (
        f"{FRAME_ORIGIN_DESCRIPTION_GENERIC}; the coarse-channel center sits at "
        f"nu = {center:.6f}"
    )


def _signed_circular_delta(a: float, b: float) -> float:
    """Signed circular difference a-b mapped into [-0.5, 0.5)."""
    return ((float(a) - float(b) + 0.5) % 1.0) - 0.5


def _resolve_frame_convention(
    profile: "ReceiverProfile", coarse_channel_index: int
) -> tuple[float, float, str]:
    """Return the explicitly declared frame center and physical DC."""
    return (
        float(profile.frame_center_normalized(coarse_channel_index)),
        float(profile.forbidden_dc_normalized(coarse_channel_index)),
        BASEBAND_FRAME_MODE_EXPLICIT,
    )


REFERENCE_FORBIDDEN_COLLISION_RULE = (
    "circular_normalized_distance <= 0.5 / detector_window_samples"
)


@dataclass(frozen=True)
class DetectorCoreLayout:
    detector_window_samples: int
    skipped_guard_bins: int
    reference_offset_bins: int

    def __post_init__(self) -> None:
        values: dict[str, int] = {}
        for field_name in (
            "detector_window_samples",
            "skipped_guard_bins",
            "reference_offset_bins",
        ):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"{field_name} must be an integer, not a boolean."
                )
            try:
                values[field_name] = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{field_name} must be an integer.") from exc
        if values["detector_window_samples"] <= 0:
            raise ValueError("detector_window_samples must be positive.")
        if values["skipped_guard_bins"] < 0:
            raise ValueError("skipped_guard_bins must be nonnegative.")
        if values["reference_offset_bins"] <= 0:
            raise ValueError("reference_offset_bins must be positive.")
        if (
            values["reference_offset_bins"]
            != values["skipped_guard_bins"] + 1
        ):
            raise ValueError(
                "reference_offset_bins must equal skipped_guard_bins + 1."
            )
        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)


def _physical_channels_from_range(value: str) -> list[int]:
    return parse_physical_channel_spec(value)


def parse_physical_channel_selection(
    *,
    physical_channels: Sequence[int] | None = None,
    physical_channel_range: str | None = None,
) -> list[int]:
    """Parse explicit and range-based ATSC physical-channel selections."""
    out: list[int] = []
    if physical_channel_range is not None:
        out.extend(_physical_channels_from_range(physical_channel_range))
    if physical_channels is not None:
        out.extend(
            validate_uhf_physical_channel(channel)
            for channel in physical_channels
        )
    seen: set[int] = set()
    unique: list[int] = []
    for channel in out:
        if channel in seen:
            continue
        seen.add(channel)
        unique.append(channel)
    if not unique:
        raise ValueError("at least one physical channel must be selected.")
    return unique


def _packed_weight_vector(
    normalized_frequency: float,
    *,
    detector_window_samples: int,
    bits_per_component: int,
) -> np.ndarray:
    sample_index = np.arange(int(detector_window_samples), dtype=np.float64)
    max_int = float((1 << (int(bits_per_component) - 1)) - 1)
    weights = max_int * np.exp(
        -2j * math.pi * float(normalized_frequency) * sample_index
    )
    return quantize_complex_numpy(
        weights[np.newaxis, :].astype(np.complex64),
        int(bits_per_component),
        1.0,
    )[0]


def _wrap_normalized_frequency(value: float) -> float:
    return float(float(value) % 1.0)


def _circular_normalized_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 1.0
    return float(min(delta, 1.0 - delta))


def _reference_collides_with_dc(
    normalized_frequency: float,
    *,
    detector_window_samples: int,
    dc_normalized: float,
) -> bool:
    fine_bin = 1.0 / float(detector_window_samples)
    return bool(
        _circular_normalized_distance(
            float(normalized_frequency),
            float(dc_normalized),
        )
        <= 0.5 * fine_bin
    )


def _dc_in_skipped_guard(
    target_normalized_frequency: float,
    *,
    reference_offset_bins: int,
    detector_window_samples: int,
    dc_normalized: float,
) -> bool:
    fine_bin = 1.0 / float(detector_window_samples)
    signed_bins_to_dc = _signed_circular_delta(
        float(dc_normalized),
        float(target_normalized_frequency),
    ) / fine_bin
    selected = int(reference_offset_bins)
    if selected == 0:
        return False
    same_side = math.copysign(1.0, signed_bins_to_dc) == math.copysign(
        1.0, selected
    )
    return bool(same_side and 0.0 < abs(signed_bins_to_dc) < abs(selected))


def _resolve_one_reference_offset(
    target_normalized_frequency: float,
    *,
    detector_window_samples: int,
    requested_offset_bins: int,
    direction: int,
    dc_normalized: float,
) -> dict[str, Any]:
    if int(direction) not in {-1, 1}:
        raise ValueError("direction must be -1 or +1")
    requested = abs(int(requested_offset_bins))
    if requested <= 0:
        raise ValueError("requested_offset_bins must be positive")

    fine_bin = 1.0 / float(detector_window_samples)
    offset = int(direction) * requested
    wrapped = False
    dc_shifted = False
    requested_raw = float(target_normalized_frequency) + offset * fine_bin
    requested_wrapped = not (0.0 <= requested_raw < 1.0)
    requested_normalized = _wrap_normalized_frequency(requested_raw)
    requested_dc_collision = _reference_collides_with_dc(
        requested_normalized,
        detector_window_samples=int(detector_window_samples),
        dc_normalized=float(dc_normalized),
    )

    for _ in range(int(detector_window_samples)):
        raw = float(target_normalized_frequency) + offset * fine_bin
        if not (0.0 <= raw < 1.0):
            wrapped = True
        normalized = _wrap_normalized_frequency(raw)
        if not _reference_collides_with_dc(
            normalized,
            detector_window_samples=int(detector_window_samples),
            dc_normalized=float(dc_normalized),
        ):
            return {
                "offset_bins": int(offset),
                "normalized_frequency": float(normalized),
                "edge_wrapped": bool(wrapped),
                "dc_shifted": bool(dc_shifted),
                "requested_offset_bins": int(direction) * requested,
                "requested_normalized_frequency": float(requested_normalized),
                "requested_edge_wrapped": bool(requested_wrapped),
                "requested_dc_collision": bool(requested_dc_collision),
            }
        dc_shifted = True
        offset += int(direction)

    raise ValueError(
        "could not choose a reference bin that avoids the forbidden DC bin"
    )


def _reference_placement(
    target_normalized_frequency: float,
    *,
    detector_window_samples: int,
    reference_offset_bins: int,
    dc_normalized: float,
    origin_description: str = FRAME_ORIGIN_DESCRIPTION_GENERIC,
) -> dict[str, Any]:
    lower = _resolve_one_reference_offset(
        target_normalized_frequency,
        detector_window_samples=int(detector_window_samples),
        requested_offset_bins=int(reference_offset_bins),
        direction=-1,
        dc_normalized=float(dc_normalized),
    )
    upper = _resolve_one_reference_offset(
        target_normalized_frequency,
        detector_window_samples=int(detector_window_samples),
        requested_offset_bins=int(reference_offset_bins),
        direction=1,
        dc_normalized=float(dc_normalized),
    )
    edge_wrapped = bool(lower["edge_wrapped"] or upper["edge_wrapped"])
    dc_shifted = bool(lower["dc_shifted"] or upper["dc_shifted"])
    requested_dc_collision = bool(
        lower["requested_dc_collision"] or upper["requested_dc_collision"]
    )
    forbidden_tone_in_skipped_guard = bool(
        _dc_in_skipped_guard(
            target_normalized_frequency,
            reference_offset_bins=int(lower["offset_bins"]),
            detector_window_samples=int(detector_window_samples),
            dc_normalized=float(dc_normalized),
        )
        or _dc_in_skipped_guard(
            target_normalized_frequency,
            reference_offset_bins=int(upper["offset_bins"]),
            detector_window_samples=int(detector_window_samples),
            dc_normalized=float(dc_normalized),
        )
    )
    if edge_wrapped and dc_shifted:
        status = REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED_AND_DC_SHIFTED
    elif edge_wrapped:
        status = REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED
    elif dc_shifted:
        status = REFERENCE_PLACEMENT_STATUS_DC_SHIFTED
    else:
        status = REFERENCE_PLACEMENT_STATUS_NOMINAL
    reasons: list[str] = []
    if edge_wrapped:
        reasons.append(REFERENCE_PLACEMENT_REASON_EDGE_WRAPPED)
    if dc_shifted:
        reasons.append(REFERENCE_PLACEMENT_REASON_DC_SHIFTED)
    warnings: list[str] = []
    if bool(lower["edge_wrapped"]):
        warnings.append(
            f"lower reference wrapped across {origin_description}"
        )
    if bool(upper["edge_wrapped"]):
        warnings.append(
            f"upper reference wrapped across {origin_description}"
        )
    if bool(lower["dc_shifted"]):
        warnings.append("lower reference shifted away from coarse-channel DC")
    if bool(upper["dc_shifted"]):
        warnings.append("upper reference shifted away from coarse-channel DC")

    return {
        "lower": lower,
        "upper": upper,
        "placement_status": status,
        "adaptive_reference_placement": bool(edge_wrapped or dc_shifted),
        "reference_placement_reason": (
            ",".join(reasons) if reasons else REFERENCE_PLACEMENT_REASON_NOMINAL
        ),
        "placement_warnings": "; ".join(warnings),
        "edge_reference_wrapped": bool(edge_wrapped),
        "dc_reference_collision": bool(requested_dc_collision),
        "dc_reference_shifted": bool(dc_shifted),
        "forbidden_tone_in_skipped_guard": bool(forbidden_tone_in_skipped_guard),
    }


def target_layout(
    *,
    physical_channel: int,
    profile: ReceiverProfile,
    core: DetectorCoreLayout | DetectorCoreProfile,
    pilot_hz_override: float | None = None,
) -> dict[str, Any]:
    # A file-backed core profile must agree with the receiver profile on the
    # deployed window length; select it with with_detector_window_samples
    # first. A bare DetectorCoreLayout stays an explicit geometry override
    # for what-if analysis.
    if isinstance(core, DetectorCoreProfile):
        core_window = int(core.detector_window_samples)
        profile_window = int(profile.detector_window_samples)
        if core_window != profile_window:
            raise ValueError(
                "receiver profile and detector core disagree on "
                f"detector_window_samples: profile={profile_window}, "
                f"core={core_window}. Select the receiver window length on "
                "the core first "
                "(DetectorCoreProfile.with_detector_window_samples)."
            )
    channel = validate_uhf_physical_channel(physical_channel)
    nominal_pilot_hz = physical_channel_to_pilot_hz(channel)
    # A station can run its carrier a few kHz off the ATSC nominal; the
    # template follows the measured carrier while the manifest keeps the
    # nominal pilot as the lookup key.
    pilot_hz = nominal_pilot_hz if pilot_hz_override is None else float(pilot_hz_override)
    if pilot_hz_override is not None and abs(pilot_hz - nominal_pilot_hz) > float(
        profile.coarse_channel_width_hz
    ) / 2.0:
        raise ValueError(
            f"measured pilot {pilot_hz:.0f} Hz is more than half a coarse channel from the "
            f"nominal pilot {nominal_pilot_hz:.0f} Hz of physical channel {channel}"
        )
    selection = receiver_frequency_to_channel(pilot_hz, profile)
    frame_center, forbidden_dc, frame_mode = _resolve_frame_convention(
        profile, int(selection.coarse_channel_index)
    )
    target_normalized = _wrap_normalized_frequency(
        frame_center
        + float(selection.fine_bin_offset_hz)
        / float(profile.coarse_channel_width_hz)
    )
    if _reference_collides_with_dc(
        target_normalized,
        detector_window_samples=int(core.detector_window_samples),
        dc_normalized=float(forbidden_dc),
    ):
        raise ValueError(
            "target pilot bin collides with the forbidden coarse-channel DC bin "
            f"for physical channel {channel}; target bins cannot be "
            "moved safely."
        )
    placement = _reference_placement(
        target_normalized,
        detector_window_samples=int(core.detector_window_samples),
        reference_offset_bins=int(core.reference_offset_bins),
        dc_normalized=float(forbidden_dc),
        origin_description=frame_origin_description(frame_center),
    )
    lower = placement["lower"]
    upper = placement["upper"]
    lower_offset = int(lower["offset_bins"])
    upper_offset = int(upper["offset_bins"])
    fine_bin_width_hz = (
        float(profile.coarse_channel_width_hz)
        / float(core.detector_window_samples)
    )
    lower_normalized = float(lower["normalized_frequency"])
    upper_normalized = float(upper["normalized_frequency"])
    lower_requested_normalized = float(lower["requested_normalized_frequency"])
    upper_requested_normalized = float(upper["requested_normalized_frequency"])
    target_offset_hz = float(selection.fine_bin_offset_hz)
    lower_reference_offset_hz = _signed_circular_delta(
        lower_normalized, frame_center
    ) * float(profile.coarse_channel_width_hz)
    upper_reference_offset_hz = _signed_circular_delta(
        upper_normalized, frame_center
    ) * float(profile.coarse_channel_width_hz)
    lower_requested_reference_offset_hz = _signed_circular_delta(
        lower_requested_normalized, frame_center
    ) * float(profile.coarse_channel_width_hz)
    upper_requested_reference_offset_hz = _signed_circular_delta(
        upper_requested_normalized, frame_center
    ) * float(profile.coarse_channel_width_hz)
    if not (0.0 <= lower_normalized < 1.0 and 0.0 <= upper_normalized < 1.0):
        raise ValueError(
            "could not choose in-channel reference bins for physical channel "
            f"{channel}."
        )
    min_selected_offset = min(abs(lower_offset), abs(upper_offset))
    # Physical coarse-channel edge diagnostics. The placement above works in
    # the periodic normalized-frequency coordinate, so a reference requested
    # beyond +-fs/2 of the channel center simply aliases to the opposite
    # channel edge; detection arithmetic is unaffected, but the reference then
    # samples the channelizer's roll-off region (both channel edges fold
    # there). These fields make that visible without changing any placement.
    half_width_hz = 0.5 * float(profile.coarse_channel_width_hz)
    lower_unwrapped_hz = float(target_offset_hz) + lower_offset * fine_bin_width_hz
    upper_unwrapped_hz = float(target_offset_hz) + upper_offset * fine_bin_width_hz
    lower_crosses_edge = bool(abs(lower_unwrapped_hz) > half_width_hz)
    upper_crosses_edge = bool(abs(upper_unwrapped_hz) > half_width_hz)
    channel_edge_notes: list[str] = []
    if lower_crosses_edge:
        channel_edge_notes.append(
            f"lower reference at {lower_unwrapped_hz:+.1f} Hz from the channel "
            f"center lies beyond the coarse-channel half-width "
            f"({half_width_hz:.1f} Hz); it aliases to "
            f"{lower_reference_offset_hz:+.1f} Hz and samples the "
            "channelizer roll-off"
        )
    if upper_crosses_edge:
        channel_edge_notes.append(
            f"upper reference at {upper_unwrapped_hz:+.1f} Hz from the channel "
            f"center lies beyond the coarse-channel half-width "
            f"({half_width_hz:.1f} Hz); it aliases to "
            f"{upper_reference_offset_hz:+.1f} Hz and samples the "
            "channelizer roll-off"
        )
    return {
        "physical_channel": channel,
        "dtv_pilot_hz": float(nominal_pilot_hz),
        "target_frequency_mhz": float(nominal_pilot_hz / HZ_PER_MHZ),
        "template_pilot_hz": float(pilot_hz),
        "pilot_offset_hz": float(pilot_hz - nominal_pilot_hz),
        "coarse_channel_index": int(selection.coarse_channel_index),
        "baseband_frame_center_normalized": float(frame_center),
        "baseband_frame_mode": frame_mode,
        "forbidden_dc_normalized": float(forbidden_dc),
        "coarse_channel_center_hz": float(selection.coarse_channel_center_hz),
        "fine_bin_offset_hz": float(selection.fine_bin_offset_hz),
        "detector_fine_bin_width_hz": float(fine_bin_width_hz),
        "target_offset_hz": float(target_offset_hz),
        "target_normalized_frequency": float(target_normalized),
        "lower_reference_normalized_frequency": float(lower_normalized),
        "upper_reference_normalized_frequency": float(upper_normalized),
        "lower_reference_offset_hz": float(lower_reference_offset_hz),
        "upper_reference_offset_hz": float(upper_reference_offset_hz),
        "lower_reference_relative_to_target_hz": float(
            lower_offset * fine_bin_width_hz
        ),
        "upper_reference_relative_to_target_hz": float(
            upper_offset * fine_bin_width_hz
        ),
        "symmetric_reference_offsets": bool(abs(lower_offset) == abs(upper_offset)),
        "reference_offset_bins_requested": int(core.reference_offset_bins),
        "skipped_guard_bins_requested": int(core.skipped_guard_bins),
        "reference_offset_bins_min_empirical": int(min_selected_offset),
        "skipped_guard_bins_min_empirical": int(max(0, min_selected_offset - 1)),
        "reference_offset_bins_selected": int(min_selected_offset),
        "skipped_guard_bins_selected": int(max(0, min_selected_offset - 1)),
        "strict_reference_offset_pass": bool(
            min_selected_offset >= int(core.reference_offset_bins)
        ),
        "lower_reference_offset_bins": int(lower_offset),
        "upper_reference_offset_bins": int(upper_offset),
        "lower_skipped_guard_bins": int(max(0, abs(lower_offset) - 1)),
        "upper_skipped_guard_bins": int(max(0, abs(upper_offset) - 1)),
        "reference_selection_score": float(
            REFERENCE_SELECTION_SCORE_ADAPTIVE
            if placement["adaptive_reference_placement"]
            else REFERENCE_SELECTION_SCORE_NOMINAL
        ),
        "adaptive_reference_placement": bool(
            placement["adaptive_reference_placement"]
        ),
        "reference_placement_reason": str(placement["reference_placement_reason"]),
        "reference_selection_method": REFERENCE_SELECTION_METHOD,
        "reference_placement_status": str(placement["placement_status"]),
        "placement_warnings": str(placement["placement_warnings"]),
        "edge_reference_wrapped": bool(placement["edge_reference_wrapped"]),
        "dc_reference_collision": bool(placement["dc_reference_collision"]),
        "dc_reference_shifted": bool(placement["dc_reference_shifted"]),
        "forbidden_tone_in_skipped_guard": bool(
            placement["forbidden_tone_in_skipped_guard"]
        ),
        "frame_origin_description": frame_origin_description(frame_center),
        "coarse_channel_half_width_hz": float(half_width_hz),
        "target_channel_edge_margin_hz": float(
            half_width_hz - abs(float(target_offset_hz))
        ),
        "lower_reference_unwrapped_offset_hz": float(lower_unwrapped_hz),
        "upper_reference_unwrapped_offset_hz": float(upper_unwrapped_hz),
        "lower_reference_crosses_channel_edge": lower_crosses_edge,
        "upper_reference_crosses_channel_edge": upper_crosses_edge,
        "reference_crosses_channel_edge": bool(
            lower_crosses_edge or upper_crosses_edge
        ),
        "channel_edge_notes": "; ".join(channel_edge_notes),
        "lower_reference_edge_wrapped": bool(lower["edge_wrapped"]),
        "upper_reference_edge_wrapped": bool(upper["edge_wrapped"]),
        "lower_reference_dc_shifted": bool(lower["dc_shifted"]),
        "upper_reference_dc_shifted": bool(upper["dc_shifted"]),
        "lower_reference_requested_offset_bins": int(lower["requested_offset_bins"]),
        "upper_reference_requested_offset_bins": int(upper["requested_offset_bins"]),
        "lower_reference_requested_normalized_frequency": float(
            lower_requested_normalized
        ),
        "upper_reference_requested_normalized_frequency": float(
            upper_requested_normalized
        ),
        "lower_reference_requested_offset_hz": float(
            lower_requested_reference_offset_hz
        ),
        "upper_reference_requested_offset_hz": float(
            upper_requested_reference_offset_hz
        ),
        "lower_reference_requested_relative_to_target_hz": float(
            int(lower["requested_offset_bins"]) * fine_bin_width_hz
        ),
        "upper_reference_requested_relative_to_target_hz": float(
            int(upper["requested_offset_bins"]) * fine_bin_width_hz
        ),
        "lower_reference_requested_edge_wrapped": bool(
            lower["requested_edge_wrapped"]
        ),
        "upper_reference_requested_edge_wrapped": bool(
            upper["requested_edge_wrapped"]
        ),
        "lower_reference_requested_dc_collision": bool(
            lower["requested_dc_collision"]
        ),
        "upper_reference_requested_dc_collision": bool(
            upper["requested_dc_collision"]
        ),
        "max_lower_reference_offset_bins": int(
            math.floor(target_normalized * int(core.detector_window_samples))
        ),
        "max_upper_reference_offset_bins": int(
            math.floor((1.0 - target_normalized) * int(core.detector_window_samples))
        ),
    }

def profile_requires_window_time_reversal(
    profile: ReceiverProfile,
    weight_coordinate_system: str,
) -> bool:
    """Return the adapter preprocessing declared by the profile."""
    coordinate_system = normalize_weight_coordinate_system(
        weight_coordinate_system
    )
    if coordinate_system == WEIGHT_COORDINATE_RAW_INPUT:
        return False
    return bool(profile.time_reverse_detector_windows_before_kernel)


def generate_weight_table_from_receiver_profile(
    *,
    profile: ReceiverProfile,
    core: DetectorCoreProfile,
    physical_channels: Sequence[int],
    weight_coordinate_system: str = WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    pilot_overrides_hz: Mapping[int, float] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return the weight table and manifest target-layout entries."""
    validate_integration_compatibility(profile=profile, detector_core=core)
    channels = [
        validate_uhf_physical_channel(channel)
        for channel in physical_channels
    ]
    # The receiver profile owns the deployed window length (K); the core
    # declares which lengths the compiled kernel family supports.
    # DetectorCoreProfile validates the selection in __post_init__.
    core = replace(
        core, detector_window_samples=int(profile.detector_window_samples)
    )
    coordinate_system = normalize_weight_coordinate_system(weight_coordinate_system)
    use_normal_override = (
        coordinate_system == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
        and not profile.time_reverse_detector_windows_before_kernel
    ) or (
        coordinate_system == WEIGHT_COORDINATE_RAW_INPUT
        and profile.time_reverse_detector_windows_before_kernel
    )
    generation_profile = (
        replace(profile, spectral_sense=SPECTRAL_SENSE_NORMAL)
        if use_normal_override
        else profile
    )
    table = np.zeros(
        (
            int(generation_profile.num_coarse_channels),
            int(core.num_weight_terms),
            int(core.detector_window_samples),
        ),
        dtype=np.int8,
    )
    layouts: list[dict[str, Any]] = []
    for channel in channels:
        layout = target_layout(
            physical_channel=channel,
            profile=generation_profile,
            core=core,
            pilot_hz_override=(pilot_overrides_hz or {}).get(int(channel)),
        )
        coarse_index = int(cast(int, layout["coarse_channel_index"]))
        if np.any(table[coarse_index]):
            raise ValueError(
                "multiple physical channels map to the same coarse channel: "
                f"coarse_channel_index={coarse_index}."
            )
        table[coarse_index, WEIGHT_TERM_TARGET] = _packed_weight_vector(
            float(layout["target_normalized_frequency"]),
            detector_window_samples=int(core.detector_window_samples),
            bits_per_component=int(core.sample_bits_per_component),
        )
        table[coarse_index, WEIGHT_TERM_REF_LOWER] = _packed_weight_vector(
            float(layout["lower_reference_normalized_frequency"]),
            detector_window_samples=int(core.detector_window_samples),
            bits_per_component=int(core.sample_bits_per_component),
        )
        table[coarse_index, WEIGHT_TERM_REF_UPPER] = _packed_weight_vector(
            float(layout["upper_reference_normalized_frequency"]),
            detector_window_samples=int(core.detector_window_samples),
            bits_per_component=int(core.sample_bits_per_component),
        )
        # Exact integer squared norms of the packed (post-quantization) weight
        # terms, recorded for provenance. null_power_ratio = 2*nt/(nl+nu) is the flat-floor
        # H0 zero-point of F; the norm-corrected positive-excess mask and the
        # kernel rational half-threshold nt:(nl+nu) divide it out.
        norm_nt, norm_nl, norm_nu = weight_term_norms_sq(
            table[coarse_index],
            bits_per_component=int(core.sample_bits_per_component),
        )
        layout = dict(layout)
        layout["target_norm_sq"] = int(norm_nt)
        layout["lower_reference_norm_sq"] = int(norm_nl)
        layout["upper_reference_norm_sq"] = int(norm_nu)
        layout["reference_norm_sum_sq"] = int(norm_nl + norm_nu)
        layout["null_power_ratio"] = float(null_power_ratio_from_weight_norms(norm_nt, norm_nl + norm_nu))
        layouts.append(layout)
    return np.ascontiguousarray(table), layouts


def _weight_header_bytes(
    *,
    core: DetectorCoreProfile,
    profile: ReceiverProfile,
    fine_bin_width_hz: float,
    weights_bytes: bytes,
) -> bytes:
    reference_name = b"profile_generated"
    profile_name = profile.receiver_profile_id.encode("utf-8")
    header_size = HEADER_FIXED_SIZE + len(reference_name) + len(profile_name) + CRC_SIZE
    fixed_without_crc = struct.pack(
        HEADER_FIXED_FMT,
        WEIGHT_MAGIC,
        int(WEIGHT_VERSION),
        int(header_size),
        int(core.detector_window_samples),
        int(core.num_weight_terms),
        int(core.reference_offset_bins),
        int(core.sample_bits_per_component),
        int(profile.num_coarse_channels),
        float(DEFAULT_DOPPLER_TOL_HZ),
        float(fine_bin_width_hz),
        int(len(reference_name)),
        int(len(profile_name)),
    )
    header_zero_crc = fixed_without_crc + reference_name + profile_name + struct.pack(
        "<I",
        0,
    )
    crc = zlib.crc32(header_zero_crc)
    crc = zlib.crc32(weights_bytes, crc) & CRC32_UNSIGNED_MASK
    return fixed_without_crc + reference_name + profile_name + struct.pack("<I", crc)


def write_weight_bank_from_receiver_profile(
    *,
    output_path: str | Path,
    profile: ReceiverProfile,
    core: DetectorCoreProfile,
    physical_channels: Sequence[int],
    weight_coordinate_system: str = WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    pilot_overrides_hz: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Write a packed weight bank plus the adjacent manifest and return manifest."""
    validate_integration_compatibility(profile=profile, detector_core=core)
    channels = [
        validate_uhf_physical_channel(channel)
        for channel in physical_channels
    ]
    core = core.with_detector_window_samples(int(profile.detector_window_samples))
    coordinate_system = normalize_weight_coordinate_system(weight_coordinate_system)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=channels,
        weight_coordinate_system=coordinate_system,
        pilot_overrides_hz=pilot_overrides_hz,
    )
    weights_bytes = table.tobytes(order="C")
    header = _weight_header_bytes(
        core=core,
        profile=profile,
        fine_bin_width_hz=float(profile.bin_enbw_hz),
        weights_bytes=weights_bytes,
    )
    payload = header + weights_bytes
    output.write_bytes(payload)
    manifest = {
        "schema_version": "pilotproxy_weight_manifest_v1",
        "pilot_overrides_hz": {
            str(int(ch)): float(hz) for ch, hz in sorted((pilot_overrides_hz or {}).items())
        },
        "weight_format_version": int(WEIGHT_VERSION),
        "weight_coordinate_system": coordinate_system,
        "input_coordinate_system": input_coordinate_system_for_weight_coordinate(
            coordinate_system
        ),
        "input_preprocessing": {
            "time_reverse_detector_windows_before_kernel": (
                profile_requires_window_time_reversal(profile, coordinate_system)
            ),
        },
        "kernel_spec": {
            "detector_window_samples": int(core.detector_window_samples),
            "num_weight_terms": int(core.num_weight_terms),
            "reference_offset_bins": int(core.reference_offset_bins),
            "skipped_guard_bins": int(core.skipped_guard_bins),
            "packed_complex_bits": int(core.packed_complex_bits),
            "sample_bits_per_component": int(core.sample_bits_per_component),
        },
        "receiver_profile_hash": receiver_profile_hash(profile),
        "receiver_profile": profile.to_dict(),
        "physical_channels": channels,
        "forbidden_tone_policy": {
            "forbidden_tone": "physical_data_dc",
            "forbidden_tone_normalized": float(
                layouts[0]["forbidden_dc_normalized"]
            ),
            "forbidden_collision_rule": REFERENCE_FORBIDDEN_COLLISION_RULE,
            "forbidden_collision_half_width_bins": 0.5,
            "forbidden_collision_half_width_normalized": float(
                0.5 / float(core.detector_window_samples)
            ),
        },
        "target_reference_layout": layouts,
        "artifacts": {
            "weights_path": str(output),
            "manifest_path": str(output.with_suffix(output.suffix + ".manifest.json")),
            "weights_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "DetectorCoreLayout",
    "generate_weight_table_from_receiver_profile",
    "parse_physical_channel_selection",
    "target_layout",
    "write_weight_bank_from_receiver_profile",
]

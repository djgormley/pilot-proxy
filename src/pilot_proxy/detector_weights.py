# coding=utf-8
"""Loader for the packed ATSC reference detector weight ROM."""

from __future__ import annotations

import json
import math
import operator
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import numpy as np

from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_RAW_INPUT,
    input_coordinate_system_for_weight_coordinate,
    normalize_weight_coordinate_system,
)
from pilot_proxy.provenance import file_sha256
from pilot_proxy.schema_identity import schema_token

from .atsc_channels import (
    ATSC_CHANNEL_WIDTH_HZ,
    ATSC_PILOT_OFFSET_HZ,
    ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ,
    ATSC_UHF_MIN_PHYSICAL_CHANNEL,
    physical_channel_to_pilot_hz,
)

WEIGHT_MAGIC = b"FSTATWGT1"
WEIGHT_VERSION = 3
HEADER_FIXED_FMT = "<9sIIIIIIIddHH"
HEADER_FIXED_SIZE = struct.calcsize(HEADER_FIXED_FMT)
CRC_SIZE = 4
# zlib.crc32 returns a signed-looking Python int on some platforms; mask to the
# unsigned 32-bit field stored in the weight-file trailer.
CRC32_UNSIGNED_MASK = 0xFFFFFFFF
# Physical ATSC pilot matching should be exact to the manifest, with a small
# tolerance for decimal MHz/Hertz roundoff in command-line inputs.
DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ = 10.0
HZ_PER_MHZ = 1.0e6
WEIGHT_MANIFEST_SCHEMA_NAME = "pilotproxy_weight_manifest"
WEIGHT_MANIFEST_SCHEMA_REVISION = 1
WEIGHT_MANIFEST_SCHEMA_TOKEN = schema_token(
    WEIGHT_MANIFEST_SCHEMA_NAME, WEIGHT_MANIFEST_SCHEMA_REVISION
)
_REFERENCE_FIELD_PART = "reference"
_OLD_GAP_FIELD_PART = "guard"
_OFFSET_FIELD_PART = "offset"
_BINS_FIELD_PART = "bins"
_NOMINAL_FIELD_PART = "nominal"
_REQUESTED_FIELD_PART = "requested"
_SELECTED_FIELD_PART = "selected"
_MIN_EMPIRICAL_FIELD_PART = "min_empirical"
DEPRECATED_DETECTOR_SPACING_FIELDS = frozenset(
    {
        "_".join((_REFERENCE_FIELD_PART, _OLD_GAP_FIELD_PART, _BINS_FIELD_PART)),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _NOMINAL_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _REQUESTED_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _SELECTED_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _MIN_EMPIRICAL_FIELD_PART,
            )
        ),
        "_".join((_OLD_GAP_FIELD_PART, _BINS_FIELD_PART)),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OFFSET_FIELD_PART,
                _BINS_FIELD_PART,
                _NOMINAL_FIELD_PART,
            )
        ),
    }
)

ATSC_CHANNEL_WIDTH_MHZ = ATSC_CHANNEL_WIDTH_HZ / HZ_PER_MHZ
ATSC_UHF_CHANNEL_14_PILOT_MHZ = (
    ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ + ATSC_PILOT_OFFSET_HZ
) / HZ_PER_MHZ
FREQUENCY_ORDER_ASCENDING_RF = "ascending_rf"
FREQUENCY_ORDER_DESCENDING_RF = "descending_rf"


@dataclass(frozen=True)
class WeightHeader:
    """Parsed header metadata for a serialized weight file."""

    magic: str
    version: int
    header_size: int
    detector_window_samples: int
    num_weight_terms: int
    reference_offset_bins: int
    component_bits: int
    num_channels: int
    doppler_tol_hz: Optional[float]
    fine_bin_width_hz: float
    reference_name: str
    profile_name: str
    crc32: int

    @property
    def K(self) -> int:
        return self.detector_window_samples

    @property
    def N(self) -> int:
        return self.num_weight_terms

    @property
    def n_channels(self) -> int:
        return self.num_channels


class InvalidWeightHeaderError(ValueError):
    """Raised when a weight file lacks the expected header."""


def _exact_integer(value: object, *, field: str) -> int:
    """Return an integer without accepting truncation or booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc


def _packed_dtype_for_component_bits(component_bits: int) -> np.dtype:
    bits = int(component_bits)
    if bits == 4:
        return np.dtype(np.int8)
    if bits == 8:
        return np.dtype(np.int16)
    raise ValueError(f"Unsupported packed component bit depth: {bits}")


def _extract_kernel_specs(expected_kernel) -> Optional[tuple[int, int, int, int]]:
    if expected_kernel is None:
        return None
    if isinstance(expected_kernel, (tuple, list)):
        if len(expected_kernel) < 4:
            raise TypeError(
                "expected_kernel must include K, N, bits, reference_offset_bins"
            )
        return (
            _exact_integer(expected_kernel[0], field="expected_kernel.K"),
            _exact_integer(expected_kernel[1], field="expected_kernel.N"),
            _exact_integer(expected_kernel[2], field="expected_kernel.bits"),
            _exact_integer(
                expected_kernel[3],
                field="expected_kernel.reference_offset_bins",
            ),
        )
    return (
        _exact_integer(getattr(expected_kernel, "K"), field="expected_kernel.K"),
        _exact_integer(getattr(expected_kernel, "N"), field="expected_kernel.N"),
        _exact_integer(
            getattr(expected_kernel, "bits"),
            field="expected_kernel.bits",
        ),
        _exact_integer(
            getattr(expected_kernel, "reference_offset_bins"),
            field="expected_kernel.reference_offset_bins",
        ),
    )


def read_header_and_weights(path: Path) -> tuple[WeightHeader, np.ndarray]:
    """Read and validate a detector weight file header plus payload."""
    path = Path(path)
    with path.open("rb") as f:
        fixed = f.read(HEADER_FIXED_SIZE)
        if len(fixed) != HEADER_FIXED_SIZE:
            raise ValueError(f"Weight file too small for header: {path}")

        (
            magic,
            version,
            header_size,
            detector_window_samples,
            num_weight_terms,
            reference_offset_bins,
            component_bits,
            num_channels,
            doppler_val,
            fine_bin_width_hz,
            reference_name_len,
            profile_name_len,
        ) = struct.unpack(HEADER_FIXED_FMT, fixed)

        if magic != WEIGHT_MAGIC:
            raise InvalidWeightHeaderError(
                "Weight file is missing the expected FSTATWGT1 header."
            )
        if version != WEIGHT_VERSION:
            raise ValueError(
                f"Unsupported weight version {version}; expected {WEIGHT_VERSION}."
            )

        expected_header_size = (
            HEADER_FIXED_SIZE
            + int(reference_name_len)
            + int(profile_name_len)
            + CRC_SIZE
        )
        if header_size != expected_header_size:
            raise ValueError(
                f"Header size mismatch: expected {expected_header_size}, "
                f"got {header_size}."
            )

        rest = f.read(header_size - HEADER_FIXED_SIZE)
        if len(rest) != header_size - HEADER_FIXED_SIZE:
            raise ValueError(f"Incomplete header data in weight file: {path}")

        reference_name_bytes = rest[:reference_name_len]
        profile_name_bytes = rest[
            reference_name_len : reference_name_len + profile_name_len
        ]
        crc = struct.unpack("<I", rest[-CRC_SIZE:])[0]
        weights_bytes = f.read()

    header_no_crc = fixed + reference_name_bytes + profile_name_bytes + struct.pack(
        "<I", 0
    )
    calc_crc = zlib.crc32(header_no_crc)
    calc_crc = zlib.crc32(weights_bytes, calc_crc) & CRC32_UNSIGNED_MASK
    if calc_crc != crc:
        raise ValueError(
            f"Weight file CRC mismatch: expected 0x{crc:08X}, "
            f"got 0x{calc_crc:08X}."
        )

    packed_dtype = _packed_dtype_for_component_bits(component_bits)
    weights = np.frombuffer(weights_bytes, dtype=packed_dtype).copy()
    doppler_tol_hz = None if math.isnan(doppler_val) else float(doppler_val)
    header = WeightHeader(
        magic=magic.decode("utf-8", errors="replace"),
        version=int(version),
        header_size=int(header_size),
        detector_window_samples=int(detector_window_samples),
        num_weight_terms=int(num_weight_terms),
        reference_offset_bins=int(reference_offset_bins),
        component_bits=int(component_bits),
        num_channels=int(num_channels),
        doppler_tol_hz=doppler_tol_hz,
        fine_bin_width_hz=float(fine_bin_width_hz),
        reference_name=reference_name_bytes.decode(errors="replace"),
        profile_name=profile_name_bytes.decode(errors="replace"),
        crc32=int(crc),
    )
    return header, weights


class DetectorWeightBank:
    """Load a prebuilt packed detector weight ROM and expose channel weights."""

    def __init__(
        self,
        profile: str = "atsc_reference",
        profile_name: str = "dtv",
        K: Optional[int] = None,
        N: int = 3,
        reference_offset_bins: Optional[int] = None,
        explicit_path: str | os.PathLike | None = None,
        expected_kernel: object | None = None,
    ) -> None:
        if explicit_path is None:
            raise ValueError("DetectorWeightBank requires explicit_path")

        self.path = Path(explicit_path)
        self.filename = str(self.path)
        self.header, flat = read_header_and_weights(self.path)
        self.manifest = _read_adjacent_manifest(self.path)
        _validate_manifest_spacing_schema(self.manifest)
        _validate_manifest_weight_binding(self.path, self.manifest)
        self.K = int(self.header.K)
        self.N = int(self.header.N)
        header_reference_offset_bins = self.header.reference_offset_bins
        if header_reference_offset_bins is None:
            raise ValueError("Weight header is missing reference_offset_bins.")
        self.reference_offset_bins = int(header_reference_offset_bins)
        self.component_bits = int(self.header.component_bits)
        self.bits = self.component_bits
        self.profile = str(profile)
        self.profile_name = str(profile_name)

        expected = _extract_kernel_specs(expected_kernel)
        if expected is not None:
            exp_k, exp_n, exp_bits, exp_offset = expected
            actual = (
                self.K,
                self.N,
                self.component_bits,
                self.reference_offset_bins,
            )
            if actual != (exp_k, exp_n, exp_bits, exp_offset):
                raise ValueError(
                    "Weight ROM does not match kernel specs: "
                    f"weights={actual}, kernel={(exp_k, exp_n, exp_bits, exp_offset)}"
                )
        if K is not None and _exact_integer(K, field="K") != self.K:
            raise ValueError(f"Weight K={self.K} does not match requested K={K}")
        if _exact_integer(N, field="N") != self.N:
            raise ValueError(f"Weight N={self.N} does not match requested N={N}")
        if (
            reference_offset_bins is not None
            and _exact_integer(
                reference_offset_bins,
                field="reference_offset_bins",
            )
            != self.reference_offset_bins
        ):
            raise ValueError(
                f"Weight reference_offset_bins={self.reference_offset_bins} "
                "does not match requested "
                f"reference_offset_bins={reference_offset_bins}"
            )

        expected_size = self.header.num_channels * self.N * self.K
        if flat.size != expected_size:
            raise ValueError(
                f"Weight payload has {flat.size} values; expected {expected_size}."
            )
        self.rom_table = np.ascontiguousarray(
            flat.reshape(self.header.num_channels, self.N, self.K)
        )

        receiver_grid = _receiver_frequency_grid_from_manifest(
            self.manifest,
            expected_num_channels=self.header.num_channels,
        )
        if receiver_grid is None:
            self.reference_freqs = np.empty(0, dtype=np.float64)
            self.detector_profile = {}
            self._receiver_band_mhz: tuple[float, float] | None = None
            self._receiver_frequency_order: str | None = None
            self._receiver_channel_width_mhz: float | None = None
        else:
            reference_freqs, detector_profile, receiver_band_mhz = receiver_grid
            self.reference_freqs = reference_freqs
            self.detector_profile = detector_profile
            self._receiver_band_mhz = receiver_band_mhz
            self._receiver_frequency_order = str(
                detector_profile["frequency_order"]
            )
            self._receiver_channel_width_mhz = float(
                detector_profile["coarse_channel_width_mhz"]
            )
        self._known_layout = _known_layout_from_manifest(self.manifest)
        self.known_pilot_frequencies_mhz = [
            float(cast(float, row["target_frequency_mhz"]))
            for row in self._known_layout
        ]
        self._layout_by_physical_channel = {
            _physical_channel_from_pilot_mhz(
                float(cast(float, row["target_frequency_mhz"]))
            ): row
            for row in self._known_layout
        }

    def _weights_for_channel_index(self, channel_index: int) -> tuple[np.ndarray | None, bool]:
        idx = _exact_integer(channel_index, field="coarse channel index")
        if idx < 0 or idx >= self.rom_table.shape[0]:
            raise ValueError(
                f"coarse channel index must be in [0, {self.rom_table.shape[0] - 1}], "
                f"got {idx}."
            )
        weights = np.ascontiguousarray(self.rom_table[idx])
        if np.any(weights):
            return weights, True
        return None, False

    def get_weights(self, freq_mhz: float) -> tuple[np.ndarray | None, bool]:
        """Return packed weights for the nearest reference-channel entry.

        This is an advanced nearest-channel lookup. Public detector paths should
        use get_weights_for_pilot_frequency or get_weights_for_physical_channel
        so the requested pilot is
        validated against the shipped weight manifest.
        """
        requested = _finite_frequency_mhz(freq_mhz)
        if self._receiver_band_mhz is None or self.reference_freqs.size == 0:
            raise ValueError(
                "Expert frequency lookup requires an adjacent weight manifest "
                "with an embedded receiver_profile."
            )
        lower_mhz, upper_mhz = self._receiver_band_mhz
        if not lower_mhz <= requested <= upper_mhz:
            raise ValueError(
                "Requested frequency is outside the receiver profile: "
                f"{requested:.6f} MHz is not in "
                f"[{lower_mhz:.6f}, {upper_mhz:.6f}] MHz."
            )
        if self._receiver_frequency_order is None:
            raise RuntimeError("receiver frequency order was not initialized")
        if self._receiver_channel_width_mhz is None:
            raise RuntimeError("receiver channel width was not initialized")
        if self._receiver_frequency_order == FREQUENCY_ORDER_DESCENDING_RF:
            channel_coordinate = (
                float(self.reference_freqs[0]) - requested
            ) / self._receiver_channel_width_mhz
        else:
            channel_coordinate = (
                requested - float(self.reference_freqs[0])
            ) / self._receiver_channel_width_mhz
        chan_idx = int(round(channel_coordinate))
        if not 0 <= chan_idx < self.rom_table.shape[0]:
            raise ValueError(
                "Requested frequency is outside the receiver profile channel "
                f"centers: {requested:.6f} MHz."
            )
        return self._weights_for_channel_index(chan_idx)

    def get_weights_for_pilot_frequency(
        self,
        freq_mhz: float,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> tuple[np.ndarray | None, bool]:
        """Return weights only when the requested MHz value matches a known pilot."""
        layout = self.layout_for_pilot_frequency(
            freq_mhz,
            tolerance_hz=tolerance_hz,
        )
        if not layout:
            raise ValueError(
                "Pilot-frequency and physical-channel lookups require the adjacent "
                "weight manifest; use get_weights() explicitly for an expert nearest-"
                "coarse-channel lookup."
            )
        return self._weights_for_channel_index(layout["coarse_channel_index"])

    def get_weights_for_physical_channel(
        self,
        channel: int,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> tuple[np.ndarray | None, bool]:
        """Return weights for an ATSC UHF physical channel."""
        physical_channel = _exact_integer(channel, field="physical channel")
        pilot_mhz = physical_channel_to_pilot_hz(physical_channel) / HZ_PER_MHZ
        return self.get_weights_for_pilot_frequency(
            pilot_mhz,
            tolerance_hz=tolerance_hz,
        )

    def layout_for_pilot_frequency(
        self,
        freq_mhz: float,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> dict[str, object]:
        """Return the manifest target/reference layout for a known pilot."""
        requested = _finite_frequency_mhz(freq_mhz)
        tolerance = _finite_nonnegative_tolerance_hz(tolerance_hz)
        if not self._known_layout:
            return {}
        known = np.asarray(self.known_pilot_frequencies_mhz, dtype=np.float64)
        nearest_idx = int(np.argmin(np.abs(known - requested)))
        delta_hz = abs(float(known[nearest_idx]) - requested) * HZ_PER_MHZ
        if delta_hz > tolerance:
            raise ValueError(
                "Requested DTV pilot frequency is not in the weight manifest: "
                f"{requested:.6f} MHz, nearest known "
                f"{float(known[nearest_idx]):.6f} MHz, delta={delta_hz:.3f} Hz."
            )
        return dict(self._known_layout[nearest_idx])

    def layout_for_physical_channel(
        self,
        channel: int,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> dict[str, object]:
        """Return the manifest target/reference layout for a physical channel."""
        physical_channel = _exact_integer(channel, field="physical channel")
        pilot_mhz = physical_channel_to_pilot_hz(physical_channel) / HZ_PER_MHZ
        return self.layout_for_pilot_frequency(
            pilot_mhz,
            tolerance_hz=tolerance_hz,
        )

    def supported_physical_channels(self) -> list[int]:
        """Return physical channels covered by this weight manifest."""
        return sorted(int(ch) for ch in self._layout_by_physical_channel)


def _read_adjacent_manifest(path: Path) -> dict[str, object]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _finite_frequency_mhz(value: object) -> float:
    """Return one finite MHz coordinate without accepting booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("freq_mhz must be a finite number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("freq_mhz must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError("freq_mhz must be finite.")
    return result


def _finite_nonnegative_tolerance_hz(value: object) -> float:
    """Return a finite nonnegative matching tolerance without boolean coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("tolerance_hz must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance_hz must be finite and nonnegative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("tolerance_hz must be finite and nonnegative.")
    return result


def _manifest_finite_number(value: object, *, field: str) -> float:
    """Read a finite numeric receiver-profile field from JSON."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Weight manifest {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Weight manifest {field} must be finite.")
    return result


def _receiver_frequency_grid_from_manifest(
    manifest: dict[str, object],
    *,
    expected_num_channels: int,
) -> tuple[np.ndarray, dict[str, object], tuple[float, float]] | None:
    """Build the ROM-index RF grid from its embedded receiver profile.

    Receiver profiles own channel order, center offset, RF bounds, and channel
    count.  Reconstructing a generic ascending 400--800 MHz grid would select
    the wrong ROM row for both the shipped CHIME and CHORD banks.
    """
    if not manifest:
        return None
    receiver_profile = manifest.get("receiver_profile")
    if not isinstance(receiver_profile, dict):
        raise ValueError(
            "Current weight manifest requires an embedded receiver_profile "
            "for frequency lookup."
        )
    # Keep this import local: integration.__init__ also exposes weight-generation
    # helpers that import this loader, so a module-level import creates a cycle.
    from pilot_proxy.integration.receiver_profile import (
        ReceiverProfile,
        receiver_frequency_to_channel,
        validate_weight_manifest_profile_hash,
    )

    # The manifest hash is the cryptographic binding between ROM row order and
    # the embedded receiver geometry.  Without this check a one-field edit to
    # frequency_axis.order silently selects a different row for every pilot.
    profile = ReceiverProfile.from_dict(dict(receiver_profile))
    validate_weight_manifest_profile_hash(manifest, profile)
    rf_band = receiver_profile.get("rf_band")
    channelizer = receiver_profile.get("channelizer")
    if not isinstance(rf_band, dict) or not isinstance(channelizer, dict):
        raise ValueError(
            "Weight manifest receiver_profile requires rf_band and channelizer "
            "objects."
        )
    frequency_axis = channelizer.get("frequency_axis")
    if not isinstance(frequency_axis, dict):
        raise ValueError(
            "Weight manifest receiver_profile.channelizer requires a "
            "frequency_axis object."
        )

    num_channels = channelizer.get("num_coarse_channels")
    if isinstance(num_channels, bool) or not isinstance(num_channels, int):
        raise ValueError(
            "Weight manifest receiver_profile.channelizer.num_coarse_channels "
            "must be an integer."
        )
    if num_channels != int(expected_num_channels):
        raise ValueError(
            "Weight manifest receiver channel count does not match the ROM: "
            f"{num_channels} != {int(expected_num_channels)}."
        )

    lower_hz = _manifest_finite_number(
        rf_band.get("lower_hz"),
        field="receiver_profile.rf_band.lower_hz",
    )
    upper_hz = _manifest_finite_number(
        rf_band.get("upper_hz"),
        field="receiver_profile.rf_band.upper_hz",
    )
    if upper_hz <= lower_hz:
        raise ValueError(
            "Weight manifest receiver_profile RF upper bound must exceed its "
            "lower bound."
        )
    center_offset_hz = _manifest_finite_number(
        channelizer.get("coarse_channel_center_offset_hz"),
        field=(
            "receiver_profile.channelizer.coarse_channel_center_offset_hz"
        ),
    )
    order = frequency_axis.get("order")
    if order not in (
        FREQUENCY_ORDER_ASCENDING_RF,
        FREQUENCY_ORDER_DESCENDING_RF,
    ):
        raise ValueError(
            "Weight manifest receiver frequency order must be "
            f"{FREQUENCY_ORDER_ASCENDING_RF!r} or "
            f"{FREQUENCY_ORDER_DESCENDING_RF!r}; got {order!r}."
        )

    channel_width_hz = (upper_hz - lower_hz) / float(num_channels)
    indices = np.arange(num_channels, dtype=np.float64)
    if order == FREQUENCY_ORDER_DESCENDING_RF:
        centers_hz = upper_hz - center_offset_hz - indices * channel_width_hz
    else:
        centers_hz = lower_hz + center_offset_hz + indices * channel_width_hz
    if not np.all(np.isfinite(centers_hz)):
        raise ValueError("Weight manifest receiver frequency grid is not finite.")
    if (
        float(np.min(centers_hz)) < lower_hz
        or float(np.max(centers_hz)) > upper_hz
    ):
        raise ValueError(
            "Weight manifest receiver frequency grid lies outside its RF band."
        )

    layout = manifest.get("target_reference_layout")
    if not isinstance(layout, list):
        raise ValueError(
            "Weight manifest target_reference_layout must be a list."
        )
    layout_channels: list[int] = []
    for row_index, row in enumerate(layout):
        if not isinstance(row, dict):
            raise ValueError(
                "Weight manifest target_reference_layout entries must be objects."
            )
        physical_channel = _exact_integer(
            row.get("physical_channel"),
            field=f"target_reference_layout[{row_index}].physical_channel",
        )
        expected_pilot_hz = physical_channel_to_pilot_hz(physical_channel)
        pilot_hz = _manifest_finite_number(
            row.get("dtv_pilot_hz"),
            field=f"target_reference_layout[{row_index}].dtv_pilot_hz",
        )
        target_mhz = _manifest_finite_number(
            row.get("target_frequency_mhz"),
            field=(
                f"target_reference_layout[{row_index}].target_frequency_mhz"
            ),
        )
        if pilot_hz != expected_pilot_hz or target_mhz != expected_pilot_hz / HZ_PER_MHZ:
            raise ValueError(
                "Weight manifest target-reference pilot identity disagrees with "
                f"physical channel {physical_channel}."
            )
        selection = receiver_frequency_to_channel(expected_pilot_hz, profile)
        coarse_index = _exact_integer(
            row.get("coarse_channel_index"),
            field=(
                f"target_reference_layout[{row_index}].coarse_channel_index"
            ),
        )
        if coarse_index != int(selection.coarse_channel_index):
            raise ValueError(
                "Weight manifest target-reference coarse index disagrees with "
                f"the receiver profile for physical channel {physical_channel}: "
                f"{coarse_index} != {int(selection.coarse_channel_index)}."
            )
        center_hz = _manifest_finite_number(
            row.get("coarse_channel_center_hz"),
            field=(
                f"target_reference_layout[{row_index}].coarse_channel_center_hz"
            ),
        )
        if center_hz != float(selection.coarse_channel_center_hz):
            raise ValueError(
                "Weight manifest target-reference coarse center disagrees with "
                f"the receiver profile for physical channel {physical_channel}."
            )
        layout_channels.append(physical_channel)
    if len(set(layout_channels)) != len(layout_channels):
        raise ValueError(
            "Weight manifest target_reference_layout physical channels must be unique."
        )
    manifest_channels = manifest.get("physical_channels")
    if not isinstance(manifest_channels, list):
        raise ValueError("Weight manifest physical_channels must be a list.")
    exact_manifest_channels = [
        _exact_integer(value, field=f"physical_channels[{index}]")
        for index, value in enumerate(manifest_channels)
    ]
    if exact_manifest_channels != layout_channels:
        raise ValueError(
            "Weight manifest physical_channels must exactly match the ordered "
            "target_reference_layout physical channels."
        )

    receiver_profile_id = receiver_profile.get("receiver_profile_id")
    if not isinstance(receiver_profile_id, str) or not receiver_profile_id:
        raise ValueError(
            "Weight manifest receiver_profile_id must be a non-empty string."
        )
    detector_profile = {
        "name": receiver_profile_id,
        "num_channels": int(num_channels),
        "band_lower_mhz": lower_hz / HZ_PER_MHZ,
        "band_upper_mhz": upper_hz / HZ_PER_MHZ,
        "bandwidth_mhz": (upper_hz - lower_hz) / HZ_PER_MHZ,
        "coarse_channel_width_mhz": channel_width_hz / HZ_PER_MHZ,
        "frequency_order": order,
    }
    return (
        np.ascontiguousarray(centers_hz / HZ_PER_MHZ),
        detector_profile,
        (lower_hz / HZ_PER_MHZ, upper_hz / HZ_PER_MHZ),
    )


def _known_layout_from_manifest(manifest: dict[str, object]) -> list[dict[str, object]]:
    layout = manifest.get("target_reference_layout", [])
    if not isinstance(layout, list):
        return []
    out: list[dict[str, object]] = []
    for row in layout:
        if not isinstance(row, dict):
            continue
        if "target_frequency_mhz" not in row or "coarse_channel_index" not in row:
            continue
        out.append(row)
    return out


def _validate_manifest_weight_binding(
    path: Path,
    manifest: dict[str, object],
) -> None:
    """Verify that an adjacent manifest names the exact binary being loaded."""
    if not manifest:
        return
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Weight manifest requires an artifacts object.")

    expected_sha256 = artifacts.get("weights_sha256")
    if expected_sha256 is None:
        raise ValueError(
            "Current weight manifests must bind the binary with "
            "artifacts.weights_sha256."
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(
            "Weight manifest/binary SHA256 mismatch: "
            f"{actual_sha256!r} != {str(expected_sha256)!r}."
        )


def _validate_manifest_spacing_schema(manifest: dict[str, object]) -> None:
    if not manifest:
        return
    schema_version = manifest.get("schema_version")
    if schema_version != WEIGHT_MANIFEST_SCHEMA_TOKEN:
        raise ValueError(
            "Unsupported weight manifest schema_version: "
            f"{schema_version!r}; expected {WEIGHT_MANIFEST_SCHEMA_TOKEN!r}."
        )
    coordinate = manifest.get("weight_coordinate_system")
    if coordinate is None:
        raise ValueError(
            "Current weight manifest requires weight_coordinate_system."
        )
    coordinate_system = normalize_weight_coordinate_system(coordinate)
    input_coordinate = manifest.get("input_coordinate_system")
    if input_coordinate is None:
        raise ValueError(
            "Current weight manifest requires input_coordinate_system."
        )
    expected_input_coordinate = input_coordinate_system_for_weight_coordinate(
        coordinate_system
    )
    if str(input_coordinate) != expected_input_coordinate:
        raise ValueError(
            "Weight manifest input_coordinate_system does not match "
            f"weight_coordinate_system: {input_coordinate!r} != "
            f"{expected_input_coordinate!r}."
        )
    input_preprocessing = manifest.get("input_preprocessing")
    if not isinstance(input_preprocessing, dict):
        raise ValueError("Current weight manifest requires input_preprocessing.")
    if "time_reverse_detector_windows_before_kernel" not in input_preprocessing:
        raise ValueError(
            "Weight manifest input_preprocessing requires "
            "time_reverse_detector_windows_before_kernel."
        )
    if (
        coordinate_system == WEIGHT_COORDINATE_RAW_INPUT
        and bool(input_preprocessing["time_reverse_detector_windows_before_kernel"])
    ):
        raise ValueError(
            "Raw input-coordinate weights must not request detector-window "
            "time reversal before the kernel."
        )
    _reject_deprecated_spacing_fields(manifest)
    kernel_spec = manifest.get("kernel_spec")
    if isinstance(kernel_spec, dict):
        _reject_deprecated_spacing_fields(kernel_spec)
    layout = manifest.get("target_reference_layout")
    if isinstance(layout, list):
        for row in layout:
            if isinstance(row, dict):
                _reject_deprecated_spacing_fields(row)


def _reject_deprecated_spacing_fields(data: dict[str, object]) -> None:
    for key in sorted(DEPRECATED_DETECTOR_SPACING_FIELDS):
        if key in data:
            raise ValueError(
                f"Deprecated detector-spacing field found: {key}. "
                "Use skipped_guard_bins or reference_offset_bins."
            )


def _physical_channel_from_pilot_mhz(pilot_mhz: float) -> int:
    channel_offset = (
        float(pilot_mhz) - ATSC_UHF_CHANNEL_14_PILOT_MHZ
    ) / ATSC_CHANNEL_WIDTH_MHZ
    return int(round(channel_offset)) + int(ATSC_UHF_MIN_PHYSICAL_CHANNEL)


__all__ = [
    "CRC32_UNSIGNED_MASK",
    "CRC_SIZE",
    "DetectorWeightBank",
    "HEADER_FIXED_FMT",
    "HEADER_FIXED_SIZE",
    "InvalidWeightHeaderError",
    "WEIGHT_MAGIC",
    "WEIGHT_VERSION",
    "WeightHeader",
    "read_header_and_weights",
]

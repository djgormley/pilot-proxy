# coding=utf-8
"""Fail-closed health repair for the completed CHIME per-pilot archive.

The archive products retain enough frame-level information to remove known
pathological rows from scalar and fine-statistic analyses.  They normally do
not retain per-frame Fourier spectra.  There is one deliberately narrow
exception: ``baseband_power_linear == 128`` proves that every decoded complex
int4 sample in the frame was ``(-8, -8)``.  Such a frame has a reconstructible
rectangular-FFT spectrum (DC only), so its contribution can be subtracted from
the two archived integrated spectra.  This module implements that exact repair
and refuses to claim a spectral repair for any excluded valid frame whose
samples are not similarly determined.

The native CHIME byte for ``(-8, -8)`` is offset-binary ``0x00``.  Lossless
repacking for the detector converts the same sample to two's-complement
``0x88``.  These byte coordinates are never interchangeable in reports.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from pilot_proxy.archived_product_keys import (
    ARCHIVED_TO_CURRENT,
    measurement,
)
from pilot_proxy.atomic_io import (
    atomic_write_json,
    create_temporary_sibling,
    fsync_directory,
)
from pilot_proxy.chime.products import atomic_savez_compressed
from pilot_proxy.config.project import default_project
from pilot_proxy.detector_geometry import (
    DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS,
    SPECTRAL_SENSE_INVERTED,
    SPECTRAL_SENSE_NORMAL,
    predicted_fine_designated_bins,
    predicted_pilot_fine_bin,
)
from pilot_proxy.fine_reduction import (
    CFAR_DEFAULT_FALLBACK_FLAG_FRACTION,
    CFAR_FALLBACK_LEFT_QUANTILE,
    CFAR_FALLBACK_LOCATION_QUANTILE,
    CFAR_LEFT_QUANTILE,
    CFAR_MODE_MEDIAN_LEFT,
    CFAR_MODE_QUANTILE_FALLBACK,
    fine_bin_frequencies_hz,
    independent_bin_mask,
    p_fa_to_threshold_k,
)
from pilot_proxy.json_utils import json_safe
from pilot_proxy.product_contract import fine_power_ratio_of, null_power_ratio_of
from pilot_proxy.provenance import file_sha256, package_source_sha256

FRAME_HEALTH_GATE_SCHEMA_VERSION = "pilotproxy_archive_frame_health_gate_v1"
ARCHIVE_HEALTH_SUMMARY_SCHEMA_VERSION = "pilotproxy_archive_health_summary_v1"
EXCLUSION_LEDGER_SCHEMA_VERSION = "pilotproxy_archive_exclusion_ledger_v1"
DIAGNOSTIC_MANIFEST_SCHEMA_VERSION = "pilotproxy_archive_diagnostic_manifest_v1"
CORRECTED_SPECTRA_SCHEMA_VERSION = "pilotproxy_archive_corrected_spectra_v1"
RESIDUAL_HEALTH_VIEW_SCHEMA_VERSION = "pilotproxy_baonoise_health_view_v1"

# Site of the project's instrument (CHIME at DRAO): east-positive longitude for
# local mean sidereal time, and the IANA zone of local civil time.
_INSTRUMENT = default_project().instrument
DRAO_LONGITUDE_DEGREES_EAST = _INSTRUMENT.site_longitude_deg
LOCAL_CIVIL_TIME_ZONE = _INSTRUMENT.local_time_zone
LMST_FORMULA_VERSION = "utc_as_ut1_iau1982_style_gmst_polynomial_v1"
LMST_FORMULA = (
    "JD = unix_utc_seconds / 86400 + 2440587.5; "
    "T = (JD - 2451545.0) / 36525; "
    "GMST_deg = 280.46061837 + 360.98564736629 * (JD - 2451545.0) "
    "+ 0.000387933 * T^2 - T^3 / 38710000; "
    "LMST_hours = ((GMST_deg + longitude_degrees_east) mod 360) / 15"
)
LMST_FORMULA_IMPLEMENTATION_SHA256 = hashlib.sha256(
    LMST_FORMULA.encode("ascii")
).hexdigest()

MEASURED_FINE_LINE_HALF_WIDTH_BINS = 2
MEASURED_FINE_ANCHOR_MIN_FRAMES = 30
MEASURED_FINE_ANCHOR_NORMALIZED_LINE_THRESHOLD = 1.5
MEASURED_FINE_ANCHOR_MIN_PERSISTENCE_FRACTION = 0.10
MEASURED_FINE_ANCHOR_MIN_COMPETITOR_MARGIN = 0.02
MEASURED_FINE_ANCHOR_COMPETITOR_CLEAR_HALF_WIDTH_BINS = 5

REASON_DETECTOR_INVALID = "detector_invalid"
REASON_DETECTOR_POWERS_ALL_ZERO = "detector_powers_all_zero"
REASON_BASEBAND_NONFINITE = "baseband_power_nonfinite"
REASON_BASEBAND_OUT_OF_BOUNDS = "baseband_power_out_of_complex_int4_bounds"
REASON_BASEBAND_CEILING = "baseband_power_at_negative_full_scale_ceiling"
REASON_COARSE_NONFINITE = "coarse_power_ratio_nonfinite"
REASON_FINE_NONFINITE = "fine_power_ratio_nonfinite"
REASON_FINE_NEGATIVE = "fine_power_ratio_negative"

REASON_DEFINITIONS: dict[str, str] = {
    REASON_DETECTOR_INVALID: (
        "The stored detector valid bit is false; the analyzer omitted this "
        "frame from both integrated spectra."
    ),
    REASON_DETECTOR_POWERS_ALL_ZERO: (
        "Both stored detector power terms are zero."
    ),
    REASON_BASEBAND_NONFINITE: (
        "The stored frame-mean baseband power is not finite."
    ),
    REASON_BASEBAND_OUT_OF_BOUNDS: (
        "The stored mean power lies outside the closed [0, 128] range of "
        "decoded complex int4 samples."
    ),
    REASON_BASEBAND_CEILING: (
        "The mean power is exactly 128, its mathematical maximum.  Every "
        "decoded sample is therefore (-8, -8): native CHIME offset-binary "
        "raw byte 0x00, equivalently detector-input/post-repack "
        "two's-complement byte 0x88."
    ),
    REASON_COARSE_NONFINITE: (
        "A detector-valid row has a non-finite coarse power ratio."
    ),
    REASON_FINE_NONFINITE: (
        "A detector-valid row contains a non-finite fine power ratio."
    ),
    REASON_FINE_NEGATIVE: (
        "A detector-valid row contains a negative fine power ratio."
    ),
}

ENCODING_INTERPRETATION = (
    "baseband_power_linear is evaluated after decoding the native CHIME "
    "offset-binary nibbles.  A frame mean of exactly 128 reaches the maximum "
    "possible sample power (-8)^2 + (-8)^2, so every native raw byte is 0x00. "
    "The losslessly repacked detector-input representation of that same "
    "sample is two's-complement 0x88."
)

SPECTRAL_LIMITATION = (
    "The NPZ files store only two accumulated power spectra, not per-frame "
    "complex FFTs.  The v1 gate can exactly remove detector-invalid rows "
    "(which contributed nothing) and the provably constant power-ceiling "
    "rows (whose DC-only spectrum is reconstructible).  It cannot apply an "
    "arbitrary new frame mask, FFT window, or threshold to the archived "
    "spectra.  Raw-voltage spectrograms and absolute calibrated PSDs remain "
    "unavailable without the source HDF5 data and calibration."
)

_REASON_ORDER = tuple(REASON_DEFINITIONS)
_VERSION_TOKEN = re.compile(r"(?P<key>[^=\s]+)=(?P<value>[^\s]+)")

_RESIDUAL_FRAME_FIELDS = (
    "frame_index",
    "valid",
    "reject_mask",
    "coarse_power_ratio",
    # RFIsher's residual reader needs the per-frame shelf estimate beside
    # the statistic; without it every health view is refused as neither a
    # current product nor a supported archived one.
    "estimated_data_shelf_snr_db",
    "frame_unit_index",
)
_RESIDUAL_METADATA_FIELDS = (
    "physical_channel",
    "freq_id",
    "chime_frequency_hz",
    "unit_time0_ctime",
    "detector_version",
    "detector_contract_json",
    "mask_rule",
    "pilot_below_data_db",
    "dtv_bandwidth_hz",
    "bin_enbw_hz",
)


class ArchiveHealthError(ValueError):
    """The archive cannot be interpreted safely under the v1 gate."""


@dataclass(frozen=True)
class FrameHealthResult:
    """Frame inclusion mask and every independently triggered reason mask."""

    include: np.ndarray
    reasons: Mapping[str, np.ndarray]

    @property
    def excluded(self) -> np.ndarray:
        return ~self.include

    @property
    def reason_counts(self) -> dict[str, int]:
        return {
            code: int(np.count_nonzero(mask))
            for code, mask in self.reasons.items()
            if np.any(mask)
        }


@dataclass(frozen=True)
class FineDiagnosticResult:
    """Health-filtered fine diagnostics with predicted and measured roles."""

    predicted_anchor_bin: int
    predicted_acquisition_bins: np.ndarray
    measured_line_half_width_bins: int
    epoch_anchor_records: tuple[Mapping[str, Any], ...]
    selected_anchor_bin_by_frame: np.ndarray
    measured_anchor_used_by_frame: np.ndarray
    null_bulk_mask: np.ndarray
    fine_frequency_hz: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    threshold: np.ndarray
    null_bulk_exceedance_fraction: np.ndarray
    fallback_mode: np.ndarray
    detected_count_all_bins: np.ndarray
    detected_count_predicted_acquisition: np.ndarray
    predicted_acquisition_peak: np.ndarray
    detected_count_selected_epoch_window: np.ndarray
    selected_epoch_window_peak: np.ndarray
    selected_anchor_value: np.ndarray


@dataclass(frozen=True)
class CorrectedSpectrumResult:
    """Exactly corrected integrated spectra for the v1 health exclusions."""

    before: np.ndarray
    after: np.ndarray
    healthy_before_count: int
    healthy_after_count: int
    ceiling_count_before: int
    ceiling_count_after: int
    ceiling_dc_power_per_frame: float
    before_parseval_expected_sum: float
    before_parseval_observed_sum: float
    before_parseval_relative_error: float
    after_parseval_expected_sum: float
    after_parseval_observed_sum: float
    after_parseval_relative_error: float
    parseval_relative_tolerance: float
    parseval_pass: bool
    exact: bool
    unavailable_reason: str | None


# current spelling -> the spelling the 2020--2026 archive used.
_ARCHIVED_SPELLING = {v: k for k, v in ARCHIVED_TO_CURRENT.items()}


def _array(product: Mapping[str, Any], name: str) -> np.ndarray:
    """Read one field, naming it in the current vocabulary.

    Archived products predate the measurement rename, so a current name that is
    absent is resolved once through the migration map. Anything still missing
    fails closed.
    """
    try:
        return measurement(product, name)
    except KeyError:
        raise ArchiveHealthError(
            f"archive product is missing required field {name!r}"
        ) from None


def _scalar(product: Mapping[str, Any], name: str) -> Any:
    arr = _array(product, name)
    if arr.size != 1:
        raise ArchiveHealthError(f"{name} must contain exactly one value; got {arr.shape}")
    return arr.reshape(()).item()


def _frame_vector(
    product: Mapping[str, Any], name: str, n_frames: int
) -> np.ndarray:
    arr = _array(product, name)
    if arr.shape == (n_frames, 1):
        return arr[:, 0]
    if arr.shape == (n_frames,):
        return arr
    raise ArchiveHealthError(
        f"{name} must have shape ({n_frames},) or ({n_frames}, 1); got {arr.shape}"
    )


def _binary_frame_vector(
    product: Mapping[str, Any], name: str, n_frames: int
) -> np.ndarray:
    values = _frame_vector(product, name, n_frames)
    if not np.all((values == 0) | (values == 1)):
        raise ArchiveHealthError(f"{name} must contain only exact 0/1 values")
    return values.astype(bool, copy=False)


def _validate_frame_identity(product: Mapping[str, Any], n_frames: int) -> None:
    frame_index = _frame_vector(product, "frame_index", n_frames)
    if not np.array_equal(frame_index, np.arange(n_frames, dtype=frame_index.dtype)):
        raise ArchiveHealthError("frame_index must be the exact positional sequence 0..N-1")
    unit_index = _frame_vector(product, "frame_unit_index", n_frames)
    frame_in_unit = _frame_vector(product, "frame_in_unit", n_frames)
    if unit_index.dtype.kind not in "iu" or frame_in_unit.dtype.kind not in "iu":
        raise ArchiveHealthError("frame identity arrays must use integer dtypes")
    source_keys = _array(product, "source_event_keys").reshape(-1)
    unit_order = _array(product, "unit_order").reshape(-1)
    if source_keys.size != unit_order.size or source_keys.size == 0:
        raise ArchiveHealthError(
            "source_event_keys and unit_order must be aligned non-empty unit axes"
        )
    if np.any(unit_index < 0) or np.any(unit_index >= source_keys.size):
        raise ArchiveHealthError("frame_unit_index contains an out-of-range unit row")
    if np.any(frame_in_unit < 0):
        raise ArchiveHealthError("frame_in_unit must be non-negative")


def evaluate_frame_health(product: Mapping[str, Any]) -> FrameHealthResult:
    """Evaluate the versioned v1 gate, excluding any unsafe row.

    Required fields and frame identities are validated before a mask is
    returned.  A malformed or unclassifiable product raises instead of
    silently treating the row as healthy.
    """

    frame_index = _array(product, "frame_index")
    if frame_index.ndim != 1:
        raise ArchiveHealthError("frame_index must be one-dimensional")
    n_frames = int(frame_index.size)
    if n_frames == 0:
        raise ArchiveHealthError("archive products with zero frames are not auditable")
    _validate_frame_identity(product, n_frames)

    valid = _binary_frame_vector(product, "valid", n_frames)
    _binary_frame_vector(product, "reject_mask", n_frames)
    p_target = _frame_vector(product, "p_target_u64", n_frames)
    p_reference = _frame_vector(product, "p_ref_sum_u64", n_frames)
    if p_target.dtype.kind not in "iu" or p_reference.dtype.kind not in "iu":
        raise ArchiveHealthError("stored detector powers must use integer dtypes")
    baseband_power = np.asarray(
        _frame_vector(product, "baseband_power_linear", n_frames), dtype=np.float64
    )
    coarse = np.asarray(
        _frame_vector(product, "coarse_power_ratio", n_frames),
        dtype=np.float64,
    )
    # The fine measurement is exact uint64 terms in current products and a
    # float32 ratio in archived ones. Integers admit neither a non-finite nor a
    # negative value, so those two checks apply only to the archived form.
    if "fine_power_u64" in product:
        fine_terms = _array(product, "fine_power_u64")
        if (
            fine_terms.ndim != 3
            or fine_terms.shape[0] != n_frames
            or fine_terms.shape[2] <= 0
        ):
            raise ArchiveHealthError(
                "fine_power_u64 must have shape (N, terms, B), B > 0; "
                f"got {fine_terms.shape}"
            )
        if fine_terms.dtype.kind != "u":
            raise ArchiveHealthError(
                "fine_power_u64 must use an unsigned integer dtype; got "
                f"{fine_terms.dtype}"
            )
        fine_nonfinite = np.zeros(n_frames, dtype=bool)
        fine_negative = np.zeros(n_frames, dtype=bool)
    else:
        fine = _array(product, "fine_power_ratio")
        if fine.ndim != 2 or fine.shape[0] != n_frames or fine.shape[1] <= 0:
            raise ArchiveHealthError(
                f"fine_power_ratio must have shape (N, B), B > 0; got {fine.shape}"
            )
        fine_nonfinite = ~np.all(np.isfinite(fine), axis=1)
        fine_negative = np.any(fine < 0.0, axis=1)

    reasons: dict[str, np.ndarray] = {
        REASON_DETECTOR_INVALID: ~valid,
        REASON_DETECTOR_POWERS_ALL_ZERO: (p_target == 0) & (p_reference == 0),
        REASON_BASEBAND_NONFINITE: ~np.isfinite(baseband_power),
        REASON_BASEBAND_OUT_OF_BOUNDS: (
            np.isfinite(baseband_power)
            & ((baseband_power < 0.0) | (baseband_power > 128.0))
        ),
        REASON_BASEBAND_CEILING: baseband_power == 128.0,
        REASON_COARSE_NONFINITE: valid & ~np.isfinite(coarse),
        REASON_FINE_NONFINITE: valid & fine_nonfinite,
        REASON_FINE_NEGATIVE: valid & fine_negative,
    }
    excluded = np.zeros(n_frames, dtype=bool)
    for mask in reasons.values():
        excluded |= mask
    return FrameHealthResult(include=~excluded, reasons=reasons)


def _residual_view_key(name: str) -> str:
    """Published column name for an internal field name.

    The view still publishes its archived spellings and schema value so older
    consumers can read it.
    """
    return _ARCHIVED_SPELLING.get(name, name)


def write_residual_health_view(source: Path, destination: Path) -> Path:
    """Write the minimal health-filtered NPZ residual-analysis view.

    The released residual APIs accept paths and load the per-frame columns
    internally.  Passing an original archive product would silently restore
    rows excluded by the v1 gate.  This deliberately incomplete derived view
    retains only the coarse-policy/residual columns those APIs consume, slices
    every frame column by the fail-closed inclusion mask, and records the
    source hash and gate identity.  It is not a replacement canonical product
    and intentionally omits the run's superseded fine-decision fields.
    """

    source_path = Path(source)
    source_digest = file_sha256(source_path)
    if source_digest is None:
        raise ArchiveHealthError(f"not a readable product file: {source_path}")
    with np.load(source_path, allow_pickle=False) as product:
        health = evaluate_frame_health(product)
        n_frames = int(health.include.size)
        arrays: dict[str, np.ndarray] = {}
        for name in _RESIDUAL_FRAME_FIELDS:
            value = _array(product, name)
            if value.ndim == 0 or value.shape[0] != n_frames:
                raise ArchiveHealthError(
                    f"residual frame field {name!r} must have leading length "
                    f"{n_frames}; got {value.shape}"
                )
            arrays[_residual_view_key(name)] = np.asarray(value[health.include])
        for name in _RESIDUAL_METADATA_FIELDS:
            arrays[_residual_view_key(name)] = np.asarray(_array(product, name))
        # Derived, not copied: the product stores the integer pair only.
        arrays[_residual_view_key("null_power_ratio")] = np.asarray(
            [null_power_ratio_of(product)], dtype=np.float64
        )

    included = int(np.count_nonzero(health.include))
    arrays["frame_index"] = np.arange(included, dtype=np.int64)
    if not np.all(np.asarray(arrays["valid"]) == 1):
        raise ArchiveHealthError(
            "health-filtered residual view unexpectedly retains an invalid row"
        )
    arrays.update(
        schema_version=np.asarray(RESIDUAL_HEALTH_VIEW_SCHEMA_VERSION),
        archive_health_gate_schema_version=np.asarray(
            FRAME_HEALTH_GATE_SCHEMA_VERSION
        ),
        source_product_name=np.asarray(source_path.name),
        source_product_sha256=np.asarray(source_digest),
        source_frame_count=np.asarray(n_frames, dtype=np.int64),
        health_included_frame_count=np.asarray(included, dtype=np.int64),
        health_excluded_frame_count=np.asarray(
            int(np.count_nonzero(health.excluded)), dtype=np.int64
        ),
        health_reason_counts_json=np.asarray(
            json.dumps(health.reason_counts, sort_keys=True)
        ),
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    return atomic_savez_compressed(output, **arrays)


@contextlib.contextmanager
def temporary_residual_health_views(
    product_paths: Sequence[Path],
) -> Iterator[list[Path]]:
    """Yield transient, v1-filtered inputs for path-only residual APIs."""

    paths = [Path(path) for path in product_paths]
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ArchiveHealthError(
            "cannot materialize residual health views with duplicate basenames"
        )
    with tempfile.TemporaryDirectory(prefix="pilotproxy-residual-health-") as root:
        root_path = Path(root)
        views = [
            write_residual_health_view(path, root_path / path.name) for path in paths
        ]
        yield views


def _spectral_sense_name(product: Mapping[str, Any]) -> str:
    value = int(_scalar(product, "sense"))
    if value == -1:
        return SPECTRAL_SENSE_INVERTED
    if value == 1:
        return SPECTRAL_SENSE_NORMAL
    raise ArchiveHealthError(f"sense must be +1 or -1, got {value}")


def _sample_rate_hz(product: Mapping[str, Any]) -> float:
    if "sample_rate_hz" in product:
        rate = float(_scalar(product, "sample_rate_hz"))
    else:
        delta = np.asarray(_array(product, "unit_delta_time"), dtype=np.float64)
        finite = delta[np.isfinite(delta) & (delta > 0.0)]
        if finite.size == 0:
            raise ArchiveHealthError("no finite positive sample period is available")
        if not np.allclose(finite, finite[0], rtol=0.0, atol=1.0e-15):
            raise ArchiveHealthError("unit_delta_time is not constant within the product")
        rate = 1.0 / float(finite[0])
    if not math.isfinite(rate) or rate <= 0.0:
        raise ArchiveHealthError("sample_rate_hz must be finite and positive")
    return rate


def corrected_fine_geometry(product: Mapping[str, Any]) -> dict[str, Any]:
    """Return the geometry-predicted acquisition neighborhood.

    This geometry is an acquisition prior, not an empirical line
    localization.  :func:`recompute_corrected_fine_diagnostics` separately
    measures narrow epoch anchors and records any refusal to do so.
    """

    nfft = int(_scalar(product, "nfft"))
    detector_window = int(_scalar(product, "detector_window_samples"))
    pad_factor = int(_scalar(product, "fine_pad_factor"))
    num_bins = int(_scalar(product, "fine_num_bins"))
    if nfft <= 0 or detector_window <= 0 or nfft % detector_window:
        raise ArchiveHealthError("nfft must be a positive multiple of detector_window_samples")
    expected_bins = pad_factor * (nfft // detector_window)
    if num_bins != expected_bins:
        raise ArchiveHealthError(
            f"fine_num_bins={num_bins} does not match pad*nfft/K={expected_bins}"
        )
    anchor = predicted_pilot_fine_bin(
        pilot_rf_hz=float(_scalar(product, "pilot_frequency_hz")),
        coarse_center_hz=float(_scalar(product, "chime_frequency_hz")),
        sample_rate_hz=_sample_rate_hz(product),
        detector_window_samples=detector_window,
        nfft=nfft,
        spectral_sense=_spectral_sense_name(product),
        pad_factor=pad_factor,
    )
    predicted_acquisition = np.asarray(
        predicted_fine_designated_bins(
            anchor,
            DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS,
            num_bins,
        ),
        dtype=np.int64,
    )
    census = np.asarray(
        _array(product, "fine_census_excluded_bins"), dtype=np.int64
    ).reshape(-1)
    guard = int(_scalar(product, "fine_guard_fine_bins"))
    bulk = independent_bin_mask(
        num_bins,
        pad_factor=pad_factor,
        designated_bins=predicted_acquisition,
        guard_fine_bins=guard,
        census_excluded_bins=census,
    )
    if np.count_nonzero(bulk) < 8:
        raise ArchiveHealthError("corrected fine null bulk has fewer than eight bins")
    stored = np.asarray(_array(product, "fine_designated_bins"), dtype=np.int64).reshape(-1)
    return {
        "predicted_anchor_bin": int(anchor),
        "predicted_acquisition_bins": predicted_acquisition,
        "predicted_acquisition_null_bulk_mask": bulk,
        "stored_designated_bins": stored,
        "stored_differs_from_predicted_acquisition": not np.array_equal(
            stored, predicted_acquisition
        ),
        "fine_frequency_hz": fine_bin_frequencies_hz(
            nfft // detector_window,
            _sample_rate_hz(product) / detector_window,
            pad_factor=pad_factor,
        ),
        "guard_fine_bins": guard,
        "pad_factor": pad_factor,
        "p_fa": float(_scalar(product, "fine_p_fa")),
        "census_excluded_bins": census,
    }


def _circular_bin_delta(measured: int, predicted: int, n_bins: int) -> int:
    """Signed measured-minus-predicted delta on a circular even-size grid."""

    return int((int(measured) - int(predicted) + n_bins // 2) % n_bins - n_bins // 2)


def _measure_fine_epoch_anchors(
    product: Mapping[str, Any],
    gate: FrameHealthResult,
    geometry: Mapping[str, Any],
    fine: np.ndarray,
) -> tuple[tuple[Mapping[str, Any], ...], np.ndarray, np.ndarray]:
    """Measure persistent narrow line anchors independently in UTC quarters."""

    n_frames, n_bins = fine.shape
    predicted = int(geometry["predicted_anchor_bin"])
    selected_anchor = np.full(n_frames, predicted, dtype=np.int32)
    measured_used = np.zeros(n_frames, dtype=bool)
    times = frame_utc_seconds(product)
    finite_time = np.isfinite(times)
    epoch = np.full(n_frames, "timestamp_unavailable", dtype="U24")
    for frame in np.flatnonzero(finite_time):
        value = dt.datetime.fromtimestamp(float(times[frame]), dt.timezone.utc)
        epoch[frame] = f"{value.year:04d}Q{(value.month - 1) // 3 + 1}"
    frequency = np.asarray(geometry["fine_frequency_hz"], dtype=np.float64)
    if frequency.size != n_bins:
        raise ArchiveHealthError("fine frequency axis and fine statistic do not align")
    fine_bin_hz = _sample_rate_hz(product) / int(
        _scalar(product, "detector_window_samples")
    ) / n_bins
    predicted_window = np.asarray(
        geometry["predicted_acquisition_bins"], dtype=np.int64
    )
    predicted_window_set = set(int(value) for value in predicted_window)
    outside_predicted_window = np.asarray(
        [index for index in range(n_bins) if index not in predicted_window_set],
        dtype=np.int64,
    )
    records: list[Mapping[str, Any]] = []
    health_epochs = sorted(set(epoch[gate.include].tolist()))
    for epoch_key in health_epochs:
        rows = np.flatnonzero(gate.include & (epoch == epoch_key))
        if epoch_key == "timestamp_unavailable":
            records.append(
                {
                    "epoch_key": epoch_key,
                    "epoch_definition": (
                        "retrospective provisional UTC calendar quarter at "
                        "frame start; not an authoritative station epoch"
                    ),
                    "first_frame_utc": None,
                    "last_frame_utc": None,
                    "health_included_frames": int(rows.size),
                    "estimator_usable_frames": 0,
                    "status": "fallback_predicted_acquisition",
                    "candidate_anchor_bin": None,
                    "selected_anchor_bin": predicted,
                    "selected_window_half_width_bins": int(
                        DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS
                    ),
                    "selected_window_bins": predicted_window.astype(int).tolist(),
                    "refusal_reasons": ["frame_timestamp_unavailable"],
                }
            )
            continue
        block = fine[rows]
        row_bulk = np.median(block, axis=1)
        usable = np.isfinite(row_bulk) & (row_bulk > 0.0)
        usable_block = block[usable]
        normalized = (
            usable_block / row_bulk[usable, np.newaxis]
            if np.any(usable)
            else np.empty((0, n_bins), dtype=np.float64)
        )
        candidate: int | None = None
        persistence: np.ndarray | None = None
        line_ratio: np.ndarray | None = None
        peak_persistence: float | None = None
        competitor_persistence: float | None = None
        competitor_margin: float | None = None
        outside_sentinel: int | None = None
        if normalized.size:
            persistence = np.mean(
                normalized >= MEASURED_FINE_ANCHOR_NORMALIZED_LINE_THRESHOLD,
                axis=0,
            )
            line_ratio = np.median(normalized, axis=0)
            candidate = max(
                predicted_window.tolist(),
                key=lambda index: (
                    float(persistence[index]),
                    float(line_ratio[index]),
                    -index,
                ),
            )
            peak_persistence = float(persistence[candidate])
            competitor_clear = {
                (candidate + offset) % n_bins
                for offset in range(
                    -MEASURED_FINE_ANCHOR_COMPETITOR_CLEAR_HALF_WIDTH_BINS,
                    MEASURED_FINE_ANCHOR_COMPETITOR_CLEAR_HALF_WIDTH_BINS + 1,
                )
            }
            competitor_persistence = float(
                max(
                    [
                        persistence[index]
                        for index in predicted_window
                        if int(index) not in competitor_clear
                    ]
                    or [0.0]
                )
            )
            competitor_margin = peak_persistence - competitor_persistence
            if outside_predicted_window.size:
                outside_sentinel = max(
                    outside_predicted_window.tolist(),
                    key=lambda index: (
                        float(persistence[index]),
                        float(line_ratio[index]),
                        -index,
                    ),
                )
        refusal: list[str] = []
        if int(np.count_nonzero(usable)) < MEASURED_FINE_ANCHOR_MIN_FRAMES:
            refusal.append("fewer_than_minimum_usable_health_frames")
        if peak_persistence is None or (
            peak_persistence < MEASURED_FINE_ANCHOR_MIN_PERSISTENCE_FRACTION
        ):
            refusal.append("persistence_below_minimum")
        if competitor_margin is None or (
            competitor_margin < MEASURED_FINE_ANCHOR_MIN_COMPETITOR_MARGIN
        ):
            refusal.append("peak_not_separated_from_off_window_competitor")
        candidate_delta = (
            _circular_bin_delta(int(candidate), predicted, n_bins)
            if candidate is not None
            else None
        )
        candidate_edge_clearance = (
            DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS - abs(candidate_delta)
            if candidate_delta is not None
            else None
        )
        if (
            candidate is not None
            and outside_sentinel is not None
            and persistence is not None
            and candidate_edge_clearance is not None
            and candidate_edge_clearance <= MEASURED_FINE_LINE_HALF_WIDTH_BINS
            and float(persistence[outside_sentinel]) > float(persistence[candidate])
        ):
            refusal.append(
                "candidate_at_acquisition_edge_with_stronger_external_sentinel"
            )
        accepted = candidate is not None and not refusal
        chosen = int(candidate) if accepted else predicted
        half_width = (
            MEASURED_FINE_LINE_HALF_WIDTH_BINS
            if accepted
            else DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS
        )
        chosen_window = np.asarray(
            predicted_fine_designated_bins(chosen, half_width, n_bins),
            dtype=np.int64,
        )
        if accepted:
            selected_anchor[rows] = chosen
            measured_used[rows] = True
        delta = candidate_delta
        records.append(
            {
                "epoch_key": epoch_key,
                "epoch_definition": (
                    "retrospective provisional UTC calendar quarter at frame "
                    "start; not an authoritative station epoch"
                ),
                "first_frame_utc": _finite_time_bound(times[rows]),
                "last_frame_utc": _finite_time_bound(times[rows], latest=True),
                "health_included_frames": int(rows.size),
                "estimator_usable_frames": int(np.count_nonzero(usable)),
                "status": (
                    "measured_narrow_line_anchor"
                    if accepted
                    else "fallback_predicted_acquisition"
                ),
                "candidate_anchor_bin": (
                    int(candidate) if candidate is not None else None
                ),
                "candidate_anchor_frequency_hz": (
                    float(frequency[candidate]) if candidate is not None else None
                ),
                "candidate_minus_predicted_circular_bins": delta,
                "candidate_minus_predicted_frequency_hz": (
                    float(delta * fine_bin_hz) if delta is not None else None
                ),
                "candidate_distance_inside_acquisition_edge_bins": (
                    int(candidate_edge_clearance)
                    if candidate_edge_clearance is not None
                    else None
                ),
                "circular_delta_sign_convention": (
                    "candidate measured-array bin minus geometry-predicted "
                    "bin, wrapped to [-N/2, N/2); positive follows increasing "
                    "fine-array index"
                ),
                "candidate_persistence_fraction": peak_persistence,
                "strongest_off_window_competitor_persistence_fraction": (
                    competitor_persistence
                ),
                "competitor_search_domain": (
                    "inside the geometry-predicted acquisition neighborhood, "
                    "excluding the candidate +/- competitor-clear width"
                ),
                "persistence_margin_over_competitor": competitor_margin,
                "candidate_median_per_frame_bulk_ratio": (
                    float(line_ratio[candidate])
                    if candidate is not None and line_ratio is not None
                    else None
                ),
                "strongest_outside_acquisition_sentinel": (
                    {
                        "bin": int(outside_sentinel),
                        "frequency_hz": float(frequency[outside_sentinel]),
                        "persistence_fraction": float(
                            persistence[outside_sentinel]
                        ),
                        "median_per_frame_bulk_ratio": float(
                            line_ratio[outside_sentinel]
                        ),
                        "interpretation": (
                            "reported as a possible unrelated/instrument line; "
                            "it cannot redefine the target pilot anchor"
                        ),
                    }
                    if outside_sentinel is not None
                    and persistence is not None
                    and line_ratio is not None
                    else None
                ),
                "selected_anchor_bin": chosen,
                "selected_window_half_width_bins": int(half_width),
                "selected_window_bins": chosen_window.astype(int).tolist(),
                "refusal_reasons": refusal,
            }
        )
    return tuple(records), selected_anchor, measured_used


def recompute_corrected_fine_diagnostics(
    product: Mapping[str, Any],
    health: FrameHealthResult | None = None,
    *,
    chunk_rows: int = 4096,
) -> FineDiagnosticResult:
    """Recompute float CFAR with distinct predicted and measured line roles.

    This reproduces :func:`pilot_proxy.fine_reduction.calibrate_cfar` in
    vectorized chunks.  The broad geometry-predicted neighborhood is an
    acquisition diagnostic.  A narrow +/-2-bin window is used only in UTC
    quarters where a health-filtered persistent line passes the declared
    evidence thresholds; other quarters explicitly fall back to the broad
    predicted neighborhood.  This remains retrospective and does not replace
    the active coarse rejection bit stored in ``reject_mask``.
    """

    gate = evaluate_frame_health(product) if health is None else health
    geometry = corrected_fine_geometry(product)
    fine = fine_power_ratio_of(product)
    n_frames, n_bins = fine.shape
    if gate.include.shape != (n_frames,):
        raise ArchiveHealthError("health mask and fine product have different frame counts")
    step = int(chunk_rows)
    if step <= 0:
        raise ValueError("chunk_rows must be positive")

    records, selected_anchor, measured_used = _measure_fine_epoch_anchors(
        product, gate, geometry, fine
    )
    predicted = np.asarray(
        geometry["predicted_acquisition_bins"], dtype=np.int64
    )
    null_excluded = set(int(value) for value in predicted)
    for record in records:
        if record["status"] == "measured_narrow_line_anchor":
            null_excluded.update(int(value) for value in record["selected_window_bins"])
    bulk = independent_bin_mask(
        n_bins,
        pad_factor=int(geometry["pad_factor"]),
        designated_bins=sorted(null_excluded),
        guard_fine_bins=int(geometry["guard_fine_bins"]),
        census_excluded_bins=np.asarray(
            geometry["census_excluded_bins"], dtype=np.int64
        ),
    )
    if np.count_nonzero(bulk) < 8:
        raise ArchiveHealthError(
            "measured-anchor-aware fine null bulk has fewer than eight bins"
        )

    fill = lambda: np.full(n_frames, np.nan, dtype=np.float64)
    location, scale, threshold = fill(), fill(), fill()
    null_fraction = fill()
    predicted_peak, selected_peak, selected_anchor_value = fill(), fill(), fill()
    fallback = np.zeros(n_frames, dtype=bool)
    detected_all = np.zeros(n_frames, dtype=np.int32)
    detected_predicted = np.zeros(n_frames, dtype=np.int32)
    detected_selected = np.zeros(n_frames, dtype=np.int32)
    anchor = int(geometry["predicted_anchor_bin"])
    k = p_fa_to_threshold_k(float(geometry["p_fa"]))

    selected = np.flatnonzero(gate.include)
    for start in range(0, selected.size, step):
        rows = selected[start : start + step]
        block = fine[rows]
        null = block[:, bulk]
        loc = np.quantile(null, 0.5, axis=1)
        left = np.quantile(null, CFAR_LEFT_QUANTILE, axis=1)
        scl = loc - left
        std = np.std(null, axis=1)
        scl = np.where(scl > 0.0, scl, np.where(std > 0.0, std, 1.0))
        thr = loc + k * scl
        frac = np.mean(null > thr[:, np.newaxis], axis=1)
        use_fallback = frac > CFAR_DEFAULT_FALLBACK_FLAG_FRACTION
        if np.any(use_fallback):
            fb_null = null[use_fallback]
            fb_loc = np.quantile(
                fb_null, CFAR_FALLBACK_LOCATION_QUANTILE, axis=1
            )
            fb_left = np.quantile(fb_null, CFAR_FALLBACK_LEFT_QUANTILE, axis=1)
            fb_scale = np.maximum((fb_loc - fb_left) / 2.0, 1.0e-12)
            fb_thr = fb_loc + k * fb_scale
            loc[use_fallback] = fb_loc
            scl[use_fallback] = fb_scale
            thr[use_fallback] = fb_thr
            frac[use_fallback] = np.mean(
                fb_null > fb_thr[:, np.newaxis], axis=1
            )

        hits = block > thr[:, np.newaxis]
        location[rows], scale[rows], threshold[rows] = loc, scl, thr
        null_fraction[rows] = frac
        fallback[rows] = use_fallback
        detected_all[rows] = np.sum(hits, axis=1, dtype=np.int32)
        detected_predicted[rows] = np.sum(
            hits[:, predicted], axis=1, dtype=np.int32
        )
        predicted_peak[rows] = np.max(block[:, predicted], axis=1)
        detected_selected[rows] = detected_predicted[rows]
        selected_peak[rows] = predicted_peak[rows]
        selected_anchor_value[rows] = block[:, anchor]
        chunk_selected_anchor = selected_anchor[rows]
        chunk_measured = measured_used[rows]
        for measured_anchor in np.unique(chunk_selected_anchor[chunk_measured]):
            local = chunk_measured & (chunk_selected_anchor == measured_anchor)
            window = np.asarray(
                predicted_fine_designated_bins(
                    int(measured_anchor),
                    MEASURED_FINE_LINE_HALF_WIDTH_BINS,
                    n_bins,
                ),
                dtype=np.int64,
            )
            detected_selected[rows[local]] = np.sum(
                hits[local][:, window], axis=1, dtype=np.int32
            )
            selected_peak[rows[local]] = np.max(
                block[local][:, window], axis=1
            )
            selected_anchor_value[rows[local]] = block[
                local, int(measured_anchor)
            ]

    return FineDiagnosticResult(
        predicted_anchor_bin=anchor,
        predicted_acquisition_bins=predicted,
        measured_line_half_width_bins=MEASURED_FINE_LINE_HALF_WIDTH_BINS,
        epoch_anchor_records=records,
        selected_anchor_bin_by_frame=selected_anchor,
        measured_anchor_used_by_frame=measured_used,
        null_bulk_mask=bulk,
        fine_frequency_hz=np.asarray(geometry["fine_frequency_hz"], dtype=np.float64),
        location=location,
        scale=scale,
        threshold=threshold,
        null_bulk_exceedance_fraction=null_fraction,
        fallback_mode=fallback,
        detected_count_all_bins=detected_all,
        detected_count_predicted_acquisition=detected_predicted,
        predicted_acquisition_peak=predicted_peak,
        detected_count_selected_epoch_window=detected_selected,
        selected_epoch_window_peak=selected_peak,
        selected_anchor_value=selected_anchor_value,
    )


def _ceiling_dc_power_per_frame(product: Mapping[str, Any]) -> float:
    """Emulate the production complex64 ``abs(FFT)**2`` DC arithmetic."""

    nfft = int(_scalar(product, "nfft"))
    streams = int(_scalar(product, "num_input_streams"))
    if nfft <= 0 or streams <= 0:
        raise ArchiveHealthError("nfft and num_input_streams must be positive")
    dc = np.asarray([complex(-8 * nfft, -8 * nfft)], dtype=np.complex64)
    # Array arithmetic is intentional: both NumPy and CuPy produce float32 for
    # abs(complex64_array)**2.  For nfft=16384 this is 34359736320 per stream,
    # not the infinite-precision 34359738368, because abs rounds before square.
    power_per_stream = float(np.asarray(np.abs(dc) ** 2, dtype=np.float64)[0])
    return float(power_per_stream * streams)


def health_correct_integrated_spectra(
    product: Mapping[str, Any],
    health: FrameHealthResult | None = None,
) -> CorrectedSpectrumResult:
    """Exactly remove v1-reconstructible exclusions from archived spectra.

    Detector-invalid rows were never accumulated.  A valid ceiling-power row
    is the constant native byte ``0x00`` and contributes only one known DC-bin
    value.  Any other excluded valid row makes exact correction unavailable.
    """

    gate = evaluate_frame_health(product) if health is None else health
    n_frames = gate.include.size
    valid = _binary_frame_vector(product, "valid", n_frames)
    reject = _binary_frame_vector(product, "reject_mask", n_frames)
    ceiling = np.asarray(
        _frame_vector(product, "baseband_power_linear", n_frames), dtype=np.float64
    ) == 128.0
    unknown_valid_exclusions = gate.excluded & valid & ~ceiling
    before_stored = np.asarray(
        _array(product, "integrated_spectrum_before_mask"), dtype=np.float64
    ).reshape(-1)
    after_stored = np.asarray(
        _array(product, "integrated_spectrum_after_mask"), dtype=np.float64
    ).reshape(-1)
    nfft = int(_scalar(product, "nfft"))
    if before_stored.shape != (nfft,) or after_stored.shape != (nfft,):
        raise ArchiveHealthError("integrated spectra must each have length nfft")
    healthy_before = int(np.count_nonzero(gate.include & valid))
    healthy_after = int(np.count_nonzero(gate.include & valid & ~reject))
    n_ceiling_before = int(np.count_nonzero(ceiling & valid))
    n_ceiling_after = int(np.count_nonzero(ceiling & valid & ~reject))
    dc_power = _ceiling_dc_power_per_frame(product)
    parseval_tolerance = 5.0e-7

    def _unchecked_result(reason: str) -> CorrectedSpectrumResult:
        return CorrectedSpectrumResult(
            before=before_stored.copy(),
            after=after_stored.copy(),
            healthy_before_count=healthy_before,
            healthy_after_count=healthy_after,
            ceiling_count_before=n_ceiling_before,
            ceiling_count_after=n_ceiling_after,
            ceiling_dc_power_per_frame=dc_power,
            before_parseval_expected_sum=float("nan"),
            before_parseval_observed_sum=float(np.sum(before_stored)),
            before_parseval_relative_error=float("nan"),
            after_parseval_expected_sum=float("nan"),
            after_parseval_observed_sum=float(np.sum(after_stored)),
            after_parseval_relative_error=float("nan"),
            parseval_relative_tolerance=parseval_tolerance,
            parseval_pass=False,
            exact=False,
            unavailable_reason=reason,
        )

    if np.any(unknown_valid_exclusions):
        return _unchecked_result(
            f"{int(np.count_nonzero(unknown_valid_exclusions))} excluded valid "
            "frame(s) do not have a reconstructible spectrum"
        )

    before = before_stored.copy()
    after = after_stored.copy()
    before[0] -= n_ceiling_before * dc_power
    after[0] -= n_ceiling_after * dc_power
    scale = max(float(np.max(before_stored)), float(np.max(after_stored)), 1.0)
    tolerance = 32.0 * np.finfo(np.float64).eps * scale
    if before[0] < -tolerance or after[0] < -tolerance:
        raise ArchiveHealthError(
            "ceiling-frame DC subtraction exceeds an archived spectrum; "
            "the product does not match the recorded FFT arithmetic"
        )
    if abs(before[0]) <= tolerance:
        before[0] = 0.0
    if abs(after[0]) <= tolerance:
        after[0] = 0.0
    if np.any(before < -tolerance) or np.any(after < -tolerance):
        raise ArchiveHealthError("health-corrected integrated spectra contain negative bins")
    baseband = np.asarray(
        _frame_vector(product, "baseband_power_linear", n_frames), dtype=np.float64
    )
    parseval_factor = float(
        int(_scalar(product, "nfft")) ** 2
        * int(_scalar(product, "num_input_streams"))
    )
    before_expected = float(
        parseval_factor * np.sum(baseband[gate.include & valid], dtype=np.float64)
    )
    after_expected = float(
        parseval_factor
        * np.sum(baseband[gate.include & valid & ~reject], dtype=np.float64)
    )
    before_observed = float(np.sum(before, dtype=np.float64))
    after_observed = float(np.sum(after, dtype=np.float64))
    before_relative = (before_observed - before_expected) / max(
        abs(before_expected), 1.0
    )
    after_relative = (after_observed - after_expected) / max(
        abs(after_expected), 1.0
    )
    parseval_pass = bool(
        abs(before_relative) <= parseval_tolerance
        and abs(after_relative) <= parseval_tolerance
    )
    if not parseval_pass:
        raise ArchiveHealthError(
            "health-corrected spectra fail the frame-power Parseval check: "
            f"before_relative_error={before_relative:.6g}, "
            f"after_relative_error={after_relative:.6g}, "
            f"tolerance={parseval_tolerance:.6g}"
        )
    return CorrectedSpectrumResult(
        before=before,
        after=after,
        healthy_before_count=healthy_before,
        healthy_after_count=healthy_after,
        ceiling_count_before=n_ceiling_before,
        ceiling_count_after=n_ceiling_after,
        ceiling_dc_power_per_frame=dc_power,
        before_parseval_expected_sum=before_expected,
        before_parseval_observed_sum=before_observed,
        before_parseval_relative_error=before_relative,
        after_parseval_expected_sum=after_expected,
        after_parseval_observed_sum=after_observed,
        after_parseval_relative_error=after_relative,
        parseval_relative_tolerance=parseval_tolerance,
        parseval_pass=parseval_pass,
        exact=True,
        unavailable_reason=None,
    )


def proportion_summary(successes: int, total: int) -> dict[str, Any]:
    """Binomial proportion, standard error, and two-sided Wilson 95% interval."""

    n = int(total)
    k = int(successes)
    if n < 0 or k < 0 or k > n:
        raise ValueError("proportion counts must satisfy 0 <= successes <= total")
    if n == 0:
        return {
            "count": k,
            "total": n,
            "fraction": None,
            "standard_error": None,
            "wilson_95": {"low": None, "high": None},
        }
    p = k / n
    z = NormalDist().inv_cdf(0.975)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return {
        "count": k,
        "total": n,
        "fraction": p,
        "standard_error": math.sqrt(p * (1.0 - p) / n),
        "wilson_95": {"low": max(0.0, center - half), "high": min(1.0, center + half)},
    }


def distribution_summary(values: np.ndarray, *, histogram_bins: int = 64) -> dict[str, Any]:
    """Finite-sample descriptive distribution with a machine-readable histogram."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        "count_total": int(arr.size),
        "count_finite": int(finite.size),
        "count_nonfinite": int(arr.size - finite.size),
    }
    if finite.size == 0:
        out.update(
            mean=None,
            standard_deviation=None,
            quantiles={},
            histogram={"edges": [], "counts": []},
        )
        return out
    probabilities = np.asarray([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    quantiles = np.quantile(finite, probabilities)
    if np.all(finite == finite[0]):
        width = max(abs(float(finite[0])) * 1.0e-6, 1.0e-12)
        hist_range = (float(finite[0] - width), float(finite[0] + width))
    else:
        hist_range = (float(np.min(finite)), float(np.max(finite)))
    counts, edges = np.histogram(finite, bins=int(histogram_bins), range=hist_range)
    out.update(
        mean=float(np.mean(finite)),
        standard_deviation=float(np.std(finite)),
        quantiles={
            f"p{int(round(probability * 100)):02d}": float(value)
            for probability, value in zip(probabilities, quantiles)
        },
        histogram={
            "edges": edges.astype(float).tolist(),
            "counts": counts.astype(int).tolist(),
        },
    )
    return out


def frame_utc_seconds(product: Mapping[str, Any]) -> np.ndarray:
    """Derive one UTC Unix timestamp per frame from the unit time axes."""

    n_frames = int(_array(product, "frame_index").size)
    unit_index = np.asarray(
        _frame_vector(product, "frame_unit_index", n_frames), dtype=np.int64
    )
    frame_in_unit = np.asarray(
        _frame_vector(product, "frame_in_unit", n_frames), dtype=np.float64
    )
    time0 = np.asarray(_array(product, "unit_time0_ctime"), dtype=np.float64).reshape(-1)
    delta = np.asarray(_array(product, "unit_delta_time"), dtype=np.float64).reshape(-1)
    if time0.size != delta.size:
        raise ArchiveHealthError("unit_time0_ctime and unit_delta_time must align")
    return time0[unit_index] + frame_in_unit * int(_scalar(product, "nfft")) * delta[unit_index]


def frame_duration_seconds(product: Mapping[str, Any]) -> np.ndarray:
    """Return the rectangular analysis duration represented by each frame."""

    n_frames = int(_array(product, "frame_index").size)
    unit_index = np.asarray(
        _frame_vector(product, "frame_unit_index", n_frames), dtype=np.int64
    )
    delta = np.asarray(_array(product, "unit_delta_time"), dtype=np.float64).reshape(-1)
    if np.any(unit_index < 0) or np.any(unit_index >= delta.size):
        raise ArchiveHealthError("frame_unit_index is outside unit_delta_time")
    return int(_scalar(product, "nfft")) * delta[unit_index]


def unix_utc_to_lmst_hours(
    unix_utc_seconds: np.ndarray | Sequence[float] | float,
    *,
    longitude_degrees_east: float = DRAO_LONGITUDE_DEGREES_EAST,
) -> np.ndarray:
    """Approximate local mean sidereal time with UTC used as the UT1 proxy.

    The dependency-free polynomial is an IAU-1982-style GMST expression.  It
    is used only to assign the survey exposure to one-hour LMST bins.  The
    products do not retain DUT1, so this function deliberately reports mean,
    not apparent, sidereal time and does not claim precision timing.
    """

    seconds = np.asarray(unix_utc_seconds, dtype=np.float64)
    julian_date = seconds / 86_400.0 + 2_440_587.5
    days_since_j2000 = julian_date - 2_451_545.0
    centuries_since_j2000 = days_since_j2000 / 36_525.0
    gmst_degrees = (
        280.46061837
        + 360.98564736629 * days_since_j2000
        + 0.000387933 * centuries_since_j2000**2
        - centuries_since_j2000**3 / 38_710_000.0
    )
    return np.mod(gmst_degrees + float(longitude_degrees_east), 360.0) / 15.0


def _exposure_bin_rows(
    labels: np.ndarray,
    durations: np.ndarray,
    include: np.ndarray,
    *,
    ordered_labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Count frames and rectangular-frame seconds in deterministic bins."""

    label_array = np.asarray(labels, dtype=str).reshape(-1)
    duration_array = np.asarray(durations, dtype=np.float64).reshape(-1)
    include_array = np.asarray(include, dtype=bool).reshape(-1)
    if not (
        label_array.size == duration_array.size == include_array.size
    ):
        raise ArchiveHealthError("exposure-bin inputs must align")
    if ordered_labels is None:
        bin_labels = sorted(np.unique(label_array).tolist())
    else:
        bin_labels = [str(value) for value in ordered_labels]
        unexpected = sorted(set(label_array.tolist()) - set(bin_labels))
        if unexpected:
            raise ArchiveHealthError(
                f"exposure labels are outside the declared bins: {unexpected}"
            )
    lookup = {label: index for index, label in enumerate(bin_labels)}
    codes = np.fromiter(
        (lookup[str(label)] for label in label_array),
        dtype=np.int64,
        count=label_array.size,
    )
    n_bins = len(bin_labels)
    duration_available = np.isfinite(duration_array) & (duration_array > 0.0)
    stored_counts = np.bincount(codes, minlength=n_bins)
    included_counts = np.bincount(
        codes, weights=include_array.astype(np.int64), minlength=n_bins
    )
    stored_duration_counts = np.bincount(
        codes, weights=duration_available.astype(np.int64), minlength=n_bins
    )
    included_duration_counts = np.bincount(
        codes,
        weights=(include_array & duration_available).astype(np.int64),
        minlength=n_bins,
    )
    stored_seconds = np.bincount(
        codes,
        weights=np.where(duration_available, duration_array, 0.0),
        minlength=n_bins,
    )
    included_seconds = np.bincount(
        codes,
        weights=np.where(include_array & duration_available, duration_array, 0.0),
        minlength=n_bins,
    )
    return [
        {
            "label": label,
            "stored_frames": int(stored_counts[index]),
            "health_included_frames": int(included_counts[index]),
            "health_excluded_frames": int(
                stored_counts[index] - included_counts[index]
            ),
            "stored_frames_with_duration": int(stored_duration_counts[index]),
            "health_included_frames_with_duration": int(
                included_duration_counts[index]
            ),
            "stored_exposure_seconds": float(stored_seconds[index]),
            "health_included_exposure_seconds": float(included_seconds[index]),
            "health_excluded_exposure_seconds": float(
                stored_seconds[index] - included_seconds[index]
            ),
        }
        for index, label in enumerate(bin_labels)
    ]


def _time_exposure_summary(
    product: Mapping[str, Any],
    times: np.ndarray,
    include: np.ndarray,
) -> dict[str, Any]:
    """Build reproducible health-filtered UTC, civil, season, and LMST exposure."""

    timestamp = np.asarray(times, dtype=np.float64).reshape(-1)
    included = np.asarray(include, dtype=bool).reshape(-1)
    duration = frame_duration_seconds(product)
    if not (timestamp.size == included.size == duration.size):
        raise ArchiveHealthError("frame time, duration, and health arrays must align")
    timestamp_available = np.isfinite(timestamp)
    duration_available = np.isfinite(duration) & (duration > 0.0)
    complete_coordinate = timestamp_available & duration_available
    temporally_resolved = timestamp_available
    resolved_times = timestamp[temporally_resolved]
    resolved_duration = duration[temporally_resolved]
    resolved_include = included[temporally_resolved]
    try:
        local_zone = ZoneInfo(LOCAL_CIVIL_TIME_ZONE)
    except ZoneInfoNotFoundError as exc:
        raise ArchiveHealthError(
            f"time-zone database lacks {LOCAL_CIVIL_TIME_ZONE!r}"
        ) from exc
    try:
        utc_datetimes = [
            dt.datetime.fromtimestamp(float(value), dt.timezone.utc)
            for value in resolved_times
        ]
    except (OSError, OverflowError, ValueError) as exc:
        raise ArchiveHealthError(
            "finite unit_time0_ctime produced an unsupported calendar date"
        ) from exc
    local_datetimes = [value.astimezone(local_zone) for value in utc_datetimes]
    utc_month = np.asarray(
        [f"{value.year:04d}-{value.month:02d}" for value in utc_datetimes]
    )
    local_month = np.asarray(
        [f"{value.year:04d}-{value.month:02d}" for value in local_datetimes]
    )
    hour_labels = [f"{hour:02d}:00-{hour + 1:02d}:00" for hour in range(24)]
    utc_hour = np.asarray([hour_labels[value.hour] for value in utc_datetimes])
    local_hour = np.asarray([hour_labels[value.hour] for value in local_datetimes])
    season_for_month = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }
    local_season = np.asarray(
        [season_for_month[value.month] for value in local_datetimes]
    )
    lmst = unix_utc_to_lmst_hours(resolved_times)
    lmst_hour_index = np.floor(lmst).astype(np.int64) % 24
    lmst_hour = np.asarray([hour_labels[int(value)] for value in lmst_hour_index])
    all_duration_available = bool(np.all(duration_available))
    all_timestamp_available = bool(np.all(timestamp_available))
    return {
        "status": (
            "available"
            if all_timestamp_available and all_duration_available
            else "partial"
        ),
        "frame_time_definition": (
            "unit_time0_ctime[frame_unit_index] + frame_in_unit * nfft * "
            "unit_delta_time[frame_unit_index], interpreted as UTC Unix "
            "seconds at the rectangular frame start"
        ),
        "frame_exposure_definition": (
            "nfft * unit_delta_time[frame_unit_index]; each complete frame's "
            "duration is credited to the bin containing its start time rather "
            "than split across a bin boundary"
        ),
        "coverage": {
            "stored_frames": int(timestamp.size),
            "health_included_frames": int(np.count_nonzero(included)),
            "frames_with_finite_timestamp": int(
                np.count_nonzero(timestamp_available)
            ),
            "health_included_frames_with_finite_timestamp": int(
                np.count_nonzero(included & timestamp_available)
            ),
            "frames_with_positive_finite_duration": int(
                np.count_nonzero(duration_available)
            ),
            "health_included_frames_with_positive_finite_duration": int(
                np.count_nonzero(included & duration_available)
            ),
            "frames_with_complete_time_and_duration": int(
                np.count_nonzero(complete_coordinate)
            ),
            "health_included_frames_with_complete_time_and_duration": int(
                np.count_nonzero(included & complete_coordinate)
            ),
            "unavailable_reason": (
                None
                if all_timestamp_available and all_duration_available
                else (
                    "one or more retained units lack a finite UTC start or a "
                    "positive finite sample period; those frames are counted "
                    "in coverage but cannot contribute complete temporal "
                    "exposure coordinates"
                )
            ),
        },
        "utc_calendar_month": {
            "coordinate": "UTC calendar year-month",
            "bins": _exposure_bin_rows(
                utc_month, resolved_duration, resolved_include
            ),
        },
        "utc_hour_of_day": {
            "coordinate": "UTC civil hour at frame start",
            "bin_edges_hours": list(range(25)),
            "bins": _exposure_bin_rows(
                utc_hour,
                resolved_duration,
                resolved_include,
                ordered_labels=hour_labels,
            ),
        },
        "local_civil_calendar_month": {
            "coordinate": "local civil calendar year-month",
            "iana_time_zone": LOCAL_CIVIL_TIME_ZONE,
            "daylight_saving_handling": "IANA tzdb conversion for each UTC instant",
            "bins": _exposure_bin_rows(
                local_month, resolved_duration, resolved_include
            ),
        },
        "local_civil_hour_of_day": {
            "coordinate": "local civil hour at frame start; not solar or sidereal time",
            "iana_time_zone": LOCAL_CIVIL_TIME_ZONE,
            "bin_edges_hours": list(range(25)),
            "bins": _exposure_bin_rows(
                local_hour,
                resolved_duration,
                resolved_include,
                ordered_labels=hour_labels,
            ),
        },
        "local_meteorological_season": {
            "coordinate": "local civil meteorological season",
            "iana_time_zone": LOCAL_CIVIL_TIME_ZONE,
            "definition": {
                "DJF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
            },
            "bins": _exposure_bin_rows(
                local_season,
                resolved_duration,
                resolved_include,
                ordered_labels=["DJF", "MAM", "JJA", "SON"],
            ),
        },
        "local_mean_sidereal_hour": {
            "coordinate": "DRAO local mean sidereal time (LMST), not LAST",
            "longitude_degrees_east": DRAO_LONGITUDE_DEGREES_EAST,
            "utc_to_earth_rotation_assumption": (
                "UTC is used as the UT1 proxy because DUT1/Earth-orientation "
                "values were not retained; this is sufficient for one-hour "
                "exposure bins but not precision astrometry"
            ),
            "formula_version": LMST_FORMULA_VERSION,
            "formula": LMST_FORMULA,
            "formula_implementation_sha256": LMST_FORMULA_IMPLEMENTATION_SHA256,
            "validation": {
                "reference": "USNO Sidereal Time API v4.0.1 LMST output",
                "maximum_reference_vector_difference_sidereal_seconds": 0.01,
            },
            "bin_edges_hours": list(range(25)),
            "bins": _exposure_bin_rows(
                lmst_hour,
                resolved_duration,
                resolved_include,
                ordered_labels=hour_labels,
            ),
        },
    }


def _version_tokens(version: str) -> dict[str, str]:
    out: dict[str, str] = {}
    fields = str(version).split()
    if fields:
        out["package"] = fields[0]
    for match in _VERSION_TOKEN.finditer(str(version)):
        out[match.group("key")] = match.group("value")
    return out


def _iso_utc(timestamp: float) -> str | None:
    if not math.isfinite(float(timestamp)):
        return None
    return dt.datetime.fromtimestamp(float(timestamp), dt.timezone.utc).isoformat()


def _finite_time_bound(values: np.ndarray, *, latest: bool = False) -> str | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    value = np.max(finite) if latest else np.min(finite)
    return _iso_utc(float(value))


def exclusion_ledger_entries(
    product: Mapping[str, Any],
    health: FrameHealthResult,
    *,
    product_path: Path,
    product_sha256: str,
) -> list[dict[str, Any]]:
    """Build stable event/frame/reason keyed records for every excluded row."""

    n_frames = health.include.size
    fid = int(_scalar(product, "freq_id"))
    channel = int(_scalar(product, "physical_channel"))
    unit_index = np.asarray(
        _frame_vector(product, "frame_unit_index", n_frames), dtype=np.int64
    )
    frame_in_unit = np.asarray(
        _frame_vector(product, "frame_in_unit", n_frames), dtype=np.int64
    )
    source_keys = np.asarray(_array(product, "source_event_keys")).reshape(-1)
    unit_order = np.asarray(_array(product, "unit_order")).reshape(-1)
    event_ids = np.asarray(_array(product, "unit_event_id")).reshape(-1)
    times = frame_utc_seconds(product)
    valid = _binary_frame_vector(product, "valid", n_frames)
    reject = _binary_frame_vector(product, "reject_mask", n_frames)
    baseband = np.asarray(
        _frame_vector(product, "baseband_power_linear", n_frames), dtype=np.float64
    )
    p_target = _frame_vector(product, "p_target_u64", n_frames)
    p_reference = _frame_vector(product, "p_ref_sum_u64", n_frames)
    rows: list[dict[str, Any]] = []
    for frame in np.flatnonzero(health.excluded):
        reason_codes = [
            code for code in _REASON_ORDER if bool(health.reasons[code][frame])
        ]
        unit = int(unit_index[frame])
        reason_token = "+".join(reason_codes)
        key = (
            f"{FRAME_HEALTH_GATE_SCHEMA_VERSION}/fid-{fid}/"
            f"source-{source_keys[unit]}/frame-{int(frame_in_unit[frame])}/"
            f"reasons-{reason_token}"
        )
        rows.append(
            {
                "schema_version": EXCLUSION_LEDGER_SCHEMA_VERSION,
                "gate_schema_version": FRAME_HEALTH_GATE_SCHEMA_VERSION,
                "ledger_key": key,
                "reason_codes": reason_codes,
                "freq_id": fid,
                "physical_channel": channel,
                "product_name": product_path.name,
                "product_sha256": product_sha256,
                "frame_index": int(frame),
                "frame_in_unit": int(frame_in_unit[frame]),
                "unit_index": unit,
                "source_event_key": str(source_keys[unit]),
                "unit_key": str(unit_order[unit]),
                "event_id": int(event_ids[unit]),
                "frame_time_utc": _iso_utc(float(times[frame])),
                "valid": bool(valid[frame]),
                "reject_mask": bool(reject[frame]),
                "baseband_power_linear": float(baseband[frame]),
                "p_target_u64": int(p_target[frame]),
                "p_ref_sum_u64": int(p_reference[frame]),
            }
        )
    return rows


def audit_product(
    path: Path,
    *,
    fine_chunk_rows: int = 4096,
) -> tuple[dict[str, Any], list[dict[str, Any]], CorrectedSpectrumResult]:
    """Audit one per-pilot product and return summary, ledger, and spectra."""

    product_path = Path(path)
    digest = file_sha256(product_path)
    if digest is None:
        raise ArchiveHealthError(f"not a readable product file: {product_path}")
    with np.load(product_path, allow_pickle=False) as product:
        health = evaluate_frame_health(product)
        fine = recompute_corrected_fine_diagnostics(
            product, health, chunk_rows=fine_chunk_rows
        )
        spectra = health_correct_integrated_spectra(product, health)
        n_frames = health.include.size
        valid = _binary_frame_vector(product, "valid", n_frames)
        reject = _binary_frame_vector(product, "reject_mask", n_frames)
        mu0 = null_power_ratio_of(product)
        coarse = np.asarray(
            _frame_vector(product, "coarse_power_ratio", n_frames),
            dtype=np.float64,
        )
        include = health.include
        corrected_ratio = np.divide(
            fine.selected_epoch_window_peak,
            fine.threshold,
            out=np.full(n_frames, np.nan),
            where=np.isfinite(fine.threshold) & (fine.threshold > 0.0),
        )
        times = frame_utc_seconds(product)
        version = str(_scalar(product, "detector_version"))
        geometry = corrected_fine_geometry(product)
        summary = {
            "freq_id": int(_scalar(product, "freq_id")),
            "physical_channel": int(_scalar(product, "physical_channel")),
            "product_name": product_path.name,
            "product_bytes": int(product_path.stat().st_size),
            "product_sha256": digest,
            "stored_schema_version": str(_scalar(product, "schema_version")),
            "detector_version": version,
            "detector_version_tokens": _version_tokens(version),
            "weight_bank_sha256": str(_scalar(product, "weight_bank_sha256")),
            "weight_manifest_sha256": str(_scalar(product, "weight_manifest_sha256")),
            "weights_hash": str(_scalar(product, "weights_hash")),
            "frame_counts": {
                "stored": n_frames,
                "stored_valid": int(np.count_nonzero(valid)),
                "included_by_health_gate": int(np.count_nonzero(include)),
                "excluded_unique": int(np.count_nonzero(health.excluded)),
                "reason_counts": health.reason_counts,
            },
            "health_exclusion_rate": proportion_summary(
                int(np.count_nonzero(health.excluded)), n_frames
            ),
            "stored_mask_rate_on_stored_valid": proportion_summary(
                int(np.count_nonzero(reject & valid)), int(np.count_nonzero(valid))
            ),
            "stored_mask_rate_on_health_included": proportion_summary(
                int(np.count_nonzero(reject & include)), int(np.count_nonzero(include))
            ),
            "recomputed_fine_any_bin_rate": proportion_summary(
                int(np.count_nonzero((fine.detected_count_all_bins > 0) & include)),
                int(np.count_nonzero(include)),
            ),
            "fine_predicted_acquisition_neighborhood_rate": proportion_summary(
                int(
                    np.count_nonzero(
                        (fine.detected_count_predicted_acquisition > 0) & include
                    )
                ),
                int(np.count_nonzero(include)),
            ),
            "fine_selected_epoch_window_rate": proportion_summary(
                int(
                    np.count_nonzero(
                        (fine.detected_count_selected_epoch_window > 0) & include
                    )
                ),
                int(np.count_nonzero(include)),
            ),
            "recomputed_fine_cfar_fallback_rate": proportion_summary(
                int(np.count_nonzero(fine.fallback_mode & include)),
                int(np.count_nonzero(include)),
            ),
            "scalar_distributions_health_included": {
                "coarse_power_ratio": distribution_summary(coarse[include]),
                "coarse_power_ratio_over_mu0": distribution_summary(
                    coarse[include] / mu0
                ),
                "normalized_coarse_power_ratio_db": distribution_summary(
                    _frame_vector(
                        product,
                        "normalized_coarse_power_ratio_db",
                        n_frames,
                    )[include]
                ),
                "pilot_excess_db": distribution_summary(
                    _frame_vector(product, "pilot_excess_db", n_frames)[include]
                ),
                "estimated_data_shelf_snr_db": distribution_summary(
                    _frame_vector(product, "estimated_data_shelf_snr_db", n_frames)[include]
                ),
                "baseband_power_linear": distribution_summary(
                    _frame_vector(product, "baseband_power_linear", n_frames)[include]
                ),
                "recomputed_fine_location": distribution_summary(fine.location[include]),
                "recomputed_fine_scale": distribution_summary(fine.scale[include]),
                "recomputed_fine_threshold": distribution_summary(fine.threshold[include]),
                "fine_selected_anchor_value": distribution_summary(
                    fine.selected_anchor_value[include]
                ),
                "fine_predicted_acquisition_peak": distribution_summary(
                    fine.predicted_acquisition_peak[include]
                ),
                "fine_selected_epoch_window_peak": distribution_summary(
                    fine.selected_epoch_window_peak[include]
                ),
                "fine_selected_epoch_window_peak_over_threshold": distribution_summary(
                    corrected_ratio[include]
                ),
            },
            "fine_anchor_audit": {
                "role": (
                    "retrospective float diagnostic only; the stored coarse "
                    "reject_mask remains the active survey decision"
                ),
                "predicted_acquisition_neighborhood": {
                    "role": (
                        "broad geometry prior for acquisition/diagnosis; not "
                        "a final calibrated narrow designation"
                    ),
                    "anchor_bin": int(fine.predicted_anchor_bin),
                    "half_width_bins": int(
                        DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS
                    ),
                    "bins": fine.predicted_acquisition_bins.astype(int).tolist(),
                },
                "measured_epoch_line_anchor": {
                    "role": (
                        "narrow line-localized diagnostic used only when the "
                        "quarter's health-filtered evidence passes every "
                        "declared threshold"
                    ),
                    "epoch_definition": (
                        "retrospective provisional UTC calendar quarter at "
                        "frame start; not an authoritative station epoch"
                    ),
                    "estimator": (
                        "within each provisional epoch, divide every health-"
                        "included fine-F row by its row median; within the broad "
                        "geometry-predicted acquisition neighborhood, select "
                        "the bin with the largest "
                        "fraction of normalized values >= the line threshold, "
                        "breaking ties by the median normalized value and then "
                        "the lower bin index"
                    ),
                    "minimum_evidence": {
                        "usable_health_frames": MEASURED_FINE_ANCHOR_MIN_FRAMES,
                        "normalized_line_threshold": (
                            MEASURED_FINE_ANCHOR_NORMALIZED_LINE_THRESHOLD
                        ),
                        "persistence_fraction": (
                            MEASURED_FINE_ANCHOR_MIN_PERSISTENCE_FRACTION
                        ),
                        "persistence_margin_over_strongest_competitor": (
                            MEASURED_FINE_ANCHOR_MIN_COMPETITOR_MARGIN
                        ),
                        "within_acquisition_competitor_clear_half_width_bins": (
                            MEASURED_FINE_ANCHOR_COMPETITOR_CLEAR_HALF_WIDTH_BINS
                        ),
                        "boundary_external_sentinel_rule": (
                            "refuse a candidate within the accepted narrow "
                            "half-width of the acquisition edge when the "
                            "strongest outside sentinel has greater persistence"
                        ),
                    },
                    "accepted_window_half_width_bins": int(
                        fine.measured_line_half_width_bins
                    ),
                    "health_included_frames_using_measured_anchor": int(
                        np.count_nonzero(fine.measured_anchor_used_by_frame & include)
                    ),
                    "health_included_frames_falling_back_to_predicted": int(
                        np.count_nonzero(~fine.measured_anchor_used_by_frame & include)
                    ),
                    "epoch_status_counts": dict(
                        Counter(str(row["status"]) for row in fine.epoch_anchor_records)
                    ),
                    "epochs": [dict(row) for row in fine.epoch_anchor_records],
                },
                "cfar_null_bulk": {
                    "construction": (
                        "independent-bin bulk after excluding the broad predicted "
                        "acquisition neighborhood, every accepted measured-line "
                        "window, guard bins, and stored census exclusions"
                    ),
                    "bins": np.flatnonzero(fine.null_bulk_mask).astype(int).tolist(),
                    "count": int(np.count_nonzero(fine.null_bulk_mask)),
                },
                "stored_ancillary_designation": {
                    "bins": np.asarray(
                        geometry["stored_designated_bins"], dtype=int
                    ).tolist(),
                    "interpretation": (
                        "historical run metadata only; it is neither the broad "
                        "predicted acquisition neighborhood nor a measured "
                        "epoch line anchor"
                    ),
                    "differs_from_predicted_acquisition": bool(
                        geometry["stored_differs_from_predicted_acquisition"]
                    ),
                },
                "cfar_modes": {
                    "ordinary": CFAR_MODE_MEDIAN_LEFT,
                    "fallback": CFAR_MODE_QUANTILE_FALLBACK,
                },
            },
            "time_span": {
                "first_stored_utc": _finite_time_bound(times),
                "last_stored_utc": _finite_time_bound(times, latest=True),
                "first_included_utc": _finite_time_bound(times[include]),
                "last_included_utc": _finite_time_bound(
                    times[include], latest=True
                ),
            },
            "health_filtered_exposure": _time_exposure_summary(
                product, times, include
            ),
            "integrated_spectrum_health_correction": {
                "exact": spectra.exact,
                "unavailable_reason": spectra.unavailable_reason,
                "healthy_before_frame_count": spectra.healthy_before_count,
                "healthy_after_frame_count": spectra.healthy_after_count,
                "ceiling_frames_subtracted_before": spectra.ceiling_count_before,
                "ceiling_frames_subtracted_after": spectra.ceiling_count_after,
                "ceiling_dc_power_per_frame": spectra.ceiling_dc_power_per_frame,
                "parseval": {
                    "before_expected_sum": spectra.before_parseval_expected_sum,
                    "before_observed_sum": spectra.before_parseval_observed_sum,
                    "before_relative_error": spectra.before_parseval_relative_error,
                    "after_expected_sum": spectra.after_parseval_expected_sum,
                    "after_observed_sum": spectra.after_parseval_observed_sum,
                    "after_relative_error": spectra.after_parseval_relative_error,
                    "relative_tolerance": spectra.parseval_relative_tolerance,
                    "pass": spectra.parseval_pass,
                    "roundoff_interpretation": (
                        "The production spectrum uses complex64 FFT/absolute-square "
                        "arithmetic, so agreement with the frame-power Parseval "
                        "prediction is bounded rather than bit-exact."
                    ),
                },
                "corrected_bins": [0] if spectra.exact and spectra.ceiling_count_before else [],
                "limitation": SPECTRAL_LIMITATION,
            },
        }
        ledger = exclusion_ledger_entries(
            product,
            health,
            product_path=product_path,
            product_sha256=digest,
        )
    return summary, ledger, spectra


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    fd, temporary = create_temporary_sibling(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(dict(row)), sort_keys=True, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _archive_npz_hashes(path: Path) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    names: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            names.append(info.filename)
            if not info.filename.lower().endswith(".npz"):
                continue
            digest = hashlib.sha256()
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            base = Path(info.filename).name
            if base in hashes:
                raise ArchiveHealthError(
                    f"product archive contains duplicate NPZ basename {base!r}"
                )
            hashes[base] = digest.hexdigest()
    return hashes, names


def _git_package_source_identity(
    repository: Path,
    revision: str,
) -> tuple[str, str]:
    """Resolve a commit and reproduce its ``package_source_sha256`` digest."""

    repo = Path(repository).resolve()
    if not repo.is_dir():
        raise ArchiveHealthError(f"source repository is not a directory: {repo}")

    def git_text(*arguments: str) -> str:
        process = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode:
            detail = process.stderr.strip() or process.stdout.strip()
            raise ArchiveHealthError(
                f"git {' '.join(arguments)} failed for {repo}: {detail}"
            )
        return process.stdout

    resolved = git_text("rev-parse", "--verify", f"{revision}^{{commit}}").strip()
    names = git_text(
        "ls-tree",
        "-r",
        "--name-only",
        resolved,
        "--",
        "src/pilot_proxy",
    ).splitlines()
    paths = sorted(name for name in names if name.endswith(".py"))
    if not paths:
        raise ArchiveHealthError(
            f"commit {resolved} has no Python package under src/pilot_proxy"
        )
    digest = hashlib.sha256()
    for name in paths:
        relative = name.removeprefix("src/pilot_proxy/").encode("utf-8")
        process = subprocess.run(
            ["git", "-C", str(repo), "show", f"{resolved}:{name}"],
            check=False,
            capture_output=True,
        )
        if process.returncode:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise ArchiveHealthError(
                f"cannot read {name} from commit {resolved}: {detail}"
            )
        payload = process.stdout
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return resolved, digest.hexdigest()


def _zip_member_by_suffix(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ArchiveHealthError(
            f"expected one {suffix!r} member in {archive.filename}, found {len(matches)}"
        )
    return matches[0]


_OBSERVATION_SCOPE_CLASS = {
    "chime.event.baseband.raw": "triggered_event",
    "chime.scheduled.baseband.raw": "scheduled",
}


def _observation_class(scope: str) -> str:
    return _OBSERVATION_SCOPE_CLASS.get(str(scope), f"unmapped_scope:{scope}")


def _inventory_observation_class_exposure(
    inventory: Sequence[Mapping[str, Any]],
    inventory_by_key: Mapping[str, Mapping[str, Any]],
    product_paths: Sequence[Path],
    ledger: Sequence[Mapping[str, Any]],
    unprocessed_inventory: Sequence[str],
) -> dict[str, Any]:
    """Join inventory scope to processed units and health-filtered exposure."""

    classes = sorted(
        {_observation_class(str(row["scope"])) for row in inventory},
        key=lambda value: (
            {"triggered_event": 0, "scheduled": 1}.get(value, 2),
            value,
        ),
    )
    counts: dict[str, Counter[str]] = {value: Counter() for value in classes}
    events: dict[str, dict[str, set[tuple[str, str]]]] = {
        value: {
            "inventory": set(),
            "processed": set(),
            "stored": set(),
            "health": set(),
        }
        for value in classes
    }
    for row in inventory:
        scope = str(row["scope"])
        category = _observation_class(scope)
        counts[category]["inventory_units"] += 1
        counts[category]["discovered_catalogued_bytes"] += int(
            row.get("size_bytes", 0)
        )
        events[category]["inventory"].add((scope, str(row["event"])))
    for key in unprocessed_inventory:
        category = _observation_class(str(inventory_by_key[key]["scope"]))
        counts[category]["inventory_units_not_processed"] += 1

    excluded_by_product: dict[str, set[int]] = {}
    for row in ledger:
        excluded_by_product.setdefault(str(row["product_name"]), set()).add(
            int(row["frame_index"])
        )
    seen_product_names: set[str] = set()
    for product_path in product_paths:
        if product_path.name in seen_product_names:
            raise ArchiveHealthError(
                f"duplicate product basename in inventory join: {product_path.name}"
            )
        seen_product_names.add(product_path.name)
        with np.load(product_path, allow_pickle=False) as product:
            unit_keys = np.asarray(_array(product, "unit_order")).reshape(-1)
            unit_rows = [inventory_by_key[str(key)] for key in unit_keys]
            unit_classes = np.asarray(
                [_observation_class(str(row["scope"])) for row in unit_rows],
                dtype=str,
            )
            unit_events = [
                (str(row["scope"]), str(row["event"])) for row in unit_rows
            ]
            frame_unit = np.asarray(
                _array(product, "frame_unit_index"), dtype=np.int64
            ).reshape(-1)
            if np.any(frame_unit < 0) or np.any(frame_unit >= unit_keys.size):
                raise ArchiveHealthError(
                    f"{product_path.name}: frame_unit_index is outside unit_order"
                )
            include = np.ones(frame_unit.size, dtype=bool)
            excluded_indices = sorted(excluded_by_product.get(product_path.name, set()))
            if excluded_indices:
                excluded = np.asarray(excluded_indices, dtype=np.int64)
                if excluded[0] < 0 or excluded[-1] >= include.size:
                    raise ArchiveHealthError(
                        f"{product_path.name}: ledger frame index is out of range"
                    )
                include[excluded] = False
            durations = frame_duration_seconds(product)
            duration_available = np.isfinite(durations) & (durations > 0.0)
            frame_classes = unit_classes[frame_unit]
            units_with_stored = set(int(value) for value in np.unique(frame_unit))
            units_with_health = set(
                int(value) for value in np.unique(frame_unit[include])
            )
            for unit, category in enumerate(unit_classes.tolist()):
                counts[category]["processed_units"] += 1
                events[category]["processed"].add(unit_events[unit])
                if unit in units_with_stored:
                    counts[category]["units_with_stored_frames"] += 1
                    events[category]["stored"].add(unit_events[unit])
                if unit in units_with_health:
                    counts[category]["units_with_health_included_frames"] += 1
                    events[category]["health"].add(unit_events[unit])
            for category in classes:
                category_frames = frame_classes == category
                category_included = category_frames & include
                category_duration = category_frames & duration_available
                category_included_duration = category_included & duration_available
                counts[category]["stored_frames"] += int(
                    np.count_nonzero(category_frames)
                )
                counts[category]["health_included_frames"] += int(
                    np.count_nonzero(category_included)
                )
                counts[category]["stored_frames_with_duration"] += int(
                    np.count_nonzero(category_duration)
                )
                counts[category]["health_included_frames_with_duration"] += int(
                    np.count_nonzero(category_included_duration)
                )
                counts[category]["stored_exposure_nanoseconds"] += int(
                    round(float(np.sum(durations[category_duration])) * 1.0e9)
                )
                counts[category]["health_included_exposure_nanoseconds"] += int(
                    round(float(np.sum(durations[category_included_duration])) * 1.0e9)
                )

    rows: dict[str, Any] = {}
    duration_complete = True
    for category in classes:
        values = counts[category]
        stored_frames = int(values["stored_frames"])
        included_frames = int(values["health_included_frames"])
        stored_duration_frames = int(values["stored_frames_with_duration"])
        included_duration_frames = int(
            values["health_included_frames_with_duration"]
        )
        stored_ns = int(values["stored_exposure_nanoseconds"])
        included_ns = int(values["health_included_exposure_nanoseconds"])
        duration_complete = duration_complete and (
            stored_frames == stored_duration_frames
            and included_frames == included_duration_frames
        )
        rows[category] = {
            "inventory_units": int(values["inventory_units"]),
            "inventory_distinct_scope_events": len(events[category]["inventory"]),
            "discovered_catalogued_bytes": int(
                values["discovered_catalogued_bytes"]
            ),
            "processed_units": int(values["processed_units"]),
            "processed_distinct_scope_events": len(events[category]["processed"]),
            "inventory_units_not_processed": int(
                values["inventory_units_not_processed"]
            ),
            "units_with_stored_frames": int(values["units_with_stored_frames"]),
            "units_with_health_included_frames": int(
                values["units_with_health_included_frames"]
            ),
            "distinct_scope_events_with_stored_frames": len(
                events[category]["stored"]
            ),
            "distinct_scope_events_with_health_included_frames": len(
                events[category]["health"]
            ),
            "stored_frames": stored_frames,
            "health_included_frames": included_frames,
            "health_excluded_frames": stored_frames - included_frames,
            "stored_frames_with_duration": stored_duration_frames,
            "health_included_frames_with_duration": included_duration_frames,
            "stored_exposure_seconds": stored_ns / 1.0e9,
            "health_included_exposure_seconds": included_ns / 1.0e9,
            "health_excluded_exposure_seconds": (stored_ns - included_ns) / 1.0e9,
        }
    unmapped = sorted(
        str(row["scope"])
        for row in inventory
        if str(row["scope"]) not in _OBSERVATION_SCOPE_CLASS
    )
    return {
        "status": "available" if duration_complete else "partial",
        "classification_source": (
            "the inventory scope field, joined exactly to unit_order by "
            "common_path/name; this class is not present in the NPZ alone"
        ),
        "scope_to_class": dict(_OBSERVATION_SCOPE_CLASS),
        "unmapped_scopes": sorted(set(unmapped)),
        "frame_exposure_definition": (
            "nfft * unit_delta_time[frame_unit_index] for each rectangular frame"
        ),
        "duration_unavailable_reason": (
            None
            if duration_complete
            else "one or more frames lack a positive finite unit_delta_time"
        ),
        "classes": rows,
    }


def verify_supporting_evidence(
    product_summaries: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    *,
    product_paths: Sequence[Path] = (),
    product_archive: Path | None = None,
    inventory_archive: Path | None = None,
    kernel_library: Path | None = None,
    source_repository: Path | None = None,
    source_commits: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify optional attachments against product hashes and source-unit keys."""

    result: dict[str, Any] = {
        "product_archive": None,
        "inventory_archive": None,
        "kernel_library": None,
        "source_repository": None,
    }
    product_hashes = {
        str(row["product_name"]): str(row["product_sha256"])
        for row in product_summaries
    }
    if bool(source_repository) != bool(source_commits):
        raise ArchiveHealthError(
            "source-history verification requires both --source-repository "
            "and at least one --source-commit"
        )
    if source_repository is not None:
        source_counts = Counter(
            str(row["detector_version_tokens"].get("source", ""))
            for row in product_summaries
        )
        matches: list[dict[str, Any]] = []
        matched_hashes: set[str] = set()
        for revision in source_commits:
            resolved, digest = _git_package_source_identity(
                source_repository, str(revision)
            )
            if digest not in source_counts:
                raise ArchiveHealthError(
                    f"source commit {resolved} hashes to {digest}, which is "
                    "not recorded by any audited product"
                )
            matched_hashes.add(digest)
            matches.append(
                {
                    "requested_revision": str(revision),
                    "resolved_commit": resolved,
                    "package_source_sha256": digest,
                    "matching_products": int(source_counts[digest]),
                }
            )
        unmatched = sorted(set(source_counts) - matched_hashes)
        result["source_repository"] = {
            "repository_basename": Path(source_repository).resolve().name,
            "verified_commits": matches,
            "unmatched_recorded_source_hashes": unmatched,
            "status": "complete" if not unmatched else "partial",
            "interpretation": (
                "Each listed clean Git commit was rehashed with the product's "
                "package_source_sha256 algorithm. Unmatched recorded hashes "
                "remain build identities, but cannot be assigned to a clean "
                "commit from the supplied revisions."
            ),
        }
    if product_archive is not None:
        archive_path = Path(product_archive)
        member_hashes, names = _archive_npz_hashes(archive_path)
        missing = sorted(set(product_hashes) - set(member_hashes))
        extra = sorted(set(member_hashes) - set(product_hashes))
        mismatched = sorted(
            name
            for name in set(product_hashes) & set(member_hashes)
            if product_hashes[name] != member_hashes[name]
        )
        if missing or extra or mismatched:
            raise ArchiveHealthError(
                "product archive does not reproduce the audited NPZ set: "
                f"missing={missing}, extra={extra}, mismatched={mismatched}"
            )
        quarantine: list[dict[str, Any]] = []
        with zipfile.ZipFile(archive_path) as archive:
            quarantine_members = [
                name for name in archive.namelist() if name.endswith("quarantine.jsonl")
            ]
            if len(quarantine_members) > 1:
                raise ArchiveHealthError(
                    "product archive contains more than one quarantine.jsonl"
                )
            if quarantine_members:
                quarantine = [
                    json.loads(line)
                    for line in archive.read(quarantine_members[0])
                    .decode("utf-8")
                    .splitlines()
                    if line
                ]
        result["product_archive"] = {
            "basename": archive_path.name,
            "size_bytes": int(archive_path.stat().st_size),
            "sha256": file_sha256(archive_path),
            "npz_members": len(member_hashes),
            "matches_all_audited_products": True,
            "contains_raw_hdf5": any(
                name.lower().endswith((".h5", ".hdf5")) for name in names
            ),
            "contains_log_named_members": any("log" in name.lower() for name in names),
            "quarantine_rows": len(quarantine),
            "quarantine_unit_keys": sorted(
                str(row.get("key", "")) for row in quarantine
            ),
        }

    if kernel_library is not None:
        library_path = Path(kernel_library)
        actual = file_sha256(library_path)
        expected = sorted(
            {
                str(row["detector_version_tokens"].get("kernel_sha256", ""))
                for row in product_summaries
            }
        )
        matches = bool(actual and expected == [actual])
        if not matches:
            raise ArchiveHealthError(
                f"kernel library SHA-256 {actual!r} does not match product values {expected}"
            )
        result["kernel_library"] = {
            "basename": library_path.name,
            "size_bytes": int(library_path.stat().st_size),
            "sha256": actual,
            "recorded_kernel_versions": sorted(
                {
                    str(row["detector_version_tokens"].get("kernel", ""))
                    for row in product_summaries
                }
            ),
            "matches_every_product": True,
        }

    if inventory_archive is not None:
        inventory_path = Path(inventory_archive)
        with zipfile.ZipFile(inventory_path) as archive:
            meta_name = _zip_member_by_suffix(archive, "inventory.meta.json")
            inventory_name = _zip_member_by_suffix(archive, "inventory.jsonl")
            no_files_name = _zip_member_by_suffix(archive, "no_files_events.jsonl")
            surveyed_name = _zip_member_by_suffix(archive, "surveyed_events.txt")
            incomplete_name = _zip_member_by_suffix(archive, "incomplete_events.txt")
            attempts_name = _zip_member_by_suffix(archive, "attempts.json")
            enum_name = _zip_member_by_suffix(archive, "enum_cache.json")
            names = archive.namelist()
            meta = json.loads(archive.read(meta_name))
            inventory = [
                json.loads(line)
                for line in archive.read(inventory_name).decode("utf-8").splitlines()
                if line
            ]
            no_files = [
                json.loads(line)
                for line in archive.read(no_files_name).decode("utf-8").splitlines()
                if line
            ]
            surveyed = [
                line
                for line in archive.read(surveyed_name).decode("utf-8").splitlines()
                if line
            ]
            incomplete = [
                line
                for line in archive.read(incomplete_name).decode("utf-8").splitlines()
                if line
            ]
            attempts = json.loads(archive.read(attempts_name))
            enum_cache = json.loads(archive.read(enum_name))

        inventory_by_key = {
            f"{row['common_path']}/{row['name']}": row for row in inventory
        }
        if len(inventory_by_key) != len(inventory):
            raise ArchiveHealthError("inventory contains duplicate source unit keys")
        flagged_keys = sorted({str(row["unit_key"]) for row in ledger})
        missing_flagged = sorted(set(flagged_keys) - set(inventory_by_key))
        if missing_flagged:
            raise ArchiveHealthError(
                f"{len(missing_flagged)} exclusion-ledger unit(s) are absent from inventory"
            )
        product_unit_counts = Counter()
        for row in product_summaries:
            if "unit_count" in row:
                product_unit_counts[int(row["freq_id"])] = int(row["unit_count"])
        inventory_counts = Counter(int(row["freq_id"]) for row in inventory)
        processed_unit_keys: set[str] = set()
        for product_path in product_paths:
            with np.load(product_path, allow_pickle=False) as product:
                processed_unit_keys.update(
                    str(value)
                    for value in np.asarray(_array(product, "unit_order")).reshape(-1)
                )
        missing_processed = sorted(processed_unit_keys - set(inventory_by_key))
        if missing_processed:
            raise ArchiveHealthError(
                f"{len(missing_processed)} processed unit(s) are absent from inventory"
            )
        unprocessed_inventory = sorted(set(inventory_by_key) - processed_unit_keys)
        quarantine_keys = set(
            (result.get("product_archive") or {}).get("quarantine_unit_keys", [])
        )
        quarantine_matches = bool(
            quarantine_keys and quarantine_keys == set(unprocessed_inventory)
        )
        if quarantine_keys and not quarantine_matches:
            raise ArchiveHealthError(
                "inventory-minus-product unit set does not equal the quarantine ledger"
            )
        processed_scope_events = {
            (str(inventory_by_key[key]["scope"]), str(inventory_by_key[key]["event"]))
            for key in processed_unit_keys
        }
        inventory_scope_events = {
            (str(row["scope"]), str(row["event"])) for row in inventory
        }
        no_file_scope_events = {
            (str(row["scope"]), str(row["event"])) for row in no_files
        }
        surveyed_scope_events = {
            tuple(line.rsplit("|", 1)) for line in surveyed
        }
        attempt_scope_events = {
            tuple(str(key).rsplit("|", 1)) for key in attempts
        }
        enum_scope_events = {
            tuple(str(key).rsplit("|", 1)) for key in enum_cache
        }
        outrigger_excluded = {
            tuple(str(key).rsplit("|", 1))
            for key, labels in enum_cache.items()
            if any("outrigger" in str(label).lower() for label in labels)
        }
        survey_scope_events = enum_scope_events - outrigger_excluded
        derived_no_common_path = (
            surveyed_scope_events - inventory_scope_events - no_file_scope_events
        )
        completed_partition_exact = bool(
            surveyed_scope_events
            == inventory_scope_events | no_file_scope_events | derived_no_common_path
            and not (inventory_scope_events & no_file_scope_events)
            and not (inventory_scope_events & derived_no_common_path)
            and not (no_file_scope_events & derived_no_common_path)
        )
        survey_accounting_exact = bool(
            enum_scope_events == survey_scope_events | outrigger_excluded
            and not (survey_scope_events & outrigger_excluded)
            and survey_scope_events == surveyed_scope_events | attempt_scope_events
            and not (surveyed_scope_events & attempt_scope_events)
        )
        if not survey_accounting_exact:
            raise ArchiveHealthError(
                "inventory enumeration/completion/attempt ledgers do not form "
                "an exact survey-scope partition"
            )
        if not completed_partition_exact:
            raise ArchiveHealthError(
                "completed survey events do not partition exactly into "
                "inventory, accepted-empty, and derived no-common-path sets"
            )
        if not inventory_scope_events <= processed_scope_events:
            raise ArchiveHealthError(
                "one or more inventory events do not join the processed product axes"
            )
        observation_class_exposure = _inventory_observation_class_exposure(
            inventory,
            inventory_by_key,
            product_paths,
            ledger,
            unprocessed_inventory,
        )
        result["inventory_archive"] = {
            "basename": inventory_path.name,
            "size_bytes": int(inventory_path.stat().st_size),
            "sha256": file_sha256(inventory_path),
            "metadata": meta,
            "inventory_rows": len(inventory),
            "unique_inventory_unit_keys": len(inventory_by_key),
            "discovered_catalogued_bytes": int(
                sum(int(row.get("size_bytes", 0)) for row in inventory)
            ),
            "transferred_bytes": None,
            "byte_count_interpretation": (
                "discovered_catalogued_bytes is the sum of inventory object "
                "sizes.  No transfer/performance log is present, so it is not "
                "a measurement of network traffic, peak storage, or bytes read."
            ),
            "unique_scope_event_pairs": len(
                {(str(row["scope"]), str(row["event"])) for row in inventory}
            ),
            "surveyed_event_lines": len(surveyed),
            "surveyed_event_lines_unique": len(set(surveyed)),
            "incomplete_event_lines": len(incomplete),
            "no_files_event_rows": len(no_files),
            "no_files_reasons": dict(Counter(str(row["reason"]) for row in no_files)),
            "retried_scope_events": len(attempts),
            "retry_attempt_count_distribution": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(int(value) for value in attempts.values()).items()
                )
            },
            "enumerated_scope_events": len(enum_scope_events),
            "outrigger_labelled_events_excluded": len(outrigger_excluded),
            "outrigger_label_filter": (
                "case-insensitive label substring 'outrigger', matching the "
                "archive survey source's exclusion rule"
            ),
            "target_survey_scope_events": len(survey_scope_events),
            "completed_survey_scope_events": len(surveyed_scope_events),
            "pending_attempt_scope_events": len(attempt_scope_events),
            "survey_scope_accounting_exact": survey_accounting_exact,
            "completed_event_partition": {
                "with_target_freq_inventory_rows": len(inventory_scope_events),
                "aged_out_accepted_empty": len(no_file_scope_events),
                "derived_no_common_path": len(derived_no_common_path),
                "partition_exact": completed_partition_exact,
                "derived_status_caveat": (
                    "The no-common-path category is set subtraction from the "
                    "surviving enum, completed, inventory, and no-files "
                    "artifacts.  No explicit per-event status ledger for these "
                    "events survived, so the category is derived rather than "
                    "directly recorded."
                ),
            },
            "inventory_rows_by_freq_id": {
                str(key): int(value) for key, value in sorted(inventory_counts.items())
            },
            "processed_units_by_freq_id": {
                str(key): int(value) for key, value in sorted(product_unit_counts.items())
            },
            "inventory_minus_processed_by_freq_id": {
                str(key): int(inventory_counts[key] - product_unit_counts.get(key, 0))
                for key in sorted(inventory_counts)
            },
            "exclusion_ledger_units": len(flagged_keys),
            "all_exclusion_ledger_units_join_inventory": True,
            "processed_unit_keys": len(processed_unit_keys),
            "all_processed_units_join_inventory": not missing_processed,
            "inventory_events_joining_processed_products": len(
                inventory_scope_events & processed_scope_events
            ),
            "all_inventory_events_join_processed_products": (
                inventory_scope_events <= processed_scope_events
            ),
            "inventory_units_not_processed": len(unprocessed_inventory),
            "inventory_minus_products_equals_quarantine": quarantine_matches,
            "health_filtered_exposure_by_observation_class": (
                observation_class_exposure
            ),
            "quarantine_inventory_bytes": int(
                sum(
                    int(inventory_by_key[key].get("size_bytes", 0))
                    for key in unprocessed_inventory
                )
            ),
            "exclusion_ledger_source_bytes_referenced": int(
                sum(int(inventory_by_key[key].get("size_bytes", 0)) for key in flagged_keys)
            ),
            "contains_raw_hdf5": any(
                name.lower().endswith((".h5", ".hdf5")) for name in names
            ),
            "contains_log_named_members": any("log" in name.lower() for name in names),
            "operational_metrics": {
                "network_bytes_transferred": None,
                "peak_storage_bytes": None,
                "gpu_time_seconds": None,
                "wall_time_seconds": None,
                "throughput_bytes_per_second": None,
                "status": (
                    "unavailable because no transfer/performance log was "
                    "supplied; these metrics are not required for the science "
                    "frame-health conclusions"
                ),
            },
            "interpretation": (
                "This attachment is an inventory/retrieval manifest, not the "
                "raw HDF5 payload.  Its CADC unit keys locate every excluded "
                "frame's source file for optional root-cause follow-up."
            ),
        }
    return result


def _generation_timestamp() -> tuple[str, str]:
    """Return an ISO timestamp, honoring reproducible-build convention."""

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is None:
        return dt.datetime.now(dt.timezone.utc).isoformat(), "wall_clock"
    try:
        timestamp = int(source_date_epoch)
    except ValueError as exc:
        raise ArchiveHealthError(
            "SOURCE_DATE_EPOCH must be an integer number of Unix seconds"
        ) from exc
    try:
        created = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise ArchiveHealthError(
            f"SOURCE_DATE_EPOCH is outside the supported range: {timestamp}"
        ) from exc
    return created.isoformat(), "SOURCE_DATE_EPOCH"


def _audit_implementation_identity() -> dict[str, Any]:
    """Identify the exact generator source, including Git state when present."""

    identity: dict[str, Any] = {
        "package_source_sha256": package_source_sha256(),
        "archive_health_module_sha256": file_sha256(Path(__file__)),
        "git_commit": None,
        "git_tracked_worktree_clean": None,
        "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH"),
        "interpretation": (
            "The digests identify the exact PilotProxy Python tree and "
            "archive-health module used to produce this audit. A non-null "
            "Git commit is additional context; the source digest remains "
            "authoritative."
        ),
    }
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return identity
    if commit.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit.stdout.strip()):
        identity["git_commit"] = commit.stdout.strip()
    if status.returncode == 0:
        identity["git_tracked_worktree_clean"] = not bool(status.stdout.strip())
    return identity


def _aggregate_summary(
    channels: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    supporting_evidence: Mapping[str, Any],
    *,
    distinct_product_unit_events: int,
    distinct_events_with_stored_frames: int,
    distinct_events_with_health_included_frames: int,
    zero_frame_units: int,
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    total_frames = total_included = 0
    for row in channels:
        counts = row["frame_counts"]
        total_frames += int(counts["stored"])
        total_included += int(counts["included_by_health_gate"])
        reason_counts.update(
            {str(k): int(v) for k, v in counts["reason_counts"].items()}
        )
    versions = Counter(str(row["detector_version"]) for row in channels)
    source_hashes = Counter(
        str(row["detector_version_tokens"].get("source", "")) for row in channels
    )
    created_utc, created_utc_source = _generation_timestamp()
    source_evidence = supporting_evidence.get("source_repository")
    source_hash_interpretation = (
        "Clean Git revisions supplied to the audit were rehashed and matched "
        "where reported under supporting_evidence.source_repository; any "
        "remaining recorded source hash is explicitly unmatched."
        if source_evidence
        else (
            "These are hashes recorded inside the products. A matching "
            "source-tree revision was not supplied by the evidence inputs; "
            "kernel-binary verification is reported separately."
        )
    )
    temporal_statuses = Counter(
        str(row["health_filtered_exposure"]["status"]) for row in channels
    )
    temporal_status = (
        "available"
        if temporal_statuses == {"available": len(channels)}
        else "partial"
    )
    inventory_evidence = supporting_evidence.get("inventory_archive")
    observation_class_exposure = (
        inventory_evidence.get("health_filtered_exposure_by_observation_class")
        if isinstance(inventory_evidence, Mapping)
        else None
    )
    return {
        "schema_version": ARCHIVE_HEALTH_SUMMARY_SCHEMA_VERSION,
        "created_utc": created_utc,
        "created_utc_source": created_utc_source,
        "health_gate": {
            "schema_version": FRAME_HEALTH_GATE_SCHEMA_VERSION,
            "policy": "fail_closed",
            "reason_definitions": REASON_DEFINITIONS,
            "encoding_interpretation": ENCODING_INTERPRETATION,
        },
        "exposure_coordinates": {
            "status": temporal_status,
            "per_channel_field": "channels[].health_filtered_exposure",
            "channel_status_counts": dict(temporal_statuses),
            "utc_calendar_month": {
                "status": temporal_status,
                "field": (
                    "channels[].health_filtered_exposure.utc_calendar_month"
                ),
            },
            "utc_hour_of_day": {
                "status": temporal_status,
                "field": "channels[].health_filtered_exposure.utc_hour_of_day",
            },
            "local_civil_calendar_month_and_hour": {
                "status": temporal_status,
                "iana_time_zone": LOCAL_CIVIL_TIME_ZONE,
                "fields": [
                    "channels[].health_filtered_exposure."
                    "local_civil_calendar_month",
                    "channels[].health_filtered_exposure.local_civil_hour_of_day",
                ],
            },
            "local_meteorological_season": {
                "status": temporal_status,
                "iana_time_zone": LOCAL_CIVIL_TIME_ZONE,
                "definition": "DJF/MAM/JJA/SON from local civil month",
                "field": (
                    "channels[].health_filtered_exposure."
                    "local_meteorological_season"
                ),
            },
            "local_mean_sidereal_hour": {
                "status": temporal_status,
                "longitude_degrees_east": DRAO_LONGITUDE_DEGREES_EAST,
                "coordinate": "LMST, not LAST and not local civil/solar hour",
                "formula_version": LMST_FORMULA_VERSION,
                "formula_implementation_sha256": (
                    LMST_FORMULA_IMPLEMENTATION_SHA256
                ),
                "field": (
                    "channels[].health_filtered_exposure."
                    "local_mean_sidereal_hour"
                ),
            },
            "triggered_versus_scheduled_observation_class": (
                {
                    "status": str(observation_class_exposure["status"]),
                    "field": (
                        "provenance.supporting_evidence.inventory_archive."
                        "health_filtered_exposure_by_observation_class"
                    ),
                    "classification_source": str(
                        observation_class_exposure["classification_source"]
                    ),
                }
                if isinstance(observation_class_exposure, Mapping)
                else {
                    "status": "unavailable",
                    "field": None,
                    "unavailable_reason": (
                        "the NPZ products do not retain trigger/scheduled "
                        "class; supply the inventory archive so unit_order can "
                        "be joined to its scope field"
                    ),
                }
            ),
        },
        "totals": {
            "products": len(channels),
            "stored_frames": total_frames,
            "included_frames": total_included,
            "excluded_unique_frames": total_frames - total_included,
            "reason_counts": dict(reason_counts),
            "exclusion_ledger_rows": len(ledger),
            "product_units": int(sum(int(row["unit_count"]) for row in channels)),
            "zero_frame_units": int(zero_frame_units),
            "distinct_product_unit_events": int(distinct_product_unit_events),
            "distinct_events_with_stored_frames": int(
                distinct_events_with_stored_frames
            ),
            "distinct_events_with_health_included_frames": int(
                distinct_events_with_health_included_frames
            ),
            "health_exclusion_rate": proportion_summary(
                total_frames - total_included, total_frames
            ),
        },
        "provenance": {
            "audit_implementation": _audit_implementation_identity(),
            "detector_versions": dict(versions),
            "recorded_source_hashes": dict(source_hashes),
            "source_hash_interpretation": source_hash_interpretation,
            "supporting_evidence": supporting_evidence,
        },
        "channels": list(channels),
        "limitations": {
            "spectra": SPECTRAL_LIMITATION,
            "fine": (
                "The corrected fine results reuse stored float fine-F arrays. "
                "They repair the geometry-derived anchor/window/null bulk, but "
                "are retrospective diagnostics and are not a newly deployed "
                "or independently calibrated mask."
            ),
            "absolute_psd": (
                "The products do not contain a flux/temperature calibration; "
                "spectra are reported only as relative power."
            ),
            "raw_spectrogram": (
                "The products do not contain per-time/per-frequency voltage or "
                "power arrays.  The fine-F UTC heatmaps are detector-statistic "
                "diagnostics, not raw-voltage spectrograms."
            ),
        },
    }


def audit_archive_products(
    product_paths: Sequence[Path],
    *,
    fine_chunk_rows: int = 4096,
    product_archive: Path | None = None,
    inventory_archive: Path | None = None,
    kernel_library: Path | None = None,
    source_repository: Path | None = None,
    source_commits: Sequence[str] = (),
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[int, CorrectedSpectrumResult],
]:
    """Audit a set of products, rejecting duplicates and partial failures."""

    paths = [Path(path) for path in product_paths]
    if not paths:
        raise ArchiveHealthError("no per-pilot products were supplied")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ArchiveHealthError(f"missing product files: {missing}")
    channels: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    spectra_by_fid: dict[int, CorrectedSpectrumResult] = {}
    seen_fids: set[int] = set()
    seen_channels: set[int] = set()
    product_unit_events: set[int] = set()
    stored_frame_events: set[int] = set()
    health_included_events: set[int] = set()
    total_zero_frame_units = 0
    for path in paths:
        summary, entries, spectra = audit_product(
            path, fine_chunk_rows=fine_chunk_rows
        )
        fid = int(summary["freq_id"])
        channel = int(summary["physical_channel"])
        if fid in seen_fids or channel in seen_channels:
            raise ArchiveHealthError(
                f"duplicate freq_id or physical channel: fid={fid}, channel={channel}"
            )
        seen_fids.add(fid)
        seen_channels.add(channel)
        # Unit counts are part of the inventory coverage invariant.
        with np.load(path, allow_pickle=False) as product:
            unit_count = int(_array(product, "unit_order").size)
            summary["unit_count"] = unit_count
            unit_event_ids = np.asarray(
                _array(product, "unit_event_id"), dtype=np.int64
            ).reshape(-1)
            frame_unit_index = np.asarray(
                _array(product, "frame_unit_index"), dtype=np.int64
            ).reshape(-1)
            units_with_frames = np.unique(frame_unit_index)
            include = np.ones(frame_unit_index.size, dtype=bool)
            if entries:
                include[
                    np.asarray([int(row["frame_index"]) for row in entries], dtype=np.int64)
                ] = False
            units_with_health_frames = np.unique(frame_unit_index[include])
            zero_frame_units = unit_count - int(units_with_frames.size)
            total_zero_frame_units += zero_frame_units
            summary["unit_coverage"] = {
                "stored_units": unit_count,
                "units_with_stored_frames": int(units_with_frames.size),
                "zero_frame_units": zero_frame_units,
                "units_with_health_included_frames": int(
                    units_with_health_frames.size
                ),
            }
            product_unit_events.update(
                int(value) for value in unit_event_ids if int(value) >= 0
            )
            stored_frame_events.update(
                int(value)
                for value in unit_event_ids[units_with_frames]
                if int(value) >= 0
            )
            health_included_events.update(
                int(value)
                for value in unit_event_ids[units_with_health_frames]
                if int(value) >= 0
            )
        channels.append(summary)
        ledger.extend(entries)
        spectra_by_fid[fid] = spectra
    channels.sort(key=lambda row: int(row["physical_channel"]))
    ledger.sort(
        key=lambda row: (
            int(row["physical_channel"]),
            int(row["frame_index"]),
            str(row["ledger_key"]),
        )
    )
    evidence = verify_supporting_evidence(
        channels,
        ledger,
        product_paths=paths,
        product_archive=product_archive,
        inventory_archive=inventory_archive,
        kernel_library=kernel_library,
        source_repository=source_repository,
        source_commits=source_commits,
    )
    return (
        _aggregate_summary(
            channels,
            ledger,
            evidence,
            distinct_product_unit_events=len(product_unit_events),
            distinct_events_with_stored_frames=len(stored_frame_events),
            distinct_events_with_health_included_frames=len(health_included_events),
            zero_frame_units=total_zero_frame_units,
        ),
        ledger,
        spectra_by_fid,
    )


def _write_corrected_spectra(
    path: Path,
    product_paths: Sequence[Path],
    spectra_by_fid: Mapping[int, CorrectedSpectrumResult],
) -> Path:
    rows: list[dict[str, Any]] = []
    for product_path in product_paths:
        with np.load(product_path, allow_pickle=False) as product:
            fid = int(_scalar(product, "freq_id"))
            rows.append(
                {
                    "fid": fid,
                    "channel": int(_scalar(product, "physical_channel")),
                    "sample_rate_hz": _sample_rate_hz(product),
                    "sense": int(_scalar(product, "sense")),
                    "nfft": int(_scalar(product, "nfft")),
                    "spectrum": spectra_by_fid[fid],
                }
            )
    rows.sort(key=lambda row: int(row["channel"]))
    if not all(row["spectrum"].exact for row in rows):
        bad = [row["fid"] for row in rows if not row["spectrum"].exact]
        raise ArchiveHealthError(
            f"refusing corrected-spectra publication; exact repair unavailable for {bad}"
        )
    nffts = {int(row["nfft"]) for row in rows}
    if len(nffts) != 1:
        raise ArchiveHealthError("corrected spectra cannot be stacked across mixed nfft")
    return atomic_savez_compressed(
        path,
        schema_version=np.asarray(CORRECTED_SPECTRA_SCHEMA_VERSION),
        physical_channel=np.asarray([row["channel"] for row in rows], dtype=np.int32),
        freq_id=np.asarray([row["fid"] for row in rows], dtype=np.int64),
        sample_rate_hz=np.asarray([row["sample_rate_hz"] for row in rows], dtype=np.float64),
        sense=np.asarray([row["sense"] for row in rows], dtype=np.int8),
        nfft=np.asarray(next(iter(nffts)), dtype=np.int64),
        integrated_spectrum_before_health_gate=np.stack(
            [row["spectrum"].before for row in rows]
        ).astype(np.float64),
        integrated_spectrum_after_mask_and_health_gate=np.stack(
            [row["spectrum"].after for row in rows]
        ).astype(np.float64),
        healthy_before_frame_count=np.asarray(
            [row["spectrum"].healthy_before_count for row in rows], dtype=np.int64
        ),
        healthy_after_frame_count=np.asarray(
            [row["spectrum"].healthy_after_count for row in rows], dtype=np.int64
        ),
        ceiling_frames_subtracted_before=np.asarray(
            [row["spectrum"].ceiling_count_before for row in rows], dtype=np.int64
        ),
        ceiling_frames_subtracted_after=np.asarray(
            [row["spectrum"].ceiling_count_after for row in rows], dtype=np.int64
        ),
        ceiling_dc_power_per_frame=np.asarray(
            [row["spectrum"].ceiling_dc_power_per_frame for row in rows],
            dtype=np.float64,
        ),
        before_parseval_expected_sum=np.asarray(
            [row["spectrum"].before_parseval_expected_sum for row in rows],
            dtype=np.float64,
        ),
        before_parseval_observed_sum=np.asarray(
            [row["spectrum"].before_parseval_observed_sum for row in rows],
            dtype=np.float64,
        ),
        before_parseval_relative_error=np.asarray(
            [row["spectrum"].before_parseval_relative_error for row in rows],
            dtype=np.float64,
        ),
        after_parseval_expected_sum=np.asarray(
            [row["spectrum"].after_parseval_expected_sum for row in rows],
            dtype=np.float64,
        ),
        after_parseval_observed_sum=np.asarray(
            [row["spectrum"].after_parseval_observed_sum for row in rows],
            dtype=np.float64,
        ),
        after_parseval_relative_error=np.asarray(
            [row["spectrum"].after_parseval_relative_error for row in rows],
            dtype=np.float64,
        ),
        parseval_relative_tolerance=np.asarray(
            [row["spectrum"].parseval_relative_tolerance for row in rows],
            dtype=np.float64,
        ),
        parseval_pass=np.asarray(
            [row["spectrum"].parseval_pass for row in rows], dtype=np.uint8
        ),
    )


def _pdf_metadata(title: str) -> dict[str, str | None]:
    """Return deterministic PDF metadata without wall-clock date fields."""

    return {
        "Title": str(title),
        "Author": "Dylan Gormley",
        "Subject": "PilotProxy archive-health dissertation diagnostic",
        "Creator": "PilotProxy with Matplotlib and Latin Modern via LaTeX",
        "CreationDate": None,
        "ModDate": None,
    }


def _month_codes(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    datetimes = np.asarray(times * 1.0e9, dtype="datetime64[ns]")
    months = datetimes.astype("datetime64[M]")
    finite = ~np.isnat(months)
    if not np.any(finite):
        raise ArchiveHealthError("no finite UTC times are available for a heatmap")
    lo, hi = months[finite].min(), months[finite].max()
    full = np.arange(lo, hi + np.timedelta64(1, "M"), dtype="datetime64[M]")
    lookup = {month: index for index, month in enumerate(full)}
    codes = np.full(months.size, -1, dtype=np.int64)
    for index in np.flatnonzero(finite):
        codes[index] = lookup[months[index]]
    labels = [str(month) for month in full]
    return codes, full, labels


def _plot_product_diagnostics(
    product_path: Path,
    spectra: CorrectedSpectrumResult,
    figure_root: Path,
    *,
    fine_chunk_rows: int,
    dissertation_style: bool,
) -> list[dict[str, Any]]:
    try:
        from pilot_proxy.plot_style import setup_matplotlib

        plt = setup_matplotlib(dissertation_style=dissertation_style)
    except ImportError as exc:
        raise ArchiveHealthError(
            "diagnostic plots require Matplotlib; install pilot-proxy[plot] "
            "or pass --no-plots"
        ) from exc

    outputs: list[dict[str, Any]] = []
    with np.load(product_path, allow_pickle=False) as product:
        health = evaluate_frame_health(product)
        fine_diag = recompute_corrected_fine_diagnostics(
            product, health, chunk_rows=fine_chunk_rows
        )
        n_frames = health.include.size
        fid = int(_scalar(product, "freq_id"))
        channel = int(_scalar(product, "physical_channel"))
        reject = _binary_frame_vector(product, "reject_mask", n_frames)
        mu0 = null_power_ratio_of(product)
        coarse = np.asarray(
            _frame_vector(product, "coarse_power_ratio", n_frames),
            dtype=np.float64,
        )
        baseband = np.asarray(
            _frame_vector(product, "baseband_power_linear", n_frames),
            dtype=np.float64,
        )
        times = frame_utc_seconds(product)
        fine = fine_power_ratio_of(product)
        include = health.include
        channel_dir = figure_root / f"channel_{channel:02d}_fid_{fid:04d}"
        channel_dir.mkdir(parents=True, exist_ok=True)

        # Precompute the three dissertation diagnostics once, then render both
        # stand-alone publication assets and a one-page per-channel atlas.
        panels = [
            (coarse[include] / mu0, r"Coarse $F/\mu_0$", 1.0),
            (
                _frame_vector(
                    product, "normalized_coarse_power_ratio_db", n_frames
                )[include],
                "Normalized coarse level [dB]",
                0.0,
            ),
            (baseband[include], "Mean baseband power [native int4 units]", None),
            (
                fine_diag.selected_epoch_window_peak[include]
                / fine_diag.threshold[include],
                r"Epoch-anchor/fallback $F_{\mathrm{peak}}/T_{\mathrm{CFAR}}$",
                1.0,
            ),
        ]
        if not spectra.exact:
            raise ArchiveHealthError(
                f"cannot plot a health-corrected spectrum for freq_id {fid}: "
                f"{spectra.unavailable_reason}"
            )
        nfft = int(_scalar(product, "nfft"))
        rate = _sample_rate_hz(product)
        raw_bin = np.arange(nfft)
        baseband_hz = np.where(raw_bin < nfft // 2, raw_bin, raw_bin - nfft) * rate / nfft
        sky_offset = (
            float(_scalar(product, "chime_frequency_hz"))
            + int(_scalar(product, "sense")) * baseband_hz
            - float(_scalar(product, "pilot_frequency_hz"))
        )
        order = np.argsort(sky_offset)
        before_mean = spectra.before / max(spectra.healthy_before_count, 1)
        after_mean = spectra.after / max(spectra.healthy_after_count, 1)
        reference_values = before_mean[np.isfinite(before_mean) & (before_mean > 0.0)]
        if reference_values.size == 0:
            raise ArchiveHealthError("health-corrected before spectrum has no positive bins")
        reference = float(np.median(reference_values))
        floor = reference * 1.0e-12
        before_db = 10.0 * np.log10(np.maximum(before_mean, floor) / reference)
        after_db = 10.0 * np.log10(np.maximum(after_mean, floor) / reference)
        codes, months, labels = _month_codes(times)
        monthly = np.full((months.size, fine.shape[1]), np.nan, dtype=np.float64)
        month_counts = np.zeros(months.size, dtype=np.int64)
        for month in range(months.size):
            select = include & (codes == month)
            month_counts[month] = int(np.count_nonzero(select))
            if month_counts[month]:
                monthly[month] = np.mean(fine[select], axis=0)
        freq = np.asarray(fine_diag.fine_frequency_hz)
        freq_order = np.argsort(freq)
        display = 10.0 * np.log10(np.maximum(monthly[:, freq_order], 1.0e-12)).T
        finite_display = display[np.isfinite(display)]
        vmin, vmax = np.percentile(finite_display, [1.0, 99.0])
        if not vmax > vmin:
            vmax = vmin + 1.0

        percent = r"\%" if dissertation_style else "%"

        def draw_histograms(axes: Sequence[Any]) -> None:
            for axis, (values, label, line) in zip(axes, panels):
                finite_values = np.asarray(values, dtype=np.float64)
                finite_values = finite_values[np.isfinite(finite_values)]
                axis.hist(finite_values, bins=80, color="#0072B2", alpha=0.82)
                if line is not None:
                    axis.axvline(line, color="#D55E00", linewidth=1.1)
                axis.set_xlabel(label)
                axis.set_ylabel("Healthy frames")
                axis.set_yscale("log")
                axis.grid(alpha=0.18)

        def draw_spectrum(axis: Any) -> None:
            axis.plot(
                sky_offset[order] / 1.0e3,
                before_db[order],
                linewidth=0.65,
                color="0.35",
                label="before stored mask, health corrected",
            )
            if spectra.healthy_after_count:
                axis.plot(
                    sky_offset[order] / 1.0e3,
                    after_db[order],
                    linewidth=0.65,
                    color="#0072B2",
                    label="after stored mask, health corrected",
                )
            else:
                axis.text(
                    0.02,
                    0.06,
                    "No health-included frame survived the stored mask.\n"
                    "The after-spectrum is unavailable (zero integrated sum) "
                    "and is not plotted.",
                    transform=axis.transAxes,
                    fontsize=8,
                    color="#0072B2",
                    bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
                )
            axis.axvline(
                0.0,
                color="#D55E00",
                linestyle="--",
                linewidth=0.9,
                label="nominal pilot",
            )
            axis.set_xlabel("RF-frequency offset from nominal pilot [kHz]")
            axis.set_ylabel("Mean power [dB relative to before-spectrum median]")
            axis.grid(alpha=0.18)
            axis.legend(fontsize=8)

        def draw_heatmap(figure: Any, axis: Any) -> None:
            image = axis.imshow(
                display,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=float(vmin),
                vmax=float(vmax),
                extent=(0, months.size, freq[freq_order][0], freq[freq_order][-1]),
                cmap="viridis",
            )
            anchor_hz = float(freq[fine_diag.predicted_anchor_bin])
            axis.axhline(
                anchor_hz,
                color="white",
                linestyle="--",
                linewidth=0.9,
                label="geometry-predicted acquisition anchor",
            )
            measured_label_used = False
            for record in fine_diag.epoch_anchor_records:
                if record["status"] != "measured_narrow_line_anchor":
                    continue
                epoch_key = str(record["epoch_key"])
                year = int(epoch_key[:4])
                quarter = int(epoch_key[-1])
                epoch_months = {
                    f"{year:04d}-{month:02d}"
                    for month in range(3 * quarter - 2, 3 * quarter + 1)
                }
                indices = [
                    index for index, label in enumerate(labels) if label in epoch_months
                ]
                if not indices:
                    continue
                measured_hz = float(freq[int(record["selected_anchor_bin"])])
                axis.plot(
                    [min(indices), max(indices) + 1],
                    [measured_hz, measured_hz],
                    color="#D55E00",
                    linewidth=1.25,
                    label=(
                        "accepted measured epoch anchor"
                        if not measured_label_used
                        else None
                    ),
                )
                measured_label_used = True
            tick_step = max(1, int(math.ceil(months.size / 9)))
            tick_index = np.arange(0, months.size, tick_step)
            axis.set_xticks(
                tick_index + 0.5,
                [labels[index] for index in tick_index],
                rotation=35,
                ha="right",
            )
            axis.set_xlabel("UTC calendar month (gaps retained)")
            axis.set_ylabel("Fine-envelope frequency [Hz]")
            axis.legend(fontsize=8, loc="upper right")
            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label(r"$10\log_{10} F_{\mathrm{fine}}$")

        # Health-filtered scalar histograms. PDF preserves vector text and
        # geometry for direct inclusion in the dissertation.
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
        draw_histograms(list(axes.flat))
        fig.suptitle(
            f"Channel {channel} (frequency ID {fid}): "
            "health-filtered scalar distributions\n"
            f"N={int(np.count_nonzero(include)):,}; stored coarse mask rate "
            f"{100*np.mean(reject[include]):.2f}{percent}"
        )
        fig.tight_layout()
        hist_path = channel_dir / "health_filtered_histograms.pdf"
        fig.savefig(
            hist_path,
            format="pdf",
            bbox_inches="tight",
            metadata=_pdf_metadata(
                f"Channel {channel} health-filtered scalar distributions"
            ),
        )
        plt.close(fig)
        outputs.append(
            {
                "path": str(hist_path),
                "kind": "health_filtered_scalar_histograms",
                "physical_channel": channel,
                "freq_id": fid,
                "format": "PDF",
                "rendering": "vector",
            }
        )

        # Exactly health-corrected relative time-averaged spectra. This is a
        # relative spectrum, not an absolutely calibrated PSD.
        fig, axis = plt.subplots(figsize=(10.2, 4.5))
        draw_spectrum(axis)
        axis.set_title(
            f"Channel {channel} (frequency ID {fid}): "
            "relative time-averaged spectra\n"
            f"exact v1 DC subtraction: {spectra.ceiling_count_before} before, "
            f"{spectra.ceiling_count_after} after"
        )
        fig.tight_layout()
        spectrum_path = channel_dir / "relative_time_averaged_spectra.pdf"
        fig.savefig(
            spectrum_path,
            format="pdf",
            bbox_inches="tight",
            metadata=_pdf_metadata(
                f"Channel {channel} relative time-averaged spectra"
            ),
        )
        plt.close(fig)
        outputs.append(
            {
                "path": str(spectrum_path),
                "kind": "relative_time_averaged_health_corrected_spectra",
                "physical_channel": channel,
                "freq_id": fid,
                "format": "PDF",
                "rendering": "vector",
                "healthy_before_frame_count": spectra.healthy_before_count,
                "healthy_after_frame_count": spectra.healthy_after_count,
                "absolute_psd": False,
                "raw_voltage_spectrogram": False,
            }
        )

        # UTC calendar-month mean fine-F heatmap. This is explicitly a
        # detector-statistic diagnostic, not a raw-voltage spectrogram.
        fig, axis = plt.subplots(figsize=(11.0, 5.0))
        draw_heatmap(fig, axis)
        axis.set_title(
            f"Channel {channel} (frequency ID {fid}): monthly mean fine F\n"
            "health-filtered detector-statistic heatmap; not a raw-voltage spectrogram"
        )
        fig.tight_layout()
        heatmap_path = channel_dir / "fine_f_utc_monthly_heatmap.png"
        fig.savefig(heatmap_path, dpi=300, format="png", bbox_inches="tight")
        plt.close(fig)
        outputs.append(
            {
                "path": str(heatmap_path),
                "kind": "fine_f_utc_monthly_mean_heatmap",
                "physical_channel": channel,
                "freq_id": fid,
                "format": "PNG",
                "rendering": "raster",
                "dpi": 300,
                "time_bin": "UTC_calendar_month",
                "time_bin_reduction": "arithmetic_mean_of_health_included_frames",
                "months_with_frames": int(np.count_nonzero(month_counts)),
                "raw_voltage_spectrogram": False,
            }
        )

        # One deterministic dissertation-ready atlas per channel. The heatmap
        # remains a raster image embedded in the otherwise vector PDF.
        atlas = plt.figure(figsize=(12.0, 15.5))
        outer = atlas.add_gridspec(
            3,
            1,
            height_ratios=(1.45, 1.0, 1.05),
            hspace=0.42,
        )
        hist_grid = outer[0].subgridspec(2, 2, hspace=0.38, wspace=0.28)
        hist_axes = [
            atlas.add_subplot(hist_grid[row, column])
            for row in range(2)
            for column in range(2)
        ]
        spectrum_axis = atlas.add_subplot(outer[1])
        heatmap_axis = atlas.add_subplot(outer[2])
        draw_histograms(hist_axes)
        draw_spectrum(spectrum_axis)
        spectrum_axis.set_title(
            "Relative time-averaged spectra "
            f"({spectra.ceiling_count_before}/{spectra.ceiling_count_after} "
            "ceiling frames subtracted before/after)"
        )
        draw_heatmap(atlas, heatmap_axis)
        heatmap_axis.set_title(
            "Monthly mean fine F: health-filtered detector statistic, "
            "not a raw-voltage spectrogram"
        )
        atlas.suptitle(
            f"Channel {channel} (frequency ID {fid}) diagnostic atlas: "
            f"{int(np.count_nonzero(include)):,} health-included frames",
            y=0.995,
        )
        atlas.subplots_adjust(top=0.95, bottom=0.055, left=0.08, right=0.92)
        atlas_path = channel_dir / f"channel_{channel:02d}_diagnostic_atlas.pdf"
        atlas.savefig(
            atlas_path,
            format="pdf",
            bbox_inches="tight",
            metadata=_pdf_metadata(f"Channel {channel} diagnostic atlas"),
        )
        plt.close(atlas)
        outputs.append(
            {
                "path": str(atlas_path),
                "kind": "per_channel_dissertation_diagnostic_atlas",
                "physical_channel": channel,
                "freq_id": fid,
                "format": "PDF",
                "rendering": "vector_with_embedded_heatmap_raster",
                "healthy_before_frame_count": spectra.healthy_before_count,
                "healthy_after_frame_count": spectra.healthy_after_count,
                "sections": [
                    "health_filtered_scalar_histograms",
                    "relative_time_averaged_health_corrected_spectra",
                    "fine_f_utc_monthly_mean_heatmap",
                ],
                "absolute_psd": False,
                "raw_voltage_spectrogram": False,
            }
        )
    return outputs


def write_archive_audit(
    product_paths: Sequence[Path],
    output_dir: Path,
    *,
    fine_chunk_rows: int = 4096,
    make_plots: bool = True,
    dissertation_style: bool = False,
    product_archive: Path | None = None,
    inventory_archive: Path | None = None,
    kernel_library: Path | None = None,
    source_repository: Path | None = None,
    source_commits: Sequence[str] = (),
    expect_products: int | None = None,
    expect_excluded_frames: int | None = None,
    expect_invalid_frames: int | None = None,
    expect_ceiling_frames: int | None = None,
) -> dict[str, Path]:
    """Run the complete audit and publish versioned machine-readable outputs."""

    paths = [Path(path) for path in product_paths]
    summary, ledger, spectra_by_fid = audit_archive_products(
        paths,
        fine_chunk_rows=fine_chunk_rows,
        product_archive=product_archive,
        inventory_archive=inventory_archive,
        kernel_library=kernel_library,
        source_repository=source_repository,
        source_commits=source_commits,
    )
    _assert_expected(
        summary,
        expect_products=expect_products,
        expect_excluded_frames=expect_excluded_frames,
        expect_invalid_frames=expect_invalid_frames,
        expect_ceiling_frames=expect_ceiling_frames,
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "archive_health_summary.json"
    ledger_path = destination / "archive_exclusion_ledger.jsonl"
    spectra_path = destination / "health_corrected_integrated_spectra.npz"
    atomic_write_json(summary_path, json_safe(summary), indent=2, sort_keys=True)
    _write_jsonl(ledger_path, ledger)
    _write_corrected_spectra(spectra_path, paths, spectra_by_fid)

    manifest: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_MANIFEST_SCHEMA_VERSION,
        "path_semantics": "release_root_relative_posix",
        "summary": summary_path.relative_to(destination).as_posix(),
        "exclusion_ledger": ledger_path.relative_to(destination).as_posix(),
        "corrected_spectra": spectra_path.relative_to(destination).as_posix(),
        "generator": summary["provenance"]["audit_implementation"],
        "figures": [],
        "figure_style": {
            "name": (
                "pilotproxy_dissertation_latin_modern_t1_v1"
                if dissertation_style
                else "pilotproxy_standard_v1"
            ),
            "latex_required": bool(dissertation_style),
            "pdf_date_metadata": "suppressed",
        },
        "figure_semantics": {
            "histograms": "health-filtered scalar distributions",
            "spectra": (
                "relative time averages from the exactly v1-corrected "
                "integrated spectra; not absolute calibrated PSDs"
            ),
            "heatmaps": (
                "UTC-month means of stored fine-F detector statistics; not "
                "raw-voltage spectrograms"
            ),
        },
        "limitations": SPECTRAL_LIMITATION,
    }
    if make_plots:
        figure_root = destination / "figures"
        for product_path in paths:
            with np.load(product_path, allow_pickle=False) as product:
                fid = int(_scalar(product, "freq_id"))
            manifest["figures"].extend(
                _plot_product_diagnostics(
                    product_path,
                    spectra_by_fid[fid],
                    figure_root,
                    fine_chunk_rows=fine_chunk_rows,
                    dissertation_style=dissertation_style,
                )
            )
        manifest["figures"].sort(
            key=lambda row: (int(row["physical_channel"]), str(row["kind"]))
        )
        for row in manifest["figures"]:
            absolute_path = Path(row["path"])
            row["sha256"] = file_sha256(absolute_path)
            relative_path = absolute_path.relative_to(destination).as_posix()
            row["path"] = relative_path
            row["relative_path"] = relative_path
    manifest_path = destination / "diagnostic_manifest.json"
    atomic_write_json(manifest_path, json_safe(manifest), indent=2, sort_keys=True)
    return {
        "summary": summary_path,
        "ledger": ledger_path,
        "corrected_spectra": spectra_path,
        "manifest": manifest_path,
    }


def _assert_expected(
    summary: Mapping[str, Any],
    *,
    expect_products: int | None = None,
    expect_excluded_frames: int | None = None,
    expect_invalid_frames: int | None = None,
    expect_ceiling_frames: int | None = None,
) -> None:
    totals = summary["totals"]
    checks = {
        "products": expect_products,
        "excluded_unique_frames": expect_excluded_frames,
    }
    reasons = totals["reason_counts"]
    if expect_invalid_frames is not None:
        checks[f"reason_counts.{REASON_DETECTOR_INVALID}"] = expect_invalid_frames
    if expect_ceiling_frames is not None:
        checks[f"reason_counts.{REASON_BASEBAND_CEILING}"] = expect_ceiling_frames
    for field, expected in checks.items():
        if expected is None:
            continue
        if field.startswith("reason_counts."):
            actual = int(reasons.get(field.split(".", 1)[1], 0))
        else:
            actual = int(totals[field])
        if actual != int(expected):
            raise ArchiveHealthError(
                f"expected {field}={int(expected)}, observed {actual}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the versioned fail-closed health gate to archived per-pilot "
            "NPZ products, repair the fine diagnostic geometry, and write "
            "machine-readable evidence plus supported plots."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--products-dir", type=Path)
    source.add_argument("--product", type=Path, action="append", dest="products")
    parser.add_argument("--glob", default="*.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fine-chunk-rows", type=int, default=4096)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--dissertation-style",
        action="store_true",
        help=(
            "Fail closed unless LaTeX can embed Latin Modern/T1 fonts in the "
            "PDF diagnostics. Use this for dissertation release assets."
        ),
    )
    parser.add_argument("--product-archive", type=Path, default=None)
    parser.add_argument("--inventory-archive", type=Path, default=None)
    parser.add_argument("--kernel-library", type=Path, default=None)
    parser.add_argument("--source-repository", type=Path, default=None)
    parser.add_argument(
        "--source-commit",
        action="append",
        dest="source_commits",
        default=[],
        help=(
            "Clean Git commit whose src/pilot_proxy tree must reproduce a "
            "source hash recorded by the products (repeatable)."
        ),
    )
    parser.add_argument("--expect-products", type=int, default=None)
    parser.add_argument("--expect-excluded-frames", type=int, default=None)
    parser.add_argument("--expect-invalid-frames", type=int, default=None)
    parser.add_argument("--expect-ceiling-frames", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = (
        sorted(Path(args.products_dir).glob(args.glob))
        if args.products_dir is not None
        else list(args.products or [])
    )
    outputs = write_archive_audit(
        paths,
        args.output_dir,
        fine_chunk_rows=args.fine_chunk_rows,
        make_plots=not args.no_plots,
        dissertation_style=args.dissertation_style,
        product_archive=args.product_archive,
        inventory_archive=args.inventory_archive,
        kernel_library=args.kernel_library,
        source_repository=args.source_repository,
        source_commits=args.source_commits,
        expect_products=args.expect_products,
        expect_excluded_frames=args.expect_excluded_frames,
        expect_invalid_frames=args.expect_invalid_frames,
        expect_ceiling_frames=args.expect_ceiling_frames,
    )
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    totals = summary["totals"]
    print(
        f"Audited {totals['products']} products / {totals['stored_frames']} frames: "
        f"included {totals['included_frames']}, excluded "
        f"{totals['excluded_unique_frames']}."
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_HEALTH_SUMMARY_SCHEMA_VERSION",
    "RESIDUAL_HEALTH_VIEW_SCHEMA_VERSION",
    "ArchiveHealthError",
    "CORRECTED_SPECTRA_SCHEMA_VERSION",
    "DIAGNOSTIC_MANIFEST_SCHEMA_VERSION",
    "DRAO_LONGITUDE_DEGREES_EAST",
    "ENCODING_INTERPRETATION",
    "EXCLUSION_LEDGER_SCHEMA_VERSION",
    "FRAME_HEALTH_GATE_SCHEMA_VERSION",
    "LMST_FORMULA_IMPLEMENTATION_SHA256",
    "LMST_FORMULA_VERSION",
    "FineDiagnosticResult",
    "FrameHealthResult",
    "CorrectedSpectrumResult",
    "REASON_BASEBAND_CEILING",
    "REASON_DETECTOR_INVALID",
    "REASON_DETECTOR_POWERS_ALL_ZERO",
    "SPECTRAL_LIMITATION",
    "audit_archive_products",
    "audit_product",
    "corrected_fine_geometry",
    "distribution_summary",
    "evaluate_frame_health",
    "exclusion_ledger_entries",
    "frame_utc_seconds",
    "frame_duration_seconds",
    "health_correct_integrated_spectra",
    "proportion_summary",
    "recompute_corrected_fine_diagnostics",
    "temporary_residual_health_views",
    "unix_utc_to_lmst_hours",
    "verify_supporting_evidence",
    "write_residual_health_view",
    "write_archive_audit",
]

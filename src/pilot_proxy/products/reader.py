"""One per-pilot product: the validated contract view and lazy access to its fields.

Two layers, read in this order:

- :func:`open_product` validates a current (v5) product mapping against the
  product and detector contracts and exposes its residual decision
  coordinates as a :class:`ProductView`: the exact-integer coarse statistic
  ``Q = F / mu_0``, the survey flag, the in-band residual estimate (the
  estimated shelf level, dB) and the acquisition-unit coordinates. A product
  that does not satisfy the contract raises :class:`ProductContractError`. A
  pre-v5 product that carries the legacy shelf field is read into the same
  coordinates with fewer checks; no release reads one, and the residual-chain
  tests use it to build small synthetic products.
- :class:`Product` keeps one ``.npz`` product open lazily (a product is up to
  1.5 GB and ``numpy.load`` decodes a member only when indexed). It exposes the
  small per-frame and per-unit fields as cached arrays, the view above, the
  frame selection the health gate leaves, and the band geometry (the nominal
  marker position on the fine axis and on the per-frame spectrum axis). The two
  large members, ``fine_power_u64`` and ``psd_frame_db_i16``, are read only in
  row chunks through ``fine_terms`` and ``psd_rows``.

Coordinates: every frequency reported outward is the RF offset from the
nominal marker, positive towards higher RF; the fine axis measures the carrier
modulo ``f_s / K`` about the coarse grid, so the nominal marker's fine bin (the
grid residual) is a band constant computed here and used to convert fine
offsets to RF.

The instrument and detector constants (sample rate, frame length, detector
window, fine bins) are read from the project profile.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Real
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from pilot_proxy.config.project import default_project

PRODUCT_SCHEMA_NAME = "pilotproxy_per_pilot_product"
PRODUCT_SCHEMA_REVISION = 5
PRODUCT_SCHEMA_TOKEN = "pilotproxy_per_pilot_product_v5"
SOURCE_EVENT_KEY_SCHEMA = "pilotproxy_namespaced_source_event_key_v1"

MASK_RULE = (
    "valid && (p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum)"
)
VALID_RULE = "p_ref_sum != 0"
FINE_MEASUREMENT_METHOD = "exact_fine_power_terms"
SCHEMA_TOKEN = "pilotproxy_per_pilot_product_v5"
HEALTH_GATE_SCHEMA = "pilotproxy_archive_frame_health_gate_v1"

_PROJECT = default_project()
_INSTRUMENT = _PROJECT.instrument
_DETECTOR = _PROJECT.detector_config
SAMPLE_RATE_HZ = _INSTRUMENT.sample_rate_hz                   # 390625 Hz
NFFT = _DETECTOR.nfft                                         # 16384
PSD_BIN_HZ = _DETECTOR.psd_bin_hz(_INSTRUMENT)                # 23.84185791 Hz
DETECTOR_WINDOW = _DETECTOR.detector_window                   # K of the campaign
COARSE_BIN_HZ = _DETECTOR.detector_bin_hz(_INSTRUMENT)        # 3051.7578125 Hz
FINE_BINS = _DETECTOR.fine_bins                               # L_F = 2 N / K
FINE_BIN_HZ = _DETECTOR.envelope_bin_hz(_INSTRUMENT)          # 11.92092896 Hz
FRAME_SECONDS = _DETECTOR.frame_seconds(_INSTRUMENT)          # 41.94 ms
PSD_DB_INVALID = -32768

FINE_TARGET, FINE_REF_LOWER, FINE_REF_UPPER = 0, 1, 2


class ProductContractError(ValueError):
    """Raised when a product cannot support the declared residual mapping."""


def _array(product: Mapping, name: str, dtype, shape) -> np.ndarray:
    if name not in product:
        raise ProductContractError(f"product is missing {name!r}")
    values = np.asarray(product[name])
    expected = np.dtype(dtype)
    if values.dtype != expected or values.shape != shape:
        raise ProductContractError(
            f"{name!r} must have dtype {expected} and shape {shape}; "
            f"got {values.dtype} and {values.shape}"
        )
    return values


def _string_scalar(product: Mapping, name: str) -> str:
    if name not in product:
        raise ProductContractError(f"product is missing {name!r}")
    values = np.asarray(product[name])
    if values.shape != () or values.dtype.kind not in {"U", "S"}:
        raise ProductContractError(f"{name!r} must be a string scalar")
    value = str(values.item())
    if not value:
        raise ProductContractError(f"{name!r} must not be empty")
    return value


def _string_vector(product: Mapping, name: str, size: int) -> tuple[str, ...]:
    if name not in product:
        raise ProductContractError(f"product is missing {name!r}")
    values = np.asarray(product[name])
    if values.shape != (size,) or values.dtype.kind not in {"U", "S"}:
        raise ProductContractError(
            f"{name!r} must be a string vector of length {size}"
        )
    return tuple(str(value) for value in values.tolist())


def _integer_scalar(product: Mapping, name: str, dtype=np.int64, *, minimum=None) -> int:
    values = _array(product, name, dtype, ())
    value = int(values.item())
    if minimum is not None and value < minimum:
        raise ProductContractError(f"{name!r} must be at least {minimum}")
    return value


def _float_scalar(product: Mapping, name: str, *, positive=False) -> float:
    values = _array(product, name, np.float64, ())
    value = float(values.item())
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ProductContractError(f"{name!r} must be {qualifier}")
    return value


def _json_scalar(product: Mapping, name: str) -> dict:
    raw = _string_scalar(product, name)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProductContractError(f"{name!r} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProductContractError(f"{name!r} must contain an object")
    return value


def _same_float_array(name: str, stored: np.ndarray,
                      derived: np.ndarray) -> None:
    if not np.allclose(stored, derived, rtol=4e-13, atol=1e-12,
                       equal_nan=True):
        finite = np.isfinite(stored) & np.isfinite(derived)
        maximum = (float(np.max(np.abs(stored[finite] - derived[finite])))
                   if finite.any() else float("nan"))
        raise ProductContractError(
            f"{name!r} disagrees with the exact power terms "
            f"(maximum absolute difference {maximum:g})"
        )


def _legacy_scalar(product: Mapping, name: str) -> float:
    if name not in product:
        raise ProductContractError(f"legacy product is missing {name!r}")
    values = np.asarray(product[name]).reshape(-1)
    if values.size != 1:
        raise ProductContractError(f"legacy field {name!r} must be scalar")
    value = float(values[0])
    if not math.isfinite(value):
        raise ProductContractError(f"legacy field {name!r} must be finite")
    return value


@dataclass(frozen=True)
class ProductView:
    """One residual coordinate system shared by current and legacy products."""

    schema: str
    physical_channel: int
    freq_id: int
    chime_frequency_hz: float
    valid: np.ndarray
    rejected: np.ndarray
    shelf_db: np.ndarray
    statistic: np.ndarray
    null_level: float
    shelf_offset_db: float
    frame_unit_index: np.ndarray
    unit_time0_ctime: np.ndarray
    normalized_excess: np.ndarray
    _p_target: np.ndarray | None = field(default=None, repr=False)
    _p_ref_sum: np.ndarray | None = field(default=None, repr=False)
    _target_norm_sq: int | None = field(default=None, repr=False)
    _reference_norm_sum_sq: int | None = field(default=None, repr=False)

    @property
    def is_current(self) -> bool:
        return self.schema == PRODUCT_SCHEMA_TOKEN

    def rejected_at_multiplier(self, eta: Real) -> np.ndarray:
        """Return the coarse decision at a positive threshold multiplier."""
        if isinstance(eta, bool) or not isinstance(eta, Real):
            raise TypeError("eta must be a number")
        value = float(eta)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("eta must be positive and finite")
        if not self.is_current:
            if (not math.isfinite(self.null_level)
                    or not np.isfinite(self.statistic).any()):
                raise ProductContractError(
                    "legacy product lacks fstat_raw/mu0 for rethresholding"
                )
            return self.valid & (self.statistic > value * self.null_level)

        eta_num, eta_den = value.as_integer_ratio()
        target = self._p_target
        reference = self._p_ref_sum
        target_norm = int(self._target_norm_sq)
        reference_norm = int(self._reference_norm_sum_sq)
        decisions = np.fromiter(
            (
                bool(is_valid)
                and int(num) * reference_norm * eta_den
                > target_norm * int(den) * eta_num
                for num, den, is_valid in zip(target, reference, self.valid)
            ),
            dtype=bool,
            count=self.valid.size,
        )
        return decisions


def _current_view(product: Mapping) -> ProductView:
    name = _string_scalar(product, "schema_name")
    revision = _integer_scalar(product, "schema_revision")
    token = _string_scalar(product, "schema_version")
    if (name != PRODUCT_SCHEMA_NAME or revision != PRODUCT_SCHEMA_REVISION
            or token != PRODUCT_SCHEMA_TOKEN):
        raise ProductContractError(
            f"unsupported product schema {name!r} revision {revision} "
            f"token {token!r}"
        )
    if (_string_scalar(product, "source_event_key_schema_version")
            != SOURCE_EVENT_KEY_SCHEMA):
        raise ProductContractError("source-event identity schema is unsupported")

    decision = _json_scalar(product, "decision_contract_json")
    active = decision.get("active_decision")
    fine = decision.get("fine_measurement")
    candidate = decision.get("fine_candidate_decision")
    if (not isinstance(active, dict)
            or active.get("method") != "coarse_normalized_positive_excess"
            or active.get("implementation") != "host_exact_integer_comparison"
            or active.get("output_field") != "reject_mask"):
        raise ProductContractError("active decision contract is incompatible")
    if (not isinstance(fine, dict)
            or fine.get("method") != FINE_MEASUREMENT_METHOD
            or fine.get("role") != "measurement_only_no_scan_time_decision"
            or fine.get("terms_field") != "fine_power_u64"
            or not isinstance(candidate, dict)
            or candidate.get("method") != "fine_order_statistic_cfar"
            or candidate.get("active") is not False):
        raise ProductContractError("fine measurement contract is incompatible")

    detector = _json_scalar(product, "detector_contract_json")
    if (detector.get("schema_version") != "pilotproxy_detector_contract_v1"
            or detector.get("num_weight_terms") != 3
            or detector.get("power_accumulator") != "uint64"
            or detector.get("power_accumulator_bits") != 64
            or detector.get("threshold_mode") != "none"
            or detector.get("per_frequency_threshold") is not False
            or detector.get("valid_rule") != VALID_RULE
            or detector.get("mask_rule") != MASK_RULE
            or detector.get("equivalent_mask_rule")
            != ("R_coarse > R_null; R_null = "
                "2*target_norm_sq/reference_norm_sum_sq")):
        raise ProductContractError("detector contract is incompatible")
    if _string_scalar(product, "mask_rule") != MASK_RULE:
        raise ProductContractError("product mask rule is incompatible")

    frame_index = np.asarray(product.get("frame_index"))
    if frame_index.dtype != np.dtype(np.int64) or frame_index.ndim != 1:
        raise ProductContractError("'frame_index' must be a one-dimensional int64 array")
    frame_count = int(frame_index.size)
    if frame_count == 0 or not np.array_equal(
            frame_index, np.arange(frame_count, dtype=np.int64)):
        raise ProductContractError("'frame_index' must be contiguous and zero-based")

    valid_u8 = _array(product, "valid", np.uint8, (frame_count, 1))
    rejected_u8 = _array(product, "reject_mask", np.uint8, (frame_count, 1))
    if np.any(valid_u8 > 1) or np.any(rejected_u8 > 1):
        raise ProductContractError("valid and reject flags must be zero or one")
    valid = valid_u8[:, 0].astype(bool)
    rejected = rejected_u8[:, 0].astype(bool)

    p_target = _array(product, "p_target_u64", np.uint64,
                      (frame_count, 1))[:, 0]
    p_ref_sum = _array(product, "p_ref_sum_u64", np.uint64,
                       (frame_count, 1))[:, 0]
    p_ref_lower = _array(product, "p_ref_lower_u64", np.uint64,
                         (frame_count, 1))[:, 0]
    p_ref_upper = _array(product, "p_ref_upper_u64", np.uint64,
                         (frame_count, 1))[:, 0]
    for row, (lower, upper, total) in enumerate(
            zip(p_ref_lower, p_ref_upper, p_ref_sum)):
        if int(lower) + int(upper) != int(total):
            raise ProductContractError(
                f"coarse reference terms disagree at frame {row}"
            )
    if not np.array_equal(valid, p_ref_sum != 0):
        raise ProductContractError("valid flags disagree with p_ref_sum != 0")

    target_norm = int(_array(
        product, "target_norm_sq", np.int64, (1,))[0])
    reference_norm = int(_array(
        product, "reference_norm_sum_sq", np.int64, (1,))[0])
    if target_norm <= 0 or reference_norm <= 0:
        raise ProductContractError("weight norms must be positive")
    exact_rejected = np.fromiter(
        (
            bool(is_valid)
            and int(num) * reference_norm > target_norm * int(den)
            for num, den, is_valid in zip(p_target, p_ref_sum, valid)
        ),
        dtype=bool,
        count=frame_count,
    )
    if not np.array_equal(rejected, exact_rejected):
        row = int(np.flatnonzero(rejected != exact_rejected)[0])
        raise ProductContractError(
            f"reject_mask disagrees with the exact decision at frame {row}"
        )

    ratio = np.full(frame_count, np.nan, dtype=np.float64)
    np.divide(
        p_target.astype(np.float64) * float(reference_norm),
        p_ref_sum.astype(np.float64) * float(target_norm),
        out=ratio,
        where=p_ref_sum > 0,
    )
    excess = ratio - 1.0
    coarse = np.full(frame_count, np.nan, dtype=np.float64)
    np.divide(2.0 * p_target.astype(np.float64),
              p_ref_sum.astype(np.float64), out=coarse,
              where=p_ref_sum > 0)
    ratio_db = np.full(frame_count, np.nan, dtype=np.float64)
    ratio_db[ratio > 0.0] = 10.0 * np.log10(ratio[ratio > 0.0])
    excess_db = np.full(frame_count, np.nan, dtype=np.float64)
    excess_db[excess > 0.0] = 10.0 * np.log10(excess[excess > 0.0])

    pilot_below = _float_scalar(product, "pilot_below_data_db")
    bin_enbw = _float_scalar(product, "bin_enbw_hz", positive=True)
    dtv_bandwidth = _float_scalar(product, "dtv_bandwidth_hz", positive=True)
    efficiency = _float_scalar(
        product, "pilot_capture_efficiency", positive=True)
    if pilot_below < 0.0 or efficiency > 1.0:
        raise ProductContractError("shelf calibration values are out of range")
    offset = (pilot_below - 10.0 * np.log10(dtv_bandwidth / bin_enbw)
              - 10.0 * np.log10(efficiency))
    shelf = excess_db + offset

    for field_name, derived in (
            ("coarse_power_ratio", coarse),
            ("normalized_coarse_power_ratio_db", ratio_db),
            ("normalized_pilot_excess", excess),
            ("pilot_excess_db", excess_db),
            ("estimated_data_shelf_snr_db", shelf)):
        stored = _array(product, field_name, np.float64,
                        (frame_count, 1))[:, 0]
        _same_float_array(field_name, stored, derived)

    if int(_array(product, "pilot_in_band", np.uint8, (1,))[0]) != 1:
        raise ProductContractError("pilot is not in the measured band")
    frame_unit = _array(product, "frame_unit_index", np.int32,
                        (frame_count,))
    unit_time0 = np.asarray(product.get("unit_time0_ctime"))
    if unit_time0.dtype != np.dtype(np.float64) or unit_time0.ndim != 1 \
            or unit_time0.size == 0 or not np.isfinite(unit_time0).all():
        raise ProductContractError(
            "'unit_time0_ctime' must be a finite float64 vector"
        )
    if np.any(frame_unit < 0) or np.any(frame_unit >= unit_time0.size):
        raise ProductContractError("frame unit indices are out of range")

    physical_channel = int(_array(
        product, "physical_channel", np.int32, (1,))[0])
    freq_id = int(_array(product, "freq_id", np.int64, (1,))[0])
    chime_frequency = float(_array(
        product, "chime_frequency_hz", np.float64, (1,))[0])
    if not math.isfinite(chime_frequency) or chime_frequency <= 0.0:
        raise ProductContractError("chime frequency must be positive and finite")

    return ProductView(
        schema=PRODUCT_SCHEMA_TOKEN,
        physical_channel=physical_channel,
        freq_id=freq_id,
        chime_frequency_hz=chime_frequency,
        valid=valid,
        rejected=rejected,
        shelf_db=shelf,
        statistic=ratio,
        null_level=1.0,
        shelf_offset_db=float(offset),
        frame_unit_index=frame_unit,
        unit_time0_ctime=unit_time0,
        normalized_excess=excess,
        _p_target=p_target,
        _p_ref_sum=p_ref_sum,
        _target_norm_sq=target_norm,
        _reference_norm_sum_sq=reference_norm,
    )


def _legacy_view(product: Mapping) -> ProductView:
    required = ("valid", "reject_mask", "snr_shelf_db",
                "physical_channel", "freq_id")
    missing = [name for name in required if name not in product]
    if missing:
        raise ProductContractError(
            "product is neither current v5 nor a supported legacy product; "
            "missing " + ", ".join(missing)
        )
    valid_values = np.asarray(product["valid"])
    if valid_values.ndim != 2 or valid_values.shape[1] != 1:
        raise ProductContractError("legacy valid array must have shape (N, 1)")
    frame_count = int(valid_values.shape[0])
    valid = valid_values[:, 0].astype(bool)
    rejected = np.asarray(product["reject_mask"])
    shelf = np.asarray(product["snr_shelf_db"])
    statistic = (np.asarray(product["fstat_raw"])
                 if "fstat_raw" in product else
                 np.full((frame_count, 1), np.nan, dtype=np.float64))
    for name, values in (("reject_mask", rejected),
                         ("snr_shelf_db", shelf),
                         ("fstat_raw", statistic)):
        if values.shape != (frame_count, 1):
            raise ProductContractError(
                f"legacy {name} array must have shape ({frame_count}, 1)"
            )
    null_level = (_legacy_scalar(product, "mu0")
                  if "mu0" in product else float("nan"))
    calibration = ("pilot_below_data_db", "dtv_bandwidth_hz", "bin_enbw_hz")
    if all(name in product for name in calibration):
        pilot_below = _legacy_scalar(product, "pilot_below_data_db")
        bandwidth = _legacy_scalar(product, "dtv_bandwidth_hz")
        bin_enbw = _legacy_scalar(product, "bin_enbw_hz")
        if bandwidth <= 0.0 or bin_enbw <= 0.0:
            raise ProductContractError("legacy bandwidths must be positive")
        offset = pilot_below - 10.0 * np.log10(bandwidth / bin_enbw)
    else:
        offset = float("nan")
    has_frame_unit = "frame_unit_index" in product
    has_unit_time = "unit_time0_ctime" in product
    if has_frame_unit != has_unit_time:
        raise ProductContractError("legacy unit coordinates are incomplete")
    if has_frame_unit:
        frame_unit = np.asarray(product["frame_unit_index"])
        unit_time0 = np.asarray(product["unit_time0_ctime"])
        if frame_unit.shape != (frame_count,) or unit_time0.ndim != 1:
            raise ProductContractError("legacy unit coordinates are malformed")
    else:
        frame_unit = np.empty(0, dtype=np.int32)
        unit_time0 = np.empty(0, dtype=np.float64)
    return ProductView(
        schema="legacy",
        physical_channel=int(np.asarray(product["physical_channel"]).reshape(-1)[0]),
        freq_id=int(np.asarray(product["freq_id"]).reshape(-1)[0]),
        chime_frequency_hz=(
            float(np.asarray(product["chime_frequency_hz"]).reshape(-1)[0])
            if "chime_frequency_hz" in product else float("nan")),
        valid=valid,
        rejected=rejected[:, 0].astype(bool),
        shelf_db=shelf[:, 0].astype(np.float64, copy=False),
        statistic=statistic[:, 0].astype(np.float64, copy=False),
        null_level=float(null_level),
        shelf_offset_db=float(offset),
        frame_unit_index=frame_unit,
        unit_time0_ctime=unit_time0,
        normalized_excess=statistic[:, 0].astype(np.float64, copy=False) - 1.0,
    )


_LEGACY_READING = False


@contextmanager
def legacy_reading():
    """Within the block, :func:`open_product` reads pre-v5 (legacy) products by default (tests only)."""
    global _LEGACY_READING
    before, _LEGACY_READING = _LEGACY_READING, True
    try:
        yield
    finally:
        _LEGACY_READING = before


def open_product(product: Mapping, *, allow_legacy: bool | None = None) -> ProductView:
    """Validate a product and expose its residual decision coordinates.

    A product that does not declare the current v5 schema is refused unless
    legacy reading is asked for (``allow_legacy=True``, or inside
    :func:`legacy_reading`), so a pre-v5 product is never read silently.
    """
    token = None
    revision = None
    if "schema_version" in product:
        values = np.asarray(product["schema_version"])
        if values.shape == () and values.dtype.kind in {"U", "S"}:
            token = str(values.item())
    if "schema_revision" in product:
        values = np.asarray(product["schema_revision"])
        if values.shape == () and np.issubdtype(values.dtype, np.integer):
            revision = int(values.item())
    if token == PRODUCT_SCHEMA_TOKEN or revision == PRODUCT_SCHEMA_REVISION:
        return _current_view(product)
    if not (_LEGACY_READING if allow_legacy is None else allow_legacy):
        raise ProductContractError(
            f"product does not declare the current schema {PRODUCT_SCHEMA_TOKEN!r} "
            f"(schema_version {token!r}, schema_revision {revision!r}); legacy products are read only on request"
        )
    return _legacy_view(product)


def is_current_product(product: Mapping) -> bool:
    """Return whether a product declares the current v5 schema."""
    if "schema_version" not in product:
        return False
    values = np.asarray(product["schema_version"])
    return bool(values.shape == () and values.dtype.kind in {"U", "S"}
                and str(values.item()) == PRODUCT_SCHEMA_TOKEN)


def coarse_reject_mask(product: Mapping, eta: Real = 1.0) -> np.ndarray:
    """Return a rethresholded coarse mask in the product's own coordinates."""
    revision = np.asarray(product.get("schema_revision"))
    declares_revision_five = bool(
        revision.shape == ()
        and np.issubdtype(revision.dtype, np.integer)
        and int(revision.item()) == PRODUCT_SCHEMA_REVISION
    )
    if is_current_product(product) or declares_revision_five:
        return _current_view(product).rejected_at_multiplier(eta)
    if "fstat_raw" not in product or "mu0" not in product:
        raise ProductContractError(
            "legacy product lacks fstat_raw/mu0 for rethresholding"
        )
    statistic = np.asarray(product["fstat_raw"])
    if statistic.ndim == 2 and statistic.shape[1] == 1:
        statistic = statistic[:, 0]
    elif statistic.ndim != 1:
        raise ProductContractError(
            "legacy fstat_raw must have shape (N,) or (N, 1)"
        )
    null = _legacy_scalar(product, "mu0")
    if isinstance(eta, bool) or not isinstance(eta, Real):
        raise TypeError("eta must be a number")
    value = float(eta)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("eta must be positive and finite")
    return statistic > value * null



def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fine_power_ratio(terms: np.ndarray) -> np.ndarray:
    """``T[f] = 2 S_0[f] / (S_1[f] + S_2[f])`` from exact terms ``(n, 3, B)``.

    The deployed fine statistic, formed the way ``pilot_proxy.fine_reduction``
    forms it: a zero denominator gives 0.0. Returned as float64 ``(n, B)``.
    """
    terms = np.asarray(terms)
    if terms.ndim != 3 or terms.shape[1] != 3:
        raise ValueError(f"fine terms must have shape (n, 3, B); got {terms.shape}")
    s_t = terms[:, FINE_TARGET].astype(np.float64)
    den = terms[:, FINE_REF_LOWER].astype(np.float64) + terms[:, FINE_REF_UPPER].astype(np.float64)
    positive = den > 0.0
    return np.where(positive, 2.0 * s_t / np.where(positive, den, 1.0), 0.0)


def grid_residual_hz(pilot_hz: float, centre_hz: float, sense: int,
                     coarse_bin_hz: float = COARSE_BIN_HZ) -> float:
    """Where an exactly-nominal pilot appears on the fine axis, in Hz.

    The receiver-frame offset ``sense * (pilot - centre)`` reduced to the
    nearest coarse-grid frequency: the fine spectrum's origin is that grid
    point, so a nominal pilot sits at this residual, not at zero.
    """
    eff = float(np.sign(sense) or 1) * (float(pilot_hz) - float(centre_hz))
    return eff - round(eff / coarse_bin_hz) * coarse_bin_hz


def fine_bin_of_hz(offset_hz: float, fine_bins: int = FINE_BINS, fine_bin_hz: float = FINE_BIN_HZ) -> int:
    """Padded fine bin (0..L_F-1) of a fine-axis offset in Hz."""
    return int(round(offset_hz / fine_bin_hz)) % fine_bins


def fine_hz_of_bin(bin_index, fine_bins: int = FINE_BINS, fine_bin_hz: float = FINE_BIN_HZ):
    """Fine-axis offset in Hz of a padded bin, centred convention ((b + L/2) mod L) - L/2."""
    b = np.asarray(bin_index)
    return (((b + fine_bins // 2) % fine_bins) - fine_bins // 2) * fine_bin_hz


def fine_offset_to_rf_hz(fine_offset_hz, grid_residual: float, sense: int) -> np.ndarray:
    """RF offset from the nominal pilot of a fine-axis offset.

    The fine offset is unwrapped about the grid residual (so an anchor a few
    bins above the residual is a small positive shift even when the wrap
    puts it at the other end of the axis), then the receiver sense is
    undone: ``rf = sign(sense) * (fine - residual)`` with ``sign(-1) = -1``.
    """
    fine = np.asarray(fine_offset_hz, dtype=float)
    delta = fine - grid_residual
    delta = (delta + COARSE_BIN_HZ / 2.0) % COARSE_BIN_HZ - COARSE_BIN_HZ / 2.0
    return float(np.sign(sense) or 1) * delta


@dataclass(frozen=True)
class Geometry:
    """Channel constants: where the nominal pilot is on each axis."""

    physical_channel: int
    freq_id: int
    pilot_hz: float
    centre_hz: float
    sense: int
    grid_residual_hz: float          # fine-axis offset of the nominal pilot (fine-array direction)
    nominal_fine_bin: int            # its padded fine bin (what fine_designated_bins is centred on)
    nominal_psd_bin: float           # its (real-valued) receiver-frame bin on the per-frame spectrum axis
    centre_line_rf_offset_hz: float  # the instrumental channel-centre line, RF offset from nominal
    stored_window_centre: int        # centre of the product's fine_designated_bins (the scan's prediction)

    @property
    def matches_stored_window(self) -> bool:
        """Whether the scan's designated window was centred on the same nominal bin."""
        return self.stored_window_centre == self.nominal_fine_bin

    def psd_rf_offset_hz(self, bins) -> np.ndarray:
        """RF offset from the nominal pilot of per-frame spectrum bins (FFT order)."""
        b = np.asarray(bins, dtype=float)
        receiver = ((b + NFFT // 2) % NFFT - NFFT // 2) * PSD_BIN_HZ
        return float(np.sign(self.sense) or 1) * receiver + (self.centre_hz - self.pilot_hz)

    def psd_bin_of_rf_offset(self, rf_offset_hz) -> np.ndarray:
        """Receiver-frame bin (real-valued, FFT order) of an RF offset from the nominal pilot."""
        receiver = (np.asarray(rf_offset_hz, dtype=float) - (self.centre_hz - self.pilot_hz)) / float(np.sign(self.sense) or 1)
        return (receiver / PSD_BIN_HZ) % NFFT


class Product:
    """One open v5 product; nothing large is read until asked for."""

    def __init__(self, path: Path | str, *, require_health: bool = False):
        self.path = Path(path)
        self.require_health = require_health
        self._z = np.load(self.path, allow_pickle=False)
        token = str(np.asarray(self._z["schema_version"]).item()) if "schema_version" in self._z.files else ""
        if token != SCHEMA_TOKEN:
            raise ValueError(f"{self.path.name}: schema {token!r}, expected {SCHEMA_TOKEN!r}")

    # ------------------------------------------------------------ raw access
    @property
    def archive(self):
        return self._z

    def scalar(self, name: str):
        return np.asarray(self._z[name]).reshape(-1)[0] if np.asarray(self._z[name]).ndim else np.asarray(self._z[name]).item()

    def frame_column(self, name: str) -> np.ndarray:
        """A per-frame field, shaped (N,)."""
        arr = np.asarray(self._z[name])
        return arr[:, 0] if arr.ndim == 2 and arr.shape[1] == 1 else arr.reshape(arr.shape[0], -1)[:, 0] if arr.ndim == 2 else arr

    # -------------------------------------------------------- small fields
    @cached_property
    def view(self) -> ProductView:
        """The exact-integer residual view (validates the v5 contracts)."""
        return open_product(self._z)

    @cached_property
    def n_frames(self) -> int:
        return int(self.view.valid.size)

    @cached_property
    def valid(self) -> np.ndarray:
        return self.view.valid

    @cached_property
    def rejected(self) -> np.ndarray:
        """The stored survey flag ``F > mu_0`` (eta = 1), re-derived exactly by the view."""
        return self.view.rejected

    @cached_property
    def statistic(self) -> np.ndarray:
        """``Q = F / mu_0`` per frame (NaN where invalid)."""
        return self.view.statistic

    @cached_property
    def level_db(self) -> np.ndarray:
        """``10 log10(F / mu_0)`` per frame (NaN where invalid or nonpositive)."""
        q = self.statistic
        out = np.full(q.shape, np.nan)
        ok = np.isfinite(q) & (q > 0)
        out[ok] = 10.0 * np.log10(q[ok])
        return out

    @cached_property
    def shelf_db(self) -> np.ndarray:
        """Estimated data-shelf SNR (finite only where the normalized excess is positive)."""
        return self.view.shelf_db

    @cached_property
    def frame_unit_index(self) -> np.ndarray:
        return np.asarray(self.view.frame_unit_index, dtype=np.int64)

    @cached_property
    def frame_in_unit(self) -> np.ndarray:
        return np.asarray(self._z["frame_in_unit"], dtype=np.int64).reshape(-1)

    @cached_property
    def unit_time0(self) -> np.ndarray:
        """Unix UTC seconds of each acquisition unit's first sample."""
        return np.asarray(self.view.unit_time0_ctime, dtype=np.float64)

    @cached_property
    def unit_delta_time(self) -> np.ndarray:
        return np.asarray(self._z["unit_delta_time"], dtype=np.float64).reshape(-1)

    @cached_property
    def unit_event_id(self) -> np.ndarray:
        return np.asarray(self._z["unit_event_id"]).reshape(-1)

    @cached_property
    def unit_input_map_sha256(self) -> np.ndarray:
        return np.asarray(self._z["unit_input_map_sha256"]).reshape(-1)

    @cached_property
    def unit_git_version_tag(self) -> np.ndarray:
        return np.asarray(self._z["unit_git_version_tag"]).reshape(-1)

    @cached_property
    def frame_time(self) -> np.ndarray:
        """Unix UTC seconds of each frame's first sample.

        ``unit_time0 + frame_in_unit * nfft * unit_delta_time``; NaN where the
        unit's sample interval is unrecorded (1-2% of units), so calendar
        statistics must state that denominator.
        """
        t0 = self.unit_time0[self.frame_unit_index]
        dt = self.unit_delta_time[self.frame_unit_index]
        return t0 + self.frame_in_unit * NFFT * dt

    @cached_property
    def unit_time(self) -> np.ndarray:
        """Unit time of each frame (its acquisition's first sample), never NaN."""
        return self.unit_time0[self.frame_unit_index]

    @cached_property
    def mu0(self) -> float:
        return 2.0 * int(self.scalar("target_norm_sq")) / int(self.scalar("reference_norm_sum_sq"))

    @cached_property
    def fine_designated_bins(self) -> np.ndarray:
        """The scan-time predicted window (nominal bin +- 30); not a measured anchor."""
        return np.asarray(self._z["fine_designated_bins"], dtype=np.int64).reshape(-1)

    @cached_property
    def fine_census_excluded_bins(self) -> np.ndarray:
        return np.asarray(self._z["fine_census_excluded_bins"], dtype=np.int64).reshape(-1)

    @cached_property
    def fine_pad_factor(self) -> int:
        return int(self.scalar("fine_pad_factor"))

    @cached_property
    def fine_guard_bins(self) -> int:
        return int(self.scalar("fine_guard_fine_bins"))

    # ------------------------------------------------------------ geometry
    @cached_property
    def geometry(self) -> Geometry:
        pilot = float(self.scalar("pilot_frequency_hz"))
        centre = float(self.scalar("chime_frequency_hz"))
        sense = int(self.scalar("sense"))
        residual = grid_residual_hz(pilot, centre, sense)
        nominal_bin = fine_bin_of_hz(residual)
        designated = self.fine_designated_bins
        stored_centre = int(designated[len(designated) // 2]) if designated.size else -1
        return Geometry(
            physical_channel=int(self.view.physical_channel),
            freq_id=int(self.view.freq_id),
            pilot_hz=pilot, centre_hz=centre, sense=sense,
            grid_residual_hz=residual, nominal_fine_bin=nominal_bin,
            # receiver-frame frequency of the pilot is sense * (pilot - centre)
            nominal_psd_bin=float((float(np.sign(sense) or 1) * (pilot - centre) / PSD_BIN_HZ) % NFFT),
            centre_line_rf_offset_hz=centre - pilot,
            stored_window_centre=stored_centre,
        )

    # -------------------------------------------------------- frame gating
    @cached_property
    def health(self) -> "HealthGate":
        return health_gate(self)

    @cached_property
    def selected(self) -> np.ndarray:
        """Frames every downstream step may use: valid and passing the health gate."""
        return self.valid & self.health.include

    # ---------------------------------------------------------- big members
    def fine_terms(self, rows) -> np.ndarray:
        """Exact fine terms for the given frame rows, shaped (len(rows), 3, 256)."""
        return np.asarray(self._z["fine_power_u64"])[np.asarray(rows)]

    def fine_terms_all(self) -> np.ndarray:
        """The whole fine-terms member (about 240 MB per channel); prefer ``fine_terms``."""
        return np.asarray(self._z["fine_power_u64"])

    def psd_rows(self, chunk: int = 2048) -> Iterator[tuple[slice, np.ndarray]]:
        """Decoded per-frame spectra in row chunks: yields (row slice, linear power (n, 16384)).

        Decoding follows the product contract: ``psd_db_reference *
        10 ** (code / 1000)`` with NaN at the invalid code. The int16 member is
        read once in full (it is a compressed zip member and cannot be sliced
        on disk); the decode is chunked so the float64 copy never exceeds the
        chunk.
        """
        codes = np.asarray(self._z["psd_frame_db_i16"])
        reference = self.frame_column("psd_db_reference").astype(np.float64)
        invalid = int(self.scalar("psd_db_invalid_code"))
        n = codes.shape[0]
        for start in range(0, n, int(chunk)):
            stop = min(n, start + int(chunk))
            block = codes[start:stop]
            power = reference[start:stop, None] * np.power(10.0, block.astype(np.float64) / 1000.0)
            power[block == invalid] = np.nan
            yield slice(start, stop), power

    def close(self) -> None:
        self._z.close()

    def __enter__(self) -> "Product":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass(frozen=True)
class HealthGate:
    """Which frames the frame-health gate admits, and which gate that was."""

    schema: str                  # HEALTH_GATE_SCHEMA
    include: np.ndarray          # bool (N,)
    reason_counts: dict[str, int]


def health_gate(product: Product) -> HealthGate:
    """Apply the v1 frame-health gate (:func:`pilot_proxy.archive_health.evaluate_frame_health`).

    The gate excludes 192 of the campaign's 770,478 frames (saturated-ceiling
    and detector-invalid frames). There is no valid-only substitute: the gate
    is part of the product's population.
    """
    from pilot_proxy.archive_health import evaluate_frame_health
    result = evaluate_frame_health(product.archive)
    include = np.asarray(result.include, dtype=bool)
    if include.shape != product.valid.shape:
        raise ValueError("health gate mask does not match the frame axis")
    return HealthGate(HEALTH_GATE_SCHEMA, include, {k: int(v) for k, v in dict(result.reason_counts).items()})

__all__ = [
    "COARSE_BIN_HZ", "DETECTOR_WINDOW", "FINE_BINS", "FINE_BIN_HZ", "FINE_MEASUREMENT_METHOD",
    "FINE_REF_LOWER", "FINE_REF_UPPER", "FINE_TARGET", "FRAME_SECONDS", "Geometry",
    "HEALTH_GATE_SCHEMA", "HealthGate", "MASK_RULE", "NFFT", "PRODUCT_SCHEMA_NAME",
    "PRODUCT_SCHEMA_REVISION", "PRODUCT_SCHEMA_TOKEN", "PSD_BIN_HZ", "PSD_DB_INVALID", "Product",
    "ProductContractError", "ProductView", "SAMPLE_RATE_HZ", "SCHEMA_TOKEN",
    "SOURCE_EVENT_KEY_SCHEMA", "VALID_RULE", "coarse_reject_mask", "fine_bin_of_hz",
    "fine_hz_of_bin", "fine_offset_to_rf_hz", "fine_power_ratio", "grid_residual_hz",
    "health_gate", "is_current_product", "legacy_reading", "open_product", "sha256_of",
]

# coding=utf-8
"""Current per-pilot product and decision-method identities.

Schema identity, scientific measurement, and rejection policy are separate
coordinates.  Keeping them explicit prevents a diagnostic fine spectrum or an
implemented kernel entry point from being mistaken for the active mask used by
an archive product.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from pilot_proxy.archived_product_keys import (
    ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_REFERENCE_NORM_SUM_SQ,
)
from pilot_proxy.atsc_channels import (
    ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    ATSC_UHF_MIN_PHYSICAL_CHANNEL,
)
from pilot_proxy.detector_contract import (
    NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    normalized_positive_excess,
)
from pilot_proxy.detector_reference import REFERENCE_WEIGHT_TERMS
from pilot_proxy.fine_reduction import (
    WEIGHT_TERM_REF_LOWER,
    WEIGHT_TERM_REF_UPPER,
    WEIGHT_TERM_TARGET,
)

PER_PILOT_PRODUCT_SCHEMA_NAME = "pilotproxy_per_pilot_product"
# v2 retains the exact uint64 fine-power terms and the lower/upper
# reference split. v1 products collapsed both, which made an exact Q16
# fine replay impossible after the fact; they are rejected, not migrated.
# v3 adds the per-frame PSD the analyzer already computed and discarded,
# so a later pass can apply a new frame mask, window, or threshold to the
# spectra instead of being limited to the two archived accumulators.
# v4 added the per-frame railed-sample counts; saturation cannot be
# recovered later because no raw samples are stored.
# v5 splits the archive's 0x00 missing-data fill signature out of the
# railed count (fill_sample_count), so incomplete baseband is no longer
# recorded as clipping. No v4 cohort was ever produced; v4 snapshots are
# rejected like every other development snapshot.
PER_PILOT_PRODUCT_SCHEMA_REVISION = 5
PER_PILOT_PRODUCT_SCHEMA_TOKEN = (
    f"{PER_PILOT_PRODUCT_SCHEMA_NAME}_v{PER_PILOT_PRODUCT_SCHEMA_REVISION}"
)

# Every per-frame array the analyzer writes (length n_frames along axis 0),
# in one authoritative place: event-keyed alignment in combine, the
# resume/compare gate, and the deep audit all treat per-frame data by this
# list, so a schema bump that adds a per-frame field must extend it here or
# the field silently escapes alignment and comparison.
PER_FRAME_PRODUCT_KEYS = (
    "frame_index",
    "p_target_u64",
    "p_ref_sum_u64",
    "p_ref_lower_u64",
    "p_ref_upper_u64",
    "coarse_power_ratio",
    "normalized_coarse_power_ratio_db",
    "pilot_excess_db",
    "estimated_data_shelf_snr_db",
    "normalized_pilot_excess",
    "reject_mask",
    "valid",
    "baseband_power_linear",
    "railed_sample_count",
    "fill_sample_count",
    "railed_sample_total",
    "fine_power_u64",
    "psd_frame_db_i16",
    "psd_db_reference",
    "frame_unit_index",
    "frame_in_unit",
)
# Per-frame PSD encoding: int16 codes at 0.01 dB per step about a
# per-frame reference. The sentinel marks a frame that never reached the
# transform and sits outside the clipped range, so it cannot be read back
# as a measured level.
PSD_DB_INVALID = -32768
PSD_DB_MIN = -32767
PSD_DB_MAX = 32767
PSD_DB_STEP = 0.01

SOURCE_EVENT_KEY_SCHEMA_VERSION = "pilotproxy_namespaced_source_event_key_v1"

COARSE_MEASUREMENT_METHOD = "coarse_local_reference_power_ratio"
FINE_MEASUREMENT_METHOD = "fine_local_reference_power_ratio"
ACTIVE_DECISION_METHOD = "coarse_normalized_positive_excess"
ACTIVE_DECISION_IMPLEMENTATION = "host_exact_integer_comparison"
FINE_TERMS_METHOD = "exact_fine_power_terms"
FINE_TERMS_ROLE = "measurement_only_no_scan_time_decision"
FINE_CANDIDATE_DECISION_METHOD = "fine_order_statistic_cfar"
FINE_CANDIDATE_CALIBRATION_STATUS = "pending_campaign"

FINE_STATUS_ENABLED = "enabled"
FINE_STATUS_PENDING = "pending"
FINE_STATUS_WITHOUT_TERMS = frozenset(
    {
        "disabled_by_option",
        "detector_fn_lacks_matched_filter_row_projections",
        "kernel_library_lacks_matched_filter_row_projections",
        "not_applicable_pilot_out_of_band",
    }
)


def current_decision_contract() -> dict[str, Any]:
    """Return a fresh JSON-safe description of product decision semantics."""
    return {
        "measurements": [COARSE_MEASUREMENT_METHOD, FINE_MEASUREMENT_METHOD],
        "active_decision": {
            "method": ACTIVE_DECISION_METHOD,
            "implementation": ACTIVE_DECISION_IMPLEMENTATION,
            "output_field": "reject_mask",
        },
        "fine_measurement": {
            "method": FINE_TERMS_METHOD,
            "role": FINE_TERMS_ROLE,
            "terms_field": "fine_power_u64",
            "terms_note": (
                "exact uint64 [3, num_bins] from the frozen fxfft256: the "
                "deployed fine statistic's numerators and denominators. The "
                "scan stores these and nothing derived from them, so the "
                "ratio, the CFAR baseline, the designated-set decision, and "
                "any threshold are recomputed in post-processing at whatever "
                "operating point is calibrated."
            ),
            "no_scan_time_decision": (
                "The scan applies no fine decision. A float ratio formed with "
                "numpy.fft was previously stored and was never bit-equal to "
                "the deployed statistic; it is no longer written."
            ),
        },
        "fine_candidate_decision": {
            "method": FINE_CANDIDATE_DECISION_METHOD,
            "calibration_status": FINE_CANDIDATE_CALIBRATION_STATUS,
            "active": False,
        },
    }


class CurrentProductContractError(ValueError):
    """A per-pilot product does not satisfy the only supported contract."""


def null_power_ratio_of(product: Mapping[str, Any]) -> float:
    """The weight-norm correction, derived rather than read.

    ``mu_0 = 2 * target_norm_sq / reference_norm_sum_sq`` is an exact rational
    of two integers every product carries, so it is never stored. Archived
    products spell the denominator differently; the migration map resolves it,
    and the derivation reproduces their stored ``mu0`` exactly on all 23
    archived channels.
    """
    target = int(np.asarray(product["target_norm_sq"]).reshape(-1)[0])
    # NpzFile is not a full Mapping, so test membership rather than use .get.
    name = (
        "reference_norm_sum_sq"
        if "reference_norm_sum_sq" in product
        else ARCHIVED_REFERENCE_NORM_SUM_SQ
    )
    denominator = int(np.asarray(product[name]).reshape(-1)[0])
    if denominator <= 0:
        raise CurrentProductContractError(
            "reference_norm_sum_sq must be positive to form the weight-norm "
            "correction"
        )
    return 2.0 * target / denominator


def fine_power_ratio_of(product: Mapping[str, Any]) -> np.ndarray:
    """The per-frame fine statistic, ``F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])``.

    Current products retain the exact ``uint64`` fine-power terms
    (``fine_power_u64``, shape ``(N, 3, B)``) and store nothing derived from
    them, so the ratio is formed here, the way ``fine_reduction.fine_reduce``
    forms it (a zero denominator gives 0.0). Archived products stored a
    ``float32`` ratio formed with ``numpy.fft`` instead; it is returned as
    stored, under either spelling. The two are the same statistic but not the
    same numbers: the archived ratio was never bit-equal to the deployed
    fixed-point transform.

    Returns ``float64`` of shape ``(N, B)``.
    """
    if "fine_power_u64" in product:
        terms = np.asarray(product["fine_power_u64"])
        if terms.ndim != 3:
            raise CurrentProductContractError(
                "fine_power_u64 must have shape (N, 3, B); got "
                f"{terms.shape}"
            )
        if terms.shape[1] == 0 or terms.shape[2] == 0:
            # The (N, 0, 0) placeholder of a product whose fine stage was off.
            raise CurrentProductContractError(
                "product retains no fine terms (fine_num_bins = 0), so no fine "
                "power ratio can be formed"
            )
        if terms.shape[1] != 3:
            raise CurrentProductContractError(
                "fine_power_u64 must have shape (N, 3, B); got "
                f"{terms.shape}"
            )
        s_t = terms[:, WEIGHT_TERM_TARGET].astype(np.float64)
        s_l = terms[:, WEIGHT_TERM_REF_LOWER].astype(np.float64)
        s_u = terms[:, WEIGHT_TERM_REF_UPPER].astype(np.float64)
        den = s_l + s_u
        positive = den > 0.0
        return np.where(positive, 2.0 * s_t / np.where(positive, den, 1.0), 0.0)
    for name in ("fine_power_ratio", ARCHIVED_FINE_POWER_RATIO):
        if name in product:
            return np.asarray(product[name], dtype=np.float64)
    raise CurrentProductContractError(
        "product carries neither fine_power_u64 nor a stored fine power ratio"
    )


def exact_integer_array(
    value: Any,
    *,
    field: str,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
    ndim: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    """Return an integer array without accepting bool/float/string coercion."""
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"} or array.dtype.kind == "b":
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have an integer dtype"
        )
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have dtype "
            f"{np.dtype(dtype)}, got {array.dtype}"
        )
    if ndim is not None and array.ndim != ndim:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be {ndim}D"
        )
    if minimum is not None and array.size and np.any(array < minimum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be >= {minimum}"
        )
    if maximum is not None and array.size and np.any(array > maximum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be <= {maximum}"
        )
    return array


def exact_integer_scalar(
    product: Mapping[str, Any],
    field: str,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    array = exact_integer_array(
        product.get(field, np.asarray([], dtype=dtype)),
        field=field,
        dtype=dtype,
        minimum=minimum,
        maximum=maximum,
    )
    if field not in product:
        raise CurrentProductContractError(
            f"current per-pilot product is missing required field {field!r}"
        )
    if array.size != 1:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be scalar"
        )
    return int(array.reshape(()).item())


def _exact_string_scalar(product: Mapping[str, Any], field: str) -> str:
    if field not in product:
        raise CurrentProductContractError(
            f"current per-pilot product is missing required field {field!r}"
        )
    value = np.asarray(product[field])
    if value.size != 1 or value.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be a string scalar"
        )
    return str(value.reshape(()).item())


def _exact_float_array(
    product: Mapping[str, Any],
    field: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
    allow_nan: bool = True,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    array = np.asarray(product[field])
    if array.dtype != np.dtype(dtype):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have dtype "
            f"{np.dtype(dtype)}, got {array.dtype}"
        )
    if shape is not None and array.shape != shape:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have shape {shape}, "
            f"got {array.shape}"
        )
    if ndim is not None and array.ndim != ndim:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be {ndim}D"
        )
    if np.any(np.isinf(array)) or (not allow_nan and np.any(np.isnan(array))):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} contains non-finite values"
        )
    finite = array[np.isfinite(array)]
    if minimum is not None and finite.size and np.any(finite < minimum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be >= {minimum} when finite"
        )
    if maximum is not None and finite.size and np.any(finite > maximum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be <= {maximum} when finite"
        )
    return array


def _exact_string_vector(
    product: Mapping[str, Any], field: str, *, length: int
) -> list[str]:
    array = np.asarray(product[field])
    if array.ndim != 1 or array.size != length or array.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be a string vector of "
            f"length {length}"
        )
    return [str(value) for value in array.tolist()]


def validate_current_product_identity(
    product: Mapping[str, Any], *, allow_empty_checkpoint: bool = False
) -> None:
    """Require the only supported schema and decision semantics.

    Development snapshots must be deleted and regenerated. This function
    intentionally contains no aliases, migration, or best-effort fallback.
    """
    schema_name = _exact_string_scalar(product, "schema_name")
    schema_revision = exact_integer_scalar(
        product,
        "schema_revision",
        dtype=np.int64,
        minimum=PER_PILOT_PRODUCT_SCHEMA_REVISION,
        maximum=PER_PILOT_PRODUCT_SCHEMA_REVISION,
    )
    schema_token = _exact_string_scalar(product, "schema_version")
    if (
        schema_name != PER_PILOT_PRODUCT_SCHEMA_NAME
        or schema_revision != PER_PILOT_PRODUCT_SCHEMA_REVISION
        or schema_token != PER_PILOT_PRODUCT_SCHEMA_TOKEN
    ):
        raise CurrentProductContractError(
            "unsupported per-pilot product identity: "
            f"schema_name={schema_name!r}, schema_revision={schema_revision!r}, "
            f"schema_version={schema_token!r}; delete the product and regenerate "
            "it with the current PilotProxy release. A checkpoint from an older "
            "schema cannot be resumed here: the per-frame railed/fill counts "
            "are taken from raw samples the product does not retain, so they "
            "cannot be backfilled -- finish that cohort on the release that "
            "started it, or rebuild from the archive"
        )

    event_key_schema = _exact_string_scalar(
        product, "source_event_key_schema_version"
    )
    if event_key_schema != SOURCE_EVENT_KEY_SCHEMA_VERSION:
        raise CurrentProductContractError(
            "unsupported source-event identity schema: "
            f"{event_key_schema!r}; delete the product and regenerate it so "
            "event keys retain their archive/campaign namespace"
        )

    raw_contract = _exact_string_scalar(product, "decision_contract_json")
    try:
        decision_contract = json.loads(raw_contract)
    except json.JSONDecodeError as exc:
        raise CurrentProductContractError(
            "current per-pilot decision_contract_json is invalid JSON"
        ) from exc
    if decision_contract != current_decision_contract():
        raise CurrentProductContractError(
            "per-pilot decision contract does not match the current release; "
            "delete the product and regenerate it"
        )

    required = {
        "physical_channel",
        "freq_id",
        "pilot_in_band",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "nfft",
        "sample_rate_hz",
        "detector_window_samples",
        "num_input_streams",
        "sense",
        "frame_index",
        "p_target_u64",
        "p_ref_sum_u64",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "fine_power_u64",
        "psd_frame_db_i16",
        "psd_db_reference",
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "reject_mask",
        "valid",
        "target_norm_sq",
        "reference_norm_sum_sq",
        "normalized_pilot_excess",
        "baseband_power_linear",
        "railed_sample_count",
        "fill_sample_count",
        "railed_sample_total",
        "integrated_spectrum_before_mask",
        "integrated_spectrum_after_mask",
        "fine_pad_factor",
        "fine_num_bins",
        "fine_p_fa",
        "fine_guard_fine_bins",
        "fine_designated_bins",
        "fine_census_excluded_bins",
        "fine_status",
        "source_event_keys",
        "unit_keys",
        "unit_order",
        "unit_time0_ctime",
        "unit_time0_fpga",
        "unit_event_id",
        "unit_delta_time",
        "archive_version",
        "unit_git_version_tag",
        "unit_input_map_sha256",
        "unit_collection_server",
        "unit_scope",
        "frame_unit_index",
        "frame_in_unit",
        "detector_contract_json",
        "max_chunks_per_file",
        "rational_overflow_count",
        "weights_hash",
        "weight_bank_sha256",
        "weight_manifest_sha256",
        "detector_version",
        "mask_rule",
        "reference_placement_json",
        "pilot_below_data_db",
        "bin_enbw_hz",
        "dtv_bandwidth_hz",
        "pilot_capture_efficiency",
    }
    missing = sorted(required.difference(product))
    if missing:
        raise CurrentProductContractError(
            "current per-pilot product is missing required arrays: "
            + ", ".join(missing)
        )

    if "mask" in product:
        raise CurrentProductContractError(
            "per-pilot product contains the ambiguous field 'mask'; the current "
            "decision field is 'reject_mask'"
        )

    exact_integer_scalar(
        product,
        "physical_channel",
        dtype=np.int32,
        minimum=ATSC_UHF_MIN_PHYSICAL_CHANNEL,
        maximum=ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    )
    exact_integer_scalar(
        product, "freq_id", dtype=np.int64, minimum=0, maximum=1023
    )
    nfft = exact_integer_scalar(
        product, "nfft", dtype=np.int64, minimum=1
    )
    exact_integer_scalar(
        product, "detector_window_samples", dtype=np.int64, minimum=1
    )
    num_input_streams = exact_integer_scalar(
        product, "num_input_streams", dtype=np.int64, minimum=1
    )
    sense = exact_integer_scalar(
        product, "sense", dtype=np.int64, minimum=-1, maximum=1
    )
    if sense not in {-1, 1}:
        raise CurrentProductContractError(
            "current per-pilot field 'sense' must be exactly -1 or 1"
        )
    exact_integer_scalar(
        product, "max_chunks_per_file", dtype=np.int64, minimum=-1
    )
    exact_integer_scalar(
        product, "rational_overflow_count", dtype=np.uint64, minimum=0
    )
    pilot_in_band = exact_integer_array(
        product["pilot_in_band"],
        field="pilot_in_band",
        dtype=np.uint8,
        ndim=1,
        minimum=0,
        maximum=1,
    )
    if pilot_in_band.shape != (1,):
        raise CurrentProductContractError(
            "current per-pilot field 'pilot_in_band' must have shape (1,)"
        )

    _exact_float_array(
        product, "pilot_frequency_hz", shape=(1,), allow_nan=False
    )
    _exact_float_array(
        product, "chime_frequency_hz", shape=(1,), allow_nan=False
    )
    sample_rate = _exact_float_array(
        product, "sample_rate_hz", shape=(), allow_nan=False, minimum=0.0
    )
    if float(sample_rate.reshape(()).item()) <= 0.0:
        raise CurrentProductContractError(
            "current per-pilot field 'sample_rate_hz' must be positive"
        )
    for field, minimum, maximum in (
        ("pilot_below_data_db", 0.0, None),
        ("bin_enbw_hz", 0.0, None),
        ("dtv_bandwidth_hz", 0.0, None),
        ("pilot_capture_efficiency", 0.0, 1.0),
    ):
        values = _exact_float_array(
            product,
            field,
            shape=(),
            allow_nan=False,
            minimum=minimum,
            maximum=maximum,
        )
        if field != "pilot_below_data_db" and float(values.reshape(()).item()) <= 0.0:
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must be positive"
            )
    for field in (
        "detector_contract_json",
        "weights_hash",
        "weight_bank_sha256",
        "weight_manifest_sha256",
        "detector_version",
        "mask_rule",
        "reference_placement_json",
    ):
        _exact_string_scalar(product, field)
    if _exact_string_scalar(product, "mask_rule") != NORMALIZED_POSITIVE_EXCESS_MASK_RULE:
        raise CurrentProductContractError(
            "current per-pilot mask_rule does not declare the normalized "
            "positive-excess policy"
        )

    frame_index = exact_integer_array(
        product["frame_index"],
        field="frame_index",
        dtype=np.int64,
        ndim=1,
        minimum=0,
    )
    frame_count = int(frame_index.size)
    if (frame_count == 0 and not allow_empty_checkpoint) or not np.array_equal(
        frame_index, np.arange(frame_count, dtype=np.int64)
    ):
        raise CurrentProductContractError(
            "current per-pilot frame_index must be a non-empty contiguous "
            "zero-based int64 vector"
        )

    unit_order_array = np.asarray(product["unit_order"])
    if unit_order_array.ndim != 1 or unit_order_array.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            "current per-pilot field 'unit_order' must be a 1D string vector"
        )
    unit_count = int(unit_order_array.size)
    if unit_count == 0 and not allow_empty_checkpoint:
        raise CurrentProductContractError(
            "current per-pilot unit_order must not be empty"
        )
    unit_order = _exact_string_vector(
        product, "unit_order", length=unit_count
    )
    unit_keys = _exact_string_vector(product, "unit_keys", length=unit_count)
    source_event_keys = _exact_string_vector(
        product, "source_event_keys", length=unit_count
    )
    _exact_string_vector(product, "archive_version", length=unit_count)
    git_version_tags = _exact_string_vector(
        product, "unit_git_version_tag", length=unit_count
    )
    input_map_sha256 = _exact_string_vector(
        product, "unit_input_map_sha256", length=unit_count
    )
    _exact_string_vector(product, "unit_collection_server", length=unit_count)
    unit_scope = _exact_string_vector(product, "unit_scope", length=unit_count)
    if any(not value.strip() for value in unit_scope):
        raise CurrentProductContractError(
            "current per-pilot unit_scope values must not be empty"
        )
    lowercase_hex = set("0123456789abcdef")
    if any(
        value and (len(value) != 64 or not set(value) <= lowercase_hex)
        for value in input_map_sha256
    ):
        raise CurrentProductContractError(
            "current per-pilot unit_input_map_sha256 values must be empty or "
            "64-character lowercase SHA-256 digests"
        )
    if any(
        scope != "local" and (not tag or not input_hash)
        for scope, tag, input_hash in zip(
            unit_scope, git_version_tags, input_map_sha256
        )
    ):
        raise CurrentProductContractError(
            "current per-pilot archive units require nonempty "
            "unit_git_version_tag and unit_input_map_sha256 values"
        )
    if len(set(unit_order)) != unit_count or len(set(unit_keys)) != unit_count:
        raise CurrentProductContractError(
            "current per-pilot unit_order and unit_keys must each be unique"
        )
    if set(unit_order) != set(unit_keys):
        raise CurrentProductContractError(
            "current per-pilot unit_order and unit_keys must contain the exact "
            "same unit identities"
        )
    if any(not value for value in unit_order + unit_keys + source_event_keys):
        raise CurrentProductContractError(
            "current per-pilot unit and source-event identities must not be empty"
        )
    if len(set(source_event_keys)) != unit_count:
        raise CurrentProductContractError(
            "current per-pilot source_event_keys must be unique"
        )

    unit_event_id = exact_integer_array(
        product["unit_event_id"],
        field="unit_event_id",
        dtype=np.int64,
        ndim=1,
        minimum=-1,
    )
    unit_time0_fpga = exact_integer_array(
        product["unit_time0_fpga"],
        field="unit_time0_fpga",
        dtype=np.uint64,
        ndim=1,
        minimum=0,
    )
    for field, values in (
        ("unit_event_id", unit_event_id),
        ("unit_time0_fpga", unit_time0_fpga),
    ):
        if values.shape != (unit_count,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({unit_count},)"
            )
    _exact_float_array(
        product, "unit_time0_ctime", shape=(unit_count,), allow_nan=True
    )
    unit_delta_time = _exact_float_array(
        product,
        "unit_delta_time",
        shape=(unit_count,),
        allow_nan=True,
        minimum=0.0,
    )
    if np.any(np.isfinite(unit_delta_time) & (unit_delta_time <= 0.0)):
        raise CurrentProductContractError(
            "current per-pilot finite unit_delta_time values must be positive"
        )
    finite_delta_time = unit_delta_time[np.isfinite(unit_delta_time)]
    if finite_delta_time.size and not np.allclose(
        finite_delta_time,
        1.0 / float(sample_rate.reshape(()).item()),
        rtol=1e-12,
        atol=0.0,
    ):
        raise CurrentProductContractError(
            "current per-pilot unit_delta_time disagrees with sample_rate_hz"
        )

    frame_unit_index = exact_integer_array(
        product["frame_unit_index"],
        field="frame_unit_index",
        dtype=np.int32,
        ndim=1,
        minimum=0,
        maximum=unit_count - 1,
    )
    frame_in_unit = exact_integer_array(
        product["frame_in_unit"],
        field="frame_in_unit",
        dtype=np.int32,
        ndim=1,
        minimum=0,
    )
    for field, values in (
        ("frame_unit_index", frame_unit_index),
        ("frame_in_unit", frame_in_unit),
    ):
        if values.shape != (frame_count,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count},)"
            )
    if frame_count == 0 and unit_count != 0:
        raise CurrentProductContractError(
            "an empty current per-pilot checkpoint must not claim consumed units"
        )
    if frame_count and set(frame_unit_index.tolist()) != set(range(unit_count)):
        raise CurrentProductContractError(
            "current per-pilot frame_unit_index must reference every consumed "
            "unit at least once"
        )

    exact_powers: dict[str, np.ndarray] = {}
    for field in (
        "p_target_u64",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "p_ref_sum_u64",
    ):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint64,
            minimum=0,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        exact_powers[field] = values
    for frame, (lower, upper, total) in enumerate(
        zip(
            exact_powers["p_ref_lower_u64"].reshape(-1),
            exact_powers["p_ref_upper_u64"].reshape(-1),
            exact_powers["p_ref_sum_u64"].reshape(-1),
        )
    ):
        combined = int(lower) + int(upper)
        if combined >= (1 << 64):
            raise CurrentProductContractError(
                "current per-pilot reference power sum exceeds uint64 at "
                f"frame {frame}"
            )
        if combined != int(total):
            raise CurrentProductContractError(
                "current per-pilot lower and upper reference powers do not "
                f"match p_ref_sum_u64 at frame {frame}"
            )
    exact_flags: dict[str, np.ndarray] = {}
    for field in ("reject_mask", "valid"):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint8,
            minimum=0,
            maximum=1,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        exact_flags[field] = values
    # ADC/F-engine saturation and 0x00 fill, counted from the raw int4 nibbles
    # while the frame was in hand. The product stores no raw samples, so an
    # absent or wrong count cannot be repaired later; validate it here rather
    # than trusting the writer.
    railed: dict[str, np.ndarray] = {}
    for field in ("railed_sample_count", "fill_sample_count", "railed_sample_total"):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint64,
            minimum=0,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        railed[field] = values
    sample_counts_fit = all(
        int(railed_count) + int(fill_count) <= int(total)
        for railed_count, fill_count, total in zip(
            railed["railed_sample_count"].reshape(-1),
            railed["fill_sample_count"].reshape(-1),
            railed["railed_sample_total"].reshape(-1),
        )
    )
    if not sample_counts_fit:
        raise CurrentProductContractError(
            "current per-pilot railed_sample_count + fill_sample_count exceeds "
            "railed_sample_total; the saturation counters and their denominator "
            "disagree"
        )
    expected_total = np.uint64(2 * nfft * num_input_streams)
    if np.any(railed["railed_sample_total"] != expected_total):
        raise CurrentProductContractError(
            "current per-pilot railed_sample_total does not equal "
            "2 * nfft * num_input_streams "
            f"({int(expected_total)}); the denominator no longer describes the "
            "frame it was counted from"
        )
    for field in PER_FRAME_PRODUCT_KEYS:
        if field not in product:
            raise CurrentProductContractError(
                f"current per-pilot product is missing per-frame field {field!r}"
            )
        values = np.asarray(product[field])
        if values.ndim == 0 or int(values.shape[0]) != frame_count:
            raise CurrentProductContractError(
                f"current per-pilot per-frame field {field!r} does not have "
                f"{frame_count} frames along axis 0"
            )
    for field, minimum in (
        ("coarse_power_ratio", 0.0),
        ("normalized_coarse_power_ratio_db", None),
        ("pilot_excess_db", None),
        ("estimated_data_shelf_snr_db", None),
        ("normalized_pilot_excess", -1.0),
        ("baseband_power_linear", 0.0),
    ):
        _exact_float_array(
            product,
            field,
            shape=(frame_count, 1),
            allow_nan=True,
            minimum=minimum,
        )

    fine_num_bins = exact_integer_scalar(
        product, "fine_num_bins", dtype=np.int64, minimum=0
    )
    # The exact deployed terms: target, lower reference, upper reference, per
    # fine bin. Every fine diagnostic and decision is recomputed from these in
    # post-processing, so the scan stores nothing derived from them.
    fine_terms = exact_integer_array(
        product["fine_power_u64"],
        field="fine_power_u64",
        dtype=np.uint64,
        ndim=3,
        minimum=0,
    )
    if fine_terms.shape[0] != frame_count:
        raise CurrentProductContractError(
            "current per-pilot fine_power_u64 frame axis must match frame_index"
        )
    if fine_terms.shape[2] != fine_num_bins:
        raise CurrentProductContractError(
            "current per-pilot fine_power_u64 width must match fine_num_bins"
        )
    fine_status = _exact_string_scalar(product, "fine_status")
    if fine_status == FINE_STATUS_ENABLED:
        if fine_num_bins <= 0:
            raise CurrentProductContractError(
                "current per-pilot enabled fine measurement must have bins"
            )
        expected_fine_shape = (
            frame_count,
            REFERENCE_WEIGHT_TERMS,
            fine_num_bins,
        )
        if fine_terms.shape != expected_fine_shape:
            raise CurrentProductContractError(
                "current per-pilot enabled fine_power_u64 must have shape "
                f"{expected_fine_shape}"
            )
    elif fine_status == FINE_STATUS_PENDING:
        if frame_count != 0:
            raise CurrentProductContractError(
                "current per-pilot fine status may be pending only in an empty "
                "checkpoint"
            )
        if fine_num_bins != 0 or fine_terms.shape != (0, 0, 0):
            raise CurrentProductContractError(
                "current per-pilot pending fine measurement must not carry terms"
            )
    elif fine_status in FINE_STATUS_WITHOUT_TERMS:
        canonical_empty = (
            fine_num_bins == 0
            and fine_terms.shape == (frame_count, 0, 0)
        )
        legacy_empty = (
            fine_num_bins > 0
            and fine_terms.shape
            == (frame_count, REFERENCE_WEIGHT_TERMS, fine_num_bins)
            and not np.any(fine_terms)
        )
        if not canonical_empty and not legacy_empty:
            raise CurrentProductContractError(
                "current per-pilot fine status without measurements must use "
                f"shape ({frame_count}, 0, 0) or an all-zero v5 placeholder"
            )
    else:
        allowed = sorted(
            {FINE_STATUS_ENABLED, FINE_STATUS_PENDING, *FINE_STATUS_WITHOUT_TERMS}
        )
        raise CurrentProductContractError(
            f"current per-pilot fine_status must be one of {allowed}"
        )

    exact_integer_scalar(
        product, "fine_pad_factor", dtype=np.int64, minimum=1
    )
    exact_integer_scalar(
        product, "fine_guard_fine_bins", dtype=np.int64, minimum=0
    )
    fine_p_fa = _exact_float_array(
        product, "fine_p_fa", shape=(), allow_nan=False, minimum=0.0, maximum=1.0
    )
    if not 0.0 < float(fine_p_fa.reshape(()).item()) < 1.0:
        raise CurrentProductContractError(
            "current per-pilot fine_p_fa must be strictly between zero and one"
        )
    for field in ("fine_designated_bins", "fine_census_excluded_bins"):
        exact_integer_array(product[field], field=field, dtype=np.int64, ndim=1)
    exact_norms: dict[str, int] = {}
    for field in ("target_norm_sq", "reference_norm_sum_sq"):
        values = exact_integer_array(
            product[field], field=field, dtype=np.int64, minimum=1
        )
        if values.shape != (1,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape (1,)"
            )
        exact_norms[field] = int(values[0])
    # The weight-norm correction is not stored. It is the exact rational
    # 2 * target_norm_sq / reference_norm_sum_sq of two integers validated
    # above, so a float copy would be its lossy form and would need its own
    # name; the contract used to check the two agreed, which is the same
    # statement as not needing both.
    if exact_norms["reference_norm_sum_sq"] <= 0:
        raise CurrentProductContractError(
            "current per-pilot reference_norm_sum_sq must be positive to form "
            "the weight-norm correction"
        )

    p_target = exact_powers["p_target_u64"].reshape(-1)
    p_ref_sum = exact_powers["p_ref_sum_u64"].reshape(-1)
    valid = exact_flags["valid"].reshape(-1)
    reject = exact_flags["reject_mask"].reshape(-1)
    expected_valid = (p_ref_sum != 0).astype(np.uint8)
    if not np.array_equal(valid, expected_valid):
        raise CurrentProductContractError(
            "current per-pilot valid flags disagree with p_ref_sum != 0"
        )
    expected_reject = np.asarray(
        [
            normalized_positive_excess(
                int(num),
                int(den),
                target_norm_sq=exact_norms["target_norm_sq"],
                reference_norm_sum_sq=exact_norms["reference_norm_sum_sq"],
            )
            for num, den in zip(p_target, p_ref_sum)
        ],
        dtype=np.uint8,
    )
    if not np.array_equal(reject, expected_reject):
        raise CurrentProductContractError(
            "current per-pilot reject_mask disagrees with the exact normalized "
            "positive-excess decision over stored powers and norms"
        )
    spectrum_before = _exact_float_array(
        product,
        "integrated_spectrum_before_mask",
        shape=(nfft,),
        allow_nan=False,
        minimum=0.0,
    )
    spectrum_after = _exact_float_array(
        product,
        "integrated_spectrum_after_mask",
        shape=(nfft,),
        allow_nan=False,
        minimum=0.0,
    )
    if np.any(spectrum_after > spectrum_before):
        raise CurrentProductContractError(
            "current per-pilot integrated spectrum after mask exceeds before mask"
        )


def current_decision_contract_json() -> str:
    """Return the canonical stable serialization stored in products."""
    return json.dumps(current_decision_contract(), sort_keys=True, separators=(",", ":"))

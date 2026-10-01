# coding=utf-8
"""CHIME DTV local-reference power-ratio archive analyzer.

The detector functionality of ``pilot_proxy.chime.runner.run_chime_analysis`` as a
streaming analyzer. It reimplements no DSP: it wraps
PilotProxy's own ``pack_chime_block_for_detector`` -> ``detector_fn`` (the CUDA kernel
via ``detect_packed_detector_input`` by default) -> ``_append_detection_rows``
maths, so the per-frame product matches the runner by construction.

Data path (per coarse channel / pilot, fanned out one ``<channel>.npz`` each):

* the ``chime-baseband-packed`` reader yields one raw ``uint8 [nfft, n_feeds]``
  frame per chunk (native offset-binary 4+4-bit);
* the analyzer reshapes it to PilotProxy's normalized ``(n_feeds, 1, nfft)`` block and
  calls ``pack_chime_block_for_detector`` with ``sample_encoding`` =
  native-offset-binary, which takes the LOSSLESS repack route (no calibration
  scale or requantization; the native int4 grid passes straight through);
* ``detector_fn(packed=packed.packed, weights=weights, kernel=kernel)`` yields
  per-frame target/reference powers; the positive-excess mask + dB metrics come
  from ``dtv_units`` exactly as ``_append_detection_rows`` computes them.

The CUDA kernel is GPU-only, so the real kernel-level + real-data parity is a
CANFAR/GPU step. ``detector_fn`` / ``kernel`` / ``weights`` are injectable (via
``ctx.options``), mirroring ``run_chime_analysis``, so a CPU reference can drive
a GPU-free plumbing parity test in the same way PilotProxy's own runner test does.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import operator
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from pilot_proxy.archive import accel
from pilot_proxy.archive.instruments import nyquist_sign
from pilot_proxy.archive.interfaces import (
    EXPERIMENTAL,
    Analyzer,
    PluginInfo,
    RunContext,
)
from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
from pilot_proxy.chime.hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    nearest_atsc_physical_channel,
)
from pilot_proxy.chime.products import SAMPLE_RATE_HZ, atomic_savez_compressed
from pilot_proxy.detect import detect_packed_detector_input
from pilot_proxy.detector_contract import (
    NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
    build_detector_contract,
    normalize_weight_coordinate_system,
    normalized_positive_excess,
    null_power_ratio_from_weight_norms,
    weight_term_norms_sq,
)
from pilot_proxy.detector_geometry import (
    DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS,
    SPECTRAL_SENSE_INVERTED,
    SPECTRAL_SENSE_NORMAL,
    predicted_fine_designated_bins,
    predicted_pilot_fine_bin,
)
from pilot_proxy.detector_reference import REFERENCE_WEIGHT_TERMS
from pilot_proxy.dtv_units import (
    DETECTOR_WINDOW_SAMPLES,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    normalized_pilot_excess_to_db,
    pilot_excess_db_to_data_shelf_snr_db,
    power_terms_to_coarse_power_ratio,
    power_terms_to_normalized_coarse_power_ratio_db,
    power_terms_to_normalized_pilot_excess,
)
from pilot_proxy.fine_reduction import (
    CFAR_DEFAULT_GUARD_FINE_BINS,
    CFAR_DEFAULT_P_FA,
    CFAR_MODE_MEDIAN_LEFT,
    CFAR_MODE_QUANTILE_FALLBACK,
    FINE_PAD_FACTOR,
    fine_bin_count,
)
from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.product_contract import (
    FINE_STATUS_WITHOUT_TERMS,
    PER_PILOT_PRODUCT_SCHEMA_NAME,
    PER_PILOT_PRODUCT_SCHEMA_REVISION,
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    PSD_DB_INVALID,
    PSD_DB_MAX,
    PSD_DB_MIN,
    PSD_DB_STEP,
    SOURCE_EVENT_KEY_SCHEMA_VERSION,
    CurrentProductContractError,
    current_decision_contract,
    current_decision_contract_json,
    validate_current_product_identity,
)
from pilot_proxy.provenance import (
    file_sha256,
    package_source_sha256,
    sidecar_manifest_path,
)

from .chime_coarse import chime_freq_id_from_hz, source_event_key
from .stream_kinds import STREAM_PACKED_COMPLEX_INT4_BASEBAND

_FINE_MODE_CODES = {
    "": 0,
    CFAR_MODE_MEDIAN_LEFT: 1,
    CFAR_MODE_QUANTILE_FALLBACK: 2,
}


def _exact_backend_u64(value: object, *, field: str) -> int:
    """Accept an exact backend integer without float/bool truncation or wrap."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"detector analyzer: backend field {field!r} must be an exact "
            "unsigned integer, not a boolean"
        )
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"detector analyzer: backend field {field!r} must be an exact "
            "unsigned integer"
        ) from exc
    if not 0 <= result < (1 << 64):
        raise ValueError(
            f"detector analyzer: backend field {field!r} is outside uint64"
        )
    return int(result)

# Native CHIME baseband packing: one uint8 per complex sample, high nibble = real,
# low nibble = imag, each a 4-bit offset-binary value (stored = signed + 8). This
# matches ``baseband_format.unpack_4bit`` and stays array-module generic so
# the integrated-spectrum FFT runs on cupy (GPU) in production and numpy in tests
# with identical arithmetic.
_INT4_OFFSET = np.float32(8.0)


def _unpack_4bit_xp(xp, packed):
    """offset-binary uint8 [..] -> complex64 [..], on numpy or cupy (`xp`)."""
    real = (packed >> 4).astype(xp.float32) - _INT4_OFFSET
    imag = (packed & np.uint8(0x0F)).astype(xp.float32) - _INT4_OFFSET
    return (real + 1j * imag).astype(xp.complex64)


# Rails per packed byte, indexed by the uint8 value. A nibble of 0 or 15 is
# signed -8 or +7, the ends of the int4 range. The 0x00 byte is excluded: it is
# also the archive's missing-data fill signature, so its components are tallied
# as fill rather than clipping (a genuinely doubly-negative-railed sample is
# indistinguishable from fill and lands in the fill count).
_PACKED_BYTE = np.arange(256, dtype=np.uint16)
_RAILS_PER_BYTE = (
    np.isin(_PACKED_BYTE >> 4, (0, 15)).astype(np.uint8)
    + np.isin(_PACKED_BYTE & 0x0F, (0, 15)).astype(np.uint8)
)
_RAILS_PER_BYTE[0] = 0


def railed_component_count(packed: np.ndarray) -> tuple[int, int]:
    """Count components on the int4 rails, and components in 0x00 fill bytes.

    A nibble of 0 or 15 is signed -8 or +7, the ends of the int4 range, so a
    sample there has been clipped by the ADC or the F-engine requantizer rather
    than measured -- unless the whole byte is 0x00, the archive's missing-data
    fill signature, which is returned as the separate fill count. Real and
    imaginary parts are counted separately, giving a shared denominator of
    ``2 * packed.size``.

    This has to be recorded while the frame is in hand: the product stores no
    raw samples (5.8 MB against 8.0 GB for one channel of the pre-flight), so a
    scan that does not count rails as it goes cannot recover them afterwards.
    ``baseband_power_linear == 128`` only catches a frame that is railed
    everywhere; partial saturation is invisible without this.
    """
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    fill = 2 * int(np.count_nonzero(packed == 0))
    railed = int(_RAILS_PER_BYTE[packed].sum(dtype=np.uint64))
    return railed, fill


def _to_host(a) -> np.ndarray:
    """Bring an accumulator to host numpy (cupy ndarray -> .get(); numpy -> asarray)."""
    get = getattr(a, "get", None)
    return np.asarray(get() if callable(get) else a)


def _callable_accepts_keyword(function: Any, keyword: str) -> bool:
    """Return whether a callable explicitly accepts a keyword argument.

    Feature discovery must happen before invocation.  Catching ``TypeError``
    around the detector call can misclassify a real implementation defect as a
    missing optional feature and can execute the detector twice.
    """
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _detector_fft_backend():
    """Return cupy only when its CUDA runtime is actually usable.

    Some CANFAR / CI images have CuPy importable through system-site packages even
    when the visible node has no compatible CUDA driver.  ``accel.import_cupy()``
    can therefore return the module, but the first allocation later fails with a
    CUDA runtime error.  The detector's integrated-spectrum FFT is only a
    reporting-side accumulator and already has a NumPy path, so fail closed to
    NumPy unless a tiny runtime probe succeeds.
    """
    try:
        cp = accel.import_cupy()
    except Exception:
        return np
    if cp is None:
        return np
    try:
        runtime = getattr(getattr(cp, "cuda", None), "runtime", None)
        get_count = getattr(runtime, "getDeviceCount", None)
        if callable(get_count) and int(get_count()) <= 0:
            return np
        probe = cp.zeros(1, dtype=cp.float32)
        # Touch the array so lazy runtime failures surface during backend
        # selection instead of later, after the analyzer has committed to cupy.
        if hasattr(probe, "sum"):
            probe.sum()
        return cp
    except Exception:
        return np


_DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ = 10.0

_RESUME_REQUIRED_FIELDS = frozenset(
    {
        "archive_version",
        "bin_enbw_hz",
        "detector_window_samples",
        "detector_contract_json",
        "detector_version",
        "decision_contract_json",
        "dtv_bandwidth_hz",
        "fine_census_excluded_bins",
        "fine_designated_bins",
        "fine_guard_fine_bins",
        "fine_num_bins",
        "fine_p_fa",
        "fine_pad_factor",
        "fine_status",
        "max_chunks_per_file",
        "mask_rule",
        "num_input_streams",
        "nfft",
        "pilot_below_data_db",
        "pilot_capture_efficiency",
        "pilot_in_band",
        "rational_overflow_count",
        "reference_placement_json",
        "sample_rate_hz",
        "sense",
        "source_event_key_schema_version",
        "unit_collection_server",
        "unit_delta_time",
        "unit_event_id",
        "unit_git_version_tag",
        "unit_input_map_sha256",
        "unit_keys",
        "unit_order",
        "unit_scope",
        "unit_time0_ctime",
        "unit_time0_fpga",
        "weight_bank_sha256",
        "weight_manifest_sha256",
        "weights_hash",
    }
)


def _validate_resume_product(data: Mapping[str, Any], path: str) -> None:
    """Require a complete current product before restoring mutable state."""
    missing = sorted(_RESUME_REQUIRED_FIELDS.difference(data))
    if missing:
        raise SystemExit(
            "pilot-proxy-detector: existing product lacks resume-critical fields "
            f"{missing}; remove it and rebuild with the current version."
        )
    try:
        validate_current_product_identity(data, allow_empty_checkpoint=True)
    except CurrentProductContractError as exc:
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has invalid current "
            f"per-pilot data ({exc}); remove it and rebuild."
        ) from exc


def _validated_resume_axes(
    data: Mapping[str, Any], path: str
) -> tuple[list[str], set[str]]:
    """Validate the frame-to-unit relationship before restoring its arrays."""
    unit_order = [
        str(value) for value in np.asarray(data["unit_order"]).reshape(-1)
    ]
    unit_key_values = [
        str(value) for value in np.asarray(data["unit_keys"]).reshape(-1)
    ]
    unit_keys = set(unit_key_values)
    if len(unit_order) != len(set(unit_order)):
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has duplicate unit_order "
            "entries; remove it and rebuild."
        )
    if len(unit_key_values) != len(unit_keys):
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has duplicate unit_keys; "
            "remove it and rebuild."
        )
    if set(unit_order) != unit_keys:
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has unit_order entries that "
            "do not match unit_keys; remove it and rebuild."
        )

    per_unit_fields = (
        "source_event_keys",
        "unit_time0_ctime",
        "unit_time0_fpga",
        "unit_event_id",
        "unit_delta_time",
        "archive_version",
        "unit_git_version_tag",
        "unit_input_map_sha256",
        "unit_collection_server",
        "unit_scope",
    )
    for name in per_unit_fields:
        size = int(np.asarray(data[name]).reshape(-1).size)
        if size != len(unit_order):
            raise SystemExit(
                f"pilot-proxy-detector: product {path} field {name!r} has "
                f"{size} entries but unit_order has {len(unit_order)}; remove "
                "it and rebuild."
            )
    try:
        saved_sample_rate = np.asarray(
            data["sample_rate_hz"], dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has an invalid "
            "sample_rate_hz; remove it and rebuild."
        ) from exc
    if (
        saved_sample_rate.size != 1
        or not np.isfinite(saved_sample_rate[0])
        or saved_sample_rate[0] <= 0.0
    ):
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has an invalid "
            "sample_rate_hz; remove it and rebuild."
        )
    try:
        finite_delta_time = np.asarray(
            data["unit_delta_time"], dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has invalid "
            "unit_delta_time values; remove it and rebuild."
        ) from exc
    finite_delta_time = finite_delta_time[
        np.isfinite(finite_delta_time) & (finite_delta_time > 0.0)
    ]
    if finite_delta_time.size and not np.allclose(
        finite_delta_time,
        1.0 / float(saved_sample_rate[0]),
        rtol=1e-12,
        atol=0.0,
    ):
        raise SystemExit(
            f"pilot-proxy-detector: product {path} sample_rate_hz disagrees "
            "with unit_delta_time; remove it and rebuild."
        )
    expected_event_keys = [
        source_event_key(key, int(np.asarray(data["freq_id"]).reshape(-1)[0]))
        for key in unit_order
    ]
    saved_event_keys = [
        str(value) for value in np.asarray(data["source_event_keys"]).reshape(-1)
    ]
    if saved_event_keys != expected_event_keys:
        raise SystemExit(
            f"pilot-proxy-detector: product {path} source_event_keys are not "
            "aligned with unit_order; remove it and rebuild."
        )

    frame_index = np.asarray(data["frame_index"], dtype=np.int64).reshape(-1)
    frame_count = int(frame_index.size)
    if not np.array_equal(frame_index, np.arange(frame_count, dtype=np.int64)):
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has a non-contiguous "
            "frame_index; remove it and rebuild."
        )
    per_frame_fields = (
        "p_target_u64",
        "p_ref_sum_u64",
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "normalized_pilot_excess",
        "reject_mask",
        "valid",
        "baseband_power_linear",
        "fine_power_u64",
        "psd_frame_db_i16",
        "psd_db_reference",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "frame_unit_index",
        "frame_in_unit",
    )
    for name in per_frame_fields:
        values = np.asarray(data[name])
        size = int(values.shape[0]) if values.ndim else 1
        if size != frame_count:
            raise SystemExit(
                f"pilot-proxy-detector: product {path} field {name!r} has "
                f"{size} frame rows but frame_index has {frame_count}; remove "
                "it and rebuild."
            )

    fine = np.asarray(data["fine_power_u64"])
    fine_bins = int(np.asarray(data["fine_num_bins"]).reshape(()).item())
    if fine.ndim != 3 or fine.shape[2] != fine_bins:
        raise SystemExit(
            f"pilot-proxy-detector: product {path} fine_power_u64 shape does "
            "not match fine_num_bins; remove it and rebuild."
        )
    nfft = int(np.asarray(data["nfft"]).reshape(()).item())
    for name in (
        "integrated_spectrum_before_mask",
        "integrated_spectrum_after_mask",
    ):
        if np.asarray(data[name]).reshape(-1).size != nfft:
            raise SystemExit(
                f"pilot-proxy-detector: product {path} field {name!r} does not "
                "match nfft; remove it and rebuild."
            )

    unit_index = np.asarray(data["frame_unit_index"], dtype=np.int64).reshape(-1)
    frame_in_unit = np.asarray(data["frame_in_unit"], dtype=np.int64).reshape(-1)
    if np.any(frame_in_unit < 0):
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has negative frame_in_unit "
            "entries; remove it and rebuild."
        )
    if np.any(unit_index < 0) or np.any(unit_index >= len(unit_order)):
        raise SystemExit(
            f"pilot-proxy-detector: product {path} has frame_unit_index entries "
            "outside unit_order; remove it and rebuild."
        )
    return unit_order, unit_keys


class PilotProxyDetectorAnalyzer(Analyzer):
    """Per-pilot CHIME local-reference power-ratio detector, parity with PilotProxy's batch runner."""

    requires_in_order = True

    info = PluginInfo(
        name="pilot-proxy-detector",
        kind="analyzer",
        summary="CHIME DTV local-reference power-ratio detector (PilotProxy parity; per-channel "
                "fixed-point pilot detection + positive-excess mask).",
        status=EXPERIMENTAL,  # a registry value, kept as it is; the local run of record supplied the real-data parity
        instruments=("chime", "kko", "gbo", "hco"),
        produces="<channel>.npz (chime_detector_outputs schema)",
        requires=("h5py", "pilot-proxy", "GPU+libfstatistic.so (CUDA kernel)"),
        accepts_stream_kinds=(STREAM_PACKED_COMPLEX_INT4_BASEBAND,),
        notes="Use with the 'chime-baseband-packed' reader. Wraps "
              "pack_chime_block_for_detector + detect_packed_detector_input; "
              "detector_fn/kernel/weights injectable via options (CPU ref for tests).",
    )

    def __init__(self) -> None:
        self._nfft = 0
        self._K = int(DETECTOR_WINDOW_SAMPLES)
        self._spectral_sense = SPECTRAL_SENSE_NORMAL
        self._physical_channel = -1
        self._freq_id = -1
        self._f0_hz = 800_000_000.0
        self._sample_rate_hz = float("nan")
        self._pilot_in_band = True
        self._pilot_rf_hz = float("nan")
        self._coarse_center_hz = float("nan")
        # injected detector pieces (defaults resolved lazily in begin)
        self._detector_fn = None
        self._detector_fn_accepts_matched_filter_row_projections = False
        self._kernel = None
        self._weights: np.ndarray | None = None
        # dB-calibration constants (overridable)
        self._pilot_below_data_db = float(PILOT_BELOW_DATA_DB)
        self._bin_enbw_hz = float(EFFECTIVE_BIN_BW_HZ)
        self._dtv_bandwidth_hz = float(DTV_BANDWIDTH_HZ)
        self._pilot_capture_efficiency = float(PILOT_CAPTURE_EFFICIENCY)
        self._max_chunks_per_file: int | None = None
        self._num_input_streams = 0
        self._detector_contract: dict[str, Any] = {}
        self._detector_contract_json = "{}"
        self._weights_hash = ""
        self._weight_bank_sha256 = ""
        self._weight_manifest_sha256 = ""
        self._weight_coefficients_sha256 = ""
        self._reference_placement: dict[str, Any] = {}
        self._detector_version = ""
        # exact integer weight-norm zero-point (set with the weights in begin())
        self._target_norm_sq = 0
        self._reference_norm_sum_sq = 0
        self._null_power_ratio = float("nan")
        self._resumed_provenance: dict[str, Any] | None = None
        # accumulators
        self._p_target: list[int] = []
        self._p_ref_sum: list[int] = []
        # Kept separately: the sum alone hides an asymmetric reference.
        self._p_ref_lower: list[int] = []
        self._p_ref_upper: list[int] = []
        self._coarse_power_ratio: list[float] = []
        # Fine-diagnostic products in the current per-pilot contract.
        self._fine_mode_opt: str = "auto"
        self._fine_supported: bool | None = None
        self._fine_status: str = "not_configured"
        self._resumed_fine_retention: bool | None = None
        self._fine_bins: int = 0
        self._fine_p_fa: float = float(CFAR_DEFAULT_P_FA)
        self._fine_guard: int = int(CFAR_DEFAULT_GUARD_FINE_BINS)
        self._fine_designated: list[int] = [0]
        self._fine_designated_resumed: bool = False
        self._fine_census: list[int] = []
        # Exact uint64 [3, FINE_BINS] terms from the frozen fxfft256 --
        # the deployed fine statistic's numerators and denominators.
        self._fine_power_u64: list[np.ndarray] = []
        self._normalized_coarse_power_ratio_db: list[float] = []
        self._pilot_excess_db: list[float] = []
        self._estimated_data_shelf_snr_db: list[float] = []
        self._normalized_pilot_excess: list[float] = []  # F/null_power_ratio - 1 (NaN if invalid)
        self._reject_mask: list[int] = []  # 1 = discard frame (positive excess)
        self._valid: list[int] = []
        self._baseband_power: list[float] = []
        self._railed_count: list[int] = []
        self._fill_count: list[int] = []
        self._railed_total: list[int] = []
        self._overflow = 0
        self._n_frames = 0
        self._keys: set = set()
        self._unit_order: list[str] = []  # consumption order, for alignment keys
        # integrated power spectra (sum over valid frames of |FFT|^2 summed over
        # feeds); rectangular window. Allocated in begin() once nfft + xp are known
        # (None here so resume() can seed them before begin() moves them onto xp).
        self._xp = np  # replaced in begin() with cupy when a GPU is present
        # Per-frame PSD, feeds summed, encoded on arrival: int16 dB codes
        # plus that frame's own reference level. Holding float64 here cost
        # 4.6 GB on the largest channel.
        self._psd_codes: list[np.ndarray] = []
        self._psd_ref: list[float] = []
        self._spec_before = None  # xp.float64[nfft]; valid frames
        self._spec_after = None   # xp.float64[nfft]; valid AND kept (not rejected)
        # per-frame unit tags -> absolute time: t[f] = unit_time0_ctime[u]
        #   + frame_in_unit[f] * nfft * unit_delta_time[u],  u = frame_unit_index[f]
        self._frame_unit_index: list[int] = []
        self._frame_in_unit: list[int] = []
        # per-unit (per-file) time axis + provenance, aligned to _unit_order
        self._unit_time0_ctime: list[float] = []
        self._unit_time0_fpga: list[int] = []
        self._unit_event_id: list[int] = []
        self._unit_delta_time: list[float] = []
        self._unit_archive_version: list[str] = []
        self._unit_git_version_tag: list[str] = []
        self._unit_input_map_sha256: list[str] = []
        self._unit_collection_server: list[str] = []
        # Acquisition scope from the source; local inputs declare "local".
        self._unit_scope: list[str] = []
    # -- selection / fan-out (per CHIME coarse channel / freq_id) -----------
    def resolve_selection(self, ctx: RunContext, spec: Any) -> Any:
        # --select is CHIME freq_id (coarse-channel indices), the namespace the
        # CADC inventory and filenames key on. One freq_id is one pilot; the
        # product is labeled with the ATSC channel that pilot falls in.
        if spec is None:
            return None
        if isinstance(spec, (list, tuple)):
            return [int(s) for s in spec]
        out: list[int] = []
        for part in str(spec).split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            elif part:
                out.append(int(part))
        return out

    def plan_runs(self, ctx: RunContext, spec: Any) -> list:
        sel = self.resolve_selection(ctx, spec)
        if not sel:
            raise ValueError(
                "pilot-proxy-detector requires an explicit freq_id selection "
                "(e.g. select=400 or select='399,400'); it has no 'all' mode, "
                "because one product holds exactly one coarse channel (one pilot) "
                "and an unscoped run would accumulate several channels under the "
                "first file's label."
            )
        dupes = sorted({fid for fid in sel if list(sel).count(fid) > 1})
        if dupes:
            raise ValueError(
                f"pilot-proxy-detector: duplicate freq_id(s) in --select: {dupes}. Each "
                f"coarse channel must appear at most once."
            )
        return [[int(fid)] for fid in sel]

    def resume(self, path: str, ctx: RunContext) -> bool:
        """Reload a checkpointed product so a killed run continues instead of
        restarting from scratch.

        Returns False when there is no prior product (fresh build). Raises
        SystemExit when a prior product exists but was built with incompatible
        parameters, so a capped smoke product is never silently "completed"
        by a full run, and a different channel's product is never resumed into
        this one.
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            with np.load(p, allow_pickle=False) as npz:
                data = {k: npz[k] for k in npz.files}
        except Exception as exc:  # unreadable / corrupt checkpoint
            raise SystemExit(
                f"pilot-proxy-detector: cannot read existing product {path} for "
                f"resume ({type(exc).__name__}: {exc}). Remove it to rebuild "
                f"from scratch, or point --output-dir at a clean directory."
            )

        _validate_resume_product(data, path)
        saved_unit_order, saved_unit_keys = _validated_resume_axes(data, path)

        # -- compatibility guards (refuse; never silently overwrite/complete) --
        saved_schema = str(data["schema_version"].item())
        if saved_schema != PER_PILOT_PRODUCT_SCHEMA_TOKEN:
            raise SystemExit(
                f"pilot-proxy-detector: product {path} has schema_version "
                f"{saved_schema!r}, but this build writes "
                f"{PER_PILOT_PRODUCT_SCHEMA_TOKEN!r}. "
                f"Remove it to rebuild."
            )
        saved_cap = int(data["max_chunks_per_file"])
        req = (ctx.options or {}).get("max_chunks_per_file", None)
        req_cap = -1 if req is None else int(req)
        if saved_cap != req_cap:
            _s = saved_cap if saved_cap >= 0 else None
            _r = req_cap if req_cap >= 0 else None
            raise SystemExit(
                f"pilot-proxy-detector: product {path} was built with "
                f"max_chunks_per_file={_s}, but this run requests {_r}. A capped "
                f"product cannot be completed by a different cap; use a clean "
                f"--output-dir."
            )
        saved_fid = int(data["freq_id"][0])
        sel = list(getattr(ctx, "selection", None) or [])
        if sel and int(sel[0]) != saved_fid:
            raise SystemExit(
                f"pilot-proxy-detector: product {path} is freq_id {saved_fid}, but "
                f"--select requests {int(sel[0])}. Refusing to resume a "
                f"different channel's product into this one."
            )

        # -- restore identity / calibration (begin() re-derives, must agree) ---
        self._freq_id = saved_fid
        self._physical_channel = int(data["physical_channel"][0])
        self._pilot_in_band = bool(int(data["pilot_in_band"][0]))
        self._pilot_rf_hz = float(data["pilot_frequency_hz"][0])
        self._coarse_center_hz = float(data["chime_frequency_hz"][0])
        self._nfft = int(data["nfft"])
        self._sample_rate_hz = float(
            np.asarray(data["sample_rate_hz"]).reshape(()).item()
        )
        self._K = int(data["detector_window_samples"])
        self._spectral_sense = (
            SPECTRAL_SENSE_INVERTED if int(data["sense"]) == -1
            else SPECTRAL_SENSE_NORMAL
        )
        self._pilot_below_data_db = float(data["pilot_below_data_db"])
        self._bin_enbw_hz = float(data["bin_enbw_hz"])
        self._dtv_bandwidth_hz = float(data["dtv_bandwidth_hz"])
        self._pilot_capture_efficiency = float(data["pilot_capture_efficiency"])
        self._max_chunks_per_file = None if saved_cap < 0 else saved_cap

        self._num_input_streams = int(
            np.asarray(data["num_input_streams"]).reshape(()).item()
        )
        if self._num_input_streams <= 0:
            raise SystemExit(
                "pilot-proxy-detector: existing product has an invalid num_input_streams; "
                "remove it and rebuild."
            )

        def _text(name: str) -> str:
            return str(np.asarray(data[name]).reshape(()).item())

        try:
            saved_contract = json.loads(_text("detector_contract_json"))
            saved_reference = json.loads(_text("reference_placement_json"))
            saved_decision_contract = json.loads(_text("decision_contract_json"))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"pilot-proxy-detector: invalid resume provenance JSON in {path}: {exc}"
            ) from exc
        if saved_decision_contract != current_decision_contract():
            raise SystemExit(
                "pilot-proxy-detector: existing product decision contract does not "
                "match the current active/diagnostic decision semantics. Delete "
                "the product and rebuild before resuming."
            )
        self._resumed_provenance = {
            "weights_hash": _text("weights_hash"),
            "detector_version": _text("detector_version"),
            "mask_rule": _text("mask_rule"),
            "target_norm_sq": int(
                np.asarray(data["target_norm_sq"]).reshape(-1)[0]
            ),
            "reference_norm_sum_sq": int(
                np.asarray(data["reference_norm_sum_sq"]).reshape(-1)[0]
            ),
            "detector_contract": saved_contract,
            "weight_bank_sha256": _text("weight_bank_sha256"),
            "weight_manifest_sha256": _text("weight_manifest_sha256"),
            "reference_placement": saved_reference,
        }

        # -- restore the per-frame accumulator, in stored order ----------------
        def _col(name: str) -> np.ndarray:
            return np.asarray(data[name]).reshape(-1)
        self._p_target = [int(x) for x in _col("p_target_u64")]
        self._p_ref_sum = [int(x) for x in _col("p_ref_sum_u64")]
        self._p_ref_lower = [int(x) for x in _col("p_ref_lower_u64")]
        self._p_ref_upper = [int(x) for x in _col("p_ref_upper_u64")]
        self._coarse_power_ratio = [float(x) for x in _col("coarse_power_ratio")]
        self._normalized_coarse_power_ratio_db = [float(x) for x in _col("normalized_coarse_power_ratio_db")]
        self._pilot_excess_db = [float(x) for x in _col("pilot_excess_db")]
        self._estimated_data_shelf_snr_db = [float(x) for x in _col("estimated_data_shelf_snr_db")]
        self._normalized_pilot_excess = [
            float(x) for x in _col("normalized_pilot_excess")
        ]
        self._reject_mask = [int(x) for x in _col("reject_mask")]
        # Restore the current fine-diagnostic arrays; current checkpoints carry
        # these fields even when there are zero frames.
        self._fine_status = str(np.asarray(data["fine_status"]).reshape(()).item())
        _fine_u64 = np.asarray(data["fine_power_u64"], dtype=np.uint64)
        legacy_empty_fine = (
            self._fine_status in FINE_STATUS_WITHOUT_TERMS
            and _fine_u64.ndim == 3
            and _fine_u64.shape[1] == REFERENCE_WEIGHT_TERMS
            and _fine_u64.shape[2] > 0
        )
        if legacy_empty_fine:
            self._fine_bins = 0
        elif _fine_u64.ndim == 3:
            self._fine_bins = int(_fine_u64.shape[2])
        else:
            self._fine_bins = 0
        psd_codes = np.asarray(data["psd_frame_db_i16"], dtype=np.int16)
        psd_ref = np.asarray(data["psd_db_reference"], dtype=np.float64)
        if psd_codes.ndim == 2 and psd_codes.shape[1]:
            self._psd_codes = [psd_codes[i] for i in range(psd_codes.shape[0])]
            self._psd_ref = [float(x) for x in psd_ref.reshape(-1)]
        else:
            self._psd_codes = []
            self._psd_ref = []
        self._fine_power_u64 = (
            []
            if legacy_empty_fine
            else [_fine_u64[i] for i in range(_fine_u64.shape[0])]
        )
        self._fine_p_fa = float(np.asarray(data["fine_p_fa"]))
        self._fine_guard = int(np.asarray(data["fine_guard_fine_bins"]))
        self._fine_designated = [int(x) for x in _col("fine_designated_bins")]
        # The resumed designation is part of the product's decision history;
        # begin() must not replace it with a freshly predicted default, or the
        # resumed-provenance contract check would refuse its own checkpoint.
        self._fine_designated_resumed = True
        self._fine_census = [
            int(x) for x in _col("fine_census_excluded_bins")
        ]
        if self._fine_status in {"enabled", "pending"}:
            self._resumed_fine_retention = True
        elif self._fine_status != "not_applicable_pilot_out_of_band":
            self._resumed_fine_retention = False
        self._valid = [int(x) for x in _col("valid")]
        self._baseband_power = [float(x) for x in _col("baseband_power_linear")]
        self._railed_count = [int(x) for x in _col("railed_sample_count")]
        self._fill_count = [int(x) for x in _col("fill_sample_count")]
        self._railed_total = [int(x) for x in _col("railed_sample_total")]
        self._frame_unit_index = [int(x) for x in _col("frame_unit_index")]
        self._frame_in_unit = [int(x) for x in _col("frame_in_unit")]
        self._n_frames = len(self._p_target)
        self._overflow = int(data["rational_overflow_count"])
        # integrated spectra: restore on host (1-D float64); begin() moves them onto
        # the FFT backend (cupy/numpy) and resumes accumulation there.
        self._spec_before = np.asarray(
            data["integrated_spectrum_before_mask"], dtype=np.float64
        ).reshape(-1)
        self._spec_after = np.asarray(
            data["integrated_spectrum_after_mask"], dtype=np.float64
        ).reshape(-1)

        # -- restore processed keys + consumption order ------------------------
        self._keys = saved_unit_keys
        self._unit_order = saved_unit_order
        # -- restore the per-unit time axis (aligned to unit_order) ------------
        self._unit_time0_ctime = [float(x) for x in _col("unit_time0_ctime")]
        self._unit_time0_fpga = [int(x) for x in _col("unit_time0_fpga")]
        self._unit_event_id = [int(x) for x in _col("unit_event_id")]
        self._unit_delta_time = [float(x) for x in _col("unit_delta_time")]
        self._unit_archive_version = [str(x) for x in _col("archive_version")]
        self._unit_git_version_tag = [str(x) for x in _col("unit_git_version_tag")]
        self._unit_input_map_sha256 = [str(x) for x in _col("unit_input_map_sha256")]
        self._unit_collection_server = [str(x) for x in _col("unit_collection_server")]
        self._unit_scope = [str(x) for x in _col("unit_scope")]

        # let begin() verify the new input's identity matches what we restored
        self._resumed_identity = (
            self._freq_id, int(self._nfft), self._sample_rate_hz,
            int(self._K), self._spectral_sense,
            self._physical_channel, self._pilot_below_data_db,
            self._bin_enbw_hz, self._dtv_bandwidth_hz,
            self._pilot_capture_efficiency,
        )
        return True

    def processed_keys(self) -> set:
        return set(self._keys)

    def processed_key_order(self) -> list[str]:
        return list(self._unit_order)

    @staticmethod
    def _fine_retention_resolution(
        opts: Mapping[str, Any],
        *,
        detector_fn: Any = None,
        kernel: Any = None,
    ) -> dict[str, str]:
        requested = str(opts.get("fine_products", "auto")).lower()
        if requested not in {"auto", "on", "off"}:
            raise ValueError(
                "detector analyzer: fine_products must be auto|on|off, got "
                f"{requested!r}."
            )
        if requested == "off":
            return {"requested": requested, "resolved": "disabled_by_option"}

        if detector_fn is None:
            detector_fn = opts.get("detector_fn")
        if detector_fn is None:
            detector_fn = detect_packed_detector_input

        if kernel is None:
            kernel = opts.get("kernel")
        if kernel is None:
            from pilot_proxy.kernel import FStatKernel
            from pilot_proxy.paths import DEFAULT_LIB_PATH

            kernel = FStatKernel(opts.get("lib_path", DEFAULT_LIB_PATH))

        detector_supports = _callable_accepts_keyword(
            detector_fn, "emit_row_projections"
        )
        probe = getattr(kernel, "supports_row_projections", None)
        kernel_supports = bool(probe()) if callable(probe) else False
        if kernel_supports and detector_supports:
            resolved = "enabled"
        elif kernel_supports:
            resolved = "detector_fn_lacks_matched_filter_row_projections"
        else:
            resolved = "kernel_library_lacks_matched_filter_row_projections"
        return {"requested": requested, "resolved": resolved}

    def fine_retention_scope(self, ctx: RunContext) -> dict[str, str]:
        return self._fine_retention_resolution(dict(ctx.options or {}))

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        """Report detector runtime problems before a long scan starts."""
        import importlib.util
        problems: list[str] = []
        opts = dict(ctx.options or {})
        # A CPU/test detector_fn replaces the GPU kernel, so it needs no cupy.
        if "detector_fn" not in opts and importlib.util.find_spec("cupy") is None:
            problems.append(
                "cupy/CUDA is not importable; pilot-proxy-detector requires a GPU runtime"
            )
        try:
            from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH
            lib_path = Path(opts.get("lib_path", DEFAULT_LIB_PATH))
            weights_path = Path(opts.get("weights_path", DEFAULT_WEIGHTS_PATH))
            if "kernel" not in opts and not lib_path.exists():
                problems.append(f"CUDA detector library not found: {lib_path}")
            if (
                "weights" not in opts
                and "weights_by_channel" not in opts
                and not weights_path.exists()
            ):
                problems.append(f"detector weight bank not found: {weights_path}")
        except Exception as exc:  # noqa: BLE001 - preflight should report rather than crash.
            problems.append(f"detector preflight failed: {type(exc).__name__}: {exc}")
        try:
            fine_retention = self.fine_retention_scope(ctx)
            if (
                fine_retention["requested"] == "on"
                and fine_retention["resolved"] != "enabled"
            ):
                problems.append(
                    "fine_products=on requires matched-filter row projections; "
                    f"resolved status is {fine_retention['resolved']}"
                )
        except Exception as exc:  # noqa: BLE001 - preflight reports all failures.
            problems.append(
                "fine-product preflight failed: "
                f"{type(exc).__name__}: {exc}"
            )
        return (not problems), problems

    # -- lifecycle ----------------------------------------------------------
    def begin(self, ctx: RunContext, first_meta: Mapping[str, Any]) -> None:
        inst = ctx.instrument
        opts = dict(ctx.options or {})
        self._nfft = int(getattr(inst, "nfft", 0) or first_meta.get("nfft") or 0)
        if self._nfft <= 0:
            raise ValueError("detector analyzer: nfft must come from the instrument")
        instrument_sample_rate_hz = float(getattr(inst, "fs_hz", 0.0) or 0.0)
        metadata_sample_rate_hz = float(first_meta.get("sample_rate_hz", 0.0) or 0.0)
        self._sample_rate_hz = (
            instrument_sample_rate_hz
            if instrument_sample_rate_hz > 0.0
            else metadata_sample_rate_hz
            if metadata_sample_rate_hz > 0.0
            else float(SAMPLE_RATE_HZ)
        )
        # Instrument definitions expose spectral sense as nyquist_zone.
        # not a "sense" attribute. nyquist_sign maps that to +1/-1.
        sense = int(nyquist_sign(int(getattr(inst, "nyquist_zone", 1) or 1)))
        self._spectral_sense = (
            SPECTRAL_SENSE_INVERTED if sense == -1 else SPECTRAL_SENSE_NORMAL
        )

        # pilot / channel geometry from the channel-center frequency
        f_center_hz = float(first_meta["f_center_hz"])
        self._coarse_center_hz = f_center_hz
        meta_streams = first_meta.get("num_input_streams")
        if meta_streams is not None:
            meta_count = int(meta_streams)
            if meta_count <= 0:
                raise ValueError("detector analyzer: num_input_streams must be positive")
            if self._num_input_streams not in (0, meta_count):
                raise SystemExit(
                    "pilot-proxy-detector: resumed product input-stream count does not "
                    f"match new input ({self._num_input_streams} != {meta_count})."
                )
            self._num_input_streams = meta_count
        # The detector also emits explicit invalid rows for selected receiver
        # channels whose nearest ATSC pilot lies outside this coarse bin.  Its
        # 3-MHz lookup is label-only; HDF5 discovery uses the strict half-bin
        # default and therefore never groups neighboring freq_ids.
        physical_channel = nearest_atsc_physical_channel(
            f_center_hz, tolerance_hz=3.0e6
        )
        if physical_channel is None:
            raise ValueError(
                f"detector analyzer: coarse center {f_center_hz:.0f} Hz is not near "
                "any ATSC pilot"
            )
        self._physical_channel = int(physical_channel)
        self._f0_hz = float(getattr(inst, "f0_mhz", 800.0)) * 1e6
        self._freq_id = chime_freq_id_from_hz(f_center_hz, self._f0_hz)
        self._pilot_rf_hz = float(physical_channel_to_pilot_hz(self._physical_channel))
        # Validate the *first* file against the requested freq_id. begin() fixes
        # the product identity here, and the per-file guard only protects later
        # files against this. A mislabeled filename or wrong inventory record would
        # otherwise silently redefine the product's freq_id.
        sel = list(getattr(ctx, "selection", None) or [])
        if sel:
            requested = int(sel[0])
            if self._freq_id != requested:
                raise ValueError(
                    f"detector analyzer: --select requested freq_id {requested}, but "
                    f"the first file's center {f_center_hz:.0f} Hz implies freq_id "
                    f"{self._freq_id}. Refusing to build a {self._freq_id} product "
                    f"under a {requested} request; check the file naming and inventory."
                )
        # In-band test: the detector's weights target the pilot bin *inside* this
        # coarse channel. If the pilot is more than fs/2 from center it lives in a
        # different coarse channel (a wrong --select freq_id), so the detection is
        # meaningless. Flag it; consume_file then emits an explicitly invalid
        # product (mask=0, valid=0) instead of fabricating detections.
        _coarse_width = self._sample_rate_hz
        _expected_offset = self._pilot_rf_hz - f_center_hz
        self._pilot_in_band = abs(_expected_offset) < (_coarse_width / 2.0)
        if not self._pilot_in_band:
            warnings.warn(
                f"detector analyzer: freq_id {self._freq_id} (center "
                f"{f_center_hz / 1e6:.4f} MHz) does not contain ATSC ch"
                f"{self._physical_channel}'s pilot; the nominal offset "
                f"{_expected_offset / 1e3:.0f} kHz exceeds +/-{_coarse_width / 2e3:.0f} "
                f"kHz. Emitting an all-invalid product (no in-band pilot); pick the "
                f"freq_id whose center is within fs/2 of the pilot.",
                RuntimeWarning,
                stacklevel=2,
            )

        # detector backend: injectable, mirroring run_chime_analysis
        self._fine_p_fa = float(opts.get("fine_p_fa", CFAR_DEFAULT_P_FA))
        self._fine_guard = int(
            opts.get("fine_guard_fine_bins", CFAR_DEFAULT_GUARD_FINE_BINS)
        )
        self._fine_designated_from_opts = "fine_designated_bins" in opts
        if self._fine_designated_from_opts:
            self._fine_designated = [
                int(b) for b in opts["fine_designated_bins"]
            ]
        elif not getattr(self, "_fine_designated_resumed", False):
            self._fine_designated = [0]
        # resumed without an explicit option: keep the checkpoint's designation
        self._fine_designated_half_width = int(
            opts.get("fine_designated_half_width_bins",
                     DEFAULT_FINE_DESIGNATED_HALF_WIDTH_BINS)
        )
        self._fine_census = [
            int(b) for b in opts.get("fine_census_excluded_bins", [])
        ]
        self._detector_fn = opts.get("detector_fn")
        if self._detector_fn is None:
            self._detector_fn = detect_packed_detector_input
        self._detector_fn_accepts_matched_filter_row_projections = _callable_accepts_keyword(
            self._detector_fn, "emit_row_projections"
        )

        self._kernel = opts.get("kernel")
        if self._kernel is None:
            from pilot_proxy.kernel import FStatKernel
            from pilot_proxy.paths import DEFAULT_LIB_PATH
            self._kernel = FStatKernel(opts.get("lib_path", DEFAULT_LIB_PATH))
        fine_retention = self._fine_retention_resolution(
            opts,
            detector_fn=self._detector_fn,
            kernel=self._kernel,
        )
        self._fine_mode_opt = fine_retention["requested"]
        resolved_fine_status = fine_retention["resolved"]
        if self._fine_mode_opt == "on" and resolved_fine_status != "enabled":
            raise RuntimeError(
                "detector analyzer: fine_products=on but the configured "
                "kernel/detector backend does not explicitly support row-sum "
                f"emission ({resolved_fine_status})."
            )
        fine_enabled = resolved_fine_status == "enabled"
        if (
            self._resumed_fine_retention is not None
            and self._resumed_fine_retention != fine_enabled
        ):
            raise SystemExit(
                "pilot-proxy-detector: resumed product fine-product retention "
                "mode differs from the current backend or fine_products option. "
                "Restore the original setting and capability, or use a clean "
                "--output-dir."
            )
        self._fine_supported = fine_enabled
        self._fine_status = (
            "not_applicable_pilot_out_of_band"
            if not self._pilot_in_band
            else "pending"
            if fine_enabled
            else resolved_fine_status
        )
        self._K = int(getattr(getattr(self._kernel, "specs", None), "K",
                              DETECTOR_WINDOW_SAMPLES))
        if int(self._nfft) % int(self._K) != 0:
            raise ValueError("detector analyzer: nfft must be divisible by kernel K")

        # Default designated bins: the *predicted* pilot line, not fine bin 0.
        # The carrier lands at the coarse-grid quantization residual in the
        # envelope spectrum, so a bin-0 designation watches an empty bin and
        # leaves the line in the CFAR null bulk. An explicit
        # fine_designated_bins option always wins; the out-of-band case keeps
        # [0] because the product is emitted all-invalid anyway.
        if (not self._fine_designated_from_opts
                and not getattr(self, "_fine_designated_resumed", False)
                and self._pilot_in_band):
            _fine_bins = fine_bin_count(int(self._nfft) // int(self._K))
            _predicted = predicted_pilot_fine_bin(
                pilot_rf_hz=self._pilot_rf_hz,
                coarse_center_hz=self._coarse_center_hz,
                sample_rate_hz=self._sample_rate_hz,
                detector_window_samples=self._K,
                nfft=self._nfft,
                spectral_sense=self._spectral_sense,
                pad_factor=FINE_PAD_FACTOR,
            )
            self._fine_designated = predicted_fine_designated_bins(
                _predicted, self._fine_designated_half_width, _fine_bins
            )

        bank = None
        weights = opts.get("weights")
        if weights is None:
            by_channel = opts.get("weights_by_channel")
            if by_channel is not None:
                weights = by_channel.get(self._physical_channel)
        if weights is None:
            from pilot_proxy.detector_weights import DetectorWeightBank
            from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH
            bank = DetectorWeightBank(
                explicit_path=opts.get("weights_path", DEFAULT_WEIGHTS_PATH),
                expected_kernel=getattr(self._kernel, "specs", None),
            )
            weights, valid = bank.get_weights_for_physical_channel(
                self._physical_channel,
                tolerance_hz=float(
                    opts.get("pilot_frequency_tolerance_hz",
                             _DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ)
                ),
            )
            if weights is None or not valid:
                raise ValueError(
                    f"detector analyzer: no valid weights for channel "
                    f"{self._physical_channel}"
                )
        self._weights = np.ascontiguousarray(weights)
        # Exact integer squared norms of the three weight terms. null_power_ratio =
        # 2*nt/(nl+nu) is the flat-floor H0 zero-point of F that int4 weight
        # quantization shifts away from 1; the mask rule and the corrected
        # pilot excess divide it out (see detector_contract).
        _nt, _nl, _nu = weight_term_norms_sq(self._weights)
        self._target_norm_sq = int(_nt)
        self._reference_norm_sum_sq = int(_nl + _nu)
        self._null_power_ratio = (
            null_power_ratio_from_weight_norms(self._target_norm_sq, self._reference_norm_sum_sq)
            if self._reference_norm_sum_sq > 0
            else float("nan")
        )

        time_reverse = self._spectral_sense == SPECTRAL_SENSE_INVERTED
        default_wc = (
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE if time_reverse
            else WEIGHT_COORDINATE_RAW_INPUT
        )
        if bank is not None:
            # Reuse the standalone runner's manifest/runtime coordinate validation.
            from pilot_proxy.chime.runner import (
                _reference_placement_summary,
                _weight_coordinate_metadata,
            )
            coordinate_metadata = _weight_coordinate_metadata(
                weight_bank=bank,
                input_spectral_sense=self._spectral_sense,
            )
            contract_weight_coordinate = str(
                coordinate_metadata["effective_weight_coordinate_system"]
            )
            if "weight_coordinate_system" in opts:
                requested_coordinate = normalize_weight_coordinate_system(
                    opts["weight_coordinate_system"]
                )
                if requested_coordinate != contract_weight_coordinate:
                    raise ValueError(
                        "Requested weight_coordinate_system disagrees with the "
                        f"weight manifest: {requested_coordinate!r} != "
                        f"{contract_weight_coordinate!r}."
                    )
            self._reference_placement = (
                _reference_placement_summary(
                    getattr(bank, "manifest", None),
                    [self._physical_channel],
                )
                or {}
            )
            self._weight_bank_sha256 = file_sha256(bank.path) or ""
            try:
                from pilot_proxy.detector_weights import (
                    read_header_and_weights,
                )

                _hdr, _coeffs = read_header_and_weights(Path(bank.path))
                self._weight_coefficients_sha256 = hashlib.sha256(
                    np.ascontiguousarray(_coeffs).tobytes()
                ).hexdigest()
            except (OSError, ValueError):
                # Never fail a scan over a provenance convenience field.
                self._weight_coefficients_sha256 = ""
            manifest_path = sidecar_manifest_path(bank.path)
            self._weight_manifest_sha256 = file_sha256(manifest_path) or ""
        else:
            contract_weight_coordinate = normalize_weight_coordinate_system(
                opts.get("weight_coordinate_system", default_wc)
            )
            self._reference_placement = {}
            self._weight_bank_sha256 = ""
            self._weight_manifest_sha256 = ""
            self._weight_coefficients_sha256 = ""

        # dB-calibration constants (overridable)
        self._pilot_below_data_db = float(opts.get("pilot_below_data_db", PILOT_BELOW_DATA_DB))
        self._bin_enbw_hz = float(opts.get("bin_enbw_hz", EFFECTIVE_BIN_BW_HZ))
        self._dtv_bandwidth_hz = float(opts.get("dtv_bandwidth_hz", DTV_BANDWIDTH_HZ))
        self._pilot_capture_efficiency = float(
            opts.get("pilot_capture_efficiency", PILOT_CAPTURE_EFFICIENCY)
        )
        self._max_chunks_per_file = opts.get("max_chunks_per_file", None)

        # Detector contract, so the combine step can emit run_config/stats that
        # validate-products accepts. Built from the kernel specs + spectral sense
        # the same way run_chime_analysis builds it; the weight coordinate system
        # defaults to raw-input and is overridable via options.
        specs = getattr(self._kernel, "specs", None)
        spec_dict: dict = {}
        ser = getattr(specs, "as_descriptive_dict", None)
        if callable(ser):
            raw = ser()
            if isinstance(raw, Mapping):
                spec_dict = dict(raw)

        def _spec(name: str, attr: str, default: int) -> int:
            if name in spec_dict and spec_dict[name] is not None:
                return int(spec_dict[name])
            return int(getattr(specs, attr, default) or default)

        ref_off = _spec("reference_offset_bins", "reference_offset_bins", 2)
        try:
            self._detector_contract = build_detector_contract(
                detector_window_samples=int(self._K),
                skipped_guard_bins=max(0, ref_off - 1),
                reference_offset_bins=ref_off,
                num_weight_terms=_spec("num_weight_terms", "N", 3),
                sample_bits_per_component=_spec(
                    "sample_bits_per_component", "bits", 4
                ),
                weight_coordinate_system=contract_weight_coordinate,
                time_reverse_detector_windows_before_kernel=time_reverse,
            )
        except ValueError as exc:
            raise SystemExit(
                "pilot-proxy-detector: detector_contract is incompatible with "
                f"the current detector ({exc})."
            ) from exc

        # If this run resumed a prior product, the identity begin() just derived
        # from the first NEW file must match the restored one. Otherwise new
        # frames would be appended under a different channel/kernel geometry (or
        # dB calibration) than the existing ones, silently corrupting the
        # product. Refuse rather than mix.
        prev = getattr(self, "_resumed_identity", None)
        if prev is not None:
            now = (
                self._freq_id, int(self._nfft), self._sample_rate_hz,
                int(self._K),
                self._spectral_sense, self._physical_channel,
                self._pilot_below_data_db, self._bin_enbw_hz,
                self._dtv_bandwidth_hz, self._pilot_capture_efficiency,
            )
            if now != prev:
                raise SystemExit(
                    "pilot-proxy-detector: resumed product identity does not match the "
                    "identity derived from new input (instrument / kernel / "
                    "--select / sample rate / dB-calibration changed between runs). "
                    "Use a clean "
                    f"--output-dir to rebuild. restored={prev} new={now}"
                )

        # -- integrated-spectrum backend + accumulators ------------------------
        # cupy when a GPU runtime is present (keeps the per-frame FFT inside the
        # download-idle GPU time on CANFAR); numpy otherwise and in tests, with
        # identical arithmetic. A fresh run allocates zeros; a resumed run has
        # had host spectra restored by resume(), so move them onto the backend.
        self._xp = _detector_fft_backend()
        if self._spec_before is None:
            self._spec_before = self._xp.zeros(int(self._nfft), dtype=self._xp.float64)
            self._spec_after = self._xp.zeros(int(self._nfft), dtype=self._xp.float64)
        else:  # resumed: arrays restored on host -> move onto the backend
            self._spec_before = self._xp.asarray(self._spec_before)
            self._spec_after = self._xp.asarray(self._spec_after)

        # -- provenance: which weights + implementation produced this product --
        self._refresh_provenance(opts)
        self._validate_resumed_provenance()

    def _ensure_fine_width(self, expected_bins: int) -> None:
        """Pin the fine-spectrum width for this product.

        A resumed product's restored width (when nonzero) must equal the
        width the current configuration would produce; nfft is not part of
        the detector contract, so a geometry change between epochs would
        otherwise surface as a shape error at flush. Zero means no frames
        have fixed the width yet (fresh product or zero-frame checkpoint)
        and is always safe to adopt.
        """
        expected = int(expected_bins)
        if self._fine_bins == 0:
            self._fine_bins = expected
            return
        if int(self._fine_bins) != expected:
            raise SystemExit(
                "pilot-proxy-detector: resumed product has fine spectra of "
                f"width {int(self._fine_bins)} but the current configuration "
                f"produces width {expected}; a product's fine dimensionality "
                "cannot change mid-product. Use a clean --output-dir."
            )

    def _refresh_provenance(self, opts: Mapping[str, Any]) -> None:
        if self._weights is None:
            raise RuntimeError("detector analyzer: weights are not initialized")
        self._weights_hash = hashlib.sha256(
            np.ascontiguousarray(self._weights).tobytes()
        ).hexdigest()
        contract_with_fine = dict(self._detector_contract)
        contract_with_fine["fine_reduction"] = {
            "pad_factor": int(FINE_PAD_FACTOR),
            "cfar_policy": (
                "null_bulk_median_left_side_scale;quantile_fallback;"
                "any_bin_detection;nd_flag_rate_monitor"
            ),
            "p_fa": float(self._fine_p_fa),
            "guard_fine_bins": int(self._fine_guard),
            "designated_bins": [int(b) for b in self._fine_designated],
            "census_excluded_bins": [int(b) for b in self._fine_census],
            "v1_marginal_identity": "exact_int64_enforced_per_frame",
        }
        self._detector_contract_json = json.dumps(
            contract_with_fine,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            import pilot_proxy as _pkg
            version = getattr(_pkg, "__version__", "unknown")
        except Exception:  # pragma: no cover
            version = "unknown"
        kernel_version_obj = getattr(self._kernel, "version", None)
        serializer = getattr(kernel_version_obj, "as_string", None)
        kernel_version = (
            str(serializer()) if callable(serializer) else str(kernel_version_obj or "unknown")
        )
        if "kernel" in opts:
            kernel_sha256 = "injected"
        else:
            from pilot_proxy.paths import DEFAULT_LIB_PATH
            kernel_sha256 = (
                file_sha256(opts.get("lib_path", DEFAULT_LIB_PATH)) or "unavailable"
            )
        self._detector_version = (
            f"pilot-proxy/{version} source={package_source_sha256()} "
            f"kernel={kernel_version} kernel_sha256={kernel_sha256} "
            f"{PER_PILOT_PRODUCT_SCHEMA_TOKEN} K={int(self._K)}"
        )

    def _validate_resumed_provenance(self) -> None:
        saved = self._resumed_provenance
        if saved is None:
            return
        current = {
            "weights_hash": self._weights_hash,
            "detector_version": self._detector_version,
            "mask_rule": NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
            "target_norm_sq": int(self._target_norm_sq),
            "reference_norm_sum_sq": int(self._reference_norm_sum_sq),
            # Compare the enriched contract (including the fine_reduction
            # block) exactly as save() persists it: _refresh_provenance() has
            # just rebuilt _detector_contract_json for the current
            # configuration, and the saved side was json.loads'd from the
            # same serialization at resume().
            "detector_contract": json.loads(self._detector_contract_json),
            "weight_bank_sha256": self._weight_bank_sha256,
            "weight_manifest_sha256": self._weight_manifest_sha256,
            "reference_placement": self._reference_placement,
        }
        mismatches = [
            key for key in current
            if saved.get(key) != current.get(key)
        ]
        # Resume is stricter than cross-pilot combine: appending frames mutates
        # one product, so it requires the exact Python implementation identity
        # (the source= tree hash embedded in detector_version).  Allowing a
        # geometry-equivalent source change here would mix implementations and
        # then relabel every prior frame with only the newest build stamp.
        if mismatches:
            details = "; ".join(
                f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}"
                for key in mismatches
            )
            raise SystemExit(
                "pilot-proxy-detector: resumed product provenance does not match the "
                f"current detector configuration ({details}). Use a clean "
                "--output-dir to rebuild."
            )

    def _check_file_meta(self, meta: Mapping[str, Any]) -> None:
        """Reject any file whose channel/nfft does not match this product.

        begin() fixes the product identity from the first file; every later file
        must agree, or we would silently accumulate multiple channels (or a
        different FFT length) into one product. This turns that into a hard error,
        which is the real guard against a wrong --select or a filename/freq mismatch.
        """
        fc = meta.get("f_center_hz")
        if fc is not None:
            fid = chime_freq_id_from_hz(float(fc), self._f0_hz)
            if fid != self._freq_id:
                raise ValueError(
                    f"detector analyzer: file center {float(fc):.0f} Hz is coarse "
                    f"channel freq_id {fid}, but this product is freq_id "
                    f"{self._freq_id} (ATSC ch{self._physical_channel}). Refusing "
                    f"to mix coarse channels in one product -- check the --select "
                    f"freq_id and the input file naming."
                )
        mnfft = meta.get("nfft")
        if mnfft is not None and int(mnfft) != int(self._nfft):
            raise ValueError(
                f"detector analyzer: file nfft {int(mnfft)} != product nfft "
                f"{int(self._nfft)}."
            )
        streams = meta.get("num_input_streams")
        if (
            streams is not None
            and self._num_input_streams > 0
            and int(streams) != self._num_input_streams
        ):
            raise ValueError(
                "detector analyzer: file input-stream count "
                f"{int(streams)} != product count {self._num_input_streams}."
            )
        if meta.get("unit_key") is not None:
            scope = meta.get("scope")
            if not isinstance(scope, str) or not scope.strip():
                raise ValueError(
                    "detector analyzer: each unit must have a non-empty source scope"
                )
            if scope != "local":
                git_version_tag = meta.get("git_version_tag")
                input_map_sha256 = meta.get("input_map_sha256")
                if not isinstance(git_version_tag, str) or not git_version_tag:
                    raise ValueError(
                        "detector analyzer: archive units require git_version_tag"
                    )
                if (
                    not isinstance(input_map_sha256, str)
                    or len(input_map_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in input_map_sha256)
                ):
                    raise ValueError(
                        "detector analyzer: archive units require a valid "
                        "input_map_sha256"
                    )

    def consume_file(self, arrays: Iterable, meta: Mapping[str, Any]) -> int:
        """Consume one unit atomically, rolling back every accumulator on error."""
        list_fields = (
            "_p_target",
            "_p_ref_sum",
            "_p_ref_lower",
            "_p_ref_upper",
            "_coarse_power_ratio",
            "_psd_codes",
            "_psd_ref",
            "_fine_power_u64",
            "_normalized_coarse_power_ratio_db",
            "_pilot_excess_db",
            "_estimated_data_shelf_snr_db",
            "_normalized_pilot_excess",
            "_reject_mask",
            "_valid",
            "_baseband_power",
            "_railed_count",
            "_fill_count",
            "_railed_total",
            "_frame_unit_index",
            "_frame_in_unit",
            "_unit_order",
            "_unit_time0_ctime",
            "_unit_time0_fpga",
            "_unit_event_id",
            "_unit_delta_time",
            "_unit_archive_version",
            "_unit_git_version_tag",
            "_unit_input_map_sha256",
            "_unit_collection_server",
            "_unit_scope",
        )
        lengths = {name: len(getattr(self, name)) for name in list_fields}
        scalar_state = {
            "_num_input_streams": self._num_input_streams,
            "_overflow": self._overflow,
            "_n_frames": self._n_frames,
            "_fine_bins": self._fine_bins,
            "_fine_supported": self._fine_supported,
            "_fine_status": self._fine_status,
        }
        keys = set(self._keys)
        spec_before = (
            None if self._spec_before is None else self._spec_before.copy()
        )
        spec_after = None if self._spec_after is None else self._spec_after.copy()
        try:
            return self._consume_file_unchecked(arrays, meta)
        except BaseException:
            for name, length in lengths.items():
                del getattr(self, name)[length:]
            for name, value in scalar_state.items():
                setattr(self, name, value)
            self._keys = keys
            self._spec_before = spec_before
            self._spec_after = spec_after
            raise

    def _consume_file_unchecked(
        self, arrays: Iterable, meta: Mapping[str, Any]
    ) -> int:
        if self._weights is None or self._detector_fn is None:
            raise RuntimeError("detector analyzer: begin() was not called")
        self._check_file_meta(meta)
        n = 0
        # this file's position in the per-unit time axis (the unit row is appended
        # at the tail below, so len(_unit_order) is this unit's index); frame_in_unit
        # is the chunk's 0-based *time* position in the file. The packed reader yields
        # contiguous full nfft chunks, so chunk position == time position, giving each
        # frame an absolute time without storing one timestamp per frame.
        unit_idx = len(self._unit_order)
        chunk_in_unit = 0
        for chunk in arrays:
            if self._max_chunks_per_file is not None and n >= int(
                self._max_chunks_per_file
            ):
                break
            arr = np.asarray(chunk)
            if arr.ndim != 2 or arr.shape[0] != self._nfft:
                continue  # ragged/partial frame -> skip
            if arr.dtype != np.uint8:
                raise ValueError(
                    "detector analyzer requires raw uint8 chunks from the "
                    "'chime-baseband-packed' reader; got dtype "
                    f"{arr.dtype!s}. Use `pilot-proxy chime-scan` with the "
                    "'chime-baseband-packed' reader."
                )
            railed_count, fill_count = railed_component_count(arr)
            railed_total = 2 * int(arr.size)
            chunk_streams = int(arr.shape[1])
            if self._num_input_streams == 0:
                self._num_input_streams = chunk_streams
            elif chunk_streams != self._num_input_streams:
                raise ValueError(
                    "detector analyzer: input-stream count changed within a product: "
                    f"{chunk_streams} != {self._num_input_streams}."
                )
            if not self._pilot_in_band:
                # No in-band pilot in this coarse channel: emit an explicitly
                # invalid frame (reject_mask=0, valid=0, zero powers) and skip the
                # kernel, so the product cannot be read as a real detection.
                # valid=0 keeps the frame out of both integrated spectra.
                self._p_target.append(0)
                self._p_ref_sum.append(0)
                self._p_ref_lower.append(0)
                self._p_ref_upper.append(0)
                self._coarse_power_ratio.append(float("nan"))
                self._normalized_coarse_power_ratio_db.append(float("nan"))
                self._pilot_excess_db.append(float("nan"))
                self._estimated_data_shelf_snr_db.append(float("nan"))
                self._normalized_pilot_excess.append(float("nan"))
                self._reject_mask.append(0)
                self._valid.append(0)
                self._baseband_power.append(float("nan"))
                # Saturation is a property of the raw samples, so it is
                # recorded even for a frame the detector cannot use.
                self._railed_count.append(int(railed_count))
                self._fill_count.append(int(fill_count))
                self._railed_total.append(int(railed_total))
                self._frame_unit_index.append(unit_idx)
                self._frame_in_unit.append(chunk_in_unit)
                n += 1
                chunk_in_unit += 1
                continue
            # reader yields native uint8 [nfft, n_feeds]; fstat wants (streams, 1, time)
            native_block = np.ascontiguousarray(arr.T)[:, np.newaxis, :]
            packed = pack_chime_block_for_detector(
                native_block,
                frame_size_samples=self._nfft,
                detector_window_samples=self._K,
                spectral_sense=self._spectral_sense,
                frames_in_chunk=1,
                sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
                selected_coarse_channel=0,   # unused on the native (lossless) path
                physical_channel=self._physical_channel,
            )
            want_fine = self._fine_mode_opt != "off"
            if want_fine and self._fine_supported is None:
                probe = getattr(self._kernel, "supports_row_projections", None)
                kernel_supports_row_projections = bool(probe()) if callable(probe) else False
                self._fine_supported = bool(
                    kernel_supports_row_projections and self._detector_fn_accepts_matched_filter_row_projections
                )
                if not self._fine_supported:
                    if self._fine_mode_opt == "on":
                        raise RuntimeError(
                            "detector analyzer: fine_products=on but the configured "
                            "kernel/detector backend does not explicitly support "
                            "row-sum emission."
                        )
                    self._fine_status = (
                        "detector_fn_lacks_matched_filter_row_projections"
                        if kernel_supports_row_projections
                        else "kernel_library_lacks_matched_filter_row_projections"
                    )
                else:
                    self._fine_status = "enabled"
            emit_fine = bool(want_fine and self._fine_supported)
            if emit_fine:
                self._fine_status = "enabled"
                self._ensure_fine_width(
                    fine_bin_count(int(self._nfft) // int(self._K))
                )
                detection = self._detector_fn(
                    packed=packed.packed,
                    weights=self._weights,
                    kernel=self._kernel,
                    emit_row_projections=True,
                )
            else:
                detection = self._detector_fn(
                    packed=packed.packed,
                    weights=self._weights,
                    kernel=self._kernel,
                )
            overflow = _exact_backend_u64(
                detection.get("rational_overflow_count", 0),
                field="rational_overflow_count",
            )
            if self._overflow + overflow >= (1 << 64):
                raise ValueError(
                    "detector analyzer: accumulated rational_overflow_count "
                    "exceeds uint64"
                )
            self._overflow += overflow
            baseband_power = np.asarray(packed.baseband_power_linear, dtype=np.float64)
            results = detection["results"]
            # The integrated spectrum (one FFT per nfft chunk) and the absolute-time
            # axis (one time per chunk) both assume one detector result per chunk --
            # which is exactly what the packed reader + frames_in_chunk=1 produce.
            # Guard it so a future packer change can't silently misalign the spectrum
            # / time axis against the per-frame arrays.
            if len(results) != 1:
                raise ValueError(
                    "detector analyzer: integrated-spectrum + time axis assume one "
                    f"detector result per nfft chunk, got {len(results)}. The "
                    "chime-baseband-packed reader yields one frame per chunk."
                )
            power_rows: list[tuple[int, int, int, int]] = []
            for row in results:
                required = (
                    "p_target_u64",
                    "p_ref_lower_u64",
                    "p_ref_upper_u64",
                    "p_ref_sum_u64",
                )
                missing = [field for field in required if field not in row]
                if missing:
                    raise ValueError(
                        "detector analyzer: backend result is missing exact power "
                        f"field(s) {missing}"
                    )
                num = _exact_backend_u64(row["p_target_u64"], field="p_target_u64")
                lower = _exact_backend_u64(
                    row["p_ref_lower_u64"], field="p_ref_lower_u64"
                )
                upper = _exact_backend_u64(
                    row["p_ref_upper_u64"], field="p_ref_upper_u64"
                )
                den = _exact_backend_u64(
                    row["p_ref_sum_u64"], field="p_ref_sum_u64"
                )
                combined = lower + upper
                if combined >= (1 << 64):
                    raise ValueError(
                        "detector analyzer: lower and upper reference power sum "
                        "exceeds uint64"
                    )
                if den != combined:
                    raise ValueError(
                        "detector analyzer: p_ref_sum_u64 does not equal the lower "
                        "and upper reference powers"
                    )
                power_rows.append((num, lower, upper, den))

            matched_filter_row_projections = (
                detection.get("matched_filter_row_projections")
                if emit_fine
                else None
            )
            if emit_fine and matched_filter_row_projections is None:
                raise RuntimeError(
                    "detector analyzer: fine support was enabled but the backend "
                    "returned no matched-filter row projections"
                )
            if matched_filter_row_projections is not None:
                rs0 = matched_filter_row_projections[0]
                # The exact terms are the measurement. The float reduction
                # and its CFAR ran here; both are recomputable from these
                # and neither was bit-equal to the deployed statistic, so
                # the scan no longer spends a transform on them.
                self._append_exact_fine_powers(detection, rs0)
            for local_index, powers in enumerate(power_rows):
                num, lower, upper, den = powers
                self._p_target.append(num)
                self._p_ref_lower.append(lower)
                self._p_ref_upper.append(upper)
                self._p_ref_sum.append(den)
                self._coarse_power_ratio.append(
                    float(power_terms_to_coarse_power_ratio(num, den))
                )
                self._normalized_coarse_power_ratio_db.append(
                    float(
                        power_terms_to_normalized_coarse_power_ratio_db(
                            num,
                            den,
                            target_norm_sq=self._target_norm_sq,
                            reference_norm_sum_sq=self._reference_norm_sum_sq,
                        )
                    )
                )
                normalized_excess = float(
                    power_terms_to_normalized_pilot_excess(
                        num,
                        den,
                        target_norm_sq=self._target_norm_sq,
                        reference_norm_sum_sq=self._reference_norm_sum_sq,
                    )
                )
                pilot_excess = float(
                    normalized_pilot_excess_to_db(normalized_excess)
                )
                self._pilot_excess_db.append(pilot_excess)
                self._estimated_data_shelf_snr_db.append(
                    float(pilot_excess_db_to_data_shelf_snr_db(
                        pilot_excess,
                        pilot_below_data_db=self._pilot_below_data_db,
                        bin_enbw_hz=self._bin_enbw_hz,
                        dtv_bandwidth_hz=self._dtv_bandwidth_hz,
                        pilot_capture_efficiency=self._pilot_capture_efficiency,
                    ))
                )
                self._normalized_pilot_excess.append(normalized_excess)
                # The backend mask is not authoritative product identity.
                # Reconstruct the declared host-exact policy from the stored
                # uint64 marginals and weight norms; this same bit controls the
                # after-mask integrated spectrum below.
                self._reject_mask.append(
                    normalized_positive_excess(
                        num,
                        den,
                        target_norm_sq=self._target_norm_sq,
                        reference_norm_sum_sq=self._reference_norm_sum_sq,
                    )
                )
                self._valid.append(1 if den > 0 else 0)
                bp = (
                    float(baseband_power[local_index])
                    if local_index < baseband_power.size
                    else float("nan")
                )
                self._baseband_power.append(bp)
                self._railed_count.append(int(railed_count))
                self._fill_count.append(int(fill_count))
                self._railed_total.append(int(railed_total))
                self._frame_unit_index.append(unit_idx)
                self._frame_in_unit.append(chunk_in_unit)
                n += 1
            # Integrated power spectrum: one FFT (rectangular window) of this chunk's
            # raw samples, |.|^2 summed over feeds, accumulated for valid frames only.
            # spec_before = every valid frame; spec_after = valid AND kept (not
            # rejected), so (before - after) is the spectrum the mask removed. Runs on
            # the kernel's GPU (cupy) in production and numpy in tests with identical
            # arithmetic; skipped for invalid frames (which enter neither spectrum).
            if not self._valid[-1]:
                self._append_psd_frame(None)
            if self._valid[-1]:
                a = self._xp.asarray(arr)
                X = self._xp.fft.fft(_unpack_4bit_xp(self._xp, a), axis=0)
                psd = (self._xp.abs(X) ** 2).sum(axis=1, dtype=self._xp.float64)
                self._append_psd_frame(
                    np.asarray(_to_host(psd), dtype=np.float64)
                )
                self._spec_before += psd
                if not self._reject_mask[-1]:
                    self._spec_after += psd
            chunk_in_unit += 1
        if n == 0:
            raise ValueError(
                "detector analyzer: unit yielded zero complete nfft frames; "
                "refusing to mark an empty or shorter-than-nfft HDF file complete"
            )
        self._n_frames += n
        key = meta.get("unit_key")
        if key is not None:
            self._keys.add(key)
            self._unit_order.append(str(key))
            # per-unit absolute-time axis + provenance, aligned 1:1 with _unit_order.
            # The reader's probe surfaces these from the file root attrs (NaN/0/""
            # when absent), so a missing attr degrades to an unusable-but-not-fatal
            # time rather than crashing the run.
            ev = meta.get("event_id")
            self._unit_time0_ctime.append(float(meta.get("time0_ctime", float("nan"))))
            self._unit_time0_fpga.append(int(meta.get("time0_fpga_count") or 0))
            self._unit_event_id.append(int(ev) if ev is not None else -1)
            self._unit_delta_time.append(float(meta.get("delta_time", float("nan"))))
            self._unit_archive_version.append(str(meta.get("archive_version", "")))
            self._unit_git_version_tag.append(str(meta.get("git_version_tag", "")))
            self._unit_input_map_sha256.append(str(meta.get("input_map_sha256", "")))
            self._unit_collection_server.append(str(meta.get("collection_server", "")))
            self._unit_scope.append(str(meta.get("scope", "")))
        return n

    def _append_exact_fine_powers(self, detection: Mapping[str, Any], rs0) -> None:
        """Retain the exact uint64 fine terms for this frame.

        Prefers the device stage (the deployed implementation). Falls back to
        the frozen CPU reference, which the kernel contract requires to be
        bit-identical, so an older library still yields the same integers.
        """
        exact = detection.get("fine_powers_u64")
        if exact is not None:
            arr = np.asarray(exact)
            expected = (1, REFERENCE_WEIGHT_TERMS, int(self._fine_bins))
            if arr.dtype != np.dtype(np.uint64) or arr.shape != expected:
                raise ValueError(
                    "detector analyzer: fine_powers_u64 must have dtype uint64 "
                    f"and shape {expected}"
                )
            self._fine_power_u64.append(np.ascontiguousarray(arr[0]))
            return
        host = rs0.get() if hasattr(rs0, "get") else np.asarray(rs0)
        host = np.asarray(host)
        if host.dtype.kind not in "iu":
            raise TypeError(
                "detector analyzer: matched-filter row projections must use an "
                "integer dtype"
            )
        self._fine_power_u64.append(
            np.ascontiguousarray(
                fine_power_fx(
                    host,
                    num_streams=int(self._num_input_streams),
                    windows_per_stream=int(self._nfft) // int(self._K),
                    num_weight_terms=REFERENCE_WEIGHT_TERMS,
                ).astype(np.uint64)
            )
        )

    def _append_psd_frame(self, psd: np.ndarray | None) -> None:
        """Encode one frame of PSD to int16 dB about its own reference.

        ``psd`` is None for a frame that never reached the transform; it stores
        the invalid sentinel, which sits outside the clipped code range and so
        cannot be read back as a measured level.
        """
        width = int(self._nfft)
        if psd is None:
            self._psd_codes.append(
                np.full(width, PSD_DB_INVALID, dtype=np.int16)
            )
            self._psd_ref.append(float("nan"))
            return
        finite = psd[np.isfinite(psd) & (psd > 0.0)]
        reference = float(np.median(finite)) if finite.size else 1.0
        if not np.isfinite(reference) or reference <= 0.0:
            reference = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(psd / reference)
        codes = np.where(
            np.isfinite(db),
            np.clip(
                np.rint(db * (1.0 / PSD_DB_STEP)),
                PSD_DB_MIN,
                PSD_DB_MAX,
            ),
            PSD_DB_INVALID,
        ).astype(np.int16)
        self._psd_codes.append(codes)
        self._psd_ref.append(reference)

    def save(self, path: str) -> None:
        n = int(self._n_frames)
        frame_index = np.arange(n, dtype=np.int64)
        col_f = lambda lst: np.asarray(lst, dtype=np.float64).reshape(n, 1)
        col_u = lambda lst, dt: np.asarray(lst, dtype=dt).reshape(n, 1)
        if self._fine_status == "enabled":
            if len(self._fine_power_u64) != n:
                raise RuntimeError(
                    "detector analyzer: enabled fine measurements do not match "
                    "the frame count"
                )
            fine_power_u64 = np.stack(self._fine_power_u64).astype(
                np.uint64, copy=False
            )
        else:
            if any(np.asarray(row).size for row in self._fine_power_u64):
                raise RuntimeError(
                    "detector analyzer: fine terms exist under a status without "
                    "measurements"
                )
            fine_power_u64 = np.zeros((n, 0, 0), dtype=np.uint64)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        atomic_savez_compressed(
            Path(path),
            # --- chime_detector_outputs schema (single pilot) ---
            schema_name=np.asarray(PER_PILOT_PRODUCT_SCHEMA_NAME),
            schema_revision=np.asarray(PER_PILOT_PRODUCT_SCHEMA_REVISION, dtype=np.int64),
            source_event_key_schema_version=np.asarray(
                SOURCE_EVENT_KEY_SCHEMA_VERSION
            ),
            decision_contract_json=np.asarray(current_decision_contract_json()),
            physical_channel=np.asarray([self._physical_channel], dtype=np.int32),
            freq_id=np.asarray([self._freq_id], dtype=np.int64),
            pilot_in_band=np.asarray([1 if self._pilot_in_band else 0], dtype=np.uint8),
            pilot_frequency_hz=np.asarray([self._pilot_rf_hz], dtype=np.float64),
            chime_frequency_hz=np.asarray([self._coarse_center_hz], dtype=np.float64),
            frame_index=frame_index,
            p_target_u64=col_u(self._p_target, np.uint64),
            p_ref_sum_u64=col_u(self._p_ref_sum, np.uint64),
            p_ref_lower_u64=col_u(self._p_ref_lower, np.uint64),
            p_ref_upper_u64=col_u(self._p_ref_upper, np.uint64),
            coarse_power_ratio=col_f(self._coarse_power_ratio),
            # --- current fine-diagnostic per-frame products ---
            # Exact deployed-statistic terms: [frames, 3, FINE_BINS] uint64.
            # Everything downstream of the transform -- ratio, CFAR, null
            # bulk, any future eta or Q16 threshold -- is exactly
            # recomputable from these; nothing recovers them afterwards.
            fine_power_u64=fine_power_u64,
            fine_pad_factor=np.asarray(int(FINE_PAD_FACTOR), dtype=np.int64),
            fine_num_bins=np.asarray(int(self._fine_bins), dtype=np.int64),
            fine_p_fa=np.asarray(float(self._fine_p_fa), dtype=np.float64),
            fine_guard_fine_bins=np.asarray(int(self._fine_guard), dtype=np.int64),
            fine_designated_bins=np.asarray(self._fine_designated, dtype=np.int64),
            fine_census_excluded_bins=np.asarray(
                self._fine_census, dtype=np.int64
            ),
            fine_status=np.asarray(str(self._fine_status)),
            normalized_coarse_power_ratio_db=col_f(self._normalized_coarse_power_ratio_db),
            pilot_excess_db=col_f(self._pilot_excess_db),
            estimated_data_shelf_snr_db=col_f(self._estimated_data_shelf_snr_db),
            normalized_pilot_excess=col_f(self._normalized_pilot_excess),
            reject_mask=col_u(self._reject_mask, np.uint8),
            valid=col_u(self._valid, np.uint8),
            # --- per-frame power + integrated spectra (rectangular window) ---
            baseband_power_linear=col_f(self._baseband_power),
            # ADC/F-engine saturation and 0x00 fill, counted while the frame
            # was in hand; unrecoverable afterwards because no raw samples are
            # stored. Both share the railed_sample_total denominator.
            railed_sample_count=col_u(self._railed_count, np.uint64),
            fill_sample_count=col_u(self._fill_count, np.uint64),
            railed_sample_total=col_u(self._railed_total, np.uint64),
            # Per-frame PSD, feeds summed, int16 dB about psd_db_reference.
            # Decode: psd_db_reference * 10 ** (codes / 1000.0).
            psd_frame_db_i16=(
                np.stack(self._psd_codes).astype(np.int16)
                if self._psd_codes
                else np.zeros((n, 0), dtype=np.int16)
            ),
            psd_db_reference=col_f(
                self._psd_ref if self._psd_ref else [float("nan")] * n
            ),
            psd_db_step_per_code=np.asarray(PSD_DB_STEP, dtype=np.float64),
            psd_db_invalid_code=np.asarray(PSD_DB_INVALID, dtype=np.int16),
            integrated_spectrum_before_mask=(
                _to_host(self._spec_before) if self._spec_before is not None
                else np.zeros(int(self._nfft), dtype=np.float64)
            ),
            integrated_spectrum_after_mask=(
                _to_host(self._spec_after) if self._spec_after is not None
                else np.zeros(int(self._nfft), dtype=np.float64)
            ),
            # --- per-frame -> per-unit time tags (see unit_* axis below) ---
            frame_unit_index=np.asarray(self._frame_unit_index, dtype=np.int32),
            frame_in_unit=np.asarray(self._frame_in_unit, dtype=np.int32),
            rational_overflow_count=np.asarray(self._overflow, dtype=np.uint64),
            # Legacy provenance and resume keys.
            schema_version=np.asarray(PER_PILOT_PRODUCT_SCHEMA_TOKEN),
            nfft=np.asarray(int(self._nfft), dtype=np.int64),
            sample_rate_hz=np.asarray(self._sample_rate_hz, dtype=np.float64),
            detector_window_samples=np.asarray(int(self._K), dtype=np.int64),
            num_input_streams=np.asarray(int(self._num_input_streams), dtype=np.int64),
            sense=np.asarray(
                -1 if self._spectral_sense == SPECTRAL_SENSE_INVERTED else 1,
                dtype=np.int64,
            ),
            unit_keys=np.asarray(sorted(str(k) for k in self._keys), dtype=str),
            unit_order=np.asarray(
                [str(k) for k in self._unit_order], dtype=str
            ),
            source_event_keys=np.asarray(
                [source_event_key(k, self._freq_id) for k in self._unit_order],
                dtype=str,
            ),
            # --- per-unit absolute-time axis (aligned to unit_order) -----------
            # per-frame time = unit_time0_ctime[u] + frame_in_unit[f]*nfft
            #                  * unit_delta_time[u],  u = frame_unit_index[f]
            unit_time0_ctime=np.asarray(self._unit_time0_ctime, dtype=np.float64),
            unit_time0_fpga=np.asarray(self._unit_time0_fpga, dtype=np.uint64),
            unit_event_id=np.asarray(self._unit_event_id, dtype=np.int64),
            unit_delta_time=np.asarray(self._unit_delta_time, dtype=np.float64),
            archive_version=np.asarray(
                [str(s) for s in self._unit_archive_version], dtype=str
            ),
            unit_git_version_tag=np.asarray(
                [str(s) for s in self._unit_git_version_tag], dtype=str
            ),
            unit_input_map_sha256=np.asarray(
                [str(s) for s in self._unit_input_map_sha256], dtype=str
            ),
            unit_collection_server=np.asarray(
                [str(s) for s in self._unit_collection_server], dtype=str
            ),
            # Triggered, scheduled, or local acquisition scope, per unit.
            unit_scope=np.asarray(
                [str(s) for s in self._unit_scope], dtype=str
            ),
            max_chunks_per_file=np.asarray(
                -1 if self._max_chunks_per_file is None
                else int(self._max_chunks_per_file),
                dtype=np.int64,
            ),
            detector_contract_json=np.asarray(self._detector_contract_json),
            reference_placement_json=np.asarray(
                json.dumps(self._reference_placement, sort_keys=True, separators=(",", ":"))
            ),
            # dB-calibration constants, recorded so a callable combine can refuse
            # to stack products reduced with different snr_shelf calibration.
            pilot_below_data_db=np.asarray(self._pilot_below_data_db, dtype=np.float64),
            bin_enbw_hz=np.asarray(self._bin_enbw_hz, dtype=np.float64),
            dtv_bandwidth_hz=np.asarray(self._dtv_bandwidth_hz, dtype=np.float64),
            pilot_capture_efficiency=np.asarray(
                self._pilot_capture_efficiency, dtype=np.float64
            ),
            # --- run provenance: which weights + build + mask rule produced this --
            target_norm_sq=np.asarray([self._target_norm_sq], dtype=np.int64),
            reference_norm_sum_sq=np.asarray([self._reference_norm_sum_sq], dtype=np.int64),
            weights_hash=np.asarray(self._weights_hash),
            weight_bank_sha256=np.asarray(self._weight_bank_sha256),
            weight_manifest_sha256=np.asarray(self._weight_manifest_sha256),
            # Digest of the coefficients alone: unchanged by header edits
            # such as a profile rename, so two banks compare on the numbers.
            weight_coefficients_sha256=np.asarray(
                self._weight_coefficients_sha256
            ),
            detector_version=np.asarray(self._detector_version),
            mask_rule=np.asarray(NORMALIZED_POSITIVE_EXCESS_MASK_RULE),
        )

    def summary(self) -> Mapping[str, Any]:
        masked = int(sum(self._reject_mask)) if self._reject_mask else 0
        return {
            "channel": self._physical_channel,
            "frames": self._n_frames,
            "masked_frames": masked,
        }


__all__ = ["PilotProxyDetectorAnalyzer"]

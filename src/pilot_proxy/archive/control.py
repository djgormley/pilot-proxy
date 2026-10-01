# coding=utf-8
"""Control-channel band monitor for non-pilot CHIME coarse channels.

The detector analyzer is *only* defined on a coarse channel that contains an
ATSC pilot: on any other ``freq_id`` it emits an explicitly invalid product
(NaN powers, frames excluded from the integrated spectra) or refuses outright
when no pilot lies within its label net. That is correct for the deployed
mask, and useless for the control measurements the method's validation needs:

* the empirical false-alarm rate of the coarse positive-excess rule on a band
  that cannot contain a pilot (the protected 608--614 MHz allocation);
* the pilot->shelf transfer premise, measured at mid-allocation coarse
  channels of strong stations and joined frame-by-frame against the pilot
  channel's mask record;
* a direct bound on sub-threshold leakage from mid-allocation power stacked
  over pilot-kept frames.

This analyzer records, per ``[nfft, n_feeds]`` complex frame:

``baseband_power_linear``
    mean ``|x|^2`` over samples and feeds, in native offset-binary integer
    units (the canonical ``chime-baseband`` reader yields integer-valued
    complex64), the same convention as the detector product's field of the
    same name -- so per-frame powers compare across freq_ids without unit
    gymnastics.

``coarse_marginal`` (one ``[K]`` row per frame, ``K = 128``)
    mean over the frame's ``nfft // K`` windows and all feeds of
    ``|DFT_K(window)|^2`` with a *rectangular* window, in native (unshifted)
    bin order and the reader's baseband orientation. ``K = 128`` is the
    baseline detector contract's ``detector_window_samples``, and a
    rectangular ``K``-point DFT bin is exactly the unit-modulus (float)
    analogue of the deployed detector's ``K``-sample weighted dot product, so
    the deployed coarse statistic is recoverable offline at *any* bin:

        F(b) = 2 * S[b] / (S[b - 2] + S[b + 2])        (indices mod K)

    with ``mu0 = 1`` exactly by construction. The deployed weights are
    int-quantized, so agreement with a detector product is a family statement,
    not a bit-exact one; quantify the difference by running this analyzer on a
    pilot ``freq_id`` and comparing (the parity gate), rather than assuming it
    away. The Parseval tie ``S.sum() == K**2 * baseband_power_linear`` holds
    per frame to float roundoff and is asserted in the tests.

The product also accumulates one full-resolution integrated spectrum
(rectangular ``|FFT_nfft|^2``, feed-averaged, ``sum`` + ``count`` so a resume
is exact) for narrow features.

Frame identity mirrors the detector product: ``unit_keys`` / ``files`` from
the accumulating base, per-frame ``frame_unit_index`` / ``frame_in_unit``, and
``source_event_keys`` normalised with the same freq_id-stripping rule the
combine step uses -- so a control product on ``freq_id`` 591 joins the pilot
product on 598 event-by-event, frame-by-frame. Absolute times are deliberately
absent: epoch dating comes through the event identity (the inventory's event
dates, or the paired pilot product's time axis).

There is no GPU requirement. These scans are network-bound (each ``freq_id``
is its own per-event baseband file); the per-frame maths is a batched
``K``-point FFT that NumPy does in milliseconds. ``--set gpu=1`` routes the
FFTs through CuPy when a GPU happens to be present, and is excluded from the
resume fingerprint because it does not change the product's meaning.

Run it against a surveyed inventory (one resumable ``<freq_id>.npz`` each)::

    pilot-proxy chime-control-scan \
        --inventory-name chime-controls \
        --select 484,491,515,545,591,745 \
        --output-dir control-products
"""
from __future__ import annotations

import datetime
import os
import sys
from typing import Any, Iterable, List, Mapping, Optional

import numpy as np

from pilot_proxy import __version__ as _PILOT_PROXY_VERSION
from pilot_proxy.archive.analyzer_base import AccumulatingAnalyzer
from pilot_proxy.archive.instruments import nyquist_sign
from pilot_proxy.archive.interfaces import (
    READY,
    STREAM_COMPLEX_BASEBAND,
    PluginInfo,
    RunContext,
)
from pilot_proxy.archive.selection import parse_freq_ids

from .chime_coarse import source_event_key

_SIGNATURE = "pilot-proxy-control"     # stamped into the product; verified on resume
SCHEMA_VERSION = "pilotproxy_control_product_v1"

# Baseline detector contract (docs/CANFAR_RUNBOOK.md): detector_window_samples.
# Fixed, not a tuning knob: the marginal is only the deployed statistic's
# unit-modulus analogue at the deployed window length.
DETECTOR_WINDOW_SAMPLES = 128
REFERENCE_OFFSET_BINS = 2              # the deployed +/-2-bin reference geometry

_HZ_PER_MHZ = 1_000_000.0
_CENTER_TOLERANCE_HZ = 1.0


# ---------------------------------------------------------------------------
# science helpers (pure NumPy; importable and testable without a scan)
# ---------------------------------------------------------------------------

def band_power(frame: np.ndarray) -> float:
    """Mean ``|x|^2`` over a ``[nfft, n_feeds]`` complex frame (float64)."""
    a = np.asarray(frame)
    return float(np.mean(a.real.astype(np.float64) ** 2
                         + a.imag.astype(np.float64) ** 2))


def coarse_marginal(frame: np.ndarray, k: int = DETECTOR_WINDOW_SAMPLES,
                    xp=np) -> np.ndarray:
    """Window-and-feed-averaged ``[k]`` power marginal of one frame.

    Rectangular ``k``-point DFT per window (native bin order), ``|.|^2``,
    mean over the ``nfft // k`` windows and all feeds. Satisfies
    ``out.sum() == k**2 * band_power(frame)`` to float roundoff (Parseval).
    """
    a = xp.asarray(frame)
    n = int(a.shape[0])
    if n % int(k) != 0:
        raise ValueError(
            f"frame length {n} is not divisible by the detector window {k}")
    # complex128 so the Parseval self-check certifies at float64 roundoff
    # (a ~0.5 GiB transient at CHIME scale; negligible against staging I/O).
    windows = a.reshape(n // int(k), int(k), -1).astype(xp.complex128)
    spec = xp.fft.fft(windows, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    out = power.mean(axis=(0, 2))
    host = out if xp is np else xp.asnumpy(out)
    return host.astype(np.float64)


def marginal_bin_for_rf_hz(rf_hz: float, *, f_center_hz: float, fs_hz: float,
                           nyquist_zone: int,
                           k: int = DETECTOR_WINDOW_SAMPLES) -> int:
    """Nearest marginal bin (native order, 0..k-1) for a sky frequency.

    The reader's sky convention is ``f_sky = f_center + sign * f_baseband``
    with ``sign = nyquist_sign(nyquist_zone)``, so the baseband offset is
    ``sign * (rf - f_center)``; the bin is its nearest multiple of
    ``fs / k``, wrapped into native FFT order. Raises if the frequency lies
    outside this coarse channel.
    """
    sign = int(nyquist_sign(int(nyquist_zone)))
    baseband_hz = sign * (float(rf_hz) - float(f_center_hz))
    if abs(baseband_hz) > float(fs_hz) / 2.0:
        raise ValueError(
            f"rf {rf_hz:.1f} Hz is {baseband_hz / 1e3:+.1f} kHz from center; "
            f"outside this +/-{fs_hz / 2e3:.1f} kHz coarse channel")
    return int(round(baseband_hz / (float(fs_hz) / float(k)))) % int(k)


def f_statistic_from_marginal(
        marginal: np.ndarray, target_bin: int,
        reference_offset_bins: int = REFERENCE_OFFSET_BINS) -> np.ndarray:
    """Deployed-geometry coarse F at ``target_bin``: ``2 S_t / (S_l + S_u)``.

    ``marginal`` is one ``[k]`` row or a stacked ``[n, k]`` array; reference
    bins wrap mod ``k`` (continuous-frequency reference tones wrap at the
    frame edge the same way). With unit-modulus float weights ``mu0 = 1``
    exactly; the empirical zero point of a channel remains the thing to
    measure, exactly as for the deployed detector.
    """
    m = np.asarray(marginal, dtype=np.float64)
    k = int(m.shape[-1])
    t = int(target_bin) % k
    lo = (t - int(reference_offset_bins)) % k
    hi = (t + int(reference_offset_bins)) % k
    denom = m[..., lo] + m[..., hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * m[..., t] / denom


def _parse_freq_ids(spec: Any, *, n_channels: Optional[int] = None) -> List[int]:
    """Explicit freq_id --select only (no 'all'): the control set is a choice."""
    if spec is None or str(spec).strip().lower() in ("", "all", "*"):
        raise SystemExit(
            "pilot-proxy-control needs explicit freq_id(s): --select 591 | "
            "484,491 | 477-491.\n('all' can't be expanded -- the control set "
            "is a scientific selection; name it explicitly.)")
    parsed = parse_freq_ids(spec, n_channels=n_channels)
    if parsed is None:
        raise SystemExit(
            f"pilot-proxy-control: --select {spec!r} resolved to no freq_ids.")
    return sorted(parsed)


# ---------------------------------------------------------------------------
# the analyzer
# ---------------------------------------------------------------------------

class ControlBandAnalyzer(AccumulatingAnalyzer):
    requires_in_order = True

    info = PluginInfo(
        name="pilot-proxy-control",
        kind="analyzer",
        summary=("Per-frame band power + K=128 coarse marginal for non-pilot "
                 "control freq_ids (event-keyed, resumable)."),
        status=READY,
        instruments=("*",),
        produces=("<freq_id>.npz (baseband_power_linear, coarse_marginal, "
                  "integrated_spectrum_sum/count, frame identity, provenance)"),
        requires=("numpy",),
        accepts_stream_kinds=(STREAM_COMPLEX_BASEBAND,),
        notes=("Rectangular-window analogue of the deployed K=128 coarse "
               "statistic at every bin; F(b) = 2 S[b] / (S[b-2] + S[b+2]) "
               "offline. No pilot, no weight bank, no GPU requirement."),
    )

    def __init__(self) -> None:
        super().__init__()
        self._power: list[float] = []            # per frame, float64
        self._marginal: list[np.ndarray] = []    # per frame, [K] float64
        self._frame_unit_index: list[int] = []   # per frame -> unit_keys row
        self._frame_in_unit: list[int] = []      # per frame, 0-based in file
        self._spectrum_sum: Optional[np.ndarray] = None   # [nfft] float64
        self._spectrum_count = 0
        self._event_keys: list[str] = []         # per unit, freq_id-stripped
        self._nfft = 0                           # observed frame length
        self._configured_nfft = 0                # instrument's requested framing
        self._n_feeds = 0                        # locked on the first frame
        self._fs = 0.0
        self._nyquist_zone = 1
        self._f_center: Optional[float] = None
        self._freq_id = -1
        self._max_frames = -1                    # per-file cap stamp (-1 = none)
        self._frame_batch_size = 1
        self._xp = np
        self._resumed = False

    # -- selection: explicit freq_ids -> one resumable product each ----------
    def resolve_selection(self, ctx: RunContext, spec: Any) -> Any:
        return _parse_freq_ids(
            spec, n_channels=getattr(ctx.instrument, "n_channels", None))

    def plan_runs(self, ctx: RunContext, spec: Any) -> list:
        return [[ch] for ch in self.resolve_selection(ctx, spec)]

    def resume_parameters(self, ctx: RunContext) -> Mapping[str, Any]:
        """Exclude execution settings that do not change the product."""
        execution_settings = {"gpu", "frame_batch_size"}
        return {
            k: v
            for k, v in dict(ctx.options or {}).items()
            if k not in execution_settings
        }

    # -- small shared lookups -------------------------------------------------
    @staticmethod
    def _expected_freq_id(ctx: RunContext):
        sel = ctx.selection
        if isinstance(sel, int):
            return int(sel)
        if isinstance(sel, (list, tuple)) and len(sel) == 1:
            return int(sel[0])
        return None

    @staticmethod
    def _run_cap(ctx: RunContext) -> int:
        v = (ctx.options or {}).get("max_frames_per_file")
        return int(v) if v else -1

    @staticmethod
    def _batch_size(ctx: RunContext) -> int:
        value = (ctx.options or {}).get("frame_batch_size", 1)
        if isinstance(value, bool):
            raise ValueError(
                "pilot-proxy-control: frame_batch_size must be a positive integer"
            )
        try:
            batch_size = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pilot-proxy-control: frame_batch_size must be a positive integer"
            ) from exc
        if batch_size < 1 or batch_size != value:
            raise ValueError(
                "pilot-proxy-control: frame_batch_size must be a positive integer"
            )
        return batch_size

    @staticmethod
    def _expected_center_hz(ctx: RunContext, freq_id: Optional[int]):
        if freq_id is None:
            return None
        mapper = getattr(getattr(ctx, "instrument", None), "freq_of_freq_id", None)
        if not callable(mapper):
            return None
        return float(mapper(freq_id)) * _HZ_PER_MHZ

    @staticmethod
    def _file_center_hz(value: Any, *, label: str) -> Optional[float]:
        if value is None:
            return None
        try:
            center = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} reports invalid f_center_hz={value!r}") from exc
        if not np.isfinite(center):
            raise ValueError(f"{label} reports non-finite f_center_hz={value!r}")
        return center

    # -- lifecycle ------------------------------------------------------------
    def begin(self, ctx: RunContext, first_meta: Mapping[str, Any]) -> None:
        self._frame_batch_size = self._batch_size(ctx)
        if (ctx.options or {}).get("gpu"):
            from pilot_proxy.archive import accel
            self._xp = accel.get_array_module(True)

        try:
            file_center = self._file_center_hz(
                first_meta.get("f_center_hz"), label="first file")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        if self._resumed:
            # Never overwrite a resumed product's invariants; check the first
            # new file against them instead.
            if (file_center is not None and self._f_center is not None
                    and abs(file_center - self._f_center) > _CENTER_TOLERANCE_HZ):
                raise SystemExit(
                    f"resumed product is freq_id {self._freq_id} (centre "
                    f"{self._f_center / _HZ_PER_MHZ:.4f} MHz) but the first new "
                    f"file is at {file_center / _HZ_PER_MHZ:.4f} MHz. "
                    "Use a fresh product.")
            return

        ch = self._expected_freq_id(ctx)
        self._freq_id = ch if ch is not None else -1
        self._fs = float(ctx.instrument.fs_hz)
        self._nyquist_zone = int(getattr(ctx.instrument, "nyquist_zone", 1) or 1)
        self._configured_nfft = int(getattr(ctx.instrument, "nfft", 0) or 0)
        self._max_frames = self._run_cap(ctx)
        if file_center is not None:
            expected_center = self._expected_center_hz(ctx, ch)
            if (expected_center is not None
                    and abs(file_center - expected_center) > _CENTER_TOLERANCE_HZ):
                raise SystemExit(
                    f"selected freq_id {ch} has instrument centre "
                    f"{expected_center / _HZ_PER_MHZ:.6f} MHz, but the first file "
                    f"reports {file_center / _HZ_PER_MHZ:.6f} MHz. The file or "
                    "inventory belongs to a different channel; refusing to label "
                    "its product with the selected freq_id.")
            self._f_center = file_center
        # nfft / accumulators are sized lazily on the first frame.

    def _size_to(self, nfft: int, n_feeds: int) -> None:
        if int(nfft) % DETECTOR_WINDOW_SAMPLES != 0:
            raise ValueError(
                f"pilot-proxy-control: frame length {nfft} is not divisible by "
                f"detector_window_samples={DETECTOR_WINDOW_SAMPLES}; the coarse "
                "marginal would not be the deployed statistic's analogue.")
        self._nfft = int(nfft)
        self._n_feeds = int(n_feeds)
        self._spectrum_sum = np.zeros(self._nfft, dtype=np.float64)

    def _prepare_frame(self, frame: Any, meta: Mapping[str, Any]):
        frame = self._xp.asarray(frame)
        if self._spectrum_sum is None:
            self._size_to(frame.shape[0], frame.shape[1] if frame.ndim > 1 else 1)
        if frame.shape[0] != self._nfft:
            return None
        feeds = int(frame.shape[1]) if frame.ndim > 1 else 1
        if feeds != self._n_feeds:
            raise ValueError(
                "pilot-proxy-control: feed count changed within a product "
                f"({feeds} != {self._n_feeds}) at "
                f"{meta.get('unit_name', '?')}; refusing to average "
                "incompatible frames into one product."
            )
        return frame

    def _consume_frame(self, frame: Any, unit_idx: int, frame_idx: int) -> None:
        xp = self._xp
        spec = xp.fft.fft(frame, axis=0)
        power_bins = spec.real ** 2 + spec.imag ** 2
        if power_bins.ndim > 1:
            power_bins = power_bins.mean(axis=tuple(range(1, power_bins.ndim)))
        host_bins = power_bins if xp is np else xp.asnumpy(power_bins)
        self._spectrum_sum += host_bins.astype(np.float64)
        self._spectrum_count += 1

        self._marginal.append(
            coarse_marginal(frame, DETECTOR_WINDOW_SAMPLES, xp=xp)
        )
        self._power.append(band_power(frame if xp is np else xp.asnumpy(frame)))
        self._frame_unit_index.append(unit_idx)
        self._frame_in_unit.append(frame_idx)

    def _consume_batch(
        self,
        frames: list[Any],
        unit_idx: int,
        first_frame_idx: int,
    ) -> None:
        if len(frames) == 1:
            self._consume_frame(frames[0], unit_idx, first_frame_idx)
            return

        xp = self._xp
        batch_count = len(frames)
        stacked = xp.stack(frames, axis=0)
        frames.clear()
        spectra = xp.fft.fft(stacked, axis=1)
        spectrum_power = spectra.real ** 2 + spectra.imag ** 2

        for offset in range(batch_count):
            power_bins = spectrum_power[offset]
            if power_bins.ndim > 1:
                power_bins = power_bins.mean(
                    axis=tuple(range(1, power_bins.ndim))
                )
            host_bins = power_bins if xp is np else xp.asnumpy(power_bins)
            self._spectrum_sum += host_bins.astype(np.float64)
            self._spectrum_count += 1
        del spectra, spectrum_power, power_bins, host_bins

        nfft = int(stacked.shape[1])
        windows = stacked.reshape(
            batch_count,
            nfft // DETECTOR_WINDOW_SAMPLES,
            DETECTOR_WINDOW_SAMPLES,
            -1,
        ).astype(xp.complex128)
        marginal_spectra = xp.fft.fft(windows, axis=2)
        marginal_power = (
            marginal_spectra.real ** 2 + marginal_spectra.imag ** 2
        )

        for offset in range(batch_count):
            marginal = marginal_power[offset].mean(axis=(0, 2))
            host_marginal = marginal if xp is np else xp.asnumpy(marginal)
            self._marginal.append(host_marginal.astype(np.float64))
            self._power.append(
                band_power(
                    stacked[offset]
                    if xp is np
                    else xp.asnumpy(stacked[offset])
                )
            )
            self._frame_unit_index.append(unit_idx)
            self._frame_in_unit.append(first_frame_idx + offset)

    def consume_file(self, arrays: Iterable, meta: Mapping[str, Any]) -> int:
        try:
            file_center = self._file_center_hz(
                meta.get("f_center_hz"),
                label=str(meta.get("unit_name", "file")))
        except ValueError as exc:
            print(f"  skip {meta.get('unit_name', '?')}: {exc}", file=sys.stderr)
            return 0
        if (file_center is not None and self._f_center is not None
                and abs(file_center - self._f_center) > _CENTER_TOLERANCE_HZ):
            print(f"  skip {meta.get('unit_name', '?')}: f_center "
                  f"{file_center / _HZ_PER_MHZ:.4f} MHz != product "
                  f"{self._f_center / _HZ_PER_MHZ:.4f} MHz", file=sys.stderr)
            return 0

        unit_idx = len(self._keys)
        n = 0
        pending: list[Any] = []
        pending_signature = None

        def flush() -> None:
            nonlocal n, pending_signature
            if not pending:
                return
            batch = list(pending)
            pending.clear()
            pending_signature = None
            batch_count = len(batch)
            self._consume_batch(batch, unit_idx, n)
            n += batch_count

        try:
            for raw_frame in arrays:
                frame = self._prepare_frame(raw_frame, meta)
                if frame is None:
                    continue
                signature = (tuple(frame.shape), str(frame.dtype))
                if pending and signature != pending_signature:
                    flush()
                pending_signature = signature
                pending.append(frame)
                if len(pending) >= self._frame_batch_size:
                    flush()
        except BaseException:
            flush()
            raise
        flush()

        if n:
            # A unit is resumably complete only if a frame actually contributed.
            self._record(meta)
            self._event_keys.append(
                source_event_key(meta.get("unit_key", ""), self._freq_id))
        return n

    # -- resume / checkpoint --------------------------------------------------
    def resume(self, path: str, ctx: RunContext) -> bool:
        if not os.path.exists(path):
            return False
        z = np.load(path, allow_pickle=False)
        if ("analysis" not in z.files) or (str(z["analysis"]) != _SIGNATURE):
            raise SystemExit(
                f"error: {path} was not written by the {_SIGNATURE} analyzer. "
                "Another analysis owns this file -- point --out elsewhere so "
                "products don't mix.")
        schema = str(z["schema_version"]) if "schema_version" in z.files else "?"
        if schema != SCHEMA_VERSION:
            raise SystemExit(
                f"error: {path} has schema {schema}; this analyzer writes "
                f"{SCHEMA_VERSION}. Use a fresh product.")

        def _mismatch(label, was, now):
            raise SystemExit(
                f"error: {path} was built with {label}={was} but this run uses "
                f"{label}={now}. Use a fresh product (--out elsewhere).")

        fs_prev = float(z["fs_hz"])
        if abs(fs_prev - float(ctx.instrument.fs_hz)) > 1.0:
            _mismatch("fs_hz", fs_prev, float(ctx.instrument.fs_hz))
        exp_ch = self._expected_freq_id(ctx)
        if exp_ch is not None and int(z["freq_id"]) != exp_ch:
            _mismatch("freq_id", int(z["freq_id"]), exp_ch)
        product_center = float(z["f_center_hz"])
        expected_center = self._expected_center_hz(ctx, int(z["freq_id"]))
        if (np.isfinite(product_center) and expected_center is not None
                and abs(product_center - expected_center) > _CENTER_TOLERANCE_HZ):
            _mismatch("f_center_hz", product_center, expected_center)
        configured_now = int(getattr(ctx.instrument, "nfft", 0) or 0)
        configured_prev = int(z["configured_nfft"])
        if (configured_prev and configured_now
                and configured_prev != configured_now):
            _mismatch("configured_nfft", configured_prev, configured_now)
        inst_zone = int(getattr(ctx.instrument, "nyquist_zone", 0) or 0)
        if inst_zone and int(z["nyquist_zone"]) != inst_zone:
            _mismatch("nyquist_zone", int(z["nyquist_zone"]), inst_zone)
        k_prev = int(z["detector_window_samples"])
        if k_prev != DETECTOR_WINDOW_SAMPLES:
            _mismatch("detector_window_samples", k_prev, DETECTOR_WINDOW_SAMPLES)
        prev_cap = int(z["max_frames_per_file"])
        cur_cap = self._run_cap(ctx)
        if prev_cap != cur_cap:
            raise SystemExit(
                f"error: {path} was built with max_frames_per_file="
                f"{prev_cap if prev_cap >= 0 else 'none'} but this run uses "
                f"{cur_cap if cur_cap >= 0 else 'none'}. A capped smoke-test "
                "product is not equivalent to a full one -- use --out "
                "elsewhere, or delete it and rerun.")

        self._power = [float(x) for x in z["baseband_power_linear"]]
        self._marginal = [np.array(row, dtype=np.float64)
                          for row in z["coarse_marginal"]]
        self._frame_unit_index = [int(x) for x in z["frame_unit_index"]]
        self._frame_in_unit = [int(x) for x in z["frame_in_unit"]]
        self._spectrum_sum = np.array(z["integrated_spectrum_sum"],
                                      dtype=np.float64)
        self._spectrum_count = int(z["integrated_spectrum_count"])
        self._nfft = int(z["nfft"])
        self._configured_nfft = configured_prev
        self._n_feeds = int(z["n_feeds"])
        self._fs = fs_prev
        self._nyquist_zone = int(z["nyquist_zone"])
        self._f_center = product_center if np.isfinite(product_center) else None
        self._freq_id = int(z["freq_id"])
        self._max_frames = prev_cap
        self._keys = [str(x) for x in z["unit_keys"]]
        self._names = [str(x) for x in z["files"]]
        self._event_keys = [str(x) for x in z["source_event_keys"]]
        self._resumed = True
        return True

    def save(self, path: str) -> None:
        n = len(self._power)
        marginal = (np.stack(self._marginal).astype(np.float64)
                    if self._marginal
                    else np.zeros((0, DETECTOR_WINDOW_SAMPLES)))
        self._atomic_savez(
            path,
            analysis=_SIGNATURE,
            schema_version=SCHEMA_VERSION,
            analyzer_version=f"pilot-proxy/{_PILOT_PROXY_VERSION}",
            freq_id=self._freq_id,
            f_center_hz=(self._f_center if self._f_center is not None else np.nan),
            fs_hz=self._fs,
            nyquist_zone=self._nyquist_zone,
            nfft=self._nfft,
            configured_nfft=self._configured_nfft,
            detector_window_samples=DETECTOR_WINDOW_SAMPLES,
            reference_offset_bins=REFERENCE_OFFSET_BINS,
            n_feeds=self._n_feeds,
            n_frames=n,
            baseband_power_linear=np.asarray(self._power, dtype=np.float64),
            coarse_marginal=marginal,
            frame_unit_index=np.asarray(self._frame_unit_index, dtype=np.int32),
            frame_in_unit=np.asarray(self._frame_in_unit, dtype=np.int32),
            integrated_spectrum_sum=(self._spectrum_sum
                                     if self._spectrum_sum is not None
                                     else np.zeros(0)),
            integrated_spectrum_count=self._spectrum_count,
            max_frames_per_file=self._max_frames,
            files=np.array(self._names),
            unit_keys=np.array(self._keys),
            source_event_keys=np.array(self._event_keys),
            created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def summary(self) -> Mapping[str, Any]:
        out: dict = {"frames": len(self._power), "files": len(self._names),
                     "freq_id": self._freq_id}
        if self._power:
            out["mean_power"] = round(float(np.mean(self._power)), 3)
        if self._marginal:
            mean_marginal = np.mean(np.stack(self._marginal), axis=0)
            peak = int(np.argmax(mean_marginal))
            out["marginal_peak_bin"] = peak
            if self._f_center is not None and self._fs:
                sign = nyquist_sign(self._nyquist_zone)
                bb = (peak if peak <= DETECTOR_WINDOW_SAMPLES // 2
                      else peak - DETECTOR_WINDOW_SAMPLES)
                bb_hz = bb * self._fs / DETECTOR_WINDOW_SAMPLES
                out["marginal_peak_sky_mhz"] = round(
                    (self._f_center + sign * bb_hz) / _HZ_PER_MHZ, 4)
        return out

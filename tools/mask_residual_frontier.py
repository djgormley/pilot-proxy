#!/usr/bin/env python3
# coding=utf-8
"""The mask-versus-residual frontier from synthetic truth, at the production geometry.

Chapter 6 (``sec:detection:synthetic``, ``fig:detection:frontier``) asks for
the frontier of ``eq:detection:frontier`` drawn against a contamination whose
value is *known*: one unsmoothed curve per rank ``rho`` in the (surviving
injected contamination, masked fraction) plane with ``eta`` labelling points
along each curve, at several duty cycles, with trial-bootstrap intervals; a
second panel setting what the histogram cumulative sums of
``eq:detection:histogram`` claim survives beside what was actually injected
into the frames the mask kept; and the before-and-after frame-averaged fine
spectra at a loud and a sub-noise shelf.

The second panel is the point of the exercise.  Every residual the archive
analysis reports is a prefix sum over a residual-score histogram whose
per-frame entries are estimates.  On the archive the truth is unknown, so the
estimator has never been checked against one.  Here the bench decides frame by
frame whether a shelf was injected and at what level, so the same prefix sums
can be totalled over the same kept frames for the claim and for the truth.

Nothing in the masking path is re-derived for the bench.  The per-rank Q16
boundary, the keep rule, the empirical candidate staircase, the retained-frame
floor and the shelf-or-floor residual convention are transcribed in
``pilot_proxy.testbench.mask_frontier`` from the detector's score and
characterization modules (``pilot_proxy.detectors.narrowband_marker.scores``,
``pilot_proxy.characterization.surface`` and ``.nulls``); a frontier drawn with a
different rule would validate nothing.  The waveform, reference-PFB
normalization, weight profile, designated set and bulk mask come from
``tools/current_geometry_sensitivity.py`` so that these frames are the frames
the crossing and representation-loss results were measured on.

The transmitter is switched on and off across frames at a declared duty cycle,
so the off frames are a null population whose membership is known rather than
inferred.  A second, disjoint off population supplies the channel floor the
archive way: the 90th percentile of its finite shelf estimates.

Stages::

    # 6000 frames at K=128, L=128, M=2048 on the GPU
    python3 tools/mask_residual_frontier.py --stage generate --gpu \
        --input-iq generated/atsc/atsc_8vsb_complex64_settled.cfile \
        --waveform-audit generated/atsc/atsc_waveform_audit_settled.json \
        --output-dir docs/evidence/mask_residual_frontier_<date>

    # the frontier, the claim-against-truth panel, the plates and the tables
    python3 tools/mask_residual_frontier.py --stage report \
        --output-dir docs/evidence/mask_residual_frontier_<date>

    # the device generator against the host reference, on real frames
    python3 tools/mask_residual_frontier.py --stage audit --gpu ...

``--num-streams`` below 2048 requires ``--allow-reduced-geometry`` and stamps
every product as reduced; a reduced run is code validation, not evidence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_TOOLS = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "current_geometry_sensitivity", _TOOLS / "current_geometry_sensitivity.py"
)
cgs = importlib.util.module_from_spec(_SPEC)
# ``dataclass`` resolves annotations through ``sys.modules``, so the sibling
# driver must be registered before it is executed.
sys.modules["current_geometry_sensitivity"] = cgs
_SPEC.loader.exec_module(cgs)

from pilot_proxy.detector_contract import weight_term_norms_sq
from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.dtv_units import (
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    pilot_capture_efficiency_db,
    spreading_loss_db_from_bin_enbw_hz,
)
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.provenance import file_sha256
from pilot_proxy.testbench.mask_frontier import (
    ALWAYS_MASKED_Q16,
    FLOOR_MIN_FRAMES,
    FLOOR_PERCENTILE,
    MASK_FRONTIER_SCHEMA,
    MAX_MULTIPLIER_Q16,
    MIN_RETAINED_FRAMES,
    Q16_SCALE,
    bootstrap_frontier,
    candidate_multipliers_q16,
    coarse_normalized_excess,
    frontier_points,
    kept_at,
    kept_frame_means,
    measured_floor_db,
    minimum_mask_envelope,
    required_multipliers_by_rank,
    residual_score_histogram,
    shelf_db_from_excess,
    systematic_residuals,
)
from pilot_proxy.testbench.sensitivity_study import canonical_seed

UTC = timezone.utc
K = cgs.K
WINDOWS = cgs.WINDOWS
FINE_BINS = cgs.FINE_BINS
BITS = cgs.BITS
PRODUCTION_STREAMS = cgs.PRODUCTION_STREAMS

OFF_LABEL = "off"
FLOOR_LABEL = "floor"

DEFAULT_SHELF_SNR_DB = (-10.0, -44.0, -55.0)
DEFAULT_DUTY_CYCLES = (0.10, 0.25, 0.50)
DEFAULT_RHO = (12, 31, 62, 93, 112)
DEFAULT_FRAMES_ON = 1000
DEFAULT_FRAMES_OFF = 2000
DEFAULT_FRAMES_FLOOR = 1000
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260907

# The bench decides these labels rather than reading them off a spectrum.
LOUD_LABEL = "loud"
SUBNOISE_LABEL = "sub-noise"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _label(value: float) -> str:
    return f"{value:+.1f}".replace(".", "p").replace("+", "p").replace("-", "m")


def _shelf_offset_db(args: argparse.Namespace) -> float:
    """The deterministic pilot-excess to data-shelf offset of this geometry."""
    return (
        float(args.pilot_below_data_db)
        - spreading_loss_db_from_bin_enbw_hz(
            float(args.bin_enbw_hz), dtv_bandwidth_hz=float(args.dtv_bandwidth_hz)
        )
        - pilot_capture_efficiency_db(float(args.pilot_capture_efficiency))
    )


class DeviceFrameSource:
    """2048-stream packed frames built on the device, exact powers off it.

    The packing is the deployed one: ``round`` to the signed 4+4 grid at the
    declared input scale, clipped, real in the high nibble.  ``--stage audit``
    checks the device packing against ``quantize_complex_numpy`` and the device
    fine powers against the exact CPU reference on real frames rather than
    asserting the port.
    """

    def __init__(self, args, profile, signal_rows, *, gpu):
        import cupy as cp

        self.cp = cp
        self.args = args
        self.profile = profile
        self.gpu = gpu
        self.streams = int(args.num_streams)
        self.scale = float(args.input_scale)
        self.signal = cp.asarray(np.asarray(signal_rows, dtype=np.complex64))
        self.max_int = (1 << (BITS - 1)) - 1
        self.mask = (1 << BITS) - 1

    def packed(self, *, seed: int, amplitude: float) -> Any:
        cp = self.cp
        rows = self.streams * WINDOWS
        generator = cp.random.default_rng(int(seed) % (1 << 63))
        real = generator.standard_normal((rows, K), dtype=cp.float32)
        imag = generator.standard_normal((rows, K), dtype=cp.float32)
        block = (real + 1j * imag).astype(cp.complex64) * cp.float32(
            1.0 / math.sqrt(2.0)
        )
        if amplitude != 0.0:
            block += cp.float32(amplitude) * cp.tile(self.signal, (self.streams, 1))
        scale = cp.float32(self.scale)
        r = cp.clip(
            cp.round(block.real * scale), -self.max_int, self.max_int
        ).astype(cp.int32)
        i = cp.clip(
            cp.round(block.imag * scale), -self.max_int, self.max_int
        ).astype(cp.int32)
        packed = ((r << BITS) | (i & self.mask)).astype(cp.int8)
        clipped = float(
            cp.mean(
                (cp.abs(block.real * scale) > self.max_int)
                | (cp.abs(block.imag * scale) > self.max_int)
            )
        )
        return packed, clipped

    def frame(self, *, seed: int, amplitude: float):
        cp = self.cp
        packed, clipped = self.packed(seed=seed, amplitude=amplitude)
        kernel = self.gpu.kernel
        diagnostic = cp.zeros(1, dtype=cp.float32)
        handle = kernel.create_raw(
            int(packed.shape[0]), packed.data.ptr, diagnostic.data.ptr
        )
        fine = cp.zeros((3, FINE_BINS), dtype=cp.uint64)
        powers = cp.zeros(3, dtype=cp.uint64)
        try:
            kernel.compute_fused_fine_u64(
                handle,
                self.profile.packed_weights.ctypes.data,
                int(fine.data.ptr),
                int(powers.data.ptr),
                0,
            )
            cp.cuda.Device().synchronize()
            host_fine = cp.asnumpy(fine).astype(np.uint64, copy=False)
            host_powers = tuple(int(value) for value in cp.asnumpy(powers))
        finally:
            kernel.destroy(handle)
        return host_fine, host_powers, clipped

    def host_packed(self, *, seed: int, amplitude: float) -> np.ndarray:
        """The same frame packed on the host, for the audit."""
        cp = self.cp
        rows = self.streams * WINDOWS
        generator = cp.random.default_rng(int(seed) % (1 << 63))
        real = generator.standard_normal((rows, K), dtype=cp.float32)
        imag = generator.standard_normal((rows, K), dtype=cp.float32)
        block = (real + 1j * imag).astype(cp.complex64) * cp.float32(
            1.0 / math.sqrt(2.0)
        )
        if amplitude != 0.0:
            block += cp.float32(amplitude) * cp.tile(self.signal, (self.streams, 1))
        return quantize_complex_numpy(cp.asnumpy(block), BITS, self.scale)


def _population_seed(args: argparse.Namespace, label: str, index: int) -> int:
    return canonical_seed(
        int(args.seed),
        "mask_frontier_frame",
        str(label),
        int(args.physical_channel),
        int(round(float(args.offset_fine_bins) * 1_000_000.0)),
        int(args.num_streams),
        int(index),
    )


def _populations(args: argparse.Namespace) -> list[tuple[str, float | None, int]]:
    populations: list[tuple[str, float | None, int]] = [
        (OFF_LABEL, None, int(args.frames_off)),
        (FLOOR_LABEL, None, int(args.frames_floor)),
    ]
    for shelf in args.shelf_snr_db:
        populations.append(
            (f"on_{_label(float(shelf))}", float(shelf), int(args.frames_on))
        )
    return populations


def _shard_path(output_dir: Path, label: str) -> Path:
    return output_dir / "frames" / f"{label}.npz"


def _run_generate(args, profile, gpu) -> int:
    config = _study_config(args, profile)
    atsc_rows, _tone_rows, meta = cgs._load_signal_cache(args, config, profile)
    source = DeviceFrameSource(args, profile, atsc_rows, gpu=gpu)
    bulk_size = int(np.count_nonzero(profile.bulk_mask))
    target_norm_sq, ref_lower_norm_sq, ref_upper_norm_sq = weight_term_norms_sq(
        profile.packed_weights, bits_per_component=BITS
    )
    reference_norm_sum_sq = ref_lower_norm_sq + ref_upper_norm_sq
    offset_db = _shelf_offset_db(args)
    (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)

    for label, shelf_db, count in _populations(args):
        path = _shard_path(args.output_dir, label)
        if path.exists() and args.resume:
            print(f"{label}: present, skipped")
            continue
        amplitude = (
            0.0
            if shelf_db is None
            else cgs._signal_amplitude_for_snr(
                shelf_db, clean_iq_power=meta["clean_iq_power"], args=args
            )
        )
        injected_linear = 0.0 if shelf_db is None else 10.0 ** (shelf_db / 10.0)
        required = np.zeros((count, bulk_size), dtype=np.uint64)
        fine = np.zeros((count, 3, FINE_BINS), dtype=np.uint64)
        marginals = np.zeros((count, 3), dtype=np.uint64)
        excess = np.zeros(count, dtype=np.float64)
        shelf_estimate = np.zeros(count, dtype=np.float64)
        clip = np.zeros(count, dtype=np.float64)
        seeds = np.zeros(count, dtype=np.uint64)
        started = time.time()
        for index in range(count):
            seed = _population_seed(args, label, index)
            powers, terms, clipped = source.frame(seed=seed, amplitude=amplitude)
            boundaries = required_multipliers_by_rank(
                powers, designated=profile.designated, bulk_mask=profile.bulk_mask
            )
            # 0 is not a deployable boundary; it is this product's
            # always-masked sentinel, so the stored column stays uint64.
            required[index] = np.asarray(
                [0 if value == ALWAYS_MASKED_Q16 else value for value in boundaries],
                dtype=np.uint64,
            )
            fine[index] = powers
            marginals[index] = np.asarray(terms, dtype=np.uint64)
            value = coarse_normalized_excess(
                terms[0],
                terms[1],
                terms[2],
                target_norm_sq=target_norm_sq,
                reference_norm_sum_sq=reference_norm_sum_sq,
            )
            excess[index] = value
            shelf_estimate[index] = shelf_db_from_excess(value, offset_db=offset_db)
            clip[index] = clipped
            seeds[index] = np.uint64(seed)
            if index and index % 250 == 0:
                rate = (time.time() - started) / index
                print(f"  {label}: {index}/{count} at {rate:.3f} s/frame", flush=True)
        elapsed = time.time() - started
        shard_meta = {
            "schema_version": MASK_FRONTIER_SCHEMA,
            "created_utc": _utc_now(),
            "population": label,
            "injected_shelf_snr_db": (
                float("nan") if shelf_db is None else float(shelf_db)
            ),
            "injected_linear": float(injected_linear),
            "signal_amplitude": float(amplitude),
            "frames": int(count),
            "num_streams": int(args.num_streams),
            "reduced_geometry": bool(int(args.num_streams) != PRODUCTION_STREAMS),
            "physical_channel": int(args.physical_channel),
            "offset_fine_bins": float(args.offset_fine_bins),
            "anchor_bin": int(profile.anchor_bin),
            "bulk_size": bulk_size,
            "cfar_rank_zero_based": int(profile.cfar_rank),
            "target_norm_sq": int(target_norm_sq),
            "reference_norm_sum_sq": int(reference_norm_sum_sq),
            "shelf_offset_db": float(offset_db),
            "always_masked_sentinel": 0,
            "seed": int(args.seed),
            "elapsed_seconds": float(elapsed),
            "generator": "cupy default_rng standard_normal, device-packed",
            "config_sha256": config["config_sha256"],
            "clean_iq_power": float(meta["clean_iq_power"]),
        }
        np.savez_compressed(
            path,
            required_multiplier_q16=required,
            fine_power_u64=fine,
            coarse_marginals_u64=marginals,
            normalized_excess=excess,
            shelf_estimate_db=shelf_estimate,
            clip_fraction=clip,
            frame_seed=seeds,
            meta_json=np.asarray(json.dumps(shard_meta, sort_keys=True)),
        )
        print(
            f"{label}: {count} frames in {elapsed:.1f} s "
            f"({elapsed / max(count, 1):.3f} s/frame) -> {path}",
            flush=True,
        )
    return 0


def _run_audit(args, profile, gpu) -> int:
    """Check the device generator against the host reference on real frames."""
    from pilot_proxy.detector_reference import (
        matched_filter_row_projections_cpu_reference_packed,
    )

    config = _study_config(args, profile)
    atsc_rows, _tone, meta = cgs._load_signal_cache(args, config, profile)
    source = DeviceFrameSource(args, profile, atsc_rows, gpu=gpu)
    rows = []
    for label, shelf_db in ((OFF_LABEL, None), ("on", float(args.shelf_snr_db[0]))):
        amplitude = (
            0.0
            if shelf_db is None
            else cgs._signal_amplitude_for_snr(
                shelf_db, clean_iq_power=meta["clean_iq_power"], args=args
            )
        )
        for index in range(int(args.audit_frames)):
            seed = _population_seed(args, f"audit_{label}", index)
            device_packed, _ = source.packed(seed=seed, amplitude=amplitude)
            host = source.host_packed(seed=seed, amplitude=amplitude)
            packing_identical = bool(
                np.array_equal(source.cp.asnumpy(device_packed), host)
            )
            projections = matched_filter_row_projections_cpu_reference_packed(
                host, profile.packed_weights, BITS
            ).astype(np.int64)
            cpu_fine = cgs._fixed_fine_powers_by_stream(
                projections.astype(np.int32), num_streams=int(args.num_streams)
            ).sum(axis=0, dtype=np.uint64)
            cpu_marginals = tuple(
                int(np.sum(projections[term, :, 0].astype(object) ** 2
                           + projections[term, :, 1].astype(object) ** 2))
                for term in range(3)
            )
            gpu_fine, gpu_marginals, _ = source.frame(seed=seed, amplitude=amplitude)
            rows.append(
                {
                    "population": label,
                    "frame": index,
                    "packing_bit_identical": packing_identical,
                    "fine_powers_bit_identical": bool(
                        np.array_equal(cpu_fine, gpu_fine)
                    ),
                    "coarse_marginals_bit_identical": bool(
                        cpu_marginals == gpu_marginals
                    ),
                }
            )
            print(rows[-1], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_strict(
        args.output_dir / "device_audit.json",
        {
            "schema_version": MASK_FRONTIER_SCHEMA,
            "created_utc": _utc_now(),
            "num_streams": int(args.num_streams),
            "frames": rows,
            "all_identical": all(
                row["packing_bit_identical"]
                and row["fine_powers_bit_identical"]
                and row["coarse_marginals_bit_identical"]
                for row in rows
            ),
        },
    )
    return 0


def _load_shard(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta_json"].item()))
        return {
            "meta": meta,
            "required": np.asarray(archive["required_multiplier_q16"]),
            "fine": np.asarray(archive["fine_power_u64"]),
            "excess": np.asarray(archive["normalized_excess"]),
            "shelf": np.asarray(archive["shelf_estimate_db"]),
            "clip": np.asarray(archive["clip_fraction"]),
        }


def _required_column(required: np.ndarray, rho: int) -> list[int]:
    """One-based rank ``rho`` with the stored zero sentinel expanded."""
    column = required[:, int(rho) - 1]
    return [
        ALWAYS_MASKED_Q16 if int(value) == 0 else int(value) for value in column
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("generate", "audit", "report", "figure"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-iq", type=Path)
    parser.add_argument("--waveform-audit", type=Path)
    parser.add_argument("--weights-path", type=Path, default=cgs.DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--lib-path", type=Path, default=cgs.DEFAULT_LIB_PATH)
    parser.add_argument("--physical-channel", type=int, default=14)
    parser.add_argument("--offset-fine-bins", type=float, default=0.0)
    parser.add_argument("--num-streams", type=int, default=PRODUCTION_STREAMS)
    parser.add_argument("--allow-reduced-geometry", action="store_true")
    parser.add_argument(
        "--shelf-snr-db", type=float, nargs="+", default=list(DEFAULT_SHELF_SNR_DB)
    )
    parser.add_argument(
        "--duty-cycle", type=float, nargs="+", default=list(DEFAULT_DUTY_CYCLES)
    )
    parser.add_argument("--rho", type=int, nargs="+", default=list(DEFAULT_RHO))
    parser.add_argument(
        "--panel-shelf-db",
        type=float,
        default=None,
        help="shelf level the frontier panel is drawn at (default: the "
             "second lowest, where the mask is partly effective)",
    )
    parser.add_argument("--frames-on", type=int, default=DEFAULT_FRAMES_ON)
    parser.add_argument("--frames-off", type=int, default=DEFAULT_FRAMES_OFF)
    parser.add_argument("--frames-floor", type=int, default=DEFAULT_FRAMES_FLOOR)
    parser.add_argument("--audit-frames", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--designated-half-width", type=int, default=2)
    parser.add_argument("--guard-fine-bins", type=int, default=1)
    parser.add_argument("--null-quantile", type=float, default=0.5)
    parser.add_argument("--spectral-sense", default="normal")
    parser.add_argument("--input-scale", type=float, default=cgs.DEFAULT_INPUT_SCALE)
    parser.add_argument(
        "--pilot-below-data-db", type=float, default=PILOT_BELOW_DATA_DB
    )
    parser.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    parser.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    parser.add_argument(
        "--pilot-capture-efficiency", type=float, default=PILOT_CAPTURE_EFFICIENCY
    )
    parser.add_argument("--iq-sample-rate-hz", type=float, default=None)
    return parser


def _prepare(args: argparse.Namespace):
    if int(args.num_streams) != PRODUCTION_STREAMS and not args.allow_reduced_geometry:
        raise SystemExit(
            f"--num-streams {args.num_streams} is not the production geometry; "
            "pass --allow-reduced-geometry to run it as code validation"
        )
    args.mode = "production"
    args.simulation_backend = cgs.SIMULATION_SUFFICIENT
    args.sufficient_pool_streams = cgs.PRODUCTION_SUFFICIENT_POOL_STREAMS
    args.trials = 1
    args.trial_start = 0
    args.p_fa = cgs.DEFAULT_P_FA
    args.bootstrap_replicates = int(args.bootstrap_replicates)
    args.historical_kernel_artifact = None
    if args.iq_sample_rate_hz is None:
        args.iq_sample_rate_hz = cgs.GNU_RADIO_ATSC_SYMBOL_RATE_HZ
    bank = DetectorWeightBank(explicit_path=args.weights_path)
    args._bank = bank
    args.physical_channel = [int(args.physical_channel)] if isinstance(
        args.physical_channel, int
    ) else args.physical_channel
    channel = int(args.physical_channel[0])
    args.physical_channel = channel
    profile = cgs._profile(args, bank, channel, float(args.offset_fine_bins))
    return profile


def _study_config(args: argparse.Namespace, profile) -> dict[str, Any]:
    """The immutable identity of one frontier study."""
    if args.input_iq is None or not Path(args.input_iq).is_file():
        raise SystemExit("--input-iq must name an existing waveform")
    audit = (
        cgs._read_passed_waveform_audit(Path(args.waveform_audit))
        if args.waveform_audit is not None
        else None
    )
    if audit is None:
        raise SystemExit("--waveform-audit is required: the waveform must have passed")
    target_norm_sq, ref_lower, ref_upper = weight_term_norms_sq(
        profile.packed_weights, bits_per_component=BITS
    )
    payload = {
        "schema_version": MASK_FRONTIER_SCHEMA,
        "geometry": {
            "detector_window_samples": K,
            "windows_per_stream": WINDOWS,
            "fine_bins": FINE_BINS,
            "num_streams": int(args.num_streams),
            "production_streams": PRODUCTION_STREAMS,
            "reduced_geometry": bool(int(args.num_streams) != PRODUCTION_STREAMS),
            "physical_channel": int(args.physical_channel),
            "offset_fine_bins": float(args.offset_fine_bins),
            "anchor_bin": int(profile.anchor_bin),
            "designated_half_width": int(args.designated_half_width),
            "guard_fine_bins": int(args.guard_fine_bins),
            "bulk_size": int(np.count_nonzero(profile.bulk_mask)),
            "spectral_sense": str(args.spectral_sense),
            "input_scale": float(args.input_scale),
        },
        "calibration": {
            "pilot_below_data_db": float(args.pilot_below_data_db),
            "bin_enbw_hz": float(args.bin_enbw_hz),
            "dtv_bandwidth_hz": float(args.dtv_bandwidth_hz),
            "pilot_capture_efficiency": float(args.pilot_capture_efficiency),
            "shelf_offset_db": _shelf_offset_db(args),
            "target_norm_sq": int(target_norm_sq),
            "reference_norm_sum_sq": int(ref_lower + ref_upper),
        },
        "campaign": {
            "shelf_snr_db": [float(value) for value in args.shelf_snr_db],
            "duty_cycle": [float(value) for value in args.duty_cycle],
            "rho": [int(value) for value in args.rho],
            "frames_on": int(args.frames_on),
            "frames_off": int(args.frames_off),
            "frames_floor": int(args.frames_floor),
            "seed": int(args.seed),
            "bootstrap_replicates": int(args.bootstrap_replicates),
        },
        "policy": {
            "minimum_retained_frames": MIN_RETAINED_FRAMES,
            "minimum_candidate_multiplier_q16": 1,
            "floor_percentile": FLOOR_PERCENTILE,
            "floor_minimum_frames": FLOOR_MIN_FRAMES,
            "always_masked_q16": str(ALWAYS_MASKED_Q16),
            "maximum_multiplier_q16": str(MAX_MULTIPLIER_Q16),
            "q16_scale": Q16_SCALE,
        },
        "inputs": {
            "input_iq": str(Path(args.input_iq).resolve()),
            "input_iq_sha256": file_sha256(Path(args.input_iq)),
            "waveform_audit_sha256": file_sha256(Path(args.waveform_audit)),
            "weights_path": str(Path(args.weights_path).resolve()),
            "weights_sha256": file_sha256(Path(args.weights_path)),
            "iq_sample_rate_hz": float(args.iq_sample_rate_hz),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["config_sha256"] = digest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "study_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != digest:
            raise SystemExit(
                f"study configuration has changed since {path} was written"
            )
        return existing
    write_json_strict(path, payload)
    return payload


def _duty_population(
    args: argparse.Namespace,
    off: dict[str, Any],
    on: dict[str, Any],
    duty: float,
    *,
    frames: int,
) -> dict[str, Any]:
    """Compose one duty cycle from the labelled off and on populations.

    Frames are taken in order from each bank, so the same duty cycle always
    draws the same frames and a larger duty cycle contains the smaller one's
    on frames.  ``frames`` is the composed population size.
    """
    on_count = int(round(float(duty) * int(frames)))
    off_count = int(frames) - on_count
    if on_count > on["required"].shape[0] or off_count > off["required"].shape[0]:
        raise SystemExit(
            f"duty cycle {duty} needs {on_count} on and {off_count} off frames; "
            f"the banks hold {on['required'].shape[0]} and "
            f"{off['required'].shape[0]}"
        )
    injected_linear = float(on["meta"]["injected_linear"])
    return {
        "required": np.concatenate(
            [off["required"][:off_count], on["required"][:on_count]], axis=0
        ),
        "shelf": np.concatenate(
            [off["shelf"][:off_count], on["shelf"][:on_count]], axis=0
        ),
        "fine": np.concatenate(
            [off["fine"][:off_count], on["fine"][:on_count]], axis=0
        ),
        "injected": np.concatenate(
            [
                np.zeros(off_count, dtype=np.float64),
                np.full(on_count, injected_linear, dtype=np.float64),
            ]
        ),
        "on_frames": on_count,
        "off_frames": off_count,
        "frames": int(frames),
        "duty_cycle": float(duty),
        "injected_shelf_snr_db": float(on["meta"]["injected_shelf_snr_db"]),
        "injected_linear": injected_linear,
    }


def _frontier_for_rank(
    population: Mapping[str, Any],
    *,
    rho: int,
    floor_linear: float,
    bulk_size: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    required = _required_column(population["required"], rho)
    claim = systematic_residuals(population["shelf"], floor_linear=floor_linear)
    unfloored = np.where(
        np.isfinite(population["shelf"]),
        10.0 ** (np.nan_to_num(population["shelf"], nan=-400.0) / 10.0),
        0.0,
    )
    truth = np.asarray(population["injected"], dtype=np.float64)
    candidates = candidate_multipliers_q16(required)
    histogram = residual_score_histogram(
        required, claim, truth, candidates=candidates, rho=rho, bulk_size=bulk_size
    )
    unfloored_histogram = residual_score_histogram(
        required,
        unfloored,
        truth,
        candidates=candidates,
        rho=rho,
        bulk_size=bulk_size,
    )
    points = frontier_points(histogram)
    unfloored_points = {
        point["eta_q16"]: point for point in frontier_points(unfloored_histogram)
    }
    intervals = bootstrap_frontier(
        required,
        claim,
        truth,
        candidates=candidates,
        rho=rho,
        bulk_size=bulk_size,
        replicates=int(replicates),
        seed=int(seed),
    )
    for point in points:
        eta = point["eta_q16"]
        interval = intervals[eta]
        point["masked_fraction_q16"] = interval["masked_fraction"][0]
        point["masked_fraction_q84"] = interval["masked_fraction"][1]
        point["r_sys_q16"] = interval["r_sys"][0]
        point["r_sys_q84"] = interval["r_sys"][1]
        point["r_injected_q16"] = interval["r_injected"][0]
        point["r_injected_q84"] = interval["r_injected"][1]
        point["r_sys_no_floor"] = unfloored_points[eta]["r_sys"]
        point["claim_minus_truth"] = point["r_sys"] - point["r_injected"]
        point["claim_over_truth"] = (
            point["r_sys"] / point["r_injected"]
            if point["r_injected"] > 0.0
            else float("inf")
        )
        point["claim_over_truth_no_floor"] = (
            point["r_sys_no_floor"] / point["r_injected"]
            if point["r_injected"] > 0.0
            else float("inf")
        )
    # The bookkeeping identity: the prefix sums against the frame list.
    audited = 0
    worst = 0.0
    for point in points[:: max(1, len(points) // 20)]:
        direct = kept_frame_means(required, claim, eta_q16=point["eta_q16"])
        if direct["kept"] != point["kept"]:
            raise AssertionError("prefix-sum kept count disagrees with the frame list")
        worst = max(worst, abs(direct["mean"] - point["r_sys"]))
        audited += 1
    return {
        "rho": int(rho),
        "points": points,
        "candidates": len(candidates),
        "prefix_sum_audit_points": audited,
        "prefix_sum_audit_max_abs_difference": float(worst),
    }


def _unmasked_reference(
    population: Mapping[str, Any], *, floor_linear: float
) -> dict[str, Any]:
    """Keep everything: what the claim and the truth say with no mask at all.

    ``transfer_closure`` is the estimator's own accuracy on the injected
    frames alone, with no floor and no mask: the mean linear shelf the frames
    report divided by the linear shelf the bench injected.  It is a property
    of the pilot-excess-to-shelf transfer, not of the masking, and separating
    it from the frontier is what lets the frontier's gap be read as masking.
    """
    shelf = np.asarray(population["shelf"], dtype=np.float64)
    truth = np.asarray(population["injected"], dtype=np.float64)
    claim = systematic_residuals(shelf, floor_linear=floor_linear)
    linear = np.where(np.isfinite(shelf), 10.0 ** (np.nan_to_num(shelf, nan=-400.0) / 10.0), 0.0)
    on = truth > 0.0
    injected_linear = float(population["injected_linear"])
    return {
        "frames": int(shelf.size),
        "frames_with_shelf_estimate": int(np.count_nonzero(np.isfinite(shelf))),
        "on_frames_with_shelf_estimate": int(
            np.count_nonzero(np.isfinite(shelf) & on)
        ),
        "r_sys_unmasked": float(claim.mean()),
        "r_injected_unmasked": float(truth.mean()),
        "r_sys_unmasked_no_floor": float(linear.mean()),
        "claim_over_truth_unmasked": (
            float(claim.mean() / truth.mean()) if truth.mean() > 0.0 else float("inf")
        ),
        "transfer_closure_on_frames": (
            float(linear[on].mean() / injected_linear)
            if on.any() and injected_linear > 0.0
            else float("nan")
        ),
        "transfer_closure_db": (
            float(10.0 * math.log10(linear[on].mean() / injected_linear))
            if on.any() and injected_linear > 0.0 and linear[on].mean() > 0.0
            else float("nan")
        ),
    }


def _spectra(population: Mapping[str, Any], *, required, eta_q16: int) -> dict[str, Any]:
    """The three normalizations of the frame-averaged fine target spectrum.

    ``all`` is the all-frame mean, ``keep_given_keep`` is what a kept frame
    looks like, and ``keep_contribution`` is what survives into the
    integration.  What the mask removes is ``all - keep_contribution``, taken
    in linear power; decibels are the last operation.
    """
    keep = kept_at(np.asarray(required, dtype=object), int(eta_q16))
    fine = np.asarray(population["fine"], dtype=np.float64)
    target = fine[:, 0, :]
    reference = 0.5 * (fine[:, 1, :] + fine[:, 2, :])
    frames = target.shape[0]
    kept = int(np.count_nonzero(keep))
    scale = float(np.median(reference)) or 1.0
    all_mean = target.mean(axis=0) / scale
    if kept:
        keep_given_keep = target[keep].mean(axis=0) / scale
        keep_contribution = target[keep].sum(axis=0) / frames / scale
    else:
        keep_given_keep = np.full(FINE_BINS, np.nan)
        keep_contribution = np.zeros(FINE_BINS)
    return {
        "frames": frames,
        "kept": kept,
        "eta_q16": int(eta_q16),
        "reference_scale": scale,
        "all_mean": all_mean,
        "keep_given_keep": keep_given_keep,
        "keep_contribution": keep_contribution,
        "removed": all_mean - keep_contribution,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(value) for name, value in row.items()})


def _run_report(args, profile) -> int:
    config = _study_config(args, profile)
    bulk_size = int(config["geometry"]["bulk_size"])
    off = _load_shard(_shard_path(args.output_dir, OFF_LABEL))
    floor_bank = _load_shard(_shard_path(args.output_dir, FLOOR_LABEL))
    floor = measured_floor_db(floor_bank["shelf"])
    if floor["evidence"] != "measured":
        raise SystemExit(f"the floor population refused a floor: {floor}")
    floor_linear = 10.0 ** (floor["floor_db"] / 10.0)

    levels: dict[float, dict[str, Any]] = {}
    for shelf in args.shelf_snr_db:
        levels[float(shelf)] = _load_shard(
            _shard_path(args.output_dir, f"on_{_label(float(shelf))}")
        )

    # One composed population size serves every duty cycle, so the curves are
    # comparable: the largest N both banks can supply at every declared duty.
    duties = [float(value) for value in args.duty_cycle]
    composed = int(
        min(
            int(args.frames_on) / max(duties),
            int(args.frames_off) / (1.0 - min(duties)),
        )
    )
    if composed < MIN_RETAINED_FRAMES:
        raise SystemExit("the composed population is smaller than the retained floor")
    frontier_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    for shelf, bank in levels.items():
        for duty in args.duty_cycle:
            population = _duty_population(
                args, off, bank, float(duty), frames=composed
            )
            key = f"snr{_label(shelf)}_duty{int(round(duty * 100)):03d}"
            curves[key] = {
                "shelf_snr_db": float(shelf),
                "duty_cycle": float(duty),
                "frames": population["frames"],
                "on_frames": population["on_frames"],
                "off_frames": population["off_frames"],
                "injected_linear": population["injected_linear"],
                "unmasked_injected_mean": float(population["injected"].mean()),
                "unmasked": _unmasked_reference(
                    population, floor_linear=floor_linear
                ),
                "ranks": {},
            }
            for rho in args.rho:
                result = _frontier_for_rank(
                    population,
                    rho=int(rho),
                    floor_linear=floor_linear,
                    bulk_size=bulk_size,
                    replicates=int(args.bootstrap_replicates),
                    seed=canonical_seed(int(args.seed), "bootstrap", key, int(rho)),
                )
                curves[key]["ranks"][str(int(rho))] = result
                audits.append(
                    {
                        "configuration": key,
                        "rho": int(rho),
                        "points": result["prefix_sum_audit_points"],
                        "max_abs_difference": result[
                            "prefix_sum_audit_max_abs_difference"
                        ],
                    }
                )
                for point in result["points"]:
                    frontier_rows.append(
                        {
                            "configuration": key,
                            "shelf_snr_db": float(shelf),
                            "duty_cycle": float(duty),
                            **{
                                name: point[name]
                                for name in (
                                    "rho",
                                    "eta_q16",
                                    "eta",
                                    "frames",
                                    "kept",
                                    "masked_fraction",
                                    "masked_fraction_q16",
                                    "masked_fraction_q84",
                                    "r_sys",
                                    "r_sys_q16",
                                    "r_sys_q84",
                                    "r_sys_no_floor",
                                    "r_injected",
                                    "r_injected_q16",
                                    "r_injected_q84",
                                    "claim_minus_truth",
                                    "claim_over_truth",
                                    "claim_over_truth_no_floor",
                                )
                            },
                        }
                    )
                # One summary row per (configuration, rho) at the least mask
                # that holds the claim inside a tolerance the claim itself
                # sets: the unmasked injected mean, which is the contamination
                # the mask exists to remove.
                target = float(population["injected"].mean())
                for name, tolerance in (
                    ("half_unmasked_truth", 0.5 * target),
                    ("tenth_unmasked_truth", 0.1 * target),
                ):
                    if tolerance <= 0.0:
                        continue
                    chosen = minimum_mask_envelope(result["points"], tolerance)
                    if chosen is None:
                        summary_rows.append(
                            {
                                "configuration": key,
                                "shelf_snr_db": float(shelf),
                                "duty_cycle": float(duty),
                                "rho": int(rho),
                                "allowance": name,
                                "tolerance": tolerance,
                                "status": "no feasible point",
                                "eta": math.nan,
                                "masked_fraction": math.nan,
                                "r_sys": math.nan,
                                "r_injected": math.nan,
                                "claim_over_truth": math.nan,
                                "truth_inside_allowance": False,
                            }
                        )
                        continue
                    summary_rows.append(
                        {
                            "configuration": key,
                            "shelf_snr_db": float(shelf),
                            "duty_cycle": float(duty),
                            "rho": int(rho),
                            "allowance": name,
                            "tolerance": tolerance,
                            "status": "selected",
                            "eta": chosen["eta"],
                            "masked_fraction": chosen["masked_fraction"],
                            "r_sys": chosen["r_sys"],
                            "r_injected": chosen["r_injected"],
                            "claim_over_truth": chosen["claim_over_truth"],
                            "truth_inside_allowance": bool(
                                chosen["r_injected"] <= tolerance
                            ),
                        }
                    )

    _write_csv(args.output_dir / "frontier_points.csv", frontier_rows)
    _write_csv(args.output_dir / "operating_points.csv", summary_rows)

    # The plates: one loud and one sub-noise shelf at the median rank, at the
    # candidate that masks closest to the injected duty cycle.
    plate_rho = int(args.rho[len(args.rho) // 2])
    plates: dict[str, Any] = {}
    duty_for_plates = float(args.duty_cycle[-1])
    for shelf in (max(args.shelf_snr_db), min(args.shelf_snr_db)):
        bank = levels[float(shelf)]
        population = _duty_population(
            args, off, bank, duty_for_plates, frames=composed
        )
        required = _required_column(population["required"], plate_rho)
        key = f"snr{_label(float(shelf))}_duty{int(round(duty_for_plates * 100)):03d}"
        points = curves[key]["ranks"][str(plate_rho)]["points"]
        chosen = min(
            points,
            key=lambda point: abs(point["masked_fraction"] - duty_for_plates),
        )
        spectra = _spectra(population, required=required, eta_q16=chosen["eta_q16"])
        plates[f"{shelf:+.1f}"] = {
            "shelf_snr_db": float(shelf),
            "regime": LOUD_LABEL if shelf == max(args.shelf_snr_db) else SUBNOISE_LABEL,
            "duty_cycle": duty_for_plates,
            "rho": plate_rho,
            "eta": chosen["eta"],
            "eta_q16": chosen["eta_q16"],
            "masked_fraction": chosen["masked_fraction"],
            "r_sys": chosen["r_sys"],
            "r_injected": chosen["r_injected"],
            "spectra": {
                name: [float(value) for value in spectra[name]]
                for name in ("all_mean", "keep_given_keep", "keep_contribution")
            },
            "kept": spectra["kept"],
            "frames": spectra["frames"],
        }

    # The verdict: does the claim reproduce the truth, and where does it fail?
    with_truth = [
        row for row in frontier_rows if float(row["r_injected"]) > 0.0
    ]
    ratios = np.asarray(
        [float(row["claim_over_truth"]) for row in with_truth], dtype=float
    )
    no_floor = np.asarray(
        [float(row["claim_over_truth_no_floor"]) for row in with_truth], dtype=float
    )
    under = [row for row in with_truth if float(row["claim_over_truth"]) < 1.0]
    zero_truth = [row for row in frontier_rows if float(row["r_injected"]) == 0.0]
    verdict = {
        "evaluable_points": len(frontier_rows),
        "points_with_surviving_injection": len(with_truth),
        "points_with_no_surviving_injection": len(zero_truth),
        "claim_over_truth_min": float(ratios.min()) if ratios.size else math.nan,
        "claim_over_truth_median": (
            float(np.median(ratios)) if ratios.size else math.nan
        ),
        "claim_over_truth_max": float(ratios.max()) if ratios.size else math.nan,
        "claim_over_truth_no_floor_min": (
            float(no_floor.min()) if no_floor.size else math.nan
        ),
        "claim_over_truth_no_floor_median": (
            float(np.median(no_floor)) if no_floor.size else math.nan
        ),
        "under_booking_points": len(under),
        "under_booking_fraction": (
            len(under) / len(with_truth) if with_truth else math.nan
        ),
        "worst_under_booking": (
            min(under, key=lambda row: float(row["claim_over_truth"]))
            if under
            else None
        ),
        "claim_at_zero_truth_min": (
            float(min(float(row["r_sys"]) for row in zero_truth))
            if zero_truth
            else math.nan
        ),
        "by_configuration": {
            key: {
                "claim_over_truth_min": float(
                    min(
                        (
                            float(row["claim_over_truth"])
                            for row in with_truth
                            if row["configuration"] == key
                        ),
                        default=math.nan,
                    )
                ),
                "claim_over_truth_max": float(
                    max(
                        (
                            float(row["claim_over_truth"])
                            for row in with_truth
                            if row["configuration"] == key
                        ),
                        default=math.nan,
                    )
                ),
                "transfer_closure_db": curves[key]["unmasked"][
                    "transfer_closure_db"
                ],
                "claim_over_truth_unmasked": curves[key]["unmasked"][
                    "claim_over_truth_unmasked"
                ],
            }
            for key in curves
        },
    }

    report = {
        "schema_version": MASK_FRONTIER_SCHEMA,
        "created_utc": _utc_now(),
        "config_sha256": config["config_sha256"],
        "reduced_geometry": bool(config["geometry"]["reduced_geometry"]),
        "num_streams": int(config["geometry"]["num_streams"]),
        "floor": {**floor, "floor_linear": floor_linear},
        "populations": {
            OFF_LABEL: off["meta"],
            FLOOR_LABEL: floor_bank["meta"],
            **{f"on_{_label(k)}": v["meta"] for k, v in levels.items()},
        },
        "prefix_sum_audit": audits,
        "verdict": verdict,
        "plates": plates,
        "points_csv": "frontier_points.csv",
        "configurations": {
            key: {
                "shelf_snr_db": value["shelf_snr_db"],
                "duty_cycle": value["duty_cycle"],
                "frames": value["frames"],
                "on_frames": value["on_frames"],
                "off_frames": value["off_frames"],
                "injected_linear": value["injected_linear"],
                "unmasked_injected_mean": value["unmasked_injected_mean"],
                "unmasked": value["unmasked"],
                "candidates_by_rho": {
                    name: value["ranks"][name]["candidates"]
                    for name in value["ranks"]
                },
                "evaluated_points_by_rho": {
                    name: len(value["ranks"][name]["points"])
                    for name in value["ranks"]
                },
            }
            for key, value in curves.items()
        },
    }
    write_json_strict(args.output_dir / "frontier_report.json", report)
    print(f"wrote {args.output_dir / 'frontier_report.json'}")
    print(
        "claim / truth over "
        f"{verdict['points_with_surviving_injection']} points with surviving "
        f"injection: min {verdict['claim_over_truth_min']:.3g}, median "
        f"{verdict['claim_over_truth_median']:.3g}, max "
        f"{verdict['claim_over_truth_max']:.3g}; "
        f"{verdict['under_booking_points']} under-booked"
    )
    print(
        f"floor: {floor['floor_db']:.2f} dB "
        f"({floor['evidence']}, p{FLOOR_PERCENTILE:.0f} of "
        f"{floor['frames_with_shelf_estimate']}/{floor['frames']} off frames)"
    )
    return 0


def _step(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """The empirical staircase, retained without smoothing."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    order = np.argsort(ys)
    return xs[order], ys[order]


def _run_figure(args: argparse.Namespace) -> int:
    from pilot_proxy.plot_style import setup_matplotlib

    setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    report = json.loads(
        (args.output_dir / "frontier_report.json").read_text(encoding="utf-8")
    )
    configurations = {
        key: dict(value) for key, value in report["configurations"].items()
    }
    with (args.output_dir / "frontier_points.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            entry = configurations[row["configuration"]].setdefault(
                row["rho"], []
            )
            entry.append({name: float(value) for name, value in row.items()
                          if name != "configuration"})
    duties = sorted({float(v["duty_cycle"]) for v in configurations.values()})
    shelves = sorted({float(v["shelf_snr_db"]) for v in configurations.values()})
    ranks = [int(value) for value in args.rho]
    floor_linear = float(report["floor"]["floor_linear"])
    reduced = bool(report["reduced_geometry"])
    streams = int(report["num_streams"])

    # Panel (a) is drawn at the shelf the mask can only partly reach: the
    # loudest level the detector does not simply remove outright.
    panel_a_shelf = (
        float(args.panel_shelf_db)
        if args.panel_shelf_db is not None
        else sorted(shelves)[min(1, len(shelves) - 1)]
    )
    if panel_a_shelf not in shelves:
        raise SystemExit(f"--panel-shelf-db {panel_a_shelf} is not a measured level")
    left = 1e-10

    figure = plt.figure(figsize=(11.0, 11.5))
    grid = gridspec.GridSpec(
        3, len(duties), figure=figure, hspace=0.42, wspace=0.30,
        height_ratios=[1.0, 1.0, 0.9],
    )
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for column, duty in enumerate(duties):
        axis = figure.add_subplot(grid[0, column])
        key = f"snr{_label(panel_a_shelf)}_duty{int(round(duty * 100)):03d}"
        entry = configurations[key]
        for index, rho in enumerate(ranks):
            points = entry[str(rho)]
            x = [p["r_injected"] for p in points]
            y = [p["masked_fraction"] for p in points]
            low = [p["r_injected_q16"] for p in points]
            high = [p["r_injected_q84"] for p in points]
            xs, ys = _step(x, y)
            axis.step(
                np.maximum(xs, left), ys, where="post",
                color=colours[index % len(colours)], lw=1.1,
                label=rf"$\rho={rho}$",
            )
            axis.fill_betweenx(
                np.asarray(y)[np.argsort(y)],
                np.maximum(np.asarray(low)[np.argsort(y)], left),
                np.maximum(np.asarray(high)[np.argsort(y)], left),
                color=colours[index % len(colours)], alpha=0.16, lw=0.0,
            )
        # eta labels along the median-rank curve
        median_rho = ranks[len(ranks) // 2]
        points = entry[str(median_rho)]
        for fraction in (0.15, 0.45, 0.75):
            target = min(
                points, key=lambda p: abs(p["masked_fraction"] - fraction)
            )
            axis.annotate(
                rf"$\eta={target['eta']:.3g}$",
                (max(target["r_injected"], left), target["masked_fraction"]),
                textcoords="offset points", xytext=(6, 3), fontsize=7,
                color=colours[(len(ranks) // 2) % len(colours)],
            )
        axis.axvline(
            entry["unmasked_injected_mean"], color="0.45", ls=":", lw=0.9,
            label="injected, no mask" if column == 0 else None,
        )
        axis.set_xscale("log")
        axis.set_xlim(left, max(2.0 * entry["unmasked_injected_mean"], 10.0 * left))
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(
            "surviving injected contamination (linear;\n"
            "the left spine is exactly zero)"
        )
        if column == 0:
            axis.set_ylabel("masked fraction")
        axis.set_title(
            f"(a) duty {duty:.2f}, shelf {panel_a_shelf:+.0f} dB", fontsize=9
        )
        axis.grid(True, which="both", alpha=0.25, lw=0.4)
        if column == 0:
            axis.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

    # Panel (b): what the cumulative sums claim against what was injected.
    axis = figure.add_subplot(grid[1, : max(1, len(duties) - 1)])
    ratio_axis = figure.add_subplot(grid[1, len(duties) - 1])
    markers = {duty: marker for duty, marker in zip(duties, ("o", "s", "^", "v"))}
    for index, shelf in enumerate(shelves):
        for duty in duties:
            key = f"snr{_label(shelf)}_duty{int(round(duty * 100)):03d}"
            entry = configurations[key]
            points = entry[str(ranks[len(ranks) // 2])]
            x = np.asarray([p["r_injected"] for p in points])
            y = np.asarray([p["r_sys"] for p in points])
            m = np.asarray([p["masked_fraction"] for p in points])
            visible = x > 0.0
            axis.plot(
                x[visible], y[visible], markers[duty], ms=2.4,
                color=colours[index % len(colours)], alpha=0.75,
                label=(
                    f"{shelf:+.0f} dB, duty {duty:.2f}"
                    if True
                    else None
                ),
            )
            ratio_axis.plot(
                m[visible], y[visible] / x[visible], markers[duty], ms=2.4,
                color=colours[index % len(colours)], alpha=0.75,
            )
    limits = [1e-9, 1e0]
    axis.plot(limits, limits, color="0.3", lw=0.9, ls="--", label="claim = truth")
    axis.axhline(
        floor_linear, color="0.55", lw=0.9, ls=":",
        label=rf"floor {10 * math.log10(floor_linear):.1f} dB",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*limits)
    axis.set_ylim(*limits)
    axis.set_xlabel("injected residual in the kept frames (truth)")
    axis.set_ylabel("residual the cumulative sums claim")
    axis.set_title(
        r"(b) claim against truth, $\rho=%d$, every $\eta$" % ranks[len(ranks) // 2],
        fontsize=9,
    )
    axis.grid(True, which="both", alpha=0.25, lw=0.4)
    axis.legend(fontsize=5.5, ncol=2, loc="upper left", framealpha=0.85)
    ratio_axis.axhline(1.0, color="0.3", lw=0.9, ls="--")
    ratio_axis.set_yscale("log")
    ratio_axis.set_xlabel("masked fraction")
    ratio_axis.set_ylabel("claim / truth")
    ratio_axis.set_title("(b) over-booking factor", fontsize=9)
    ratio_axis.grid(True, which="both", alpha=0.25, lw=0.4)

    # Panel (c): the plates, one loud and one sub-noise shelf.
    plates = report["plates"]
    ordered = sorted(plates.values(), key=lambda p: -float(p["shelf_snr_db"]))
    anchor = int(report["populations"][OFF_LABEL]["anchor_bin"])
    for column, plate in enumerate(ordered):
        if column >= len(duties):
            break
        axis = figure.add_subplot(grid[2, column])
        shift = FINE_BINS // 2 - anchor
        bins = np.arange(FINE_BINS) - FINE_BINS // 2
        for label, style in (
            ("all_mean", "-"),
            ("keep_given_keep", "--"),
        ):
            values = np.roll(
                np.asarray(plate["spectra"][label], dtype=float), shift
            )
            axis.plot(
                bins, 10.0 * np.log10(np.maximum(values, 1e-18)), style, lw=1.0,
                label={
                    "all_mean": r"$\bar P_{\rm all}$",
                    "keep_given_keep": r"$\bar P_{\rm keep|keep}$",
                }[label],
            )
        axis.set_xlabel("fine bin relative to the anchor")
        if column == 0:
            axis.set_ylabel("fine power (dB, arbitrary zero)")
        axis.set_title(
            f"(c) {plate['regime']} shelf {plate['shelf_snr_db']:+.0f} dB, "
            f"duty {plate['duty_cycle']:.2f}, masked {plate['masked_fraction']:.2f}",
            fontsize=8,
        )
        axis.grid(True, alpha=0.25, lw=0.4)
        axis.legend(fontsize=7)

    if len(duties) > len(ordered):
        axis = figure.add_subplot(grid[2, len(ordered)])
        for plate in ordered:
            shift = FINE_BINS // 2 - anchor
            removed = np.roll(
                np.asarray(plate["spectra"]["all_mean"], dtype=float)
                - np.asarray(plate["spectra"]["keep_contribution"], dtype=float),
                shift,
            )
            axis.plot(
                np.arange(FINE_BINS) - FINE_BINS // 2, removed, lw=1.0,
                label=f"{plate['shelf_snr_db']:+.0f} dB",
            )
        axis.set_yscale("symlog", linthresh=1e-4)
        axis.set_xlabel("fine bin relative to the anchor")
        axis.set_ylabel(r"$\bar P_{\rm all}-\bar P_{\rm keep,contrib}$")
        axis.set_title("(c) what the mask removed, linear power", fontsize=8)
        axis.grid(True, alpha=0.25, lw=0.4)
        axis.legend(fontsize=7)

    banner = (
        f"K=128, L=128, M={streams} streams, fine designated set, exact Q16 rule"
        + (
            "  --  REDUCED GEOMETRY: code validation, not evidence"
            if reduced
            else ""
        )
    )
    figure.suptitle(
        "Synthetic mask-versus-residual frontier against injected truth\n" + banner,
        fontsize=10,
    )
    path = args.output_dir / "mask_residual_frontier.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")
    return 0


def run(args: argparse.Namespace) -> int:
    profile = _prepare(args)
    if args.stage == "report":
        return _run_report(args, profile)
    if args.stage == "figure":
        return _run_figure(args)
    gpu = cgs._initialize_gpu(args)
    if gpu is None:
        raise SystemExit("--gpu is required: the frame generator runs on the device")
    if args.stage == "generate":
        return _run_generate(args, profile, gpu)
    if args.stage == "audit":
        return _run_audit(args, profile, gpu)
    raise AssertionError(args.stage)


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

# coding=utf-8
"""Every literal the project profile replaces, and the loader value that reproduces it.

Each row names where the literal sat (``file:line`` at pilot-proxy bdfcdce or
RFIsher e041d5e), the literal exactly as it was written (so its float arithmetic
is repeated here), and the profile value that stands in for it. The profile value
must have the same type and compare exactly equal.

``replaced`` rows were removed from pilot-proxy ``src/`` and now read the
profile; ``REPLACED_NAMES`` below checks that they are no longer literals.
``pinned`` rows are copies that stay where they are for now (the detector's own
constants, RFIsher, tools and frozen records); the refactor switches them to the
profile in later milestones, and these rows fix the value they must keep.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from pilot_proxy import paths  # noqa: E402
from pilot_proxy.config.instrument import nyquist_sign  # noqa: E402
from pilot_proxy.config.project import default_project  # noqa: E402

P = default_project()
DATA = Path(__file__).resolve().parent / "data"


def _screened():
    return P.frequency_plan.bands("screened")


def _labels_as_ints(role="screened"):
    return tuple(int(label) for label in P.frequency_plan.labels(role))


def _freq_ids(role="screened"):
    return [P.target_freq_id(b) for b in P.frequency_plan.bands(role)]


# (location, status, literal as written, profile value)
LITERALS = {
    # ---- frequency plan and interference template -------------------------------
    "atsc_channel_width": (
        "pilot-proxy src/pilot_proxy/atsc_channels.py:12 ATSC_CHANNEL_WIDTH_HZ",
        "replaced", 6.0e6, lambda: P.interference.band_width_hz),
    "atsc_pilot_offset": (
        "pilot-proxy src/pilot_proxy/atsc_channels.py:13 ATSC_PILOT_OFFSET_HZ",
        "replaced", 309_441.0, lambda: P.interference.marker.offset_hz),
    "atsc_ch14_lower_edge": (
        "pilot-proxy src/pilot_proxy/atsc_channels.py:17 ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ",
        "replaced", 470.0e6, lambda: _screened()[0].low_hz),
    "dtv_bandwidth": (
        "pilot-proxy src/pilot_proxy/dtv_units.py:27 DTV_BANDWIDTH_HZ",
        "replaced", 6.0e6, lambda: P.interference.band_width_hz),
    "pilot_below_data_db": (
        "pilot-proxy src/pilot_proxy/dtv_units.py:39 PILOT_BELOW_DATA_DB",
        "replaced", 11.3, lambda: P.interference.marker.marker_to_band_db),
    "pilot_capture_efficiency": (
        "pilot-proxy src/pilot_proxy/dtv_units.py:40 PILOT_CAPTURE_EFFICIENCY",
        "replaced", 1.0, lambda: P.interference.marker.capture_efficiency),
    "rfisher_pilot_below_shelf_db": (
        "RFIsher src/rfisher/selection_policy.py transfer.nominal_pilot_below_shelf_db",
        "pinned", 11.3, lambda: P.interference.marker.marker_to_band_db),
    "rfisher_pilot_capture_efficiency": (
        "RFIsher src/rfisher/selection_policy.py transfer.pilot_capture_efficiency",
        "pinned", 1.0, lambda: P.interference.marker.capture_efficiency),
    "rfisher_ch14_lower_edge_mhz": (
        "RFIsher src/rfisher/channels.py:57 ATSC_CH14_LOWER_EDGE",
        "pinned", 470.0, lambda: _screened()[0].low_mhz),
    "rfisher_atsc_width_mhz": (
        "RFIsher src/rfisher/channels.py:58 ATSC_WIDTH",
        "pinned", 6.0, lambda: _screened()[0].high_mhz - _screened()[0].low_mhz),
    "rfisher_atsc_channels": (
        "RFIsher src/rfisher/channels.py:59 ATSC_DTV_CHANNELS",
        "pinned", tuple(range(14, 37)), _labels_as_ints),
    "rfisher_atsc_upper_edge_mhz": (
        "RFIsher src/rfisher/channels.py:60 ATSC_DTV_UPPER_EDGE",
        "pinned", 470.0 + len(range(14, 37)) * 6.0, lambda: _screened()[-1].high_mhz),
    "rfisher_acceptance_channels": (
        "RFIsher src/rfisher/archive_acceptance.py:21 EXPECTED_CHANNELS",
        "pinned", tuple(range(14, 37)), _labels_as_ints),
    "rfisher_tolerance_channels": (
        "RFIsher src/rfisher_results/archive/tolerances.py:24 CHANNELS",
        "pinned", tuple(range(14, 37)), _labels_as_ints),
    "export_expected_channels": (
        "pilot-proxy src/pilot_proxy/dissertation_exports.py:372 range(14, 37) (frozen v1 export)",
        "pinned", set(range(14, 37)), lambda: set(_labels_as_ints())),
    "rfisher_pilot_base_hz": (
        "RFIsher src/rfisher/archive_acceptance.py:30 EXPECTED_PILOT_BASE_HZ",
        "pinned", 470.309441e6, lambda: P.marker_hz(_screened()[0])),
    "rfisher_pilot_spacing_hz": (
        "RFIsher src/rfisher/archive_acceptance.py:31 EXPECTED_PILOT_SPACING_HZ",
        "pinned", 6e6, lambda: P.marker_hz(_screened()[1]) - P.marker_hz(_screened()[0])),
    "audit_pilot_base_mhz": (
        "pilot-proxy tools/audit_per_pilot.py:35 PILOT_BASE_MHZ",
        "pinned", 470.309441, lambda: P.marker_hz(_screened()[0]) / 1e6),
    "quantize_default_pilot_hz": (
        "pilot-proxy src/pilot_proxy/testbench/quantize.py:38 DEFAULT_DTV_PILOT_HZ",
        "pinned", 470_309_441.0, lambda: P.marker_hz(_screened()[0])),
    "quantize_channel_width": (
        "pilot-proxy src/pilot_proxy/testbench/quantize.py:36 ATSC_CHANNEL_WIDTH_HZ",
        "pinned", 6.0e6, lambda: P.interference.band_width_hz),
    "quantize_pilot_offset": (
        "pilot-proxy src/pilot_proxy/testbench/quantize.py:37 ATSC_PILOT_OFFSET_HZ",
        "pinned", 309_441.0, lambda: P.interference.marker.offset_hz),
    "generator_channel_width": (
        "pilot-proxy src/pilot_proxy/testbench/generate_atsc_signal.py:17 ATSC_CHANNEL_WIDTH_HZ",
        "pinned", 6.0e6, lambda: P.interference.band_width_hz),
    # ---- freq_id maps -----------------------------------------------------------
    "audit_freq_table": (
        "pilot-proxy tools/audit_per_pilot.py:36 FREQ_TABLE",
        "pinned",
        {14 + i: fid for i, fid in enumerate(
            [844, 829, 813, 798, 783, 767, 752, 736, 721, 706, 690, 675, 660,
             644, 629, 614, 598, 583, 568, 552, 537, 521, 506])},
        lambda: dict(zip(_labels_as_ints(), _freq_ids()))),
    "frame_policy_pilot_freq_ids": (
        "pilot-proxy tools/capture/frame_analysis/frame_policy_v1.py:15 PILOT (record c49ae37f)",
        "pinned",
        dict(zip(range(14, 37), [844, 829, 813, 798, 783, 767, 752, 736,
                                 721, 706, 690, 675, 660, 644, 629, 614, 598, 583, 568,
                                 552, 537, 521, 506])),
        lambda: dict(zip(_labels_as_ints(), _freq_ids()))),
    "control_target_freq_id": (
        "amendment 18 (2026-09-24) decision 13: channel 37 control read at freq_id 491",
        "pinned", {37: 491}, lambda: dict(zip(_labels_as_ints("control"), _freq_ids("control")))),
    "control_target_is_nominal_marker_channel": (
        "amendment 18 decision 13: 491 is the channel of the nominal pilot position 608.309 MHz",
        "pinned", 491,
        lambda: P.instrument.freq_id_of_hz(P.marker_hz(P.frequency_plan.band("37")))),
    # ---- instrument -------------------------------------------------------------
    "chime_band_top": (
        "pilot-proxy src/pilot_proxy/archive/chime_coarse.py:24 CHIME_BAND_TOP_HZ",
        "replaced", 800_000_000.0, lambda: P.instrument.f0_hz),
    "chime_n_coarse": (
        "pilot-proxy src/pilot_proxy/archive/chime_coarse.py:25 CHIME_N_COARSE_CHANNELS",
        "replaced", 1024, lambda: P.instrument.n_channels),
    "chime_coarse_width": (
        "pilot-proxy src/pilot_proxy/archive/chime_coarse.py:26 CHIME_COARSE_WIDTH_HZ",
        "replaced", 400_000_000.0 / 1024, lambda: P.instrument.channel_width_hz),
    "chime_products_sample_rate": (
        "pilot-proxy src/pilot_proxy/chime/products.py:59 SAMPLE_RATE_HZ",
        "replaced", 390_625.0, lambda: P.instrument.sample_rate_hz),
    "drao_longitude": (
        "pilot-proxy src/pilot_proxy/archive_health.py:81 DRAO_LONGITUDE_DEGREES_EAST",
        "replaced", -119.6175, lambda: P.instrument.site_longitude_deg),
    "local_time_zone": (
        "pilot-proxy src/pilot_proxy/archive_health.py:82 LOCAL_CIVIL_TIME_ZONE",
        "replaced", "America/Vancouver", lambda: P.instrument.local_time_zone),
    "hdf5_coarse_width": (
        "pilot-proxy src/pilot_proxy/chime/hdf5_input.py:26 CHIME_COARSE_WIDTH_HZ",
        "pinned", 400_000_000.0 / 1024.0, lambda: P.instrument.channel_width_hz),
    "hdf5_band_top": (
        "pilot-proxy src/pilot_proxy/chime/hdf5_input.py:343,350 800_000_000.0",
        "pinned", 800_000_000.0, lambda: P.instrument.f0_hz),
    "detector_analyzer_default_top": (
        "pilot-proxy src/pilot_proxy/archive/detector.py:504 800_000_000.0 (reset from the "
        "instrument in begin())", "pinned", 800_000_000.0, lambda: P.instrument.f0_hz),
    "runner_sample_rate": (
        "pilot-proxy src/pilot_proxy/chime/runner.py:342 390_625.0",
        "pinned", 390_625.0, lambda: P.instrument.sample_rate_hz),
    "cleaning_tradeoff_width_mhz": (
        "pilot-proxy src/pilot_proxy/chime/cleaning_tradeoff.py:37 "
        "CHIME_COARSE_CHANNEL_BANDWIDTH_MHZ",
        "pinned", 400.0 / 1024.0, lambda: P.instrument.bandwidth_mhz / P.instrument.n_channels),
    "reference_bandwidth": (
        "pilot-proxy src/pilot_proxy/dtv_units.py:24 REFERENCE_BANDWIDTH_HZ (detector)",
        "pinned", 400.0e6, lambda: P.instrument.bandwidth_mhz * 1e6),
    "reference_num_channels": (
        "pilot-proxy src/pilot_proxy/dtv_units.py:25 REFERENCE_NUM_CHANNELS (detector)",
        "pinned", 1024, lambda: P.instrument.n_channels),
    "loader_default_nfft": (
        "pilot-proxy src/pilot_proxy/config/instrument.py DEFAULT_NFFT (loader fallback)",
        "pinned", 16384, lambda: P.instrument.nfft),
    "generate_results_coarse_mhz": (
        "pilot-proxy scripts/generate_results.py:89 CHIME_COARSE_MHZ",
        "pinned", 400.0 / 1024.0, lambda: P.instrument.bandwidth_mhz / P.instrument.n_channels),
    "audit_hz_per_channel": (
        "pilot-proxy tools/audit_per_pilot.py:34 CHIME_HZ_PER_CHANNEL",
        "pinned", 400e6 / 1024.0, lambda: P.instrument.channel_width_hz),
    "rfisher_products_sample_rate": (
        "RFIsher src/rfisher_results/archive/products.py:35 SAMPLE_RATE_HZ",
        "pinned", 390625.0, lambda: P.instrument.sample_rate_hz),
    "rfisher_acceptance_sample_rate": (
        "RFIsher src/rfisher/archive_acceptance.py:23 EXPECTED_SAMPLE_RATE_HZ",
        "pinned", 390625.0, lambda: P.instrument.sample_rate_hz),
    "rfisher_acceptance_input_streams": (
        "RFIsher src/rfisher/archive_acceptance.py:24 EXPECTED_INPUT_STREAMS",
        "pinned", 2048, lambda: P.instrument.n_inputs),
    "rfisher_acceptance_sense": (
        "RFIsher src/rfisher/archive_acceptance.py:26 EXPECTED_SENSE",
        "pinned", -1, lambda: nyquist_sign(P.instrument.nyquist_zone)),
    "rfisher_frequency_min_mhz": (
        "RFIsher src/rfisher/constants.py:5 CHIME_FREQUENCY_MIN_MHZ",
        "pinned", 400.0, lambda: P.instrument.f0_mhz - P.instrument.bandwidth_mhz),
    "rfisher_frequency_max_mhz": (
        "RFIsher src/rfisher/constants.py:6 CHIME_FREQUENCY_MAX_MHZ",
        "pinned", 800.0, lambda: P.instrument.f0_mhz),
    # ---- detector geometry ------------------------------------------------------
    "detector_window": (
        "pilot-proxy src/pilot_proxy/detector_constants.py:4 DEFAULT_DETECTOR_WINDOW_SAMPLES "
        "(detector)", "pinned", 128, lambda: P.detector_config.detector_window),
    "dtv_units_detector_window": (
        "pilot-proxy src/pilot_proxy/dtv_units.py:28 DETECTOR_WINDOW_SAMPLES (detector)",
        "pinned", 128, lambda: P.detector_config.detector_window),
    "fine_bins": (
        "pilot-proxy src/pilot_proxy/fine_decision.py:78 FINE_BINS (detector)",
        "pinned", 256, lambda: P.detector_config.fine_bins),
    "generate_results_fine_bin_hz": (
        "pilot-proxy scripts/generate_results.py:90 FINE_BIN_HZ",
        "pinned", 390625.0 / 128.0, lambda: P.detector_config.detector_bin_hz(P.instrument)),
    "rfisher_products_nfft": (
        "RFIsher src/rfisher_results/archive/products.py:36 NFFT",
        "pinned", 16384, lambda: P.detector_config.nfft),
    "rfisher_products_psd_bin": (
        "RFIsher src/rfisher_results/archive/products.py:37 PSD_BIN_HZ",
        "pinned", 390625.0 / 16384, lambda: P.detector_config.psd_bin_hz(P.instrument)),
    "rfisher_products_detector_window": (
        "RFIsher src/rfisher_results/archive/products.py:38 DETECTOR_WINDOW",
        "pinned", 128, lambda: P.detector_config.detector_window),
    "rfisher_products_coarse_bin": (
        "RFIsher src/rfisher_results/archive/products.py:39 COARSE_BIN_HZ",
        "pinned", 390625.0 / 128, lambda: P.detector_config.detector_bin_hz(P.instrument)),
    "rfisher_products_fine_bins": (
        "RFIsher src/rfisher_results/archive/products.py:40 FINE_BINS",
        "pinned", 256, lambda: P.detector_config.fine_bins),
    "rfisher_products_fine_bin": (
        "RFIsher src/rfisher_results/archive/products.py:41 FINE_BIN_HZ",
        "pinned", 390625.0 / 128 / 256, lambda: P.detector_config.envelope_bin_hz(P.instrument)),
    "rfisher_acceptance_nfft": (
        "RFIsher src/rfisher/archive_acceptance.py:22 EXPECTED_NFFT",
        "pinned", 16384, lambda: P.detector_config.nfft),
    "rfisher_acceptance_detector_window": (
        "RFIsher src/rfisher/archive_acceptance.py:25 EXPECTED_DETECTOR_WINDOW",
        "pinned", 128, lambda: P.detector_config.detector_window),
    "rfisher_coarse_null_dof": (
        "RFIsher src/rfisher_results/archive/nulls.py:92 COARSE_DOF",
        "pinned", (524288, 1048576), lambda: P.detector_config.coarse_null_dof(P.instrument)),
    "rfisher_histogram_null_p": (
        "RFIsher scripts/compare_coarse_histograms_v5.py P = 262144",
        "pinned", 262144, lambda: P.detector_config.summed_terms(P.instrument)),
    # ---- integration model ------------------------------------------------------
    "rfisher_frame_seconds": (
        "RFIsher src/rfisher/constants.py:4 CHIME_FRAME_SECONDS",
        "pinned", 16384 * 2.56e-6, lambda: P.integration_model.frame_seconds),
    "rfisher_products_frame_seconds": (
        "RFIsher src/rfisher_results/archive/products.py:42 FRAME_SECONDS",
        "pinned", 16384 * 2.56e-6, lambda: P.detector_config.frame_seconds(P.instrument)),
    "rfisher_coherence_cap": (
        "RFIsher src/rfisher/residual.py:110 MAX_TAU_C_SECONDS (= correlation.sidereal_day_seconds)",
        "pinned", 86164.0905, lambda: P.integration_model.coherence_cap_seconds),
    "ruling_frame_seconds": (
        "pilot-proxy tools/capture/frame_analysis/ruling_baseline.py:17 TF (record)",
        "pinned", 16384 * 2.56e-6, lambda: P.integration_model.frame_seconds),
    "ruling_cap": (
        "pilot-proxy tools/capture/frame_analysis/ruling_baseline.py:17 CAP (record)",
        "pinned", 86164.0905, lambda: P.integration_model.coherence_cap_seconds),
    # ---- eras -------------------------------------------------------------------
    "rfisher_sign_on_off_through": (
        "RFIsher src/rfisher/residual.py:135 SIGN_ON_OFF_THROUGH",
        "pinned", {35: "2021-10"}, lambda: P.eras.off_through()),
    "rfisher_sign_off_from": (
        "RFIsher src/rfisher/residual.py:152 SIGN_OFF_FROM",
        "pinned", {19: "2024-12", 20: "2022-09", 26: "2023-04", 27: "2022-10", 32: "2023-02"},
        lambda: P.eras.off_from()),
    "release_author_eras_file": (
        "results/archive_author_eras_2026-09-23/author_eras.json (byte copy)",
        "pinned", "a988e813fcaca817670aa88b4db59b9e5ad92f130159a46524bc0bb722c95ffd",
        lambda: P.eras.source_sha256),
    "release_era_overrides_digest": (
        "results/archive_author_eras_2026-09-23/archive/ledger/run.json era_overrides_sha256",
        "pinned", "81f5f20e075c8f6bde41928ecfdd8215b87488b2ee3ff512b5c9ab8343c81aae",
        lambda: P.eras.overrides_sha256()),
    # ---- register ---------------------------------------------------------------
    "min_frames_per_false_alarm": (
        "pilot-proxy src/pilot_proxy/chime/injection_recovery.py:48 MIN_FRAMES_PER_FALSE_ALARM",
        "replaced", 10.0, lambda: P.register.value("detection.minimum_frames_per_false_alarm")),
    "ruling_allowance_db": (
        "pilot-proxy tools/capture/frame_analysis/ruling_baseline.py:17 ALLOW_DB (record)",
        "pinned", 3.0, lambda: P.register.value("measurement_allowance_db")),
    "rfisher_min_retained_frames": (
        "RFIsher src/rfisher/selection_policy.py selection.minimum_retained_frames",
        "pinned", 30, lambda: P.register.value("selection.minimum_retained_frames")),
    "mask_frontier_min_retained_frames": (
        "pilot-proxy src/pilot_proxy/testbench/mask_frontier.py:53 MIN_RETAINED_FRAMES",
        "pinned", 30, lambda: P.register.value("selection.minimum_retained_frames")),
    "mask_frontier_floor_percentile": (
        "pilot-proxy src/pilot_proxy/testbench/mask_frontier.py:54 FLOOR_PERCENTILE",
        "pinned", 90.0, lambda: P.register.value("floor.upper_percentile")),
    "mask_frontier_floor_min_frames": (
        "pilot-proxy src/pilot_proxy/testbench/mask_frontier.py:55 FLOOR_MIN_FRAMES",
        "pinned", 30, lambda: P.register.value("floor.minimum_null_frames")),
    "rfisher_floor_percentile": (
        "RFIsher src/rfisher_results/archive/nulls.py:99 FLOOR_PERCENTILE",
        "pinned", 90.0, lambda: P.register.value("floor.upper_percentile")),
    "rfisher_floor_min_frames": (
        "RFIsher src/rfisher_results/archive/nulls.py:100 FLOOR_MIN_FRAMES",
        "pinned", 30, lambda: P.register.value("floor.minimum_null_frames")),
    "rfisher_min_null_frames": (
        "RFIsher src/rfisher_results/archive/nulls.py:101 MIN_NULL_FRAMES",
        "pinned", 30, lambda: P.register.value("floor.minimum_null_frames")),
    "rfisher_as_coded_probes": (
        "RFIsher src/rfisher_results/archive/nulls.py:97 AS_CODED_PROBES",
        "pinned", ((32.0, 1.0), (5.0, 1.96), (0.3, 2.9677)),
        lambda: P.register.value("floor.null_scale_probes")),
}


@pytest.mark.parametrize("key", sorted(LITERALS))
def test_profile_reproduces_literal(key):
    location, status, literal, profile_value = LITERALS[key]
    assert status in {"replaced", "pinned"}, location
    value = profile_value()
    assert type(value) is type(literal), (location, value, literal)
    assert value == literal, (location, value, literal)
    if isinstance(literal, float):
        assert math.copysign(1.0, value) == math.copysign(1.0, literal)
        assert value.hex() == literal.hex(), location


# Module-level names whose literal was replaced by a profile read, per file.
REPLACED_NAMES = {
    "atsc_channels.py": {
        "ATSC_CHANNEL_WIDTH_HZ": "atsc_channel_width",
        "ATSC_PILOT_OFFSET_HZ": "atsc_pilot_offset",
        "ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ": "atsc_ch14_lower_edge",
    },
    "archive/chime_coarse.py": {
        "CHIME_BAND_TOP_HZ": "chime_band_top",
        "CHIME_N_COARSE_CHANNELS": "chime_n_coarse",
        "CHIME_COARSE_WIDTH_HZ": "chime_coarse_width",
    },
    "archive_health.py": {
        "DRAO_LONGITUDE_DEGREES_EAST": "drao_longitude",
        "LOCAL_CIVIL_TIME_ZONE": "local_time_zone",
    },
    "chime/injection_recovery.py": {
        "MIN_FRAMES_PER_FALSE_ALARM": "min_frames_per_false_alarm",
    },
    "chime/products.py": {
        "SAMPLE_RATE_HZ": "chime_products_sample_rate",
    },
    "dtv_units.py": {
        "DTV_BANDWIDTH_HZ": "dtv_bandwidth",
        "PILOT_BELOW_DATA_DB": "pilot_below_data_db",
        "PILOT_CAPTURE_EFFICIENCY": "pilot_capture_efficiency",
    },
}


def _assigned_literals(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value
    return found


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except ValueError:
        return isinstance(node, ast.BinOp) and _is_literal(node.left) and _is_literal(node.right)


@pytest.mark.parametrize("relative", sorted(REPLACED_NAMES))
def test_replaced_names_read_the_profile(relative):
    import importlib

    path = paths.PACKAGE_ROOT / relative
    assigned = _assigned_literals(path)
    module = importlib.import_module(
        "pilot_proxy." + relative.removesuffix(".py").replace("/", "."))
    for name, key in REPLACED_NAMES[relative].items():
        assert name in assigned, f"{relative}: {name} is no longer defined"
        assert not _is_literal(assigned[name]), f"{relative}: {name} is still a literal"
        location, status, literal, _ = LITERALS[key]
        assert status == "replaced", key
        value = getattr(module, name)
        assert type(value) is type(literal) and value == literal, (relative, name)


def test_every_replaced_row_is_checked_in_its_module():
    checked = {key for names in REPLACED_NAMES.values() for key in names.values()}
    replaced = {key for key, row in LITERALS.items() if row[1] == "replaced"}
    assert checked == replaced


def test_screened_bands_follow_the_uhf_raster():
    # The explicit list reproduces 470 + 6 (ch - 14) MHz edges and the pilot
    # 309.441 kHz above each lower edge, bit for bit, on every screened band.
    for band in _screened():
        ch = int(band.label)
        low = 470.0e6 + (ch - 14) * 6.0e6
        assert band.low_hz == low and band.high_hz == low + 6.0e6
        assert P.marker_hz(band) == low + 309_441.0
    control = P.frequency_plan.band("37")
    assert (control.low_mhz, control.high_mhz, control.role) == (608.0, 614.0, "control")


def test_hz_freq_id_helpers_agree_on_every_target():
    from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz

    for band in P.frequency_plan.bands():
        marker = P.marker_hz(band)
        assert chime_freq_id_from_hz(marker) == P.instrument.freq_id_of_hz(marker)
        assert P.instrument.freq_id_of_freq(marker / 1e6) == P.instrument.freq_id_of_hz(marker)
        centre = P.instrument.hz_of_freq_id(P.target_freq_id(band))
        assert abs(centre - marker) <= P.instrument.channel_width_hz / 2


def test_register_copies_the_detector_entries_value_for_value():
    golden = json.loads((DATA / "selection_policy_e041d5e_detector_entries.json").read_text())
    assert golden["snapshot_sha256"] == P.register.source["snapshot_sha256"]
    copied = [d for d in golden["decisions"] if not d["id"].startswith("transfer.")]
    assert len(copied) == 59
    for record in copied:
        mine = P.register.decision(record["id"]).record()
        assert mine.pop("side") == "detector"
        assert mine == record, record["id"]
    added = set(P.register.ids()) - {d["id"] for d in copied}
    assert added == set(P.register.source["added"])
    transfer = {d["id"]: d["value"] for d in golden["decisions"] if d["id"].startswith("transfer.")}
    marker = P.interference.marker
    assert transfer == {"transfer.nominal_pilot_below_shelf_db": marker.marker_to_band_db,
                        "transfer.pilot_capture_efficiency": marker.capture_efficiency}


# A pinned pilot-proxy row that names a module-level constant is also held to the
# code: the name must still be assigned that literal where the row says it is.
# (Rows citing an inline literal, RFIsher or a record file are checked by the
# golden files and the rows above.)
_PINNED_NAME = re.compile(r"^pilot-proxy (\S+?)(?::\d+(?:,\d+)*)? ([A-Z][A-Z0-9_]*)\b")
_PURE_BUILTINS = {"dict": dict, "enumerate": enumerate, "range": range, "set": set,
                  "tuple": tuple, "zip": zip}


def _pinned_code_rows():
    rows = []
    for key, (location, status, _literal, _) in sorted(LITERALS.items()):
        match = _PINNED_NAME.match(location)
        if status == "pinned" and match:
            rows.append((key, match.group(1), match.group(2)))
    return rows


def _evaluate_pure(node: ast.AST):
    """Evaluate an assignment built only from literals and pure builtins."""
    loop_names = {sub.id for comp in ast.walk(node) if isinstance(comp, ast.comprehension)
                  for sub in ast.walk(comp.target) if isinstance(sub, ast.Name)}
    free = {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}
    free -= loop_names | set(_PURE_BUILTINS)
    if free:
        pytest.fail(f"not a pure literal ({sorted(free)}): {ast.unparse(node)}")
    code = compile(ast.Expression(node), "<pinned>", "eval")
    return eval(code, {"__builtins__": dict(_PURE_BUILTINS)})  # noqa: S307


def _module_assignment(path: Path, name: str):
    """The module-level value node assigned to ``name`` in ``path``, or None.

    A record script may use syntax newer than this interpreter (the capture
    ruling's PEP 701 f-strings need Python 3.12). Its constants sit on one-line
    top-level statements, so those lines are parsed on their own instead.
    """
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        assignment = re.compile(rf"^(?:[^#]*;\s*)?{re.escape(name)}\s*=")
        lines = [line for line in text.splitlines() if assignment.match(line)]
        tree = ast.parse("\n".join(lines), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def test_pinned_code_rows_are_found():
    names = {key for key, _, _ in _pinned_code_rows()}
    assert {"detector_window", "fine_bins", "reference_bandwidth", "hdf5_coarse_width",
            "quantize_default_pilot_hz", "audit_freq_table", "frame_policy_pilot_freq_ids",
            "ruling_cap", "mask_frontier_floor_percentile",
            "cleaning_tradeoff_width_mhz", "loader_default_nfft"} <= names


@pytest.mark.parametrize("key,relative,name", _pinned_code_rows(),
                         ids=[row[0] for row in _pinned_code_rows()])
def test_pinned_row_matches_the_code(key, relative, name):
    root = paths.SOURCE_CHECKOUT_ROOT
    if root is None:
        pytest.skip("needs a source checkout")
    location, _status, literal, _ = LITERALS[key]
    node = _module_assignment(root / relative, name)
    assert node is not None, f"{location}: {name} is not assigned at module level"
    value = _evaluate_pure(node)
    assert type(value) is type(literal) and value == literal, (location, value, literal)
    if isinstance(literal, float):
        assert value.hex() == literal.hex(), location

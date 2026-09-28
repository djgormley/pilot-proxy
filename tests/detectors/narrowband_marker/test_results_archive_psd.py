"""The per-frame spectrum containment analysis on synthetic v5 products."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.detectors.narrowband_marker import psd
from pilot_proxy.products.reader import NFFT, PSD_BIN_HZ, SAMPLE_RATE_HZ, Product

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("v5_fixture", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)

INVALID = -32768
FLOOR_REFERENCE = 10.0 ** 0.3       # a second per-frame reference whose floor code (-300) decodes exactly to 1.0
PEAK_DB, CURVATURE_DB, SUB_BIN = 20.0, 6.0, 0.3


def _weight(rf, lo, hi):
    """Fraction of a 23.84 Hz bin centred at rf inside [lo, hi], written out longhand."""
    return max(0.0, min(1.0, (min(rf + PSD_BIN_HZ / 2, hi) - max(rf - PSD_BIN_HZ / 2, lo)) / PSD_BIN_HZ))


def _bin_of(geometry, rf_hz):
    return int(round(float(geometry.psd_bin_of_rf_offset(rf_hz)))) % NFFT


def _write_psd_product(tmp_path, channel, lines, *, excursion=None):
    """A v5 fixture with per-frame spectra: a flat floor of 1.0, single-bin lines
    ``(rf_hz, dB)``, a three-bin log-parabolic pilot ``("pilot", rf_hz)`` with a
    sub-bin shift, a centre-line tone ``("centre", dB)``, and, on the frames of
    ``excursion = (frames, rf_hz, dB)``, one extra strong line.

    Returns (path, geometry, expected-linear-spectrum, {name: (bin, rf)}).
    """
    path = v5_fixture._write_product(tmp_path / f"{channel}.npz", channel)
    with Product(path) as p:
        frames = p.n_frames
        g = p.geometry
    v5_fixture._replace(path, baseband_power_linear=np.full((frames, 1), 4.0))
    with Product(path) as p:
        g = p.geometry
    spectrum = np.ones(NFFT)
    marks = {}
    for line in lines:
        if line[0] == "pilot":
            b0 = _bin_of(g, line[1])
            rf0 = float(g.psd_rf_offset_hz([b0])[0])
            for b in (b0 - 1, b0, b0 + 1):
                x = (float(g.psd_rf_offset_hz([b % NFFT])[0]) - rf0) / PSD_BIN_HZ
                spectrum[b % NFFT] *= 10.0 ** ((PEAK_DB - CURVATURE_DB * (x - SUB_BIN) ** 2) / 10.0)
            marks["pilot"] = (b0, rf0)
        elif line[0] == "centre":
            b = _bin_of(g, g.centre_line_rf_offset_hz)
            spectrum[b] *= 10.0 ** (line[1] / 10.0)
            marks["centre"] = (b, float(g.psd_rf_offset_hz([b])[0]))
        else:
            rf, db = line
            b = _bin_of(g, rf)
            spectrum[b] *= 10.0 ** (db / 10.0)
            marks[rf] = (b, float(g.psd_rf_offset_hz([b])[0]))
    reference = np.where(np.arange(frames) % 2 == 0, 1.0, FLOOR_REFERENCE)
    per_frame = np.tile(spectrum, (frames, 1))
    if excursion is not None:
        rows, rf, db = excursion
        b = _bin_of(g, rf)
        per_frame[np.asarray(rows), b] *= 10.0 ** (db / 10.0)
        marks["excursion"] = (b, float(g.psd_rf_offset_hz([b])[0]))
    codes = np.round(1000.0 * np.log10(per_frame / reference[:, None])).astype(np.int16)
    v5_fixture._replace(
        path,
        psd_frame_db_i16=codes,
        psd_db_reference=reference.reshape(frames, 1),
        psd_db_step_per_code=np.asarray(0.01, dtype=np.float64),
        psd_db_invalid_code=np.asarray(INVALID, dtype=np.int16),
    )
    return path, g, spectrum, marks


EXCURSION_ROWS = [3, 9, 15, 21, 27, 33]          # coarse-detected frames of the fixture (i % 6 == 3)


@pytest.fixture
def channel14(tmp_path):
    """Channel 14: the centre line at +3059 Hz sits in W and inside the K = 256 upper reference."""
    lines = [("pilot", 500.0), (-1200.0, 13.0), (-psd.span_half_width_hz(256), 12.0), (10_000.0, 10.0), ("centre", 25.0)]
    return _write_psd_product(tmp_path, 14, lines, excursion=(EXCURSION_ROWS, -2000.0, 30.0))


def test_decode_follows_the_product_contract(channel14):
    path, g, spectrum, marks = channel14
    with Product(path) as p:
        codes = np.asarray(p.archive["psd_frame_db_i16"])
        codes[0, :] = INVALID
        codes[2, marks["pilot"][0]] = INVALID
        v5_fixture._replace(path, psd_frame_db_i16=codes)
    with Product(path) as p:
        rows, power = next(p.psd_rows(chunk=8))
        assert rows == slice(0, 8) and power.shape == (8, NFFT)
        assert np.isnan(power[0]).all()
        assert np.isnan(power[2, marks["pilot"][0]]) and np.isfinite(power[2]).sum() == NFFT - 1
        ref = p.frame_column("psd_db_reference")
        expected = ref[1] * 10.0 ** (codes[1].astype(float) / 1000.0)
        np.testing.assert_allclose(power[1], expected)
        # both per-frame references decode the same floor exactly
        assert power[1, marks[10_000.0][0] - 5] == pytest.approx(1.0, abs=1e-12)
        assert power[3, marks[10_000.0][0] - 5] == pytest.approx(1.0, abs=1e-12)
        # the era mean is NaN-aware: the invalid frame and bin drop out without biasing the mean
        mask = np.ones(p.n_frames, bool)
        mask[EXCURSION_ROWS] = False
        spec = psd.accumulate_spectra(p, mask)
        assert spec.frames == p.n_frames - len(EXCURSION_ROWS)
        assert spec.frames_with_spectrum == spec.frames - 1
        order = np.argsort(psd.centred_offset_hz(g.psd_rf_offset_hz(np.arange(NFFT))), kind="stable")
        np.testing.assert_allclose(spec.mean, spectrum[order], rtol=1e-9)
        counts = dict(zip(spec.rf_offset_hz, spec.count))
        assert counts[marks["pilot"][1]] == spec.frames - 2
        assert counts[marks[10_000.0][1]] == spec.frames - 1
        with pytest.raises(ValueError, match="mask must have shape"):
            psd.accumulate_spectra(p, mask[:-1])


def test_lobes_energy_references_and_disposition(channel14):
    path, g, spectrum, marks = channel14
    with Product(path) as p:
        mask = p.selected.copy()
        mask[EXCURSION_ROWS] = False                     # the era excludes the excursion frames
        result = psd.containment(p, mask)
    row = result.row
    assert row.channel == 14 and row.freq_id == 844 and row.frames == 114
    assert row.centre_line_rf_offset_hz == pytest.approx(3059.0)
    assert row.window_median_power == pytest.approx(1.0)
    assert row.centre_line_db == pytest.approx(25.0, abs=0.01)
    # dominant lobe: the pilot, not the stronger centre line (excluded), bin-centre then sub-bin refined
    b0, rf0 = marks["pilot"]
    assert row.dominant_offset_hz == pytest.approx(rf0)
    assert row.dominant_db == pytest.approx(PEAK_DB - CURVATURE_DB * SUB_BIN ** 2, abs=0.01)
    assert row.dominant_refined_offset_hz == pytest.approx(rf0 + SUB_BIN * PSD_BIN_HZ, abs=0.05)
    assert result.dominant.refined_db == pytest.approx(PEAK_DB, abs=0.01)
    assert row.dominant_excess_db == pytest.approx(row.dominant_db, abs=1e-6)   # baseline is the flat floor
    assert row.near_offset_hz == row.in_span_offset_hz == row.dominant_offset_hz
    assert row.in_span_recovered and math.isnan(row.out_of_span_offset_hz)
    # pilot-associated energy: excess over the floor, spans nested about nominal, edge bin fractional
    def excess_at(rf):
        return spectrum[marks[rf][0]] - 1.0
    pilot = sum(spectrum[(b0 + k) % NFFT] - 1.0 for k in (-1, 0, 1))
    b_edge, rf_edge = marks[-psd.span_half_width_hz(256)]
    w_edge = _weight(rf_edge, -psd.span_half_width_hz(256), psd.span_half_width_hz(256))
    assert 0.0 < w_edge < 1.0
    total = pilot + excess_at(-1200.0) + excess_at(-psd.span_half_width_hz(256)) + excess_at(10_000.0)
    assert row.excess_total == pytest.approx(total, rel=1e-6)
    assert row.e_64 == pytest.approx((pilot + excess_at(-1200.0) + excess_at(-psd.span_half_width_hz(256))) / total, rel=1e-6)
    assert row.e_128 == pytest.approx(row.e_64, rel=1e-9)
    assert row.e_256 == pytest.approx((pilot + w_edge * excess_at(-psd.span_half_width_hz(256))) / total, rel=1e-6)
    assert row.e_64 >= row.e_128 >= row.e_256
    # reference contamination: the centre line in the K = 256 upper passband, the 10 kHz line in K = 64's
    assert row.ref_contaminant_256 == "centre_line"
    in_span_256 = pilot + w_edge * excess_at(-psd.span_half_width_hz(256))
    assert row.ref_contamination_256 == pytest.approx((10.0 ** 2.5 - 1.0) / in_span_256, rel=1e-6)
    assert row.ref_contaminant_64.startswith("feature@") and "10.0dB" in row.ref_contaminant_64
    assert row.ref_contamination_64 == pytest.approx(excess_at(10_000.0) / (pilot + excess_at(-1200.0) + excess_at(-psd.span_half_width_hz(256))), rel=1e-6)
    assert row.ref_contaminant_128 == "" and row.ref_contamination_128 == 0.0
    # straddle loss and margins from the refined in-span lobe
    offset = row.in_span_refined_offset_hz
    for k in psd.SPANS:
        assert getattr(row, f"straddle_loss_db_{k}") == pytest.approx(float(psd.straddle_loss_db(offset, k)))
        assert getattr(row, f"margin_hz_{k}") == pytest.approx(psd.span_half_width_hz(k) - abs(offset))
        assert getattr(row, f"margin_fraction_{k}") == pytest.approx(1.0 - abs(offset) / psd.span_half_width_hz(k))
    assert row.straddle_loss_db_256 > row.straddle_loss_db_128 > row.straddle_loss_db_64 > 0.0
    assert row.disposition == psd.SUPPORTED and row.reasons == ""
    # no detected frames in this mask -> the peak columns are blank, not wrong
    assert row.detected_frames == 54 and row.frames_in_span_64 == 1.0


def test_per_frame_peaks_use_the_detected_frames_of_the_mask(channel14):
    path, g, spectrum, marks = channel14
    with Product(path) as p:
        result = psd.containment(p, p.selected)
        detected = int((p.selected & p.rejected).sum())
    row = result.row
    assert row.frames == 120 and row.detected_frames == detected == 60
    rf_pilot = marks["pilot"][1]
    rf_ex = marks["excursion"][1]
    assert rf_ex == pytest.approx(-2000.0, abs=PSD_BIN_HZ / 2)
    peaks = result.spectrum.peak_offsets_hz
    assert peaks.shape == (60,)
    assert np.sum(np.isclose(peaks, rf_ex)) == len(EXCURSION_ROWS) and np.sum(np.isclose(peaks, rf_pilot)) == 54
    assert row.peak_abs_median_hz == pytest.approx(abs(rf_pilot))
    assert row.peak_abs_p99_hz == pytest.approx(abs(rf_ex))
    assert row.peak_signed_median_hz == pytest.approx(rf_pilot) and row.peak_signed_p90_hz == pytest.approx(rf_pilot)
    assert row.peak_signed_p99_hz == pytest.approx(rf_pilot)
    assert row.frames_in_span_64 == 1.0 and row.frames_in_span_128 == pytest.approx(0.9) and row.frames_in_span_256 == pytest.approx(0.9)
    # a caller may supply its own detected mask
    only = np.zeros(p.n_frames, bool)
    only[EXCURSION_ROWS] = True
    with Product(path) as p:
        spec = psd.accumulate_spectra(p, p.selected, detected=only)
    assert spec.detected_frames == 6 and np.allclose(spec.peak_offsets_hz, rf_ex)


def test_out_of_span_carrier_sets_the_sentinel_and_no_lobe_is_unsupported(tmp_path):
    # channel 33's case: a stronger co-channel carrier at -3.7 kHz, the configured target in span
    path, g, spectrum, marks = _write_psd_product(tmp_path, 33, [("pilot", -300.0), (-3700.0, 26.0)])
    with Product(path) as p:
        result = psd.containment(p, p.selected)
    row = result.row
    assert row.dominant_offset_hz == pytest.approx(marks[-3700.0][1])
    assert row.near_offset_hz == row.dominant_offset_hz              # within +-5 kHz too
    assert row.in_span_offset_hz == pytest.approx(marks["pilot"][1])
    assert row.in_span_recovered
    assert row.out_of_span_offset_hz == pytest.approx(marks[-3700.0][1])
    assert row.e_64 < 0.8 and row.e_64 >= row.e_128 >= row.e_256
    assert row.disposition == psd.SUPPORTED_SENTINEL
    assert "stronger out-of-span feature at -37" in row.reasons and "E_128" in row.reasons
    # only the floor and the centre line: nothing to recover
    path, g, spectrum, marks = _write_psd_product(tmp_path, 28, [("centre", 20.0)])
    with Product(path) as p:
        result = psd.containment(p, p.selected)
    row = result.row
    assert g.centre_line_rf_offset_hz == pytest.approx(-12566.0)
    assert not row.in_span_recovered and row.disposition == psd.UNSUPPORTED
    assert "no in-span lobe" in row.reasons
    assert math.isnan(row.straddle_loss_db_128) and math.isnan(row.margin_hz_128)
    assert math.isnan(row.e_128)                                     # no excess anywhere: undefined, not zero
    # channel 28's centre line at -12.566 kHz lies in the K = 64 lower reference
    assert row.ref_contaminant_64 == "centre_line" and row.ref_contaminant_128 == "" and row.ref_contaminant_256 == ""


def test_dirichlet_straddle_loss_and_overlap_weights():
    for k in psd.SPANS:
        assert float(psd.straddle_loss_db(0.0, k)) == 0.0
        assert float(psd.straddle_loss_db(psd.span_half_width_hz(k), k)) == pytest.approx(3.92, abs=0.005)
        assert float(psd.dirichlet_power(0.5, k)) == pytest.approx((2 / math.pi) ** 2, rel=1e-3)   # 2/pi is the K -> inf limit
    assert float(psd.dirichlet_power(1.5, 128)) == pytest.approx(0.045, abs=0.001)   # the +-2-bin self-leakage bound
    assert psd.span_half_width_hz(128) == pytest.approx(1525.87890625)
    assert psd.reference_centres_hz(128) == (-2 * SAMPLE_RATE_HZ / 128, 2 * SAMPLE_RATE_HZ / 128)
    # 20 Hz bins centred at -30..30 against [-11.92, 11.92]: the edge bins enter by their overlap fraction
    rf = np.array([-30.0, -10.0, 0.0, 10.0, 30.0])
    np.testing.assert_allclose(psd.overlap_weights(rf, -11.92, 11.92, bin_hz=20.0), [0.0, 0.596, 1.0, 0.596, 0.0], atol=1e-9)


def _row(channel, e64, e128, e256, disposition=psd.SUPPORTED):
    nan = float("nan")
    values = {f: nan for f in psd.COLUMNS}
    values.update(channel=channel, freq_id=0, frames=0, frames_with_spectrum=0, detected_frames=0,
                  in_span_recovered=True, e_64=e64, e_128=e128, e_256=e256,
                  ref_contaminant_64="", ref_contaminant_128="", ref_contaminant_256="",
                  disposition=disposition, reasons="")
    return psd.ContainmentRow(**values)


def test_k_star_rule_names_the_binding_channel_and_drops_sentinels():
    rows = [_row(1, 1.0, 0.98, 0.95), _row(2, 0.99, 0.92, 0.6), _row(3, 0.97, 0.85, 0.5),
            _row(4, 0.3, 0.2, 0.1), _row(5, float("nan"), float("nan"), float("nan"), psd.UNSUPPORTED)]
    table = {k.e_min: k for k in psd.k_star_table(rows)}
    assert set(table) == {0.8, 0.9, 0.95}
    policy = table[0.9]
    assert policy.k_star == 64 and policy.failing_k == 128 and policy.binding_channel == 3 and policy.binding_e == 0.85
    assert policy.sentinels == (4,) and policy.eligible == (1, 2, 3, 4)
    loose = table[0.8]
    assert loose.k_star == 128 and loose.failing_k == 256 and loose.binding_channel == 3 and loose.binding_e == 0.5
    strict = table[0.95]
    assert strict.k_star == 64 and strict.binding_channel == 3
    # every eligible channel passing every K: K* is the largest candidate and nothing binds
    clean = psd.k_star([_row(1, 1.0, 0.99, 0.97), _row(2, 0.99, 0.96, 0.95)], 0.9)
    assert clean.k_star == 256 and clean.failing_k is None and clean.binding_channel is None and math.isnan(clean.binding_e)
    # a channel failing already at K = 64 is a sentinel; with nothing else counted the rule is undefined
    only_sentinel = psd.k_star([_row(1, 0.5, 0.4, 0.3)], 0.9)
    assert only_sentinel.k_star is None and only_sentinel.sentinels == (1,) and only_sentinel.eligible == (1,)
    assert psd.k_star([], 0.9).k_star is None
    # a sentinel beside a counted channel does not drag the choice
    with_sentinel = psd.k_star([_row(1, 0.5, 0.4, 0.3), _row(2, 0.99, 0.96, 0.95)], 0.9)
    assert with_sentinel.k_star == 256 and with_sentinel.sentinels == (1,)


def test_writers_round_trip(tmp_path, channel14):
    path, g, spectrum, marks = channel14
    with Product(path) as p:
        result = psd.containment(p, p.selected)
    csv_path = psd.write_containment_csv([result.row], tmp_path / "containment.csv")
    text = csv_path.read_text(encoding="utf-8")
    header = text.splitlines()[0].split(",")
    assert header == list(psd.COLUMNS) and header[0] == "channel" and header[-1] == "reasons"
    assert "\r" not in text and ",nan" not in text
    back = psd.read_containment_csv(csv_path)
    assert len(back) == 1 and back[0].channel == 14 and back[0].in_span_recovered is True
    assert back[0].e_128 == result.row.e_128 and math.isnan(back[0].out_of_span_offset_hz)
    kstar_path = psd.write_kstar_csv(psd.k_star_table([result.row]), tmp_path / "kstar.csv")
    lines = kstar_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(psd.KSTAR_COLUMNS) and len(lines) == 4
    json_path = psd.write_spectra_json([result], tmp_path / "spectra.json", provenance="fixture")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    ch = payload["channels"]["14"]
    n = len(ch["rf_offset_hz"])
    assert n == len(ch["mean_db"]) == len(ch["baseline_db"]) == len(ch["excess"]) == len(ch["count"])
    assert all(abs(x) <= psd.WINDOW_HZ for x in ch["rf_offset_hz"])
    assert ch["dominant"]["offset_hz"] == pytest.approx(marks["pilot"][1])
    # the per-frame peaks come back losslessly as a count per window bin
    assert len(ch["peak_counts"]) == n and sum(ch["peak_counts"]) == result.row.detected_frames == 60
    by_rf = dict(zip(ch["rf_offset_hz"], ch["peak_counts"]))
    assert by_rf[round(marks["excursion"][1], 4)] == len(EXCURSION_ROWS)
    assert by_rf[round(marks["pilot"][1], 4)] == 54
    assert payload["parameters"]["window_hz"] == 15000.0 and payload["provenance"] == "fixture"
    npz_path = psd.write_spectra_npz([result], tmp_path / "spectra.npz")
    with np.load(npz_path) as z:
        assert list(z["channels"]) == [14] and z["mean_14"].shape == (NFFT,)
        assert np.all(np.diff(z["rf_offset_hz_14"]) > 0)


def test_pilot_near_the_coarse_channel_edge_reads_the_window_on_the_circular_axis(tmp_path):
    """Channel 21's pilot sits about 4.7 kHz from a coarse-channel edge: the far side of W and the lower
    reference passbands are content aliased from the channel's other edge, read where the detector reads it."""
    path, g, spectrum, marks = _write_psd_product(tmp_path, 21, [("pilot", 0.0), (-6100.0, 12.0), (-12200.0, 9.0)])
    edge = psd.edge_distance_hz(g)
    assert 4000.0 < edge < 5000.0
    with Product(path) as p:
        row = psd.containment(p, p.selected).row
    assert row.edge_distance_hz == pytest.approx(edge)
    assert row.window_aliased_hz == pytest.approx(psd.WINDOW_HZ - edge, abs=1e-6)
    assert row.in_span_recovered and abs(row.in_span_refined_offset_hz) < 2 * PSD_BIN_HZ
    # the lines beyond the edge are found at their aliased offsets, inside the K = 128 and K = 64 lower passbands
    assert row.ref_aliased_128 and row.ref_aliased_64 and not row.ref_aliased_256
    at = lambda rf: f"feature@{psd.centred_offset_hz(marks[rf][1]):.0f}Hz"      # the placed bin's centre, on the circular axis
    assert row.ref_contamination_128 > 0 and at(-6100.0) in row.ref_contaminant_128
    assert row.ref_contamination_64 > 0 and at(-12200.0) in row.ref_contaminant_64
    assert marks[-6100.0][1] > 300_000.0                                          # unwrapped, the bin sits at the far edge
    # the window arrays cover the full +-W about the pilot
    with Product(path) as p:
        window, _ = psd.window_spectrum(psd.accumulate_spectra(p, p.selected), g)
    assert window.rf_offset_hz.min() < -psd.WINDOW_HZ + PSD_BIN_HZ and window.rf_offset_hz.max() > psd.WINDOW_HZ - PSD_BIN_HZ
    # channel 36's pilot is far from both edges: nothing is aliased
    path36, g36, _, _ = _write_psd_product(tmp_path, 36, [("pilot", 0.0)])
    with Product(path36) as p:
        row36 = psd.containment(p, p.selected).row
    assert row36.window_aliased_hz == 0.0 and not (row36.ref_aliased_64 or row36.ref_aliased_128 or row36.ref_aliased_256)
    assert row36.edge_distance_hz > psd.WINDOW_HZ + 2 * SAMPLE_RATE_HZ / 64


def test_anchor_lobe_disagreement_is_measured_in_fine_bins_and_sets_the_sentinel():
    from pilot_proxy.products.reader import FINE_BIN_HZ
    assert psd.anchor_lobe_offset_bins(14.87, -151.5) == pytest.approx((14.87 + 151.5) / FINE_BIN_HZ)
    assert math.isnan(psd.anchor_lobe_offset_bins(float("nan"), -151.5))
    assert math.isnan(psd.anchor_lobe_offset_bins(3.0, float("nan")))
    lobe = psd.Lobe(index=0, offset_hz=0.0, refined_offset_hz=0.0, db=20.0, refined_db=20.0, excess_db=20.0)
    verdict, reasons = psd.disposition(lobe, float("nan"), 0.99, 0.0, anchor_aliases=True,
                                       anchor_note="fine anchor +14.0 bins from the PSD in-span lobe")
    assert verdict == psd.SUPPORTED_SENTINEL and reasons == "fine anchor +14.0 bins from the PSD in-span lobe"
    assert psd.disposition(lobe, float("nan"), 0.99, 0.0)[0] == psd.SUPPORTED

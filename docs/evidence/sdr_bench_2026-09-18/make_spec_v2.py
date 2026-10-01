#!/usr/bin/env python3
"""Build the predeclared single-input bench protocol for the kit in hand (2026-09-18).

Supersedes make_spec.py, whose ladder assumed pads in 6 dB steps. The available
attenuation is two 30 dB pads and one cable, so the analog ladder has three
settings only (0, 30, 60 dB). The fine ladder is therefore the commanded tone
amplitude, in six 6.02 dB steps, and the pads move the whole ladder by a known
30 dB. Two configurations that reach the same level by different routes are the
cross-check the bench exists for: the analog step and the digital step must
agree. Amplitude is only ever reduced, so the declared conducted bound falls with
it and no slot is louder than the 2026-09-09 records.

One waveform per spec, so one spec per amplitude rung: rungs/a<k>/ holds the
payload, its stationarity audit, its power declaration and protocol_spec.json.
Never opens a radio.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path("/home/djg/rail/pilot-proxy")
sys.path.insert(0, str(REPO/"src"))
from pilot_proxy.testbench.sdr_protocol import SCHEMA, digest, settings_identity  # noqa: E402

RATE = 2_000_000.
RECORD_SAMPLES = 4_000_000            # 2 s: the geometry the unchanged analyzer accepts
BASE_AMPLITUDE = .005                 # the 2026-09-09 tone amplitude, the top rung
N_RUNGS = 6                           # 6.02 dB apart: 0 to -30.1 dB
PADS_DB = [0, 30, 60]                 # cable only, one pad, both pads
BUILD = "/home/djg/rail/output/sdr-bench-2026-09-17/build-v2.json"


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def payload(amplitude):
    """tools/lime_reference_capture_v1.py prepare(mode='tone') with its amplitude scaled."""
    waveform = np.zeros(RECORD_SAMPLES+600000, dtype=np.complex64)
    n = RECORD_SAMPLES+400000
    tone = amplitude*np.exp(2j*np.pi*100000*np.arange(n)/RATE)
    ramp = 2000
    tone[:ramp] *= np.sin(np.linspace(0, math.pi/2, ramp))**2
    tone[-ramp:] *= np.sin(np.linspace(math.pi/2, 0, ramp))**2
    waveform[100000:100000+n] = tone.astype(np.complex64)
    return waveform


def audit(wave_path, wave, start, samples):
    segments, count = [], 7
    edges = np.linspace(start, start+samples, count+1).astype(int)
    powers = []
    for a, b in zip(edges[:-1], edges[1:]):
        block = wave[a:b].astype(np.complex128)
        power = float(np.mean(np.abs(block)**2))
        spectrum = np.abs(np.fft.fft(block))
        peak_hz = float(np.fft.fftfreq(block.size, 1/RATE)[int(np.argmax(spectrum))])
        powers.append(power)
        segments.append({"start_sample": int(a), "stop_sample": int(b), "mean_power": power,
                         "peak_hz": peak_hz, "passed": abs(peak_hz-100000.) <= RATE/block.size})
    mean = float(np.mean(powers))
    for seg in segments:
        seg["passed"] = bool(seg["passed"] and abs(seg["mean_power"]-mean)/mean <= 1e-6)
    return {"schema": "pilotproxy-stationary-input-audit-v1", "waveform_sha256": digest(wave_path),
            "start_sample": start, "samples": samples, "source_startup_discard_samples": 300000,
            "stationarity_passed": all(s["passed"] for s in segments),
            "method": "Digital command payload only: seven disjoint chronological segments covering the declared interval; per-segment mean power and FFT peak frequency. This audits the commanded waveform, not any RF measurement.",
            "thresholds": {"relative_mean_power_deviation_max": 1e-6, "peak_frequency_hz": 100000., "peak_tolerance_hz": "one FFT bin of the segment"},
            "segments": segments}


def pad_order(rung):
    """Ascending pads on even rungs, descending on odd: two pad operations per rung change."""
    return PADS_DB if rung % 2 == 0 else PADS_DB[::-1]


def slots(rung, amplitude_db):
    pads = pad_order(rung)
    first = pads[0]
    plan = [("tx_zero", "txzero", None, "terminated TX-zero before the rung"),
            ("signal_only", "tone", first, f"cabled tone at {first} dB, cabling check"),
            ("noise_only", "noise", None, "terminated null before the rung")]
    plan += [("mixture", "tone", pad, f"rung {rung} ({amplitude_db:+.2f} dB) at pad {pad} dB") for pad in pads]
    result = []
    for index, (kind, mode, pad, label) in enumerate(plan):
        terminated = pad is None
        rx = "50 ohm terminator T1" if terminated else "SMA cable C1 from the pad stack"
        tx = "50 ohm terminator T2" if terminated else (
            "TX2 to SMA cable C1 direct" if pad == 0 else
            f"TX2 to {pad//30} x 30 dB pad to SMA cable C1")
        result.append({"slot_index": index, "kind": kind, "seed": 1000+100*rung+index,
                       "data_shelf_snr_db": (amplitude_db-float(pad) if kind == "mixture" else None),
                       "helper_mode": mode, "bench_label": label,
                       "commanded_attenuation_db": pad,
                       "commanded_amplitude_db_rel": amplitude_db,
                       "rx_port": rx, "tx_port": tx,
                       "record_samples": RECORD_SAMPLES, "record_seconds": 2.0})
    return result


def build_rung(rung, build, sources):
    amplitude = BASE_AMPLITUDE/2**rung
    amplitude_db = 20*math.log10(amplitude/BASE_AMPLITUDE)
    out = HERE/"rungs"/f"a{rung}"
    out.mkdir(parents=True, exist_ok=True)
    wave_path = out/"tx_payload_2s.cfile"
    wave = payload(amplitude)
    wave.tofile(wave_path)
    start, samples = 200000, 4_200_000
    audit_path = out/"stationarity_audit.json"
    audit_path.write_text(json.dumps(audit(wave_path, wave, start, samples), indent=2)+"\n")
    spec = {
        "schema": SCHEMA, "run_id": f"SDR-BENCH-2026-09-18-single-input-reference-v2-rung{rung}",
        "bench_scope": "Conducted single-input reference on the LimeSDR Mini: terminated nulls and a cabled tone stepped by commanded amplitude and by two 30 dB pads. Closes the physical single-input reference only; no sensitivity, ROC, telescope deployment or absolute dBm claim.",
        "waveform": {**ref(wave_path), "start_sample": start, "samples": samples,
                     "source_startup_discard_samples": 300000},
        "implementation_sources": [ref(p) for p in sources],
        "noise_model": "independent_complex_gaussian",
        "radio": {"frequency_hz": 500_000_000., "sample_rate_hz": RATE,
                  "emission_bandwidth_hz": 2_000_000., "radio_filter_bandwidth_hz": 1_500_000.,
                  "tx_bandwidth_hz": 5_000_000., "tx_rms": amplitude,
                  "device_serial": "1D423D9108F273", "device": "LimeSDR Mini",
                  "antenna_separation_cm": 30., "antenna_separation_meaning": "no antennas: nominal SMA cable run length, conducted bench",
                  "antenna_polarization": "not applicable, conducted",
                  "clock_source": "LimeSDR Mini internal VCTCXO",
                  "native_tx_gain_db": 50, "tx_gain_setting_db": 38.,
                  "native_rx_gain_db": 30, "rx_gain_setting_db": 18.,
                  "rx_path": "LNAW", "active_tx_path": "TX2",
                  "tx_antenna": "none: TX2 into the pad stack and SMA cable C1, or 50 ohm terminator T2",
                  "rx_antenna": "none: SMA cable C1 from the pad stack, or 50 ohm terminator T1"},
        "capture": {"settle_samples": 100000, "marker_guard_samples": 100000, "capture_samples": RECORD_SAMPLES},
        "guards": {"clip_level": .98, "rms_min": 1e-5, "rms_max": .5, "clip_fraction_max": 0., "marker_correlation_min": .15},
        "slots": slots(rung, amplitude_db),
        "ladder": {"rung_index": rung, "commanded_amplitude": amplitude,
                   "commanded_amplitude_db_rel": amplitude_db,
                   "commanded_attenuation_db": pad_order(rung),
                   "pad_inventory": "two 30 dB SMA pads and one SMA cable: settings 0, 30 and 60 dB only",
                   "amplitude_ladder_db_rel": [round(-6.0206*k, 4) for k in range(N_RUNGS)],
                   "level_rule": "slot level = commanded_amplitude_db_rel - commanded_attenuation_db, so rung k at pad p and rung k-5 at pad p-30 are the same commanded level; those coincidences are the analog-versus-digital cross-check",
                   "passes": ["pads ascending on even rungs, descending on odd"],
                   "repeats_per_rung": 1, "record_seconds": 2.0, "record_samples": RECORD_SAMPLES,
                   "capture_maximum_seconds": 4.0, "helper": "tools/lime_reference_capture_v1.py, unchanged",
                   "overload_rule": "A tone slot whose receipt carries worker error 'RX overload guard' is recorded as overloaded and the session continues; the record is kept, counts as attempted not successful, and never enters analysis. Any other failure stops the session.",
                   "analysis": "tools/analyze_sdr_antenna_controls_v1.py analyze_capture, unchanged: mix -100 kHz, 25/128 resampler, K=L=128, bins 0/-2/+2, sample scale 500"},
        "rf_limits": {"frequency_low_hz": 499_000_000., "frequency_high_hz": 501_000_000.,
                      "power_metric": "average_conducted_dbm", "maximum_power_dbm": 0.,
                      "provided_by": "Author's conducted-bench constraint of 2026-09-17: TX port into a 50 ohm pad stack, cable or terminator only, no antenna, no radiated emission, RX port input kept at or below 0 dBm. Not a jurisdictional authorization."},
    }
    power = {"schema": "pilotproxy-measured-rf-power-v1", "settings_sha256": settings_identity(spec),
             "power_metric": "average_conducted_dbm", "covers_all_slots_markers_transitions": True,
             "measurement_id": f"BENCH-2026-09-18-declared-conducted-bound-rung{rung}",
             "method": "Declared upper bound, not a meter reading: LimeSDR Mini specified maximum TX output near +10 dBm at full digital scale and full gain; the rung's commanded amplitude is 46.0 dB below full scale at rung 0 and a further 6.02 dB per rung; bound stated at -30 dBm at rung 0, scaled with the rung, with 12 dB uncertainty covering gain-table and device spread. Applies to every slot, the zero-IQ slots and the ramps, whose amplitude never exceeds the tone. Absolute dBm is outside this protocol's claims.",
             "conducted_power_dbm": -30.+amplitude_db, "total_uncertainty_db": 12.}
    power_path = out/"power_bound_declaration.json"
    power_path.write_text(json.dumps(power, indent=2)+"\n")
    spec["stationarity_audit"] = ref(audit_path)
    spec["power_conversion"] = ref(power_path)
    (out/"protocol_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True)+"\n")
    return {"rung": rung, "dir": str(out), "amplitude": amplitude,
            "amplitude_db_rel": round(amplitude_db, 3), "pads": pad_order(rung),
            "slots": len(spec["slots"])}


def main():
    build = json.loads(Path(BUILD).read_text())
    sources = [REPO/"tools/lime_reference_capture_v1.py", REPO/"tools/lime_reference_worker_v1.cpp",
               REPO/"tools/analyze_sdr_antenna_controls_v1.py", REPO/"tools/run_sdr_distribution_reference_v1.py",
               REPO/"src/pilot_proxy/integration/sdr_upgrade_adapter.py", Path(BUILD), Path(build["worker"])]
    report = [build_rung(k, build, sources) for k in range(N_RUNGS)]
    (HERE/"rungs"/"index.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Append one bench slot to session_log.csv from its receipt. usage: log_slot.py <rung> <slot_dir> <pad|term> [notes]"""
import csv, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
rung, d, pad = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
notes = sys.argv[4] if len(sys.argv) > 4 else ""
r = json.load(open(d/"receipt.json")); w = r["worker_report"]
spec = json.load(open(HERE/"rungs"/rung/"protocol_spec.json"))
slot = int(d.name.split("-")[1])
s = spec["slots"][slot]
row = dict(rung=rung, slot=slot, utc=r["completed_utc"], kind=s["kind"], helper_mode=w["mode"],
           tone_component=spec["radio"]["tx_rms"], pad_db=pad,
           parts=("T1+T2" if pad == "term" else f"C1 + {int(pad)//30} x 30 dB pad"),
           success=r["success"], rx_qualified=w["rx_qualified"], rx_error=w["rx_error"],
           rx_max_chunk_rms=w["rx_max_chunk_rms"], rx_peak_component=w["rx_peak_component"],
           accepted_samples=w["accepted_samples"], qualified_dropped=w["qualified_rx_dropped"],
           qualified_overrun=w["qualified_rx_overrun"], qualified_underrun=w["qualified_rx_underrun"],
           tx_underrun=w["tx_underrun"], notes=notes)
p = HERE/"session_log.csv"; new = not p.exists()
with p.open("a", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(row))
    if new: wr.writeheader()
    wr.writerow(row)
print(",".join(f"{k}={row[k]}" for k in ("rung","slot","kind","pad_db","success","rx_qualified","rx_max_chunk_rms","rx_peak_component")))

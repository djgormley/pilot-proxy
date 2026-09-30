# Single-input SDR bench, v2 (2026-09-18): the kit in hand

Supersedes BENCH_RUNBOOK_v1_superseded.md, whose ladder assumed pads in 6 dB steps.

Kit: LimeSDR Mini serial 1D423D9108F273, two 50 ohm SMA terminators (T1, T2), two
30 dB SMA pads (P1, P2), one SMA cable (C1), two antennas (not used here).

The pads give three analog settings only: 0, 30 and 60 dB. The fine ladder is
therefore the commanded tone amplitude, six steps of 6.02 dB, and the pads move
the whole ladder by a known 30 dB. Level of a slot is
`commanded_amplitude_db - pad_db`, so rung 0 at pad 30 and rung 5 at pad 0 are
the same level, as are rung 0 at pad 60 and rung 5 at pad 30. Those two
coincidences are the point of the bench: the analog 30 dB and the digital
30.1 dB must give the same reading, which separates the analog chain from the
digital scaling. The amplitude is only ever reduced from the 2026-09-09 value,
so no slot is louder than the records already taken and the declared conducted
bound falls with the rung.

## Fixed settings (unchanged from the 2026-09-09 controls)

500 MHz LO, 2 MHz complex sample rate, RX gain setting 18 (native 30) on LNAW,
RX LPF 1.5 MHz, TX gain setting 38 (native 50) on TX2, TX LPF 5 MHz, tone at
+100 kHz, rung 0 amplitude 0.005. Analysis: mix by -100 kHz, the 25/128 FIR
resampler, K = L = 128, bins 0/-2/+2, sample scale 500, unchanged helper and
worker.

## Wiring

- Terminated slots: T1 on RX, T2 on TX2. Nothing else connected.
- Pad 0: TX2 to C1 to RX.
- Pad 30: TX2 to P1 to C1 to RX.
- Pad 60: TX2 to P1 to P2 to C1 to RX.

The pads screw directly onto TX2 and onto each other, so one cable is enough.
Never connect an antenna while TX is enabled; this protocol declares a conducted
bench and no radiated emission.

## Protocols

`rungs/a0` to `rungs/a5`, one per amplitude rung, each with its own payload,
stationarity audit, power declaration and `protocol_spec.json`. All six validate
(`tools/prepare_sdr_protocol.py validate --spec rungs/a<k>/protocol_spec.json`
prints `protocol_valid: true`). Run them in rung order 0 to 5. The pad order
alternates by rung, ascending on even rungs and descending on odd, so only two
pad operations separate one rung from the next; the spec's slot order is the
order to record in and is not to be changed during the session.

Each rung is six records: terminated TX-zero, a cabled tone at the rung's first
pad as a cabling check, a terminated null, then the three pad settings. Thirty-six
records in all, about forty minutes with the pad changes.

## Level budget (from the 2026-09-09 records, written before any bench data)

The 2026-09-09 over-the-air tone (30 cm, same settings) sits at raw rms 0.0122
to 0.0140 and resampled tone amplitude about 0.0025. Against that level: the
worker overload guard is reached about 28 dB above it and the peak guard about
26 dB above it; the analyzer's 4-bit quantizer at sample scale 500 saturates a
resampled component about 12 dB above it, so a clean rung stays at or below
about +10 dB; the qualification floor, 10 dB above the terminated null in the
target bin, is about 19 dB below it. A qualified slot therefore lies in a window
about 30 dB wide.

The conducted level at pad 0, rung 0 is the 30 cm level plus the over-the-air
link loss L, which was never measured and is plausibly 20 to 45 dB. The
eighteen (rung, pad) slots span 90 dB in 6 dB steps, so the window holds five or
six qualified slots for any L between about 0 and 60 dB. That is why this ladder
is run instead of an analog-only one: with three pad settings alone, at most two
slots could be qualified and only for a narrow range of L.

## Steps

Interpreter (`$PY`): `/home/djg/rail/output/fisher-fixes-2026-09-08/analysis-venv/bin/python`,
the analysis venv of the 2026-09-09 run. Set `PYTHONPATH=/home/djg/rail/pilot-proxy/src`
and the numerical thread variables to 1. Power the radio and wait ten minutes
before the first record.

Rung amplitudes, passed to the helper as `--tone-component`:

| rung | amplitude | level |
|---|---|---|
| a0 | 0.005 | 0 dB |
| a1 | 0.0025 | -6.02 dB |
| a2 | 0.00125 | -12.04 dB |
| a3 | 0.000625 | -18.06 dB |
| a4 | 0.0003125 | -24.08 dB |
| a5 | 0.00015625 | -30.10 dB |

1. Create `session/` and `session_log.csv` with columns slot, utc_start, rung,
   helper_mode, tone_component, commanded_attenuation_db, pad_parts, cable,
   terminator_ids, room_temperature_c, notes. Photograph the bench once before
   the first record. Label the parts: cable C1, pads P1 and P2 by their body
   marks, terminators T1 and T2, and use the same identities in every row.
2. For each rung k in 0 to 5, read `rungs/a<k>/protocol_spec.json` and work its
   six slots in `slot_index` order. The first three are the fixed controls
   (terminated TX-zero, a cabled tone, a terminated null), so two connector
   changes per rung are spent on them; that order is fixed by the protocol and
   is not to be rearranged.
3. Per slot: connect the ports as the slot's `tx_port` and `rx_port` say, fill
   the log row, then run

       $PY /home/djg/rail/pilot-proxy/tools/lime_reference_capture_v1.py prepare \
         --output session/a<k>/slot-<NN>-<mode> \
         --build-manifest /home/djg/rail/results/sdr_reference_transport_2026-09-09/build-v1/build.json \
         --serial 0x1d423d9108f273 --mode <noise|txzero|tone> \
         --tone-component <rung amplitude> --record-seconds 2

       $PY /home/djg/rail/pilot-proxy/tools/lime_reference_capture_v1.py capture \
         --output session/a<k>/slot-<NN>-<mode> --hardware-authorized --rf-confined-authorized

   adding `--transmit` on the txzero and tone slots. Change pads only between
   records, with the helper not running.
4. On a failure, read `rx_error` in the slot's `receipt.json`: `RX overload
   guard` on a tone slot is the predeclared exception, so log it, keep the
   record, and go on. Anything else stops the session.
5. After the last slot, `sha256sum` every `receipt.json` and the log into
   `session/manifest.sha256` before any analysis.
6. Analyse each successful record with the unchanged function:

       $PY -c "import sys; sys.path.insert(0, '/home/djg/rail/pilot-proxy/tools'); \
       from analyze_sdr_antenna_controls_v1 import analyze_capture; \
       analyze_capture('session/a<k>/slot-<NN>-<mode>', 'analysis/a<k>/slot-<NN>-<mode>')"

   Tabulate `natural_mean_frame_term_powers`, `natural_ratio`, `packed_ratio`
   and `adapter_metadata.sample_quantization.saturated_component_count` from each
   `record.json`, with the receipt's `rx_peak_component` and `rx_max_chunk_rms`,
   into `analysis/ladder.csv`: one row per slot, with its rung, pad, commanded
   level and whether it qualified. The two coincidence pairs and the slope are
   read off that table.

## Acceptance criteria (written before the data)

A slot is qualified when its record completes, carries zero clipped samples and
zero saturated resampled components, and its target-bin power is at least 10 dB
above the terminated null recorded in the same rung.

1. Transfer: over the qualified slots, measured resampled tone amplitude in dB
   against commanded level in dB has slope 1.00 within 0.05 and residual scatter
   at or below 0.5 dB.
2. Coincidence: each of the two pairs that reach the same commanded level by
   different routes agrees within 0.5 dB.
3. Nulls: the twelve terminated records agree within 1 dB, and the first and
   last agree within 0.5 dB.
4. Guards: zero clipped samples and zero saturated components on every qualified
   slot; an overloaded slot is recorded, kept, counted as attempted and excluded
   from analysis, and the session continues. A qualified-RX underrun, overrun or
   drop is a transport fault under AMENDMENT_1_transport_faults.md (2026-09-19,
   written before any retry): the record is kept and marked, re-recorded once in
   the same wiring, and the session stops if the repeat also fails; the fault and
   repeat counts are reported with the result. Any other failure stops the
   session.

Nothing is tuned after a record. If the slope or a coincidence fails, that is the
result and it is reported as measured, not as passed.

## Scope

This closes the single-input physical reference of the evidence matrix: a
terminated noise floor and a known-level tone through one input, measured by the
unchanged estimator. It does not establish sensitivity, a receiver operating
characteristic, absolute dBm, array degrees of freedom, or anything about the
CHIME deployment. The two antennas are not part of it; an over-the-air record
would be a radiated emission outside this protocol's declaration and is covered
by the 2026-09-09 protocol instead.

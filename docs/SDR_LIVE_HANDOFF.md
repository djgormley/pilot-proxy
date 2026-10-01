# Live antenna RX to CPU host handoff

`tools/run_sdr_live_handoff.py` measures a short **file-backed live host
handoff** using the unchanged qualified native LimeSDR reference worker and
the existing CPU streaming digital adapter. It does not measure USB-call
latency, physical RF calibration, a thermal-noise distribution, sustained
operation or live Pathfinder readiness.

The separate engineering smoke accepts two seconds of RX. The primary plan
fixes three four-second accepted intervals before primary acquisition. Four
seconds is the unchanged capture helper's maximum; no native limits are
extended. Each attempt uses 500 MHz LO, 2 MHz complex sample rate, RX gain 30
and 1.5 MHz RX LPF, on the antenna-connected receiver. Native mode `noise`
means TX-disabled accepted RX with before/after register evidence. No TX
stream or payload is started. The existing LMS_Init internal gain calibration
and LPF tuning still precede the accepted interval.

Processing uses the previously fixed 100 kHz digital mixer, target bin 0,
sample scale 500 and single-input 16,384-sample upgrade output geometry.
These are digital choices; they do not calibrate RF frequency response, absolute
power or a telescope array. Every ratio status and clipping count is retained.

## Publication, ownership and bounded state

The native worker writes returned samples and accepted sample spans to regular
files, then emits a newline-terminated `rx-chunks.jsonl` record. A partial JSON
line is unpublished. The wrapper verifies each published accepted span's
offset/count, exact hardware sample timestamps, continuity, status/counters,
finite data and overload limits before reading exactly those bytes. Publication
is a visibility commit, not per-chunk fsync durability or final qualification.
The worker publishes before its own final checks; all live results remain
provisional until native acceptance, complete byte count and cleanup succeed.

A bounded queue holds at most 32 chunks of at most 8,192 complex64 samples
(2 MiB payload). The reader copies each payload slice before queueing it;
ownership then belongs to the single consumer. Queue overflow fails the attempt
and stops the native helper. The regular file spool is separate: its finite
four-second accepted payload is 64 MB, and native startup/qualified capture
files are also retained. Backlog reports distinguish validated committed
samples from accepted-file bytes visible before publication. Sample backlog
divided by 2 MHz describes buffered sample duration, not measured time latency.

The consumer owns one StreamingDigitalAdapter. It uses explicit hardware
sample indices, with a new segment origin for every separately initialized
record. It neither fabricates missing samples nor stitches captures. The
unchanged worker refuses qualified transport gaps; any failure ends this
attempt, retains partial evidence and leaves later attempts unattempted.

## Timing and concurrency evidence

The wrapper records local monotonic times of completed-row observation,
enqueue, service start/end and frame production, along with separately timed post-service file
size, sample backlog and queue occupancy. These clocks are never equated to
RF arrival or USB receive-call completion. A pidfd identifies the actual native
child, avoiding PID-reuse ambiguity. Each successful attempt must produce at
least one frame while that native child is observed alive **and** accepted-file
size is still below its fixed final sample count. Worker-alive evidence alone
could include cleanup after capture and is insufficient.
The final publication and first observed complete-file size are distinct.
Last-input service completion is measured from the observed final publication.
Final adapter boundary processing occurs after the helper exits, and its
separately recorded interval includes native cleanup/EOF waiting. That longer
interval is not queue-drain latency or an invented hardware stop timestamp.

CPU service and observed backlog are descriptive. The queue bound is an
implementation limit, not a calibrated deadline. The native helper has its
existing finite deadlines; the wrapper has a 40-second host budget per attempt.
The CPU consumer and its publication reader share one pinned logical CPU;
the previously qualified native worker keeps the invoking affinity and its
existing I/O threads. Numerical library thread requests are one. This does
not imply zero interference with other host work.

## End-of-record validation

After native completion and confirmed cleanup, the wrapper rehashes all accepted
bytes and verifies that each was consumed once. It replays the same bytes
through the streaming adapter with independent 8,192-sample partitioning and
compares every frame's exact float/packed/projection hash and output fields.
It also runs the frozen batch reference, checks numerical float agreement
(absolute tolerance 2e-12) and exact packed/projection equality for these
particular records. The minimum quantizer half-step distance is retained;
universal floating-point quantization parity is not claimed.

The final incomplete output frame is explicitly discarded with its support
receipt. All expected complete frames, clipping, finite/infinite/undefined
ratios and failed attempts remain visible. No fitting, rescaling, interval
selection, retries or gate changes follow primary outcomes. Smoke attempts
are development and are separate from the primary records.

## Commands

Set `PYTHONPATH=src` and `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.

```
python tools/run_sdr_live_handoff.py prepare NEW_OUTPUT --kind smoke \
  --build-manifest /home/djg/rail/results/sdr_reference_transport_2026-09-09/build-v1/build.json \
  --serial 0x1d423d9108f273
python tools/run_sdr_live_handoff.py run NEW_OUTPUT \
  --hardware-authorized --rf-confined-authorized
```

Use a separately prepared new output with `--kind primary` for the three fixed
four-second records. Preparation does not open hardware. Launch refuses a
changed frozen source/plan/runtime or a prior launch. `--transmit` is not an
option in this wrapper. Release evidence includes original capture receipts,
all chunk publications, host timing, boundary receipts and offline parity.
All source identities are checked again after attempted records, retaining
failed integrity explicitly. SIGINT/SIGTERM stop the owned helper and preserve
failure state; exceptional forced cleanup never qualifies an attempt.

# Bench amendment 1, 2026-09-18: transport faults are re-recorded, not fatal

Written after rung a1 slot 3 stopped the session and before any further record.

What happened. The worker refused rung a1 slot 3 with "Qualified RX stream
underrun/overrun/drop": three RX samples were dropped inside the measurement
window, so the accepted interval closed after 217,504 of 4,000,000 samples and
the record was marked unsuccessful. The declared rule in BENCH_RUNBOOK.md makes
only the RX overload guard survivable and stops the session on anything else, so
the session is stopped as declared.

Why the rule is wrong for this fault. Every record of this session has carried
RX drops: 1 to 16 per record, all of them in the startup phase, with none in the
measurement window until this one. The drops are a property of the USB transport,
which in this session is a usbipd attachment of the radio into WSL, set up for
this bench and not the path used for the 2026-09-09 records. The fault is in the
link between the radio and the host, not in the radio, the estimator, the levels
or the wiring, and the worker's own guard detected it and refused the record
rather than letting a gapped stream reach the analysis. That is a retryable
transport fault, unlike an overload, which is a statement about the level.

The amendment. A record that fails with a qualified-RX underrun, overrun or drop
is retained with its receipt, marked transport_fault in the session log, and
re-recorded once in the same wiring. If the repeat also fails, the session stops.
Both records are kept and the number of transport faults and repeats is reported
with the result, so the reader can judge whether the session was healthy. No
level, wiring, amplitude, pad, guard or acceptance criterion changes, and a
repeat is never selected on its value: the first successful record in a slot is
the one analysed.

What it cannot change. A transport fault carries no information about the signal,
so retrying one cannot move a qualified slot's reading, and the qualification
criteria are untouched. If transport faults become frequent enough that the
session is mostly repeats, the honest conclusion is that this transport cannot
support the bench, and the session is reported as not established rather than
passed.

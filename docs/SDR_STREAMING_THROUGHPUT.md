# CPU streaming adapter throughput

The fixed synthetic benchmark in
`../results/sdr_streaming_throughput_2026-09-09` measures the service rate of
`StreamingDigitalAdapter`, including its mixer, FIR resampler, four-bit
packing, integer projections, public-state probes, and a checksum output sink.
It exercises the same 16,384-sample output geometry as the upgrade adapter.
There is no SDR, USB, transport queue or paced input source in this benchmark.

## Frozen workload and result

The plan freezes one deterministic complex64 record of 4,194,304 input samples,
five repeated fresh-adapter runs, one 524,288-sample warmup, and fixed chunks
of 8,192 samples. The input is independent uniform digital noise plus a
100 kHz tone, mixed by 100 kHz, with target bin zero and sample scale five.
At 2 MHz the record represents **2.097152 seconds** of input per timed run.
Each timed run emits **49 complete frames** and discards 16,320 remaining
full-support output samples; the warmup emits six frames. The same IQ bytes
are reused for the five timing trials. They are not independent physical or
statistical trials, and there is no retry or timing-result selection.

On this WSL host, a 13th Gen Intel Core i9-13950HX, the process was pinned to
logical CPU 31 at nice 10. Requested numerical-library thread counts were one;
the observed process thread count was also one before and after the trials.

| Trial | Total wall time (s) | Input rate (million samples/s) |
| --- | ---: | ---: |
| 1 | 1.207137 | 3.474587 |
| 2 | 1.242652 | 3.375285 |
| 3 | 1.282742 | 3.269796 |
| 4 | 1.297989 | 3.231387 |
| 5 | 1.277081 | 3.284290 |

All five observed average rates exceed 2 MHz; the slowest is **1.6157 times**
the nominal input rate. All output packed/projection checksums agree, all
expected frame counts and geometry checks pass, and all sampled state
occupancies remain within their declared capacities.

Average rate alone does not establish a live deadline. Of 2,560 timed push
calls, **139 (5.43%)** exceed the nominal **4.096 ms** chunk-arrival interval.
Instrumented per-chunk service time has a median of **2.204 ms**, 95th percentile
**4.131 ms**, 99th percentile **5.112 ms**, and maximum **34.288 ms**. These are
observed wall-clock service durations, including scheduling effects and the
instrumented output sink. They are neither measured transport delays nor
queue/backlog measurements. Real receiver buffering and scheduling still need
measurement before a live rate is accepted.

## Timing and state boundaries

The primary wall timer encloses the entire push loop and final `finish()` call.
It includes slicing preloaded IQ, each adapter call, public-state reads,
per-frame checksums, receipt-array assignments, and ordinary loop overhead.
It excludes input generation/hashing, adapter construction and filter/weight
setup, imports, result-file writes, hardware input and application output
transport. Per-chunk timers enclose slicing, `push`, checksum consumption and
the state read; additional loop bookkeeping is covered by the total timer.
The checksum sink promptly releases returned frames instead of accumulating
their arrays. Constructor and first-use startup costs require separate
consideration in an operational receiver.

Between push calls the observed occupancy maxima are zero pending input
samples, **384 mixed history samples**, and **16,320 pending frame samples**.
The allocated input buffer has capacity 8,192, the declared history bound is
455, and the output buffer has capacity 16,384. These sampled occupancies are
not transient peak allocation or external queue bounds. Reported lifetime
process RSS includes imports and generation of the complete IQ array; it
must not be attributed to retained adapter state alone.

## Reproduction and audit

`tools/benchmark_sdr_upgrade_streaming.py` supports separate `freeze` and
`run` commands. Freeze before generating IQ or collecting timings:

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
nice -n 10 python tools/benchmark_sdr_upgrade_streaming.py freeze NEW_OUTPUT

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
nice -n 10 python tools/benchmark_sdr_upgrade_streaming.py run NEW_OUTPUT
```

`run` refuses changed frozen sources, changed NumPy/SciPy versions, missing
thread settings, incorrect priority, or an existing timing directory. It pins
only its own process to the highest-numbered currently allowed logical CPU.
It retains every attempted run and warmup, per-call durations, output counts,
occupancies, process metadata, source snapshots and checksums. Exact versions,
CPU identity and command arguments are recorded in the release. The frozen
primary comparison requires all five complete runs to reach the nominal rate
with matching output identities and valid counts/geometry/state bounds.
That is an observed engineering comparison, not a future guarantee.

`tools/audit_sdr_streaming_throughput.py` independently recomputes rate,
quantile, count and state summaries from every retained timing record and
regenerates the input hash. All **18,514 checks** pass. Exact input byte identity
requires the recorded complex-exponential expression and operation order;
an algebraically equivalent float-phase sine/cosine expansion need not yield
the same complex64 record. Timing measurements are not rerun by the audit.

This short, memory-resident CPU test does not measure sustained-hour behavior,
USB loss, SDR acquisition, clock continuity, transport buffering, physical RF,
production output handling, multi-input array throughput, shared GPU load,
physical detection quality or live Pathfinder operation. Host load, power
management, CPU affinity, Python/NumPy/SciPy versions and instrumentation can
change observed service times.

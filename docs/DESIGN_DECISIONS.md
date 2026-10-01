# Design decisions

This file records choices that affect the interpretation of CHIME DTV
products. We keep the choice, its reason, and its boundary together so the
same question does not have to be reconstructed during review.

## The CHIME default is norm-corrected positive excess

The CHIME real-data path uses the following comparison:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the exact integer form of `F > null_power_ratio`, where

```text
null_power_ratio = 2 * target_norm_sq / reference_norm_sum_sq
```

is the flat-floor `H0` zero point implied by the packed weights. Int4
quantization leaves the target and reference norms unequal. Independent
recomputation from the shipped ATSC 14--36 manifest gives a `null_power_ratio` range of
0.9853298815 to 1.0111111111 rather than exactly 1.

We compare each channel with its own `null_power_ratio` so that the rule is defined by the
shipped weights and can be reproduced from integer powers. This avoids
fitting a separate operational threshold before the bounded CANFAR test.
Products written under the earlier `F > 1` rule retain that rule in
`mask_rule` and must be interpreted under their recorded convention.

## `K = 128` is the CHIME baseline; `K = 64` is the CHORD baseline

The compiled detector family and detector-core profile support `K = 64` and
`K = 128`. CHIME retains `K = 128` because it matches the CHIME receiver
profile, shipped CHIME manifest, and regression data. The full and pathfinder
CHORD profiles select `K = 64`; their 8,192-sample frames still produce the
frozen 128 windows per stream.

`K = 256` is not implemented or tested and would require a separate fixed-point
bound, weight bank, contract, and validation set.

`K = 512` is outside the current candidate set.

## Guard and reference terms have separate names

The public terms are:

```text
skipped_guard_bins
reference_offset_bins
```

They obey:

```text
reference_offset_bins = skipped_guard_bins + 1
```

A user-authored detector-core configuration specifies
`skipped_guard_bins`. Generated metadata may also record
`reference_offset_bins` so that the selected geometry can be audited.

## Reference placement is adaptive and recorded

The resolver retains the requested reference offset when the receiver
geometry permits it. If a reference reaches a circular FFT edge, it wraps. If
a reference would collide with the forbidden coarse-channel DC tone, it moves
away. A target/DC collision is invalid because moving the target would change
the signal being tested.

In the shipped ATSC 14--36 bank (rebuilt for the verified DC-centered frame
convention), DTV 14 is the only adaptive case: its lower reference wraps
across the coarse-channel edge. No shipped channel collides with, or shifts
around, the forbidden DC tone.

Run products retain the reference-placement summary. Validators and later
analyses should use the recorded placement rather than reconstructing it from
the nominal offset.

## The runtime bundle prepares, but does not implement, Kotekan

The runtime exporter writes:

```text
detector_contract.json
pilot_profiles.json
weights.bin
weights.manifest.json
sha256sums.txt
```

The planned deployment model is:

```text
same detector software on every node
same weight bundle on every node
first-frame integer CHIME channel ID selects active profile
non-pilot channel disables detector
pilot channel selects one weight-bank pointer
```

This repository does not contain a Kotekan stage. The bundle defines the
inputs that a later stage would consume; it is not evidence of deployment.

## Fine designated-set CFAR is implemented but inactive

The fine front end was designed as a candidate replacement for the logic after the dot product:
instead of squaring each window's dot product and summing incoherently over
all rows, the windows are integrated coherently (the padded window-axis
FFT, per feed) and only the feed sum remains incoherent. For W = 128
windows this recovers up to 10 log10(sqrt(W)) ~ 10.5 dB of deflection
sensitivity over the coarse reduction (less scalloping, bounded at 3.92 dB at
the fine-span edge), and the calibrated any-bin CFAR turns the refined
spectrum into a decision with a pre-registered false-alarm rate.

Candidate-selection record (2026-07): measured ROCs motivated implementing a
real-time fine designated-set CFAR on the device. Coarse-only operating points
were evaluated first: the coarse null bulk and
the weak-signal bulk overlap almost completely, so every coarse operating
point is one of two failures. The null_power_ratio point holds Pd ~ 1.0 everywhere but
spends 48.5% of verified-quiet time by construction, and null-calibrated
fixed-Pfa points collapse on weak channels (null = ch35 off-state,
n = 66; signal = 2024+ on-epochs of the first three scanned channels):

| Pfa (null quantile) | theta (F/null_power_ratio) | ch36 Pd | ch35 Pd | ch34 Pd |
|---|---|---|---|---|
| 0.10 (P90) | 1.114 | 0.904 | 0.996 | 0.099 |
| 0.05 (P95) | 1.124 | 0.872 | 0.996 | 0.089 |
| 0.015 (null max) | 1.146 | 0.805 | 0.996 | 0.071 |

The "null" of this record is a dated transmitter-off epoch, inferred from the
archive (ch35's off state), not a verified signal-free population; a
false-alarm limit for deployment is set only on a band's own verified,
null-like population (`characterization/false_alarm.py`).

The same frames carry a fine designated-set Pd ~ 0.99 on all three
channels at a measured Pfa of 0.091, with the crude +/-2-bin set and an
uncalibrated CFAR multiplier. The fine axis is the only one that holds
both ends at once --- more complete than any calibrated coarse point,
cheaper than null_power_ratio --- which was the design intent of the fine front end.
The candidate mask implementation therefore lives in the kernel: after the exact
int32 row sums, the device computes the padded window-axis FFT, the
incoherent feed sum, the per-frame null-bulk CFAR estimate, and the
designated-set compare, and emits one mask bit per aligned frame. The
queue handoff design is unchanged. Status: the power stage landed in
kernel core 2.1.0 (`FStat_Compute_FinePowers_U64` --- fxfft256 v1 on
device plus exact uint64 feed sums, bit-gated by
`tests/kernel/test_fine_powers_gpu.py`); the CFAR estimate and
designated-set compare remained downstream at that milestone pending the
calibration campaign.

Target form (decided 2026-08-01): one solid kernel. Block-per-stream
fusion computes the row sums in shared memory (never materialized to
global), runs the frozen fxfft256 in place, and accumulates the fine and
coarse powers in the same pass --- making the bit-exact marginal
identity an internal property of the launch --- with a last-block
epilogue (atomic completion counter) forming the designated-set decision
over the finalized feed sums and emitting the mask bit. Interface:
packed samples, packed weights, and per-channel bundle constants in; one
mask bit per aligned frame out (1 = reject, zero-reference forced 0);
row sums and exact powers become optional debug output pointers. The
epilogue is deterministic because it reads order-independent exact
integer sums. Governing rule unchanged: bit-identity with the frozen
reference and golden vectors. Staging: fuse-through-powers is a pure
refactor gateable against the existing golden vectors at any time; the
decision epilogue lands after the calibration campaign supplies anchors
and multipliers.

Status (2026-08-04): fuse-through-powers landed in kernel core 2.2.0
(`FStat_Compute_FusedFine_U64`): one launch, block-per-stream, row sums
in shared memory with the marginal identity internal, cooperative
fxfft256 butterflies (disjoint writes per stage, so bits are
schedule-independent), row sums exposed only through an optional debug
tap. Verified two ways: a threaded CPU emulation of the extracted device
code (one pthread per CUDA thread, a real barrier for `__syncthreads`,
runs bit-identical to the numpy/fxfft reference chain and stable across
repeated concurrent runs), and the on-GPU gate
`tests/kernel/test_fused_fine_gpu.py` (bit-equality against the composed
2.0.0/2.1.0 chain and the reference, batch case, tap-optional case, and
the footprint report: fused-vs-composed and tap-vs-no-tap timings). The
decision epilogue remains the only unfused stage, still gated on the
calibration campaign.

Status (2026-08-05): the decision epilogue landed in kernel core 2.3.0
(`FStat_Compute_FusedFineMask_U64`), completing the one-solid-kernel
form: packed samples, packed weights, and bundle constants in; one mask
bit per aligned frame out; marginals and row sums demoted to optional
debug taps. The epilogue runs in the last-arriving block per frame
(completion counter folded into the mask element itself;
threadfence-then-atomic visibility) over the finalized exact fine sums.

The decision arithmetic is frozen as **fine decision v1** (reference
`src/pilot_proxy/fine_decision.py`, gated by
`tests/core/test_fine_decision.py` against a `fractions.Fraction`
oracle): `F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])` held as exact rationals;
null bulk = the bundle's 256-bit mask (built from the pre-registered
`independent_bin_mask`: independent bins, guard- and census-excluded)
intersected with live denominators; CFAR estimate = the bundle-specified
*rank* of the ascending bulk F2 values (order-statistic CFAR; rank
selection by counting, so the selected value is unique under ties and
any representative yields the same decision); fire iff some designated
bin (anchor +/- width, modulo 256 --- the survey's window convention)
exceeds `multiplier_q16 / 2^16` times the rank value. Comparisons are
exact 128/192-bit integer cross products (fuzz-gated, 2M trials vs
Python big ints). Degenerate denominators never fire; a bulk not deeper
than the rank forces mask 0.

Rank-based rather than the survey's recorded location+scale CFAR, by
the same reasoning that chose the fixed-point FFT: float medians and
interpolated quantiles cannot be made bit-exact across
numpy/nvcc/compiler versions, while the rank statistic is exact,
needs no contamination fallback mode (order statistics reject the
measured 0.29% bulk tail by construction), and is calibrated by the
same null-quantile program --- the campaign picks (rank, multiplier)
from measured off-epoch nulls to hit the target false-alarm rate, and
the survey products' exact `fine_power_u64` terms --- target,
lower-reference, and upper-reference uint64 per fine bin --- supply
those quantiles, because `F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])` is
reproduced from them exactly. Archived schema v1 products, which kept
only the float32 ratio, cannot supply them. The operating point
(anchor, width, bulk mask, rank,
multiplier) is runtime-bundle data with provenance
(`fine_calibration` block, exported `pending_campaign` and validated
in both states by the bundle validator); the arithmetic is code. Activation
still requires calibrated bundle data, a schema/decision-contract revision,
and a survey epoch boundary. Under the current contract,
`fine_candidate_decision.active` is false and `reject_mask` remains the coarse
norm-corrected positive-excess decision.

Verification: threaded CPU emulation of the extracted device code
(real barrier concurrency; mask, fine, tap, and marginal outputs
bit-identical to the Python reference chain across noise, injected
pilot, wrap-anchor, degenerate-bulk, and tap-unbound scenarios; stable
across repeated concurrent runs), and the on-GPU gate
`tests/kernel/test_fused_mask_gpu.py` (reference bit-equality per
frame, injection firing, device-side degenerate rules, production
no-tap form, 20-repeat last-block determinism, and the epilogue-cost
rate report). `FStat_Compute_FusedFine_U64` remains bit-identical to
core 2.2.0.

Measured (A100, 2026-08-05, all 19 kernel gates passing): fused
production form 0.769 ms per 2048-stream frame (x55 margin against the
41.9 ms cadence; composed three-launch chain 2.144 ms, so fusion saves
1.375 ms; debug tap +5 us); candidate mask form 0.895 ms (x47 margin;
decision epilogue +0.127 ms). The complete implemented candidate costs ~2%
of the frame cadence.

The gating requirement is a deterministic verification FFT. The project's
validation standard is bit-exact agreement between the CUDA path and the
Python reference; cuFFT and numpy cannot bit-match each other (different
algorithms, operation order, FMA), so an activatable FFT must be owned by
the project and implemented identically on both sides.

Resolved (2026-07-31): the fixed-point variant is chosen and frozen as
**fxfft256 v1** --- reference `src/pilot_proxy/fxfft.py`, C port template
`cuda/fxfft256_ref.c`, golden vectors `tests/data/fxfft256_golden_v1.npz`
(56 vectors including the survey-measured pilot offsets and int4-dot
row-sum arithmetic), bit-compare gate `tests/core/test_fxfft256.py`. Spec
in brief: 256-point radix-2 DIT, Q15 twiddle table frozen as source
literals (nearest-integer, no rounding ties; +-32768 held in int32 and
exact under the rounding rule), butterfly rounding floor((v + 2^14) /
2^15), no stage scaling, inputs |re|,|im| <= 2^20 provably overflow-free
in int32 (production-bound row sums reach 2^14; contract-limit stress measures 28
of 31 bits). Measured against the analyzer's complex128 pipeline:
spectrum error <= 6.3e-4 of the spectrum peak across all golden families,
and the end-to-end fine statistic moves by <= 5.9e-4 relative (-0.00026
dB at a +1287 Hz detection peak) --- five orders below the 10.5 dB
coherent gain (`tools/fxfft_report.py`). After the rounded FFT every
downstream quantity (|X|^2, feed sums, the F2 numerators and
denominators) is exact uint64, so the candidate fine decision is
deterministic integer arithmetic end to end. The deterministic-float32
alternative (identical butterfly order, FMA disabled) is recorded as not
taken: it verifies equally but ports worse and leaves the integer
culture. Library FFTs remain cross-check oracles in tests; the CUDA
implementation must reproduce the golden vectors bit-for-bit. The offline
analyzer keeps its float pipeline; whether offline products switch to
fxfft at a possible activation epoch (so archive and real-time paths match
bit-for-bit) is an open item for the epoch-boundary migration.

Activation calibration inputs are all survey-supplied. Per-channel
measured line anchors and designated-set widths need on-epochs only,
which every detectable channel supplies. Per-channel CFAR threshold
multipliers need verified nulls: the measured null is ~19x wider than the
iid model, so the design-point P_FA is not achieved without empirical
correction --- conservative margins apply until enough verified
off-epochs accumulate, and the null-universality test with the
pooled-null program (classify epochs on/off by fine-line persistence and
census; test universality of the null across channels with off-epochs;
pool if universal) remains the path to deep, shared multipliers. The
bundle records anchors, sets, multipliers, and their calibration
provenance: epoch lists, quantiles, and source product hashes.

The survey recording is unchanged: the null_power_ratio positive-excess flag remains
the recorded convention mid-survey (exact, calibration-free, archive-
comparable), and any active-decision change must land at a survey epoch
boundary. Rate margin must be re-measured with the fine stages on
device, though 2048 256-point FFTs per ~42 ms frame should leave the
detector's lead over the correlator essentially intact. Measured
(2026-07-31, A100/sm_80, `test_fine_powers_gpu.py` report): 1.4 ms per
2048-stream frame for the fine-power stage alone, a x30 margin against
the 41.9 ms frame cadence --- the lead holds.

## The recorded threshold is data, not architecture

The survey keeps the norm-corrected positive excess (F > null_power_ratio) as the
recorded flag: it is exact in integer arithmetic, requires no calibration
input, and keeps the multi-year archive comparable. It is a recording
convention, not a tuned operating point --- scored against the ch35
off-state null it reaches Pd ~ 1.0 only at Pfa ~ 0.485 (Youden
J = Pd - Pfa ~ 0.51) on every channel; it buys completeness with half of
clean time.

Nothing in the architecture pins that choice. The operating point is
carried as per-channel rational data in the runtime bundle
(`positive_excess_half_threshold_num`/`_den` in `pilot_profiles.json`);
the CUDA comparison is threshold-agnostic, and `coarse_power_ratio` plus the fine
spectrum are persisted per frame, so every recorded epoch can be re-scored
offline under a different rule. Retuning a deployment is a bundle
regeneration and revalidation, not a kernel or schema change. Future
designs are free to move the threshold; two defensible selection policies
are recorded for whoever does:

- Fixed false alarm (CFAR): measure a transmitter-off null, set the
  threshold at its (1 - P_FA) quantile, and let Pd follow. This is the
  policy the fine candidate implements (see the section above), and the right one
  when clean integration time is the scarce resource.
- ROC-derived (Youden's J): sweep the threshold and maximize
  J = Pd - Pfa. Equal-cost by construction --- on a weak channel it will
  buy detections with false alarms one-for-one (see ch34 below) --- so it
  is the neutral default when no cost model exists for missed
  contamination versus discarded clean time. Measured on this survey for
  reference; not adopted for deployment, because the masking costs are
  not equal.

The deployable thresholds of a band now form an interval: its lower end is
the false-alarm limit eta_Pfa (where a verified null exists; otherwise the
tightest evaluable threshold), its upper end the science limit eta_sci, the
largest threshold whose kept residual the science side's tolerance accepts.
The detector side writes every evaluable candidate and chooses none; this is
also why `characterization/roc.py` reports Youden's J and never selects a
threshold with it: J prices a missed contamination and a discarded clean
frame equally, and they are not equal.

First calibration, 2020-2026 survey products (null = ch35 off-state,
n = 66 frames, proxied across channels; Pfa granularity ~1.5%):

| channel | coarse J_max (theta) | Pd / Pfa at J_max | fine designated set, measured line +/-2 bins |
|---|---|---|---|
| ch36 (506) | 0.837 (1.126) | 0.867 / 0.030 | Pd = 1.000 (no verified null) |
| ch35 (521) | 0.996 (1.157) | 0.996 / 0.000 | Pd = 0.996, Pfa = 0.091, J = 0.905 |
| ch34 (537) | 0.752 (1.030) | 0.979 / 0.227 | Pd = 0.991 (no verified null) |

Three readings. The coarse J-optimal thresholds land at or just above the
null bulk edge when separation is good --- J rediscovers the CFAR
intuition --- and dive into the bulk only when it is not. The fine path at
its fixed CFAR point dominates the coarse J-optimum exactly where the
coarse axis struggles: the coherent gain expressed as ROC separation
rather than dB. And the designated set must anchor at the measured pilot
line, not the nominal bin --- every detected transmitter sits at a stable
nonzero offset, and nominal-bin sets collapse. The measurement is
`pilot-proxy characterize roc` (`src/pilot_proxy/characterization/roc.py`,
formerly `analysis/youden_j.py@bdfcdce`); it recomputes the table from the
released per-pilot products.

## The source stamp is memoized per process

`detector_version` embeds a hash of every `*.py` under `src/pilot_proxy`.
The imported code cannot change within a process, so the hash is computed
once per process (first observation) and reused, rather than re-read from
disk at each channel's `begin()`. A source-only stamp difference is
forgivable for cross-pilot combine and stacking, which compare only the
geometry tokens; resume is stricter --- it compares the full
`detector_version` and aborts on any source-tree or package-version
change, so a new build needs a clean `--output-dir`. The stamp-to-commit
mapping is recoverable by recomputing the hash at each candidate
checkout.

## The stored fine surface is the exact integer terms

Schema v3 records the fine axis as `fine_power_u64`, shape
`(N, 3, fine_num_bins)` uint64 --- the target, lower-reference, and
upper-reference terms per fine bin --- together with the fine-decision
contract scalars (`fine_status`, `fine_num_bins`, `fine_pad_factor`,
`fine_guard_fine_bins`, `fine_p_fa`, `fine_designated_bins`,
`fine_census_excluded_bins`, `decision_contract_json`). Everything derived
from those terms --- the float ratio, the CFAR location, scale, threshold and
mode, the null-bulk exceedance fraction, and the ragged exceedance list --- is
recomputed in post-processing and is not written. The ragged list
(`fine_threshold_exceedance_frame`, `fine_threshold_exceedance_bin`,
partitioned by `fine_threshold_exceedance_count`) was a schema v1 diagnostic
surface and is retired. Products that violate the public contract are
unsupported development outputs and must be regenerated from authoritative
inputs.

## Deferred work

The following items are outside the minimum detector result and remain
separate tasks:

- tone catalog and intermodulation classification;
- LimeSDR loopback;
- Kotekan integration;
- a `K = 256` implementation;
- production integration of additional threshold modes;
- a common-mode power veto; and
- threshold fitting from CANFAR-measured means.

The analysis directory can test alternative thresholds after the scan. That
does not change the shipped real-data mask or add a new production mode. The
current paper should therefore distinguish the recorded detector output from
post-hoc sensitivity studies.

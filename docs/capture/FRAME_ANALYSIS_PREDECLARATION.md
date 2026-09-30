# Frame-level analysis of the matched-capture dumps: predeclaration

Written 2026-09-17 before any science dump has landed. Applies to D1, D2, D3 (3 s, 71 frames) and, as a dry run, to the
pilot (0.5 s, 11 frames). Inputs are the per-frequency products of reduce_dump.py (one file per coarse bin: stacked
visibilities per frame over the 7,155 redundant baseline classes, per-input powers, block fine spectra, pilot cutouts)
and the kernel-of-record outputs of chime-run on the same dump (per-frame mask and pilot excess per channel). Frames are
aligned: both start at the common FPGA origin of the dump, so frame k is the same 16384 samples everywhere.

## Quantities, in the board's units

The board's A is a power ratio per kept frame: retained DTV power over the thermal power of the bin (board_method.md,
tradeoffs.csv `evaluation_G1_allowance`), and lambda_q is the tolerance in the same ratio (channels.csv
`conditional_reference_amplitude`). The frame-level physical residual is built as the same ratio in the visibility domain.

Per coarse bin b and baseline class s (below), with V_k the stack mean of the raw products in frame k (sum over the
16384 samples, averaged over the products in the class) and P_k the mean live-input power in frame k:

1. Coherent power over a frame set K, noise-bias free: C_s,b(K) = ( |sum_K V_k|^2 - sum_K |V_k|^2 ) / (|K| (|K|-1)).
   Two frames of pure noise contribute nothing on average; a component with a fixed phase across frames survives.
2. Amplitude ratio: a_s,b(K) = sqrt(max(C, 0)) / (16384 * mean_K P_k). For a plane wave from one transmitter that
   reaches every input with the same power this is the DTV-to-thermal power ratio of the bin, the board's A.
3. Sky and instrument reference: r_s,b = the 10th percentile of a_s,b(all frames) over the bins within 40 MHz of b.
   The DTV excess is e_s,b(K) = a_s,b(K) - r_s,b, floored at zero. Stated limitation: the reference is a lower
   envelope over neighbouring bins, not a fit; on a band where every neighbour carries DTV it is an upper bound on the
   sky term and the excess is a lower bound. That is the conservative direction for an exclusion.
4. Per-input retained power (the covariance route's U_i): for input i, U_i,b(K) = (mean_K auto_i,b,k - ref_i) / ref_i
   with ref_i the same input's power in the reference bins of the same frames; reported as the distribution over inputs.
5. Frame-lag structure: on the projection alpha_k = Re( V_k conj(m) ) / |m| with m = sum_K V_k, the structure function
   D(l) = mean over pairs of (alpha_k - alpha_{k+l})^2 for l = 1 .. |K|-1, normalised by the variance of alpha. The
   fast-variance fraction is phi_fast = D(1) / (2 var), and the within-dump coherence gain is
   G_dump = 1 + 2 sum_{l} rho(l) with rho(l) = 1 - D(l) / (2 var), truncated where rho first crosses zero. G_dump is a
   lower bound on G when the plateau is not reached within the dump; the archive's 300 s bound (30 s on 31 and 35)
   caps it from above.

Frame sets: all frames; kept frames (kernel mask 0); rejected frames (mask 1). A channel with fewer than 5 kept frames
reports its kept-frame quantities as absent, not as zero.

Baseline classes s: same-polarisation classes with (EW cylinder step, NS feed step) in
{(0,1), (0,2), (0,4), (0,8), (1,0), (1,1), (2,0), (3,0)} for both polarisations, keyed as in the products
(cylinder = id // 512, pol block = (id // 256) % 2, position = id % 256; EW 22.0 m per cylinder, NS 0.3048 m per feed).

## What is reported per channel

- The pilot bin and the in-band median over its bins of e_s,b for each class, on all, kept and rejected frames, with
  the kept-frame value carried as the physical A candidate; its ratio to the kernel's pilot-inferred shelf on the same
  frames is the proxy-to-physical transfer of RULE_CHANGE_2026-09-16.md.
- phi_fast and G_dump per class at the pilot bin and the in-band median, and the lag at which rho first crosses zero.
- U_i statistics: median, 90th percentile, fraction of inputs above lambda_q.
- Frame count, kept count, and the kernel's median pilot excess.

## Outcome rule (unchanged from MATCHED_CAPTURE.md item 5)

A channel whose kept-frame excess lower bound times G_low exceeds lambda_q on two of three dumps is excluded on physical
evidence; G_low = 1 where tau_needed is below one frame, otherwise G_dump. A channel whose kept-frame excess upper bound
(the all-frames value, which includes rejected frames) times the archive's G bound sits under lambda_q passes the
tolerance and proceeds to the floor gate and covariance route. Everything else is reported, not ruled.

No threshold in this document is tuned after seeing the data; changes are logged as amendments.

## Status of the amendments, as of 2026-09-19

This document is chronological, so a linear reader meets superseded text before learning it was
superseded. These tables are the index. They also separate the two kinds of entry, because sixteen
undifferentiated amendments read as sixteen changes of mind, and that is not what happened: six are
design decisions taken as the data arrived, one is housekeeping, and the rest are corrections of
defects found by audit, several of them defects in the analysis rather than in the design.

### The decisions. These are the method, and they are what a reader should judge.

| # | when | the decision | still in force |
|---|---|---|---|
| 1 | before any science product was opened | Keep-all is the top rung of the policy ladder, and a channel with no transmitter of its own simply passes at keep-all. Two conditions: (a) the era carries no pilot, and (b) the in-band excess is below tolerance. If (a) holds and (b) fails, something radiates without a pilot and the channel goes down the ladder. | yes |
| 3 | before any cadence product was opened | The coherence time is the lag at which the structure function of the per-epoch level reaches 1 - 1/e of its plateau. This sets the gain and is the most consequential definition in the ruling. | yes |
| 4 | on the manuscript review | The level is the larger of the two same-polarisation readings, because a residual is a bound and a rescue must survive the worse feed. | yes |
| 5 | after a preview of eleven epochs | The coherence time is read on the polarisation that sets the level, and the archive's trim probes are mirrored, after a burst on channel 27 faked a short coherence time. | yes |
| 6 | after the measured-gain table | The lowest-epoch bound for channels with no live pilot node: if even the quietest of fourteen epochs fails, no mask could have saved it. This is what excises 30 and 31. | yes |
| 7 | after the audit checks | Rule on the baselines that carry the BAO measurement, keeping the 0.3 m reading as a reported worst case, with a 3 dB allowance. | yes |
| 8 | after the first adversarial audit | Channel 37 is the control: an excess counts only where it exceeds what the estimator returns on a channel with no allocation. This made the ruling conservative and removed many apparent excisions. | yes |

Two later entries are also decisions rather than corrections: amendment 13's admission of the 2020
full-array cohort as a separate basis for channel 24, and amendment 15's rule that a range's own
trim probes may show a channel cannot be excised though never that it can.

### Housekeeping

| # | what |
|---|---|
| 2 | The dump cap is 546,875 samples, so a science dump holds 33 frames and not the 71 the predeclaration assumed. A metadata correction; no estimator changed. |

### The corrections. These are defects found after the fact, most of them mine.

| # | what was wrong | direction | count after |
|---|---|---|---|
| 9 | Acted on an audit finding that the forecast prices no baseline below 20 m. The finding was false and was not verified before being applied. | removed 8 excisions wrongly | 5 |
| 10 | Reversed 9. The audit had read a hardcoded label in the forecast bank's metadata instead of the settings beside it. | restored them | 7 |
| 11 | A class's coherence time was read on a different polarisation from the level it priced. Channel 22 was excised on a coherence time belonging to a polarisation sitting at the noise floor. | removed 1 excision | 6 |
| 12 | A systematic audit of every amendment against its code. 68 differences claimed, 15 survived, none moved a verdict. | none | 6 |
| 13 addendum | The pilot-to-in-band correction came from a file that could not be reproduced. Withdrawn and measured properly. | weakened channel 24's claim | 6 + 24 |
| 14 | The published bound was priced from the wrong threshold; the positive control's negative half was circular. | none | 6 + 24 |
| 16 | The even-count aggregation was never declared. Worth up to 14.9 dB of published bar, and identical verdicts under all four conventions. | none | 6 + 24 |

The honest summary of the corrections: one of them, amendment 9, was itself a mistake, and the rest
of the sequence exists because reversing it prompted checking everything else. That is how the
polarisation defect and the undeclared convention were found. No correction after 11 moved a verdict.

The ruling as it stands: six channels excised on the 2026 capture, 15 on the short BAO baselines and
17, 30, 31, 33 and 35 on the long; channel 24 excised on the separate 2020 basis, conditionally; no
channel keeps anywhere; twelve channels carry a published bound; channels 25 and 27 are shown not
excisable; channel 16 is marginal.

## Amendment 1 (author, 2026-09-16 evening local, revised the same night before any science-dump product was opened): the keep-all policy

Every channel is judged by the most forgiving mask policy that passes tolerance. The top rung of that ladder is
keep-all: a ceiling threshold set 6 dB above the highest per-frame pilot excess of the channel's current era, so that
in normal operation no frame is masked (retention reported, expected 100 percent) while a transmitter that turns on,
moves or strengthens is caught on the day. There is no separate "clean" verdict; a channel with no transmitter of its
own simply has keep-all as its passing policy.

A channel passes at keep-all when (a) its current era is representative and carries no pilot: the era begins at the
last persistent level change in the dating run (transmitter_dating/changes.csv; no change means the whole archive)
and spans at least 90 days and 60 archived units, within which the per-unit pilot-to-noise median is below 0 dB and
no unit exceeds 6 dB; and (b) in the dumps the all-frames in-band excess over the sky reference is below lambda_q
with the residual white at frame level (phi_fast at or above 0.9) on two of three dumps. The measured excess and its
upper bound are reported on every row; "below tolerance" is a statement about the bound, never a claim of zero.

If (a) holds and (b) fails, something radiates in band without a pilot of its own and the channel goes down the
policy ladder like any other. Candidates for (a) on today's evidence: 34 (never a pilot, 2018 to 2026), and the
switched-off channels 19 (off since 2024-11-26), 26 (2023-04-09), 27 (by 2022-10-19), 32 (by 2023-02-10).

Table of record. The deliverable is one row per freq_id 477 to 844: channel; role (pilot bin or data bin; the pilot
bin is recorded on every channel whatever its verdict, so that changes can be seen); disposition and passing policy
without the delay-filter credit; the same with the deployed 200 ns cut credited; the measured excess and its bound;
lambda_q in each world; the rule and the evidence file that decided the row. Dispositions are keep (with its policy),
pilot, or excise (no policy passes with every credit granted); excised rows carry the subtraction bar, the dB a
competing method would have to recover on kept frames. Rows the data cannot decide carry the reason.

## Amendment 2 (2026-09-17, on reading the first product metadata, before any analysis): 33 frames per dump

CHIME's kotekan caps a baseband dump at 546,875 samples (1.4 s), so each science dump holds 33 full frames, not 71,
and the structure function reaches lag 32 (1.3 s), not 70. Every estimator above is unchanged; frame counts in the
outcome rule are read as 33. Additional dumps at the same sidereal slots add frames without changing the design.

## Amendment 3 (2026-09-17, 19:46 UTC, written while the ten cadence events were still on the CHIME buffer or in transfer and before any cadence product was opened): the coherence time from the cadence campaign, and how it enters the table of record

Purpose. About twenty channels are undetermined in the provisional table because they pass at the within-dump gain
(lags to 1.3 s) and fail at the archive's gain bound, so their verdict is decided by where the coherence time tau_c
falls between 1.4 s and the bound. The cadence campaign (ten 0.2 s dumps at 0, 15, 30, 60, 120, 240, 480, 720, 960
and 1200 s after 16:02:08 UTC on 2026-09-17) supplies the lags from 15 s to 20 min that the archive does not have,
and the pilot and science dumps supply the lags from 2 h to one sidereal day. The estimator mirrors the archive's
definition of tau_c (RFIsher residual.py, correlation_time) so that the measured value replaces the archive's bound
in the same formula, G = min(tau_c, cap) / T_frame with the cap one sidereal day.

Per-epoch level. For each channel, on the baseline class (0, 1) with both polarisations averaged, and for two bin
sets (the pilot bin; the in-band median over the channel's bins), the epoch's level is the noise-bias-free amplitude
ratio of item 2 over all frames of the dump, x_d = sqrt(max(C_d, 0)) - r_d, with r_d the epoch's own sky reference of
item 3 (10th percentile over the bins within 40 MHz) and no floor at zero. Its noise variance is
sigma_d^2 = var_k(alpha_k) / n_d from the per-frame projections of item 5. The per-frame normalisation divides each
frame's stack mean by that frame's own live-input power rather than by the dump mean; the difference is below one
percent on every dump so far and is stated here so that it is not tuned later.

Structure function. Over every pair of epochs (a, b) less than one sidereal day apart,
D_ab = 0.5 [ (x_a - x_b)^2 - sigma_a^2 - sigma_b^2 ], the archive's noise-corrected form. Pairs are grouped to the
nearest predeclared lag class in log lag (15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 600, 720, 960, 1200 s, then
1, 5, 10 and 24 h); a class is populated with one or more pairs (the archive's 40-pair gate is a sparse-sampling
guard that the campaign's per-pair noise, reported beside every D, makes unnecessary). The plateau is the mean D over
pairs with lag above 7200 s (the archive's plateau_start_seconds), with its uncertainty from the scatter of those
pairs. tau_c is the lag at which the class-mean D first reaches (1 - 1/e) of the plateau, linearly interpolated
between adjacent populated classes, requiring at least three populated classes; these are the archive's
crossing_fraction and minimum_populated_lag_bins.

Status and the gain. Four outcomes, in the order tested:
- constant: the plateau is not positive at two standard errors. The level does not vary within the day at the
  campaign's precision, the residual is a fixed bias over the integration, and G takes the cap (2,054,312), exactly
  as the archive books a refused tau_c. This is the pessimistic end and is deliberate.
- measured: D crosses the target inside the populated cadence classes. G = tau_c / T_frame.
- bound: the plateau is positive but D stays below the target through the longest populated cadence class. tau_c is
  at least that lag and at most 7200 s; G is reported as the range, and only the lower end is used, so that the bound
  can excise but never keep.
- no cadence lags: fewer than three populated classes below 7200 s (a failed campaign); the row keeps its provisional
  reason.

Table of record. Two columns are added, the measured G (with tau_c and its status) and R at that G in each world, and
the disposition on every channel with a live pilot bin is then read at the measured G: keep if the loosest policy
passes at the measured G with the deployed cut (marked "also without credit" when it passes in world none); excise
if no policy passes at the measured G, or at its lower bound, with the deployed cut. Excised rows carry two
subtraction bars: the dB over tolerance at the measured G with the deployed cut, which is what a competing method
must recover from the residual it leaves, and the dB at G = 1, the least any method could face if it also
decorrelated the residual frame to frame. The archive-bound keep of the provisional table stays valid where it
already held (channels 15 and 29). Channels with no live pilot bin are unchanged by this amendment.

What is not done. The class-mean D is not fitted to a model; the crossing is read off as in the archive. The
between-epoch phasor coherence of cadence_lags.py (a statement about phase stationarity, used in section 15 of the
methods notes) is reported alongside but does not enter G, because the archive's tau_c is defined on the level.

## Amendment 4 (2026-09-17, 22:35 UTC, on the manuscript review of the provisional table, before the fourth science dump or any cadence product beyond event 1 was placed): the physical A is the larger of the two polarisations

The provisional table read the in-band excess on the (0, 1) class from polarisation 0 alone. The review found that
polarisation 1 reads higher on most channels (channel 15: 0.0094 to 0.013 against a clipped zero; channel 29: 0.0023 to
0.0031 against zero; channel 16: 0.013 to 0.018 against 0.0017 to 0.010), consistent with a horizontally polarised
transmitter coupling unequally to the two feeds. A is a bound on the retained DTV-to-thermal ratio, and a rescue must
survive the worse reading, so from this amendment the table's A is the larger of the two same-polarisation readings of
the (0, 1) class on the kept dumps, with both readings carried as columns; G_dump and phi_fast are taken from the same
polarisation as A. The keep-all bound, the tolerance ratios in both worlds and the dispositions are recomputed from it.
The cadence level of amendment 3 is unchanged (both polarisations averaged; it measures time structure, not a bound).
Nothing else in the estimators changes. The polarisation-0 table is retained as table_of_record_provisional_pol0.csv.

## Amendment 5 (2026-09-18, 01:35 UTC, written after a preview of eleven of the fourteen epochs, before the last three cadence events were placed): the cadence level follows amendment 4, and the archive's trim probes are mirrored

Two omissions of amendment 3 are corrected, and the circumstances are stated so that the reader can weigh them.

1. Polarisation. Amendment 3 averaged the two polarisations' complex stack means before taking the level. That is a
   fixed linear functional of the field and tracks the DTV power, but it can be small where the two polarisations'
   instrumental phases oppose, and it is not the quantity the table rules on since amendment 4. From this amendment the
   epoch level is the per-polarisation amplitude ratio of item 2 on the polarisation that sets the table's A (the larger
   of the two readings on the kept dumps), and the other polarisation's result is reported beside it.
2. Trim probes. The archive's correlation_time trims the upper tail of the level at the 90th percentile before the
   structure function, probes 75, 90 and 95, and refuses when tau_c moves by more than a factor two across the probes.
   Amendment 3 omitted this. It is mirrored here on the epoch levels: the primary estimate drops the epochs above the
   90th percentile of the level (one of fourteen), the probes drop those above the 75th and 95th, and a spread above two
   across the probes gives the status refused, which takes the cap as the archive does. A status of bound at every
   probe is consistent. The untrimmed result is reported beside the trimmed one.

Why now. The preview showed one epoch (C240, 16:06:08 UTC) on channel 27 with a level three times its neighbours,
decaying within the dump's 0.17 s, which alone drove a crossing at 101 s; the archive's trim exists for exactly this
tail. The ruling's outcome is insensitive to the choice: no channel with a live pilot bin can pass at a gain above about
240 (channel 29 needs tau_c below 10 s; every other channel less), and at the campaign's shortest lag of 15 s the
structure function is below 0.4 of the plateau on every channel whose level is measurable, so no channel's tau_c lies
below 15 s. The amendment changes which number is printed for tau_c, not which channels pass.

The implementation is cadence_tau.py with --level polmax --pol-table <table_of_record csv> --trim archive; the
amendment-3 form remains available as --level polavg --trim none and is run for the report.

## Amendment 6 (2026-09-18, 03:16 UTC, after the table of record with the measured gain was built and with the author's decision): the lowest-epoch bound for channels whose pilot bin is not in the dumps

Facts that prompted it. Six channels (19, 20, 24, 30, 31, 34) have no live node on their pilot bin in the September 2026
dumps, so the per-dump mask statistic that the calibrated policies read cannot be formed, and the provisional rule left
them undetermined. The archive's own pilot-bin history shows: the bins were recorded until 2026-04-16 (19, 24, 31),
2026-06-13 (20) and 2026-07-06 (34), so five of the six nodes were dropped within the months before the dumps (30's
stopped in 2023-09); the current era's nominal pilot is at the floor on 19 (lost 2024-11-26, from 14.9 to -19.5 dB),
on 20 (lost 2022-08-31, from 3.0 to -9.9 dB and declining) and on 34 (never above the floor, 2018 to 2026), while the
same channels show DTV on their live in-band bins at every epoch (19: 4e-3 to 1.6e-2; 20: 6e-3 to 2.3e-2), so the
transmitter is on and the nominal pilot is not where the detector looks, the channel 33 situation; 30 and 31 carry
strong nominal pilots in the archive (Q medians 154 and 25). The dating run's "transmitter off" on 19 is therefore a
loss of the nominal pilot, not of the transmitter, and is reworded.

The bound. Every masking policy keeps some subset of the epochs, so the residual it leaves cannot lie below the
residual of the lowest epoch. For a channel whose pilot bin is not in the dumps, A_low is the minimum over all fourteen
epochs (the four dumps and the ten cadence dumps, all frames) of the in-band median excess on the polarisation that
sets A (amendment 4), and the channel is excised when A_low, with the deployed cut, fails tolerance at the measured
coherence gain (or the lower end of its bound, or the cap on a refusal): no policy can pass, whatever the mask would
have done. The bars are reported at A_low: the dB over tolerance at that gain, and at G = 1. A channel whose A_low
passes at that gain stays undetermined (a mask might pass; it cannot be evaluated). A channel with no products (24)
stays undetermined. The same column is reported for every channel; for the seventeen with a pilot bin it can only
confirm the verdict already made, since their kept-set residuals are above their lowest epoch.

Why it is valid. A_low is a lower bound on the kept-frame residual of any policy because masking removes frames and
cannot lower the residual of the frames it keeps below the lowest epoch's; the cadence epochs of four frames are
included so that the minimum is taken over the finest sampling available, which makes the bound weaker, that is,
harder to excise on. The direction is the conservative one for an exclusion.

## Amendment 7 (2026-09-18, 03:49 UTC, after the audit checks of the table of record, on the author's decision): the ruling is read on the baselines that carry the BAO measurement

Facts that prompted it. The table read A and the coherence time on the (0, 1) class, the 0.3 m baseline, where the
two feeds see the transmitter almost identically and the residual is largest. The 21 cm BAO scales at these redshifts
(k of 0.03 to 0.3 per Mpc) lie at baselines of roughly 10 to 100 m; the shortest baselines probe smaller k and do not
carry that measurement. On the three east-west classes in the same products, (1, 0), (2, 0) and (3, 0) at 22, 44 and
66 m, the in-band residual is 3 to 22 dB lower than on the 0.3 m baseline, and its level decorrelates sooner where the
measurement has the precision to say (tens of seconds to forty minutes), consistent with the low between-epoch phase
coherence on those baselines (section 15 of the notes). The rule's own logic, that an exclusion must survive every
credit the data can give, requires the exclusion to be read there.

The basis. For every channel, A_bao is the median over the three east-west classes of the per-class in-band excess
(each class read on the larger of its two polarisations, amendment 4; the median over the kept dumps of the policy
under test). The coherence time tau_bao is the median of the measured east-west class values of the amendment-5
estimator run on each class; where a class is bound its lower end is used; where every east-west class is refused
(the level near the estimator's floor, the trim probes disagreeing), the shortest-baseline result is used as the upper
bound on persistence and the row says so. G_bao = min(tau_bao, cap) / T_frame. The lowest-epoch bound of amendment 6
is read on the same basis (the minimum over the fourteen epochs of the east-west median).

Dispositions on this basis, with the 3 dB allowance the record section already carries: excise when no policy passes
at G_bao with the deployed cut by more than 3 dB; keep when the loosest policy passes at G_bao with the deployed cut
(marked also without credit where world none passes too), stated as keep on baselines longer than about 10 m, since
the 0.3 m reading is over tolerance on every channel; undetermined when the best policy lies within the 3 dB allowance
(marginal), or when the east-west coherence is unmeasurable and the verdict flips between the fallback and the
measured gains. The 0.3 m reading is kept in every row as the worst case (the columns of the previous table), and the
subtraction bars are reported on the BAO basis and on the worst case.

What it changes. From the table with the measured gain: channel 27 keeps (9 dB under on the BAO basis), channel 29 is
marginal (about 2 dB over), channel 14 is undetermined (its east-west coherence is refused in every class and the
fallback gain decides the sign); the other nineteen measurable channels remain excised by 9 dB or more on the BAO
basis, before the concession checks, which are repeated on this basis. Every bar falls; the direction is the one a
reviewer will accept.

Tolerance wording. The tolerance against which the excisions hold is the least strict the forecast prices: the primary
systematic budget of one sigma (zeta = 1) in the deployed world, with the 200 ns delay cut credited at 11.4 dB. The
ground-filter (sidereal-mean) credit is not included; it is granted in the robustness check, where on the east-west
baselines the measured between-epoch coherence of 0.1 to 0.7 makes it small.

## Amendment 8 (2026-09-18, 04:04 UTC, after the adversarial audit of the ruling, finding S1): the control floor and the baseline-resolved ruling

Facts that prompted it. Channel 37 (608 to 614 MHz, no DTV allocation) is the one control in the dumps. Read through
the same estimator it returns a positive, persistent excess on every class and epoch: 2.1e-3 on the 0.3 m class,
3.8e-4 at 2.4 m, 3e-5 to 8e-5 on the classes from 9.8 to 78 m and 22 to 66 m (baseline_floor.csv). The tolerance on A
at the ruling gains (1e-6 to 3e-6 at G of 85,831; 1e-7 at the cap) lies 16 to 45 dB below that reading. A channel whose
excess is not measured above the control's cannot be shown to be over tolerance, or within it, by this capture. The
predeclaration's item 3 called the excess a lower bound; the control shows it has a floor.

The floor. For each class and polarisation the floor is the mean of the channel 37 bins' excess over the four science
dumps, with its scatter over those bins and dumps. A channel's excess on a class is measured when it exceeds the floor
by more than three scatters; the excess net of the floor, with the scatter as its uncertainty, is what enters R.

The ranges. Baseline classes are grouped by what they carry: the shortest (0.3 to 2.4 m; below the BAO scales; the
worst case, kept as such), the short BAO baselines (9.8 and 19.5 m north-south), and the long BAO baselines (39 and
78 m north-south; 22, 44 and 66 m east-west), the 21 cm BAO modes at these redshifts lying at 10 to 100 m. The
coherence time is measured per class with the amendment-5 estimator; each range takes the median of the measured
values of its classes, the lower end of a bound, and the 0.3 m result as the persistence upper bound where every
class in the range is refused (stated on the row).

Dispositions. A channel is excised on a range when its excess is measured above the floor on at least one class of the
range and, net of the floor, fails tolerance by more than the 3 dB allowance at the range's gain with the deployed
cut under every policy; it is undetermined at the floor on a range when its excess is not measured above the floor
there (the capture cannot decide it either way, and the row states by how many dB the floor itself exceeds the
tolerance at that gain); it keeps on a range only when its excess is measured and passes, which the floor makes
impossible at these gains, so the table reports no keep. The row's verdict is given per range, and the summary
verdict for the 21 cm BAO measurement is: excise where excised on the long BAO range; excise on the short BAO
baselines only, where excised there and at the floor beyond; undetermined at the floor where at the floor on both.

What lowers the floor. Calibrated gains before stacking (the redundant stacks are uncalibrated and add incoherently),
a control block outside the DTV allocation and the 600 MHz cellular band in the post-upgrade dump, and longer
captures; the rows say so.

Addendum to amendment 8 (04:20 UTC, on reading the north-south class coherence times, before the per-range table was
built): where every class of a range is refused by the level estimator, the persistence bound is used only where the
between-epoch phasor coherence of the range's classes over the six dump pairs (5 to 22 h; cadence_lags on the
reference-subtracted excess) exceeds 0.7 in median, which is a measurement of persistence by a different estimator;
otherwise the range's gain is unmeasured between the within-dump gain and the persistence bound and the channel is
undetermined on that range (its measured excess above the floor is reported as unpriced). The per-class coherence
times are in cadence_tau_ns{32,64,128,255}.csv and cadence_tau_ew{1,2,3}.csv; the phasor coherences in
lag_coherence_allclasses_D.csv (classes (0,1), (0,8), (1,0) only; the north-south long classes are computed for this
addendum with the same script).

Second addendum to amendment 8 (04:35 UTC, author's decision): every credit the survey applies enters the ruling at
its measured value. The deployed 200 ns delay cut is credited at 11.4 dB on A as before. The ground filter (sidereal-
mean subtraction) is credited per baseline class from the between-epoch phasor coherence rho of the reference-
subtracted excess over the six dump pairs (5 to 22 h; the quantity the subtraction acts on, under RFIsher's model in
which a residual constant within a day is removed whatever it does day to day): the surviving power fraction is
1 - rho^2 with rho the median over the range's classes clipped to [0, 0.95]. The ratio with both credits is the ruling
quantity in the deployed world; the ratio with no credit at all remains the world-none column for the rescue
direction; the bar on an excised row is reported with both credits, and the credit itself is printed.

## Amendment 9 [ITEM 1 SUPERSEDED BY AMENDMENT 10] (2026-09-18, 05:30 UTC, after the second adversarial audit of the committed ruling, RULING_AUDIT_ROUND2_2026-09-18.md, findings R1 to R4; author's standing instruction that the ruling must survive every objection): the ruling is confined to the forecast's baseline domain and to measured coherence times

1. The forecast's baseline domain (R3). The forecast of record is the RadioFisher CHIME layout with baselines from
   20 to 128 m and a synthetic baseline density with zero weight below 20 m. Its tolerance therefore prices no
   residual on the 0.3, 2.4, 9.8 or 19.5 m classes. The range at 9.8 and 19.5 m is renamed "below the forecast's
   baseline cut"; its readings are reported (the excess above the control floor, the tolerance applied there for
   information) but rule nothing. The ruling is made on the long range alone (39 and 78 m north-south; 22, 44 and
   66 m east-west), which lies inside the domain. Rebuilding the bank with the as-built density is future work.
2. The lowest-epoch bound (R1). An epoch on which no class of the range is measured above the floor is an epoch at
   the floor, and the bound is then at the floor and cannot excise; the implementation is corrected to the rule as
   written in amendment 6.
3. The persistence fallback (R2). The between-epoch phasor coherence measures the share the ground filter removes,
   not the coherence time of what it leaves, so it no longer supplies a gain. Where the level estimator returns no
   measured class on the long range, the channel's own measured coherence time on the 19.5 m class is used if there is
   one (a measurement on the same channel at the nearest priced-adjacent baseline; stated on the row); otherwise the
   range is unpriced and the channel undetermined.
4. Marginal readings (R4). A bar within the 3 dB allowance is marginal, as before; a bar whose probe range (the three
   trim probes of the measured class) crosses the allowance is reported with that range.

The count this leaves is the one both audits support: the channels excised for the 21 cm BAO measurement are those
measured above the control floor on the long range and over the least strict tolerance there with both credits;
every other channel is undetermined, with its measured excess below the cut reported for the collaboration.

## Amendment 10 (2026-09-18, 19:40 UTC, on re-deriving the forecast's baseline domain from the response banks themselves rather than from the bank meta's label; evidence in forecast_baseline_domain.py and FORECAST_BASELINE_DOMAIN.txt): item 1 of amendment 9 is void, and the short BAO range rules

Amendment 9 item 1 withdrew the 9.8 and 19.5 m north-south range from the ruling on the ground, taken from audit
finding R3, that the forecast of record is the RadioFisher CHIME layout with baselines from 20 to 128 m and a
synthetic baseline density with zero weight below 20 m. That ground is false, and the finding is refuted.

1. What the banks were built on. The two tolerance worlds are read from four response banks in
   results/canfar_reanalysis_2026-09-09/archive/response-banks: the world none tolerance from
   fisher_bank_chime2022_pres_dense.npz, and the deployed world from the three kfg_fac banks at 22, 44 and 80. All
   four carry config chime2022, whose experiment dictionary is CHIME's as-built one from
   RadioFisher/chime2021/experiments_CHIME.py, with Dmin 0.305 m, Dmax 102 m, and the as-built baseline density
   chime2021/array_config/nx_CHIME_800.dat. The density's sha256 recorded in each bank matches the file on disk at
   b6c8c109. Its support runs from 0.176 to 101.8 m, and 11.0 per cent of its integral lies below 20 m over 57
   nonzero rows.
2. Where the audit went wrong. Every bank's meta carries the string "CHIME (RadioFisher 'yCHIME', mode icyl)". That
   string is a hardcoded literal at fisherbank.py line 517, written into the meta of every bank whatever the config.
   The audit read the label and inferred the generic RadioFisher CHIME dictionary, Dmin 20 m and Dmax 128 m, and the
   synthetic table nx_CHIME_800_synth.dat. The settings block recorded a few lines below the label in the same meta
   is what was actually built, and it names the as-built density. The synthetic table is the default of
   survey.chime_experiment, which is the bull2015 path; the ruling's tolerances do not come from it.
3. Why Dmin and Dmax do not confine the domain. In RadioFisher's interferometer_response, a supplied n(x) table is
   interpolated directly and Dmin and Dmax are never consulted; they gate only the uniform-density fallback taken
   when no table is given. The priced domain is therefore the support of the table, 0.18 to 102 m, and modes outside
   it are given 1/INF_NOISE and carry no weight. Every baseline class the ruling reads is inside the support, with
   density from 4.1e5 at 78 m to 1.8e8 at 0.3 m; the short classes carry the larger density, as a close-packed
   cylinder array requires.
4. What changes. Item 1 of amendment 9 is void. The 9.8 and 19.5 m north-south range is priced by the forecast of
   record, is renamed the short BAO baselines again, and rules as the long range does, under the same rule: excised
   when measured above the control floor and over the least strict tolerance with both credits. Items 2, 3 and 4 of
   amendment 9 stand unchanged, and no other rule, estimator, credit or threshold moves.
5. What does not change. The 0.3 and 2.4 m classes are also priced, and this is now stated rather than denied, but
   they continue to be reported and not to rule. Two reasons. The rule has always ruled on ranges rather than single
   classes, and the shortest range is one class; and under the asymmetric ruling rule, declining to rule on a class
   that shows large excess is the conservative direction, since it can only move a channel toward undetermined and
   never toward excision. Their readings are carried on every row for the collaboration.
6. Standing. No rebuild of the Fisher bank is needed or contemplated. The bank of record already is the as-built
   density, so the future work named in amendment 9 item 1 is discharged by this amendment, not by a computation.
7. Probe robustness becomes a rule (closes audit finding R4). Restoring the short range put channel 16 over the
   allowance at 4.1 dB on a class whose own three trim probes place it between 2.8 and 4.5 dB, straddling the
   allowance. The asymmetric ruling rule excises only with every credit the data can give, so a bar that falls inside
   the allowance at the channel's own least trim probe is not an excision. From here a range excises only if the bar
   exceeds the allowance both at the coherence time of record and at the least of the measured class's trim probes;
   otherwise it is marginal. Every row carries the least-probe coherence time and the decibels it costs. The rule was
   written before the result was read, and it demotes exactly one channel, 16, to marginal. The five excisions
   standing before this amendment survive it with 15.9 dB or more, which is the check that it is a rule and not a
   choice.
8. The rescue coherence is stated on every excised row (closes audit finding R7). The ground credit is 1 - rho^2 in
   the channel's measured between-epoch phasor coherence, capped at rho = 0.95. That cap is a stipulation, and an
   excision that depends on it is an excision resting on a number the collaboration did not agree to. Rather than
   defend the cap, every excised row now carries the rescue coherence: the coherence a per-day ground filter would
   have to reach for the bar to fall inside the allowance. It is computed from the same bar and the same measured rho
   that produced the verdict, as rho_rescue = sqrt(1 - (1 - rho^2) 10^((allowance - bar)/10)). This replaces an
   argument about a ceiling with a condition the collaboration can test.

### The ruling under amendments 9 and 10

Seven channels are excised: 15, 17, 22, 30, 31, 33 and 35. Five (17, 30, 31, 33, 35) are excised on the long BAO
baselines, as before; two (15, 22) on the short BAO baselines, which amendment 10 returns to the ruling. Each is
measured above the control floor, over the least strict tolerance with both credits, at the coherence time of record
and at its own least trim probe, and each would be rescued only by a per-day ground filter reaching a between-epoch
coherence of 0.982 to 0.999 against a measured 0.34 to 0.84. Channel 16 is marginal. Every other channel is
undetermined, with its measured excess reported.

### Disclosure on the two short-range excisions (not a rule change)

The seven are not one population. The five excised on the long BAO baselines are measured above the control floor by 19
to 320 floor scatters, on both polarisations and on every class of the range. The two excised on the short range are
measured by 3.1 to 4.1 scatters against a gate of three: channel 15 on the 9.8 m class of one polarisation and the
19.5 m class of the other, at 3.1 and 3.2; channel 22 on the 9.8 m class of one polarisation alone, at 4.1. Their bars
are large because the coherence gain on that range is large, 49,999 and 49,310, not because the excess is strongly
detected. The three-scatter gate was fixed in amendment 8, before the ranges were read, and it is not moved now that
it decides two channels; the margin is disclosed instead, so the collaboration can weigh the two populations
differently. Nothing here changes a verdict.

Checked and recorded: amendment 10 moves no tolerance number. The lambda_none, lambda_deployed and suppression_db
columns are identical on all 368 rows before and after it. It changes no estimator, no credit, no gate and no
threshold. What it changes is which measured ranges are admitted to the ruling, and it adds one test that can only
move a channel toward undetermined.

## Amendment 11 (2026-09-19, on checking the polarisation of every coherence time against the polarisation whose level it prices; the check was prompted by an adversarial review of the amendment-10 table): the gain must price the residual the level measures

Amendment 5 item 1 reads the cadence level, and therefore the coherence time, "on the polarisation that sets the
table's A". When amendment 8 made the ruling per baseline class, that became a per-class choice. It was not
implemented as one. The class files of record (cadence_tau_ns32, ns64, ns128, ns255, ew1, ew2, ew3) were produced
without the predeclared --pol-table, so cadence_tau.py fell back to a per-class argmax of the raw median level over
all fourteen epochs, while the ruling prices, per class, the polarisation with the largest excess net of that
polarisation's own control floor over the four science dumps. Those are different quantities, and on four of the
seven excised channels they select different polarisations. The product A x G then multiplied one polarisation's
level by the other polarisation's persistence.

1. The coherence time on a class is read on the polarisation that sets A on that class. The estimator is unchanged
   and is re-run per class on each polarisation (cadence_tau.py gains a --pol option; the fallback run reproduces the
   files of record exactly, so the only difference is which polarisation is selected). The ruling then takes the row
   matching the class's priced polarisation.
2. A range's gain is aggregated only over the classes whose excess is measured above the control floor. G prices the
   persistence of the residual that A measures, and a class at the floor has no measured residual to persist. This is
   the same principle as item 1, applied to classes rather than polarisations.

What it changes. Channel 22 is no longer excised and becomes undetermined. Its only measured class on the ruling
range is the 9.8 m class on polarisation 0, where the estimator REFUSES (trim spread 471, probes constant 86164 /
183 / 218). The 2068.2 s that produced its 16.2 dB bar was measured on polarisation 1, whose level on that class
reads 1.6 floor scatters, that is, at the floor. Channel 33's bar moves from 25.8 to 24.9 dB and channel 35's from
17.3 to 17.2 dB; 15, 17, 30 and 31 do not move. Item 2 removes the table's only keep cell, channel 18 on the short
range, whose gain came from a class at the floor; no verdict changes with it.

The direction test. This amendment removes an excision and adds none. Every amendment that only ever adds excisions
should be suspected of being fitted to the result; this one is a correctness fix, and it costs the ruling a channel.

The ruling under amendments 9 to 11. Six channels are excised: 15 on the short BAO baselines at 19.8 dB, and 17, 30,
31, 33 and 35 on the long at 16.5, 28.6, 26.4, 24.9 and 17.0 dB. Each clears the allowance at the coherence time of
record and at its own least trim probe, and each would be rescued only by a per-day ground filter reaching a
between-epoch coherence of 0.988 to 0.999. Channel 16 is marginal. Every other channel is undetermined.

No rescue exists anywhere. One keep cell survives in the table, channel 23 on the short BAO baselines, and it is a
credited keep only: at the no-credit rescue test of the asymmetric rule it reads 28.27, which is 14.5 dB over
tolerance. No channel and no range passes the rescue test.

## Amendment 12 (2026-09-19, after a systematic audit of every amendment against the code that implements it): the corrections the audit confirmed, none of which moves a verdict

Amendment 11 was found by accident. The defect it fixed was of one kind, the predeclaration saying one thing and the
code doing another, so the whole predeclaration was then checked against the whole implementation, amendment by
amendment, with every claimed difference verified adversarially against the products. Sixty-eight differences were
claimed, twenty-four were carried to verification, fifteen survived it and nine were refuted. Not one of the fifteen
moves a verdict. The six excised channels, every range verdict and every disposition are the same before and after
this amendment. What follows are the corrections adopted and, as importantly, the ones tested and declined.

1. A bound class enters its range's coherence time at its lower end. Amendment 7 reads a range's coherence time as
   "the median of the measured east-west class values ...; where a class is bound its lower end is used", and the code
   admitted a bound class only where every class of the range was bound. Three channels have a bound class on their
   ruling range, all already excised: channel 17's 78 m class, channel 30's 22 m class and channel 33's 44 m class.
   Their coherence times become 211, 1136 and 2505 s, from 38, 686 and 1744, and their bars 24.0, 30.8 and 26.5 dB,
   from 16.5, 28.6 and 24.9. No verdict moves and no channel is added. The status is printed as "measured with a
   bound class" so the reader can see which rows it touches.
2. The least trim probe is read at the precision of the coherence time of record. The probes column prints whole
   seconds and the record value 0.1 s, so a class whose least probe is its primary could report a least probe above
   its own record value. It is clamped. The largest effect anywhere is 0.03 dB.
3. The lowest-epoch bound is read on the same polarisation as the rest of the ruling. Amendment 11 item 1 fixed the
   per-class polarisation for the coherence time and the level; amendment 6's bound still ran its own argmax at each
   individual epoch, so it could take one polarisation at one epoch and the other at the next. It now uses the
   per-class polarisation of amendment 11. No verdict moves.
4. The decibel figure an at-floor row reports is the detection gate, not the floor. The row said "the floor itself is
   X dB over tolerance" while X was the floor mean plus the three scatters of the detection gate, which is 5.6 to 7.3
   dB larger. The number is the useful one, being the smallest excess the capture could have detected, so the label is
   corrected rather than the quantity.

Tested and declined, each with the reason, so that a reader who repeats the audit finds them already answered:

5. The zero floor of the original item 3 is not applied to the class excesses or to the control floor, and stays that
   way. Censoring at zero biases the control's mean up and its scatter down, which lowers the gate the ruling turns
   on. Applying it as written was tested: every disposition, range verdict, coherence time and gain is identical, the
   largest bar shift is 0.17 dB, and exactly two cells cross the gate, neither changing a verdict.
6. Amendment 5 item 2's parenthetical is wrong and the code is right. On fourteen epochs the 90th-percentile trim
   drops the top two, not one; the 75 probe drops four and the 95 probe one. Promoting the 95 probe to primary, which
   is the amendment's literal reading, was tested and gives zero verdict differences, because since amendment 10 item
   7 the least-probe leg binds and does not depend on which probe is primary.
7. The ground credit's coherence is read with both polarisations averaged, as the second addendum to amendment 8
   specifies, and is not moved to the priced polarisation. On channel 16 the priced polarisation's coherence is 0.10
   against the pol-averaged 0.77, so the change would shrink the credit and raise the bar into an excision. The
   asymmetric rule excises only with every credit the data can give, so the larger credit is the one the rule
   requires; the smaller would excise a channel by withdrawing a credit, which is the opposite of the rule.
8. A refused or constant class is not booked at the sidereal cap. Booking the largest possible gain on a class where
   the estimator returned nothing would excise on the absence of a measurement. It would excise channel 16 at 17.6 dB
   and price channel 26, and it is refused for the same reason the persistence fallback was withdrawn by amendment 9.

The ruling under amendments 9 to 12. Six channels are excised: 15 on the short BAO baselines at 19.8 dB, and 17, 30,
31, 33 and 35 on the long at 24.0, 30.8, 26.4, 26.5 and 17.0 dB. Each clears the allowance at its own least trim
probe, the bars falling by at most 0.9 dB there, and each would be rescued only by a per-day ground filter reaching a
between-epoch coherence of 0.991 or more. Channel 16 is marginal. Every other channel is undetermined.

## Amendment 13 [ITS CORRECTION FIGURE SUPERSEDED BY ITS OWN ADDENDUM] (2026-09-19, on the author's decision after the campaign study): channel 24 is ruled on a separate basis, the 2020 cohort is published as a positive control, and every undetermined row carries a number

The ruling left seventeen channels undetermined. One of them, channel 24, was undetermined for want
of data rather than for want of a measurement: no live node records freq_id 676 to 691, so the 2026
capture has no products for it at all. The other sixteen were measured and could not be decided.
This amendment addresses the first directly, and makes the second say what it measured.

1. Channel 24 on the 2020 full-array basis. The CANFAR per-pilot baseband cohort
   (~/rail/datasets/baseband/canfar_pilots_10s, one directory per DTV pilot bin, 23 in all) covers
   channel 24 at freq_id 690 over sixteen epochs in 2020. Read through the same estimator on the
   same long BAO classes, channel 24's excess is 1.509e-02, which is 48.6 scatters above channel 35,
   whose transmitter was off until 2021-10 and which therefore supplies a transmitter-off null in
   the same frames. With the delay cut at 11.4 dB and the ground credit at its 0.95 ceiling, the
   most generous the rule allows, and on the worst case of the pilot-bin to in-band correction
   measured on the eighteen channels the 2026 capture does record (median -1.03 dB, worst case
   -3.55 dB), channel 24 is over tolerance by 30.4 dB at its own archive-measured correlation time
   of 7777 s, and by 3.2 dB even at 15 s, the shortest coherence time amendment 5 states any channel
   can have. It is therefore over tolerance at every coherence time the estimator can return.
   The condition, stated as a condition: the transmitter must still be on. The archive ledger
   (ch24_fid690.json) records it transmitting continuously from 2018-12 to 2026-04 with no declared
   off epoch, zero off frames and 9,470 frames flagged transmitter-on, its bin ceasing on
   2026-04-16 because no live node covers it and not because it went quiet.
   This row is NOT merged into the table of record and does not change any 2026 disposition. The
   2020 cohort has no channel 37 bin, carries one coarse bin per file rather than the in-band shelf
   the ruling reads, and has 8 frames per epoch against 33. Its verdict is carried separately, with
   its own control, its own correction and its own caveat, and is reproduced by
   ch24_2020_basis.py in the campaign directory.
2. The 2020 cohort as a positive control. The same estimator on the same baselines detects every
   transmitter known to have been on in 2020 and does not detect the one known to have been off.
   Channel 35, off until 2021-10, reads 0.3 scatters BELOW its own null. Channels 26, 27 and 32 are
   detected at 23.0, 59.0 and 112.5 scatters in 2020 and read at the control floor in the 2026
   capture, which is what a transmitter that has since been switched off looks like and matches the
   dating from the archive pilot series. Those three are therefore undetermined for a stated
   physical reason, not for a failure to measure. This is a control on the method, not an excision:
   their 2020 emission is a transmitter that no longer exists.
3. Every row publishes its bound. An undetermined-at-the-floor row now carries the detection gate
   in decibels over tolerance as a column of its own, {range}_gate_db, rather than only inside its
   reason string. Twelve channels acquire a long-range bound, from -1.3 dB on channel 27 and 5.3 dB
   on channel 29 up to 23.8 dB on channel 32. Five (14, 22, 24, 28 and channel 23 on the long range)
   carry none, because they are blocked on the gain rather than on the floor and no gate can be
   formed without a coherence time; their rows say so. Two of the published bounds are negative,
   channel 27's long range at -1.3 dB and channel 23's short at -2.0 dB, which means the capture's
   detection threshold there lies below tolerance and those ranges have a real acceptance region.
   No verdict changes: the column is a report of a quantity the ruling already computed.

### Addendum to amendment 13 (2026-09-19, on re-deriving the pilot-bin to in-band correction from the products): the correction was larger than stated and channel 24's excision is conditional on its coherence time as well as its transmitter

Amendment 13 carried a pilot-bin to in-band correction of median -1.03 dB and worst case -3.55 dB,
taken from a file produced during the campaign study. That file could not be reproduced: its four
values per channel do not match the pilot and in-band levels computed from the reduced products with
the ruling's own estimator, on any ordering of its columns, on any class tested. It is withdrawn.

The correction is now measured directly and reproducibly by pilot_to_inband.py, which writes
pilot_to_inband.csv: for each of the eighteen channels the 2026 capture records with both, the
median over the four science dumps and the five long BAO classes of the larger-polarisation excess
on the pilot bin and on the in-band median, with the same sky reference and the same estimator the
ruling uses. Over all eighteen the correction is -1.66 dB median and -5.70 dB worst. It tracks
transmitter strength, as it must: a channel at the control floor reads the same on its pilot bin and
in band, so a weak channel's ratio is near unity and carries no information. Over the four strong
transmitters, the population channel 24 belongs to, it is -4.55 dB median and -5.66 dB worst, and
that is the figure carried.

What changes. On the worst case for a strong transmitter, channel 24 is over tolerance by 28.3 dB at
its own archive-measured correlation time of 7777 s, not the 30.4 dB amendment 13 stated. More
importantly, amendment 13's claim that it is over tolerance "at every coherence time the estimator
can return" is WITHDRAWN. At 15 s, the campaign-wide floor of amendment 5, the bar is 1.1 dB, inside
the allowance. The excision requires a coherence time above 23.2 s. The archive measures 7777 s on
this channel, a factor of 336 above that threshold, so the excision stands with a large margin, but
it rests on this channel's own measured correlation time and not on the campaign floor alone, and
must be stated as conditional on both the transmitter and the coherence time.

Why this is recorded rather than quietly fixed. The error was of the kind this project has now been
bitten by three times: a derived quantity taken from an intermediate file whose construction was not
checked. It was found by re-deriving the quantity from the products rather than by reading the file
again. The lesson is in the method, and the method is now in the tree: every number the ruling
carries should be reproducible from the products by a script in this directory, and the two scripts
this addendum adds, pilot_to_inband.py and ch24_2020_basis.py, are.

## Amendment 14 (2026-09-19, after a six-lens adversarial audit of the banked position, which had never been applied to amendment 13): the published bound is corrected, and the positive control is stated at its true strength

Amendments 1 to 12 had been through three audits; amendment 13 had been through none. It was audited,
and ten defects survived adversarial verification. None moves a verdict: the six excisions on the 2026
capture, channel 24's excision on the 2020 basis, and every disposition are unchanged, and the table
reproduces. All ten are reporting or wording, and they are corrected here.

1. The published bound was the wrong quantity. Amendment 13 item 3 published the detection gate as
   the median over a range's classes of the floor mean plus three scatters, priced through the
   ruling. But the quantity that enters R is A_net, the excess already NET of the floor mean, so the
   threshold on it is three scatters, not the floor mean plus three scatters. And a range is detected
   when any one class clears, so the binding threshold is the LEAST class's, not the median. The
   column is redefined as the least class's three scatters, priced at the range's gain. Every
   published bound falls, by 2.6 to 5.8 dB: the long-range bounds now run from -4.0 dB on channel 27
   to 19.3 dB on channel 32, and channel 29, the closest to detectable, needs 1.6 dB rather than the
   5.3 dB previously published. No verdict moves, because the detection test itself was always
   computed correctly on A against the floor mean plus three scatters; only the reported bound was
   priced from the wrong threshold.
2. The negative half of the positive control is withdrawn as vacuous. Amendment 13 item 2 offered
   "channel 35, off air in 2020, reads 0.3 scatters below its own null" as evidence. It is arithmetic:
   channel 35 IS the null, the gate being its own mean plus three of its own scatters, and a sample
   median cannot exceed that. It demonstrates nothing and is struck. The positive half stands: the
   estimator detects the transmitters the archive records as on.
3. The control is weaker than amendment 13 claimed, and is restated. Its detection set depends on
   which channel is chosen as the null, and channel 35 is the most favourable of the twelve
   candidates; rebuilding the same gate on other nulls changes which channels are detected on nine of
   eleven. And the truth table is not independent of the statistic being tested, since the cohort and
   the archive pilot reading use the same coarse bin of the same files at the same sixteen epochs.
   What the cohort supports is therefore narrower than a blind control: the estimator, run on 2020
   full-array data, recovers the transmitters the archive independently records as transmitting, and
   channel 24 is among the loudest of them. That is what is claimed from here.
4. "Channels 26, 27 and 32 read at the control floor now" is false for two of the three. On the long
   BAO range all three are at the floor. On the SHORT range channels 26 and 27 are measured above it,
   at 4.16 and 4.67 floor scatters against the three-scatter gate, which is why they sit in the
   group whose coherence time is unmeasured rather than in the at-floor group. The dating claim is
   narrowed accordingly: their long-range excess has fallen to the control floor since 2020,
   consistent with the archive's transmitter dating, while a short-range excess remains that this
   capture cannot price. Only channel 32 is at the floor on both ranges.
5. Four stale passages in the record section, left behind by amendments 10 to 12 and corrected here:
   a sentence still describing an excise row as dropped "on baselines of 20 m and longer", which
   amendment 10 superseded; the short-range coherence-time list, still carrying pre-amendment-11
   values and the withdrawn phrase "below the forecast's cut"; a seven-channel grouping of the
   unmeasured-gain population, which amendment 11 reduced to four; and five places reading "sixteen
   channels" over tolerance on the 0.3 m worst case, which amendment 11 made seventeen.

### Addendum to amendment 14 (2026-09-19): the gate sensitivity of each excision, and two audit claims refuted on checking

Two further things were established while acting on the audit, one a disclosure the reader is owed
and one a claim that did not survive.

1. The gate sensitivity, measured. The detection gate is three floor scatters, fixed by amendment 8
   before any range was read. Re-running the ruling at gates of 3.00, 3.16, 3.30, 3.50 and 4.00
   scatters, changing nothing else, gives: at 3.00 the excised set is 15, 17, 30, 31, 33 and 35; at
   every value from 3.16 upward it is 17, 30, 31, 33 and 35. Five of the six excisions are therefore
   insensitive to the gate over a third of its own value, and channel 15 alone depends on it, being
   removed by a gate 5 per cent higher than the predeclared one. This is stated because channel 15 is
   the one excision a reader can remove with a defensible change of a single constant, and they should
   learn it here rather than find it. The gate is not moved: it was predeclared, and moving it after
   the result is known is the fault the whole amendment ledger exists to prevent.
2. Refuted on checking: the audit proposed that channel 24's coherence time could be measured inside
   the 2020 cohort, citing an epoch pair 4448 s apart. No such pair exists. Channel 24's sixteen
   epochs give a shortest pair of 8658 s, above the 7200 s plateau start, so the cohort has zero pairs
   below the plateau and can supply no crossing at all. The channel 24 row therefore keeps its
   dependence on the archive's own measured correlation time of 7777 s, as the addendum to amendment
   13 states, and the conditional stands as written.
3. Also refuted: that the 2020 cohort could supply a floor for the floor-limited channels. It cannot,
   for the same reason, and its non-detections are in any case at 8 frames per epoch against the 2026
   capture's 33.

## Amendment 15 (2026-09-19, on the author's instruction to seek a confident decision channel by channel even where the estimator refuses): the trim probes may show that a channel cannot be excised, never that it can

Nine channels are blocked because the coherence estimator returns nothing on the class carrying their
measured excess. But the estimator publishes three trim probes per class and refuses when they spread
by more than a factor of two. A refusal says the coherence time is imprecise. It does not by itself
say the VERDICT is undetermined: if the bar is inside the allowance even at the greatest coherence
time the probes admit, then no coherence time the estimator admits can excise the channel, and the
imprecision never reaches the decision.

1. The rule. Where a range has no measured and no bound class, and no borrow under amendment 9 item
   3, it is priced at the greatest of the measured and bound trim probes of its classes that are
   measured above the control floor. If the bar there is inside the 3 dB allowance, the range is not
   excisable, and the row says so with the probe it was tested at. If the bar there is outside the
   allowance, NOTHING is concluded and the range stays unpriced.
2. The asymmetry, which is the whole content of the rule. The greatest probe is the largest gain and
   therefore the least favourable reading for the channel, so surviving it is evidence. The converse
   is not: failing it would be an excision on a gain the estimator declined to measure, which is what
   amendment 12 item 8 refused when it declined to book a refused class at the sidereal cap. A probe
   that returns "constant" is the estimator returning nothing and is excluded from the set entirely.
3. Why this scoping and no wider. The rule was first written to admit probes into the range gain
   generally, and that was tested and rejected. Admitting them generally moves the bars of the banked
   excisions: channel 31 falls from 26.4 dB to between 14.9 and 7.0, channel 35 from 17.0 to between
   15.0 and 13.5, channel 33 from 26.5 to between 24.9 and 20.1. Booking the least rather than the
   greatest probe creates three keep cells and breaks the no-keep result. Booking the greatest probe
   into the verdict generally ADDS four excisions, on channels 14, 18, 22 and 26, several of them at
   the sidereal cap, which is the defect amendment 12 item 8 exists to prevent. Only the scoping
   above is safe, and it was arrived at by testing the wider forms and discarding them, not by
   choosing the one with the best outcome.
4. What it changes. Channel 27's short BAO range moves from unpriced to not excisable: at its
   greatest admitted probe, 368 s, the bar is 0.8 dB, inside the allowance, against a threshold of
   617 s. Nothing else moves. The six excised channels keep identical bars, no new keep cell appears,
   and no channel disposition changes. Channel 27's summary remains undetermined, because its long
   range is at the control floor, but the reason is now a measurement rather than a blank.
5. Recorded beside it, verified independently: channel 25's short range is already priced and reads
   inside the allowance at every one of its probes, 1.1 dB at 66 s, 1.2 dB at 68 s and 2.7 dB at
   95 s, against a threshold of 102 s. It was already marginal and stays so; the probe invariance is
   what makes that verdict robust rather than incidental.
6. The direction test. This amendment adds no excision and removes none. It converts one blind cell
   into a decided one, in the direction of not excisable, and every wider form of it that would have
   added excisions was tested and rejected.

## Amendment 16 (2026-09-19, closing an undeclared convention the second audit asked for on 2026-09-18 and which was not logged): the even-count aggregation of a range's coherence time

Amendment 7 says a range's coherence time is "the median of the measured class values". Where a range
has an EVEN number of contributing classes the median is not defined by that sentence, and the code
has always taken the geometric mean of the two middle values (ruling_baseline.py, gmed at line 113,
applied at line 159). The word "geometric" appears nowhere in this predeclaration. Audit R8 asked a
day ago for the convention to be logged as a numbered amendment and it was not. It is logged here,
with its sensitivity, because it is load-bearing on the published bars and a reviewer who finds an
undeclared convention will assume the worst about it.

The convention stands as the code has always applied it: the geometric mean of the two middle values.
It is the right choice for a quantity that enters the ruling logarithmically, a coherence time whose
bar is proportional to its decibels, and it is the choice that was in force before any range was read.

What it is worth, every other input held fixed:

| aggregation | 15 | 17 | 30 | 31 | 33 | 35 | excised |
|---|---|---|---|---|---|---|---|
| geometric mean, as ruled | 19.8 | 24.0 | 30.8 | 26.4 | 26.5 | 17.0 | all six |
| arithmetic mean | 19.8 | 28.6 | 31.4 | 26.4 | 26.7 | 17.0 | all six |
| lower of the two middle | 19.8 | 16.5 | 28.6 | 26.4 | 24.9 | 17.0 | all six |
| upper of the two middle | 19.8 | 31.4 | 33.0 | 26.4 | 28.0 | 17.0 | all six |

The convention is worth up to 14.9 dB of published bar on channel 17 and 4.4 dB on channel 30, and
nothing at all on channels 15, 31 and 35, whose ranges contribute an odd number of classes. It
changes no verdict: the excised set is identical under all four conventions, and under the most
conservative of them, the lower of the two middle values, the six bars are 19.8, 16.5, 28.6, 26.4,
24.9 and 17.0 dB, every one clear of the 3 dB allowance. That invariance is the point of logging it.
The manuscript quotes the ruling's own figures and states the conservative alternative beside them.

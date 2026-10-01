# Literal fine-detector validation

This campaign replaces the earlier preliminary finite-pool sensitivity study
with literal frames at the current production geometry. The experiment lives
at `../results/fine_detector_validation_2026-09-09/`; the generating program is
`tools/validate_fine_detector.py`. Its frozen plan, rather than this overview,
is the authoritative definition of a particular run.

The scientific run began on 2026-09-09 at 14:21 UTC. Its plan SHA256 is
`0ceac022b515568cf42fc4f3320a2acba8387bcfdc0753a60bbd7f3132ac44e7`.
The 14-stage local supervisor records its current state in
`campaign/execution/state.json`, with a separate log for each stage. It has
started the 4,000-draw null calibration. Calibration, independent validation,
discovery, evaluation, stress and spatial stages use distinct frozen noise
identities. The statistical reports run automatically; subsequent independent
release review and dissertation figure import remain explicit work.

The engineering checks and preparation are complete. Scientific sensitivity,
false-alarm and representation-loss conclusions require the independent
campaign outcomes. Passing a software audit does not fill the dissertation's
sensitivity figure or establish its 0.10 dB equivalence requirement.

## What is exercised

Each literal frame contains 2,048 streams, 128 windows per stream and 128
complex samples per window. The fine transform is padded to 256 bins. The
same independently generated noise realization is paired across the ideal
tone, GNU Radio ATSC waveform, input quantizer, weight quantizer, joint
quantizers, floating transform and exact fixed transform. The selected current
CUDA 2.3.0 binary produces the integer rows and powers; CPU replay checks those integers
exactly. Its binary hash is distinct from the historical campaign artifact;
the experiment does not silently certify another build. Device Q16 decision
checks are a separate explicit audit.

The floating detector uses complex128 projections and FFTs, with float64 power
sums. Its inputs are literal complex64 signal-plus-noise samples. This is not
a claim of a double-precision noise generator. The integer input quantizer
uses symmetric -7 through +7 components with ties-to-even rounding. Every
frame retains clipping counts and the full fine and coarse powers.

The study covers the 23 individually quantized weight profiles for physical
channels 14 through 36, including the circular band-edge geometry of channel
14. There are seven frequency conditions per channel: the nominal frequency,
exact padded-bin alignment, an exact half-padded-bin offset, and literal
offsets of -1,000, +1,000, -1,400 and +1,400 Hz. The padded-bin spacing is
11.920928955078125 Hz; the unpadded spacing is twice that. The analytic GNU
Radio pilot-frequency correction is recorded separately from the injected
offset.

Each condition has a static anchor declared from its known synthetic carrier
before detector outcomes. A separate replay retains the nominal anchor under
frequency error. These answer different questions: sensitivity with a known
calibrated anchor, and sensitivity when that anchor is stale. Neither tests
acquisition of an unknown transmitter frequency.

The cached PFB streams use normalized frequency coordinates. The scientific
detector input uses the CHIME inverted convention: conjugate the coarse time
stream and apply the receiver's reversal within each 128-sample window. The
slow-time fine coordinate is consequently reflected, and the anchors use
their declared inverted counterparts. Ideal floating powers obey the expected
reflection identity. Quantized powers need not match the normalized-coordinate
experiment, so those integer results are not interchanged. A separate native
adapter audit verifies CPU/GPU agreement directly on the adapted input.

Under the declared independent circular Gaussian model, adding noise after
this conjugation/permutation has the same distribution as adapting raw noise.
The symmetric -7 through +7 quantizer commutes with that operation. This does
not change the model into a measurement of the telescope's full native ADC
range or noise covariance.

The null bulk guards the anchor by one native bin and excludes the five
designated padded bins. It contains 125 or 126 even bins, depending on the
anchor parity. Applying this guard around each designated bin, as in the old
study, would remove additional bins and change the detector being validated.

## Waveforms and statistical population

Four newly seeded GNU Radio ATSC sources were generated with startup samples
discarded and long transport streams. Engineering qualification used eight
disjoint full-frame input windows per source. Three sources passed the
predeclared RMS, pilot-amplitude and reference-band variation limits. Source
01 failed the reference-band variation limit and remains a separately labelled
stress fixture. It was not replaced or quietly included in the qualified
family. These finite-window limits are engineering consistency checks, not a
statistical proof of stationarity.

Source 00 supplies discovery data. Evaluation trials draw a predeclared
independent 50:50 mixture of fixed sources 02 and 03. The source assignment and
noise seed remain paired across SNRs, offsets, profiles and arithmetic stages.
The population is this explicit two-fixture digital mixture. The number of
profiles, bins or policy evaluations is not an additional independent-trial
count, and two sources do not establish behaviour for arbitrary broadcasts.

Only window 0 of each source is used in this campaign. Each source passes
through the full reference real-ADC/PFB model at every profile and frequency
condition, producing 644 cached fixtures. The ideal tone retains its
continuous frequency and phase across the entire frame. Its amplitude comes
from projection of that finite clean stream, so finite-window data leakage
into the projection remains part of the declared tone comparison.

The source normalization, requested shelf SNR and injected truth are digital
model quantities. Noise added at the PFB output is independent across samples
and streams. Measured telescope noise correlations, feed gains, propagation,
physical pilot-to-shelf transfer and visibility preservation are outside this
experiment.

## Calibration and inference

Null calibration and independent null validation have disjoint frozen seeds.
The intended scientific counts are 4,000 calibration and 10,000 validation
draws per profile. Sharing each draw across profiles saves generation work
without increasing its statistical independence.

The primary nominal false-alarm probabilities are 0.01 and 0.05. A nominal
0.001 result is also retained as a tail diagnostic, with its actual uncertainty;
it is not silently given the precision of the primary results. Four rank
fractions are evaluated: one quarter, one half, three quarters and the maximum.
The calibrated statistic already maximizes over the five designated bins, so
the empirical threshold includes that search. No extra search correction is
applied to the same empirical probability.

The primary false-alarm gate reports an exact pointwise binomial interval and
a simultaneous upper bound for the frozen family of fixed Q16 policies and
coarse comparators. Its upper cap is twice the nominal probability. Passing
that engineering cap is not a claim that the true probability equals the
nominal calibration target. All failed, invalid or unsupported cases remain
visible.

Discovery uses 24 independent noise identities and only source 00. A
deterministic rule in the plan chooses the evaluation SNR grid from discovery
crossing brackets and freezes it before evaluation. Evaluation uses 512
independent noise/fixture identities per grid point. The source-01 stress
family uses 128 separate identities on that same frozen grid. An unresolved
evaluation crossing is not repaired by inspecting results and adding points.

Detection curves retain raw binomial counts and intervals. Crossing reports
show every upward and downward bracket, ambiguity and censoring. Linear
interpolation is a descriptive crossing model; grid discretization is a
separate uncertainty. Paired bootstrap comparisons recalibrate the null in
each resample and retain every censored replicate. A pointwise interval alone
does not establish equivalence on all profiles and offsets. The 0.10 dB
requirement remains an evaluated criterion, not an assumption or an adjustable
target.

## Scaling and interpretation

The scaling experiment uses powers of two from one through 2,048 streams on
all 23 aligned profiles. A fixed-bin ideal fine ratio has the central
F(2M,4M) null model. The designated maximum divided by a bulk order statistic
is a different statistic and is calibrated empirically. Coarse sums have
additional independent-window degrees of freedom under the declared model.

The ideal F(2,4) distribution at M=1 has infinite variance. A finite sample
standard deviation at that endpoint is descriptive; the theoretical standard
deviation must be omitted. Robust quantile widths remain usable. Replicating a
single captured stream supplies a correlation stress example, not M
independent measurements and not a universal covariance bound.

The familiar 5 log10(128), approximately 10.54 dB, fine-axis gain is an
aligned-tone benchmark. The designated search, bulk-rank calibration, carrier
offset, data shelf and quantization may change the observed crossing gain.

Completion requires checking planned coverage, independent source and seed
identities, raw-power integrity, arithmetic and Q16 parity, and the scientific
gates before importing figures or acceptance claims into the dissertation.
Earlier preliminary releases remain unchanged and labelled with their
original scope.

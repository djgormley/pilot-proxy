# The handoff: what the detector side gives the science side

This page is for an engineer who wants to know what this repository computes,
what it hands on, and where in the code each step is. The science side
([RFIsher](https://github.com/WVURAIL/RFIsher)) has the matching page,
`docs/HANDOFF.md` there, for what it does with these files.

## What crosses the boundary

Files only. Nothing in this repository imports the science code, and nothing on
the science side imports this one (`tests/core/test_no_rfisher_import.py` holds
the first half). Two handoffs are written, each a directory with a manifest
that records the sha256 of every file it lists:

| Handoff | Written by | Files | Schema |
|---|---|---|---|
| `operating_characteristic_v1` | `pilot-proxy characterize archive` | `oc_table.csv`, `oc_summary.csv`, `manifest.json` | `docs/schemas/operating_characteristic_v1.schema.json` (package copy `src/pilot_proxy/characterization/schemas/`) |
| `capture_operating_characteristic_v1` | `pilot-proxy capture oc-table` | `capture_oc_table.csv`, `capture_oc_summary.csv`, `capture_tau_bounds.csv`, `manifest.json` | `docs/schemas/capture_operating_characteristic_v1.schema.json` (package copy `src/pilot_proxy/capture/schemas/`) |

The schema files are pinned by `tests/characterization/test_oc_schema.py` and
`tests/capture/test_capture_oc_schema.py`, which also rebuild the golden
handoffs under `tests/golden/handoff/` from synthetic inputs and compare them
byte for byte. A producer whose files validate against a schema can feed the
science side; the science side vendors the schemas and refuses a file that
does not match.

What an operating characteristic is here: for each candidate threshold `eta`
of a band, the masked fraction `f(eta)` and the kept residual `r_sys(eta)`
(residual interference power over system noise power per frame, in the
reference the manifest declares), and `P_fa(eta)` where a verified
signal-free population exists. It is not P_d against SNR. The mask rule is
`statistic > eta`; equality keeps the frame. Floats are written as Python
`repr`, so a file reads back bit for bit.

## Reading path

Reading these modules in this order covers the whole method. Everything else
is a record of the 2026 campaign, an adapter for one instrument or emitter, or
infrastructure.

| Order | Module | Question it answers |
|---|---|---|
| 1 | `src/pilot_proxy/products/reader.py` | What is in one product? |
| 2 | `src/pilot_proxy/detectors/narrowband_marker/frames.py` | How do products become per-frame statistics and residual estimates? |
| 3 | `src/pilot_proxy/characterization/nulls.py` | What is the null, and why is it not verified here? |
| 4 | `src/pilot_proxy/characterization/false_alarm.py` | How would `eta_pfa` be set, and when is it available? |
| 5 | `src/pilot_proxy/characterization/residual_chain.py` | How is `r_sys` built (floor, coherence time, gain)? |
| 6 | `src/pilot_proxy/characterization/surface.py`, then `oc_table.py` | What does each threshold cost, and what does it leave behind? |

The capture path, for the matched capture of 2026-09:

| Order | Module | Question it answers |
|---|---|---|
| 1 | `src/pilot_proxy/capture/units.py` | What do the class amplitude, `N` and `n_b` mean, and how does an amplitude become a power ratio? |
| 2 | `src/pilot_proxy/capture/frame_residual.py` | What is the residual of one dump, per band and baseline class? |
| 3 | `src/pilot_proxy/capture/cadence.py` | How long does the residual stay coherent (the cadence campaign)? |
| 4 | `src/pilot_proxy/capture/oc_table.py` | What crosses into the capture handoff? |

The detector adapter's interface is `src/pilot_proxy/detectors/interface.py`;
the narrowband-marker adapter under `detectors/narrowband_marker/` is the one
this project uses.

## What the science side does with it

RFIsher turns a cosmological forecast into a tolerance `r_tol` per band and
scenario (`tolerances.csv`), the only science input of its verdict. The
verdict reads one handoff directory and that one file: it finds `eta_sci`, the
largest threshold whose kept residual is within the tolerance, forms the
deployable set of thresholds between the false-alarm limit (or the
evaluability bound) and `eta_sci`, selects inside it, and screens each band as
keep, excise or undetermined. The capture ruling of record is the science
side's too, with this repository's capture handoff as its detector input.

## Operating characteristic

This section describes the candidate families and the thresholds as the code
has them at this commit. Later changes to the rank rule or to how `eta` is
chosen are made here and in the code together.

**Two statistics, one `eta` column.** `statistic` is `Q` for the coarse
statistic (`F / mu_0`, the power ratio over its flat-noise null; `eta` is a
multiplier on it) or `Z_rho` for the fine order-statistic CFAR of rank `rho`
(`eta = eta_q16 / 2^16`, the 16-bit fixed-point multiplier the kernel
compares). Candidate sets are `fine_surface`, `coarse_surface` and
`coarse_ladder` (the `cal_q` retention rungs on `Q`).

**The rank family.** The fine surface enumerates every one-based rank
supported by all accepted frames of the band,
`detectors.narrowband_marker.scores.candidate_rho_values(minimum_valid_bulk_count)`,
which is `1 .. minimum_valid_bulk_count`; `characterization.stability`
asserts that the full grid was characterized. Each rank's candidate
thresholds are its exact deployable staircase
(`scores.candidate_eta_q16`: the distinct per-frame keep boundaries plus the
policy floor of one). The register entries `preparation.candidate_rho_grid`,
`preparation.rank_index_mapping` and `stability.surface`
(`projects/chime_atsc/detector_register.json`) describe these; no code reads
them. A candidate is written only when it is evaluable: at least 30 kept
frames, and both calendar halves of the era keep enough frames.

**The false-alarm limit.** `eta_pfa` is set only from a band's own
transmitter-off population that is independently verified and null-like,
inside the band's current era (`characterization.false_alarm.false_alarm_limit`).
At this commit it is the "higher" order statistic of `Q` at
`k = ceil((1 - alpha)(n - 1))` on that population, reported with a day-block
bootstrap design effect, the effective sample count and a one-sided 95 %
Clopper-Pearson upper bound on the achieved rate; it is `available` only when
the effective count reaches `minimum_frames_per_false_alarm / alpha`. Its
status says why it is missing when it is (`eta_pfa_status`; a fine row reads
`not computed: statistic Z_rho`), and `null_rejection_reason` says why a
band's population was not accepted as a null. The ideal-noise quantile and
the product's OS-CFAR design value are reported beside it as references and
never used as `eta_pfa`. `eta_eval` is the tightest evaluable threshold of a
family, the lower end of the deployable set when no `eta_pfa` is available.
The dissertation gives the wording these terms carry there.

**What this side does not choose.** It does not choose `eta`. The table holds
every evaluable candidate; the choice is the science side's.

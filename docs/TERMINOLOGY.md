# Terminology

This page gives the dissertation's symbols with the names the code uses for
them, the words that have one meaning only, and the frozen tokens: names that
files already written carry, kept as they are wherever they appear. The
science side's page (RFIsher `docs/terminology.md`) repeats the rows both
sides use, with the same code names.

## Symbols and code names

| Dissertation | Code |
|---|---|
| F (coarse power ratio) | `coarse_power_ratio` (product field) |
| μ0 | `null_power_ratio` (`pilot_proxy.detector_contract.null_power_ratio_from_weight_norms`) |
| Q = F/μ0 | `normalized_coarse_power_ratio`; OC column `statistic` = `Q` |
| e_p = Q − 1 | `normalized_pilot_excess` |
| Z_ρ, the fine OS-CFAR score | `pilot_proxy.detectors.narrowband_marker.scores.required_eta_q16_by_rank`; OC column `statistic` = `Z_rho` |
| ρ, η, η_q16 | `rho`, `eta`, `eta_q16` |
| M (marker bins), B (bulk) | `marker_bins`, `bulk` |
| f | `masked_fraction` |
| exposure cost 1/(1 − f) | `exposure_cost_uniform_loss` |
| C = (1 + r_var)/(1 − f) | `rfisher.verdict.interval.cost` |
| r_sys, r_sys at G = 1 | `r_sys`, `r_sys_incoherent` |
| G, τ_c, n_coh | `chain_gain` (with `chain_gain_status`), `tau_c_minutes`, `n_coh` |
| η_Pfa (the false-alarm limit) | `eta_pfa` |
| η_eval | `eta_eval` |
| r_tol | `r_tol` (`tolerance_basis` says which basis) |
| S | `suppression_db` |
| η_sci | `eta_sci` |
| 𝒟 (deployable set) | `interval_low`, `interval_high`, `interval_status`, `in_deployable_set` |
| R = r_sys/r_tol | `R` |
| "preparation" | `pilot_proxy.characterization.stability.prepare_family` |
| "selector" | `rfisher.verdict.interval.select` |
| "OC table" | `operating_characteristic_v1` |

## One meaning each

- **𝒟** is only the deployable set. The anchors' former "designated set" is
  the **marker bins M** (`marker_bins`).
- **"Floor"** means only the **residual floor**, the level below which a
  frame's residual is not resolved. η_Pfa is the **false-alarm limit**; the
  capture's control-band reading (band 37) is the **control-band level**.
- **Q** means only the statistic F/μ0. η_q16 is η in 16-bit fixed point;
  `cal_q0.5` names the rung that keeps the frames whose Q is at or below the
  calibration block's median Q.
- **γ** is the phasor coherence of the capture; the per-feed SNR of the
  dissertation's appendix C is ψ.
- **Scenario**, never "world": a delay-filter suppression S with its role in
  the verdict (`rule` or `sensitivity`). The file name `worlds.csv` and the
  column prefixes `none_*` and `deployed_*` are frozen tokens.
- **Credit** is not a term of the model. The model's term is the
  **post-processing sensitivity**, an assumed downstream suppression (the
  delay filter S, the ground filter, the coherence factor G → 1) that this
  work does not measure; it is applied only on the science side
  (RFIsher `verdict/sensitivity.py`). "Credit" survives in frozen tokens and
  in the text of the 2026 records (`GROUND_CREDIT`, `credit_readings/`, the
  readings of revisions 2 and 3).
- **Operating characteristic** means f and r_sys against η, not P_d against
  SNR.
- **`proxy-low`, `proxy-high`**: an era whose marker level is low (consistent
  with the transmitter off or weak) or high, the states of the author-dated
  era list.
- **Shelf**: the emitter's in-band power plateau, estimated per frame.
- **Anchor**: the fine-axis bin where the marker sits; the marker bins M are
  defined around it.
- **Transmitter-off (dated)** is not **verified signal-free**: a dated off
  epoch is inferred from the archive; only an independently verified,
  null-like population sets η_Pfa.
- **System noise power** in the code (the reference of every `r`) is the
  dissertation's **receiver noise power**.
- **`lb_priced`** (capture handoff, `capture_tau_bounds.csv`): the
  within-dump lower bound on the coherence time that the science side prices.
  It is a frozen v1 column name; this side prices nothing.
- **`eta`** in the capture handoff is a rung of the coarse ladder on Q (the
  archive threshold of a `cal_q` policy), not a deployable threshold.
  `eta_rung` and `board_G_archive` are candidate names for a v2 of that
  handoff.

## Frozen tokens

Names that frozen files carry; the lints exempt them by name, and only these:

- `pilotproxy_per_pilot_product_v5`, `pilotproxy_baonoise_health_view_v1`,
  `urn:baonoise:*`, the bank provenance key `baonoise`;
- `section-8.1-eras-v2`, the rung names `cal_q0.1`, `cal_q0.5`, `cal_q0.9`;
- `r_tol_dilation` and the wide tolerance columns, `worlds.csv` with its
  `none_*` and `deployed_*` columns, and the release basis name
  `coarse_world_sensitivity_2026-09-09`;
- `rfisher_results.archive.psd window spectra v1` (`spectra_window.json`),
  `rfisher-archive-ledger` and `WVURAIL/RFIsher` (the ledger),
  `rfisher-archive-report` (the report), `rfisher-dissertation-numbers`;
- `FStat_NumDenToPilotExcess*`, `LOCKED_RAW_PILOT_EXCESS_DEFINITION`;
- the frozen v1 dissertation export
  (`src/pilot_proxy/records/chime_atsc_2026/dissertation_export_v1.py`): its
  table names `bao_time_vs_masking`, `bao_convergence`, `bao_two_walls`,
  `bao_policy_case`, the owner strings `external-fisher-forecast` and
  `pilot-proxy+external-fisher-forecast`, and its producer name
  `pilot_proxy.dissertation_exports`;
- the capture handoff's `lb_priced` and `eta` columns (above).

## Detector measurement terms

PilotProxy stores a target-to-local-reference **power ratio**, not an exact
F-distributed statistic:

\[
R_\mathrm{coarse}
  = \frac{2P_\mathrm{target}}
         {P_\mathrm{ref,lower}+P_\mathrm{ref,upper}}.
\]

The exact integer squared norms of the packed target and reference weights set
the flat-noise null ratio

\[
R_\mathrm{null}
  = \frac{2\lVert w_\mathrm{target}\rVert^2}
         {\lVert w_\mathrm{ref,lower}\rVert^2
          +\lVert w_\mathrm{ref,upper}\rVert^2}.
\]

The normalized ratio and physical pilot excess are

\[
Q_\mathrm{coarse}=R_\mathrm{coarse}/R_\mathrm{null},
\qquad
e_p=Q_\mathrm{coarse}-1.
\]

`reject_mask` uses the exact integer form of `Q_coarse > 1`.
`normalized_coarse_power_ratio_db = 10 log10(Q_coarse)` is the plotted level,
so the null and the active positive-excess boundary are both exactly 0 dB.
The reported `pilot_excess_db` is `10 log10(e_p)` where `e_p > 0`, and
`estimated_data_shelf_snr_db` is derived from that same normalized excess.
The linear `coarse_power_ratio` remains available for exact reconstruction;
`raw_pilot_excess = R_coarse - 1` is diagnostic-only and is not a physical PNR.

The fine product stores

\[
R_\mathrm{fine}[b]
  = \frac{2S_\mathrm{target}[b]}
         {S_\mathrm{ref,lower}[b]+S_\mathrm{ref,upper}[b]}.
\]

Its null is calibrated empirically from the independent non-designated bins.
The name *fine power ratio* therefore states exactly what is measured without
claiming an exact parametric null distribution. Since schema v3 the ratio is
not stored: the scan writes the exact `fine_power_u64` terms and the ratio is
recomputed from them in post-processing. `fine_power_ratio` survives only as
the spelling of the corresponding key in the archived survey products.

Under independent complex-Gaussian projections with equal reference scales,
the corresponding idealized ratio can be written as an F variate.  The shipped
int4 weights do not satisfy the equal-scale condition exactly (the lower and
upper reference norms can differ), and telescope streams can be correlated or
non-Gaussian.  "F-statistic" is therefore retained only where it names the
existing CUDA/C ABI (`FStat_*`, `libfstatistic.so`) or discusses the idealized
model.  A later ABI-only series can rename implementation symbols without
mixing that mechanical change with scientific product semantics.

Frozen evidence and manuscript provenance are not rewritten by this hard cut.
They document earlier development snapshots and are not valid current product
inputs; regenerate current products and figures rather than adding readers for
those retired fields.

## Configuration and implementation identities

Current receiver profiles use stable descriptive IDs without development
suffixes: `reference_800mhz_pfb`, `chime_dtv_fengine`,
`chord_dtv_fengine`, and `chord_pathfinder_dtv_fengine`.  Every profile
binds to the single detector-core identity
`pilotproxy_cuda_local_reference_power_ratio`.  Stream maps and packed
weight headers record those same IDs, so provenance does not depend on the
order in which pre-release designs were tried.

The compiled C/CUDA ABI remains `FStat_*` / `libfstatistic.so` in this
series.  Those symbol and library names are an implementation boundary and
are changed only by the separate ABI series; no compatibility aliases are
introduced here.

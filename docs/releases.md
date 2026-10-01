# Detector-side releases and evidence bundles

Each frozen output the detector side reproduces, with the commit to replay it
from, the interpreter of record, and the row of the refactor's equivalence
table (DESIGN F.2) that reproduces it. A release directory under the workspace
`results/` tree is named, not its path. Replay a row from a worktree at its
commit (`git worktree add <dir> <commit>`) with its interpreter; the
commands of record are in each release's manifest and in the gate records of
the refactor.

The interpreters of record are: the release environment (Python 3.12.3,
numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.1), the archive environment
(`archive-local`: Python 3.12.3, numpy 2.5.2, with cupy for the kernel
paths), the dissertation environment (Python 3.12.3, numpy 2.3.5) and the
system Python (`/usr/bin/python3`, Python 3.12.3, matplotlib 3.10.8) for
row 11.

| Row | Release or bundle | What is reproduced | Replay commit | Interpreter of record |
|---|---|---|---|---|
| 1a, 1b, 19 | `archive_author_eras_2026-09-23` (archive layer) | era, anchor, containment, K*, flagger tables; nulls, held-out spectra, spectrum windows (`pilot-proxy characterize archive`) | 30e50c2 | release environment |
| 20 | `archive_no_split_2026-09-24_r2` (archive layer) | the same tables at the off setting | 30e50c2 | release environment |
| 4 | `histogram_author_eras_2026-09-23c` | histogram frames and analysis (`characterize histogram-frames`, `characterize histograms`) | 30e50c2 | release environment |
| 5 | the archive spectrogram vendored by the dissertation | `spectrogram.npz` and its era files (`characterize spectrogram`) | 30e50c2 | release environment |
| 24 | `ch37_control_491_20260924` (export) | the channel 37 control export and its comparison | 30e50c2 | release environment |
| 10 | the archive report vendored by the dissertation | accounting, census, crossbuild and detection tables and numbers (`characterize report`) | 30e50c2 | release environment |
| 11 | `docs/evidence/mask_residual_frontier_2026-09-07` | the frontier report and figures from the frozen frames (`tools/mask_residual_frontier.py report`, `figure`); the generate and audit stages ran on the GPU and are not rerun | 30e50c2 | system Python, matplotlib 3.10.8 |
| 12 | `docs/evidence/estimator_transfer_2026-08-25`, `docs/evidence/sdr_ota_transfer_2026-08-25` | the two estimator-transfer figures (`bench estimator-transfer`) | 30e50c2 | release environment |
| 14 | `fine_null_diagnostics_2026-09-09` | the null-calibration data through the release driver on `characterization/false_alarm.py` | 30e50c2 | release environment |
| 15 | `inband_lag_diagnostics_2026-09-09` | the lag diagnostics through the release driver on `capture/lag_moments.py` (6254a771) | 30e50c2 | release environment |
| 16 | `docs/evidence/youden_j_2026-09-07` | the Youden J table (`characterize roc`) and the table of `DESIGN_DECISIONS.md` | 30e50c2 | release environment |
| 17 | `noise_signal_references_2026-09-09` | the coarse power response and the coarse reference-law figures (`tools/plot_coarse_power_response.py`, `tools/plot_coarse_reference_laws.py`) | 30e50c2 | release environment, with the plotter's `/usr/bin/python3.12` identity check bypassed as the refactor's ruling R16 records |
| 18 | the archive acceptance report | `products check-archive` | 30e50c2 | release environment |
| 6 | the capture's detector tables vendored by the dissertation, and the frozen `CADENCE_REPORT.md` | frame residuals, cadence and lag coherence (`capture frame-residual`, `capture cadence`), the cadence report up to its table of record section (`capture cadence-report`; the science side appends the rest) | 30e50c2 | archive environment for the frame residuals, release environment for the cadence tables, dissertation environment for `marker-to-inband` |
| 9 | the frame policies of the 2026 capture | `records frame-policy` (`records/chime_atsc_2026/frame_policy.py`, c49ae37f) | 30e50c2 | archive environment |
| 22 | the Stokes I study inputs | the Stokes I cadence, class excess and summary inputs (the capture's `--products stokes_i` options) | 30e50c2 | release environment |
| 7, 21 | the capture handoff | the detector half of the table of record and of the capture ruling (`capture oc-table`) | 30e50c2 | release environment |

Rows 1c to 1f, 2, 3, 8 and 13 and the science halves of rows 7 and 21 are the
science side's (RFIsher `docs/releases.md`).

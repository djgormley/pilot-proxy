# Capture characterization

The 2026-09 matched capture recorded manual CHIME baseband dumps of the DTV band: four science dumps (the 0.5 s
marker-bin dump of 2026-09-16 and three 1.4 s dumps of 2026-09-17) and ten short cadence dumps 0 to 1200 s apart.
The capture characterization measures, on those dumps, what the detector side owns: the class excess of each band
and the control band's level, the frame residual, the cadence coherence time and its within-dump lower bounds, the
phasor coherence between dumps, the lag moments of the in-band voltages, and where each dump sits on the band's
archive ladder. It writes one handoff for the science side, `capture_operating_characteristic_v1`. No tolerance,
credit, gain model or disposition is computed here: the capture ruling (the table of record) is the science side's.
One command still prints science values: `capture cadence-report` copies the table of record's dispositions and R
values from the table it is given into its last section, as the frozen report did. It computes none of them, and
that section is to move to RFIsher.

Reading CHIME baseband data requires CHIME/FRB authorization. The dumps, their reduced products and the detector
runs are collaboration data and are not published here.

## The pieces

| Code | What it measures | Output (frozen names) |
|---|---|---|
| `pilot_proxy.instruments.chime.capture.reduce_dump` | One converted baseband file per frequency, per 16384-sample frame: the 2048-input correlation stacked into redundant baseline classes, per-input powers, the fine spectrum per 256-input block, per-input marker cutouts. A standalone script (numpy, scipy, h5py), copied to the analysis host. | `<freq_id>.npz` |
| `capture frame-residual` | Per band, class, polarisation and frame set (all, the detector's kept and rejected frames, the archive's ladder): the noise-bias-free coherent amplitude less the sky reference, in-band median and maximum, the marker bin's value and structure function; the per-input retained power against the control band (`U_i`). No tolerance is read: the fraction of inputs above one is the science side's, from the per-input sidecar. | `frame_residual_<event>.csv`, `frame_residual_<event>_inputs.csv` |
| `capture cadence tau` | The cadence coherence time of a class's level across the dumps (predeclaration, amendments 3 and 5), with the archive's trim probes. `--level polavg`, `stokesI` or `polmax`. | `cadence_tau[_<class>][_pol<p>].csv`, `..._structure.csv` |
| `capture cadence lags` | The phasor coherence between dumps per class and lag class, raw and less the reference bands' mean phasor. `--classes three` or `ten`. | `lag_coherence_14.csv`, `lag_coherence_tenclasses_D.csv` |
| `capture cadence-report` | Renders the cadence tables and the table of record's dispositions (computes nothing on the table). | `CADENCE_REPORT.md` |
| `capture control-level` | Per band and class, A over the science dumps and the control band's level (mean and scatter of its per-bin excess). | `baseline_floor.csv` |
| `capture class-excess` | Per epoch, class and product (`xx,yy` or `stokes_i`), each band's in-band excess, and the control band's per-bin excess. | `class_excess_epochs.csv`, `class_floor_bins.csv` |
| `capture marker-map`, `marker_to_inband` | The marker-bin to in-band correction on the long baselines. `marker_to_inband` imports numpy and the standard library only (it ran under an interpreter without a YAML reader); the map is written first under a full interpreter. | `pilot_to_inband.csv` |
| `capture ladder-place` | Each dump's median Q and the fraction of its frames each rung of the band's archive ladder keeps (population `all_valid_frames`). | stdout |
| `capture tau-bounds` | The within-dump 95 % lower bounds on tau (amendment 19 item 35) from the autocorrelation lane and its check, at the settings of releases r5 and r5.1/r5.2. The lane's own files are inputs. The lane's grouping of classes into ranges is a science rule and is not carried. | `capture_tau_bounds.csv` |
| `capture oc-table` | The handoff: `capture_oc_table.csv`, `capture_oc_summary.csv`, `capture_tau_bounds.csv`, `manifest.json` (schema `docs/schemas/capture_operating_characteristic_v1.schema.json`). | `<out>/capture/` |
| `capture band-shape`, `first-look`, `line-check`, `bearing` | Diagnostics of one dump. | `band_shape.csv`, `first_look.csv`, stdout |
| `records frame-policy` | The frozen coarse frame policies replayed on timestamp-matched capture correlations (`records/chime_atsc_2026/frame_policy.py`, byte-frozen; see below). | `frame_policies/<name>/` |
| `pilot_proxy.capture.lag_moments` | Support-matched lag moments of contiguous series (the in-band lag diagnostics). | library |

The capture record of the campaign (its dumps, their labels, the detector runs made on them, the reference bands
of the lag coherence) is `pilot_proxy.records.chime_atsc_2026.capture_campaign`.

Names fixed on disk stay as they were written: `pilot_*` and `*_pilot*` columns, `is_pilot`, `bins=pilot`, the
`pilot` label of the first science dump and `role=pilot` in the table of record all name the marker bin. Channel
37 has no licensed emitter; it is the control band, read at freq_id 491, which the capture prints as "control bin
491 (nominal ATSC pilot position)". It is not a pilot.

## Commands of record

`E` names the reduced products root (`pilot_reduce_<event>/`), `FRZ` the frozen reduction tree of the capture
(`output/channel-ruling-execution-2026-09-14/rebuild/author_actions/capture-runbook/reduce`, with the detector runs
`kernel_<event>_k230` and `kernel_<event>_k230_ch33measured` and `frame_analysis/epochs_14.txt`), `DEV` the
decision evidence of 2026-09-24.

```
pilot-proxy capture frame-residual --products $E/pilot_reduce_<event> --detector-run $FRZ/kernel_<event>_k230 --out frame_residual_<event>.csv
pilot-proxy capture cadence tau --level polmax --pol-table $FRZ/frame_analysis/table_of_record_provisional_3dumps_pol.csv --trim archive cadence_tau.csv <label>=<dir> ...
pilot-proxy capture cadence tau --level polmax --trim archive --class 0,32 [--pol 0|1] cadence_tau_ns32[_pol0|_pol1].csv <label>=<dir> ...
pilot-proxy capture cadence tau --level stokesI --trim archive --class 0,32 cadence_tau_ns32.csv <label>=<dir> ...
pilot-proxy capture cadence tau cadence_tau_a3.csv <label>=<dir> ...
pilot-proxy capture cadence lags --classes three --reference-bands 34,37 lag_coherence_14.csv <label>=<dir> ...
pilot-proxy capture cadence lags --classes ten --reference-bands 34,37 lag_coherence_tenclasses_D.csv pilot=<dir> D1=<dir> D2=<dir> D3=<dir>
pilot-proxy capture cadence-report CADENCE_REPORT.md cadence_tau.csv lag_coherence_14.csv table_of_record_v10.csv cadence_tau_a3.csv
pilot-proxy capture control-level --datasets $E --out-dir DIR
pilot-proxy capture class-excess --epochs $FRZ/frame_analysis/epochs_14.txt --datasets $E --products xx,yy --out-dir DIR
pilot-proxy capture class-excess --epochs $FRZ/frame_analysis/epochs_14.txt --datasets $E --products stokes_i --out-dir DIR
pilot-proxy capture marker-map --out map.json
python -m pilot_proxy.detectors.narrowband_marker.marker_to_inband --marker-map map.json --datasets $E --out pilot_to_inband.csv
pilot-proxy capture ladder-place --archive-products <archive per-band products> $FRZ/kernel_<event>_k230 ...
pilot-proxy capture tau-bounds --lane $DEV/autocorrelation --check $DEV/check_ac --b3-null $DEV/b_list/B-3/null_bounds.csv \
    --b4-bounds $DEV/b_list/B-4/lane_ch15_ew3/bounds_b4.json --out capture_tau_bounds.csv
pilot-proxy capture oc-table --capture-dir <xx,yy capture files> --stokes-dir <Stokes I capture files> \
    --lags lag_coherence_tenclasses_D.csv --detector-runs $FRZ --datasets $E --archive-products <archive per-band products> \
    --measured-marker-rescan <channel 33 rescan products> --tau-bounds capture_tau_bounds.csv \
    --board <channel ruling board_final.csv> --out DIR
pilot-proxy records frame-policy --root <workspace> --output <new directory>
```

The cadence runs of record are the 0.3 m class and the seven classes 0,32, 0,64, 0,128, 0,255, 1,0, 2,0 and 3,0,
each on the per-band level, on pol 0 and on pol 1 (the `xx,yy` capture files), and on Stokes I (the Stokes I
capture files), over the fourteen dumps of `epochs_14.txt`. The operations scripts that fired, reduced and
processed the dumps are in `scripts/capture/` (see `scripts/capture/dumps/README.md`).

## The capture handoff

`capture_oc_table.csv` has one row per band x class x product x bank x policy x event: each dump's class excess, its
frame count, its median Q, whether the policy keeps it and the admissible minimum coherence gain at its frame
count; and a `kept` row per policy with the median excess over the kept dumps, the control band's level, the net
amplitude, the amplitude test (`A > mean + k sd`, `k = capture.detection_gate_sigma` of the detector register,
3.0) and the power form `A_net^2 N n_b`. The cadence coherence time of the product's level file and the phasor
coherence between the science dumps repeat on every row of a band x class x product. `capture_oc_summary.csv` has
one row per band: its bins, the archive ladder per bank, the frozen archive gain bound of the channel ruling
(`board_final.csv`, 2026-09-14) and the frame-residual readings of the table of record. The manifest records the
schema, every input's sha256, the integration model and the null policy: no band has an eta_Pfa on the capture
(eta_Pfa is read only on a band's own verified, signal-free population, and the capture holds none; the control
band's is not defined).

## Frame policies (the record)

`records/chime_atsc_2026/frame_policy.py` applies the frozen current-era calibration thresholds from the September
9 coarse release to individual capture frames. It uses exact integer comparisons and joins the detector and
correlation products by FPGA sample identity within each event, with the source event and UTC/FPGA origins checked
before selection. Missing marker data, incompatible banks, missing timing metadata and unmatched frames refuse the
affected comparison. It is byte-frozen (sha256 `c49ae37f`): its receipt records its own sha256, and the release's
`analysis_source.py` is a copy of it. The four evaluated policies are the three saved calibration quantiles and
keep-all; the quantiles are not refitted to the captures. The nominal channel 33 bank remains a separate
diagnostic. The output directory must not already exist; `--root` is the workspace root that holds `output/`,
`results/`, `products/` and `datasets/`.

The sample covariance describes fluctuations among the retained frames in that capture, not the covariance of a
survey mean; the signed cross-frame product and lag-one covariance are diagnostics. `count` in the source
reduction is redundant-product multiplicity. The captures are short, separated records: summed retained duration
is not a contiguous integration.

## Records kept elsewhere

- The method and its amendments: `FRAME_ANALYSIS_PREDECLARATION.md`, `amendments.md` (in plain English), and the
  two ruling audits of 2026-09-18.
- The science scripts of the capture ruling (`ruling_baseline.py`, `table_of_record.py` and the credit, robustness
  and baseline-domain scripts) stay in `tools/capture/frame_analysis/` until their port to the science side lands.
- The frozen outputs of the campaign are in the frozen reduction tree and its `frame_analysis/` directory; the SDR
  bench of 2026-09-18 is `docs/evidence/sdr_bench_2026-09-18/`.

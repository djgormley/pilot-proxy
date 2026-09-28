#!/usr/bin/env python3
"""Full-depth three-arm mask evaluation with the REAL common-mode power veto.

Arms (trusted channels):
  1. ceiling    : mask F > mu_hat
  2. band floor : mask F < mu_hat - 12e-3*null_power_ratio
  3. power veto : mask P/base_unit > 1 + 5*sigma_P   (high side only)
Untrusted channels (ch24, ch30): one-sided manifest ceiling + power veto,
matching the adopted operating point.

base_unit = median baseband power over the unit's F-core frames (emulates a
slow tracker whose state is set by the quiet majority of time); units without
>=2 core frames inherit the channel median baseline.
sigma_P = 1.4826*MAD of P/base over core frames (per channel).

Outputs: deliverables/threearm_fulldepth.csv, fig_threearm_veto.png/pdf,
printed ladder (analytic ceiling -> measured ceiling -> +floor -> +veto).
"""
import csv

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
PCT = r"\%" if plt.rcParams["text.usetex"] else "%"
OUT = _paths.OUT
COARSE_MHZ = 0.390625
PW = np.load(_paths.POWER)
PF = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}
chans = sorted(study)


def boot_p999(vals, units, n_boot=300, q=0.999, seed=5):
    """68% per-event block-bootstrap interval on the p99.9 of vals.

    Frames cluster within capture units; resample whole units via
    multiplicity weights on the sorted values (weighted percentile)."""
    if vals.size < 100:
        return (float("nan"), float("nan"))
    order = np.argsort(vals)
    v = vals[order]
    uu, uinv = np.unique(units[order], return_inverse=True)
    U = uu.size
    if U < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, U, U), minlength=U)
        cw = np.cumsum(m[uinv])
        if cw[-1] == 0:
            out[b] = np.nan
            continue
        out[b] = v[min(np.searchsorted(cw, q * cw[-1]), v.size - 1)]
    lo, hi = np.nanpercentile(out, [16, 84])
    return (float(lo), float(hi))

rows = []
ccdf_data = {}
for ch in chans:
    s = study[ch]
    null_power_ratio = float(s["null_power_ratio_analytic"])
    mu_hat = float(s["null_power_ratio_empirical"])
    trusted = s["zero_point_trusted"] == "1"
    pt = PF[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = PF[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = PF[f"ch{ch}_valid"].astype(bool)
    fui = PF[f"ch{ch}_frame_unit_index"].astype(int)
    P = PW[f"ch{ch}_baseband_power_linear"]
    n_units = PF[f"ch{ch}_unit_time0_ctime"].size
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    ok = valid & np.isfinite(f) & np.isfinite(P) & (P > 0)

    # F-arm keep regions
    keep_c_an = ok & (f <= null_power_ratio)                       # analytic ceiling
    if trusted:
        keep_c = ok & (f <= mu_hat)                   # measured ceiling
        keep_b = keep_c & (f >= mu_hat - 12e-3 * null_power_ratio)  # + band floor
        core = ok & (np.abs(f - mu_hat) <= 6e-3 * null_power_ratio)
    else:
        keep_c = keep_c_an                            # manifest one-sided
        keep_b = keep_c
        core = ok & (np.abs(f - null_power_ratio) <= 6e-3 * null_power_ratio)

    # per-unit power baseline from core frames (slow-tracker emulation)
    base = np.full(n_units, np.nan)
    n_core_u = np.bincount(fui[core], minlength=n_units)
    for u in np.nonzero(n_core_u >= 2)[0]:
        base[u] = np.median(P[core & (fui == u)])
    fallback = np.nanmedian(base)
    n_fallback = int((~np.isfinite(base)).sum())
    base = np.where(np.isfinite(base), base, fallback)
    Pn = P / base[fui]

    # veto threshold from core-frame scatter
    pn_core = Pn[core]
    sig = 1.4826 * np.median(np.abs(pn_core - np.median(pn_core)))
    thr = 1.0 + 5.0 * sig
    veto = Pn > thr
    keep_f = keep_b & ~veto

    nv = int(ok.sum())
    kc_an = keep_c_an.sum() / nv
    kc = keep_c.sum() / nv
    kb = keep_b.sum() / nv
    kf = keep_f.sum() / nv
    inc = 1.0 - kf / kb if kb > 0 else 0.0
    with np.errstate(divide="ignore"):
        pn_db = 10 * np.log10(np.maximum(Pn, 1e-12))
    q_band = (np.percentile(pn_db[keep_b], [50, 99.9])
              if keep_b.sum() > 100 else (np.nan, np.nan))
    q_kept = (np.percentile(pn_db[keep_f], [50, 99.9])
              if keep_f.sum() > 100 else (np.nan, np.nan))
    b_lo, b_hi = boot_p999(pn_db[keep_b], fui[keep_b])
    k_lo, k_hi = boot_p999(pn_db[keep_f], fui[keep_f])
    rows.append(dict(
        ch=ch, fid=int(float(s["freq_id"])), n=nv, trusted=int(trusted),
        kc_an=kc_an, kc=kc, kb=kb, kf=kf, inc=inc, sig_pct=100 * sig,
        thr_db=10 * np.log10(thr), n_fallback_units=n_fallback,
        band_p50_db=q_band[0], band_p999_db=q_band[1],
        kept_p50_db=q_kept[0], kept_p999_db=q_kept[1],
        band_p999_lo=b_lo, band_p999_hi=b_hi,
        kept_p999_lo=k_lo, kept_p999_hi=k_hi))
    ccdf_data[ch] = dict(pn_db=pn_db[keep_b], thr_db=10 * np.log10(thr))

L1 = sum(r["kc_an"] for r in rows) * COARSE_MHZ
L2 = sum(r["kc"] for r in rows) * COARSE_MHZ
L3 = sum(r["kb"] for r in rows) * COARSE_MHZ
L4 = sum(r["kf"] for r in rows) * COARSE_MHZ
tot = len(rows) * COARSE_MHZ

with open(OUT / "threearm_fulldepth.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["atsc_channel", "freq_id", "n_valid", "zero_point_trusted",
                "kept_ceiling_analytic", "kept_ceiling_measured",
                "kept_band", "kept_final", "veto_incremental_frac",
                "veto_sigma_pct", "veto_thr_db", "units_fallback_baseline",
                "band_p50_db", "band_p999_db", "kept_p50_db",
                "kept_p999_db", "band_p999_lo68", "band_p999_hi68",
                "kept_p999_lo68", "kept_p999_hi68"])
    for r in rows:
        w.writerow([r["ch"], r["fid"], r["n"], r["trusted"],
                    f"{r['kc_an']:.4f}", f"{r['kc']:.4f}", f"{r['kb']:.4f}",
                    f"{r['kf']:.4f}", f"{r['inc']:.4f}",
                    f"{r['sig_pct']:.2f}", f"{r['thr_db']:.3f}",
                    r["n_fallback_units"],
                    f"{r['band_p50_db']:.3f}", f"{r['band_p999_db']:.3f}",
                    f"{r['kept_p50_db']:.3f}", f"{r['kept_p999_db']:.3f}",
                    f"{r['band_p999_lo']:.3f}", f"{r['band_p999_hi']:.3f}",
                    f"{r['kept_p999_lo']:.3f}", f"{r['kept_p999_hi']:.3f}"])

hdr = (f"{'ch':>3} {'tr':>2} {'keep_an':>8} {'keep_hat':>8} {'band':>6} "
       f"{'final':>6} {'veto%':>6} {'sig%':>5} {'thr_dB':>7} "
       f"{'band_p99.9':>10} {'kept_p99.9':>10}")
print(hdr)
for r in rows:
    trusted_raw = r.get("trusted")
    trusted_disp = "Y" if trusted_raw is True else ("N" if trusted_raw is False else "?")
    print(f"{r['ch']:>3} {trusted_disp:>2} {r['kc_an']:>8.3f} "
          f"{r['kc']:>8.3f} {r['kb']:>6.3f} {r['kf']:>6.3f} "
          f"{100*r['inc']:>6.2f} {r['sig_pct']:>5.2f} {r['thr_db']:>7.3f} "
          f"{r['band_p999_db']:>10.2f} {r['kept_p999_db']:>10.2f}")
print(f"\nladder (MHz of {tot:.3f}):")
print(f"  analytic ceiling            : {L1:.3f}")
print(f"  measured ceiling            : {L2:.3f}")
print(f"  + band floor                : {L3:.3f}")
print(f"  + common-mode power veto    : {L4:.3f}")

# ---------------- figure -------------------------------------------------------
# Panel (a) channels span the observed classes: quiet cores (15, 20, 26),
# heavy episodic tails (31, 32), and one calibration-refused channel (24).
SHOWA = {15: "#0072B2", 31: "#7B4FA6", 32: "#D55E00", 24: "#A85C85",
         26: "#009E73", 20: "#E69F00"}
fig = plt.figure(figsize=(11.4, 4.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35],
                      height_ratios=[2.6, 1.0], hspace=0.12, wspace=0.20)
ax1 = fig.add_subplot(gs[:, 0])
for ch, c in SHOWA.items():
    d = ccdf_data[ch]["pn_db"]
    if d.size < 100:
        continue
    xs = np.sort(d)
    ccdf = 1.0 - np.arange(xs.size) / xs.size
    ax1.semilogy(xs, ccdf, color=c, lw=1.3, label=f"ch{ch}")
    t = ccdf_data[ch]["thr_db"]
    yt = max((d > t).mean(), 1.2e-5)
    ax1.plot([t], [yt], "v", color=c, ms=6, zorder=5)
ax1.set_xlabel("band-kept frame power vs unit baseline [dB]")
ax1.set_ylabel("CCDF (fraction of band-kept frames beyond)")
ax1.set_title("(a) power of frames the F band would keep\n"
              r"($\blacktriangledown$ = per-channel veto threshold)",
              fontsize=10)
ax1.set_ylim(1e-5, 1.1)
ax1.set_xlim(-3, 15.5)   # ch24's -15 dB in-band fade tail runs off-axis
ax1.legend(fontsize=7.5, ncol=2)
ax1.grid(color="0.92", lw=0.6, which="both")
ax1.set_axisbelow(True)

ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
x = np.arange(len(rows))
b999 = np.array([r["band_p999_db"] for r in rows])
k999 = np.array([r["kept_p999_db"] for r in rows])
b_err = np.array([[r["band_p999_db"] - r["band_p999_lo"] for r in rows],
                  [r["band_p999_hi"] - r["band_p999_db"] for r in rows]])
k_err = np.array([[r["kept_p999_db"] - r["kept_p999_lo"] for r in rows],
                  [r["kept_p999_hi"] - r["kept_p999_db"] for r in rows]])
b_err = np.nan_to_num(np.clip(b_err, 0, None))
k_err = np.nan_to_num(np.clip(k_err, 0, None))
inc = np.array([100 * r["inc"] for r in rows])
for xi, b, k in zip(x, b999, k999):
    ax2.plot([xi, xi], [k, b], "-", color="0.75", lw=1.4, zorder=1)
ax2.errorbar(x, b999, yerr=b_err, fmt="o", ms=5, color="#D55E00",
             capsize=1.5, elinewidth=0.8, lw=0,
             label="before veto (band only)")
ax2.errorbar(x, k999, yerr=k_err, fmt="o", ms=5, color="#0072B2",
             capsize=1.5, elinewidth=0.8, lw=0, label="after veto (kept)")
ax2.set_ylabel("p99.9 kept-frame power\nvs baseline [dB]")
ax2.set_title("(b) veto clips the residual power tail of the kept data\n"
              f"(68{PCT} per-event block-bootstrap intervals)", fontsize=10)
ax2.tick_params(labelbottom=False)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(color="0.92", lw=0.6)
ax2.set_axisbelow(True)
ax3.bar(x, inc, width=0.55, color="0.8")
ax3.set_xticks(x)
ax3.set_xticklabels([str(r["ch"]) for r in rows], fontsize=7)
ax3.set_xlabel("ATSC channel")
ax3.set_ylabel(f"vetoed [{PCT}]")
ax3.set_ylim(0, max(8.0, 1.15 * np.nanmax(inc)))
ax3.grid(color="0.92", lw=0.6, axis="y")
ax3.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_threearm_veto.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_threearm_veto.pdf", bbox_inches="tight")
print("\nwrote threearm_fulldepth.csv + fig_threearm_veto")

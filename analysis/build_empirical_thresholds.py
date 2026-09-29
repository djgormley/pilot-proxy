#!/usr/bin/env python3
"""Exact-integer threshold constants for the measured zero points.

Kernel form (unchanged): the shipped ceiling rule is
    mask_hi :  p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum
i.e. pt*Q > P*pr with P/Q = null_power_ratio/2. Deploying the measured zero point is a
CONSTANTS SWAP: replace (P,Q) = (target_norm_sq, reference_norm_sum_sq) with a
rational approximation of mu_hat/2. The band floor is the same compare with
its own (P_lo, Q_lo) and the opposite sense:
    mask_lo :  p_target * Q_lo < P_lo * p_ref_sum
Untrusted channels (ch24, ch30) keep the exact analytic constants and no
floor (one-sided manifest ceiling, matching the adopted operating point).

Verification: the integer masks are compared frame-for-frame against the
float rules (F > mu_hat, F < mu_hat - 12e-3*null_power_ratio) on all perframe data.
"""
import csv
from fractions import Fraction

import numpy as np

import _paths
OUT = _paths.OUT
DEN_LIMIT = 1 << 16          # keeps pt*Q, P*pr well inside u64 (see headroom)

PF = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}
chans = sorted(study)

rows = []
worst_bits = 0
for ch in chans:
    s = study[ch]
    null_power_ratio = float(s["null_power_ratio_analytic"])
    mu_hat = float(s["null_power_ratio_empirical"])
    trusted = s["zero_point_trusted"] == "1"
    scal = PF[f"ch{ch}_scalars"]          # [null_power_ratio, tns, rnss, fid, ...]
    tns, rnss = int(scal[1]), int(scal[2])
    pt = PF[f"ch{ch}_p_target_u64"]
    pr = PF[f"ch{ch}_p_ref_sum_u64"]
    valid = PF[f"ch{ch}_valid"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt.astype(np.float64) / pr.astype(np.float64)
    ok = valid & np.isfinite(f)

    if trusted:
        hf = Fraction(mu_hat / 2.0).limit_denominator(DEN_LIMIT)
        P, Q = hf.numerator, hf.denominator
        tau_lo = mu_hat - 12e-3 * null_power_ratio
        lf = Fraction(tau_lo / 2.0).limit_denominator(DEN_LIMIT)
        P_lo, Q_lo = lf.numerator, lf.denominator
    else:
        P, Q = tns, rnss                   # exact analytic, one-sided
        P_lo = Q_lo = 0

    # integer masks (object ints -> no overflow anywhere in the check itself)
    ptQ = pt.astype(object) * Q
    Ppr = pr.astype(object) * P
    m_hi_int = np.array([a > b for a, b in zip(ptQ, Ppr)]) & ok
    mu_cmp = mu_hat if trusted else null_power_ratio
    m_hi_flt = ok & (f > mu_cmp)
    mm_hi = int((m_hi_int != m_hi_flt).sum())

    mm_lo = 0
    if trusted:
        ptQl = pt.astype(object) * Q_lo
        Plpr = pr.astype(object) * P_lo
        m_lo_int = np.array([a < b for a, b in zip(ptQl, Plpr)]) & ok
        m_lo_flt = ok & (f < tau_lo)
        mm_lo = int((m_lo_int != m_lo_flt).sum())

    bits = max(int(pt.max()) * max(Q, Q_lo) if trusted else int(pt.max()) * Q,
               int(pr.max()) * max(P, P_lo)).bit_length()
    worst_bits = max(worst_bits, bits)
    err_hi = abs(2 * P / Q - (mu_hat if trusted else null_power_ratio))
    err_lo = abs(2 * P_lo / Q_lo - tau_lo) if trusted else 0.0
    rows.append(dict(ch=ch, fid=int(float(s["freq_id"])), trusted=int(trusted),
                     P=P, Q=Q, P_lo=P_lo, Q_lo=Q_lo, err_hi=err_hi,
                     err_lo=err_lo, mm_hi=mm_hi, mm_lo=mm_lo, bits=bits,
                     n=int(ok.sum())))

with open(OUT / "empirical_thresholds.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["atsc_channel", "freq_id", "zero_point_trusted",
                "ceil_num_P", "ceil_den_Q", "floor_num_P", "floor_den_Q",
                "ceil_abs_err", "floor_abs_err",
                "verify_mismatch_ceiling_frames",
                "verify_mismatch_floor_frames", "max_product_bits",
                "n_frames_verified"])
    for r in rows:
        w.writerow([r["ch"], r["fid"], r["trusted"], r["P"], r["Q"],
                    r["P_lo"], r["Q_lo"], f"{r['err_hi']:.3e}",
                    f"{r['err_lo']:.3e}", r["mm_hi"], r["mm_lo"], r["bits"],
                    r["n"]])

tot_mm = sum(r["mm_hi"] + r["mm_lo"] for r in rows)
tot_n = sum(r["n"] for r in rows)
print("\nrules: mask_hi = pt*Q > P*pr ; mask_lo = pt*Q_lo < P_lo*pr "
      "(trusted channels only)")
print(f"verification: {tot_mm} mask mismatches across {tot_n} frames")
print(f"worst product width: {worst_bits} bits (u64 headroom "
      f"{64 - worst_bits} bits)")
print("wrote empirical_thresholds.csv")

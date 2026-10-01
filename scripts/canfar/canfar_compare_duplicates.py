#!/usr/bin/env python3
"""Field-level comparison of the channels two shards both built, NaN-aware.

    python scripts/canfar/canfar_compare_duplicates.py <local root>

where <local root> holds shard1/, shard2/, shard3/ each with _per_pilot/.

Byte identity (sha256) is the headline for every pair. The field pass is the
confirmation: NaN in the same positions counts as equal, since NaN != NaN
would otherwise report an identical file as differing."""
import hashlib, sys
import numpy as np
if len(sys.argv) < 2:
    sys.exit("usage: canfar_compare_duplicates.py <local root>")
D = sys.argv[1]
PAIRS = [(767, 1, "merged"), (783, 2, "merged"),
         (813, 1, "independent"), (829, 1, "independent"),
         (798, 2, "independent"), (844, 2, "independent")]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
def eq(x, y):
    if x.shape != y.shape or x.dtype != y.dtype: return False
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        return np.array_equal(x, y, equal_nan=True)
    return np.array_equal(x, y)
worst = 0
for ch, owner, kind in PAIRS:
    a_p = f"{D}/shard{owner}/_per_pilot/{ch}.npz"; b_p = f"{D}/shard3/_per_pilot/{ch}.npz"
    sa, sb = sha(a_p), sha(b_p)
    tag = "BYTE-IDENTICAL" if sa == sb else "BYTES DIFFER"
    line = f"{ch} shard{owner} vs shard3 [{kind}]: {tag}  sha256 {sa[:16]}"
    if kind == "merged":
        print(line); worst = max(worst, 0 if sa == sb else 2); continue
    with np.load(a_p, allow_pickle=False) as A, np.load(b_p, allow_pickle=False) as Bz:
        if set(A.files) != set(Bz.files):
            print(line + "  KEY SETS DIFFER"); worst = 2; continue
        n = len(A["unit_order"]); nan_fields = 0; differ = []
        for k in sorted(A.files):
            x, y = A[k], Bz[k]
            if np.issubdtype(x.dtype, np.floating) and np.isnan(x).any(): nan_fields += 1
            if not eq(x, y): differ.append(k)
        print(f"{line}; {n} units, {len(A.files)} fields all equal (NaN-aware), "
              f"{nan_fields} fields carry NaN" if not differ else
              f"{line}; {len(differ)} fields differ: {differ}")
        worst = max(worst, 1 if differ else 0)
print("RESULT:", ["every pair identical", "some fields differ", "STRUCTURAL DIFFERENCE"][worst])
sys.exit(worst)

#!/usr/bin/env python3
"""Reduce one CHIME baseband .h5 file (one frequency) of a manual dump to frame products.

Per 16384-sample frame (0.04194304 s, the archive frame) this writes
  stacks    [n_frames, n_stack] complex64   full 2048-input correlation, averaged into redundant
                                            baseline classes (EW cylinder step, NS feed step, pol pair)
  autos     [n_frames, 2048]    float32     per-input power, chan_id order
  block_psd [n_frames, 8, 16384] float32    16384-point fine spectrum summed over each 256-input block
  pilot_cut [n_frames, 2048, 129] complex64 fine spectrum of every input around the pilot line (pilot files only)
Frames sit on a common FPGA grid (--grid-fpga) so every frequency's frame k covers the same samples.
Single process, single thread: the analysis host's docker cannot start threads.
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
import scipy.fft

NFFT = 16384
NINPUT = 2048
W_MHZ = 0.390625
PILOT_OFFSET_MHZ = 0.309441
CUT_HALF = 64
SEARCH_HALF = 600          # +-14 kHz: stations run their pilot a few kHz off nominal (ch29 sits +3.1 kHz)

def channel_of(freq_id):
    f = 800.0 - freq_id * W_MHZ
    ch = 14 + int((f - 470.0) // 6.0)
    lo = 470.0 + (ch - 14) * 6.0
    return ch, lo

def pilot_fine_bin(freq_id):
    """Fine bin of the channel's pilot if this coarse bin holds it, else None (naive sense; both signs searched)."""
    ch, lo = channel_of(freq_id)
    centre = 800.0 - freq_id * W_MHZ
    off = lo + PILOT_OFFSET_MHZ - centre
    if abs(off) > W_MHZ / 2:
        return ch, None
    return ch, int(round(off / (W_MHZ / NFFT)))

def build_stack_map():
    """Product (a<=b, chan_id order, upper triangle incl. autos) -> stack id for the redundant layout
    cylinder = id // 512, pol block = (id // 256) % 2, position = id % 256."""
    a, b = np.triu_indices(NINPUT)
    cyl_a, cyl_b = a // 512, b // 512
    pol_a, pol_b = (a // 256) % 2, (b // 256) % 2
    pos_a, pos_b = a % 256, b % 256
    ew = cyl_b - cyl_a                      # 0..3 since a <= b
    ns = pos_b - pos_a                      # -255..255
    key = ((ew * 4 + pol_a * 2 + pol_b) * 511 + (ns + 255)).astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    count = np.bincount(inv).astype(np.int32)
    ew_u = uniq // (4 * 511); rem = uniq % (4 * 511); pp = rem // 511; ns_u = rem % 511 - 255
    keys = np.stack([ew_u, ns_u, pp // 2, pp % 2], axis=1).astype(np.int16)
    return inv.astype(np.int32), count, keys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5")
    p.add_argument("out_dir")
    p.add_argument("--grid-fpga", type=int, required=True, help="FPGA count of frame 0 (common to all files)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-corr", action="store_true", help="skip the 2048-input correlation (timing tests)")
    p.add_argument("--workers", type=int, default=8, help="fft threads (BLAS threads come from OPENBLAS_NUM_THREADS)")
    a = p.parse_args()
    t_all = time.time()
    f = h5py.File(a.h5, "r")
    freq_id = int(f.attrs["freq_id"]); freq = float(f.attrs["freq"])
    t0_fpga = int(f.attrs["time0_fpga_count"]); t0_ctime = float(f.attrs["time0_ctime"])
    dt = float(f.attrs["delta_time"])
    bb = f["baseband"]
    n_time = bb.shape[0]
    chan = f["index_map/input"]["chan_id"][:].astype(np.int64)
    if sorted(chan.tolist()) != list(range(NINPUT)):
        sys.exit(f"{a.h5}: chan_id is not a permutation of 0..2047")
    order = np.argsort(chan)                      # column i of X is chan_id i
    j0 = a.grid_fpga - t0_fpga
    if j0 < 0 or j0 >= n_time:
        sys.exit(f"{a.h5}: grid origin {a.grid_fpga} outside file (time0 {t0_fpga}, n {n_time})")
    n_frames = (n_time - j0) // NFFT
    if a.max_frames:
        n_frames = min(n_frames, a.max_frames)
    ch, kp = pilot_fine_bin(freq_id)
    is_pilot = kp is not None
    lut = (((np.arange(256) >> 4) - 8) + 1j * ((np.arange(256) & 15) - 8)).astype(np.complex64)
    stack_map, count, keys = build_stack_map()
    iu = np.triu_indices(NINPUT)
    n_stack = count.size
    stacks = np.zeros((n_frames, n_stack), np.complex64)
    autos = np.zeros((n_frames, NINPUT), np.float32)
    block_psd = np.zeros((n_frames, 8, NFFT), np.float32)
    pilot_cut = np.zeros((n_frames, NINPUT, 2 * CUT_HALF + 1), np.complex64) if is_pilot else None
    peak_bin = np.full(n_frames, -1, np.int32)
    peak_ratio = np.zeros(n_frames, np.float32)       # line power over the median fine bin, summed over inputs
    frame_fpga0 = np.zeros(n_frames, np.int64)
    timing = {"read": 0.0, "unpack": 0.0, "fft": 0.0, "corr": 0.0, "stack": 0.0}
    for k in range(n_frames):
        s = j0 + k * NFFT
        frame_fpga0[k] = t0_fpga + s
        t = time.time(); raw = bb[s:s + NFFT, :]; timing["read"] += time.time() - t
        t = time.time(); X = np.ascontiguousarray(lut[raw][:, order]); timing["unpack"] += time.time() - t
        autos[k] = (X.real ** 2 + X.imag ** 2).mean(axis=0)
        t = time.time()
        F = scipy.fft.fft(X, axis=0, workers=a.workers)
        P = F.real ** 2 + F.imag ** 2
        block_psd[k] = P.reshape(NFFT, 8, 256).sum(axis=2).T
        if is_pilot:
            tot = P.sum(axis=1)
            best = None
            for sign in (1, -1):
                c = (sign * kp) % NFFT
                idx = (c + np.arange(-SEARCH_HALF, SEARCH_HALF + 1)) % NFFT
                j = idx[np.argmax(tot[idx])]
                if best is None or tot[j] > tot[best]:
                    best = j
            peak_bin[k] = best
            peak_ratio[k] = tot[best] / np.median(tot)
            idx = (best + np.arange(-CUT_HALF, CUT_HALF + 1)) % NFFT
            pilot_cut[k] = F[idx, :].T
        timing["fft"] += time.time() - t
        if not a.no_corr:
            t = time.time(); V = X.conj().T @ X; timing["corr"] += time.time() - t
            t = time.time(); vf = V[iu]
            stacks[k] = (np.bincount(stack_map, weights=vf.real, minlength=n_stack)
                         + 1j * np.bincount(stack_map, weights=vf.imag, minlength=n_stack)) / count
            timing["stack"] += time.time() - t
        del raw, X, F, P
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{freq_id}.npz")
    meta = dict(file=os.path.basename(a.h5), freq_id=freq_id, freq_mhz=freq, channel=ch, is_pilot=bool(is_pilot),
                pilot_fine_bin_naive=kp, time0_fpga=t0_fpga, time0_ctime=t0_ctime, delta_time=dt, n_time=int(n_time),
                grid_fpga=a.grid_fpga, j0=int(j0), n_frames=int(n_frames), nfft=NFFT,
                layout="cylinder=id//512, polblock=(id//256)%2, position=id%256; stack key (ew, ns, pol_a, pol_b)",
                fft="scipy fft along time, bin order 0..16383 (negative frequencies in the upper half)",
                seconds=round(time.time() - t_all, 1), timing={k: round(v, 1) for k, v in timing.items()})
    np.savez(out, stacks=stacks, count=count, keys=keys, autos=autos, block_psd=block_psd, peak_bin=peak_bin, peak_ratio=peak_ratio,
             frame_fpga0=frame_fpga0, meta=json.dumps(meta),
             **({"pilot_cut": pilot_cut} if is_pilot else {}))
    print(json.dumps(meta))

if __name__ == "__main__":
    main()

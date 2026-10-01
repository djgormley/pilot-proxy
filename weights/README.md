# Weight banks

The files directly under this directory are the active, profile-backed banks:

- `chime_dtv_weights_k128.bin` and its manifest select the verified
  DC-centered CHIME baseband convention;
- `chord_dtv_weights_k64.bin` and its manifest select the CHORD geometry.

Treat each binary and adjacent manifest as one immutable pair. Runtime loading
checks the binary digest recorded by the manifest; documentation builds derive
their displayed digest prefixes from these active files.

## Historical CHIME bank

`legacy_halfband/` preserves the bank used by the 2026-07 manuscript sweeps and
the first archive epoch. It assumed the earlier half-band baseband convention
and must not be selected for a current detector run. The manuscript's provenance
manifest named these files explicitly so its historical hashes do not look like
hashes of the active bank.

The JSON stored beside the legacy binary is itself a preserved historical
artifact. Its embedded `artifacts.weights_path` and `manifest_path` record the
paths at which it was originally produced; they do not make it an active bank.
Do not rewrite that manifest merely to relocate it, because doing so would
destroy the manuscript's byte-level provenance.

The manuscript was removed from this repository; its provenance hash list is
`paper/manuscript/provenance/hashes.sha256@30e50c2`, checked from a worktree at
that commit with `sha256sum --check paper/manuscript/provenance/hashes.sha256
--ignore-missing`.

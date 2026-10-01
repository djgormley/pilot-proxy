# Adapting the detector side to another project

The detector side is written so that a new telescope, a new emitter or a new
science goal changes data and adapters, not the method. Four tiers, from the
smallest change to the largest:

| Tier | Example | What changes | What does not |
|---|---|---|---|
| (a) A new telescope, the same interference | ATSC pilots seen by another array | `src/pilot_proxy/instruments/<name>.yaml` (channelization, feed count, site, feed layout), a reader under `src/pilot_proxy/instruments/<name>/`, the project's detector configuration and integration model | `detectors/`, `characterization/`, the handoff, the science side |
| (b) Another emitter with a narrowband marker | A carrier or pilot of another standard | the project's frequency plan (an explicit band list) and interference template values, possibly a new weight bank and a new testbench generator | the detector adapter and the characterization |
| (c) An emitter with no marker | wideband OFDM, pulsed radar | a new detector adapter under `src/pilot_proxy/detectors/` that implements the `DetectorAdapter` protocol of `detectors/interface.py`: it produces the frame table and declares its null law, its residual estimator and its candidate families; perhaps a new kernel | `characterization/`, the handoff contract, the science side |
| (d) A new science model | another Fisher code, a target other than BAO | nothing here: a new science model and profile on the science side (RFIsher `docs/adapting.md`) | this repository |

## A project profile

A project is a directory under `projects/` (this one is `projects/chime_atsc/`):
`project.yaml` names the instrument and the files of the profile, which are
the detector configuration, the frequency plan, the interference template, the
integration model, the era lists (`eras/`) and the detector register
(`detector_register.json`, every entry tagged `side: detector`; the loader
refuses a science entry). The characterization manifests record the sha256 of
every profile file. Code reads values from the profile and the register; it
does not carry them as literals (`tests/config/test_profile_literals.py`
records every literal that was replaced, and the value that stands in for it).

## What stays project-specific

These live only in profiles, adapters or records:

- CHIME: `src/pilot_proxy/instruments/chime*`, `configs/receiver_profiles/*`,
  `configs/stream_maps/*`, the weight banks;
- ATSC: `projects/chime_atsc/{frequency_plan,interference_template}.yaml` and
  the ATSC testbench generator;
- the kernel C API names and the kotekan stage names;
- the era lists and transmitter-off intervals (`projects/chime_atsc/eras/`);
- the 2026 campaign's records (`src/pilot_proxy/records/chime_atsc_2026/`):
  the capture campaign, its frame policy, the archive releases and the frozen
  dissertation export;
- on-disk tokens: names that frozen files carry and that are kept as they are
  (the list is in [TERMINOLOGY.md](TERMINOLOGY.md), "Frozen tokens").

The general layers (`characterization/`, `products/`, `detectors/interface.py`)
name no emitter, instrument or band; `tests/core/test_general_layers.py`
checks their vocabulary and that they hold no band count, and
`tests/core/test_detector_side.py` that the library holds no cosmology term.

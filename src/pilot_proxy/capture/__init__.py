"""Capture characterization: frame-level measurements on reduced baseband dumps.

A capture product is one reduced file per coarse channel (``<freq_id>.npz``)
with the class stacks of redundant baselines per frame (``stacks``), the
redundant-product multiplicity of each class (``count``), per-input powers
(``autos``) and the file's ``meta``. The modules here measure on those
products what the detector side owns: the class excess and its control level,
the frame residual, the cadence coherence time and its within-dump lower
bounds, the lag moments, and the placement of each dump on the archive's
coarse ladder. ``oc_table`` writes the handoff the science side reads
(``capture_operating_characteristic_v1``). No tolerance, credit, gain model or
disposition lives here.
"""

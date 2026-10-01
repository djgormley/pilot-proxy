# Bench amendment 2, 2026-09-18: the ladder moves from commanded amplitude to transmit gain

Written after the amplitude sweep and before any transmit-gain record.

Why. The six-rung protocol steps the commanded tone amplitude. Measured on one
wiring, back to back, with no saturation and an unchanged path, that knob is not
a level control on this device: 0.005 to 0.0025 gives -18.0 dB against 6.02
commanded, 0.0025 to 0.00125 gives a further -1.0 dB, and the two lowest
amplitudes read -44.9 and -44.0 dB, the lower one higher than the one above it.
The response is a quantization floor in the transmit path, so the rung protocols
cannot be run and any slope fitted through them would be meaningless.

The ladder. Commanded amplitude is fixed at 0.005, the value of the 2026-09-09
records and the only one this device reproduces. The level is stepped by the
native transmit gain setting, a hardware attenuator, over 50, 44, 38, 32 and 26,
five points 6 dB apart spanning 24 dB, in the wiring in place when this was
written: one 30 dB pad between TX and the cable, no terminators. Each record is
2 s, the helper and worker are unchanged, and the requested and expected gain
readbacks are equal so a device that does not honour the setting fails the
record rather than reporting a wrong level.

Acceptance, fixed before the records. Over the points whose target-bin power is
at least 10 dB above the floor measured in the same wiring, with zero saturated
components, measured power in dB against commanded gain in dB has slope 1.00
within 0.10 and residual scatter at or below 1.0 dB. The first and last points
are recorded at gain 50 to bracket drift, and they must agree within 0.5 dB.

What it can and cannot establish. It measures the estimator's response to a
known analog level change through one input, which is the single-input physical
reference. It says nothing about sensitivity, receiver operating characteristics,
absolute power, the array, or the CHIME deployment, and it does not rehabilitate
the amplitude ladder, whose failure is reported as a finding in its own right.

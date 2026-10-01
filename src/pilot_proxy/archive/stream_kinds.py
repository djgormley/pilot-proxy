# coding=utf-8
"""Semantic stream kinds shared by Pilot Proxy archive components.

Stream kinds describe the in-memory reader/analyzer boundary, independently of
the receiver or on-disk container that supplied the samples.
"""
from __future__ import annotations

# One uint8 per complex sample: offset-binary int4 real in the high nibble and
# offset-binary int4 imaginary in the low nibble, yielded as baseband frames.
STREAM_PACKED_COMPLEX_INT4_BASEBAND = "packed-complex-int4-baseband-frame"


__all__ = ["STREAM_PACKED_COMPLEX_INT4_BASEBAND"]

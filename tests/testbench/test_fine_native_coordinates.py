"""Independent algebra and native-byte regression checks for the engineering audit."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_coordinate_audit", ROOT / "tools/audit_fine_native_coordinates.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_reversal_does_not_reverse_slow_time():
    rows = (np.arange(256).reshape(2, 128) + 1j).astype(np.complex64)
    adapted = AUDIT.raw_adapt(rows)
    assert adapted[0, 0] == 127 - 1j
    assert adapted[1, 0] == 255 - 1j
    np.testing.assert_array_equal(AUDIT.raw_adapt(adapted), rows)


def test_noninteger_weight_phase_and_mirrored_absolute_powers():
    # Non-bin-centred exponentials prevent a trivial integer-bin phase shortcut.
    sample = np.arange(128)
    weights = np.exp(-2j * np.pi * np.array([0.11731, -0.31917, 0.27641])[:, None] * sample)
    t = np.arange(128 * 128)
    rows = (np.exp(0.07123j * t) + 0.4 * np.exp(-0.2811j * t)).reshape(128, 128)
    AUDIT.assert_float_mapping(rows, AUDIT.raw_adapt(rows), weights, 1)


def test_native_offset_repack_all_bytes_including_negative_eight():
    from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
    from pilot_proxy.chime.hdf5_input import CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4
    offset = np.tile(np.arange(256, dtype=np.uint8), 64).reshape(1, 1, 16384)
    packed = pack_chime_block_for_detector(
        offset, frame_size_samples=16384, detector_window_samples=128,
        spectral_sense="inverted", frames_in_chunk=1,
        sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        selected_coarse_channel=843, physical_channel=14,
    ).packed[0]
    expected = np.bitwise_xor(offset.reshape(128, 128)[:, ::-1], 0x88).view(np.int8)
    np.testing.assert_array_equal(packed, expected)

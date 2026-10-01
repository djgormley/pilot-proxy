"""The float64 excess-8 decoder of the in-band lag moments agrees in value with the reader's float32 unpacking."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("yaml")

from pilot_proxy.chime.baseband_format import decode_excess8, unpack_4bit  # noqa: E402


def test_every_byte_decodes_to_the_same_value_in_float64():
    b = np.arange(256, dtype=np.uint8).reshape(16, 16)
    x = decode_excess8(b)
    assert x.dtype == np.complex128 and x.shape == b.shape
    y = unpack_4bit(b)
    assert y.dtype == np.complex64
    np.testing.assert_array_equal(x, y.astype(np.complex128))
    assert x[0, 0] == complex(-8, -8) and x[15, 15] == complex(7, 7)
    assert decode_excess8(np.array([0x9A], np.uint8))[0] == complex(1, 2)    # high nibble real, low imaginary


def test_non_byte_input_is_refused():
    with pytest.raises(TypeError):
        decode_excess8(np.arange(4, dtype=np.int16))

"""External archives are eagerly owned, validated and never unpickled."""
import numpy as np
import pytest

from pilot_proxy.products.npzio import load_npz


@pytest.mark.parametrize("compressed", [False, True])
def test_numeric_and_string_archives_survive_file_removal(tmp_path, compressed):
    path = tmp_path / "survey.npz"
    source = {"power": np.arange(12, dtype=np.uint64).reshape(3, 4),
              "empty": np.empty(0), "metadata": np.array('{"version": 5}')}
    (np.savez_compressed if compressed else np.savez)(path, **source)
    loaded = load_npz(path, required=iter(source))
    path.unlink()
    for name, expected in source.items():
        np.testing.assert_array_equal(loaded[name], expected)
        assert loaded[name].flags.owndata
    loaded["power"][0, 0] = 99
    assert source["power"][0, 0] == 0


def test_missing_fields_are_reported_together_in_stable_order(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, present=[1])
    with pytest.raises(ValueError, match="incomplete.npz.*alpha, zeta"):
        load_npz(path, required=("zeta", "alpha", "alpha", "present"))


def test_even_unrequested_object_metadata_is_rejected_without_unpickling(tmp_path):
    path = tmp_path / "untrusted.npz"
    marker = tmp_path / "pickle-executed"

    class Payload:
        def __reduce__(self):
            # Saving records this callable; only an unsafe load would run it.
            return (eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",))

    np.savez(path, power=[1], metadata=np.array([Payload()], dtype=object))
    with pytest.raises(ValueError, match="untrusted.npz.*pickle-backed"):
        load_npz(path, required=["power"])
    assert not marker.exists()


def test_missing_and_corrupt_archives_fail(tmp_path):
    path = tmp_path / "broken.npz"
    with pytest.raises(FileNotFoundError):
        load_npz(path)
    path.write_bytes(b"not a numpy archive")
    with pytest.raises(ValueError):
        load_npz(path)

"""The producer record of a run: informational, and fixed at first use."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pilot_proxy import __version__, provenance
from pilot_proxy.config.project import default_project


def test_the_producer_names_the_package_its_sources_and_the_register():
    identity = provenance.producer_identity()
    assert identity["repository"] == "WVURAIL/pilot-proxy" and identity["package"] == "pilot_proxy"
    assert identity["version"] == __version__
    assert identity["register_sha256"] == default_project().register.sha256()
    assert identity["source_digest"] == provenance.package_source_sha256()
    assert len(identity["commit"]) in (0, 40) or identity["commit"] == "unknown"
    assert identity["dirty"] in (True, False, None)
    assert Path(identity["source_root"]).name == "pilot_proxy"
    assert provenance.producer_identity() == identity


def test_further_roots_are_digested(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    identity = provenance.producer_identity({"other": tmp_path})
    assert set(identity["other_source_digests"]) == {"other"}


def test_the_file_digest_is_sha256(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"pilot-proxy\n")
    assert provenance.sha256_file(path) == hashlib.sha256(b"pilot-proxy\n").hexdigest()

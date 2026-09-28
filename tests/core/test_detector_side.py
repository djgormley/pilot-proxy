"""The detector side carries no cosmology: no forecast, tolerance or survey term in the new packages or the profile."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "pilot_proxy"
PACKAGES = ("characterization", "products", "detectors", "records", "testbench/harness")
TOKENS = re.compile(r"redshift|r_tol|fisher|cosmolog|bao|omega", re.IGNORECASE)
# Frozen on-disk values the dissertation's importers check (K6); exempt by name, and only these.
FROZEN = ("rfisher_results.archive.psd window spectra v1", "rfisher-archive-report", "WVURAIL/RFIsher",
          "rfisher-dissertation-numbers")
# The detector register cites the release code each entry was read from; the sibling repository's name in a
# provenance path is not a cosmology term.
PROVENANCE = re.compile(r"(src/)?rfisher/[A-Za-z_/]+\.py|RFIsher|rfisher-threshold-decisions-v\d+")


def _files():
    for package in PACKAGES:
        yield from sorted((SRC / package).rglob("*.py"))
    yield from sorted((ROOT / "projects").rglob("*.yaml"))
    yield from sorted((ROOT / "projects").rglob("*.json"))


@pytest.mark.parametrize("path", list(_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_cosmology_token(path):
    text = path.read_text(encoding="utf-8")
    for token in FROZEN:
        text = text.replace(token, "")
    if path.suffix != ".py":
        text = PROVENANCE.sub("", text)
    found = sorted({m.group(0) for m in TOKENS.finditer(text)})
    assert not found, found


def test_the_frozen_tokens_are_still_written():
    report = (SRC / "characterization" / "report" / "core.py").read_text(encoding="utf-8")
    assert '"rfisher-archive-report"' in report and '"WVURAIL/RFIsher"' in report
    assert '"rfisher_results.archive.psd window spectra v1"' in (
        SRC / "detectors" / "narrowband_marker" / "psd.py").read_text(encoding="utf-8")

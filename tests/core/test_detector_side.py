"""The detector side carries no cosmology: no forecast, tolerance or survey term in the library or the profile."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "pilot_proxy"
TOKENS = re.compile(r"redshift|\br_tol\b|fisher|cosmolog|bao|omega", re.IGNORECASE)
# Frozen on-disk values: exempt by name, and only these.
FROZEN = ("rfisher_results.archive.psd window spectra v1", "rfisher-archive-report", "WVURAIL/RFIsher",
          "rfisher-dissertation-numbers",
          # the capture frame-policy record writes this limitation into its receipt.json (record c49ae37f)
          "Sample covariance is not a temporal covariance model, confidence bound, or Fisher residual.",
          # the archive health view's schema name, written into every view
          "pilotproxy_baonoise_health_view_v1",
          # the frozen v1 dissertation export (records/chime_atsc_2026/dissertation_export_v1.py): its table names,
          # their owner strings and one table's description, as its exports carry them
          "bao_time_vs_masking", "bao_convergence", "bao_two_walls", "bao_policy_case", "external-fisher-forecast",
          "Uniform-mask observing-time curves from the BAO forecast code.")
# The sibling repository's name (in a provenance path, a command it runs, or the repository itself) is not a cosmology
# term; importing it is refused by test_no_rfisher_import.
REPOSITORY = re.compile(r"\bRFIsher\b|\brfisher\b")
# The detector register cites the release code each entry was read from; the sibling repository's name in a
# provenance path is not a cosmology term.
PROVENANCE = re.compile(r"(src/)?rfisher/[A-Za-z_/]+\.py|RFIsher|rfisher-threshold-decisions-v\d+")


def _files():
    yield from sorted(SRC.rglob("*.py"))
    yield from sorted((ROOT / "projects").rglob("*.yaml"))
    yield from sorted((ROOT / "projects").rglob("*.json"))


@pytest.mark.parametrize("path", list(_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_cosmology_token(path):
    text = path.read_text(encoding="utf-8")
    for token in FROZEN:
        text = text.replace(token, "")
    if path.suffix != ".py":
        text = PROVENANCE.sub("", text)
    text = REPOSITORY.sub("", text)
    found = sorted({m.group(0) for m in TOKENS.finditer(text)})
    assert not found, found


def test_the_frozen_tokens_are_still_written():
    report = (SRC / "characterization" / "report" / "core.py").read_text(encoding="utf-8")
    assert '"rfisher-archive-report"' in report and '"WVURAIL/RFIsher"' in report
    assert '"rfisher_results.archive.psd window spectra v1"' in (
        SRC / "detectors" / "narrowband_marker" / "psd.py").read_text(encoding="utf-8")

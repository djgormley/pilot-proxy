"""The score bundle's producer record (residual_score_producer_v2) and its round trip into a prepared family."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from test_residual_scores import _bulk, _measured, _synthetic_product
from pilot_proxy import __version__
from pilot_proxy.detectors.narrowband_marker.scores import (
    SCORE_BUNDLE_SCHEMA, build_score_bundle, load_score_bundle)

ROOT = Path(__file__).resolve().parents[3]


def test_the_bundle_names_the_code_that_scored_it_and_enters_preparation(tmp_path):
    product = tmp_path / "product.npz"
    _synthetic_product(product)
    bundle = build_score_bundle(product, np.ones(48, dtype=bool), anchor_bin=0, designated_half_width=0,
                                bulk_mask=_bulk(10, 11, 12, 13))
    loaded = load_score_bundle(bundle.save(tmp_path / "scores.npz"))
    assert SCORE_BUNDLE_SCHEMA == "residual_score_bundle_v2"
    assert loaded.frame_count == 48 and loaded.supported_rho_count == 4
    assert loaded.manifest["source"]["product_sha256"] == hashlib.sha256(product.read_bytes()).hexdigest()
    producer = loaded.manifest["producer"]
    assert producer["schema_version"] == "residual_score_producer_v2"
    assert (producer["package"], producer["package_version"]) == ("pilot-proxy", __version__)
    expected = {label: hashlib.sha256((ROOT / label).read_bytes()).hexdigest() for label in (
        "src/pilot_proxy/characterization/surface.py", "src/pilot_proxy/detectors/narrowband_marker/scores.py")}
    assert producer["source_files"] == expected
    digest = hashlib.sha256(b"residual-score-producer-v2\0")
    for label, file_sha256 in sorted(expected.items()):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256))
    assert producer["source_sha256"] == digest.hexdigest()
    assert loaded.content_sha256 == bundle.content_sha256
    prepared = loaded.prepare_threshold_family(
        np.full(48, 0.02), variance_residuals=np.zeros(48), era_label="synthetic-current-era", latest_era=True,
        additive_residuals=True, score=_measured("exact score"), correlation=_measured("correlation"),
        transfer=_measured("transfer"), max_cost_ratio=1.1, max_systematic_residual_ratio=1.1,
        minimum_half_retained_frames=10, minimum_observed_months=1, minimum_span_days=1e-6)
    assert prepared.source_id == loaded.source_id and set(prepared.histograms_by_rho) == {1, 2, 3, 4}

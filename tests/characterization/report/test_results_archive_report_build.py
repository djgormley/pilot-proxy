"""The report registry: every registered module exposes its builders."""
from __future__ import annotations

from pilot_proxy.characterization.report import build as rb


def test_registry_imports_every_table_module_and_yields_callables():
    builders = rb.table_builders()
    assert len(builders) >= len(rb.TABLE_MODULES) and all(callable(b) for b in builders)
    names = [b.__module__.rsplit(".", 1)[-1] for b in builders]
    assert set(rb.TABLE_MODULES) <= set(names)


def test_figure_modules_expose_render():
    for name in rb.FIGURE_MODULES:
        mod = rb._module(name)
        assert callable(getattr(mod, "render", None))


def test_build_report_applies_the_document_style_before_rendering(tmp_path, monkeypatch):
    """The figures are set in the document's own typography; a report rendered without it cannot be vendored."""
    import json

    from pilot_proxy import figure_style as style

    calls = []
    monkeypatch.setattr(style, "configure", lambda *, require_tex=True: calls.append(require_tex))
    ledger = tmp_path / "ledger"
    (ledger / "channels").mkdir(parents=True)
    (ledger / "run.json").write_text(json.dumps({"generated": "x", "producer": {"commit": "a" * 40}, "channels": []}))
    monkeypatch.setattr(rb, "TABLE_MODULES", ())
    monkeypatch.setattr(rb, "FIGURE_MODULES", ())
    rb.build_report(tmp_path, tmp_path / "out", commit="a" * 40, generated="g")
    assert calls == [True]
    rb.build_report(tmp_path, tmp_path / "out2", commit="a" * 40, generated="g", require_tex=False)
    assert calls == [True, False]
    rb.build_report(tmp_path, tmp_path / "out3", commit="a" * 40, generated="g", figures=False)
    assert calls == [True, False]                      # no figures, no style to apply

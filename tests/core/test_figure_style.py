"""Headless diagram layout, semantic colors, and cleanup on errors."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfFile
import pytest

from pilot_proxy import figure_style as style


@pytest.fixture
def axes(monkeypatch):
    monkeypatch.setattr(style.shutil, "which", lambda name: None)
    with matplotlib.rc_context():
        style.configure(require_tex=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        yield ax
        plt.close(fig)


def test_missing_tex_is_explicit_and_preview_remains_available(axes):
    assert not matplotlib.rcParams["text.usetex"]
    with pytest.raises(RuntimeError, match="requires latex"):
        style.configure(require_tex=True)


@pytest.mark.parametrize("body", ["", "short body", "long explanatory words " * 40, r"$x^2 + y^2 = z^2$"])
def test_diagram_boxes_handle_empty_short_wrapped_and_math_text(axes, body):
    patch = style.diagram_box(axes, (0.05, 0.2), (0.4, 0.45), title="Measured", body=body, status="measured")
    style.diagram_arrow(axes, (0.5, 0.4), (0.8, 0.4), status="conditional")
    style.panel_label(axes, "a")
    style.clean_axes(axes)
    axes.figure.canvas.draw()
    assert patch in axes.patches
    assert patch.get_facecolor() == matplotlib.colors.to_rgba(style.LIGHT_BLUE)
    assert all(text.get_fontsize() >= 5.6 for text in axes.texts)
    assert not axes.spines["top"].get_visible()
    assert not any(text.get_alpha() == 0 for text in axes.texts)  # no measuring probes left behind


def test_subset_hook_restores_even_after_renderer_failure(monkeypatch):
    original = lambda charset: "ORIGINAL+"
    monkeypatch.setattr(PdfFile, "_get_subset_prefix", staticmethod(original), raising=False)
    with pytest.raises(RuntimeError, match="renderer failed"):
        with style.stable_pdf_subset_tags():
            assert PdfFile._get_subset_prefix(["b", "a"]) == style.stable_subset_prefix(["a", "b"])
            raise RuntimeError("renderer failed")
    assert PdfFile._get_subset_prefix([]) == "ORIGINAL+"
    monkeypatch.delattr(PdfFile, "_get_subset_prefix")
    with style.stable_pdf_subset_tags():
        assert not hasattr(PdfFile, "_get_subset_prefix")


def test_legend_has_distinct_semantic_labels_and_colors():
    statuses = ["measured", "model", "conditional", "failure", "pending"]
    handles = style.status_handles(statuses)
    assert len({handle.get_color() for handle in handles}) == 5
    assert [handle.get_label() for handle in handles] == ["measured", "model / transfer",
                                                         "conditional / feasible", "failure / excision", "pending / unmeasured"]

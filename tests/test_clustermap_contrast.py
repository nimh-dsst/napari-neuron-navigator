"""Contrast changes must reveal distances without changing analysis results."""

import os
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from dataclasses import asdict
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from napari_neuron_navigator.analysis.clustering import ClusterResult
from napari_neuron_navigator.analysis.clustermap_contrast import (
    ClustermapContrast,
    compute_clustermap_contrast_statistics,
)
from napari_neuron_navigator.analysis.export import (
    build_clustermap_figure,
    save_dendrogram_figure,
)


def _result(pairs):
    pairs = np.asarray(pairs, dtype=np.float32)
    matrix = squareform(pairs)
    tree = linkage(pairs, method="ward")
    return ClusterResult(
        correlation_matrix=1.0 - matrix,
        distance_matrix=matrix,
        linkage_matrix=tree,
        neuron_ids=[f"file-{i}" for i in range(len(matrix))],
        reorder_indices=leaves_list(tree),
    )


def _heatmap(figure):
    return figure.axes[4].images[0]


@pytest.mark.parametrize("legacy", [False, True])
def test_auto_uses_full_off_diagonal_distribution_without_mutation(legacy):
    pairs = np.linspace(0.991, 1.002, 4950, dtype=np.float32)
    if legacy:
        pairs[2000:] = 2.0  # Legacy disjoint pairs must remain distance 2.
    pairs[0] = 0.0
    pairs[-1] = 2.0
    result = _result(pairs)
    originals = {
        name: value.copy()
        for name, value in asdict(result).items()
        if isinstance(value, np.ndarray)
    }
    statistics = result.clustermap_contrast_statistics
    assert statistics is result.clustermap_contrast_statistics
    np.testing.assert_allclose(
        [statistics.auto_minimum, statistics.auto_maximum],
        np.percentile(pairs, [1, 99]),
        rtol=0,
        atol=1e-12,
    )
    assert statistics.extension(statistics.resolve()) == ("min" if legacy else "both")
    figures = [
        build_clustermap_figure(result, max_render_size=size) for size in (8, 100)
    ]
    try:
        for figure in figures:
            assert _heatmap(figure).get_clim() == (
                statistics.auto_minimum,
                statistics.auto_maximum,
            )
            assert _heatmap(figure).colorbar.extend == statistics.extension(
                statistics.resolve()
            )
            assert figure.axes[5].get_title() == "Auto"
        for name, original in originals.items():
            np.testing.assert_array_equal(getattr(result, name), original)
        assert "clustermap_contrast_statistics" not in asdict(result)
    finally:
        for figure in figures:
            plt.close(figure)


def test_coincident_percentiles_fall_back_to_pair_extrema():
    pairs = np.ones(4950, dtype=np.float32)
    pairs[0], pairs[-1] = 0.5, 1.5
    statistics = compute_clustermap_contrast_statistics(squareform(pairs))
    assert statistics.resolve() == ClustermapContrast(0.5, 1.5, "auto")
    assert "off-diagonal range" in statistics.explanation(statistics.resolve())
    assert statistics.explanation(statistics.resolve("full")) == ""
    assert statistics.resolve("full") == ClustermapContrast(0.0, 1.5, "full")


@pytest.mark.parametrize("value", [0.0, 1.0, 2.0])
def test_constant_distances_are_uniform_and_explained(value):
    result = _result([value] * 6)
    statistics = result.clustermap_contrast_statistics
    contrast = statistics.resolve()
    assert contrast.minimum < value < contrast.maximum
    # SciPy's zero-height dendrogram expands identical axis limits; the
    # distance heatmap must still render uniformly without changing its data.
    axis_warning = (
        pytest.warns(UserWarning, match="identical low and high")
        if value == 0
        else nullcontext()
    )
    with axis_warning:
        figure = build_clustermap_figure(result)
    try:
        image = _heatmap(figure)
        data = image.get_array()
        colors = image.to_rgba(data)
        off_diagonal = ~np.eye(len(data), dtype=bool)
        assert np.unique(colors[off_diagonal], axis=0).shape[0] == 1
        assert any("no contrast to stretch" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_nonfinite_pairs_are_ignored_and_no_finite_pairs_are_explained():
    matrix = squareform([0.995, np.nan, 1.001, np.inf, 1.002, 0.998])
    statistics = compute_clustermap_contrast_statistics(matrix)
    np.testing.assert_allclose(
        [statistics.auto_minimum, statistics.auto_maximum],
        np.percentile([0.995, 1.001, 1.002, 0.998], [1, 99]),
    )
    empty = compute_clustermap_contrast_statistics(np.full((2, 2), np.nan))
    assert "No finite off-diagonal" in empty.message
    assert empty.resolve().minimum < empty.resolve().maximum


@pytest.mark.parametrize(
    "limits", [(1, 1), (2, 1), (np.nan, 1), (0, np.inf), (-np.inf, 1)]
)
def test_invalid_manual_limits_are_rejected(limits):
    with pytest.raises(ValueError, match="finite.*minimum below maximum"):
        ClustermapContrast(*limits)


def test_preview_and_export_share_manual_limits_and_unrounded_colorbar(tmp_path):
    result = _result([0.9999996, 0.9999997, 0.9999998, 1.0, 1.0000001, 1.0000002])
    contrast = ClustermapContrast(0.9999997, 1.0000001)
    preview = build_clustermap_figure(result, contrast=contrast, max_render_size=2)
    saved = []
    original_build = build_clustermap_figure

    def capture(*args, **kwargs):
        figure = original_build(*args, **kwargs)
        saved.append(figure)
        return figure

    try:
        with patch(
            "napari_neuron_navigator.analysis.export.build_clustermap_figure",
            side_effect=capture,
        ):
            output = save_dendrogram_figure(
                tmp_path / "contrast.png", result, contrast=contrast, dpi=100
            )
        assert output.exists()
        for figure in (preview, saved[0]):
            image = _heatmap(figure)
            assert image.get_clim() == (contrast.minimum, contrast.maximum)
            assert image.colorbar.extend == "both"
            assert figure.axes[5].get_title() == "Manual"
            formatter = image.colorbar.formatter
            assert formatter(0.9999997) != formatter(1.0000001)
            assert float(formatter(0.9999997)) == 0.9999997
        tiny = ClustermapContrast(1.000000000001, 1.000000000003)
        assert tiny.format_value(tiny.minimum) != tiny.format_value(tiny.maximum)
        assert float(tiny.format_value(tiny.minimum)) == tiny.minimum
    finally:
        plt.close(preview)
        for figure in saved:
            plt.close(figure)


def test_real_contrast_controls_and_export_snapshot(tmp_path):
    """Isolate real Qt from other tests' lightweight widget stand-ins."""
    script = textwrap.dedent("""
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from qtpy.QtWidgets import QApplication
        from napari_neuron_navigator.widgets.analysis_tab import AnalysisTabWidget
        from tests.test_analysis_region_options import _DummyViewer
        from tests.test_clustermap_contrast import _result

        app = QApplication.instance() or QApplication([])
        widget = AnalysisTabWidget(_DummyViewer())
        assert not widget._clustermap_lock_cb.isChecked()
        assert not widget._clustermap_contrast_controls.isEnabled()
        first = _result([0.991, 0.996, 1.001, 0.999, 1.0, 1.002])
        second = _result([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        third = _result([10, 20, 30, 40, 50, 60])
        for result in (first, second):
            result.metadata = SimpleNamespace(distance_metric="one_minus_pearson_r")
        third.metadata = SimpleNamespace(distance_metric="euclidean_um")

        def select(result):
            widget._last_cluster_result = result
            widget._update_button_states()

        select(first)
        auto = widget._clustermap_contrast
        assert auto.mode == "auto"
        assert widget._clustermap_contrast_controls.isEnabled()
        widget._draw_clustermap(first)
        initial_image = widget._figure.axes[4].images[0]
        assert initial_image.get_clim() == (auto.minimum, auto.maximum)
        for lower, upper in (("nan", "1"), ("0", "inf"), ("2", "1"), ("1", "1"), ("", "1")):
            widget._clustermap_min_edit.setText(lower)
            widget._clustermap_max_edit.setText(upper)
            widget._clustermap_min_edit.editingFinished.emit()
            assert widget._clustermap_contrast is auto
            assert widget._figure.axes[4].images[0] is initial_image
            assert "previous valid scale" in widget._clustermap_contrast_label.text()
        widget._clustermap_min_edit.setText(format(auto.minimum, ".10g"))
        widget._clustermap_max_edit.setText(format(auto.maximum, ".10g"))
        widget._clustermap_max_edit.editingFinished.emit()
        assert widget._clustermap_contrast is auto
        assert "Auto contrast" in widget._clustermap_contrast_label.text()
        assert "previous valid scale" not in widget._clustermap_contrast_label.text()
        widget._clustermap_min_edit.setText("9.95e-1")
        widget._clustermap_max_edit.setText("1.001")
        widget._clustermap_max_edit.editingFinished.emit()
        manual = widget._clustermap_contrast
        assert manual.mode == "manual"
        assert (manual.minimum, manual.maximum) == (0.995, 1.001)
        assert widget._figure.axes[4].images[0].get_clim() == (0.995, 1.001)
        widget._clustermap_min_edit.setText("1.000000000001")
        widget._clustermap_max_edit.setText("1.000000000003")
        widget._clustermap_max_edit.editingFinished.emit()
        assert widget._clustermap_min_edit.text() != widget._clustermap_max_edit.text()
        widget._clustermap_min_edit.setText("0.995")
        widget._clustermap_max_edit.setText("1.001")
        widget._clustermap_max_edit.editingFinished.emit()
        manual = widget._clustermap_contrast
        widget._prompt_analysis_export_path = lambda *args: "unused.png"
        with patch("napari_neuron_navigator.analysis.export.save_dendrogram_figure") as save:
            widget._save_dendrogram()
            assert save.call_args.kwargs["contrast"] is manual

        # A repeated control refresh must neither recompute nor reset the scale.
        with patch("napari_neuron_navigator.analysis.clustermap_contrast.compute_clustermap_contrast_statistics", side_effect=AssertionError("recomputed")):
            widget._update_button_states()
            widget._draw_clustermap(first)
            assert widget._clustermap_contrast is manual
        widget._clustermap_full_btn.click()
        assert widget._clustermap_contrast.mode == "full"
        assert widget._clustermap_contrast.minimum == 0
        widget._clustermap_auto_btn.click()
        assert widget._clustermap_contrast == auto
        widget._clustermap_lock_cb.setChecked(True)
        select(second)
        assert widget._clustermap_lock_cb.isChecked()
        assert widget._clustermap_contrast.mode == "manual"
        assert (widget._clustermap_contrast.minimum, widget._clustermap_contrast.maximum) == (auto.minimum, auto.maximum)
        assert not widget._clustermap_rendered  # No stale plot from the previous run.
        select(third)
        assert not widget._clustermap_lock_cb.isChecked()
        assert widget._clustermap_contrast.mode == "auto"
        assert widget._clustermap_contrast.minimum > 1
        assert "metric changed" in widget._clustermap_contrast_label.text()
        select(second)
        assert widget._clustermap_contrast == second.clustermap_contrast_statistics.resolve()
        select(None)
        assert widget._clustermap_contrast is None
        assert not widget._clustermap_contrast_controls.isEnabled()
        widget.close()
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "MPLCONFIGDIR": str(tmp_path),
        },
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

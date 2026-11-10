"""Exercise multi-viewer OpenGL cleanup through the full plugin and Qt loop."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(
    os.environ.get("NN_RUN_RENDERING_TESTS") != "1",
    reason="Requires desktop OpenGL; run pixi run test-rendering",
)
def test_style_dropdown_keeps_scene_and_binds_cleanup_context():
    script = textwrap.dedent("""
        import logging
        import os
        import sys
        import traceback
        from unittest.mock import patch
        import napari
        import numpy as np
        from qtpy.QtCore import Qt, QTimer
        from qtpy.QtGui import QOpenGLContext
        from qtpy.QtTest import QTest
        from qtpy.QtWidgets import QTabWidget
        from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

        class FailOnRenderingError(logging.Handler):
            def emit(self, record):
                if "Error drawing visual" in record.getMessage():
                    print(record.getMessage(), flush=True)
                    os._exit(3)

        logging.getLogger("vispy").addHandler(FailOnRenderingError())
        main = napari.Viewer(title="Flatmap regression: controls")
        with patch.object(NeuronViewerWidget, "_start_cached_template_autoload"):
            owner = NeuronViewerWidget(main)
        main.window.add_dock_widget(owner, name="Neuron Navigator")
        main.window._qt_window.resize(1200, 900)
        reference = main.add_image(np.arange(4096).reshape(8, 16, 32), name="Reference")
        widget = owner._flatmap_tab
        for tabs in owner.findChildren(QTabWidget):
            for index in range(tabs.count()):
                if tabs.tabText(index) == "Flatmap":
                    tabs.setCurrentIndex(index)
        widget._lookup_files_section.set_expanded(True)
        display = owner._get_or_create_flatmap_viewer()

        def assert_deletion_context(event):
            expected = display.window._qt_viewer.canvas.native.context()
            if QOpenGLContext.currentContext() != expected:
                print("GPU cleanup used another viewer's context", flush=True)
                os._exit(4)

        display.layers.events.removing.connect(assert_deletion_context)
        changes = 0
        previous_layers = []
        vertices = np.array([[1, 2, 3], [6, 2, 3], [1, 12, 3], [1, 2, 22]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        vectors = np.array([[[1, 2, 3], [5, 0, 0]], [[1, 2, 3], [0, 10, 0]]], dtype=float)

        def later(delay, callback):
            def guarded():
                try:
                    callback()
                except BaseException:
                    traceback.print_exc()
                    sys.stderr.flush()
                    os._exit(1)
            QTimer.singleShot(delay, guarded)

        def populate():
            assert not display.layers
            metadata = {
                "napari_neuron_navigator_space": "flatmap",
                "flatmap_style": widget._current_style_key(),
                "flatmap_render_mode": "heatmap",
                "flatmap_plane_mode": "depth",
                "flatmap_soma_space_render_mode": "heatmap",
            }
            image = display.add_image(
                np.arange(4096).reshape(8, 16, 32), name="Isocortex Flatmap Heatmap",
                metadata=metadata.copy(), colormap="magma",
            )
            widget._projection_layer = image
            widget._region_labels_layer = display.add_labels(
                np.ones((8, 16, 32), dtype=np.int32), name="Flatmap Region Labels",
                metadata=metadata.copy(),
            )
            widget._region_surfaces_layers = [
                display.add_surface(
                    (vertices + offset, faces, np.ones(4)),
                    name=f"Flatmap Region Surfaces: MOs{index}", opacity=0.3,
                    metadata=metadata.copy(),
                ) for index, offset in enumerate((0, 2))
            ]
            widget._region_outlines_layers = [
                display.add_vectors(
                    vectors + offset, name=f"Flatmap Region Outlines: MOs{index}",
                    metadata=metadata.copy(),
                ) for index, offset in enumerate((0, 2))
            ]
            widget._soma_layer = display.add_points(
                [[4, 8, 16]], name="Isocortex Flatmap Somas", size=2,
                metadata=metadata.copy(),
            )
            display.reset_view()
            owner._on_flatmap_display_viewer_ready(display, image)
            # Screenshots make the secondary canvas current and can mask this
            # bug. Let the native application loop render without readbacks.
            later(300, open_popup)

        def open_popup():
            global previous_layers
            if changes == 6:
                print("Verified six full-plugin style changes and owning-context cleanups", flush=True)
                # Window dismissal has its own manual test; this child tests
                # live rendering without invoking secondary-window shutdown.
                os._exit(0)
            previous_layers = list(display.layers)
            main.window._qt_window.raise_()
            main.window._qt_window.activateWindow()
            widget._style_combo.showPopup()
            later(100, choose_style)

        def choose_style():
            global changes
            combo = widget._style_combo
            target = 1 - combo.currentIndex()
            rect = combo.view().visualRect(combo.model().index(target, 0))
            QTest.mouseClick(combo.view().viewport(), Qt.LeftButton, pos=rect.center())
            assert combo.currentIndex() == target
            assert list(display.layers) == previous_layers
            assert all(layer.visible for layer in previous_layers)
            assert widget._flatmap_render_mode_change_pending
            changes += 1
            later(150, project)

        def project():
            # This reproduces a main-window button press after its popup has
            # closed. The wrong current context must be corrected by cleanup.
            main.window._qt_viewer.canvas._scene_canvas.set_current()
            widget._project()
            assert list(main.layers) == [reference]
            assert reference.visible

        # Replace only the data-query worker; use the real Project callback,
        # queued layer deletion, viewer factory, rendering, and Qt event loop.
        widget._start_precomputed_heatmap_worker = lambda: later(200, populate)
        later(300, populate)
        later(60000, lambda: os._exit(2))
        napari.run()
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

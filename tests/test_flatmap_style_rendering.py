"""Check flatmap style changes against a real Qt/VisPy desktop canvas."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(
    os.environ.get("NN_RUN_RENDERING_TESTS") != "1",
    reason="Requires desktop OpenGL; run pixi run test-rendering",
)
def test_style_dropdown_retires_rendered_overlays_without_crashing():
    # Native crashes must stay isolated from pytest and from Qt stubs installed
    # by the widget unit tests. The child opens two real napari windows.
    script = textwrap.dedent("""
        import os
        import sys
        import napari
        import numpy as np
        from qtpy.QtCore import Qt
        from qtpy.QtTest import QTest
        from qtpy.QtWidgets import QApplication
        from napari_neuron_navigator.widgets.flatmap import FlatmapProjectionWidget

        main = napari.Viewer(title="Flatmap regression: controls")
        display = napari.Viewer(ndisplay=3, title="Flatmap regression: projection")
        for viewer in (main, display):
            viewer.window._qt_viewer.canvas.native.setFixedSize(640, 480)
        reference = main.add_image(np.arange(48).reshape(6, 8), name="Reference")
        widget = FlatmapProjectionWidget(
            main,
            database_provider=lambda: None,
            selected_file_ids_provider=lambda: [],
            table_file_ids_provider=lambda: [],
            color_map_provider=lambda: {},
            display_viewer_provider=lambda **kwargs: display,
        )
        main.window.add_dock_widget(widget, name="Flatmap regression controls")
        main.window._qt_window.resize(1200, 800)
        QTest.qWait(300)
        original_camera = (main.scene.camera.center, main.scene.camera.zoom)

        grid = np.indices((8, 16, 32))
        volume = np.exp(-sum((axis / size - 0.5)**2 for axis, size in zip(
            grid, (8, 16, 32)
        )) * 12).astype(np.float32)
        vertices = np.array([[1, 2, 3], [6, 2, 3], [1, 12, 3], [1, 2, 22]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        vectors = np.array([[[1, 2, 3], [5, 0, 0]], [[1, 2, 3], [0, 10, 0]]], dtype=float)

        def populate():
            widget._projection_layer = display.add_image(
                volume, name="Isocortex Flatmap Heatmap", colormap="magma"
            )
            widget._region_labels_layer = display.add_labels(
                (volume > 0.8).astype(np.int32), name="Flatmap Region Labels"
            )
            widget._region_surfaces_layers = [
                display.add_surface(
                    (vertices + offset, faces, np.ones(4)),
                    name=f"Flatmap Region Surfaces: MOs{index}", opacity=0.3,
                )
                for index, offset in enumerate((0, 2))
            ]
            widget._region_outlines_layers = [
                display.add_vectors(
                    vectors + offset, name=f"Flatmap Region Outlines: MOs{index}"
                )
                for index, offset in enumerate((0, 2))
            ]
            widget._soma_layer = display.add_points(
                [[4, 8, 16]], name="Isocortex Flatmap Somas", size=2
            )
            display.reset_view()
            QTest.qWait(150)
            frame = display.screenshot(canvas_only=True)
            assert np.ptp(frame[..., :3]) > 0, "No rendered scene"
            return list(display.layers)

        for target_index in (1, 0, 1, 0, 1, 0):
            old_layers = populate()
            combo = widget._style_combo
            combo.showPopup()
            QTest.qWait(50)
            view = combo.view()
            rect = view.visualRect(combo.model().index(target_index, 0))
            QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=rect.center())
            QTest.qWait(200)
            QApplication.processEvents()
            assert combo.currentIndex() == target_index
            assert len(display.layers) == 0, "Stale style layers survived"
            assert all(not layer.visible for layer in old_layers)
            assert list(main.layers) == [reference]
            assert reference.visible
            assert (main.scene.camera.center, main.scene.camera.zoom) == original_camera

        populate()
        QTest.qWait(300)
        print("Verified six live style switches with labels, surfaces, outlines, and somas")
        sys.stdout.flush()
        # Test window closing separately. End this isolated renderer process
        # without exercising the secondary-viewer shutdown path on macOS.
        os._exit(0)
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

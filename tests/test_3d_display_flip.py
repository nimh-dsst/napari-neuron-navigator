"""Exercise the AP/ML presentation correction with a real Qt/VisPy canvas.

Run outside the sandbox: pixi run test tests/test_3d_display_flip.py
"""

from copy import deepcopy

import numpy as np
import pytest


@pytest.mark.parametrize("synced", [True, False])
@pytest.mark.parametrize("initial_ndisplay", [2, 3])
def test_3d_correction_preserves_layers_and_2d_view(tmp_path, synced, initial_ndisplay):
    import napari
    from qtpy.QtWidgets import QApplication, QWidget

    from napari_neuron_navigator.widgets.neuron_viewer import NeuronViewerWidget

    viewer = napari.Viewer(ndisplay=initial_ndisplay)
    viewer.window._qt_viewer.canvas.native.setFixedSize(640, 480)
    original_orientation = viewer.scene.camera.orientation
    # Exercise the actual Visualization controls without loading an atlas or
    # starting the plugin's unrelated background workers.
    widget = NeuronViewerWidget.__new__(NeuronViewerWidget)
    QWidget.__init__(widget)
    widget.viewer = viewer
    widget._current_neuron_layers = []
    widget._3d_flip_original_orientation = None
    widget._3d_flip_axis_order = None
    widget._setup_viz_tab(widget)
    viewer.dims.events.ndisplay.connect(widget._on_ndisplay_changed)
    viewer.dims.events.order.connect(widget._update_3d_display_flip)

    def screenshot(name):
        QApplication.processEvents()
        QApplication.processEvents()
        return viewer.screenshot(path=str(tmp_path / f"{name}.png"))

    def red_center(frame):
        rgb = frame[..., :3].astype(float)
        selected = (rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
        rows, columns = np.nonzero(selected)
        assert len(rows) > 10
        return np.array([columns.mean(), rows.mean()])

    try:
        assert widget._flip_3d_display_cb.isChecked()
        assert viewer.scene.camera.orientation == (
            ("away", "down", "left") if initial_ndisplay == 3 else original_orientation
        )
        viewer.dims.ndisplay = 2
        with viewer._layer_slicer.force_sync():
            image = np.zeros((24, 32, 40), dtype=np.uint8)
            image[:, 4:28, 4:36] = 40
            viewer.add_image(image, contrast_limits=(0, 255))
            viewer.add_points([[12, 10, 9]], size=4, face_color="red", border_width=0)
            vertices = np.array([[12, 18, 26], [12, 18, 33], [12, 25, 26]])
            viewer.add_surface(
                (vertices, np.array([[0, 1, 2]])), colormap="cyan", shading="none"
            )
            viewer.dims.set_point(0, 12)
            viewer.scene.camera.synced = synced
            viewer.scene.camera.center = (12, 16, 20)
            viewer.scene.camera.zoom = 10
            assert viewer.scene.camera.orientation == original_orientation
            layers_before = [
                (
                    deepcopy(layer.data),
                    layer.scale.copy(),
                    layer.translate.copy(),
                    layer.affine.affine_matrix.copy(),
                )
                for layer in viewer.layers
            ]
            before_2d = screenshot("before_2d")

            # Enabled by default, but 2D remains unchanged. Disabling and
            # re-enabling in 2D takes effect only when entering 3D.
            widget._flip_3d_display_cb.setChecked(False)
            np.testing.assert_array_equal(screenshot("disabled_2d"), before_2d)
            widget._flip_3d_display_cb.setChecked(True)
            np.testing.assert_array_equal(screenshot("enabled_2d"), before_2d)
            assert viewer.scene.camera.orientation == original_orientation

            viewer.dims.ndisplay = 3
            assert viewer.scene.camera.orientation == ("away", "down", "left")
            widget._flip_3d_display_cb.setChecked(False)
            viewer.scene.camera.angles = (0, 0, 0)
            viewer.scene.camera.center = (12, 16, 20)
            viewer.scene.camera.zoom = 10
            before_3d = screenshot("before_3d")
            center = viewer.scene.camera.center
            zoom = viewer.scene.camera.zoom
            widget._flip_3d_display_cb.setChecked(True)
            mirrored_3d = screenshot("mirrored_3d")
            assert viewer.scene.camera.center == center
            assert viewer.scene.camera.zoom == zoom
            assert viewer.scene.camera.orientation == ("away", "down", "left")
            # The off-center marker crosses to the opposite horizontal side.
            before = red_center(before_3d)
            after = red_center(mirrored_3d)
            assert abs(before[0] - after[0]) > 100
            assert abs(before[1] - after[1]) < 2

            # Turning off restores the exact rendered scene.
            widget._flip_3d_display_cb.setChecked(False)
            np.testing.assert_array_equal(screenshot("restored_3d"), before_3d)
            widget._flip_3d_display_cb.setChecked(True)

            # Oblique rotation, repeated 2D/3D switches, and later-added layers
            # must not accumulate reflections or change the scene geometry.
            viewer.scene.camera.angles = (20, 30, 40)
            oblique = screenshot("oblique_3d")
            for _ in range(3):
                viewer.dims.ndisplay = 2
                assert viewer.scene.camera.orientation == original_orientation
                np.testing.assert_array_equal(screenshot("returned_2d"), before_2d)
                viewer.dims.ndisplay = 3
                np.testing.assert_array_equal(screenshot("returned_3d"), oblique)

            # AP must still flip when it becomes the horizontal axis in a
            # sagittal view. Reordering must never substitute a DV flip.
            expected_by_order = {
                (0, 1, 2): ("away", "down", "left"),
                (0, 2, 1): ("away", "up", "right"),
                (1, 0, 2): ("towards", "up", "left"),
                (1, 2, 0): ("towards", "up", "left"),
                (2, 0, 1): ("away", "up", "right"),
                (2, 1, 0): ("away", "down", "left"),
            }
            for order, expected in expected_by_order.items():
                viewer.dims.order = order
                assert viewer.scene.camera.orientation == expected
            viewer.scene.camera.angles = (0, 0, 0)
            viewer.scene.camera.center = (20, 16, 16)
            widget._flip_3d_display_cb.setChecked(False)
            before_ap = red_center(screenshot("before_ap_flip"))
            widget._flip_3d_display_cb.setChecked(True)
            after_ap = red_center(screenshot("after_ap_flip"))
            assert abs(before_ap[0] - after_ap[0]) > 100
            assert abs(before_ap[1] - after_ap[1]) < 2
            for layer, (data, scale, translate, affine) in zip(
                viewer.layers, layers_before, strict=True
            ):
                if isinstance(data, tuple):
                    for actual, expected in zip(layer.data, data, strict=True):
                        np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_array_equal(layer.data, data)
                np.testing.assert_array_equal(layer.scale, scale)
                np.testing.assert_array_equal(layer.translate, translate)
                np.testing.assert_array_equal(layer.affine.affine_matrix, affine)
            added = viewer.add_points([[12, 15, 30]], size=4)
            np.testing.assert_array_equal(added.data, [[12, 15, 30]])
            assert viewer.scene.camera.orientation[2] != original_orientation[2]
            widget._restore_3d_display_flip()
            assert viewer.scene.camera.orientation == original_orientation
            print(f"Display-flip screenshots: {tmp_path}")
    finally:
        viewer.dims.events.ndisplay.disconnect(widget._on_ndisplay_changed)
        viewer.dims.events.order.disconnect(widget._update_3d_display_flip)
        widget._restore_3d_display_flip()
        widget.close()
        viewer.close()

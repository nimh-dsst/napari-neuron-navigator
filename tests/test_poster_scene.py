"""Screenshot calibration must leave the CCF scene and live projections alone."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel

SCRIPT = Path(__file__).resolve().parents[1] / "poster" / "setup_poster_scene.py"
SETUP = runpy.run_path(str(SCRIPT))
configure = SETUP["configure_poster_scene"]


@pytest.fixture
def viewer():
    model = ViewerModel(ndisplay=3)
    try:
        with model._layer_slicer.force_sync():
            yield model
    finally:
        state = getattr(model, SETUP["_STATE_KEY"], None)
        if state is not None:
            state.remove()
        model._layer_slicer.shutdown()


def _renderer(bar, zoom):
    visual = SETUP["VispyVoxelMicronScaleBar"].__new__(
        SETUP["VispyVoxelMicronScaleBar"]
    )
    visual.overlay = bar
    visual.viewer = SimpleNamespace(
        scene=SimpleNamespace(camera=SimpleNamespace(zoom=zoom))
    )
    visual.node = SimpleNamespace(text=SimpleNamespace(text=""))
    visual._target_length = 150.0
    visual._on_rendering_change = lambda: None
    visual._on_unit_change()
    return visual


@pytest.mark.parametrize("zoom", [0.5, 1.0, 4.0, 12.0])
def test_default_bar_is_forty_voxels_for_one_millimeter(zoom):
    bar = SETUP["VoxelMicronScaleBar"]()
    visual = _renderer(bar, zoom)
    assert visual._current_length == pytest.approx(40 * zoom)
    assert visual.node.text.text == "1 mm"
    # Changing a fixed length at the same zoom must update immediately.
    bar.length = 500
    visual._on_size_or_zoom_change()
    assert visual._current_length == pytest.approx(20 * zoom)
    assert visual.node.text.text == "500 µm"


@pytest.mark.parametrize("zoom", [0.5, 1.0, 4.0, 12.0])
def test_automatic_label_matches_drawn_physical_distance(zoom):
    bar = SETUP["VoxelMicronScaleBar"](length=None, voxel_size_um=25)
    visual = _renderer(bar, zoom)
    import pint

    physical_length = pint.get_application_registry()(visual.node.text.text).to("um")
    assert physical_length.magnitude == pytest.approx(
        visual._current_length / zoom * 25
    )


def test_setup_rerun_and_new_layers_preserve_geometry_and_camera(viewer):
    image = viewer.add_image(np.zeros((20, 30, 40), dtype=np.uint8))
    points = viewer.add_points(
        [[100, 200, 300]], scale=(0.04,) * 3, translate=(2, 3, 4)
    )
    viewer.scene.camera.center = (5, 10, 15)
    viewer.scene.camera.zoom = 3.5
    viewer.scene.camera.angles = (20, 30, 40)
    snapshot = [
        (layer.scale.copy(), layer.translate.copy(), layer.units)
        for layer in viewer.layers
    ]
    camera = viewer.scene.camera.model_dump()
    point = viewer.dims.point
    original_bar = viewer.canvas.overlays.scale_bar

    state = configure(viewer)
    assert configure(viewer) is state
    # A second console run creates new Python classes; it must still reuse state.
    second_run = runpy.run_path(str(SCRIPT))
    assert second_run["configure_poster_scene"](viewer) is state
    assert viewer.scene.camera.model_dump() == camera
    assert viewer.dims.point == point
    for layer, (scale, translate, units) in zip(viewer.layers, snapshot):
        np.testing.assert_array_equal(layer.scale, scale)
        np.testing.assert_array_equal(layer.translate, translate)
        assert layer.units == units
        assert layer.axis_labels == ("AP", "DV", "ML")
    np.testing.assert_array_equal(points.data, [[100, 200, 300]])
    assert viewer.scene.overlays.axes.visible
    assert viewer.dims.axis_labels == ("AP", "DV", "ML")

    for _ in range(3):
        layer = viewer.add_points([[200, 300, 400]], scale=(0.04,) * 3)
        assert layer.axis_labels == ("AP", "DV", "ML")
        np.testing.assert_array_equal(layer.scale, (0.04,) * 3)
        viewer.layers.remove(layer)

    state.remove()
    state.remove()  # Safe repeated cleanup.
    assert viewer.canvas.overlays.scale_bar is original_bar
    assert image.axis_labels == ("-3", "-2", "-1")
    assert points.axis_labels == ("-3", "-2", "-1")


@pytest.mark.parametrize("kind", ["microns", "flatmap"])
def test_incompatible_scene_is_rejected_before_changes(viewer, kind):
    kwargs = (
        {"units": "um"}
        if kind == "microns"
        else {"metadata": {"napari_neuron_navigator_space": "flatmap"}}
    )
    image = viewer.add_image(np.zeros((2, 3, 4), dtype=np.uint8), **kwargs)
    bar = viewer.canvas.overlays.scale_bar
    labels = image.axis_labels
    with pytest.raises(ValueError):
        configure(viewer)
    assert viewer.canvas.overlays.scale_bar is bar
    assert image.axis_labels == labels


def test_real_canvas_soma_and_line_projections_survive_setup_and_recreation(tmp_path):
    """Run outside the sandbox: exercise real Qt/Vispy and both projectors."""
    import napari
    from qtpy.QtWidgets import QApplication

    from napari_neuron_navigator.widgets.slice_projection import (
        NeuronSliceProjector,
        SomaSliceProjector,
    )

    gui = napari.Viewer()
    soma = SomaSliceProjector(gui, tolerance=50)
    lines = NeuronSliceProjector(gui, tolerance=50)
    try:
        with gui._layer_slicer.force_sync():
            background = np.zeros((64, 96, 96), dtype=np.uint8)
            background[:, 20:80, 20:80] = 80
            gui.add_image(background, name="Allen Template", contrast_limits=(0, 255))
            gui.dims.set_point(0, 20)
            soma.set_scale([0.04] * 3)
            lines.set_scale([0.04] * 3)
            soma.add_soma_data("file-a", np.array([[500.0, 1000.0, 1500.0]]))
            lines.add_neuron_data(
                "file-a",
                np.array([[500.0, 1000.0, 1500.0], [500.0, 1500.0, 1000.0]]),
                np.array([[0, 1]]),
            )
            soma.enabled = lines.enabled = True
            soma._do_update_projection()
            lines._do_update_projection()
            before = soma.projection_layer.data.copy()

            state = configure(gui)
            # Reloading the script updates an existing scene's renderer too.
            reloaded = runpy.run_path(str(SCRIPT))
            assert reloaded["configure_poster_scene"](gui) is state
            gui.scene.camera.zoom = 4
            QApplication.processEvents()
            visuals = gui.window._qt_viewer.canvas._viewer_overlay_to_visual[
                state.scale_bar
            ]
            assert len(visuals) == 1
            assert visuals[0].node.text.text == "1 mm"
            assert visuals[0]._current_length == pytest.approx(160)

            # Repeat the plugin's normal render reset, which broke the old snippet.
            for projector in (soma, lines):
                projector.set_scale([0.04] * 3)
                projector._remove_projection_layer()
                projector._do_update_projection()
                assert projector._projection_layer.axis_labels == ("AP", "DV", "ML")
            np.testing.assert_array_equal(soma.projection_layer.data, before)
            assert len(soma.projection_layer._view_data) == 1

            # Empty slab, back to populated slab, and 2D/3D changes.
            for step, expected in [(40, 0), (21, 1), (20, 1)]:
                gui.dims.set_point(0, step)
                soma._do_update_projection()
                lines._do_update_projection()
                assert len(soma.projection_layer._view_data) == expected
            gui.dims.ndisplay = 3
            QApplication.processEvents()
            gui.dims.ndisplay = 2
            gui.dims.set_point(0, 20)
            soma._do_update_projection()
            lines._do_update_projection()
            QApplication.processEvents()
            assert len(soma.projection_layer._view_data) == 1
            assert gui.dims.axis_labels == ("AP", "DV", "ML")
            screenshot = gui.screenshot(path=str(tmp_path / "poster_scene.png"))
            assert screenshot.ndim == 3
            assert screenshot[..., :3].max() > screenshot[..., :3].min()
            print(f"Screenshot for inspection: {tmp_path / 'poster_scene.png'}")
            state.remove()
    finally:
        soma.cleanup()
        lines.cleanup()
        gui.close()

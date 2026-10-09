"""Set up screenshot overlays in a fresh Neuron Navigator 25 um CCF scene.

Run in the MAIN atlas viewer's napari Python console (not a terminal):

    %run -i /Users/lawrimorejg/repos/napari-neuron-navigator/poster/setup_poster_scene.py

Run before or after loading/rendering neurons. Leave this setup active while
scrolling slices, switching 2D/3D, adding regions, and re-rendering neurons.
Running it again is safe. Start a fresh viewer if you used the earlier snippets
that multiplied layer scales by 25; this script deliberately does not undo them.

Defaults: a fixed 1 mm bar and AP / DV / ML scene-axis labels.
Change the constants below, or adjust the bar afterwards in the console:

    viewer.canvas.overlays.scale_bar.length = 1000  # 1 mm; input is micrometers
    viewer.canvas.overlays.scale_bar.length = None  # automatic length
    viewer.canvas.overlays.scale_bar.font_size = 18
    viewer.canvas.overlays.axes.visible = True  # optional corner axes

Capture with napari's screenshot command or, for example:

    viewer.screenshot(path="/absolute/path/to/figure.png", canvas_only=True)

The bar converts world distances using 25 um per atlas voxel. A 1 mm bar is
40 world voxels long. Layer scales, units, data, slice positions, camera framing,
and projector caches stay untouched. This is display calibration for screenshots,
not a change to the coordinate system used for measurements or exports.

Only use with the isotropic 25 um CCF atlas in its original voxel coordinates.
Flatmaps have no equivalent uniform physical calibration. In 3D use orthographic
viewing for a uniform scale; perspective gives different scales at different depths.
The custom overlay uses napari 0.9's internal renderer API and checks that version.

To remove this setup and restore the previous overlays/labels:

    poster_scene.remove()
"""

from __future__ import annotations

import math
from weakref import WeakKeyDictionary

import napari
import pint
from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari._vispy.utils.visual import overlay_to_visual
from napari.components.overlays import ScaleBarOverlay
from napari.utils.events import disconnect_events
from pydantic import Field

VOXEL_SIZE_UM = 25.0
SCALE_BAR_LENGTH_UM = 1000.0  # 1 mm. Use None for an automatic length.
AXIS_LABELS = ("AP", "DV", "ML")
SCALE_BAR_FONT_SIZE = 16

_STATE_KEY = "_neuron_navigator_poster_scene"


class VoxelMicronScaleBar(ScaleBarOverlay):
    """Scale-bar length is in micrometers; the scene remains in atlas voxels."""

    voxel_size_um: float = Field(default=25.0, gt=0, allow_inf_nan=False)
    length: float | None = Field(default=SCALE_BAR_LENGTH_UM, gt=0, allow_inf_nan=False)

    # napari 0.9's renderer still connects to events.unit. Psygnal does not
    # recreate inherited property signals on subclasses, so declare it here.
    @property
    def unit(self):
        return None

    @unit.setter
    def unit(self, value):
        raise AttributeError("Use voxel_size_um to calibrate this scale bar.")


class VispyVoxelMicronScaleBar(VispyScaleBarOverlay):
    """Calibrate just this overlay, without changing any layer transforms."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.overlay.events.voxel_size_um.connect(self._on_unit_change)

    def _on_unit_change(self):
        self._unit = pint.get_application_registry().Quantity(
            self.overlay.voxel_size_um, "um"
        )
        self._on_size_or_zoom_change(force=True)

    def _on_size_or_zoom_change(self, *, force=False):
        # Recalculate on length edits too, even if the zoom has not changed.
        self._scale = 1 / self.viewer.scene.camera.zoom
        if self.overlay.length is None:
            # The inherited rounding routine handles the 25 um/world-voxel
            # quantity, including compact automatic labels such as "1 mm".
            world_length, quantity = self._calculate_best_length(
                self._target_length * self._scale
            )
        else:
            world_length = self.overlay.length / self.overlay.voxel_size_um
            quantity = pint.get_application_registry().Quantity(
                self.overlay.length, "um"
            )
        self._current_length = world_length / self._scale
        self.node.text.text = f"{quantity:g~#P}"
        self._on_rendering_change()

    def close(self):
        disconnect_events(self.viewer.scene.camera.events, self)
        disconnect_events(self.viewer.dims.events, self)
        for callback in (
            self._on_unit_change,
            self._on_size_or_zoom_change,
            self._on_rendering_change,
            self._on_font_size_change,
            self._on_position_change,
            self._on_box_change,
        ):
            self.overlay.events.disconnect(callback)
        super().close()


def _validate_layer(layer):
    if layer.metadata.get("napari_neuron_navigator_space") == "flatmap":
        raise ValueError(
            "Run this script in the main CCF atlas viewer, not the flatmap."
        )
    if layer.ndim != 3:
        raise ValueError(
            f"Expected a 3D CCF layer; {layer.name!r} has {layer.ndim} axes."
        )
    if any(str(unit) not in {"pixel", "dimensionless"} for unit in layer.units):
        raise ValueError(
            f"{layer.name!r} already has physical units. Start a fresh voxel-space "
            "scene without the earlier scale-conversion snippets."
        )


class PosterScene:
    """Keep anatomical captions on existing and newly rendered CCF layers."""

    def __init__(self, viewer, *, voxel_size_um, length_um, axis_labels, font_size):
        self.viewer = viewer
        self.axis_labels = tuple(axis_labels)
        self._original_labels = WeakKeyDictionary()
        self._previous_bar = viewer.canvas.overlays.scale_bar
        self._previous_axes_visible = viewer.scene.overlays.axes.visible
        self._previous_axes_labels = viewer.scene.overlays.axes.labels
        self._previous_dims_labels = viewer.dims.axis_labels
        self._removed = False

        overlay_to_visual[VoxelMicronScaleBar] = VispyVoxelMicronScaleBar
        self.scale_bar = VoxelMicronScaleBar(
            voxel_size_um=voxel_size_um,
            length=length_um,
            font_size=font_size,
            visible=True,
            position="bottom_right",
            box=False,
        )
        viewer.canvas.overlays["scale_bar"] = self.scale_bar
        for layer in viewer.layers:
            self._label_layer(layer)
        viewer.scene.overlays.axes.visible = True
        viewer.scene.overlays.axes.labels = True
        viewer.layers.events.inserted.connect(self._on_layer_added)

    def _label_layer(self, layer):
        _validate_layer(layer)
        self._original_labels.setdefault(layer, layer.axis_labels)
        layer.axis_labels = self.axis_labels

    def _on_layer_added(self, event):
        try:
            self._label_layer(event.value)
        except ValueError as exc:
            # An incompatible new layer must never leave a plausible-looking
            # but incorrect physical scale bar in a screenshot.
            self.scale_bar.visible = False
            from napari.utils.notifications import show_warning

            show_warning(f"Poster scale bar hidden: {exc}")

    def remove(self):
        """Restore labels and the original napari scale-bar overlay."""
        if self._removed:
            return
        self.viewer.layers.events.inserted.disconnect(self._on_layer_added)
        for layer in self.viewer.layers:
            if layer in self._original_labels:
                layer.axis_labels = self._original_labels[layer]
        if len(self._previous_dims_labels) == self.viewer.dims.ndim:
            self.viewer.dims.axis_labels = self._previous_dims_labels
        self.viewer.canvas.overlays["scale_bar"] = self._previous_bar
        self.viewer.scene.overlays.axes.visible = self._previous_axes_visible
        self.viewer.scene.overlays.axes.labels = self._previous_axes_labels
        self.viewer.__dict__.pop(_STATE_KEY, None)
        self._removed = True


def configure_poster_scene(
    viewer,
    *,
    voxel_size_um=VOXEL_SIZE_UM,
    length_um=SCALE_BAR_LENGTH_UM,
    axis_labels=AXIS_LABELS,
    font_size=SCALE_BAR_FONT_SIZE,
):
    """Install/reconfigure the overlays without rescaling the scene."""
    if viewer is None:
        raise RuntimeError(
            "Run this script from the main atlas viewer's Python console."
        )
    if napari.__version__.split(".")[:2] != ["0", "9"]:
        raise RuntimeError("This screenshot overlay supports napari 0.9.x.")
    if not math.isfinite(voxel_size_um) or voxel_size_um <= 0:
        raise ValueError("voxel_size_um must be finite and positive.")
    if length_um is not None and (not math.isfinite(length_um) or length_um <= 0):
        raise ValueError("length_um must be None or finite and positive.")
    if len(axis_labels) != 3:
        raise ValueError("Supply three labels in CCF data-axis order: AP, DV, ML.")
    for layer in viewer.layers:
        _validate_layer(layer)

    existing = getattr(viewer, _STATE_KEY, None)
    if existing is not None:
        # A console rerun defines new classes. Replace the old renderer so
        # updates to label formatting take effect in an already-open scene.
        if type(existing.scale_bar) is not VoxelMicronScaleBar:
            overlay_to_visual[VoxelMicronScaleBar] = VispyVoxelMicronScaleBar
            existing.scale_bar = VoxelMicronScaleBar(**existing.scale_bar.model_dump())
            # Equal model values suppress napari's dict change event, even
            # though the new class needs a new renderer.
            viewer.canvas.overlays.pop("scale_bar")
            viewer.canvas.overlays["scale_bar"] = existing.scale_bar
        existing.scale_bar.voxel_size_um = voxel_size_um
        existing.scale_bar.length = length_um
        existing.scale_bar.font_size = font_size
        existing.scale_bar.visible = True
        existing.axis_labels = tuple(axis_labels)
        for layer in viewer.layers:
            existing._label_layer(layer)
        viewer.scene.overlays.axes.visible = True
        viewer.scene.overlays.axes.labels = True
        return existing

    state = PosterScene(
        viewer,
        voxel_size_um=voxel_size_um,
        length_um=length_um,
        axis_labels=axis_labels,
        font_size=font_size,
    )
    viewer.__dict__[_STATE_KEY] = state
    return state


if __name__ == "__main__":
    poster_scene = configure_poster_scene(
        globals().get("viewer") or napari.current_viewer()
    )
    print(
        "Poster scene ready: 25 um/voxel scale bar; AP / DV / ML axes. "
        "Layer coordinates and slice projections are unchanged."
    )

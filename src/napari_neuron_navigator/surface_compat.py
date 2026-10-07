"""Local compatibility for napari 0.9's fully displayed surface axes.

In napari 0.9.0–0.9.2, ``_SurfaceSliceRequest`` returns canonical vertices
without reordering them when no dimensions are sliced away. Keep the source
geometry in atlas coordinates and correct only the display response. All
private napari API access for this workaround lives here; a behavior probe
disables it when upstream already handles the permutation.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import permutations
from types import MethodType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from napari.components import ViewerModel
    from napari.layers import Surface


_INSTALLED = "_neuron_navigator_axis_order_fix"


@lru_cache(maxsize=1)
def _needs_surface_axis_order_fix() -> bool:
    """Distinguish the known defect from corrected or unexpected behavior."""
    # Import lazily: importing the plugin must not initialize napari layers.
    from napari.components import Dims
    from napari.layers import Surface

    vertices = np.asarray(
        [[1, 2, 3], [4, 2, 3], [1, 6, 3], [1, 2, 8]], dtype=np.float32
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    correct = []
    canonical = []
    try:
        layer = Surface((vertices, faces))
        for order in permutations(range(3)):
            layer._slice_dims(Dims(ndim=3, ndisplay=3, order=order))
            correct.append(np.array_equal(layer._view_vertices, vertices[:, order]))
            canonical.append(np.array_equal(layer._view_vertices, vertices))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot verify napari Surface axis-order compatibility. "
            "The surface slicing API has changed."
        ) from exc

    if all(correct):
        return False
    if all(canonical):
        return True
    raise RuntimeError(
        "Unexpected napari Surface axis-order behavior; "
        "cannot safely apply the Neuron Navigator compatibility fix."
    )


def _install_surface_axis_order_fix(layer: Surface) -> None:
    """Wrap one affected layer's sync/async response handler exactly once."""
    state = layer._slicing_state
    if getattr(state, _INSTALLED, False):
        return
    original_update = state._update_slice_response

    def update_slice_response(_state, response):
        slice_input = response.slice_input
        # A 2D slice already orders its columns correctly. Extra value axes
        # also take napari's normal slicing path and must remain untouched.
        if (
            slice_input.ndisplay == 3
            and not slice_input.not_displayed
            and response.vertices.ndim == 2
            and response.vertices.shape[1] == 3
        ):
            order = tuple(slice_input.displayed)
            if order != (0, 1, 2):
                response = replace(response, vertices=response.vertices[:, order])
        original_update(response)

    state._update_slice_response = MethodType(update_slice_response, state)
    setattr(state, _INSTALLED, True)
    # Apply to the initial view as well as future responses. In async mode,
    # refresh requests another slice; pending responses use this handler too.
    layer.refresh()


def add_surface_with_axis_order(viewer: ViewerModel, data, **kwargs) -> Surface:
    """Add an ordinary Surface that follows the viewer's displayed axis order.

    Only surfaces created through this helper receive the compatibility fix.
    Coordinates, transforms, metadata, and serialization remain native napari
    state. No viewer event connections or global patches are installed.
    """
    needs_fix = _needs_surface_axis_order_fix()
    layer = viewer.add_surface(data, **kwargs)
    if needs_fix:
        try:
            _install_surface_axis_order_fix(layer)
        except Exception:
            # Do not leave a misleading, uncorrected mesh in the scene.
            viewer.layers.remove(layer)
            raise
    return layer

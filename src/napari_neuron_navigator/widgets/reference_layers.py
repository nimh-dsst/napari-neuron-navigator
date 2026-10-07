"""Allen template and region mesh rendering for napari.

This module provides functions to add Allen CCF reference images and
brain region meshes to a napari viewer.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING

import numpy as np

from ..logging_utils import startup_timing
from ..region_appearance import RegionAppearanceStore, structure_catalog
from ..surface_compat import add_surface_with_axis_order

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

    from ..isocortex_layers import CustomRegionSelectionGroup

logger = logging.getLogger(__name__)


def _region_appearance(
    atlas: BrainGlobeAtlas,
    region_id: int,
    appearance_store: RegionAppearanceStore | None,
    structures=None,
):
    store = appearance_store or RegionAppearanceStore()
    return store.resolve(
        int(region_id),
        structures if structures is not None else getattr(atlas, "structures", None),
    )


def _region_fill_rgba(
    atlas: BrainGlobeAtlas,
    region_id: int,
    appearance_store: RegionAppearanceStore | None,
    structures=None,
) -> np.ndarray:
    return _region_appearance(
        atlas,
        region_id,
        appearance_store,
        structures,
    ).fill_rgba


def _layer_base_visible(layer, *, default: bool = True) -> bool:
    """Preserve a napari visibility toggle as the global style gate."""
    current = bool(getattr(layer, "visible", default))
    base = bool(getattr(layer, "_napari_neuron_navigator_region_base_visible", default))
    previous = getattr(layer, "_napari_neuron_navigator_region_applied_visible", None)
    if previous is not None and current != bool(previous):
        base = current
        setattr(layer, "_napari_neuron_navigator_region_base_visible", base)
    return base


def _set_layer_applied_visible(layer, visible: bool) -> None:
    layer.visible = bool(visible)
    setattr(layer, "_napari_neuron_navigator_region_applied_visible", bool(visible))


def _array_startup_metadata(array) -> dict[str, object]:
    """Return compact array metadata without forcing a copy."""
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    nbytes = getattr(array, "nbytes", None)
    metadata: dict[str, object] = {}
    if shape is not None:
        metadata["shape"] = tuple(int(value) for value in shape)
    if dtype is not None:
        metadata["dtype"] = dtype
    if nbytes is not None:
        metadata["size_mb"] = float(nbytes) / (1024.0 * 1024.0)
    return metadata


def _atlas_structure_for_region_id(
    atlas: BrainGlobeAtlas,
    region_id: int,
):
    """Return one atlas structure by numeric ID across catalog variants."""
    structures = getattr(atlas, "structures", None)
    if structures is None:
        return None
    try:
        return structures[int(region_id)]
    except (KeyError, TypeError):
        pass

    items = getattr(structures, "items", None)
    if not callable(items):
        return None
    for key, structure in items():
        if not isinstance(structure, Mapping):
            continue
        try:
            structure_id = int(structure.get("id", key))
        except (TypeError, ValueError):
            continue
        if structure_id == int(region_id):
            return structure
    return None


def _atlas_region_rgba(structure) -> np.ndarray:
    """Return one structure's atlas color as float32 RGBA."""
    rgb = (
        structure.get("rgb_triplet", [128, 128, 128])
        if structure is not None
        else [128, 128, 128]
    )
    return np.asarray(
        [
            float(rgb[0]) / 255.0,
            float(rgb[1]) / 255.0,
            float(rgb[2]) / 255.0,
            1.0,
        ],
        dtype=np.float32,
    )


def add_allen_template(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    name: str = "Allen Template",
    opacity: float = 0.5,
    colormap: str = "gray",
    visible: bool = True,
) -> napari.layers.Image:
    """Add the Allen CCF template image to the viewer.

    All layers are in voxel/pixel space (not microns) to match brainrender-napari.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for the template.
    name : str, default="Allen Template"
        Name for the layer.
    opacity : float, default=0.5
        Layer opacity.
    colormap : str, default="gray"
        Colormap for the image.
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Image
        The created image layer.
    """
    with startup_timing(logger, "add_allen_template", layer=name) as timing:
        with startup_timing(
            logger,
            "add_allen_template_phase",
            phase="atlas.reference",
            layer=name,
        ) as reference_timing:
            reference = atlas.reference
            metadata = _array_startup_metadata(reference)
            reference_timing.set(**metadata)
            timing.set(**metadata)

        with startup_timing(
            logger,
            "add_allen_template_phase",
            phase="viewer.add_image",
            layer=name,
            **metadata,
        ):
            layer = viewer.add_image(
                reference,
                name=name,
                opacity=opacity,
                colormap=colormap,
                visible=visible,
                blending="additive",
            )

        return layer


def add_annotation_volume(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    name: str = "Allen Annotations",
    opacity: float = 0.3,
    visible: bool = False,
) -> napari.layers.Labels:
    """Add the Allen CCF annotation volume as a labels layer.

    All layers are in voxel/pixel space (not microns) to match brainrender-napari.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for annotations.
    name : str, default="Allen Annotations"
        Name for the layer.
    opacity : float, default=0.3
        Layer opacity.
    visible : bool, default=False
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Labels
        The created labels layer.
    """
    # Annotation volume (in voxel space)
    annotation = atlas.annotation

    layer = viewer.add_labels(
        annotation,
        name=name,
        opacity=opacity,
        visible=visible,
    )

    return layer


def add_region_segmentation(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    acronyms: list[str],
    name: str = "Region Segmentation",
    opacity: float = 0.3,
    visible: bool = True,
    appearance_store: RegionAppearanceStore | None = None,
) -> napari.layers.Labels | None:
    """Add a filtered annotation volume showing only selected brain regions.

    Each selected region (and all its descendants in the annotation hierarchy)
    is shown with the region's atlas-defined RGB color. Voxels outside the
    selected regions are transparent (label 0).

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for annotations and structure metadata.
    acronyms : list[str]
        List of region acronyms to display (parent-level).
    name : str, default="Region Segmentation"
        Name for the layer.
    opacity : float, default=0.3
        Layer opacity.
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Labels or None
        The created labels layer, or None if no valid regions found.
    """
    if not acronyms:
        return None

    annotation = atlas.annotation

    # For each selected parent acronym, collect all descendant annotation IDs
    # while retaining each ID for independent appearance control.
    all_selected_ids: set[int] = set()
    color_dict: dict[int | None, np.ndarray] = {
        None: np.array([0, 0, 0, 0], dtype=np.float32),  # unmapped: transparent
        0: np.array([0, 0, 0, 0], dtype=np.float32),  # background: transparent
    }
    appearance_catalog = structure_catalog(getattr(atlas, "structures", None))

    for acronym in acronyms:
        try:
            structure = atlas.structures[acronym]
        except KeyError:
            logger.warning(f"Region '{acronym}' not found in atlas structures")
            continue

        parent_id = structure["id"]
        # Collect all descendants: any structure whose structure_id_path
        # contains this parent_id is a descendant (or the region itself).
        descendant_ids: set[int] = set()
        for key, struct in atlas.structures.items():
            if isinstance(key, int):
                path = struct.get("structure_id_path", [])
                if parent_id in path:
                    descendant_ids.add(key)

        all_selected_ids.update(descendant_ids)

        # Keep every annotation ID so child overrides and atlas-default colors
        # remain independently controllable in the combined Labels layer.
        for did in descendant_ids:
            color_dict[did] = _region_fill_rgba(
                atlas,
                did,
                appearance_store,
                appearance_catalog,
            )

    if not all_selected_ids:
        logger.warning("No valid annotation IDs found for selected regions")
        return None

    # Build filtered annotation volume (keep only selected region voxels)
    id_array = np.array(sorted(all_selected_ids), dtype=annotation.dtype)
    mask = np.isin(annotation, id_array)
    filtered = np.where(mask, annotation, np.zeros_like(annotation))

    logger.info(
        f"Region segmentation: {len(acronyms)} regions, "
        f"{len(all_selected_ids)} annotation IDs, "
        f"{mask.sum():,} voxels"
    )

    # Use DirectLabelColormap (napari >= 0.5.0)
    from napari.utils import DirectLabelColormap

    colormap = DirectLabelColormap(color_dict=color_dict)

    layer = viewer.add_labels(
        filtered,
        name=name,
        opacity=opacity,
        visible=visible,
        colormap=colormap,
        metadata={
            "region_layer_kind": "segmentation",
            "selected_region_acronyms": list(acronyms),
            "represented_region_ids": sorted(all_selected_ids),
        },
    )

    return layer


def add_region_id_segmentation(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    region_ids: list[int] | tuple[int, ...],
    name: str = "Region Segmentation",
    opacity: float = 0.3,
    visible: bool = True,
    appearance_store: RegionAppearanceStore | None = None,
) -> napari.layers.Labels | None:
    """Show exactly the requested atlas IDs without descendant expansion."""
    selected_ids = tuple(
        sorted({int(region_id) for region_id in region_ids if int(region_id) > 0})
    )
    if not selected_ids:
        return None

    annotation = atlas.annotation
    valid_ids: list[int] = []
    color_dict: dict[int | None, np.ndarray] = {
        None: np.array([0, 0, 0, 0], dtype=np.float32),
        0: np.array([0, 0, 0, 0], dtype=np.float32),
    }
    appearance_catalog = structure_catalog(getattr(atlas, "structures", None))
    for region_id in selected_ids:
        structure = _atlas_structure_for_region_id(atlas, region_id)
        if structure is None:
            logger.warning("Region ID %d not found in atlas structures", region_id)
            continue
        valid_ids.append(region_id)
        color_dict[region_id] = _region_fill_rgba(
            atlas,
            region_id,
            appearance_store,
            appearance_catalog,
        )

    if not valid_ids:
        logger.warning("No valid annotation IDs found for exact region selection")
        return None

    id_array = np.asarray(valid_ids, dtype=annotation.dtype)
    mask = np.isin(annotation, id_array)
    filtered = np.where(mask, annotation, np.zeros_like(annotation))

    from napari.utils import DirectLabelColormap

    colormap = DirectLabelColormap(color_dict=color_dict)
    return viewer.add_labels(
        filtered,
        name=name,
        opacity=opacity,
        visible=visible,
        colormap=colormap,
        metadata={
            "region_layer_kind": "segmentation",
            "region_selection_source": "custom",
            "selected_region_ids": list(valid_ids),
            "represented_region_ids": list(valid_ids),
            "exact_region_ids": True,
        },
    )


def region_label_colormap(
    atlas: BrainGlobeAtlas,
    region_ids: list[int] | tuple[int, ...],
    appearance_store: RegionAppearanceStore | None = None,
):
    """Build a direct label colormap from the shared region appearance state."""
    from napari.utils import DirectLabelColormap

    transparent = np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    color_dict: dict[int | None, np.ndarray] = {None: transparent, 0: transparent}
    appearance_catalog = structure_catalog(getattr(atlas, "structures", None))
    for region_id in sorted({int(value) for value in region_ids if int(value) > 0}):
        color_dict[region_id] = _region_fill_rgba(
            atlas,
            region_id,
            appearance_store,
            appearance_catalog,
        )
    return DirectLabelColormap(color_dict=color_dict)


def apply_region_appearance_to_layer(
    layer,
    atlas: BrainGlobeAtlas,
    appearance_store: RegionAppearanceStore,
) -> bool:
    """Restyle one existing CCF region layer without rebuilding its data."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    kind = str(metadata.get("region_layer_kind", "") or "")
    if kind == "segmentation":
        region_ids = metadata.get("represented_region_ids", ()) or ()
        layer.colormap = region_label_colormap(
            atlas,
            [int(value) for value in region_ids],
            appearance_store,
        )
    elif kind == "mesh":
        region_id = int(metadata.get("region_id", 0) or 0)
        if region_id <= 0:
            return False
        effective = _region_appearance(atlas, region_id, appearance_store)
        vertices = np.asarray(layer.data[0])
        layer.vertex_colors = np.repeat(
            np.asarray(effective.color_rgba, dtype=np.float32)[None, :],
            len(vertices),
            axis=0,
        )
        base_opacity = float(
            getattr(layer, "_napari_neuron_navigator_region_base_opacity", layer.opacity)
        )
        base_visible = _layer_base_visible(layer)
        layer.opacity = base_opacity * effective.fill_opacity
        _set_layer_applied_visible(layer, base_visible and effective.fill_visible)
    elif kind == "mesh_group":
        region_ids = getattr(layer, "_napari_neuron_navigator_region_vertex_ids", None)
        if region_ids is None:
            return False
        normalized_ids = np.asarray(region_ids, dtype=np.int64)
        colors = np.empty((len(normalized_ids), 4), dtype=np.float32)
        any_visible = False
        for region_id in np.unique(normalized_ids):
            effective = _region_appearance(atlas, int(region_id), appearance_store)
            colors[normalized_ids == region_id] = effective.fill_rgba
            any_visible = any_visible or effective.fill_visible
        layer.vertex_colors = colors
        layer.opacity = float(
            getattr(layer, "_napari_neuron_navigator_region_base_opacity", layer.opacity)
        )
        _set_layer_applied_visible(layer, _layer_base_visible(layer) and any_visible)
    else:
        return False

    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()
    return True


def remove_region_segmentation(
    viewer: napari.Viewer,
    name: str = "Region Segmentation",
) -> bool:
    """Remove the region segmentation layer if it exists.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    name : str, default="Region Segmentation"
        The layer name to look for.

    Returns
    -------
    bool
        True if a layer was removed, False otherwise.
    """
    for layer in viewer.layers:
        if layer.name == name:
            viewer.layers.remove(layer)
            return True
    return False


def add_region_mesh(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    acronym: str,
    opacity: float = 0.4,
    color: str | tuple | None = None,
    name: str | None = None,
    visible: bool = True,
    appearance_store: RegionAppearanceStore | None = None,
) -> napari.layers.Surface | None:
    """Add a brain region mesh to the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for the mesh.
    acronym : str
        The region acronym (e.g., "VISp").
    opacity : float, default=0.4
        Mesh opacity.
    color : str or tuple, optional
        Mesh color. If None, uses the atlas's color for the region.
    name : str, optional
        Name for the layer. If None, uses "Region: {acronym}".
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Surface or None
        The created surface layer, or None if the region mesh is not available.
    """
    # Get structure info - StructuresDict supports direct acronym access via []
    try:
        structure = atlas.structures[acronym]
    except KeyError:
        logger.warning(f"Region '{acronym}' not found in atlas structures")
        return None

    # Get mesh using BrainGlobe API
    try:
        mesh = atlas.mesh_from_structure(acronym)
        logger.info(f"Loaded mesh for '{acronym}' with {len(mesh.points)} vertices")
    except (KeyError, FileNotFoundError) as e:
        logger.warning(f"Could not load mesh for '{acronym}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading mesh for '{acronym}': {e}")
        return None

    # Get vertices and faces (meshio format)
    # Convert to float32/int32 for better vispy/OpenGL compatibility
    vertices = mesh.points.astype(np.float32)
    faces = mesh.cells[0].data.astype(np.int32)

    # Scale mesh from microns to pixel/voxel space to match the reference image
    scale = [1.0 / res for res in atlas.resolution]

    # Create layer name
    if name is None:
        name = f"Region: {acronym}"

    # Determine vertex colors
    effective = _region_appearance(
        atlas,
        int(structure["id"]),
        appearance_store,
    )
    if color is None or appearance_store is not None:
        rgb = [component * 255.0 for component in effective.color_rgba[:3]]
    else:
        rgb = [int(c * 255) if isinstance(c, float) and c <= 1 else c for c in color]

    # Create vertex colors array (RGB 0-1 for each vertex, float32 for vispy)
    vertex_colors = np.repeat(
        [[float(c) / 255 for c in rgb]], len(vertices), axis=0
    ).astype(np.float32)

    logger.info(
        f"Creating surface layer '{name}': {len(vertices)} vertices, {len(faces)} faces"
    )
    layer = add_surface_with_axis_order(
        viewer,
        (vertices, faces),
        scale=scale,
        name=name,
        opacity=opacity * effective.fill_opacity,
        blending="translucent_no_depth",
        vertex_colors=vertex_colors,
        visible=visible and effective.fill_visible,
        metadata={
            "region_layer_kind": "mesh",
            "region_id": int(structure["id"]),
            "region_acronym": str(acronym),
        },
    )
    setattr(layer, "_napari_neuron_navigator_region_base_opacity", float(opacity))
    setattr(layer, "_napari_neuron_navigator_region_base_visible", bool(visible))
    setattr(layer, "_napari_neuron_navigator_region_applied_visible", bool(layer.visible))

    logger.info(f"Added region mesh layer: {layer}")
    return layer


def add_region_mesh_group(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    group: CustomRegionSelectionGroup,
    opacity: float = 0.4,
    name: str | None = None,
    visible: bool = True,
    appearance_store: RegionAppearanceStore | None = None,
) -> tuple[napari.layers.Surface | None, tuple[str, ...]]:
    """Combine selected terminal atlas meshes into one colored surface layer."""
    vertex_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    rendered_ids: list[int] = []
    rendered_acronyms: list[str] = []
    missing_acronyms: list[str] = []
    vertex_region_ids: list[np.ndarray] = []
    vertex_offset = 0

    for region_id, acronym in zip(
        group.region_ids,
        group.acronyms,
        strict=True,
    ):
        try:
            mesh = atlas.mesh_from_structure(acronym)
            vertices = np.asarray(mesh.points, dtype=np.float32)
            triangle_parts = [
                np.asarray(cell.data, dtype=np.int32)
                for cell in mesh.cells
                if np.asarray(cell.data).ndim == 2
                and np.asarray(cell.data).shape[1] == 3
            ]
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not len(vertices)
                or not triangle_parts
            ):
                raise ValueError("mesh has no usable vertices or triangle faces")
            faces = np.concatenate(triangle_parts, axis=0)
        except Exception as exc:
            logger.warning("Could not load mesh for '%s': %s", acronym, exc)
            missing_acronyms.append(str(acronym))
            continue

        vertex_parts.append(vertices)
        face_parts.append(faces + vertex_offset)
        rgba = _region_fill_rgba(atlas, int(region_id), appearance_store)
        color_parts.append(np.repeat(rgba[None, :], len(vertices), axis=0))
        vertex_region_ids.append(np.full(len(vertices), int(region_id), dtype=np.int32))
        rendered_ids.append(int(region_id))
        rendered_acronyms.append(str(acronym))
        vertex_offset += len(vertices)

    if not vertex_parts:
        return None, tuple(missing_acronyms)

    vertices = np.concatenate(vertex_parts, axis=0)
    faces = np.concatenate(face_parts, axis=0)
    vertex_colors = np.concatenate(color_parts, axis=0).astype(
        np.float32,
        copy=False,
    )
    any_visible = bool(np.any(vertex_colors[:, 3] > 0.0))
    layer_name = name or f"Region: Custom {group.label}"
    layer = add_surface_with_axis_order(
        viewer,
        (vertices, faces),
        scale=[1.0 / float(resolution) for resolution in atlas.resolution],
        name=layer_name,
        opacity=opacity,
        blending="translucent_no_depth",
        vertex_colors=vertex_colors,
        visible=visible and any_visible,
        metadata={
            "region_selection_source": "custom",
            "region_layer_kind": "mesh_group",
            "custom_region_group": group.label,
            "selected_region_ids": rendered_ids,
            "selected_region_acronyms": rendered_acronyms,
            "missing_region_acronyms": list(missing_acronyms),
        },
    )
    setattr(
        layer,
        "_napari_neuron_navigator_region_vertex_ids",
        np.concatenate(vertex_region_ids),
    )
    setattr(layer, "_napari_neuron_navigator_region_base_opacity", float(opacity))
    setattr(layer, "_napari_neuron_navigator_region_base_visible", bool(visible))
    setattr(layer, "_napari_neuron_navigator_region_applied_visible", bool(layer.visible))
    return layer, tuple(missing_acronyms)


def add_region_meshes(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    acronyms: list[str],
    opacity: float = 0.3,
    use_atlas_colors: bool = True,
    visible: bool = True,
) -> list[napari.layers.Surface]:
    """Add multiple brain region meshes to the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for meshes.
    acronyms : list[str]
        List of region acronyms to add.
    opacity : float, default=0.3
        Mesh opacity.
    use_atlas_colors : bool, default=True
        If True, use the atlas's colors for each region.
    visible : bool, default=True
        Whether the layers are visible by default.

    Returns
    -------
    list[napari.layers.Surface]
        List of created surface layers.
    """
    layers = []
    for acronym in acronyms:
        layer = add_region_mesh(
            viewer,
            atlas,
            acronym,
            opacity=opacity,
            color=None if use_atlas_colors else (0.5, 0.5, 0.5),
            visible=visible,
        )
        if layer is not None:
            layers.append(layer)

    return layers


def add_brain_outline(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    opacity: float = 0.2,
    name: str = "Brain Outline",
    visible: bool = True,
) -> napari.layers.Surface | None:
    """Add the whole brain outline mesh to the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use.
    opacity : float, default=0.2
        Mesh opacity.
    name : str, default="Brain Outline"
        Name for the layer.
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Surface or None
        The created surface layer, or None if not available.
    """
    # Get the root mesh using BrainGlobe API
    try:
        mesh = atlas.mesh_from_structure("root")
        logger.info(f"Loaded root mesh with {len(mesh.points)} vertices")
    except (KeyError, FileNotFoundError) as e:
        logger.warning(f"Could not load root mesh: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading root mesh: {e}")
        return None

    # Get vertices and faces (meshio format)
    # Convert to float32/int32 for better vispy/OpenGL compatibility
    vertices = mesh.points.astype(np.float32)
    faces = mesh.cells[0].data.astype(np.int32)

    # Scale mesh from microns to pixel/voxel space to match the reference image
    scale = [1.0 / res for res in atlas.resolution]

    # Gray color for outline (float32 for vispy)
    vertex_colors = np.repeat([[0.5, 0.5, 0.5]], len(vertices), axis=0).astype(
        np.float32
    )

    logger.info(
        f"Creating brain outline surface: {len(vertices)} vertices, {len(faces)} faces"
    )

    layer = add_surface_with_axis_order(
        viewer,
        (vertices, faces),
        scale=scale,
        name=name,
        opacity=opacity,
        blending="translucent_no_depth",
        vertex_colors=vertex_colors,
        visible=visible,
    )

    logger.info(f"Added brain outline layer: {layer}")
    return layer


def remove_region_layers(
    viewer: napari.Viewer,
    prefix: str = "Region:",
) -> int:
    """Remove all region mesh layers from the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    prefix : str, default="Region:"
        Prefix to match for layer names.

    Returns
    -------
    int
        Number of layers removed.
    """
    to_remove = [layer for layer in viewer.layers if layer.name.startswith(prefix)]

    for layer in to_remove:
        viewer.layers.remove(layer)

    return len(to_remove)

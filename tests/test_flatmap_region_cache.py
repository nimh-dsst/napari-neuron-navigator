from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import napari_neuron_navigator.flatmap_region_cache as region_cache_module
from napari_neuron_navigator.flatmap_region_cache import (
    LEGACY_REGION_CACHE_FORMAT,
    REGION_CACHE_MANIFEST_FILENAME,
    RegionCacheError,
    RegionCacheCancelled,
    RegionCacheValidationError,
    build_region_cache_profile,
    materialize_allen_layer_region_selection,
    materialize_flat_region_outlines,
    materialize_flat_region_selection,
    materialize_region_outlines,
    materialize_region_selection,
    materialize_region_surface,
    open_region_cache,
    structure_catalog_id,
)
from napari_neuron_navigator.flatmap_heatmap import resolve_flatmap_bin_counts
from napari_neuron_navigator.isocortex_layers import AllenIsocortexLayerMap


def _lookup_arrays():
    shape = (2, 2, 2)
    shaped = np.empty((*shape, 2), dtype=np.float32)
    square = np.empty((*shape, 2), dtype=np.float32)
    for first in range(shape[0]):
        for second in range(shape[1]):
            shaped[first, second, :, 0] = first
            shaped[first, second, :, 1] = second
            square[first, second, :, 0] = first * 2
            # Kept proportional to x so the square style's stats-derived bounds
            # stay ratio 1.0 and the derived x count is 2, not 1.  With 2 bins,
            # value 2 under bounds (0,2) clips to bin 1 exactly as value 3 under
            # (0,3) did, so no bin index changes -- only the recorded bounds.
            square[first, second, :, 1] = second * 2
    depth = np.zeros(shape, dtype=np.float32)
    depth[:, :, 1] = 10.0
    # Region 3 at this voxel recovers depth zero from the opposite hemisphere.
    depth[0, 0, 1] = -1.0
    # A region-4 source voxel is deliberately outside both flatmap lookups.
    shaped[1, 0, 1] = (-1.0, -1.0)
    square[1, 0, 1] = (-1.0, -1.0)
    annotation = np.array(
        [
            [[2, 3], [2, 0]],
            [[3, 4], [0, 4]],
        ],
        dtype=np.int32,
    )
    return annotation, shaped, square, depth


def _build(cache_dir: Path, **kwargs):
    annotation, shaped, square, depth = _lookup_arrays()
    return build_region_cache_profile(
        cache_dir,
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="portable-lookup-set",
        atlas_name="test_mouse",
        atlas_version="1.2",
        atlas_resolution_um=(10, 10, 10),
        region_descendants={
            1: {2, 3},
            2: {2},
            3: {3},
            4: {4},
            5: {5},
        },
        y_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
        **kwargs,
    )


def _allen_layer_map() -> AllenIsocortexLayerMap:
    return AllenIsocortexLayerMap(
        atlas_name="test_mouse",
        isocortex_region_id=1,
        region_to_layer_index={10: 0, 11: 0, 12: 1, 13: 2},
        region_ids_by_layer=((10, 11), (12,), (13,), (), (), ()),
    )


def _build_planar_cache(cache_dir: Path):
    shape = (2, 2, 2)
    shaped = np.full((*shape, 2), 0.25, dtype=np.float32)
    square = np.full((*shape, 2), 0.25, dtype=np.float32)
    depth = np.asarray(
        [
            [[0.0, 10.0], [0.0, 10.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    annotation = np.asarray(
        [
            [[10, 10], [11, 11]],
            [[12, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    bounds = {
        style: {
            "x_bounds": (0.0, 1.0),
            "y_bounds": (0.0, 1.0),
            "depth_bounds_um": (0.0, 20.0),
        }
        for style in ("shaped", "square")
    }
    return build_region_cache_profile(
        cache_dir,
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="planar-lookup-set",
        atlas_name="test_mouse",
        atlas_resolution_um=10,
        y_bins=2,
        depth_bin_um=10,
        bounds_by_style=bounds,
        chunk_voxels=2,
    )


def _install_windows_memmap_guard(monkeypatch):
    """Simulate Windows refusing to remove or move mapped NPY files."""
    tracked: list[tuple[Path, np.memmap]] = []
    removed_run_dirs: list[Path] = []
    published_temp_dirs: list[Path] = []
    real_load = region_cache_module.np.load
    real_open_memmap = region_cache_module.np.lib.format.open_memmap
    real_rmtree = region_cache_module.shutil.rmtree
    real_replace = region_cache_module.os.replace

    def track(path, array):
        if isinstance(array, np.memmap):
            tracked.append((Path(path), array))
        return array

    def tracked_load(path, *args, **kwargs):
        return track(path, real_load(path, *args, **kwargs))

    def tracked_open_memmap(path, *args, **kwargs):
        return track(path, real_open_memmap(path, *args, **kwargs))

    def open_paths_below(directory: Path) -> list[Path]:
        directory = Path(directory)
        return [
            path
            for path, array in tracked
            if path.is_relative_to(directory)
            and getattr(array, "_mmap", None) is not None
            and not array._mmap.closed
        ]

    def guarded_rmtree(path, *args, **kwargs):
        path = Path(path)
        locked = open_paths_below(path)
        if locked:
            raise PermissionError(
                32,
                "The process cannot access the file because it is being used "
                "by another process",
                str(locked[0]),
            )
        if path.name in {".occupancy-runs", ".geometry-runs"}:
            removed_run_dirs.append(path)
        return real_rmtree(path, *args, **kwargs)

    def guarded_replace(source, destination, *args, **kwargs):
        source_path = Path(source)
        if source_path.is_dir() and source_path.name.endswith(".tmp"):
            locked = open_paths_below(source_path)
            if locked:
                raise PermissionError(
                    32,
                    "The process cannot access the file because it is being used "
                    "by another process",
                    str(locked[0]),
                )
            published_temp_dirs.append(source_path)
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(region_cache_module.np, "load", tracked_load)
    monkeypatch.setattr(
        region_cache_module.np.lib.format,
        "open_memmap",
        tracked_open_memmap,
    )
    monkeypatch.setattr(region_cache_module.shutil, "rmtree", guarded_rmtree)
    monkeypatch.setattr(region_cache_module.os, "replace", guarded_replace)
    return tracked, removed_run_dirs, published_temp_dirs


def test_region_cache_round_trip_and_parent_geometry(tmp_path):
    profile = _build(tmp_path / "cache")
    assert profile.lookup_set_id == "portable-lookup-set"
    assert set(profile.styles) == {"shaped", "square"}
    assert profile.style("both_shaped").output_shape == (2, 2, 2)
    assert profile.style("square").grid_spec["x_bounds"] == [0.0, 2.0]

    cache = open_region_cache(tmp_path / "cache")
    assert list(cache.profiles) == [profile.profile_id]
    style = cache.profile().style("shaped")
    assert isinstance(style.array("occupancy_linear_bins"), np.memmap)
    assert style._data["counts"]["mirrored_depth_source_voxels"] == 1

    result = materialize_region_selection(
        cache,
        [2, 3],
        style="shaped",
        direct_region_ids=[1, 5],
    )
    assert result.labels.shape == (2, 2, 2)
    # Region 2 and 3 each contribute one voxel to bin zero: smaller ID wins.
    assert result.labels.reshape(-1)[0] == 2
    assert result.summary.collision_bins == 1
    assert result.represented_region_ids == (2, 3)
    assert [surface.region_id for surface in result.surfaces] == [1, 5]
    assert result.surfaces[0].vertices.shape[1:] == (3,)
    assert result.surfaces[0].faces.shape[1:] == (3,)
    assert result.surfaces[1].vertices.shape == (0, 3)
    assert result.outlines[0].vectors.shape[1:] == (2, 3)
    assert result.outlines[1].vectors.shape == (0, 2, 3)
    assert np.all(result.surfaces[0].faces >= 0)
    assert np.max(result.surfaces[0].faces) < len(result.surfaces[0].vertices)

    assert materialize_region_surface(profile, 1, style="shaped").component_count >= 1
    assert materialize_region_outlines(profile, 1, style="shaped").vectors.shape[
        1:
    ] == (2, 3)


@pytest.mark.parametrize("style", ["shaped", "square"])
def test_allen_layer_region_selection_collapses_depth_and_resolves_collisions(
    tmp_path,
    style,
):
    profile = _build_planar_cache(tmp_path / "cache")

    result = materialize_allen_layer_region_selection(
        profile,
        [1, 10, 11, 12],
        style=style,
        layer_map=_allen_layer_map(),
    )

    assert result.labels.shape == (6, 2, 2)
    assert result.labels.dtype == np.int32
    # Regions 10 and 11 each contribute two source voxels after depth
    # collapse, so the smaller region ID wins their shared planar bin.
    assert result.labels[0, 0, 0] == 10
    # Region 12 occupies the same XY bin in another Allen plane.
    assert result.labels[1, 0, 0] == 12
    assert np.count_nonzero(result.labels) == 2
    assert result.selected_region_ids == (1, 10, 11, 12)
    assert result.layer_mapped_region_ids == (10, 11, 12)
    assert result.represented_region_ids == (10, 11, 12)
    assert result.layer_labels == ("L1", "L2/3", "L4", "L5", "L6a", "L6b")
    assert result.summary.collision_bins == 1
    assert result.summary.source_voxel_count == 5
    assert result.summary.excluded_non_layer_region_count == 1
    assert result.summary.to_dict()["output_shape"] == [6, 2, 2]
    assert result.grid_spec["coordinate_order"] == ["allen_layer", "y", "x"]
    assert result.grid_spec["plane_mode"] == "allen_layers"
    assert result.style == style


def test_allen_layer_region_selection_reports_mapped_but_empty_regions(tmp_path):
    profile = _build_planar_cache(tmp_path / "cache")

    result = materialize_allen_layer_region_selection(
        profile,
        [13],
        style="shaped",
        layer_map=_allen_layer_map(),
    )

    assert result.layer_mapped_region_ids == (13,)
    assert result.represented_region_ids == ()
    assert result.summary.labeled_bins == 0
    assert not np.any(result.labels)


def test_allen_layer_region_selection_compacts_selected_stack_planes(tmp_path):
    profile = _build_planar_cache(tmp_path / "cache")

    result = materialize_allen_layer_region_selection(
        profile,
        [10, 11, 12],
        style="shaped",
        layer_map=_allen_layer_map(),
        selected_layer_indices=[1],
        output_mode="stack",
    )

    assert result.labels.shape == (1, 2, 2)
    assert result.layer_labels == ("L2/3",)
    assert result.labels[0, 0, 0] == 12
    assert result.layer_mapped_region_ids == (12,)
    assert result.summary.excluded_unselected_layer_region_count == 2
    assert result.summary.excluded_non_layer_region_count == 0
    assert result.summary.to_dict()["selected_layer_labels"] == ["L2/3"]
    assert result.grid_spec["selected_layer_indices"] == [1]


def test_allen_layer_region_projection_uses_majority_across_selected_layers(
    tmp_path,
):
    profile = _build_planar_cache(tmp_path / "cache")

    result = materialize_allen_layer_region_selection(
        profile,
        [10, 11, 12],
        style="shaped",
        layer_map=_allen_layer_map(),
        selected_layer_indices=[0, 1],
        output_mode="projection",
    )

    assert result.labels.shape == (2, 2)
    # All three regions meet at this XY bin. Regions 10 and 11 have the
    # greatest equal occupancy, so the smaller region ID wins.
    assert result.labels[0, 0] == 10
    assert np.count_nonzero(result.labels) == 1
    assert result.summary.collision_bins == 1
    assert result.summary.output_mode == "projection"
    assert result.grid_spec["coordinate_order"] == ["y", "x"]
    assert result.grid_spec["plane_mode"] == "allen_layer_projection"


def _nested_structures():
    """Two single-child parents that are absent from cache occupancy."""
    return {
        6: {"id": 6, "acronym": "P6", "structure_id_path": [6]},
        7: {"id": 7, "acronym": "P7", "structure_id_path": [7]},
        2: {"id": 2, "acronym": "C2", "structure_id_path": [6, 2]},
        3: {"id": 3, "acronym": "C3", "structure_id_path": [7, 3]},
    }


@pytest.mark.parametrize("style", ["shaped", "square"])
def test_flat_region_selection_collapses_depth_into_one_plane(tmp_path, style):
    profile = _build_planar_cache(tmp_path / "cache")

    result = materialize_flat_region_selection(
        profile,
        [1, 10, 11, 12],
        style=style,
    )

    assert result.labels.shape == (2, 2)
    assert result.labels.dtype == np.int32
    # Regions 10 and 11 each contribute two source voxels after depth
    # collapse, so the smaller region ID wins their shared planar bin.
    assert result.labels[0, 0] == 10
    assert result.summary.labeled_bins == 1
    assert result.summary.collision_bins == 1
    assert result.summary.source_voxel_count == 5
    assert result.summary.output_shape == (2, 2)
    assert result.summary.to_dict()["output_shape"] == [2, 2]
    assert result.selected_region_ids == (1, 10, 11, 12)
    assert result.represented_region_ids == (10, 11, 12)
    assert result.represented_source_region_ids == (10, 11, 12)
    assert result.grid_spec["coordinate_order"] == ["y", "x"]
    assert result.grid_spec["plane_mode"] == "flat"
    assert result.grid_spec["output_shape"] == [2, 2]
    assert result.style == style


def test_flat_region_selection_retains_descendants_beneath_one_root(tmp_path):
    profile = _build(tmp_path / "cache")

    result = materialize_flat_region_selection(
        profile,
        [2, 3],
        style="shaped",
        direct_region_ids=[1],
    )

    # Descendant IDs remain in the label plane so the colormap can restyle them
    # independently. Regions 2 and 3 compete in their shared flatmap column;
    # their counts tie, so the smaller region ID wins that bin.
    assert result.summary.collision_bins == 1
    assert result.represented_region_ids == (1,)
    assert result.represented_source_region_ids == (2, 3)
    assert result.grid_spec["label_grouping"] == "source_region"
    assert result.grid_spec["geometry_grouping"] == "selected_root"
    assert sorted(set(np.unique(result.labels).tolist())) == [0, 2, 3]
    assert np.array_equal(result.labels, np.array([[2, 3], [2, 0]], dtype=np.int32))
    assert result.summary.labeled_bins == 3
    assert result.summary.source_voxel_count == 4


@pytest.mark.parametrize("hierarchy", ["region_descendants", "atlas_structures"])
def test_flat_region_selection_assigns_members_to_several_roots(tmp_path, hierarchy):
    profile = _build(tmp_path / "cache")
    kwargs = (
        {"region_descendants": {6: (2,), 7: (3,)}}
        if hierarchy == "region_descendants"
        else {"atlas_structures": _nested_structures()}
    )

    result = materialize_flat_region_selection(
        profile,
        [2, 3],
        style="shaped",
        direct_region_ids=[6, 7],
        **kwargs,
    )

    assert result.represented_region_ids == (6, 7)
    assert result.represented_source_region_ids == (2, 3)
    assert np.array_equal(result.labels, np.array([[2, 3], [2, 0]], dtype=np.int32))
    # The one shared bin is resolved from source occupancy across depth.
    assert result.summary.collision_bins == 1
    assert result.grid_spec["label_grouping"] == "source_region"
    assert result.grid_spec["geometry_grouping"] == "selected_root"
    assert [outline.region_id for outline in result.outlines] == [6, 7]
    assert result.outlines[0].represented_region_ids == (2,)
    assert result.outlines[1].represented_region_ids == (3,)


def test_flat_region_selection_requires_a_hierarchy_for_unmapped_members(tmp_path):
    profile = _build(tmp_path / "cache")

    with pytest.raises(ValueError, match="atlas_structures"):
        materialize_flat_region_selection(
            profile,
            [2, 3],
            style="shaped",
            direct_region_ids=[6, 7],
        )


def test_flat_region_selection_labels_terminal_selections_without_a_hierarchy(tmp_path):
    """Custom terminal selections name their own labels, so no atlas is needed."""
    profile = _build(tmp_path / "cache")

    result = materialize_flat_region_selection(
        profile,
        [2, 3],
        style="shaped",
        direct_region_ids=[2, 3],
    )

    assert result.represented_region_ids == (2, 3)
    assert np.array_equal(result.labels, np.array([[2, 3], [2, 0]], dtype=np.int32))


def test_flat_region_outlines_are_two_dimensional_and_occupancy_derived(tmp_path):
    from napari_neuron_navigator.flatmap_region_cache import _outlines_for_bins

    profile = _build(tmp_path / "cache")

    outlines = materialize_flat_region_outlines(
        profile,
        1,
        style="shaped",
        region_descendants={1: (2, 3)},
    )

    assert outlines is not None
    assert outlines.region_id == 1
    assert outlines.vectors.dtype == np.float32
    assert outlines.vectors.shape[1:] == (2, 2)
    assert outlines.planar_bin_count == 3
    assert outlines.represented_region_ids == (2, 3)
    # Regions 2 and 3 collapse onto planar bins 0, 1, and 2 -- an L of three
    # cells whose perimeter is eight unit edges.
    expected = _outlines_for_bins(np.array([0, 1, 2]), (1, 2, 2))[:, :, 1:]
    assert np.array_equal(outlines.vectors, expected)
    assert len(outlines.vectors) == 8

    assert (
        materialize_flat_region_outlines(
            profile,
            5,
            style="shaped",
            region_descendants={5: (5,)},
        )
        is None
    )


def test_flat_and_allen_layer_collapse_agree_bin_for_bin(tmp_path):
    """Both collapses share one primitive, so their footprints must match."""
    profile = _build_planar_cache(tmp_path / "cache")

    flat = materialize_flat_region_selection(profile, [10, 11, 12], style="shaped")
    allen = materialize_allen_layer_region_selection(
        profile,
        [10, 11, 12],
        style="shaped",
        layer_map=_allen_layer_map(),
    )

    assert np.array_equal((allen.labels != 0).any(axis=0), flat.labels != 0)


def test_flat_materializers_do_not_change_profile_identity(tmp_path):
    """The 2D overlays are read-time derivations, not a new cache format."""
    first = _build(tmp_path / "first")
    manifest_path = tmp_path / "first" / REGION_CACHE_MANIFEST_FILENAME
    before = manifest_path.read_text(encoding="utf-8")

    materialize_flat_region_selection(
        first,
        [2, 3],
        style="shaped",
        direct_region_ids=[1],
    )
    materialize_flat_region_outlines(
        first,
        1,
        style="shaped",
        region_descendants={1: (2, 3)},
    )

    second = _build(tmp_path / "second")
    assert second.profile_id == first.profile_id
    assert manifest_path.read_text(encoding="utf-8") == before


def test_open_accepts_pre_rename_cache_format(tmp_path):
    profile = _build(tmp_path / "cache")
    manifest_path = tmp_path / "cache" / REGION_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = LEGACY_REGION_CACHE_FORMAT
    manifest_path.write_text(json.dumps(manifest))

    cache = open_region_cache(tmp_path / "cache")

    assert cache.profile(profile.profile_id).profile_id == profile.profile_id
    cache.close()


def test_region_cache_close_releases_loaded_maps_and_is_terminal(tmp_path):
    profile = _build(tmp_path / "cache")
    cache = open_region_cache(tmp_path / "cache")
    style = cache.profile(profile.profile_id).style("shaped")
    loaded = style.array("occupancy_linear_bins")
    assert isinstance(loaded, np.memmap)
    assert not loaded._mmap.closed

    cache.close()
    cache.close()

    assert loaded._mmap.closed
    with pytest.raises(RegionCacheError, match="closed"):
        style.array("occupancy_linear_bins")

    profile_style = profile.style("square")
    profile_loaded = profile_style.array("surface_faces")
    profile.close()
    profile.close()
    assert profile_loaded._mmap.closed
    with pytest.raises(RegionCacheError, match="closed"):
        profile_style.array("surface_faces")


def test_surface_and_outline_are_voxel_faithful_for_adjacent_bins():
    from napari_neuron_navigator.flatmap_region_cache import (
        _outlines_for_bins,
        _surface_for_bins,
    )

    vertices, faces = _surface_for_bins(np.array([0, 1]), (1, 2, 2))
    outlines = _outlines_for_bins(np.array([0, 1]), (1, 2, 2))
    # Two cubes sharing one face expose ten quads and have six XY perimeter edges.
    assert vertices.shape == (12, 3)
    assert faces.shape == (20, 3)
    assert outlines.shape == (6, 2, 3)
    # A single depth plane reduces the stored tracer to a 2D perimeter, which is
    # what the flat overlays slice off. Both the start depth and every direction
    # depth must be zero for that slice to be lossless.
    assert np.all(outlines[:, :, 0] == 0)
    assert outlines[:, :, 1:].shape == (6, 2, 2)
    undirected_edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0),
        axis=1,
    )
    _edges, counts = np.unique(undirected_edges, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_multiple_profiles_duplicate_rejection_and_relocation(tmp_path):
    root = tmp_path / "cache"
    first = _build(root)
    with pytest.raises(FileExistsError, match="already exists"):
        _build(root)

    annotation, shaped, square, depth = _lookup_arrays()
    second = build_region_cache_profile(
        root,
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="portable-lookup-set",
        atlas_name="test_mouse",
        atlas_version="1.2",
        atlas_resolution_um=10,
        region_descendants={1: {2, 3}},
        y_bins=3,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    assert second.profile_id != first.profile_id
    assert len(open_region_cache(root).profiles) == 2

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    reopened = open_region_cache(relocated)
    assert set(reopened.profiles) == {first.profile_id, second.profile_id}
    assert reopened.profile(first.profile_id).directory.is_relative_to(relocated)


def test_profile_reports_exact_compatibility_mismatches(tmp_path):
    profile = _build(tmp_path / "cache")
    assert not profile.compatibility_mismatches(
        lookup_set_id="portable-lookup-set",
        atlas_name="test_mouse",
        atlas_version="1.2",
        atlas_resolution_um=(10, 10, 10),
        annotation_shape=(2, 2, 2),
        style="both_square",
    )
    mismatches = profile.compatibility_mismatches(
        lookup_set_id="different",
        atlas_name="other",
        atlas_version="2",
        annotation_shape=(3, 2, 2),
    )
    assert any("lookup_set_id" in message for message in mismatches)
    assert any("atlas family" in message for message in mismatches)
    assert any("atlas version" in message for message in mismatches)
    assert any("annotation shape" in message for message in mismatches)


def test_depth_mirror_policy_affects_profile_identity_and_compatibility(tmp_path):
    root = tmp_path / "cache"
    default = _build(root)
    different_axis = _build(root, mirror_coord_axis=1)
    no_fallback = _build(root, mirror_depth_fallback=False)

    assert (
        len({default.profile_id, different_axis.profile_id, no_fallback.profile_id})
        == 3
    )
    assert not default.compatibility_mismatches(
        mirror_depth_fallback=True,
        mirror_coord_axis=2,
    )

    mismatches = default.compatibility_mismatches(
        mirror_depth_fallback=False,
        mirror_coord_axis=1,
    )
    assert any("depth mirror fallback" in message for message in mismatches)
    assert any("depth mirror axis" in message for message in mismatches)


def test_structure_catalog_and_atlas_family_are_resolution_independent(tmp_path):
    annotation, shaped, square, depth = _lookup_arrays()
    structures = {
        "root": {
            "id": 1,
            "acronym": "ROOT",
            "name": "Root",
            "structure_id_path": [1],
            "rgb_triplet": [1, 2, 3],
        },
        "child": {
            "id": 2,
            "acronym": "C",
            "name": "Child",
            "structure_id_path": [1, 2],
            "rgb_triplet": [4, 5, 6],
        },
    }
    profile = build_region_cache_profile(
        tmp_path / "cache",
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="lookup",
        atlas_name="allen_mouse_10um",
        atlas_version="1.2",
        atlas_resolution_um=10,
        atlas_structures=structures,
        y_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    catalog_id = structure_catalog_id(structures)
    assert profile.atlas["family"] == "allen_mouse"
    assert profile.atlas["structure_catalog_id"] == catalog_id
    # Viewing may use a different atlas resolution when its family/catalog match.
    assert not profile.compatibility_mismatches(
        atlas_name="allen_mouse_25um",
        atlas_version="1.2",
        structure_catalog_id=catalog_id,
    )
    assert any(
        "structure catalog" in message
        for message in profile.compatibility_mismatches(
            atlas_name="allen_mouse_25um",
            structure_catalog_id="different",
        )
    )


def test_open_rejects_missing_corrupt_and_invalid_arrays(tmp_path):
    root = tmp_path / "cache"
    profile = _build(root)
    manifest = json.loads((root / REGION_CACHE_MANIFEST_FILENAME).read_text())
    style_data = manifest["profiles"][profile.profile_id]["styles"]["shaped"]
    profile_dir = (
        root / manifest["profiles"][profile.profile_id]["directory"] / "shaped"
    )
    profile.close()
    offsets_path = (
        profile_dir / style_data["arrays"]["occupancy_region_offsets"]["path"]
    )
    offsets = np.load(offsets_path, allow_pickle=False)
    offsets[0] = 1
    np.save(offsets_path, offsets, allow_pickle=False)
    with pytest.raises(RegionCacheValidationError, match="offsets"):
        open_region_cache(root)

    offsets[0] = 0
    np.save(offsets_path, offsets, allow_pickle=False)
    faces_path = profile_dir / style_data["arrays"]["surface_faces"]["path"]
    faces = np.load(faces_path, allow_pickle=False)
    faces[0, 0] = np.iinfo(np.int32).max
    np.save(faces_path, faces, allow_pickle=False)
    with pytest.raises(RegionCacheValidationError, match="face index"):
        open_region_cache(root)

    faces_path.unlink()
    with pytest.raises(RegionCacheValidationError, match="missing"):
        open_region_cache(root)


def test_open_rejects_contract_dtype_even_when_file_matches_manifest(tmp_path):
    root = tmp_path / "cache"
    profile = _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    style_data = manifest["profiles"][profile.profile_id]["styles"]["shaped"]
    profile_dir = (
        root / manifest["profiles"][profile.profile_id]["directory"] / "shaped"
    )
    profile.close()
    region_ids_spec = style_data["arrays"]["occupancy_region_ids"]
    region_ids_path = profile_dir / region_ids_spec["path"]
    region_ids = np.load(region_ids_path, allow_pickle=False).astype(np.int64)
    np.save(region_ids_path, region_ids, allow_pickle=False)
    region_ids_spec["dtype"] = "int64"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        RegionCacheValidationError,
        match="occupancy_region_ids.*must use int32",
    ):
        open_region_cache(root)


@pytest.mark.parametrize("corruption", ["dtype", "shape"])
def test_open_closes_array_rejected_before_registration(
    tmp_path, monkeypatch, corruption
):
    root = tmp_path / "cache"
    profile = _build(root)
    manifest = json.loads((root / REGION_CACHE_MANIFEST_FILENAME).read_text())
    style_data = manifest["profiles"][profile.profile_id]["styles"]["shaped"]
    profile_dir = (
        root / manifest["profiles"][profile.profile_id]["directory"] / "shaped"
    )
    profile.close()
    region_ids_path = profile_dir / style_data["arrays"]["occupancy_region_ids"]["path"]
    region_ids = np.load(region_ids_path, allow_pickle=False)
    if corruption == "dtype":
        region_ids = region_ids.astype(np.int64)
    else:
        region_ids = np.concatenate((region_ids, region_ids[:1]))
    np.save(region_ids_path, region_ids, allow_pickle=False)

    tracked, _removed_run_dirs, _published_temp_dirs = _install_windows_memmap_guard(
        monkeypatch
    )
    with pytest.raises(RegionCacheValidationError, match=corruption):
        open_region_cache(root)

    rejected = [array for path, array in tracked if path == region_ids_path]
    assert rejected
    assert all(array._mmap.closed for array in rejected)


def test_open_rejects_incorrect_validation_count_metadata(tmp_path):
    root = tmp_path / "cache"
    profile = _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    style_data = manifest["profiles"][profile.profile_id]["styles"]["shaped"]
    style_data["counts"]["surface_face_count"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        RegionCacheValidationError,
        match="validation count 'surface_face_count' differs",
    ):
        open_region_cache(root)


def test_cancelled_build_keeps_existing_manifest_and_cleans_temporary_profile(tmp_path):
    root = tmp_path / "cache"
    _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    original = manifest_path.read_bytes()
    annotation, shaped, square, depth = _lookup_arrays()
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(RegionCacheCancelled):
        build_region_cache_profile(
            root,
            annotation=annotation,
            shaped_flatmap=shaped,
            square_flatmap=square,
            depth=depth,
            lookup_set_id="other-lookup",
            atlas_name="test_mouse",
            atlas_version="1.2",
            atlas_resolution_um=10,
            y_bins=3,
            depth_bin_um=10,
            chunk_voxels=2,
            cancel_callback=cancel,
        )
    assert manifest_path.read_bytes() == original
    assert not list((root / "profiles").glob("*.tmp"))
    assert len(open_region_cache(root).profiles) == 1


def test_build_closes_all_temporary_memmaps_before_cleanup_and_publication(
    tmp_path,
    monkeypatch,
):
    tracked, removed_run_dirs, published_temp_dirs = _install_windows_memmap_guard(
        monkeypatch
    )
    annotation, shaped, square, depth = _lookup_arrays()
    profile = build_region_cache_profile(
        tmp_path / "cache",
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="windows-handles",
        atlas_name="test_mouse",
        atlas_version="1.2",
        atlas_resolution_um=10,
        # Both directly selectable regions deliberately share one footprint,
        # exercising the footprint comparison mapping from the reported bug.
        region_descendants={1: {2}, 2: {2}},
        y_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )

    np.testing.assert_array_equal(
        profile.style("shaped").array("geometry_region_footprint_indices"),
        [0, 0],
    )
    temporary_names = {
        path.name
        for path, _array in tracked
        if ".occupancy-runs" in path.parts or ".geometry-runs" in path.parts
    }
    assert any(name.startswith("keys-") for name in temporary_names)
    assert any(name.startswith("counts-") for name in temporary_names)
    assert any(name.endswith("-bins.npy") for name in temporary_names)
    assert any(name.endswith("-vertices.npy") for name in temporary_names)
    assert any(name.endswith("-faces.npy") for name in temporary_names)
    assert any(name.endswith("-outlines.npy") for name in temporary_names)
    assert sum(path.name == ".occupancy-runs" for path in removed_run_dirs) == 2
    assert sum(path.name == ".geometry-runs" for path in removed_run_dirs) == 2
    assert len(published_temp_dirs) == 1

    # Keep strong references in ``tracked`` so this verifies explicit closure,
    # rather than CPython reference counting incidentally releasing handles.
    prepublication_maps = [
        array for path, array in tracked if path.is_relative_to(published_temp_dirs[0])
    ]
    assert prepublication_maps
    assert all(array._mmap.closed for array in prepublication_maps)


def test_failed_build_preserves_original_error_when_temp_cleanup_is_locked(
    tmp_path,
    monkeypatch,
    caplog,
):
    root = tmp_path / "cache"
    _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    original_manifest = manifest_path.read_bytes()
    real_rmtree = region_cache_module.shutil.rmtree

    def fail_build_style(**_kwargs):
        raise RuntimeError("injected region processing failure")

    def sharing_violation(path, *args, **kwargs):
        if Path(path).name.endswith(".tmp"):
            raise PermissionError(
                32,
                "The process cannot access the file because it is being used "
                "by another process",
                str(path),
            )
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(region_cache_module, "_build_style", fail_build_style)
    monkeypatch.setattr(region_cache_module.shutil, "rmtree", sharing_violation)
    caplog.set_level("WARNING", logger=region_cache_module.__name__)

    annotation, shaped, square, depth = _lookup_arrays()
    with pytest.raises(RuntimeError, match="injected region processing failure"):
        build_region_cache_profile(
            root,
            annotation=annotation,
            shaped_flatmap=shaped,
            square_flatmap=square,
            depth=depth,
            lookup_set_id="failed-windows-build",
            atlas_name="test_mouse",
            atlas_version="1.2",
            atlas_resolution_um=10,
            y_bins=2,
            depth_bin_um=10,
            chunk_voxels=2,
        )

    assert manifest_path.read_bytes() == original_manifest
    assert list((root / "profiles").glob("*.tmp"))
    assert ".tmp" in caplog.text
    assert len(open_region_cache(root).profiles) == 1


def test_replace_keeps_new_profile_when_old_mapped_directory_is_locked(
    tmp_path,
    monkeypatch,
    caplog,
):
    root = tmp_path / "cache"
    old_profile = _build(root)
    old_directory = old_profile.directory
    real_rmtree = region_cache_module.shutil.rmtree

    def sharing_violation(path, *args, **kwargs):
        if Path(path) == old_directory:
            raise PermissionError(
                32,
                "The process cannot access the file because it is being used "
                "by another process",
                str(path),
            )
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(region_cache_module.shutil, "rmtree", sharing_violation)
    caplog.set_level("WARNING", logger=region_cache_module.__name__)
    replacement = _build(root, replace=True)

    assert replacement.directory != old_directory
    assert old_directory.is_dir()
    reopened = open_region_cache(root).profile(old_profile.profile_id)
    assert reopened.directory == replacement.directory
    assert replacement.directory.is_dir()
    assert str(old_directory) in caplog.text


def test_build_rejects_annotation_lookup_shape_mismatch(tmp_path):
    _annotation, shaped, square, depth = _lookup_arrays()
    with pytest.raises(ValueError, match="exactly match"):
        build_region_cache_profile(
            tmp_path / "cache",
            annotation=np.zeros((3, 2, 2), dtype=np.int32),
            shaped_flatmap=shaped,
            square_flatmap=square,
            depth=depth,
            lookup_set_id="lookup",
            atlas_name="test_mouse",
            atlas_resolution_um=10,
            y_bins=2,
            depth_bin_um=10,
        )


def test_build_uses_lookup_set_grid_contract_and_requires_exact_resolution(
    tmp_path,
):
    from napari_neuron_navigator.flatmap_profiles import FlatmapGridSpec

    annotation, shaped, square, depth = _lookup_arrays()

    def grid(style: str, x_upper: float, y_upper: float) -> FlatmapGridSpec:
        return FlatmapGridSpec(
            grid_spec_id=f"grid-{style}",
            style=style,
            lookup_coordinate_order=("x", "y", "z"),
            flatmap_coordinate_order=("x_flat", "y_flat"),
            render_coordinate_order=("depth", "y", "x"),
            spatial_shape=(2, 2, 2),
            flatmap_shape=(2, 2, 2, 2),
            depth_shape=(2, 2, 2),
            lookup_resolution_um=(10.0, 10.0, 10.0),
            space_directions=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
            space_origin=(0.0, 0.0, 0.0),
            x_bounds=(0.0, x_upper),
            y_bounds=(0.0, y_upper),
            depth_bounds_um=(0.0, 10.0),
            invalid_zero_sentinel=False,
            invalid_negative_one_sentinel=True,
        )

    lookup_set = SimpleNamespace(
        lookup_set_id="lookup-contract",
        lookup_resolution_um=(10.0, 10.0, 10.0),
        spatial_shape=(2, 2, 2),
        shaped_grid=grid("both_shaped", 1.0, 1.0),
        # Ratio 1.0 so the derived x count is 2, matching the (2, 2, 2) shape
        # this test asserts.
        square_grid=grid("both_square", 2.0, 2.0),
    )
    profile = build_region_cache_profile(
        tmp_path / "cache",
        lookup_set,
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        atlas_name="test_mouse_10um",
        atlas_resolution_um=10,
        y_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    style_cache = profile.style("shaped")
    assert style_cache.lookup_grid_spec == lookup_set.shaped_grid
    assert style_cache.grid_spec["x_bounds"] == [0.0, 1.0]

    with pytest.raises(ValueError, match="resolution match"):
        build_region_cache_profile(
            tmp_path / "bad-cache",
            lookup_set,
            annotation=annotation,
            shaped_flatmap=shaped,
            square_flatmap=square,
            depth=depth,
            atlas_name="test_mouse_25um",
            atlas_resolution_um=25,
            y_bins=2,
            depth_bin_um=10,
        )


def test_empty_annotation_regions_round_trip(tmp_path):
    annotation, shaped, square, depth = _lookup_arrays()
    profile = build_region_cache_profile(
        tmp_path / "cache",
        annotation=np.zeros_like(annotation),
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="empty-lookup",
        atlas_name="test_mouse",
        atlas_resolution_um=10,
        region_descendants={1: {1}},
        y_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    result = materialize_region_selection(
        profile,
        [1],
        style="shaped",
        direct_region_ids=[1],
    )
    assert not np.any(result.labels)
    assert result.summary.labeled_bins == 0
    assert result.surfaces[0].vertices.shape == (0, 3)
    assert result.outlines[0].vectors.shape == (0, 2, 3)


def _rectangular_lookup_arrays():
    """A lookup whose valid x extent is twice its y extent.

    ``y_bins=2`` therefore derives ``x_bins=4``, so every encode/decode step is
    exercised on a grid where a ``(y, x)`` transpose cannot go unnoticed.
    """
    shape = (2, 2, 2)
    flat = np.empty((*shape, 2), dtype=np.float32)
    for first in range(shape[0]):
        for second in range(shape[1]):
            for third in range(shape[2]):
                # x spans 0..2 while y spans 0..1, an aspect ratio of exactly 2.
                flat[first, second, third, 0] = first * 2.0
                flat[first, second, third, 1] = float(second)
    depth = np.zeros(shape, dtype=np.float32)
    depth[:, :, 1] = 10.0
    annotation = np.full(shape, 7, dtype=np.int32)
    return annotation, flat, depth


def _build_rectangular(cache_dir: Path):
    annotation, flat, depth = _rectangular_lookup_arrays()
    return build_region_cache_profile(
        cache_dir,
        annotation=annotation,
        shaped_flatmap=flat,
        square_flatmap=flat.copy(),
        depth=depth,
        lookup_set_id="rectangular-lookup-set",
        atlas_name="test_mouse",
        atlas_resolution_um=10,
        region_descendants={7: {7}},
        y_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )


def test_rectangular_profile_records_per_axis_counts(tmp_path):
    """The grid spec must carry both counts and a matching output shape."""
    profile = _build_rectangular(tmp_path / "cache")
    grid = profile.style("shaped").grid_spec
    assert grid["y_bins"] == 2
    # x spans 0..2 and y spans 0..1, a ratio of 2, so x gets twice the bins.
    assert grid["x_bins"] == 4
    assert grid["coordinate_order"] == ["depth", "y", "x"]
    assert grid["output_shape"] == [profile.style("shaped").output_shape[0], 2, 4]
    assert profile.style("shaped").output_shape[1:] == (2, 4)
    # The single-count key is gone, so a stale reader fails loudly.
    assert "xy_bins" not in grid


def test_rectangular_region_cache_round_trips_the_linear_encoding(tmp_path):
    """Decoded labels must match an independent ``z*y*x + y*x + x`` recompute.

    The encode side packs ``region_id << 32 | linear``; if the linear stride used
    ``y_bins`` where it needed ``x_bins`` the labels would scatter, which only a
    non-square grid can reveal.
    """
    profile = _build_rectangular(tmp_path / "cache")
    depth_bins, y_bins, x_bins = profile.style("shaped").output_shape
    assert (y_bins, x_bins) == (2, 4)

    result = materialize_region_selection(
        profile,
        [7],
        style="shaped",
        direct_region_ids=[7],
    )
    assert result.labels.shape == (depth_bins, y_bins, x_bins)

    annotation, flat, depth = _rectangular_lookup_arrays()
    grid = profile.style("shaped").grid_spec
    x_bounds = tuple(float(value) for value in grid["x_bounds"])
    y_bounds = tuple(float(value) for value in grid["y_bounds"])
    depth_bounds = tuple(float(value) for value in grid["depth_bounds_um"])
    depth_bin_um = float(grid["depth_bin_um"])

    expected = np.zeros(depth_bins * y_bins * x_bins, dtype=np.int32)
    for index in np.ndindex(annotation.shape):
        x_value = float(flat[index][0])
        y_value = float(flat[index][1])
        depth_value = float(depth[index])
        x_bin = min(
            x_bins - 1,
            int(
                np.floor((x_value - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * x_bins)
            ),
        )
        y_bin = min(
            y_bins - 1,
            int(
                np.floor((y_value - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) * y_bins)
            ),
        )
        depth_bin = min(
            depth_bins - 1,
            int(np.floor((depth_value - depth_bounds[0]) / depth_bin_um)),
        )
        linear = depth_bin * y_bins * x_bins + y_bin * x_bins + x_bin
        expected[linear] = 7

    np.testing.assert_array_equal(
        result.labels.reshape(-1),
        expected,
    )
    # A real rectangle, so the stride above is not the square special case.
    assert y_bins != x_bins


def test_rectangular_flat_and_allen_materializers_keep_the_x_axis(tmp_path):
    """Collapsing depth must not collapse the rectangle."""
    profile = _build_rectangular(tmp_path / "cache")
    flat = materialize_flat_region_selection(
        profile,
        [7],
        style="shaped",
        direct_region_ids=[7],
    )
    assert flat.labels.shape == (2, 4)

    outlines = materialize_flat_region_outlines(profile, 7, style="shaped")
    assert outlines.vectors.shape[1:] == (2, 2)

    allen = materialize_allen_layer_region_selection(
        profile,
        [7],
        style="shaped",
        layer_map=AllenIsocortexLayerMap(
            atlas_name="test_mouse",
            isocortex_region_id=1,
            region_to_layer_index={7: 0},
            region_ids_by_layer=((7,), (), (), (), (), ()),
        ),
    )
    assert allen.labels.shape == (6, 2, 4)
    assert allen.grid_spec["y_bins"] == 2
    assert allen.grid_spec["x_bins"] == 4


def test_version_one_manifest_is_rejected_with_a_rebuild_message(tmp_path):
    """An old cache must say "rebuild", not fail on a missing grid key.

    Version 1 stored a single ``xy_bins`` and binned its arrays on that grid, so
    it cannot be reinterpreted -- only rebuilt.  Without the version bump the
    first failure would be a ``KeyError`` surfacing as "invalid fixed-grid
    dimensions", which points at corruption rather than at an outdated cache.
    """
    root = tmp_path / "cache"
    _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_version"] == 2
    manifest["format_version"] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RegionCacheValidationError, match="[Rr]ebuild") as excinfo:
        open_region_cache(root)
    message = str(excinfo.value)
    assert "older version" in message
    assert "format version 1" in message
    assert "invalid fixed-grid dimensions" not in message


def test_uint32_key_guard_names_the_control_to_reduce(tmp_path, monkeypatch):
    """The portable-key guard must be actionable on a rectangular grid.

    ``region_id << 32 | linear`` needs ``linear`` under 2**32.  Deriving x makes
    the grid wider than the old square one, so the ceiling arrives sooner and the
    message has to name a control the user actually has.
    """
    annotation, flat, depth = _rectangular_lookup_arrays()
    with pytest.raises(ValueError) as excinfo:
        build_region_cache_profile(
            tmp_path / "cache",
            annotation=annotation,
            shaped_flatmap=flat,
            square_flatmap=flat.copy(),
            depth=depth,
            lookup_set_id="too-fine",
            atlas_name="test_mouse",
            atlas_resolution_um=10,
            region_descendants={7: {7}},
            # Ratio 2, so 60000 y bins derive 120000 x bins: with 2 depth bins
            # that is 1.44e10 linear bins, far over the 2**32 ceiling.
            y_bins=60000,
            depth_bin_um=10,
            chunk_voxels=2,
        )
    message = str(excinfo.value)
    assert "portable key format" in message
    assert "Reduce Y bins" in message
    # The offending grid is reported so the user can see how far over it is.
    derived = resolve_flatmap_bin_counts(
        x_bounds=(0.0, 2.0),
        y_bounds=(0.0, 1.0),
        y_bins=60000,
    )
    assert str(derived.x_bins) in message
    assert str(derived.y_bins) in message

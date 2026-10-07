"""Tests for atlas-backed region lookup helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from napari_neuron_navigator.region import (
    get_dataset_region_structure_ids,
    get_region_at_coords,
    get_region_ids_vectorized,
)


def test_dataset_region_structure_ids_handles_unknown_ids_and_invalid_paths():
    atlas = SimpleNamespace(
        structures={
            997: {"structure_id_path": [997]},
            184: {"structure_id_path": [997, 184]},
            68: {"structure_id_path": [997, "184", None, "invalid", 68]},
            667: {"structure_id_path": [997, 184, 667]},
            500: {"structure_id_path": None},
        }
    )

    assert get_dataset_region_structure_ids(atlas, {68, 500, 999999}) == {
        997,
        184,
        68,
        500,
    }
    assert get_dataset_region_structure_ids(atlas, set()) == set()
    assert get_dataset_region_structure_ids(None, {68}) == set()


class _DummyStructureTree:
    """Minimal structure tree stub for direct region lookup tests."""

    def __init__(self, structures_by_id):
        self._structures_by_id = structures_by_id

    def get_structures_by_id(self, ids: list[int]) -> list[dict]:
        return [self._structures_by_id[i] for i in ids if i in self._structures_by_id]


def test_get_region_ids_vectorized_truncates_instead_of_rounding():
    """Micron-to-voxel conversion should match BrainGlobe's truncation rule."""
    annotation = np.zeros((1, 1, 4), dtype=np.int32)
    annotation[0, 0, 1] = 5
    annotation[0, 0, 2] = 11

    coords = np.array([[0.0, 0.0, 37.6]])

    region_ids = get_region_ids_vectorized(coords, annotation, resolution=25)

    assert region_ids.tolist() == [5]


def test_get_region_at_coords_truncates_single_coordinate_lookup():
    """Single-point lookup should use the same truncation convention."""
    annotation = np.zeros((1, 1, 4), dtype=np.int32)
    annotation[0, 0, 1] = 5
    annotation[0, 0, 2] = 11
    structure_tree = _DummyStructureTree(
        {
            5: {
                "id": 5,
                "name": "Voxel One",
                "acronym": "V1",
                "structure_id_path": [5],
                "color_hex_triplet": "ffffff",
            },
            11: {
                "id": 11,
                "name": "Voxel Two",
                "acronym": "V2",
                "structure_id_path": [11],
                "color_hex_triplet": "000000",
            },
        }
    )

    region = get_region_at_coords((0.0, 0.0, 37.6), annotation, structure_tree, resolution=25)

    assert region is not None
    assert region["id"] == 5
    assert region["acronym"] == "V1"

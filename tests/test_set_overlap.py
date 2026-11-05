"""Spatial comparison checks real coordinates and the source cohort snapshot."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from napari_neuron_navigator.analysis.set_overlap import (
    build_set_intersection_mask,
    compare_set_regions,
    source_set_metadata,
)


def _region(set_id, members, values):
    snapshot = {"set_id": set_id, "name": set_id, "rule": {}, "file_ids": members}
    return SimpleNamespace(
        data=np.asarray(values, dtype=np.uint8).reshape(1, 2, 2),
        metadata={
            "mask_query_source": True,
            "atlas_name": "test_atlas",
            "neuron_set": deepcopy(snapshot),
            "source_file_ids": list(members),
        },
        scale=(1, 1, 1),
        translate=(0, 0, 0),
    ), snapshot


def _compare(a, b, sa, sb):
    return compare_set_regions(a, b, set_a=sa, set_b=sb, atlas_name="test_atlas")


def test_spatial_overlap_independent_of_shared_neurons():
    a, sa = _region("a", ["subject-a/shared"], [1, 1, 0, 0])
    b, sb = _region("b", ["subject-b/shared"], [0, 1, 1, 0])
    result = _compare(a, b, sa, sb)
    assert result["intersection"] == 1
    assert result["union"] == 3
    assert result["a_only"] == result["b_only"] == 1
    assert result["jaccard"] == pytest.approx(1 / 3)
    assert result["dice"] == 0.5
    assert result["fraction_a"] == result["fraction_b"] == 0.5
    b.data[:] = 0
    assert _compare(a, b, sa, sb)["jaccard"] == 0
    a.data[:] = 0
    assert _compare(a, b, sa, sb)["jaccard"] is None


@pytest.mark.parametrize(
    "operation", [compare_set_regions, build_set_intersection_mask]
)
@pytest.mark.parametrize(
    "defect, message",
    [
        ("atlas", "atlas"),
        ("members", "membership has changed"),
        ("sources", "source neurons"),
        ("set_id", "does not belong"),
        ("shape", "shape"),
        ("scale", "transforms"),
        ("translation", "transforms"),
        ("rotation", "transforms"),
        ("raw", "thresholded"),
    ],
)
def test_incomparable_regions_rejected(defect, message, operation):
    a, sa = _region("a", ["a"], [1, 0, 0, 0])
    b, sb = _region("b", ["b"], [1, 0, 0, 0])
    if defect == "atlas":
        b.metadata["atlas_name"] = "other"
    elif defect == "members":
        sb["file_ids"].append("new")
    elif defect == "sources":
        b.metadata["source_file_ids"] = ["other"]
    elif defect == "set_id":
        b.metadata["neuron_set"]["set_id"] = "other"
    elif defect == "shape":
        b.data = b.data[:, :, :1]
    elif defect == "scale":
        b.scale = (1, 1, 2)
    elif defect == "translation":
        b.translate = (1, 0, 0)
    elif defect == "rotation":
        b.data_to_world = lambda p: (p[1], p[0], p[2])
    elif defect == "raw":
        b.metadata["mask_query_source"] = False
    with pytest.raises(ValueError, match=message):
        operation(a, b, set_a=sa, set_b=sb, atlas_name="test_atlas")


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
        ([2, 0, 3, 0], [4, 0, 0, 5], [1, 0, 0, 0]),
    ],
)
def test_intersection_mask_contains_only_shared_nonzero_voxels(left, right, expected):
    a, sa = _region("a", ["a"], left)
    b, sb = _region("b", ["b"], right)
    # Exercise multiple slabs and a final partial slab.
    a.data = np.tile(a.data, (19, 1, 1))
    b.data = np.tile(b.data, (19, 1, 1))
    before_a, before_b = a.data.copy(), b.data.copy()
    mask = build_set_intersection_mask(
        a, b, set_a=sa, set_b=sb, atlas_name="test_atlas"
    )
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(
        mask, np.tile(np.array(expected).reshape(1, 2, 2), (19, 1, 1))
    )
    np.testing.assert_array_equal(a.data, before_a)
    np.testing.assert_array_equal(b.data, before_b)
    assert mask.sum() == _compare(a, b, sa, sb)["intersection"]


def test_provenance_only_propagates_identical_set_snapshots():
    a, _ = _region("a", ["a"], [1, 0, 0, 0])
    b = deepcopy(a)
    assert source_set_metadata([a, b]) == {"neuron_set": a.metadata["neuron_set"]}
    b.metadata["neuron_set"]["file_ids"] = ["a", "b"]
    assert source_set_metadata([a, b]) == {}
    assert source_set_metadata([a, SimpleNamespace(metadata={})]) == {}

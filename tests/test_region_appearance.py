from __future__ import annotations

import json

import numpy as np
import pytest

from napari_neuron_navigator.region_appearance import (
    LEGACY_REGION_PALETTE_FORMAT,
    REGION_PALETTE_FORMAT,
    EffectiveRegionAppearance,
    RegionAppearanceOverride,
    RegionAppearanceStore,
    atlas_identity,
    load_region_palette,
    prepare_region_palette_import,
    save_region_palette,
)


STRUCTURES = {
    1: {
        "id": 1,
        "name": "Root",
        "acronym": "ROOT",
        "structure_id_path": [1],
        "rgb_triplet": [255, 0, 0],
    },
    2: {
        "id": 2,
        "name": "Child",
        "acronym": "CH",
        "structure_id_path": [1, 2],
        "rgb_triplet": [0, 255, 0],
    },
    3: {
        "id": 3,
        "name": "Leaf",
        "acronym": "LF",
        "structure_id_path": "/1/2/3/",
        "rgb_triplet": [0, 0, 255],
    },
}


def test_default_appearance_uses_each_regions_own_atlas_color() -> None:
    appearance = RegionAppearanceStore()

    root = appearance.resolve(1, STRUCTURES)
    leaf = appearance.resolve(3, STRUCTURES)

    assert root.color_rgba == (1.0, 0.0, 0.0, 1.0)
    assert leaf.color_rgba == (0.0, 0.0, 1.0, 1.0)
    assert leaf.fill_visible
    assert leaf.fill_opacity == 1.0
    assert leaf.outline_visible
    assert leaf.outline_opacity == 1.0


def test_nearest_explicit_ancestor_values_are_inherited_per_property() -> None:
    appearance = RegionAppearanceStore(
        overrides={
            1: RegionAppearanceOverride(
                color_mode="custom",
                color_rgb=(0.2, 0.3, 0.4),
                fill_visible=False,
                outline_opacity=0.25,
            ),
            2: RegionAppearanceOverride(fill_opacity=0.6, outline_visible=False),
        }
    )

    leaf = appearance.resolve(3, STRUCTURES)

    assert leaf == EffectiveRegionAppearance(
        color_rgba=(0.2, 0.3, 0.4, 1.0),
        fill_visible=False,
        fill_opacity=0.6,
        outline_visible=False,
        outline_opacity=0.25,
    )
    np.testing.assert_allclose(leaf.fill_rgba, [0.2, 0.3, 0.4, 0.0])
    np.testing.assert_allclose(leaf.outline_rgba, [0.2, 0.3, 0.4, 0.0])


def test_explicit_atlas_color_breaks_parent_color_inheritance() -> None:
    appearance = RegionAppearanceStore(
        overrides={
            1: RegionAppearanceOverride(color_mode="custom", color_rgb=(1.0, 1.0, 0.0)),
            2: RegionAppearanceOverride(color_mode="atlas"),
        }
    )

    assert appearance.resolve(3, STRUCTURES).color_rgba == (0.0, 1.0, 0.0, 1.0)


def test_child_can_override_hidden_parent() -> None:
    appearance = RegionAppearanceStore(
        overrides={
            1: RegionAppearanceOverride(fill_visible=False),
            3: RegionAppearanceOverride(fill_visible=True),
        }
    )

    assert appearance.resolve(2, STRUCTURES).fill_visible is False
    assert appearance.resolve(3, STRUCTURES).fill_visible is True


@pytest.mark.parametrize(
    "override",
    [
        RegionAppearanceOverride(color_mode="atlas"),
        RegionAppearanceOverride(
            color_mode="custom",
            color_rgb=(0.1, 0.2, 0.3),
            fill_visible=False,
            fill_opacity=0.4,
            outline_visible=True,
            outline_opacity=0.8,
        ),
    ],
)
def test_palette_state_round_trip(override: RegionAppearanceOverride) -> None:
    source = RegionAppearanceStore(
        atlas_name="allen_mouse_25um",
        atlas_version="1.2",
        overrides={3: override},
    )

    payload = source.to_palette_dict()
    restored = RegionAppearanceStore.from_palette_dict(payload)

    assert payload["format"] == REGION_PALETTE_FORMAT
    assert payload["overrides"] == {"3": override.to_dict()}
    assert restored == source


def test_palette_state_accepts_pre_rename_format() -> None:
    payload = RegionAppearanceStore(overrides={2: RegionAppearanceOverride()}).to_palette_dict()
    payload["format"] = LEGACY_REGION_PALETTE_FORMAT

    restored = RegionAppearanceStore.from_palette_dict(payload)

    assert restored.region_ids == ()


def test_palette_file_round_trip(tmp_path) -> None:
    source = RegionAppearanceStore(
        atlas_name="allen_mouse_25um",
        overrides={
            2: RegionAppearanceOverride(
                color_mode="custom", color_rgb=(0.25, 0.5, 0.75)
            )
        },
    )

    path = save_region_palette(tmp_path / "regions.json", source)

    assert json.loads(path.read_text())["atlas"]["name"] == "allen_mouse_25um"
    assert load_region_palette(path) == source


@pytest.mark.parametrize("existing_file", [False, True])
def test_empty_palette_export_does_not_create_or_overwrite_files(
    tmp_path, existing_file: bool
) -> None:
    path = tmp_path / "regions.json"
    original = None
    if existing_file:
        save_region_palette(
            path,
            RegionAppearanceStore(
                overrides={2: RegionAppearanceOverride(fill_opacity=0.5)}
            ),
        )
        original = path.read_bytes()

    with pytest.raises(ValueError, match="no appearance overrides"):
        save_region_palette(
            path, RegionAppearanceStore(atlas_name="allen_mouse_25um")
        )

    if existing_file:
        assert path.read_bytes() == original
    else:
        assert not path.exists()


def test_empty_override_removes_existing_entry() -> None:
    store = RegionAppearanceStore(
        overrides={1: RegionAppearanceOverride(fill_visible=False)}
    )

    store.set_override(1, RegionAppearanceOverride())

    assert store.region_ids == ()


def test_invalid_override_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires color_rgb"):
        RegionAppearanceOverride(color_mode="custom")
    with pytest.raises(ValueError, match="between 0 and 1"):
        RegionAppearanceOverride(fill_opacity=1.1)
    with pytest.raises(ValueError, match="exactly three"):
        RegionAppearanceOverride(color_mode="custom", color_rgb=(1.0, 0.0))


def test_palette_rejects_duplicate_ids() -> None:
    payload = {
        "format": REGION_PALETTE_FORMAT,
        "version": 1,
        "atlas": {},
        "overrides": [
            {"region_id": 1, "fill_visible": True},
            {"region_id": 1, "fill_visible": False},
        ],
    }

    with pytest.raises(ValueError, match="Duplicate"):
        RegionAppearanceStore.from_palette_dict(payload)


def test_palette_import_filters_unknown_ids_and_reports_version_mismatch() -> None:
    imported = RegionAppearanceStore(
        atlas_name="allen_mouse_25um",
        atlas_version="1.1",
        overrides={
            2: RegionAppearanceOverride(fill_visible=False),
            999: RegionAppearanceOverride(outline_visible=False),
        },
    )

    summary = prepare_region_palette_import(
        imported,
        atlas_name="allen_mouse_25um",
        atlas_version="1.2",
        known_region_ids=STRUCTURES,
    )

    assert summary.version_mismatch
    assert summary.unknown_region_ids == (999,)
    assert summary.store.atlas_version == "1.2"
    assert summary.store.region_ids == (2,)


def test_palette_import_rejects_atlas_name_mismatch() -> None:
    imported = RegionAppearanceStore(atlas_name="allen_mouse_10um")

    with pytest.raises(ValueError, match="atlas mismatch"):
        prepare_region_palette_import(
            imported,
            atlas_name="allen_mouse_25um",
            atlas_version="1.2",
            known_region_ids=STRUCTURES,
        )


def test_palette_merge_and_replace_have_distinct_semantics() -> None:
    existing = RegionAppearanceStore(
        atlas_name="allen_mouse_25um",
        overrides={1: RegionAppearanceOverride(fill_visible=False)},
    )
    imported = RegionAppearanceStore(
        atlas_name="allen_mouse_25um",
        overrides={2: RegionAppearanceOverride(outline_visible=False)},
    )

    merged = existing.copy()
    merged.merge(imported)
    replaced = existing.copy()
    replaced.replace_with(imported)

    assert merged.region_ids == (1, 2)
    assert replaced.region_ids == (2,)


def test_atlas_identity_uses_metadata_version_fallback() -> None:
    atlas = type(
        "Atlas",
        (),
        {"atlas_name": "allen_mouse_25um", "metadata": {"version": "1.2"}},
    )()

    assert atlas_identity(atlas) == ("allen_mouse_25um", "1.2")

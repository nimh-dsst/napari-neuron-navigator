"""Shared, atlas-scoped appearance settings for anatomical regions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


REGION_PALETTE_FORMAT = "napari_neuron_navigator.region_palette"
LEGACY_REGION_PALETTE_FORMAT = "napari_swc_viewer.region_palette"
REGION_PALETTE_FORMATS = frozenset(
    {REGION_PALETTE_FORMAT, LEGACY_REGION_PALETTE_FORMAT}
)
REGION_PALETTE_VERSION = 1

ColorMode = Literal["custom", "atlas"]


def _normalise_rgb(value: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(component) for component in value)
    if len(values) != 3:
        raise ValueError("Region colors must contain exactly three RGB values.")
    if not all(
        np.isfinite(component) and 0.0 <= component <= 1.0 for component in values
    ):
        raise ValueError("Region RGB values must be finite numbers between 0 and 1.")
    return values


def _normalise_opacity(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    opacity = float(value)
    if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return opacity


@dataclass(frozen=True, kw_only=True)
class RegionAppearanceOverride:
    """Explicit appearance values for one atlas region.

    ``None`` means that the property inherits from the nearest ancestor with an
    explicit value.  A color mode of ``"atlas"`` is an explicit inheritance
    break: it selects this region's catalog color even when an ancestor has a
    custom color.
    """

    color_mode: ColorMode | None = None
    color_rgb: tuple[float, float, float] | None = None
    fill_visible: bool | None = None
    fill_opacity: float | None = None
    outline_visible: bool | None = None
    outline_opacity: float | None = None

    def __post_init__(self) -> None:
        if self.color_mode not in (None, "custom", "atlas"):
            raise ValueError("color_mode must be 'custom', 'atlas', or None.")
        if self.color_rgb is not None:
            object.__setattr__(self, "color_rgb", _normalise_rgb(self.color_rgb))
        if self.color_mode == "custom" and self.color_rgb is None:
            raise ValueError("A custom region color requires color_rgb.")
        if self.color_mode != "custom" and self.color_rgb is not None:
            raise ValueError("color_rgb is only valid when color_mode is 'custom'.")
        object.__setattr__(
            self,
            "fill_opacity",
            _normalise_opacity(self.fill_opacity, "fill_opacity"),
        )
        object.__setattr__(
            self,
            "outline_opacity",
            _normalise_opacity(self.outline_opacity, "outline_opacity"),
        )

    @property
    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.color_mode,
                self.fill_visible,
                self.fill_opacity,
                self.outline_visible,
                self.outline_opacity,
            )
        )

    def updated(self, **changes: object) -> RegionAppearanceOverride:
        """Return a validated copy with the requested fields changed."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.color_mode is not None:
            payload["color_mode"] = self.color_mode
        if self.color_rgb is not None:
            payload["color_rgb"] = list(self.color_rgb)
        for field_name in (
            "fill_visible",
            "fill_opacity",
            "outline_visible",
            "outline_opacity",
        ):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegionAppearanceOverride:
        color_rgb = payload.get("color_rgb")
        return cls(
            color_mode=payload.get("color_mode"),  # type: ignore[arg-type]
            color_rgb=(
                _normalise_rgb(color_rgb)  # type: ignore[arg-type]
                if color_rgb is not None
                else None
            ),
            fill_visible=_optional_bool(payload.get("fill_visible"), "fill_visible"),
            fill_opacity=_optional_float(payload.get("fill_opacity")),
            outline_visible=_optional_bool(
                payload.get("outline_visible"), "outline_visible"
            ),
            outline_opacity=_optional_float(payload.get("outline_opacity")),
        )


@dataclass(frozen=True, kw_only=True)
class EffectiveRegionAppearance:
    """Fully resolved appearance used by one rendered atlas region."""

    color_rgba: tuple[float, float, float, float]
    fill_visible: bool
    fill_opacity: float
    outline_visible: bool
    outline_opacity: float

    @property
    def fill_rgba(self) -> np.ndarray:
        alpha = self.fill_opacity if self.fill_visible else 0.0
        return np.asarray((*self.color_rgba[:3], alpha), dtype=np.float32)

    @property
    def outline_rgba(self) -> np.ndarray:
        alpha = self.outline_opacity if self.outline_visible else 0.0
        return np.asarray((*self.color_rgba[:3], alpha), dtype=np.float32)


def _optional_bool(value: object, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true, false, or omitted.")
    return value


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def atlas_identity(atlas: object | None) -> tuple[str, str]:
    """Return stable atlas name/version strings for appearance persistence."""
    if atlas is None:
        return "", ""
    name = str(getattr(atlas, "atlas_name", "") or "")
    metadata = getattr(atlas, "metadata", None)
    metadata_version = (
        metadata.get("version") if isinstance(metadata, Mapping) else None
    )
    raw_version = (
        getattr(atlas, "local_version", None)
        or getattr(atlas, "atlas_version", None)
        or getattr(atlas, "version", None)
        or metadata_version
        or ""
    )
    version = (
        ".".join(str(part).strip() for part in raw_version)
        if isinstance(raw_version, (tuple, list))
        else str(raw_version).strip()
    )
    return name, version


def _structure_items(structures: object | None):
    if structures is None:
        return ()
    items = getattr(structures, "items", None)
    if callable(items):
        return items()
    if isinstance(structures, (list, tuple)):
        return enumerate(structures)
    return ()


def structure_catalog(structures: object | None) -> dict[int, Mapping[str, Any]]:
    """Normalize BrainGlobe catalog variants to an ID-keyed mapping."""
    if isinstance(structures, dict) and all(
        isinstance(key, int) and isinstance(value, Mapping)
        for key, value in structures.items()
    ):
        return structures
    result: dict[int, Mapping[str, Any]] = {}
    for key, structure in _structure_items(structures):
        if not isinstance(structure, Mapping):
            continue
        try:
            region_id = int(structure.get("id", key))
        except (TypeError, ValueError):
            continue
        result[region_id] = structure
    return result


def structure_path(
    region_id: int,
    catalog: Mapping[int, Mapping[str, Any]],
) -> tuple[int, ...]:
    """Return a root-to-region path that always ends with ``region_id``."""
    normalized_id = int(region_id)
    structure = catalog.get(normalized_id)
    raw_path = structure.get("structure_id_path", ()) if structure else ()
    if isinstance(raw_path, str):
        raw_values = [value for value in raw_path.strip("/").split("/") if value]
    else:
        try:
            raw_values = list(raw_path or ())
        except TypeError:
            raw_values = []
    path: list[int] = []
    for value in raw_values:
        try:
            path.append(int(value))
        except (TypeError, ValueError):
            continue
    if not path or path[-1] != normalized_id:
        path.append(normalized_id)
    return tuple(dict.fromkeys(path))


def atlas_region_rgb(
    region_id: int,
    catalog: Mapping[int, Mapping[str, Any]],
) -> tuple[float, float, float]:
    """Return one region's catalog RGB, with a neutral missing-ID fallback."""
    structure = catalog.get(int(region_id))
    raw_rgb = structure.get("rgb_triplet", (128, 128, 128)) if structure else (128,) * 3
    try:
        values = tuple(float(component) for component in raw_rgb)
    except (TypeError, ValueError):
        values = (128.0, 128.0, 128.0)
    if len(values) != 3:
        values = (128.0, 128.0, 128.0)
    if any(component > 1.0 for component in values):
        values = tuple(component / 255.0 for component in values)
    return tuple(float(np.clip(component, 0.0, 1.0)) for component in values)


class RegionAppearanceStore:
    """Committed or draft appearance overrides for one atlas identity."""

    def __init__(
        self,
        *,
        atlas_name: str = "",
        atlas_version: str = "",
        overrides: Mapping[int, RegionAppearanceOverride] | None = None,
    ) -> None:
        self.atlas_name = str(atlas_name or "")
        self.atlas_version = str(atlas_version or "")
        self._overrides: dict[int, RegionAppearanceOverride] = {}
        for region_id, override in (overrides or {}).items():
            self.set_override(int(region_id), override)

    def copy(self) -> RegionAppearanceStore:
        return RegionAppearanceStore(
            atlas_name=self.atlas_name,
            atlas_version=self.atlas_version,
            overrides=self._overrides,
        )

    @property
    def region_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._overrides))

    def override_for(self, region_id: int) -> RegionAppearanceOverride:
        return self._overrides.get(int(region_id), RegionAppearanceOverride())

    def set_override(
        self,
        region_id: int,
        override: RegionAppearanceOverride,
    ) -> None:
        normalized_id = int(region_id)
        if normalized_id <= 0:
            raise ValueError("Region appearance IDs must be positive integers.")
        if not isinstance(override, RegionAppearanceOverride):
            raise TypeError("override must be a RegionAppearanceOverride.")
        if override.is_empty:
            self._overrides.pop(normalized_id, None)
        else:
            self._overrides[normalized_id] = override

    def clear_override(self, region_id: int) -> None:
        self._overrides.pop(int(region_id), None)

    def replace_with(self, other: RegionAppearanceStore) -> None:
        self.atlas_name = other.atlas_name
        self.atlas_version = other.atlas_version
        self._overrides = dict(other._overrides)

    def merge(self, other: RegionAppearanceStore) -> None:
        for region_id in other.region_ids:
            self.set_override(region_id, other.override_for(region_id))

    def resolve(
        self,
        region_id: int,
        structures: object | None,
    ) -> EffectiveRegionAppearance:
        """Resolve inherited values for one target region."""
        catalog = structure_catalog(structures)
        path = structure_path(int(region_id), catalog)

        color = atlas_region_rgb(int(region_id), catalog)
        fill_visible = True
        fill_opacity = 1.0
        outline_visible = True
        outline_opacity = 1.0
        color_resolved = False
        fill_visible_resolved = False
        fill_opacity_resolved = False
        outline_visible_resolved = False
        outline_opacity_resolved = False

        for ancestor_id in reversed(path):
            override = self._overrides.get(int(ancestor_id))
            if override is None:
                continue
            if not color_resolved and override.color_mode is not None:
                color = (
                    override.color_rgb
                    if override.color_mode == "custom"
                    else atlas_region_rgb(int(ancestor_id), catalog)
                )
                color_resolved = True
            if not fill_visible_resolved and override.fill_visible is not None:
                fill_visible = override.fill_visible
                fill_visible_resolved = True
            if not fill_opacity_resolved and override.fill_opacity is not None:
                fill_opacity = override.fill_opacity
                fill_opacity_resolved = True
            if not outline_visible_resolved and override.outline_visible is not None:
                outline_visible = override.outline_visible
                outline_visible_resolved = True
            if not outline_opacity_resolved and override.outline_opacity is not None:
                outline_opacity = override.outline_opacity
                outline_opacity_resolved = True

        return EffectiveRegionAppearance(
            color_rgba=(*color, 1.0),
            fill_visible=bool(fill_visible),
            fill_opacity=float(fill_opacity),
            outline_visible=bool(outline_visible),
            outline_opacity=float(outline_opacity),
        )

    def to_palette_dict(self) -> dict[str, object]:
        return {
            "format": REGION_PALETTE_FORMAT,
            "version": REGION_PALETTE_VERSION,
            "atlas": {
                "name": self.atlas_name,
                "version": self.atlas_version,
            },
            "overrides": {
                str(region_id): self._overrides[region_id].to_dict()
                for region_id in sorted(self._overrides)
            },
        }

    @classmethod
    def from_palette_dict(cls, payload: Mapping[str, object]) -> RegionAppearanceStore:
        if payload.get("format") not in REGION_PALETTE_FORMATS:
            raise ValueError("Unsupported region palette format.")
        try:
            version = int(payload.get("version", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid region palette version.") from exc
        if version != REGION_PALETTE_VERSION:
            raise ValueError(f"Unsupported region palette version: {version}.")
        atlas = payload.get("atlas", {})
        if not isinstance(atlas, Mapping):
            raise ValueError("Region palette atlas metadata must be an object.")
        store = cls(
            atlas_name=str(atlas.get("name", "") or ""),
            atlas_version=str(atlas.get("version", "") or ""),
        )
        raw_overrides = payload.get("overrides", {})
        if isinstance(raw_overrides, Mapping):
            override_entries = []
            for raw_region_id, raw_override in raw_overrides.items():
                if not isinstance(raw_override, Mapping):
                    raise ValueError("Each region palette override must be an object.")
                override_entries.append(
                    {"region_id": raw_region_id, **dict(raw_override)}
                )
        elif isinstance(raw_overrides, list):
            # Accept the pre-release list representation so early exported
            # palettes and project bundles remain readable.
            override_entries = raw_overrides
        else:
            raise ValueError("Region palette overrides must be an object.")
        seen: set[int] = set()
        for entry in override_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Each region palette override must be an object.")
            try:
                region_id = int(entry["region_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Each region palette override needs a region_id."
                ) from exc
            if region_id in seen:
                raise ValueError(
                    f"Duplicate region palette override for ID {region_id}."
                )
            seen.add(region_id)
            store.set_override(region_id, RegionAppearanceOverride.from_dict(entry))
        return store

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RegionAppearanceStore)
            and self.atlas_name == other.atlas_name
            and self.atlas_version == other.atlas_version
            and self._overrides == other._overrides
        )


@dataclass(frozen=True, kw_only=True)
class RegionPaletteImportSummary:
    """Validated, atlas-filtered palette import ready for user confirmation."""

    store: RegionAppearanceStore
    unknown_region_ids: tuple[int, ...]
    version_mismatch: bool


def prepare_region_palette_import(
    imported: RegionAppearanceStore,
    *,
    atlas_name: str,
    atlas_version: str,
    known_region_ids: Iterable[int],
) -> RegionPaletteImportSummary:
    """Validate atlas identity and skip overrides absent from the catalog."""
    current_name = str(atlas_name or "")
    current_version = str(atlas_version or "")
    if imported.atlas_name and imported.atlas_name != current_name:
        raise ValueError(
            f"Region palette atlas mismatch: {imported.atlas_name} != {current_name}."
        )
    version_mismatch = bool(
        imported.atlas_version
        and current_version
        and imported.atlas_version != current_version
    )
    valid_ids = {int(value) for value in known_region_ids}
    filtered = RegionAppearanceStore(
        atlas_name=current_name,
        atlas_version=current_version,
    )
    unknown: list[int] = []
    for region_id in imported.region_ids:
        if region_id not in valid_ids:
            unknown.append(region_id)
            continue
        filtered.set_override(region_id, imported.override_for(region_id))
    return RegionPaletteImportSummary(
        store=filtered,
        unknown_region_ids=tuple(unknown),
        version_mismatch=version_mismatch,
    )


def save_region_palette(path: str | Path, store: RegionAppearanceStore) -> Path:
    """Write one reusable, human-readable region palette JSON file."""
    if not store.region_ids:
        raise ValueError("Cannot export a region palette with no appearance overrides.")
    output_path = Path(path)
    output_path.write_text(
        json.dumps(store.to_palette_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_region_palette(path: str | Path) -> RegionAppearanceStore:
    """Read and validate one reusable region palette JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Region palette root must be a JSON object.")
    return RegionAppearanceStore.from_palette_dict(payload)

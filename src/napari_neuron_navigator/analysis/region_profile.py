"""Build and validate sparse, atlas-aware regional neuron profiles.

The canonical profile contains one row for each non-empty
``file_id x direct region_id x laterality x compartment`` bucket.  Region IDs
are never expanded to ancestors in this table; :func:`build_hierarchy_closure`
provides the separate mapping used by later query stages.

This module is deliberately independent of Qt.  A caller may run
:func:`build_region_profile` in any cancellable worker and then register the
validated sidecar in a connection-local DuckDB temporary table.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .region_profile_traversal import AtlasAxisSide, AtlasVoxelTraverser

if TYPE_CHECKING:
    from numpy.typing import NDArray


PROFILE_FORMAT_VERSION = 1
BUILDER_ALGORITHM_VERSION = "regional_profile_builder_v1"
LENGTH_METHOD = "atlas_voxel_traversal_v1"
REGION_PROFILE_METADATA_KEY = b"napari_neuron_navigator.region_profile_json"
DEFAULT_ROW_GROUP_SIZE = 122_880
DEFAULT_MAX_BATCH_NEURONS = 400
DEFAULT_MAX_BATCH_ROWS = 5_000_000
DEFAULT_LENGTH_ABSOLUTE_TOLERANCE_UM = 1e-9
DEFAULT_LENGTH_RELATIVE_TOLERANCE = 1e-12

REQUIRED_SOURCE_COLUMNS = (
    "file_id",
    "node_id",
    "type",
    "x",
    "y",
    "z",
    "parent_id",
    "region_id",
)

REGION_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("file_id", pa.string(), nullable=False),
        pa.field("region_id", pa.int32(), nullable=False),
        pa.field(
            "laterality",
            pa.dictionary(pa.int8(), pa.string()),
            nullable=False,
        ),
        pa.field(
            "compartment",
            pa.dictionary(pa.int8(), pa.string()),
            nullable=False,
        ),
        pa.field("cable_length_um", pa.float64(), nullable=False),
        pa.field("node_count", pa.int64(), nullable=False),
        pa.field("terminus_count", pa.int64(), nullable=False),
    ]
)

_RAW_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("file_id", pa.string(), nullable=False),
        pa.field("region_id", pa.int32(), nullable=False),
        pa.field("laterality", pa.string(), nullable=False),
        pa.field("compartment", pa.string(), nullable=False),
        pa.field("cable_length_um", pa.float64(), nullable=False),
        pa.field("node_count", pa.int64(), nullable=False),
        pa.field("terminus_count", pa.int64(), nullable=False),
    ]
)

_SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOURCE_SHA256_CACHE: dict[tuple[object, ...], str] = {}


class Laterality(str, Enum):
    """A profile bucket's physical side relative to its neuron's soma."""

    IPSILATERAL = "ipsilateral"
    CONTRALATERAL = "contralateral"
    MIDLINE = "midline"
    UNKNOWN = "unknown"


class Compartment(str, Enum):
    """The two morphology compartments represented by profile version 1."""

    SOMA = "soma"
    NEURITE = "neurite"


class RegionProfileError(RuntimeError):
    """Base exception for regional-profile failures."""


class RegionProfileValidationError(RegionProfileError):
    """Raised when source, atlas, or sidecar invariants are invalid."""

    def __init__(self, message: str, *, issues: Sequence[str] = ()) -> None:
        self.issues = tuple(str(issue) for issue in issues)
        detail = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"{message}\n{detail}" if detail else message)


class RegionProfileBuildCancelled(RegionProfileError):
    """Raised when a caller cancels profile construction."""


@dataclass(frozen=True)
class HierarchyLink:
    """One ancestor-to-direct-region relationship."""

    ancestor_region_id: int
    direct_region_id: int
    depth: int


@dataclass(frozen=True)
class RegionHierarchyClosure:
    """Canonical closure derived solely from ``structure_id_path`` values."""

    links: tuple[HierarchyLink, ...]
    digest: str

    def represented_direct_ids(self, region_id: int) -> tuple[int, ...]:
        """Return direct IDs represented by an ancestor, including itself."""
        selected = {
            link.direct_region_id
            for link in self.links
            if link.ancestor_region_id == int(region_id)
        }
        return tuple(sorted(selected))

    def to_arrow(self) -> pa.Table:
        """Return the closure as a small, deterministic Arrow table."""
        return pa.Table.from_pydict(
            {
                "ancestor_region_id": [link.ancestor_region_id for link in self.links],
                "direct_region_id": [link.direct_region_id for link in self.links],
                "depth": [link.depth for link in self.links],
            },
            schema=pa.schema(
                [
                    pa.field("ancestor_region_id", pa.int32(), nullable=False),
                    pa.field("direct_region_id", pa.int32(), nullable=False),
                    pa.field("depth", pa.int32(), nullable=False),
                ]
            ),
        )


def _structure_mapping(structures: object) -> dict[int, Mapping[str, Any]]:
    items = getattr(structures, "items", None)
    if not callable(items):
        raise TypeError("atlas structures must be an ID-keyed mapping")
    result: dict[int, Mapping[str, Any]] = {}
    for raw_id, structure in items():
        if not isinstance(structure, Mapping):
            continue
        try:
            region_id = int(structure.get("id", raw_id))
        except (TypeError, ValueError):
            continue
        result[region_id] = structure
    if not result:
        raise ValueError("atlas structures contain no numeric region IDs")
    return result


def _structure_path(region_id: int, structure: Mapping[str, Any]) -> tuple[int, ...]:
    raw_path = structure.get("structure_id_path") or ()
    if isinstance(raw_path, str):
        values: Sequence[object] = tuple(
            value for value in raw_path.strip("/").split("/") if value
        )
    else:
        try:
            values = tuple(raw_path)
        except TypeError as exc:
            raise ValueError(
                f"region {region_id} has an invalid structure_id_path"
            ) from exc
    path: list[int] = []
    for value in values:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"region {region_id} has a non-numeric structure_id_path value"
            ) from exc
        if normalized not in path:
            path.append(normalized)
    if not path or path[-1] != region_id:
        path.append(region_id)
    return tuple(path)


def build_hierarchy_closure(structures: object) -> RegionHierarchyClosure:
    """Build a deterministic ancestor closure from atlas structure paths.

    The closure always includes ``(region_id, region_id, depth=0)`` and never
    relies on ``parent_structure_id`` being available.
    """
    catalog = _structure_mapping(structures)
    links: set[HierarchyLink] = set()
    for direct_id, structure in catalog.items():
        path = _structure_path(direct_id, structure)
        for index, ancestor_id in enumerate(path):
            links.add(
                HierarchyLink(
                    ancestor_region_id=ancestor_id,
                    direct_region_id=direct_id,
                    depth=len(path) - index - 1,
                )
            )
    ordered = tuple(
        sorted(
            links,
            key=lambda item: (
                item.ancestor_region_id,
                item.direct_region_id,
                item.depth,
            ),
        )
    )
    payload = [asdict(link) for link in ordered]
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return RegionHierarchyClosure(links=ordered, digest=digest)


def _normalise_resolution(
    resolution_um: float | Sequence[float],
) -> tuple[float, float, float]:
    values = np.asarray(resolution_um, dtype=np.float64)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    if values.shape != (3,):
        raise ValueError("atlas resolution must be a scalar or three values")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("atlas resolution values must be finite and positive")
    return tuple(float(value) for value in values)


def resolve_physical_midline(
    hemisphere_annotation: NDArray[np.integer],
    resolution_um: float | Sequence[float],
    *,
    left_right_axis: int,
    hemisphere_codes: tuple[int, int] = (1, 2),
) -> tuple[float, int, int]:
    """Resolve the physical hemisphere boundary from BrainGlobe labels.

    Returns ``(midline_um, lower_code, upper_code)``.  Codes 1 and 2 must form
    adjacent, non-overlapping index ranges on the left-right axis.  The result
    is the shared voxel *boundary*, not the midpoint between voxel centers.
    """
    volume = np.asarray(hemisphere_annotation)
    if volume.ndim != 3 or not np.issubdtype(volume.dtype, np.integer):
        raise ValueError("hemisphere_annotation must be a 3D integer array")
    axis = int(left_right_axis)
    if axis not in (0, 1, 2):
        raise ValueError("left_right_axis must be 0, 1, or 2")
    resolution = _normalise_resolution(resolution_um)

    # Scan one left-right slab at a time.  A full-volume ``nonzero`` would
    # allocate several index arrays that are much larger than the uint8 atlas.
    codes = tuple(int(code) for code in hemisphere_codes)
    if len(codes) != 2 or codes[0] == codes[1]:
        raise ValueError("hemisphere_codes must contain two distinct values")
    axis_first = np.moveaxis(volume, axis, 0)
    positions: dict[int, list[int]] = {code: [] for code in codes}
    for index, slab in enumerate(axis_first):
        for code in codes:
            if np.any(slab == code):
                positions[code].append(index)
    ranges: dict[int, tuple[int, int]] = {}
    for code in codes:
        if not positions[code]:
            raise ValueError(f"hemisphere annotation does not contain code {code}")
        ranges[code] = (positions[code][0], positions[code][-1])

    first, second = sorted(codes, key=lambda code: ranges[code][0])
    first_range = ranges[first]
    second_range = ranges[second]
    if first_range[1] >= second_range[0]:
        raise ValueError("hemisphere codes overlap on the left-right axis")
    if first_range[1] + 1 != second_range[0]:
        raise ValueError("hemisphere codes do not meet at one voxel boundary")
    boundary_index = second_range[0]
    return boundary_index * resolution[axis], first, second


def _atlas_version(atlas: object) -> str:
    metadata = getattr(atlas, "metadata", None)
    metadata_version = (
        metadata.get("version") if isinstance(metadata, Mapping) else None
    )
    value = (
        getattr(atlas, "local_version", None)
        or getattr(atlas, "atlas_version", None)
        or getattr(atlas, "version", None)
        or metadata_version
        or ""
    )
    if isinstance(value, (tuple, list)):
        return ".".join(str(part).strip() for part in value)
    return str(value).strip()


@dataclass(frozen=True)
class RegionalProfileAtlas:
    """The atlas inputs and physical geometry required by the builder."""

    atlas_name: str
    atlas_version: str
    annotation: NDArray[np.integer]
    resolution_um: tuple[float, float, float]
    structures: Mapping[int, Mapping[str, Any]]
    left_right_axis: int
    midline_um: float
    lower_hemisphere_code: int | None = None
    upper_hemisphere_code: int | None = None

    def __post_init__(self) -> None:
        annotation = np.asarray(self.annotation)
        if annotation.ndim != 3 or not np.issubdtype(annotation.dtype, np.integer):
            raise ValueError("atlas annotation must be a 3D integer array")
        resolution = _normalise_resolution(self.resolution_um)
        axis = int(self.left_right_axis)
        if axis not in (0, 1, 2):
            raise ValueError("left_right_axis must be 0, 1, or 2")
        midline = float(self.midline_um)
        extent = annotation.shape[axis] * resolution[axis]
        if not math.isfinite(midline) or not 0.0 <= midline <= extent:
            raise ValueError("midline_um must lie within the atlas bounds")
        object.__setattr__(self, "annotation", annotation)
        object.__setattr__(self, "resolution_um", resolution)
        object.__setattr__(self, "structures", _structure_mapping(self.structures))
        object.__setattr__(self, "left_right_axis", axis)
        object.__setattr__(self, "midline_um", midline)

    @classmethod
    def from_atlas(
        cls,
        atlas: object,
        *,
        left_right_axis: int = 2,
        hemisphere_annotation: NDArray[np.integer] | None = None,
    ) -> RegionalProfileAtlas:
        """Create builder inputs from a loaded BrainGlobe atlas."""
        annotation = np.asarray(atlas.annotation)
        resolution = _normalise_resolution(atlas.resolution)
        hemisphere_volume = hemisphere_annotation
        if hemisphere_volume is None:
            hemisphere_volume = getattr(atlas, "hemispheres", None)
        if hemisphere_volume is None:
            raise ValueError(
                "the loaded atlas does not expose hemisphere labels; pass "
                "hemisphere_annotation explicitly"
            )
        midline, lower_code, upper_code = resolve_physical_midline(
            np.asarray(hemisphere_volume),
            resolution,
            left_right_axis=left_right_axis,
            hemisphere_codes=(
                int(getattr(atlas, "left_hemisphere_value", 1)),
                int(getattr(atlas, "right_hemisphere_value", 2)),
            ),
        )
        return cls(
            atlas_name=str(getattr(atlas, "atlas_name", "") or ""),
            atlas_version=_atlas_version(atlas),
            annotation=annotation,
            resolution_um=resolution,
            structures=atlas.structures,
            left_right_axis=left_right_axis,
            midline_um=midline,
            lower_hemisphere_code=lower_code,
            upper_hemisphere_code=upper_code,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.annotation.shape)

    @property
    def left_right_extent_um(self) -> float:
        return (
            self.shape[self.left_right_axis] * self.resolution_um[self.left_right_axis]
        )


def _hash_ndarray(array: NDArray[Any]) -> str:
    """Hash array identity and C-order contents without one full-size copy."""
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    if values.ndim == 0:
        digest.update(values.tobytes())
    else:
        for index in range(values.shape[0]):
            digest.update(np.ascontiguousarray(values[index]).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class AtlasIdentity:
    atlas_name: str
    atlas_version: str
    resolution_um: tuple[float, float, float]
    annotation_shape: tuple[int, int, int]
    annotation_dtype: str
    annotation_digest: str
    hierarchy_digest: str
    left_right_axis: int
    midline_um: float
    lower_hemisphere_code: int | None
    upper_hemisphere_code: int | None

    @classmethod
    def from_profile_atlas(cls, atlas: RegionalProfileAtlas) -> AtlasIdentity:
        closure = build_hierarchy_closure(atlas.structures)
        return cls(
            atlas_name=atlas.atlas_name,
            atlas_version=atlas.atlas_version,
            resolution_um=atlas.resolution_um,
            annotation_shape=atlas.shape,
            annotation_dtype=str(atlas.annotation.dtype),
            annotation_digest=_hash_ndarray(atlas.annotation),
            hierarchy_digest=closure.digest,
            left_right_axis=atlas.left_right_axis,
            midline_um=atlas.midline_um,
            lower_hemisphere_code=atlas.lower_hemisphere_code,
            upper_hemisphere_code=atlas.upper_hemisphere_code,
        )


@dataclass(frozen=True)
class SourceFastToken:
    size_bytes: int
    mtime_ns: int
    ctime_ns: int | None
    device: int | None
    inode: int | None
    parquet_row_count: int
    parquet_row_group_count: int
    arrow_schema_digest: str
    footer_digest: str

    def cache_key(self) -> tuple[object, ...]:
        return tuple(asdict(self).values())


@dataclass(frozen=True)
class SourceFingerprint:
    content_sha256: str
    fast_token: SourceFastToken
    required_columns: tuple[tuple[str, str], ...]
    resolved_path: str


def _parquet_footer_digest(path: Path) -> str:
    size = path.stat().st_size
    if size < 8:
        raise RegionProfileValidationError(f"{path} is too small to be a Parquet file")
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        if trailer[4:] != b"PAR1":
            raise RegionProfileValidationError(f"{path} has an invalid Parquet trailer")
        metadata_length = int.from_bytes(trailer[:4], "little", signed=False)
        if metadata_length > size - 8:
            raise RegionProfileValidationError(f"{path} has an invalid Parquet footer")
        stream.seek(-(metadata_length + 8), os.SEEK_END)
        footer = stream.read(metadata_length + 8)
    return hashlib.sha256(footer).hexdigest()


def _schema_digest(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _required_source_types(schema: pa.Schema) -> tuple[tuple[str, str], ...]:
    missing = [name for name in REQUIRED_SOURCE_COLUMNS if name not in schema.names]
    if missing:
        raise RegionProfileValidationError(
            "Source Parquet is missing regional-profile columns.",
            issues=[f"missing column: {name}" for name in missing],
        )
    fields = tuple(
        (name, str(schema.field(name).type)) for name in REQUIRED_SOURCE_COLUMNS
    )
    issues: list[str] = []
    for name in ("node_id", "type", "parent_id", "region_id"):
        if not pa.types.is_integer(schema.field(name).type):
            issues.append(f"column {name!r} must have an integer type")
    for name in ("x", "y", "z"):
        if not (
            pa.types.is_floating(schema.field(name).type)
            or pa.types.is_integer(schema.field(name).type)
        ):
            issues.append(f"column {name!r} must have a numeric type")
    if not (
        pa.types.is_string(schema.field("file_id").type)
        or pa.types.is_large_string(schema.field("file_id").type)
    ):
        issues.append("column 'file_id' must have a string type")
    if issues:
        raise RegionProfileValidationError(
            "Source Parquet has incompatible regional-profile column types.",
            issues=issues,
        )
    return fields


def source_fast_token(
    path: str | Path,
) -> tuple[SourceFastToken, tuple[tuple[str, str], ...]]:
    """Read filesystem and Parquet-footer identity without scanning node data."""
    source = Path(path)
    stat = source.stat()
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    required_types = _required_source_types(schema)
    token = SourceFastToken(
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        ctime_ns=int(stat.st_ctime_ns) if hasattr(stat, "st_ctime_ns") else None,
        device=int(stat.st_dev) if hasattr(stat, "st_dev") else None,
        inode=int(stat.st_ino) if hasattr(stat, "st_ino") else None,
        parquet_row_count=int(parquet.metadata.num_rows),
        parquet_row_group_count=int(parquet.metadata.num_row_groups),
        arrow_schema_digest=_schema_digest(schema),
        footer_digest=_parquet_footer_digest(source),
    )
    return token, required_types


def _streaming_sha256(
    path: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            _check_cancelled(cancel_check)
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(
    path: str | Path,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> SourceFingerprint:
    """Return the full build-time source fingerprint pinned by Stage 0."""
    source = Path(path)
    token, required_types = source_fast_token(source)
    key = token.cache_key()
    content_digest = _SOURCE_SHA256_CACHE.get(key)
    if content_digest is None:
        content_digest = _streaming_sha256(source, cancel_check=cancel_check)
        _SOURCE_SHA256_CACHE[key] = content_digest
    return SourceFingerprint(
        content_sha256=content_digest,
        fast_token=token,
        required_columns=required_types,
        resolved_path=str(source.resolve()),
    )


@dataclass(frozen=True)
class RegionProfileBuildSummary:
    source_row_count: int
    neuron_count: int
    profile_row_count: int
    batch_count: int
    total_edge_count: int
    zero_length_edge_count: int
    total_cable_length_um: float
    allocated_cable_length_um: float
    node_count: int
    terminus_count: int
    soma_laterality_counts: Mapping[str, int]
    elapsed_seconds: float


@dataclass(frozen=True)
class RegionProfileMetadata:
    profile_format_version: int
    builder_algorithm_version: str
    source: SourceFingerprint
    atlas: AtlasIdentity
    length_method: str
    length_absolute_tolerance_um: float
    length_relative_tolerance: float
    compartment_definition: str
    terminus_definition: str
    build_timestamp_utc: str
    build_summary: RegionProfileBuildSummary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RegionProfileMetadata:
        source_raw = raw["source"]
        token_raw = source_raw["fast_token"]
        atlas_raw = raw["atlas"]
        summary_raw = raw["build_summary"]
        source = SourceFingerprint(
            content_sha256=str(source_raw["content_sha256"]),
            fast_token=SourceFastToken(**token_raw),
            required_columns=tuple(
                (str(name), str(data_type))
                for name, data_type in source_raw["required_columns"]
            ),
            resolved_path=str(source_raw.get("resolved_path", "")),
        )
        atlas = AtlasIdentity(
            atlas_name=str(atlas_raw["atlas_name"]),
            atlas_version=str(atlas_raw["atlas_version"]),
            resolution_um=tuple(float(value) for value in atlas_raw["resolution_um"]),
            annotation_shape=tuple(
                int(value) for value in atlas_raw["annotation_shape"]
            ),
            annotation_dtype=str(atlas_raw["annotation_dtype"]),
            annotation_digest=str(atlas_raw["annotation_digest"]),
            hierarchy_digest=str(atlas_raw["hierarchy_digest"]),
            left_right_axis=int(atlas_raw["left_right_axis"]),
            midline_um=float(atlas_raw["midline_um"]),
            lower_hemisphere_code=(
                None
                if atlas_raw.get("lower_hemisphere_code") is None
                else int(atlas_raw["lower_hemisphere_code"])
            ),
            upper_hemisphere_code=(
                None
                if atlas_raw.get("upper_hemisphere_code") is None
                else int(atlas_raw["upper_hemisphere_code"])
            ),
        )
        summary = RegionProfileBuildSummary(
            source_row_count=int(summary_raw["source_row_count"]),
            neuron_count=int(summary_raw["neuron_count"]),
            profile_row_count=int(summary_raw["profile_row_count"]),
            batch_count=int(summary_raw["batch_count"]),
            total_edge_count=int(summary_raw["total_edge_count"]),
            zero_length_edge_count=int(summary_raw["zero_length_edge_count"]),
            total_cable_length_um=float(summary_raw["total_cable_length_um"]),
            allocated_cable_length_um=float(summary_raw["allocated_cable_length_um"]),
            node_count=int(summary_raw["node_count"]),
            terminus_count=int(summary_raw["terminus_count"]),
            soma_laterality_counts={
                str(key): int(value)
                for key, value in summary_raw["soma_laterality_counts"].items()
            },
            elapsed_seconds=float(summary_raw["elapsed_seconds"]),
        )
        return cls(
            profile_format_version=int(raw["profile_format_version"]),
            builder_algorithm_version=str(raw["builder_algorithm_version"]),
            source=source,
            atlas=atlas,
            length_method=str(raw["length_method"]),
            length_absolute_tolerance_um=float(raw["length_absolute_tolerance_um"]),
            length_relative_tolerance=float(raw["length_relative_tolerance"]),
            compartment_definition=str(raw["compartment_definition"]),
            terminus_definition=str(raw["terminus_definition"]),
            build_timestamp_utc=str(raw["build_timestamp_utc"]),
            build_summary=summary,
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> RegionProfileMetadata:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        raw = json.loads(payload)
        if not isinstance(raw, Mapping):
            raise TypeError("regional-profile metadata must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class RegionProfileInspection:
    sidecar_path: Path
    valid: bool
    compatible: bool
    row_count: int
    metadata: RegionProfileMetadata | None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NeuronData:
    file_id: str
    node_ids: NDArray[np.int64]
    node_types: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    parent_ids: NDArray[np.int64]
    region_ids: NDArray[np.int64]


@dataclass(frozen=True)
class _ProfileRow:
    file_id: str
    region_id: int
    laterality: str
    compartment: str
    cable_length_um: float
    node_count: int
    terminus_count: int


@dataclass(frozen=True)
class _NeuronResult:
    rows: tuple[_ProfileRow, ...]
    edge_count: int
    zero_length_edge_count: int
    cable_length_um: float
    allocated_cable_length_um: float
    node_count: int
    terminus_count: int
    soma_laterality: str


def _relative_laterality(
    physical_side: AtlasAxisSide,
    soma_side: AtlasAxisSide | None,
) -> Laterality:
    if physical_side is AtlasAxisSide.MIDLINE:
        return Laterality.MIDLINE
    if soma_side not in (AtlasAxisSide.LOWER, AtlasAxisSide.UPPER):
        return Laterality.UNKNOWN
    if physical_side is soma_side:
        return Laterality.IPSILATERAL
    return Laterality.CONTRALATERAL


def _node_axis_side(coordinate: float, midline_um: float) -> AtlasAxisSide:
    if coordinate == midline_um:
        return AtlasAxisSide.MIDLINE
    if coordinate < midline_um:
        return AtlasAxisSide.LOWER
    return AtlasAxisSide.UPPER


def _soma_axis_side(
    coordinate: float,
    *,
    midline_um: float,
    extent_um: float,
) -> AtlasAxisSide | None:
    if not 0.0 <= coordinate < extent_um:
        return None
    return _node_axis_side(coordinate, midline_um)


@dataclass
class _MutableBucket:
    cable_lengths: list[float] = field(default_factory=list)
    node_count: int = 0
    terminus_count: int = 0


def _process_neuron(
    neuron: _NeuronData,
    atlas: RegionalProfileAtlas,
    traverser: AtlasVoxelTraverser,
    *,
    absolute_tolerance_um: float,
    relative_tolerance: float,
) -> _NeuronResult:
    file_id = neuron.file_id
    node_count = len(neuron.node_ids)
    issues: list[str] = []
    if node_count == 0:
        issues.append(f"{file_id}: neuron has no nodes")
    if len(np.unique(neuron.node_ids)) != node_count:
        issues.append(f"{file_id}: duplicate node_id values")
    if not np.all(np.isfinite(neuron.coordinates)):
        issues.append(f"{file_id}: coordinates must all be finite")
    if np.any(neuron.region_ids < 0):
        issues.append(f"{file_id}: region_id values must be non-negative")
    unknown_region_ids = sorted(
        {int(value) for value in neuron.region_ids} - {0} - set(atlas.structures)
    )
    if unknown_region_ids:
        sample = ", ".join(str(value) for value in unknown_region_ids[:5])
        issues.append(
            f"{file_id}: region_id values absent from atlas hierarchy: {sample}"
        )

    root_indices = np.flatnonzero(neuron.parent_ids == -1)
    if len(root_indices) != 1:
        issues.append(
            f"{file_id}: expected exactly one root, found {len(root_indices)}"
        )
    soma_indices = np.flatnonzero(neuron.node_types == 1)
    if len(soma_indices) != 1:
        issues.append(
            f"{file_id}: expected exactly one soma, found {len(soma_indices)}"
        )
    if len(soma_indices) == 1 and neuron.parent_ids[soma_indices[0]] != -1:
        issues.append(f"{file_id}: the soma node must be the root (parent_id = -1)")

    node_index = {int(node_id): index for index, node_id in enumerate(neuron.node_ids)}
    dangling = sorted(
        {
            int(parent_id)
            for parent_id in neuron.parent_ids
            if parent_id != -1 and int(parent_id) not in node_index
        }
    )
    if dangling:
        sample = ", ".join(str(value) for value in dangling[:5])
        issues.append(f"{file_id}: dangling parent_id values: {sample}")
    if len(root_indices) == 1 and not dangling and node_count:
        children: defaultdict[int, list[int]] = defaultdict(list)
        for index, parent_id in enumerate(neuron.parent_ids):
            if int(parent_id) != -1:
                children[int(parent_id)].append(int(neuron.node_ids[index]))
        reachable: set[int] = set()
        pending = [int(neuron.node_ids[int(root_indices[0])])]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(children.get(current, ()))
        if len(reachable) != node_count:
            issues.append(
                f"{file_id}: {node_count - len(reachable)} node(s) are not "
                "connected to the root (the graph may contain a cycle)"
            )
    if issues:
        raise RegionProfileValidationError(
            f"Neuron graph validation failed for {file_id}.", issues=issues
        )

    soma_index = int(soma_indices[0])
    axis = atlas.left_right_axis
    soma_side = _soma_axis_side(
        float(neuron.coordinates[soma_index, axis]),
        midline_um=atlas.midline_um,
        extent_um=atlas.left_right_extent_um,
    )
    soma_laterality = (
        Laterality.UNKNOWN.value
        if soma_side is None
        else (
            Laterality.MIDLINE.value
            if soma_side is AtlasAxisSide.MIDLINE
            else Laterality.IPSILATERAL.value
        )
    )

    child_node_ids = {
        int(parent_id) for parent_id in neuron.parent_ids if int(parent_id) != -1
    }
    terminus_mask = np.asarray(
        [
            int(node_id) not in child_node_ids and int(node_type) != 1
            for node_id, node_type in zip(
                neuron.node_ids, neuron.node_types, strict=True
            )
        ],
        dtype=bool,
    )

    buckets: defaultdict[tuple[int, str, str], _MutableBucket] = defaultdict(
        _MutableBucket
    )
    for index in range(node_count):
        compartment = (
            Compartment.SOMA
            if int(neuron.node_types[index]) == 1
            else Compartment.NEURITE
        )
        physical_side = _node_axis_side(
            float(neuron.coordinates[index, axis]), atlas.midline_um
        )
        laterality = _relative_laterality(physical_side, soma_side)
        key = (int(neuron.region_ids[index]), laterality.value, compartment.value)
        bucket = buckets[key]
        bucket.node_count += 1
        if terminus_mask[index]:
            bucket.terminus_count += 1

    expected_lengths: list[float] = []
    allocated_lengths: list[float] = []
    edge_count = 0
    zero_length_edges = 0
    for child_index, raw_parent_id in enumerate(neuron.parent_ids):
        parent_id = int(raw_parent_id)
        if parent_id == -1:
            continue
        edge_count += 1
        parent_index = node_index[parent_id]
        child_coord = neuron.coordinates[child_index]
        parent_coord = neuron.coordinates[parent_index]
        expected_length = math.dist(child_coord, parent_coord)
        expected_lengths.append(expected_length)
        if expected_length == 0.0:
            zero_length_edges += 1
            continue
        portions = traverser.traverse_validated(child_coord, parent_coord)
        compartment = (
            Compartment.SOMA
            if int(neuron.node_types[child_index]) == 1
            else Compartment.NEURITE
        )
        for portion in portions:
            if portion.region_id != 0 and portion.region_id not in atlas.structures:
                raise RegionProfileValidationError(
                    f"Atlas annotation contains region {portion.region_id}, which "
                    "is absent from its hierarchy."
                )
            laterality = _relative_laterality(portion.axis_side, soma_side)
            key = (portion.region_id, laterality.value, compartment.value)
            buckets[key].cable_lengths.append(portion.length_um)
            allocated_lengths.append(portion.length_um)

    expected_total = math.fsum(expected_lengths)
    allocated_total = math.fsum(allocated_lengths)
    tolerance = max(
        absolute_tolerance_um,
        relative_tolerance * max(1.0, abs(expected_total)),
    )
    if not math.isclose(
        expected_total,
        allocated_total,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance_um,
    ):
        raise RegionProfileValidationError(
            f"Cable-length conservation failed for {file_id}.",
            issues=[
                f"edge length = {expected_total:.17g} um",
                f"allocated length = {allocated_total:.17g} um",
                f"allowed tolerance = {tolerance:.3g} um",
            ],
        )

    rows = tuple(
        _ProfileRow(
            file_id=file_id,
            region_id=region_id,
            laterality=laterality,
            compartment=compartment,
            cable_length_um=math.fsum(bucket.cable_lengths),
            node_count=bucket.node_count,
            terminus_count=bucket.terminus_count,
        )
        for (region_id, laterality, compartment), bucket in sorted(buckets.items())
    )
    if sum(row.node_count for row in rows) != node_count:
        raise RegionProfileValidationError(
            f"Node-count conservation failed for {file_id}."
        )
    total_termini = int(np.count_nonzero(terminus_mask))
    if sum(row.terminus_count for row in rows) != total_termini:
        raise RegionProfileValidationError(
            f"Terminus-count conservation failed for {file_id}."
        )
    return _NeuronResult(
        rows=rows,
        edge_count=edge_count,
        zero_length_edge_count=zero_length_edges,
        cable_length_um=expected_total,
        allocated_cable_length_um=allocated_total,
        node_count=node_count,
        terminus_count=total_termini,
        soma_laterality=soma_laterality,
    )


def _escape_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").replace("'", "''")


def _neuron_row_counts(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
) -> list[tuple[str, int]]:
    query = f"""
        SELECT CAST(file_id AS VARCHAR) AS file_id, COUNT(*)::BIGINT AS row_count
        FROM read_parquet('{_escape_path(source_path)}')
        GROUP BY file_id
        ORDER BY file_id
    """
    rows = connection.execute(query).fetchall()
    if not rows:
        raise RegionProfileValidationError("Source Parquet contains no neuron rows.")
    issues = (
        ["source contains a null or empty file_id"]
        if any(row[0] is None or not str(row[0]) for row in rows)
        else []
    )
    if issues:
        raise RegionProfileValidationError("Invalid neuron identity.", issues=issues)
    return [(str(file_id), int(row_count)) for file_id, row_count in rows]


def _complete_neuron_batches(
    counts: Sequence[tuple[str, int]],
    *,
    max_neurons: int,
    max_rows: int,
) -> tuple[tuple[str, ...], ...]:
    if max_neurons < 1 or max_rows < 1:
        raise ValueError("batch neuron and row limits must be positive")
    batches: list[tuple[str, ...]] = []
    current: list[str] = []
    current_rows = 0
    for file_id, row_count in counts:
        would_overflow = current and (
            len(current) >= max_neurons or current_rows + row_count > max_rows
        )
        if would_overflow:
            batches.append(tuple(current))
            current = []
            current_rows = 0
        current.append(file_id)
        current_rows += row_count
        if row_count > max_rows or len(current) >= max_neurons:
            batches.append(tuple(current))
            current = []
            current_rows = 0
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _read_complete_batch(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
    file_ids: Sequence[str],
) -> pd.DataFrame:
    selection = pa.Table.from_pydict({"file_id": list(file_ids)})
    connection.register("_region_profile_selected_files", selection)
    try:
        query = f"""
            SELECT
                CAST(p.file_id AS VARCHAR) AS file_id,
                CAST(p.node_id AS BIGINT) AS node_id,
                CAST(p.type AS BIGINT) AS type,
                CAST(p.x AS DOUBLE) AS x,
                CAST(p.y AS DOUBLE) AS y,
                CAST(p.z AS DOUBLE) AS z,
                CAST(p.parent_id AS BIGINT) AS parent_id,
                CAST(p.region_id AS BIGINT) AS region_id
            FROM read_parquet('{_escape_path(source_path)}') p
            JOIN _region_profile_selected_files f
              ON CAST(p.file_id AS VARCHAR) = f.file_id
        """
        frame = connection.execute(query).fetch_df()
    finally:
        connection.unregister("_region_profile_selected_files")
    return _validate_batch_frame(frame, file_ids)


def _validate_batch_frame(
    frame: pd.DataFrame,
    file_ids: Sequence[str],
) -> pd.DataFrame:
    """Validate one materialized batch contains complete required values."""
    if len(frame) == 0:
        raise RegionProfileValidationError(
            "A complete-neuron batch unexpectedly returned no source rows."
        )
    null_columns = [name for name in frame.columns if frame[name].isna().any()]
    if null_columns:
        issues = []
        coordinate_columns = sorted(set(null_columns) & {"x", "y", "z"})
        if coordinate_columns:
            issues.append(
                "coordinates must all be finite; null values found in "
                + ", ".join(repr(name) for name in coordinate_columns)
            )
        issues.extend(
            f"null values in column {name!r}"
            for name in null_columns
            if name not in coordinate_columns
        )
        raise RegionProfileValidationError(
            "Source rows contain null values required by the regional profile.",
            issues=issues,
        )
    observed = {str(value) for value in frame["file_id"].unique()}
    missing = sorted(set(file_ids) - observed)
    if missing:
        raise RegionProfileValidationError(
            "A complete-neuron batch did not contain every requested file_id.",
            issues=[f"missing file_id: {file_id}" for file_id in missing],
        )
    return frame


def _stage_complete_batches(
    source_path: Path,
    staging_path: Path,
    batches: Sequence[Sequence[str]],
    *,
    row_group_size: int,
    max_batch_rows: int,
    cancel_check: Callable[[], bool] | None,
    progress_callback: Callable[[int, int], None] | None,
) -> None:
    """Partition required source columns in one cancellable Parquet scan.

    Partitioning once avoids rescanning a large source Parquet for every
    complete-neuron batch.  Arrow's dataset writer bounds open files and emits
    one directory per batch while the generator checks cancellation between
    source record batches.
    """
    file_to_batch = {
        file_id: batch_index
        for batch_index, file_ids in enumerate(batches)
        for file_id in file_ids
    }
    source_parquet = pq.ParquetFile(source_path)
    required_schema = pa.schema(
        [source_parquet.schema_arrow.field(name) for name in REQUIRED_SOURCE_COLUMNS]
    )
    staging_schema = required_schema.append(
        pa.field("_profile_batch_id", pa.int32(), nullable=False)
    )

    total_rows = int(source_parquet.metadata.num_rows)
    staged_row_count = 0

    def staged_batches():
        nonlocal staged_row_count
        for source_batch in source_parquet.iter_batches(
            batch_size=min(max_batch_rows, 262_144),
            columns=list(REQUIRED_SOURCE_COLUMNS),
            use_threads=True,
        ):
            _check_cancelled(cancel_check)
            file_ids = source_batch.column(
                source_batch.schema.get_field_index("file_id")
            )
            encoded = pc.dictionary_encode(file_ids)
            if encoded.null_count:
                raise RegionProfileValidationError(
                    "Source Parquet contains a null file_id."
                )
            try:
                dictionary_batches = pa.array(
                    [
                        file_to_batch[str(file_id)]
                        for file_id in encoded.dictionary.to_pylist()
                    ],
                    type=pa.int32(),
                )
            except KeyError as exc:
                raise RegionProfileValidationError(
                    f"Source staging found an unregistered file_id: {exc.args[0]}"
                ) from exc
            batch_ids = pc.take(dictionary_batches, encoded.indices)
            staged_row_count += source_batch.num_rows
            if progress_callback is not None:
                progress_callback(staged_row_count, total_rows)
            yield source_batch.append_column("_profile_batch_id", batch_ids)

    file_format = ds.ParquetFileFormat()
    write_options = file_format.make_write_options(
        compression="zstd",
        use_dictionary=["file_id"],
    )
    ds.write_dataset(
        staged_batches(),
        staging_path,
        schema=staging_schema,
        format=file_format,
        file_options=write_options,
        partitioning=["_profile_batch_id"],
        basename_template="part-{i}.parquet",
        max_open_files=64,
        max_rows_per_file=max(max_batch_rows, row_group_size),
        max_rows_per_group=row_group_size,
        existing_data_behavior="error",
    )


def _read_staged_batch(
    connection: duckdb.DuckDBPyConnection,
    staging_path: Path,
    batch_index: int,
    file_ids: Sequence[str],
) -> pd.DataFrame:
    partition_path = staging_path / str(batch_index)
    if not partition_path.is_dir():
        raise RegionProfileValidationError(
            f"Staged source batch {batch_index} is missing."
        )
    query = f"""
        SELECT
            CAST(file_id AS VARCHAR) AS file_id,
            CAST(node_id AS BIGINT) AS node_id,
            CAST(type AS BIGINT) AS type,
            CAST(x AS DOUBLE) AS x,
            CAST(y AS DOUBLE) AS y,
            CAST(z AS DOUBLE) AS z,
            CAST(parent_id AS BIGINT) AS parent_id,
            CAST(region_id AS BIGINT) AS region_id
        FROM read_parquet('{_escape_path(partition_path / "*.parquet")}')
    """
    return _validate_batch_frame(connection.execute(query).fetch_df(), file_ids)


def _frame_neurons(frame: pd.DataFrame) -> tuple[_NeuronData, ...]:
    neurons: list[_NeuronData] = []
    for file_id, group in frame.groupby("file_id", sort=False, observed=True):
        neurons.append(
            _NeuronData(
                file_id=str(file_id),
                node_ids=group["node_id"].to_numpy(dtype=np.int64, copy=True),
                node_types=group["type"].to_numpy(dtype=np.int64, copy=True),
                coordinates=group[["x", "y", "z"]].to_numpy(
                    dtype=np.float64, copy=True
                ),
                parent_ids=group["parent_id"].to_numpy(dtype=np.int64, copy=True),
                region_ids=group["region_id"].to_numpy(dtype=np.int64, copy=True),
            )
        )
    return tuple(neurons)


def _rows_to_table(rows: Sequence[_ProfileRow], schema: pa.Schema) -> pa.Table:
    payload = {
        "file_id": [row.file_id for row in rows],
        "region_id": [row.region_id for row in rows],
        "laterality": [row.laterality for row in rows],
        "compartment": [row.compartment for row in rows],
        "cable_length_um": [row.cable_length_um for row in rows],
        "node_count": [row.node_count for row in rows],
        "terminus_count": [row.terminus_count for row in rows],
    }
    return pa.Table.from_pydict(payload, schema=schema)


def _temporary_path(destination: Path, label: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.{label}.",
        suffix=".parquet",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RegionProfileBuildCancelled(
            "Regional-profile construction was cancelled."
        )


def _emit_progress(
    callback: Callable[[str, int, int], None] | None,
    message: str,
    current: int,
    total: int,
) -> None:
    if callback is not None:
        callback(message, current, total)


def _fsync_file(path: Path) -> None:
    # Windows' _commit/FlushFileBuffers requires a writable handle. Reopen
    # without truncating the completed Parquet, whose writer is already closed.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def default_region_profile_path(source_path: str | Path) -> Path:
    source = Path(source_path)
    return source.with_name(f"{source.stem}.region_profile.parquet")


def _write_sorted_profile(
    raw_path: Path,
    destination: Path,
    metadata: RegionProfileMetadata,
    *,
    row_group_size: int,
    cancel_check: Callable[[], bool] | None,
) -> int:
    schema = REGION_PROFILE_SCHEMA.with_metadata(
        {REGION_PROFILE_METADATA_KEY: metadata.to_json().encode("utf-8")}
    )
    connection = duckdb.connect()
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        query = f"""
            SELECT *
            FROM read_parquet('{_escape_path(raw_path)}')
            ORDER BY region_id, laterality, compartment, file_id
        """
        reader = connection.execute(query).fetch_record_batch(row_group_size)
        writer = pq.ParquetWriter(
            destination,
            schema,
            compression="zstd",
            use_dictionary=["laterality", "compartment"],
        )
        for record_batch in reader:
            _check_cancelled(cancel_check)
            table = pa.Table.from_batches([record_batch]).cast(schema, safe=True)
            writer.write_table(table, row_group_size=row_group_size)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
        connection.close()
    return row_count


def _metadata_from_schema(schema: pa.Schema) -> RegionProfileMetadata:
    payload = (schema.metadata or {}).get(REGION_PROFILE_METADATA_KEY)
    if payload is None:
        raise RegionProfileValidationError(
            "Parquet does not contain regional-profile metadata."
        )
    try:
        return RegionProfileMetadata.from_json(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RegionProfileValidationError(
            "Regional-profile metadata is malformed."
        ) from exc


def _profile_schema_issues(schema: pa.Schema) -> list[str]:
    issues: list[str] = []
    for expected in REGION_PROFILE_SCHEMA:
        try:
            actual = schema.field(expected.name)
        except KeyError:
            issues.append(f"missing profile column: {expected.name}")
            continue
        if actual.type != expected.type:
            issues.append(
                f"profile column {expected.name!r} has type {actual.type}, "
                f"expected {expected.type}"
            )
    extra = sorted(set(schema.names) - set(REGION_PROFILE_SCHEMA.names))
    if extra:
        issues.append(f"unexpected profile columns: {', '.join(extra)}")
    return issues


def _validate_profile_contents(
    sidecar_path: Path,
    metadata: RegionProfileMetadata,
) -> list[str]:
    issues: list[str] = []
    connection = duckdb.connect()
    try:
        path_sql = _escape_path(sidecar_path)
        row = connection.execute(
            f"""
            SELECT
                COUNT(*)::BIGINT,
                COUNT(DISTINCT (file_id, region_id, laterality, compartment))::BIGINT,
                COALESCE(SUM(node_count), 0)::BIGINT,
                COALESCE(SUM(terminus_count), 0)::BIGINT,
                COALESCE(SUM(cable_length_um), 0)::DOUBLE,
                COUNT(*) FILTER (
                    WHERE region_id < 0 OR cable_length_um < 0
                       OR node_count < 0 OR terminus_count < 0
                       OR laterality NOT IN ('ipsilateral', 'contralateral', 'midline', 'unknown')
                       OR compartment NOT IN ('soma', 'neurite')
                )::BIGINT
            FROM read_parquet('{path_sql}')
            """
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    row_count, unique_count, nodes, termini, cable, invalid_count = row
    summary = metadata.build_summary
    if int(row_count) != summary.profile_row_count:
        issues.append(
            f"profile row count {row_count} does not match metadata "
            f"{summary.profile_row_count}"
        )
    if int(unique_count) != int(row_count):
        issues.append("profile contains duplicate canonical bucket keys")
    if int(nodes) != summary.node_count:
        issues.append(
            f"node total {nodes} does not match metadata {summary.node_count}"
        )
    if int(termini) != summary.terminus_count:
        issues.append(
            f"terminus total {termini} does not match metadata {summary.terminus_count}"
        )
    if not math.isclose(
        float(cable),
        summary.allocated_cable_length_um,
        rel_tol=metadata.length_relative_tolerance,
        abs_tol=metadata.length_absolute_tolerance_um,
    ):
        issues.append(
            f"cable total {cable} does not match metadata "
            f"{summary.allocated_cable_length_um}"
        )
    if int(invalid_count):
        issues.append(f"profile contains {invalid_count} invalid bucket rows")
    return issues


def _source_compatibility_issues(
    source_path: Path,
    expected: SourceFingerprint,
) -> list[str]:
    try:
        token, required_types = source_fast_token(source_path)
    except (OSError, RegionProfileValidationError) as exc:
        return [f"source validation failed: {exc}"]
    if required_types != expected.required_columns:
        return ["source required-column types differ from the profile build"]
    if token == expected.fast_token:
        return []
    digest = _SOURCE_SHA256_CACHE.get(token.cache_key())
    if digest is None:
        digest = _streaming_sha256(source_path)
        _SOURCE_SHA256_CACHE[token.cache_key()] = digest
    if digest != expected.content_sha256:
        return ["source Parquet content differs from the profile build"]
    return []


def _atlas_compatibility_issues(
    atlas: RegionalProfileAtlas,
    expected: AtlasIdentity,
) -> list[str]:
    actual = AtlasIdentity.from_profile_atlas(atlas)
    issues: list[str] = []
    for name in asdict(expected):
        if getattr(actual, name) != getattr(expected, name):
            issues.append(f"atlas {name} differs from the profile build")
    return issues


def inspect_region_profile(
    sidecar_path: str | Path,
    *,
    source_path: str | Path | None = None,
    atlas: RegionalProfileAtlas | object | None = None,
    validate_contents: bool = False,
) -> RegionProfileInspection:
    """Inspect sidecar metadata and optional compatibility without loading rows.

    Setting ``validate_contents`` performs aggregate checks over the compact
    sidecar itself.  Source compatibility first uses only stat/footer data and
    hashes full source content only if that fast token changed.
    """
    sidecar = Path(sidecar_path)
    issues: list[str] = []
    metadata: RegionProfileMetadata | None = None
    row_count = 0
    try:
        parquet = pq.ParquetFile(sidecar)
        row_count = int(parquet.metadata.num_rows)
        issues.extend(_profile_schema_issues(parquet.schema_arrow))
        metadata = _metadata_from_schema(parquet.schema_arrow)
    except (OSError, pa.ArrowException, RegionProfileValidationError) as exc:
        issues.append(str(exc))
        return RegionProfileInspection(
            sidecar_path=sidecar,
            valid=False,
            compatible=False,
            row_count=row_count,
            metadata=metadata,
            issues=tuple(issues),
        )

    if metadata.profile_format_version != PROFILE_FORMAT_VERSION:
        issues.append("profile format version is unsupported")
    if metadata.builder_algorithm_version != BUILDER_ALGORITHM_VERSION:
        issues.append("profile builder version is incompatible")
    if metadata.length_method != LENGTH_METHOD:
        issues.append("profile cable-length method is incompatible")
    if row_count != metadata.build_summary.profile_row_count:
        issues.append("Parquet row count differs from profile metadata")
    if validate_contents:
        issues.extend(_validate_profile_contents(sidecar, metadata))

    structural_issue_count = len(issues)
    if source_path is not None:
        issues.extend(_source_compatibility_issues(Path(source_path), metadata.source))
    if atlas is not None:
        profile_atlas = (
            atlas
            if isinstance(atlas, RegionalProfileAtlas)
            else RegionalProfileAtlas.from_atlas(atlas)
        )
        issues.extend(_atlas_compatibility_issues(profile_atlas, metadata.atlas))
    return RegionProfileInspection(
        sidecar_path=sidecar,
        valid=structural_issue_count == 0,
        compatible=len(issues) == 0,
        row_count=row_count,
        metadata=metadata,
        issues=tuple(issues),
    )


def build_region_profile(
    source_path: str | Path,
    atlas: RegionalProfileAtlas | object,
    *,
    output_path: str | Path | None = None,
    max_batch_neurons: int = DEFAULT_MAX_BATCH_NEURONS,
    max_batch_rows: int = DEFAULT_MAX_BATCH_ROWS,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    length_absolute_tolerance_um: float = DEFAULT_LENGTH_ABSOLUTE_TOLERANCE_UM,
    length_relative_tolerance: float = DEFAULT_LENGTH_RELATIVE_TOLERANCE,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> RegionProfileMetadata:
    """Build, fully validate, and atomically publish a regional-profile sidecar."""
    started = perf_counter()
    source = Path(source_path)
    destination = (
        default_region_profile_path(source)
        if output_path is None
        else Path(output_path)
    )
    if source.resolve() == destination.resolve():
        raise ValueError("output_path must not replace the source Parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile_atlas = (
        atlas
        if isinstance(atlas, RegionalProfileAtlas)
        else RegionalProfileAtlas.from_atlas(atlas)
    )
    if row_group_size < 1:
        raise ValueError("row_group_size must be positive")
    if length_absolute_tolerance_um < 0 or length_relative_tolerance < 0:
        raise ValueError("cable-length tolerances must be non-negative")

    raw_path = _temporary_path(destination, "raw")
    final_path = _temporary_path(destination, "pending")
    staging_root: Path | None = None
    staging_path: Path | None = None
    raw_writer: pq.ParquetWriter | None = None
    published = False
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        _check_cancelled(cancel_check)
        _emit_progress(progress_callback, "Fingerprinting source Parquet...", 0, 1)
        fingerprint = source_fingerprint(source, cancel_check=cancel_check)
        atlas_identity = AtlasIdentity.from_profile_atlas(profile_atlas)
        connection = duckdb.connect()
        counts = _neuron_row_counts(connection, source)
        source_row_count = sum(row_count for _, row_count in counts)
        if source_row_count != fingerprint.fast_token.parquet_row_count:
            raise RegionProfileValidationError(
                "Grouped source row count does not match the Parquet footer."
            )
        batches = _complete_neuron_batches(
            counts,
            max_neurons=max_batch_neurons,
            max_rows=max_batch_rows,
        )
        staging_steps = int(len(batches) > 1)
        total_steps = len(batches) + staging_steps + 2
        _emit_progress(
            progress_callback,
            f"Building regional profiles for {len(counts):,} neurons...",
            0,
            total_steps,
        )

        if staging_steps:
            _check_cancelled(cancel_check)
            staging_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.source_batches.",
                    dir=destination.parent,
                )
            )
            staging_path = staging_root / "partitions"
            _emit_progress(
                progress_callback,
                "Staging complete-neuron source batches in one scan...",
                0,
                total_steps,
            )
            _stage_complete_batches(
                source,
                staging_path,
                batches,
                row_group_size=row_group_size,
                max_batch_rows=max_batch_rows,
                cancel_check=cancel_check,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda staged, total: _emit_progress(
                        progress_callback,
                        f"Staging source rows: {staged:,}/{total:,}...",
                        0,
                        total_steps,
                    )
                ),
            )
            _emit_progress(
                progress_callback,
                "Complete-neuron source batches staged.",
                1,
                total_steps,
            )

        profile_rows = 0
        edge_count = 0
        zero_length_edges = 0
        expected_lengths: list[float] = []
        allocated_lengths: list[float] = []
        total_nodes = 0
        total_termini = 0
        soma_lateralities: Counter[str] = Counter()
        traverser = AtlasVoxelTraverser(
            annotation=profile_atlas.annotation,
            resolution_um=profile_atlas.resolution_um,
            left_right_axis=profile_atlas.left_right_axis,
            midline_um=profile_atlas.midline_um,
        )

        raw_writer = pq.ParquetWriter(
            raw_path,
            _RAW_PROFILE_SCHEMA,
            compression="zstd",
            use_dictionary=["laterality", "compartment", "file_id"],
        )
        count_by_file_id = dict(counts)
        for batch_index, batch_file_ids in enumerate(batches, start=1):
            _check_cancelled(cancel_check)
            frame = (
                _read_complete_batch(connection, source, batch_file_ids)
                if staging_path is None
                else _read_staged_batch(
                    connection,
                    staging_path,
                    batch_index - 1,
                    batch_file_ids,
                )
            )
            expected_batch_rows = sum(
                count_by_file_id[file_id] for file_id in batch_file_ids
            )
            if len(frame) != expected_batch_rows:
                raise RegionProfileValidationError(
                    "A batch did not contain the complete rows for its file_ids.",
                    issues=[
                        f"expected {expected_batch_rows} rows, received {len(frame)}"
                    ],
                )
            batch_rows: list[_ProfileRow] = []
            for neuron in _frame_neurons(frame):
                _check_cancelled(cancel_check)
                result = _process_neuron(
                    neuron,
                    profile_atlas,
                    traverser,
                    absolute_tolerance_um=length_absolute_tolerance_um,
                    relative_tolerance=length_relative_tolerance,
                )
                batch_rows.extend(result.rows)
                edge_count += result.edge_count
                zero_length_edges += result.zero_length_edge_count
                expected_lengths.append(result.cable_length_um)
                allocated_lengths.append(result.allocated_cable_length_um)
                total_nodes += result.node_count
                total_termini += result.terminus_count
                soma_lateralities[result.soma_laterality] += 1
            raw_writer.write_table(
                _rows_to_table(batch_rows, _RAW_PROFILE_SCHEMA),
                row_group_size=row_group_size,
            )
            profile_rows += len(batch_rows)
            _emit_progress(
                progress_callback,
                f"Built batch {batch_index:,} of {len(batches):,}...",
                batch_index + staging_steps,
                total_steps,
            )

        raw_writer.close()
        raw_writer = None
        total_expected_length = math.fsum(expected_lengths)
        total_allocated_length = math.fsum(allocated_lengths)
        if not math.isclose(
            total_expected_length,
            total_allocated_length,
            rel_tol=length_relative_tolerance,
            abs_tol=length_absolute_tolerance_um,
        ):
            raise RegionProfileValidationError(
                "Global cable-length conservation failed."
            )
        if total_nodes != source_row_count:
            raise RegionProfileValidationError(
                "Global node-count conservation failed.",
                issues=[
                    f"source rows = {source_row_count}, profile nodes = {total_nodes}"
                ],
            )

        summary = RegionProfileBuildSummary(
            source_row_count=source_row_count,
            neuron_count=len(counts),
            profile_row_count=profile_rows,
            batch_count=len(batches),
            total_edge_count=edge_count,
            zero_length_edge_count=zero_length_edges,
            total_cable_length_um=total_expected_length,
            allocated_cable_length_um=total_allocated_length,
            node_count=total_nodes,
            terminus_count=total_termini,
            soma_laterality_counts=dict(sorted(soma_lateralities.items())),
            elapsed_seconds=perf_counter() - started,
        )
        metadata = RegionProfileMetadata(
            profile_format_version=PROFILE_FORMAT_VERSION,
            builder_algorithm_version=BUILDER_ALGORITHM_VERSION,
            source=fingerprint,
            atlas=atlas_identity,
            length_method=LENGTH_METHOD,
            length_absolute_tolerance_um=length_absolute_tolerance_um,
            length_relative_tolerance=length_relative_tolerance,
            compartment_definition=(
                "soma means type == 1; neurite means every non-soma node, "
                "including undefined type 0; cable uses the child compartment"
            ),
            terminus_definition=(
                "childless non-soma node computed from the complete file_id graph "
                "before region, side, or type allocation"
            ),
            build_timestamp_utc=datetime.now(UTC).isoformat(),
            build_summary=summary,
        )

        _check_cancelled(cancel_check)
        _emit_progress(
            progress_callback,
            "Sorting and compacting the regional profile...",
            len(batches) + staging_steps + 1,
            total_steps,
        )
        sorted_count = _write_sorted_profile(
            raw_path,
            final_path,
            metadata,
            row_group_size=row_group_size,
            cancel_check=cancel_check,
        )
        if sorted_count != profile_rows:
            raise RegionProfileValidationError(
                "Final profile compaction changed the row count."
            )
        _fsync_file(final_path)
        inspection = inspect_region_profile(
            final_path,
            source_path=source,
            atlas=profile_atlas,
            validate_contents=True,
        )
        if not inspection.compatible:
            raise RegionProfileValidationError(
                "The completed regional-profile sidecar failed validation.",
                issues=inspection.issues,
            )
        _check_cancelled(cancel_check)
        os.replace(final_path, destination)
        published = True
        _fsync_directory(destination.parent)
        _emit_progress(
            progress_callback,
            f"Regional profile ready: {destination.name}",
            total_steps,
            total_steps,
        )
        return metadata
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if connection is not None:
            connection.close()
        for temporary in (raw_path, final_path):
            if temporary.exists() and (temporary != final_path or not published):
                try:
                    temporary.unlink()
                except OSError:
                    pass
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def register_region_profile(
    connection: duckdb.DuckDBPyConnection,
    sidecar_path: str | Path,
    *,
    source_path: str | Path,
    atlas: RegionalProfileAtlas | object,
    table_name: str = "region_profile",
) -> RegionProfileInspection:
    """Validate and materialize a profile in a connection-local temp table."""
    if _SAFE_SQL_IDENTIFIER.fullmatch(table_name) is None:
        raise ValueError("table_name must be a simple SQL identifier")
    inspection = inspect_region_profile(
        sidecar_path,
        source_path=source_path,
        atlas=atlas,
        validate_contents=False,
    )
    if not inspection.compatible:
        raise RegionProfileValidationError(
            "Regional-profile sidecar is missing, invalid, or incompatible.",
            issues=inspection.issues,
        )
    path_sql = _escape_path(sidecar_path)
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table_name} AS
        SELECT
            file_id,
            region_id,
            CAST(laterality AS VARCHAR) AS laterality,
            CAST(compartment AS VARCHAR) AS compartment,
            cable_length_um,
            node_count,
            terminus_count
        FROM read_parquet('{path_sql}')
        """
    )
    return inspection


__all__ = [
    "BUILDER_ALGORITHM_VERSION",
    "DEFAULT_MAX_BATCH_NEURONS",
    "DEFAULT_MAX_BATCH_ROWS",
    "DEFAULT_ROW_GROUP_SIZE",
    "LENGTH_METHOD",
    "PROFILE_FORMAT_VERSION",
    "REGION_PROFILE_METADATA_KEY",
    "REGION_PROFILE_SCHEMA",
    "AtlasIdentity",
    "Compartment",
    "HierarchyLink",
    "Laterality",
    "RegionHierarchyClosure",
    "RegionProfileBuildCancelled",
    "RegionProfileBuildSummary",
    "RegionProfileError",
    "RegionProfileInspection",
    "RegionProfileMetadata",
    "RegionProfileValidationError",
    "RegionalProfileAtlas",
    "SourceFastToken",
    "SourceFingerprint",
    "build_hierarchy_closure",
    "build_region_profile",
    "default_region_profile_path",
    "inspect_region_profile",
    "register_region_profile",
    "resolve_physical_midline",
    "source_fast_token",
    "source_fingerprint",
]

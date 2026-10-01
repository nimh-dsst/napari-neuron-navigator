"""Typed Boolean queries over validated regional-profile sidecars.

The public query model in this module is immutable and independent of Qt.
Numeric atlas IDs are authoritative; acronyms and names are retained only as
stable display metadata.  Queries compile to parameterized DuckDB set
operations over a registered regional profile and never inspect the source
node relation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

import duckdb
import pandas as pd
import pyarrow as pa

from .region_profile import (
    AtlasIdentity,
    Compartment,
    RegionalProfileAtlas,
    RegionHierarchyClosure,
    RegionProfileInspection,
    RegionProfileMetadata,
    build_hierarchy_closure,
    register_region_profile,
)

QUERY_FORMAT_VERSION = 1
_SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STANDARD_RESULT_COLUMNS = ("file_id", "neuron_id", "subject")


class RegionQueryError(RuntimeError):
    """Base exception for regional-query failures."""


class RegionQueryValidationError(RegionQueryError, ValueError):
    """Raised when a query, scope, or catalog is invalid."""


class UnknownRegionError(RegionQueryValidationError):
    """Raised before execution when a selection contains foreign atlas IDs."""


class QueryLaterality(str, Enum):
    """Laterality restriction for profile conditions."""

    EITHER = "either"
    IPSILATERAL = "ipsilateral"
    CONTRALATERAL = "contralateral"
    MIDLINE = "midline"
    UNKNOWN = "unknown"


class RegionalMeasurement(str, Enum):
    """Measurements stored in a version-1 regional profile."""

    CABLE_LENGTH_UM = "cable_length_um"
    NODE_COUNT = "node_count"
    TERMINUS_COUNT = "terminus_count"


class ThresholdComparison(str, Enum):
    """Supported threshold comparisons."""

    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    EQUAL = "="


class QueryScopeKind(str, Enum):
    """The universe against which a query, especially ``NOT``, is evaluated."""

    WHOLE_PARQUET = "whole_parquet"
    EXPLICIT_FILE_IDS = "explicit_file_ids"


@dataclass(frozen=True)
class RegionReference:
    """One authoritative atlas ID and its reproducible display metadata."""

    region_id: int
    acronym: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        try:
            region_id = int(self.region_id)
        except (TypeError, ValueError) as exc:
            raise RegionQueryValidationError("region_id must be an integer") from exc
        if isinstance(self.region_id, bool) or region_id <= 0:
            raise RegionQueryValidationError("region_id must be a positive integer")
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "acronym", str(self.acronym).strip())
        object.__setattr__(self, "name", str(self.name).strip())


@dataclass(frozen=True)
class RegionSelection:
    """Atlas regions selected directly or with all represented descendants."""

    regions: tuple[RegionReference, ...]
    include_descendants: bool = False

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions:
            raise RegionQueryValidationError(
                "a regional condition must select at least one region"
            )
        if any(not isinstance(region, RegionReference) for region in regions):
            raise RegionQueryValidationError(
                "regions must contain RegionReference values"
            )
        ordered = tuple(sorted(regions, key=lambda region: region.region_id))
        region_ids = [region.region_id for region in ordered]
        if len(set(region_ids)) != len(region_ids):
            raise RegionQueryValidationError(
                "a region selection cannot repeat the same numeric atlas ID"
            )
        if not isinstance(self.include_descendants, bool):
            raise RegionQueryValidationError("include_descendants must be Boolean")
        object.__setattr__(self, "regions", ordered)

    @classmethod
    def from_ids(
        cls,
        region_ids: Sequence[int],
        *,
        include_descendants: bool = False,
        structures: Mapping[int, Mapping[str, object]] | None = None,
    ) -> RegionSelection:
        """Create a selection, optionally filling display data from an atlas."""
        references: list[RegionReference] = []
        for raw_region_id in region_ids:
            region_id = int(raw_region_id)
            structure = None if structures is None else structures.get(region_id)
            references.append(
                RegionReference(
                    region_id=raw_region_id,
                    acronym=(
                        "" if structure is None else str(structure.get("acronym", ""))
                    ),
                    name=(
                        "" if structure is None else str(structure.get("name", ""))
                    ),
                )
            )
        return cls(tuple(references), include_descendants=include_descendants)

    @property
    def region_ids(self) -> tuple[int, ...]:
        return tuple(region.region_id for region in self.regions)


class RegionQuery:
    """Marker base class for typed regional-query nodes."""


@dataclass(frozen=True)
class AllOf(RegionQuery):
    """Match file IDs present in every child result."""

    children: tuple[RegionQuery, ...]

    def __post_init__(self) -> None:
        children = tuple(self.children)
        _validate_children(children, "AllOf")
        object.__setattr__(self, "children", children)


@dataclass(frozen=True)
class AnyOf(RegionQuery):
    """Match file IDs present in one or more child results."""

    children: tuple[RegionQuery, ...]

    def __post_init__(self) -> None:
        children = tuple(self.children)
        _validate_children(children, "AnyOf")
        object.__setattr__(self, "children", children)


@dataclass(frozen=True)
class Not(RegionQuery):
    """Match the active query scope except for the child result."""

    child: RegionQuery

    def __post_init__(self) -> None:
        if not isinstance(self.child, RegionQuery):
            raise RegionQueryValidationError("Not.child must be a regional query")


@dataclass(frozen=True)
class SomaIn(RegionQuery):
    """Match neurons whose soma node lies in a selected regional set."""

    region_selection: RegionSelection

    def __post_init__(self) -> None:
        if not isinstance(self.region_selection, RegionSelection):
            raise RegionQueryValidationError(
                "SomaIn.region_selection must be a RegionSelection"
            )


@dataclass(frozen=True)
class NeuriteIntersects(RegionQuery):
    """Match neurons with non-soma morphology in a selected regional set."""

    region_selection: RegionSelection
    laterality: QueryLaterality = QueryLaterality.EITHER

    def __post_init__(self) -> None:
        if not isinstance(self.region_selection, RegionSelection):
            raise RegionQueryValidationError(
                "NeuriteIntersects.region_selection must be a RegionSelection"
            )
        object.__setattr__(self, "laterality", _coerce_laterality(self.laterality))


@dataclass(frozen=True)
class RegionalThreshold(RegionQuery):
    """Compare an aggregated regional-profile measurement with a threshold."""

    compartment: Compartment
    measurement: RegionalMeasurement
    comparison: ThresholdComparison
    value: float
    region_selection: RegionSelection
    laterality: QueryLaterality = QueryLaterality.EITHER

    def __post_init__(self) -> None:
        if not isinstance(self.region_selection, RegionSelection):
            raise RegionQueryValidationError(
                "RegionalThreshold.region_selection must be a RegionSelection"
            )
        try:
            compartment = Compartment(self.compartment)
            measurement = RegionalMeasurement(self.measurement)
            comparison = ThresholdComparison(self.comparison)
            laterality = _coerce_laterality(self.laterality)
            value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise RegionQueryValidationError(
                "regional threshold contains an unsupported typed value"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise RegionQueryValidationError(
                "regional threshold value must be finite and non-negative"
            )
        object.__setattr__(self, "compartment", compartment)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(self, "laterality", laterality)
        object.__setattr__(self, "value", value)


RegionalQueryNode: TypeAlias = (
    AllOf | AnyOf | Not | SomaIn | NeuriteIntersects | RegionalThreshold
)


def _validate_children(children: tuple[RegionQuery, ...], label: str) -> None:
    if not children:
        raise RegionQueryValidationError(f"{label} must contain at least one child")
    if any(not isinstance(child, RegionQuery) for child in children):
        raise RegionQueryValidationError(
            f"{label}.children must contain only regional queries"
        )


def _coerce_laterality(value: object) -> QueryLaterality:
    try:
        return QueryLaterality(value)
    except (TypeError, ValueError) as exc:
        raise RegionQueryValidationError(
            f"unsupported regional-query laterality: {value!r}"
        ) from exc


@dataclass(frozen=True)
class QueryScope:
    """Whole-profile or explicit-``file_id`` query universe."""

    kind: QueryScopeKind = QueryScopeKind.WHOLE_PARQUET
    file_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            kind = QueryScopeKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise RegionQueryValidationError("unsupported query scope") from exc
        raw_file_ids = tuple(self.file_ids)
        if any(value is None for value in raw_file_ids):
            raise RegionQueryValidationError("scope file_ids cannot contain null")
        file_ids = tuple(sorted({str(value) for value in raw_file_ids}))
        if any(not value for value in file_ids):
            raise RegionQueryValidationError("scope file_ids cannot be empty strings")
        if kind is QueryScopeKind.WHOLE_PARQUET and file_ids:
            raise RegionQueryValidationError(
                "whole-Parquet scope cannot contain explicit file_ids"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "file_ids", file_ids)

    @classmethod
    def whole_parquet(cls) -> QueryScope:
        return cls(QueryScopeKind.WHOLE_PARQUET)

    @classmethod
    def explicit(cls, file_ids: Sequence[object]) -> QueryScope:
        values = tuple(file_ids)
        if any(value is None for value in values):
            raise RegionQueryValidationError("scope file_ids cannot contain null")
        return cls(
            QueryScopeKind.EXPLICIT_FILE_IDS,
            tuple(str(file_id) for file_id in values),
        )

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "file_ids": list(self.file_ids)}


def _region_selection_to_dict(selection: RegionSelection) -> dict[str, object]:
    return {
        "include_descendants": selection.include_descendants,
        "regions": [
            {
                "region_id": region.region_id,
                "acronym": region.acronym,
                "name": region.name,
            }
            for region in selection.regions
        ],
    }


def query_to_dict(query: RegionQuery) -> dict[str, object]:
    """Return the typed query node as a JSON-compatible canonical mapping."""
    if isinstance(query, AllOf):
        return {
            "type": "all_of",
            "children": [query_to_dict(child) for child in query.children],
        }
    if isinstance(query, AnyOf):
        return {
            "type": "any_of",
            "children": [query_to_dict(child) for child in query.children],
        }
    if isinstance(query, Not):
        return {"type": "not", "child": query_to_dict(query.child)}
    if isinstance(query, SomaIn):
        return {
            "type": "soma_in",
            "region_selection": _region_selection_to_dict(query.region_selection),
        }
    if isinstance(query, NeuriteIntersects):
        return {
            "type": "neurite_intersects",
            "laterality": query.laterality.value,
            "region_selection": _region_selection_to_dict(query.region_selection),
        }
    if isinstance(query, RegionalThreshold):
        return {
            "type": "regional_threshold",
            "compartment": query.compartment.value,
            "measurement": query.measurement.value,
            "comparison": query.comparison.value,
            "value": query.value,
            "laterality": query.laterality.value,
            "region_selection": _region_selection_to_dict(query.region_selection),
        }
    raise RegionQueryValidationError(
        f"unsupported regional-query node: {type(query).__name__}"
    )


def serialize_query(query: RegionQuery) -> str:
    """Serialize a query to stable, versioned canonical JSON."""
    payload = {"format_version": QUERY_FORMAT_VERSION, "query": query_to_dict(query)}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _selection_from_dict(raw: object) -> RegionSelection:
    if not isinstance(raw, Mapping):
        raise RegionQueryValidationError("region_selection must be an object")
    raw_regions = raw.get("regions")
    if not isinstance(raw_regions, list):
        raise RegionQueryValidationError("region_selection.regions must be a list")
    references: list[RegionReference] = []
    for item in raw_regions:
        if not isinstance(item, Mapping) or "region_id" not in item:
            raise RegionQueryValidationError(
                "each serialized region must contain a numeric region_id"
            )
        references.append(
            RegionReference(
                region_id=item["region_id"],
                acronym=item.get("acronym", ""),
                name=item.get("name", ""),
            )
        )
    include_descendants = raw.get("include_descendants", False)
    if not isinstance(include_descendants, bool):
        raise RegionQueryValidationError("include_descendants must be Boolean")
    return RegionSelection(
        tuple(references), include_descendants=include_descendants
    )


def query_from_dict(raw: object) -> RegionQuery:
    """Reconstruct and validate one typed query node from a mapping."""
    if not isinstance(raw, Mapping):
        raise RegionQueryValidationError("serialized query node must be an object")
    node_type = raw.get("type")
    if node_type in {"all_of", "any_of"}:
        raw_children = raw.get("children")
        if not isinstance(raw_children, list):
            raise RegionQueryValidationError(f"{node_type}.children must be a list")
        children = tuple(query_from_dict(child) for child in raw_children)
        return AllOf(children) if node_type == "all_of" else AnyOf(children)
    if node_type == "not":
        if "child" not in raw:
            raise RegionQueryValidationError("not.child is required")
        return Not(query_from_dict(raw["child"]))
    if node_type == "soma_in":
        return SomaIn(_selection_from_dict(raw.get("region_selection")))
    if node_type == "neurite_intersects":
        return NeuriteIntersects(
            _selection_from_dict(raw.get("region_selection")),
            laterality=raw.get("laterality", QueryLaterality.EITHER.value),
        )
    if node_type == "regional_threshold":
        required = ("compartment", "measurement", "comparison", "value")
        missing = [name for name in required if name not in raw]
        if missing:
            raise RegionQueryValidationError(
                "regional_threshold is missing: " + ", ".join(missing)
            )
        return RegionalThreshold(
            compartment=raw["compartment"],
            measurement=raw["measurement"],
            comparison=raw["comparison"],
            value=raw["value"],
            region_selection=_selection_from_dict(raw.get("region_selection")),
            laterality=raw.get("laterality", QueryLaterality.EITHER.value),
        )
    raise RegionQueryValidationError(f"unsupported serialized query type: {node_type!r}")


def deserialize_query(payload: str | bytes | Mapping[str, object]) -> RegionQuery:
    """Deserialize versioned canonical JSON into an immutable typed query."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            raw: object = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RegionQueryValidationError("query JSON is malformed") from exc
    else:
        raw = payload
    if not isinstance(raw, Mapping):
        raise RegionQueryValidationError("serialized query must be an object")
    if raw.get("format_version") != QUERY_FORMAT_VERSION:
        raise RegionQueryValidationError("query format version is unsupported")
    if "query" not in raw:
        raise RegionQueryValidationError("serialized query is missing its query node")
    return query_from_dict(raw["query"])


@dataclass(frozen=True)
class ResolvedCondition:
    """One leaf's hierarchy resolution, intentionally excluding executable SQL."""

    path: str
    condition_type: str
    selected_region_ids: tuple[int, ...]
    include_descendants: bool
    resolved_direct_region_ids: tuple[int, ...]
    compartment: str
    laterality: str
    measurement: str | None = None
    comparison: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class RegionQueryExplanation:
    """Safe debug representation of a resolved typed query."""

    canonical_query: str
    conditions: tuple[ResolvedCondition, ...]


@dataclass(frozen=True)
class RegionQueryExecutionMetadata:
    """Provenance and timing for one profile query execution."""

    canonical_query: str
    scope: QueryScope
    matched_count: int
    returned_catalog_count: int
    missing_catalog_file_ids: tuple[str, ...]
    sidecar_identity: str
    atlas_identity: AtlasIdentity
    runtime_seconds: float


@dataclass(frozen=True)
class RegionQueryResult:
    """Standard neuron catalog rows and their regional-query provenance."""

    rows: pd.DataFrame
    metadata: RegionQueryExecutionMetadata


@dataclass(frozen=True)
class _CompiledQuery:
    sql: str
    params: tuple[object, ...]


def _validate_identifier(value: str, label: str) -> str:
    identifier = str(value)
    if _SAFE_SQL_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(f"{label} must be a simple SQL identifier")
    return identifier


def _prepare_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("catalog must be a pandas DataFrame")
    missing = [name for name in _STANDARD_RESULT_COLUMNS if name not in catalog]
    if missing:
        raise RegionQueryValidationError(
            "neuron catalog is missing columns: " + ", ".join(missing)
        )
    result = catalog.loc[:, list(_STANDARD_RESULT_COLUMNS)].copy()
    if result["file_id"].isna().any():
        raise RegionQueryValidationError("neuron catalog file_id values cannot be null")
    result["file_id"] = result["file_id"].astype(str)
    if (result["file_id"] == "").any():
        raise RegionQueryValidationError("neuron catalog file_id values cannot be empty")
    if result["file_id"].duplicated().any():
        duplicates = sorted(
            result.loc[result["file_id"].duplicated(False), "file_id"].unique()
        )
        raise RegionQueryValidationError(
            "neuron catalog must contain one row per file_id; duplicates: "
            + ", ".join(duplicates[:5])
        )
    for column in ("neuron_id", "subject"):
        result[column] = result[column].fillna("").astype(str)
    return result.sort_values("file_id").reset_index(drop=True)


def _empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_STANDARD_RESULT_COLUMNS))


def _sidecar_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class RegionQueryEngine:
    """Evaluate typed regional queries against one validated profile sidecar."""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        sidecar_path: str | Path,
        source_path: str | Path,
        atlas: RegionalProfileAtlas | object,
        catalog: pd.DataFrame,
        profile_table_name: str = "region_profile",
        catalog_table_name: str = "region_query_catalog",
    ) -> None:
        self._connection = connection
        self._profile_table = _validate_identifier(
            profile_table_name, "profile_table_name"
        )
        self._catalog_table = _validate_identifier(
            catalog_table_name, "catalog_table_name"
        )
        if self._profile_table == self._catalog_table:
            raise ValueError("profile and catalog table names must differ")
        prepared_catalog = _prepare_catalog(catalog)
        profile_atlas = (
            atlas
            if isinstance(atlas, RegionalProfileAtlas)
            else RegionalProfileAtlas.from_atlas(atlas)
        )
        inspection = register_region_profile(
            connection,
            sidecar_path,
            source_path=source_path,
            atlas=profile_atlas,
            table_name=self._profile_table,
        )
        if inspection.metadata is None:  # Defensive: registration guarantees this.
            raise RegionQueryValidationError(
                "validated regional profile has no metadata"
            )
        input_name = _validate_identifier(
            f"{self._catalog_table}_input", "catalog input name"
        )
        catalog_table = pa.Table.from_pandas(
            prepared_catalog,
            preserve_index=False,
        )
        connection.register(input_name, catalog_table)
        try:
            connection.execute(f"""
                CREATE OR REPLACE TEMP TABLE {self._catalog_table} AS
                SELECT
                    CAST(file_id AS VARCHAR) AS file_id,
                    CAST(neuron_id AS VARCHAR) AS neuron_id,
                    CAST(subject AS VARCHAR) AS subject
                FROM {input_name}
            """)
        finally:
            connection.unregister(input_name)

        self._inspection = inspection
        self._metadata = inspection.metadata
        self._atlas = profile_atlas
        self._closure = build_hierarchy_closure(profile_atlas.structures)
        self._known_region_ids = frozenset(int(value) for value in profile_atlas.structures)
        self._resolved_selection_cache: dict[RegionSelection, tuple[int, ...]] = {}
        self._sidecar_identity = _sidecar_sha256(sidecar_path)

    @property
    def inspection(self) -> RegionProfileInspection:
        return self._inspection

    @property
    def profile_metadata(self) -> RegionProfileMetadata:
        return self._metadata

    @property
    def hierarchy_closure(self) -> RegionHierarchyClosure:
        return self._closure

    @property
    def sidecar_identity(self) -> str:
        """Return the SHA-256 content digest of the registered sidecar."""
        return self._sidecar_identity

    def _resolve_selection(self, selection: RegionSelection) -> tuple[int, ...]:
        cached = self._resolved_selection_cache.get(selection)
        if cached is not None:
            return cached
        unknown = sorted(set(selection.region_ids) - self._known_region_ids)
        if unknown:
            raise UnknownRegionError(
                "query contains region IDs absent from the compatible atlas: "
                + ", ".join(str(region_id) for region_id in unknown)
            )
        represented: set[int] = set()
        for region_id in selection.region_ids:
            if selection.include_descendants:
                represented.update(self._closure.represented_direct_ids(region_id))
            else:
                represented.add(region_id)
        resolved = tuple(sorted(represented))
        if not resolved:
            raise UnknownRegionError(
                "query region selection resolves to no direct atlas regions"
            )
        self._resolved_selection_cache[selection] = resolved
        return resolved

    def explain(self, query: RegionQuery) -> RegionQueryExplanation:
        """Resolve hierarchy semantics without returning or executing raw SQL."""
        if not isinstance(query, RegionQuery):
            raise RegionQueryValidationError("query must be a typed RegionQuery")
        conditions: list[ResolvedCondition] = []

        def visit(node: RegionQuery, path: str) -> None:
            if isinstance(node, (AllOf, AnyOf)):
                for index, child in enumerate(node.children):
                    visit(child, f"{path}.children[{index}]")
                return
            if isinstance(node, Not):
                visit(node.child, f"{path}.child")
                return
            if isinstance(node, SomaIn):
                selection = node.region_selection
                conditions.append(
                    ResolvedCondition(
                        path=path,
                        condition_type="soma_in",
                        selected_region_ids=selection.region_ids,
                        include_descendants=selection.include_descendants,
                        resolved_direct_region_ids=self._resolve_selection(selection),
                        compartment=Compartment.SOMA.value,
                        laterality=QueryLaterality.EITHER.value,
                    )
                )
                return
            if isinstance(node, NeuriteIntersects):
                selection = node.region_selection
                conditions.append(
                    ResolvedCondition(
                        path=path,
                        condition_type="neurite_intersects",
                        selected_region_ids=selection.region_ids,
                        include_descendants=selection.include_descendants,
                        resolved_direct_region_ids=self._resolve_selection(selection),
                        compartment=Compartment.NEURITE.value,
                        laterality=node.laterality.value,
                    )
                )
                return
            if isinstance(node, RegionalThreshold):
                selection = node.region_selection
                conditions.append(
                    ResolvedCondition(
                        path=path,
                        condition_type="regional_threshold",
                        selected_region_ids=selection.region_ids,
                        include_descendants=selection.include_descendants,
                        resolved_direct_region_ids=self._resolve_selection(selection),
                        compartment=node.compartment.value,
                        laterality=node.laterality.value,
                        measurement=node.measurement.value,
                        comparison=node.comparison.value,
                        value=node.value,
                    )
                )
                return
            raise RegionQueryValidationError(
                f"unsupported regional-query node: {type(node).__name__}"
            )

        visit(query, "query")
        return RegionQueryExplanation(
            canonical_query=serialize_query(query),
            conditions=tuple(conditions),
        )

    def _region_placeholders(self, region_ids: tuple[int, ...]) -> str:
        return ", ".join("?" for _ in region_ids)

    def _laterality_sql(
        self,
        laterality: QueryLaterality,
        *,
        column: str = "laterality",
    ) -> tuple[str, tuple[object, ...]]:
        if laterality is QueryLaterality.EITHER:
            return "", ()
        return f" AND {column} = ?", (laterality.value,)

    def _compile(self, query: RegionQuery) -> _CompiledQuery:
        if isinstance(query, (AllOf, AnyOf)):
            compiled = [self._compile(child) for child in query.children]
            operator = " INTERSECT " if isinstance(query, AllOf) else " UNION "
            return _CompiledQuery(
                sql="(" + operator.join(item.sql for item in compiled) + ")",
                params=tuple(
                    parameter for item in compiled for parameter in item.params
                ),
            )
        if isinstance(query, Not):
            child = self._compile(query.child)
            return _CompiledQuery(
                sql=(
                    "(SELECT file_id FROM query_scope EXCEPT " + child.sql + ")"
                ),
                params=child.params,
            )

        if isinstance(query, SomaIn):
            region_ids = self._resolve_selection(query.region_selection)
            placeholders = self._region_placeholders(region_ids)
            return _CompiledQuery(
                sql=(
                    f"(SELECT DISTINCT file_id FROM {self._profile_table} "
                    "WHERE compartment = ? AND node_count > 0 "
                    f"AND region_id IN ({placeholders}))"
                ),
                params=(Compartment.SOMA.value, *region_ids),
            )

        if isinstance(query, NeuriteIntersects):
            region_ids = self._resolve_selection(query.region_selection)
            placeholders = self._region_placeholders(region_ids)
            laterality_sql, laterality_params = self._laterality_sql(
                query.laterality
            )
            return _CompiledQuery(
                sql=(
                    f"(SELECT DISTINCT file_id FROM {self._profile_table} "
                    "WHERE compartment = ? "
                    "AND (node_count > 0 OR cable_length_um > 0.0) "
                    f"AND region_id IN ({placeholders}){laterality_sql})"
                ),
                params=(
                    Compartment.NEURITE.value,
                    *region_ids,
                    *laterality_params,
                ),
            )

        if isinstance(query, RegionalThreshold):
            region_ids = self._resolve_selection(query.region_selection)
            placeholders = self._region_placeholders(region_ids)
            laterality_sql, laterality_params = self._laterality_sql(
                query.laterality,
                column="profile.laterality",
            )
            measurement = query.measurement.value
            comparison = query.comparison.value
            return _CompiledQuery(
                sql=(
                    "(SELECT scope.file_id FROM query_scope AS scope "
                    f"LEFT JOIN {self._profile_table} AS profile "
                    "ON profile.file_id = scope.file_id "
                    "AND profile.compartment = ? "
                    f"AND profile.region_id IN ({placeholders})"
                    f"{laterality_sql} "
                    "GROUP BY scope.file_id "
                    f"HAVING COALESCE(SUM(profile.{measurement}), 0) "
                    f"{comparison} ?)"
                ),
                params=(
                    query.compartment.value,
                    *region_ids,
                    *laterality_params,
                    query.value,
                ),
            )
        raise RegionQueryValidationError(
            f"unsupported regional-query node: {type(query).__name__}"
        )

    def _scope_sql(self, scope: QueryScope) -> tuple[str, tuple[object, ...]]:
        if scope.kind is QueryScopeKind.WHOLE_PARQUET:
            return (
                f"SELECT DISTINCT file_id FROM {self._profile_table}",
                (),
            )
        values = ", ".join("(?)" for _ in scope.file_ids)
        requested_scope_sql = (
            "SELECT DISTINCT CAST(file_id AS VARCHAR) AS file_id "
            f"FROM (VALUES {values}) AS requested(file_id) "
            "INTERSECT "
            f"SELECT DISTINCT file_id FROM {self._profile_table}"
        )
        return (
            requested_scope_sql,
            tuple(scope.file_ids),
        )

    def execute(
        self,
        query: RegionQuery,
        *,
        scope: QueryScope | None = None,
    ) -> RegionQueryResult:
        """Execute a query and return unique standard neuron catalog rows."""
        resolved_scope = QueryScope.whole_parquet() if scope is None else scope
        if not isinstance(resolved_scope, QueryScope):
            raise RegionQueryValidationError("scope must be a QueryScope")
        started = perf_counter()
        explanation = self.explain(query)
        if (
            resolved_scope.kind is QueryScopeKind.EXPLICIT_FILE_IDS
            and not resolved_scope.file_ids
        ):
            metadata = RegionQueryExecutionMetadata(
                canonical_query=explanation.canonical_query,
                scope=resolved_scope,
                matched_count=0,
                returned_catalog_count=0,
                missing_catalog_file_ids=(),
                sidecar_identity=self._sidecar_identity,
                atlas_identity=self._metadata.atlas,
                runtime_seconds=perf_counter() - started,
            )
            return RegionQueryResult(_empty_result_frame(), metadata)

        scope_sql, scope_params = self._scope_sql(resolved_scope)
        compiled = self._compile(query)
        sql = f"""
            WITH query_scope AS (
                {scope_sql}
            ),
            matched_ids AS (
                (SELECT file_id FROM query_scope)
                INTERSECT
                {compiled.sql}
            )
            SELECT
                matched.file_id,
                catalog.file_id IS NOT NULL AS has_catalog,
                catalog.neuron_id,
                catalog.subject
            FROM matched_ids AS matched
            LEFT JOIN {self._catalog_table} AS catalog
                ON catalog.file_id = matched.file_id
            ORDER BY matched.file_id
        """
        frame = self._connection.execute(
            sql, [*scope_params, *compiled.params]
        ).fetchdf()
        missing_file_ids = tuple(
            frame.loc[~frame["has_catalog"], "file_id"].astype(str).tolist()
        )
        rows = (
            frame.loc[frame["has_catalog"], list(_STANDARD_RESULT_COLUMNS)]
            .reset_index(drop=True)
        )
        metadata = RegionQueryExecutionMetadata(
            canonical_query=explanation.canonical_query,
            scope=resolved_scope,
            matched_count=len(frame),
            returned_catalog_count=len(rows),
            missing_catalog_file_ids=missing_file_ids,
            sidecar_identity=self._sidecar_identity,
            atlas_identity=self._metadata.atlas,
            runtime_seconds=perf_counter() - started,
        )
        return RegionQueryResult(rows=rows, metadata=metadata)


__all__ = [
    "QUERY_FORMAT_VERSION",
    "AllOf",
    "AnyOf",
    "NeuriteIntersects",
    "Not",
    "QueryLaterality",
    "QueryScope",
    "QueryScopeKind",
    "RegionQuery",
    "RegionQueryEngine",
    "RegionQueryError",
    "RegionQueryExecutionMetadata",
    "RegionQueryExplanation",
    "RegionQueryResult",
    "RegionQueryValidationError",
    "RegionReference",
    "RegionSelection",
    "RegionalMeasurement",
    "RegionalQueryNode",
    "RegionalThreshold",
    "ResolvedCondition",
    "SomaIn",
    "ThresholdComparison",
    "UnknownRegionError",
    "deserialize_query",
    "query_from_dict",
    "query_to_dict",
    "serialize_query",
]

"""Cohort regional measurements from a validated regional-profile relation.

Distinct-neuron measurements are calculated from file IDs at every hierarchy
level. Summing children's neuron counts would count the same neuron repeatedly.
Exclusions remove direct-region contributions, never neurons from the cohort.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import duckdb
import pyarrow as pa

from .region_profile import build_hierarchy_closure

COMBINED_LATERALITY = "combined"
REPORT_LATERALITIES = ("ipsilateral", "contralateral", "midline", "unknown")
REPORT_COLUMNS = (
    "Region",
    "Neurons present",
    "Cohort present (%)",
    "Total cable (mm)",
    "Mean cable per cohort neuron (mm)",
    "Neurons with termini (%)",
    "Total termini",
)


@dataclass(frozen=True)
class RegionReportMeasurements:
    laterality: str
    neurons_present: int
    cohort_present_percent: float
    total_cable_mm: float
    mean_cable_per_cohort_neuron_mm: float
    neurons_with_termini_percent: float
    total_termini: int

    def values(self) -> tuple[int | float, ...]:
        """Return the numeric cells in report column order."""
        return (
            self.neurons_present,
            self.cohort_present_percent,
            self.total_cable_mm,
            self.mean_cable_per_cohort_neuron_mm,
            self.neurons_with_termini_percent,
            self.total_termini,
        )


@dataclass(frozen=True)
class RegionReportMembers:
    """Positive neurite contributors to one retained direct-region bucket."""

    laterality: str
    file_ids: frozenset[str]


@dataclass(frozen=True)
class RegionReportSelection:
    """One report row, including its anatomical and laterality scope."""

    region_id: int
    laterality: str = COMBINED_LATERALITY
    include_descendants: bool = True


@dataclass(frozen=True)
class RegionReportRowMatch:
    selection: RegionReportSelection
    label: str
    file_ids: frozenset[str]


@dataclass(frozen=True)
class RegionReportNode:
    region_id: int
    parent_id: int | None
    name: str
    acronym: str
    direct: tuple[RegionReportMeasurements, ...]
    aggregate: tuple[RegionReportMeasurements, ...]
    members: tuple[RegionReportMembers, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.acronym})" if self.acronym else self.name


@dataclass(frozen=True)
class RegionReport:
    cohort_file_ids: tuple[str, ...]
    nodes: tuple[RegionReportNode, ...]
    # Before exclusions, so excluded branches remain available to uncheck.
    available_region_ids: frozenset[int]
    split_laterality: bool

    @property
    def cohort_count(self) -> int:
        return len(self.cohort_file_ids)

    def match_rows(
        self, selections: Sequence[RegionReportSelection]
    ) -> tuple[RegionReportRowMatch, ...]:
        """Resolve selected rows against this report's immutable cohort snapshot.

        Ancestor rows union retained descendants. Explicit [direct] rows use
        their own bucket. Exclusions, soma filtering, and side restrictions are
        already reflected in the stored contributors; no node scan is needed.
        """
        nodes = {node.region_id: node for node in self.nodes}
        children: dict[int, list[int]] = defaultdict(list)
        for node in self.nodes:
            if node.parent_id is not None:
                children[node.parent_id].append(node.region_id)
        result = []
        for selection in dict.fromkeys(selections):
            node = nodes.get(selection.region_id)
            if node is None:
                raise ValueError("A selected report region is no longer available.")
            if not any(
                value.laterality == selection.laterality for value in node.aggregate
            ):
                raise ValueError("A selected report laterality is no longer available.")
            pending = [selection.region_id]
            matching: set[str] = set()
            while pending:
                region_id = pending.pop()
                for members in nodes[region_id].members:
                    if members.laterality == selection.laterality:
                        matching.update(members.file_ids)
                if selection.include_descendants:
                    pending.extend(children.get(region_id, ()))
            label = node.acronym or node.name
            if not selection.include_descendants:
                label += " [direct]"
            if selection.laterality != COMBINED_LATERALITY:
                label += f" ({selection.laterality})"
            result.append(RegionReportRowMatch(selection, label, frozenset(matching)))
        return tuple(sorted(result, key=lambda match: match.label))


def aggregate_region_report(
    connection: duckdb.DuckDBPyConnection,
    *,
    structures: Mapping[int, Mapping[str, Any]],
    file_ids: Sequence[object],
    split_laterality: bool = False,
    excluded_region_ids: Sequence[int] = (),
) -> RegionReport:
    """Summarize a connection-local, validated ``region_profile`` table.

    The caller registers the sidecar with ``register_region_profile`` first.
    No source node rows, geometry, or compartment labels are reinterpreted here.
    Ancestors are retained only when they contain a positive neurite contribution.
    Region zero is an independent outside/unmapped bucket, outside atlas roots.
    """
    if any(value is None or not str(value).strip() for value in file_ids):
        raise ValueError("The report cohort contains an empty file_id.")
    cohort = tuple(sorted({str(value) for value in file_ids}))
    if not cohort:
        return RegionReport((), (), frozenset(), split_laterality)

    closure = build_hierarchy_closure(structures)
    links = closure.to_arrow().to_pylist()
    links.append({"ancestor_region_id": 0, "direct_region_id": 0, "depth": 0})
    excluded: set[int] = set()
    for region_id in excluded_region_ids:
        region_id = int(region_id)
        if region_id == 0:
            excluded.add(0)
        elif region_id not in structures:
            raise ValueError(f"Excluded region {region_id} is absent from the atlas.")
        else:
            excluded.update(closure.represented_direct_ids(region_id))

    connection.register("_report_cohort", pa.table({"file_id": cohort}))
    connection.register(
        "_report_closure",
        pa.Table.from_pylist(links, schema=closure.to_arrow().schema),
    )
    connection.register(
        "_report_excluded",
        pa.table({"region_id": pa.array(sorted(excluded), pa.int32())}),
    )
    # Share cohort strings across membership sets rather than retaining a new
    # Python string for every direct-region occurrence of a neuron.
    cohort_ids = {file_id: file_id for file_id in cohort}
    members_by_region: dict[int, list[RegionReportMembers]] = defaultdict(list)
    try:
        missing = connection.execute("""
            SELECT file_id FROM _report_cohort
            EXCEPT SELECT file_id FROM region_profile
            ORDER BY file_id
        """).fetchall()
        if missing:
            raise ValueError(
                f"{len(missing)} cohort file_id(s) are missing from the regional "
                "profile. Reload the Data table or build a compatible profile."
            )
        laterality_sql = "p.laterality" if split_laterality else "'combined'"
        connection.execute(f"""
            CREATE OR REPLACE TEMP TABLE _report_neurites AS
            SELECT p.file_id, p.region_id, {laterality_sql} AS laterality,
                   p.node_count, p.cable_length_um, p.terminus_count
            FROM region_profile p
            JOIN _report_cohort c ON p.file_id = c.file_id
            WHERE p.compartment = 'neurite'
              AND (p.node_count > 0 OR p.cable_length_um > 0)
        """)
        available = frozenset(
            int(row[0])
            for row in connection.execute("""
                SELECT DISTINCT h.ancestor_region_id
                FROM _report_closure h
                JOIN _report_neurites p ON p.region_id = h.direct_region_id
            """).fetchall()
        )
        rows = connection.execute("""
            WITH retained AS (
                SELECT p.* FROM _report_neurites p
                WHERE NOT EXISTS (
                    SELECT 1 FROM _report_excluded e WHERE e.region_id = p.region_id
                )
            ), allocations AS (
                SELECT false AS is_aggregate, p.region_id, p.laterality,
                       p.file_id, p.node_count, p.cable_length_um, p.terminus_count
                FROM retained p
                UNION ALL
                SELECT true AS is_aggregate, h.ancestor_region_id AS region_id,
                       p.laterality, p.file_id, p.node_count,
                       p.cable_length_um, p.terminus_count
                FROM retained p
                JOIN _report_closure h ON h.direct_region_id = p.region_id
            )
            SELECT is_aggregate, region_id, laterality,
                   count(DISTINCT file_id)::BIGINT AS neurons_present,
                   sum(cable_length_um)::DOUBLE AS cable_um,
                   count(DISTINCT file_id) FILTER (
                       WHERE terminus_count > 0
                   )::BIGINT AS neurons_with_termini,
                   sum(terminus_count)::BIGINT AS termini
            FROM allocations
            GROUP BY is_aggregate, region_id, laterality
            ORDER BY region_id, laterality
        """).fetchall()
        membership_cursor = connection.execute("""
            SELECT p.region_id, p.laterality, list(DISTINCT p.file_id)
            FROM _report_neurites p
            WHERE NOT EXISTS (
                SELECT 1 FROM _report_excluded e WHERE e.region_id = p.region_id
            )
            GROUP BY p.region_id, p.laterality
        """)
        while batch := membership_cursor.fetchmany(64):
            for region_id, side, members in batch:
                members_by_region[int(region_id)].append(
                    RegionReportMembers(
                        str(side), frozenset(cohort_ids[file_id] for file_id in members)
                    )
                )
    finally:
        connection.execute("DROP TABLE IF EXISTS _report_neurites")
        for name in ("_report_cohort", "_report_closure", "_report_excluded"):
            connection.unregister(name)

    direct: dict[int, list[RegionReportMeasurements]] = defaultdict(list)
    aggregate: dict[int, list[RegionReportMeasurements]] = defaultdict(list)
    count = len(cohort)
    for is_aggregate, region_id, side, present, cable_um, with_termini, termini in rows:
        cable_mm = float(cable_um) / 1000.0
        values = RegionReportMeasurements(
            laterality=str(side),
            neurons_present=int(present),
            cohort_present_percent=100.0 * int(present) / count,
            total_cable_mm=cable_mm,
            mean_cable_per_cohort_neuron_mm=cable_mm / count,
            neurons_with_termini_percent=100.0 * int(with_termini) / count,
            total_termini=int(termini),
        )
        (aggregate if is_aggregate else direct)[int(region_id)].append(values)

    parents = {
        int(link.direct_region_id): int(link.ancestor_region_id)
        for link in closure.links
        if link.depth == 1
    }
    nodes = []
    for region_id in sorted(aggregate):
        structure = structures.get(region_id, {})
        nodes.append(
            RegionReportNode(
                region_id=region_id,
                parent_id=parents.get(region_id),
                name=str(
                    structure.get(
                        "name",
                        "Outside / unmapped"
                        if region_id == 0
                        else f"Region {region_id}",
                    )
                ),
                acronym=str(
                    structure.get("acronym", "unmapped" if region_id == 0 else "")
                ),
                direct=tuple(direct[region_id]),
                aggregate=tuple(aggregate[region_id]),
                members=tuple(
                    sorted(members_by_region[region_id], key=lambda m: m.laterality)
                ),
            )
        )
    return RegionReport(cohort, tuple(nodes), available, split_laterality)

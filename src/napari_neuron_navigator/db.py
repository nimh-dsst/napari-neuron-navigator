"""DuckDB query interface for neuron data.

This module provides a NeuronDatabase class for efficient querying of
neuron data stored in Parquet format using DuckDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd

from .atlas_utils import mask_to_swc_xyz_bounds, swc_coords_xyz_to_atlas_voxels
from .swc import NodeType, normalize_node_types
from .terminals import TERMINUS_NODE_TYPES, TerminusCoverage, query_termini

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NeuronDatabase:
    """DuckDB-based interface for querying neuron data from Parquet files.

    Parameters
    ----------
    parquet_path : Path or str
        Path to the Parquet file containing neuron data.

    Examples
    --------
    >>> db = NeuronDatabase("neurons.parquet")
    >>> # Get all neurons in a specific region
    >>> neurons = db.get_neurons_by_region(["VISp"])
    >>> # Get soma locations for all neurons
    >>> somas = db.get_soma_locations()
    """

    def __init__(self, parquet_path: Path | str):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        self.conn = duckdb.connect()
        self._setup_view()

    def _setup_view(self) -> None:
        """Create a view for the Parquet file."""
        path_str = str(self.parquet_path).replace("\\", "/")
        self.conn.execute(
            f"CREATE OR REPLACE VIEW neurons AS SELECT * FROM read_parquet('{path_str}')"
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def has_column(self, column_name: str) -> bool:
        """Return whether the loaded Parquet-backed view exposes a column."""
        rows = self.conn.execute("DESCRIBE neurons").fetchall()
        names = {str(row[0]) for row in rows}
        return str(column_name) in names

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _append_file_id_filter(
        where_parts: list[str],
        params: list[object],
        file_ids: list[object] | tuple[object, ...] | None,
    ) -> bool:
        """Append an optional ``file_id`` restriction to a WHERE clause."""
        if file_ids is None:
            return True
        if not file_ids:
            return False

        placeholders = ", ".join(["?"] * len(file_ids))
        where_parts.append(f"file_id IN ({placeholders})")
        params.extend(file_ids)
        return True

    @staticmethod
    def _append_file_id_exclusion(
        where_parts: list[str],
        params: list[object],
        exclude_file_ids: list[object] | tuple[object, ...] | None,
    ) -> None:
        """Append an optional ``file_id`` exclusion to a WHERE clause."""
        if not exclude_file_ids:
            return

        placeholders = ", ".join(["?"] * len(exclude_file_ids))
        where_parts.append(f"file_id NOT IN ({placeholders})")
        params.extend(exclude_file_ids)

    @staticmethod
    def _empty_neuron_result() -> pd.DataFrame:
        """Return an empty query result with the standard neuron columns."""
        return pd.DataFrame(columns=["file_id", "neuron_id", "subject"])

    @staticmethod
    def _resolve_node_type_filter(
        node_types: list[int] | tuple[int, ...] | None,
        soma_only: bool,
    ) -> tuple[int, ...] | None:
        """Return the effective node-type filter for compatibility callers."""
        if node_types is None and soma_only:
            return (NodeType.SOMA,)
        return normalize_node_types(node_types)

    @staticmethod
    def _append_node_type_filter(
        where_parts: list[str],
        params: list[object],
        node_types: tuple[int, ...] | None,
    ) -> bool:
        """Append an optional SWC ``type`` filter to a WHERE clause."""
        if node_types is None:
            return True
        if not node_types:
            return False

        placeholders = ", ".join(["?"] * len(node_types))
        where_parts.append(f"type IN ({placeholders})")
        params.extend(int(node_type) for node_type in node_types)
        return True

    def get_neurons_by_region(
        self,
        region_acronyms: list[str],
        include_children: bool = False,
        soma_only: bool = False,
        file_ids: list[object] | tuple[object, ...] | None = None,
        exclude_file_ids: list[object] | tuple[object, ...] | None = None,
        node_types: list[int] | tuple[int, ...] | None = None,
    ) -> pd.DataFrame:
        """Get neurons that have nodes in the specified regions.

        Parameters
        ----------
        region_acronyms : list[str]
            List of region acronyms to query (e.g., ["VISp", "VISl"]).
        include_children : bool, default=False
            If True, include child regions in the query.
        soma_only : bool, default=False
            If True, only match soma/body rows (``type = 1``).
        node_types : list[int], optional
            Explicit SWC node type IDs to match. If provided, this takes
            precedence over ``soma_only``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, neuron_id, subject
        """
        if not region_acronyms:
            return self._empty_neuron_result()

        placeholders = ", ".join(["?"] * len(region_acronyms))
        where_parts = [f"region_acronym IN ({placeholders})"]
        params: list[object] = list(region_acronyms)
        effective_node_types = self._resolve_node_type_filter(
            node_types,
            soma_only,
        )
        if not self._append_node_type_filter(
            where_parts,
            params,
            effective_node_types,
        ):
            return self._empty_neuron_result()
        if not self._append_file_id_filter(where_parts, params, file_ids):
            return self._empty_neuron_result()
        self._append_file_id_exclusion(where_parts, params, exclude_file_ids)
        query = f"""
            SELECT DISTINCT file_id, neuron_id, subject
            FROM neurons
            WHERE {" AND ".join(where_parts)}
            ORDER BY file_id
        """
        return self.conn.execute(query, params).fetchdf()

    def get_neurons_by_region_id(
        self,
        region_ids: list[int],
        soma_only: bool = False,
        file_ids: list[object] | tuple[object, ...] | None = None,
        exclude_file_ids: list[object] | tuple[object, ...] | None = None,
        node_types: list[int] | tuple[int, ...] | None = None,
    ) -> pd.DataFrame:
        """Get neurons that have nodes in the specified region IDs.

        Parameters
        ----------
        region_ids : list[int]
            List of Allen CCF region IDs.
        soma_only : bool, default=False
            If True, only match soma/body rows (``type = 1``).
        node_types : list[int], optional
            Explicit SWC node type IDs to match. If provided, this takes
            precedence over ``soma_only``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, neuron_id, subject
        """
        if not region_ids:
            return self._empty_neuron_result()

        placeholders = ", ".join(["?"] * len(region_ids))
        where_parts = [f"region_id IN ({placeholders})"]
        params: list[object] = [int(region_id) for region_id in region_ids]
        effective_node_types = self._resolve_node_type_filter(
            node_types,
            soma_only,
        )
        if not self._append_node_type_filter(
            where_parts,
            params,
            effective_node_types,
        ):
            return self._empty_neuron_result()
        if not self._append_file_id_filter(where_parts, params, file_ids):
            return self._empty_neuron_result()
        self._append_file_id_exclusion(where_parts, params, exclude_file_ids)
        query = f"""
            SELECT DISTINCT file_id, neuron_id, subject
            FROM neurons
            WHERE {" AND ".join(where_parts)}
            ORDER BY file_id
        """
        return self.conn.execute(query, params).fetchdf()

    def get_unique_regions(self) -> pd.DataFrame:
        """Get all unique regions in the dataset with counts.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: region_id, region_acronym, region_name, node_count
        """
        query = """
            SELECT
                region_id,
                region_acronym,
                MAX(region_name) as region_name,
                COUNT(*) as node_count
            FROM neurons
            WHERE region_id > 0
            GROUP BY region_id, region_acronym
            ORDER BY node_count DESC
        """
        return self.conn.execute(query).fetchdf()

    def get_neuron_catalog(self) -> pd.DataFrame:
        """Return one deterministic display row per unique ``file_id``.

        ``file_id`` is the identity key. ``neuron_id`` and ``subject`` are
        display metadata only and are never part of the grouping key.
        """
        columns = {
            str(row[0]) for row in self.conn.execute("DESCRIBE neurons").fetchall()
        }
        if "file_id" not in columns:
            raise ValueError("The loaded Parquet does not contain a file_id column.")
        neuron_expr = (
            "COALESCE(MIN(CAST(neuron_id AS VARCHAR)), '')"
            if "neuron_id" in columns
            else "''"
        )
        subject_expr = (
            "COALESCE(MIN(CAST(subject AS VARCHAR)), '')"
            if "subject" in columns
            else "''"
        )
        return self.conn.execute(f"""
            SELECT
                CAST(file_id AS VARCHAR) AS file_id,
                {neuron_expr} AS neuron_id,
                {subject_expr} AS subject
            FROM neurons
            WHERE file_id IS NOT NULL
            GROUP BY CAST(file_id AS VARCHAR)
            ORDER BY file_id
        """).fetchdf()

    def get_soma_locations(
        self,
        file_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get soma locations for neurons.

        Parameters
        ----------
        file_ids : list[str], optional
            Filter to specific files. If None, return all somas.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, neuron_id, x, y, z, region_acronym
        """
        if file_ids:
            placeholders = ", ".join(["?"] * len(file_ids))
            query = f"""
                SELECT
                    file_id, neuron_id,
                    AVG(x) as x, AVG(y) as y, AVG(z) as z,
                    MAX(region_acronym) as region_acronym
                FROM neurons
                WHERE type = 1 AND file_id IN ({placeholders})
                GROUP BY file_id, neuron_id
                ORDER BY file_id
            """
            return self.conn.execute(query, file_ids).fetchdf()
        else:
            query = """
                SELECT
                    file_id, neuron_id,
                    AVG(x) as x, AVG(y) as y, AVG(z) as z,
                    MAX(region_acronym) as region_acronym
                FROM neurons
                WHERE type = 1
                GROUP BY file_id, neuron_id
                ORDER BY file_id
            """
            return self.conn.execute(query).fetchdf()

    def get_soma_nodes_for_rendering(
        self,
        file_ids: list[str],
    ) -> pd.DataFrame:
        """Get full soma node rows for rendering/projection.

        Like :meth:`get_neurons_for_rendering` but filtered to soma nodes
        (``type = 1``) inside DuckDB, so only soma rows are materialized
        instead of every node of every neuron. All columns are returned,
        preserving any precomputed flatmap/depth columns.

        Parameters
        ----------
        file_ids : list[str]
            List of file IDs to retrieve soma nodes for.

        Returns
        -------
        pd.DataFrame
            DataFrame with all node columns for soma nodes only.
        """
        if not file_ids:
            return pd.DataFrame()

        placeholders = ", ".join(["?"] * len(file_ids))
        query = f"""
            SELECT *
            FROM neurons
            WHERE type = 1 AND file_id IN ({placeholders})
            ORDER BY file_id, node_id
        """
        return self.conn.execute(query, file_ids).fetchdf()

    def get_soma_points(
        self,
        file_ids: list[str],
    ) -> pd.DataFrame:
        """Get raw soma/body node coordinates for rendering or projection.

        Parameters
        ----------
        file_ids : list[str]
            List of file IDs to retrieve soma nodes for.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, neuron_id, x, y, z.
        """
        if not file_ids:
            return pd.DataFrame(columns=["file_id", "neuron_id", "x", "y", "z"])

        placeholders = ", ".join(["?"] * len(file_ids))
        query = f"""
            SELECT file_id, neuron_id, x, y, z
            FROM neurons
            WHERE type = 1 AND file_id IN ({placeholders})
            ORDER BY file_id, node_id
        """
        return self.conn.execute(query, file_ids).fetchdf()

    def get_termini(
        self,
        file_ids: list[str] | None = None,
        node_types: list[int] | tuple[int, ...] | None = TERMINUS_NODE_TYPES,
    ) -> tuple[pd.DataFrame, TerminusCoverage]:
        """Get childless nodes of the selected types, with a coverage report.

        Childless nodes of an axon-typed compartment are that axon's termini.
        The node-type restriction narrows only which termini are reported; the
        child lookup always spans every node of each selected cell, so a node
        whose only child carries another type is never reported as a terminus.

        Apply any region restriction to the returned frame rather than to
        ``file_ids``: filtering nodes out of the source would make their parents
        look childless.

        Parameters
        ----------
        file_ids : list[str], optional
            Restrict to these neurons. ``None`` uses every neuron in the source.
        node_types : list[int] or tuple[int, ...], optional
            SWC types to report. Defaults to axon only. ``None`` reports every
            childless node; an empty sequence reports none.

        Returns
        -------
        tuple[pd.DataFrame, TerminusCoverage]
            Terminus nodes, and a report of which neurons were skipped because
            they contain no nodes of the selected types.
        """
        return query_termini(
            self.conn,
            "neurons",
            node_types=node_types,
            file_ids=file_ids,
        )

    def get_neurons_for_rendering(
        self,
        file_ids: list[str],
    ) -> pd.DataFrame:
        """Get full neuron data for rendering.

        Parameters
        ----------
        file_ids : list[str]
            List of file IDs to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame with all neuron node data for the specified files.
        """
        if not file_ids:
            return pd.DataFrame()

        placeholders = ", ".join(["?"] * len(file_ids))
        query = f"""
            SELECT *
            FROM neurons
            WHERE file_id IN ({placeholders})
            ORDER BY file_id, node_id
        """
        return self.conn.execute(query, file_ids).fetchdf()

    def get_neuron_lines(
        self,
        file_id: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Get line segments for a single neuron.

        Parameters
        ----------
        file_id : str
            The file ID of the neuron.

        Returns
        -------
        tuple[NDArray, NDArray]
            (vertices, edges) where vertices is (N, 3) coordinates and
            edges is (M, 2) indices into vertices.
        """
        query = """
            SELECT node_id, x, y, z, parent_id
            FROM neurons
            WHERE file_id = ?
            ORDER BY node_id
        """
        df = self.conn.execute(query, [file_id]).fetchdf()

        if df.empty:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 2)

        # Build coordinate array
        coords = df[["x", "y", "z"]].values.astype(np.float64)

        # Build node_id to index mapping
        node_ids = df["node_id"].values
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Build edges
        edges = []
        for idx, parent_id in enumerate(df["parent_id"].values):
            if parent_id in id_to_idx:
                parent_idx = id_to_idx[parent_id]
                edges.append([parent_idx, idx])

        edges = np.array(edges, dtype=np.int32) if edges else np.array([]).reshape(0, 2)

        return coords, edges

    def get_neuron_lines_batch(
        self,
        file_ids: list[str],
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.int32]]]:
        """Get line segments for multiple neurons in a single query.

        Parameters
        ----------
        file_ids : list[str]
            The file IDs of the neurons.

        Returns
        -------
        dict[str, tuple[NDArray, NDArray]]
            Mapping of file_id to (vertices, edges) where vertices is (N, 3)
            coordinates and edges is (M, 2) indices into vertices.
        """
        if not file_ids:
            return {}

        placeholders = ", ".join(["?"] * len(file_ids))
        query = f"""
            SELECT file_id, node_id, x, y, z, parent_id
            FROM neurons
            WHERE file_id IN ({placeholders})
            ORDER BY file_id, node_id
        """
        df = self.conn.execute(query, file_ids).fetchdf()

        result = {}
        for file_id, group in df.groupby("file_id"):
            coords = group[["x", "y", "z"]].values.astype(np.float64)
            node_ids = group["node_id"].values
            id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

            parent_ids = group["parent_id"].values
            edges = []
            for idx, parent_id in enumerate(parent_ids):
                if parent_id in id_to_idx:
                    edges.append([id_to_idx[parent_id], idx])

            edges_arr = (
                np.array(edges, dtype=np.int32) if edges else np.array([]).reshape(0, 2)
            )
            result[file_id] = (coords, edges_arr)

        return result

    def get_statistics(self) -> dict:
        """Get summary statistics for the database.

        Returns
        -------
        dict
            Dictionary with keys: n_nodes, n_files, n_subjects, n_regions
        """
        stats = {}

        result = self.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()
        stats["n_nodes"] = result[0]

        result = self.conn.execute(
            "SELECT COUNT(DISTINCT file_id) FROM neurons"
        ).fetchone()
        stats["n_files"] = result[0]

        result = self.conn.execute(
            "SELECT COUNT(DISTINCT subject) FROM neurons"
        ).fetchone()
        stats["n_subjects"] = result[0]

        result = self.conn.execute(
            "SELECT COUNT(DISTINCT region_id) FROM neurons WHERE region_id > 0"
        ).fetchone()
        stats["n_regions"] = result[0]

        return stats

    def get_unique_node_types(self) -> tuple[int, ...]:
        """Return sorted SWC node-type codes represented in the dataset."""
        rows = self.conn.execute(
            "SELECT DISTINCT type FROM neurons WHERE type IS NOT NULL ORDER BY type"
        ).fetchall()
        values: list[int] = []
        for row in rows:
            try:
                values.append(int(row[0]))
            except (TypeError, ValueError):
                continue
        return tuple(values)

    def get_dendrite_label_coverage(
        self,
        file_ids: list[str] | tuple[str, ...] | None = None,
    ):
        """Return whole-neuron basal/apical label coverage keyed by ``file_id``."""
        from .analysis.voxel_filter import (
            DendriteLabelCoverage,
            VoxelNodeFilter,
            prepare_voxel_node_filter,
        )

        settings = VoxelNodeFilter(require_dendrite_labels=True)
        prepared = prepare_voxel_node_filter(
            self.conn,
            "neurons",
            settings,
            file_ids=file_ids,
        )
        return DendriteLabelCoverage(
            input_file_ids=prepared.input_file_ids,
            labeled_file_ids=prepared.dendrite_labeled_file_ids or (),
            dendrite_node_types=settings.dendrite_node_types,
        )

    def get_neurons_by_mask(
        self,
        mask_volume: NDArray[np.bool_] | NDArray[np.uint8] | np.ndarray,
        atlas: Any,
        soma_only: bool = False,
        file_ids: list[object] | tuple[object, ...] | None = None,
        exclude_file_ids: list[object] | tuple[object, ...] | None = None,
        node_types: list[int] | tuple[int, ...] | None = None,
    ) -> pd.DataFrame:
        """Get neurons whose nodes fall inside a binary atlas-space mask."""
        mask = np.asarray(mask_volume) > 0
        if mask.ndim != 3:
            raise ValueError(f"Expected a 3D mask volume, got shape {mask.shape}")

        atlas_shape = tuple(np.asarray(atlas.annotation).shape)
        if mask.shape != atlas_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match atlas shape {atlas_shape}"
            )

        bounds = mask_to_swc_xyz_bounds(mask, atlas)
        if bounds is None:
            return self._empty_neuron_result()

        lower_xyz, upper_xyz = bounds
        where_parts = [
            "x >= ?",
            "x <= ?",
            "y >= ?",
            "y <= ?",
            "z >= ?",
            "z <= ?",
        ]
        params: list[object] = [
            float(lower_xyz[0]),
            float(upper_xyz[0]),
            float(lower_xyz[1]),
            float(upper_xyz[1]),
            float(lower_xyz[2]),
            float(upper_xyz[2]),
        ]
        effective_node_types = self._resolve_node_type_filter(
            node_types,
            soma_only,
        )
        if not self._append_node_type_filter(
            where_parts,
            params,
            effective_node_types,
        ):
            return self._empty_neuron_result()
        if not self._append_file_id_filter(where_parts, params, file_ids):
            return self._empty_neuron_result()
        self._append_file_id_exclusion(where_parts, params, exclude_file_ids)

        query = f"""
            SELECT file_id, neuron_id, subject, x, y, z
            FROM neurons
            WHERE {" AND ".join(where_parts)}
            ORDER BY file_id
        """
        candidates = self.conn.execute(query, params).fetchdf()
        if candidates.empty:
            return self._empty_neuron_result()

        coords = candidates[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
        voxel_coords = swc_coords_xyz_to_atlas_voxels(coords, atlas)
        in_bounds = np.all(
            (voxel_coords >= 0) & (voxel_coords < np.asarray(mask.shape)),
            axis=1,
        )
        hits = np.zeros(len(candidates), dtype=bool)
        valid = voxel_coords[in_bounds]
        if len(valid) > 0:
            hits[in_bounds] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]

        matched = candidates.loc[hits, ["file_id", "neuron_id", "subject"]]
        if matched.empty:
            return self._empty_neuron_result()
        return matched.drop_duplicates().sort_values("file_id").reset_index(drop=True)

    def get_region_neuron_counts(self) -> pd.DataFrame:
        """Get neuron counts per region.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: region_acronym, region_name, neuron_count
        """
        query = """
            SELECT
                region_acronym,
                MAX(region_name) as region_name,
                COUNT(DISTINCT file_id) as neuron_count
            FROM neurons
            WHERE region_acronym != ''
            GROUP BY region_acronym
            ORDER BY neuron_count DESC
        """
        return self.conn.execute(query).fetchdf()

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Execute a custom SQL query.

        Parameters
        ----------
        sql : str
            SQL query string. Use 'neurons' as the table name.
        params : list, optional
            Query parameters for placeholders.

        Returns
        -------
        pd.DataFrame
            Query results as a DataFrame.

        Examples
        --------
        >>> db.query("SELECT * FROM neurons WHERE type = 2 LIMIT 10")
        >>> db.query("SELECT * FROM neurons WHERE file_id = ?", ["file.swc"])
        """
        if params:
            return self.conn.execute(sql, params).fetchdf()
        return self.conn.execute(sql).fetchdf()

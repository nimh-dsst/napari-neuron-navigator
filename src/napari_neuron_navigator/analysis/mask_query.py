"""Bounded Parquet scans for arbitrary masks on the native atlas grid.

Reads directly through Arrow: no eager SQL result, node sorting, spill files,
or spatial index is needed. Only matched neuron identities survive a batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..swc import normalize_node_types

MASK_QUERY_BATCH_SIZE = 65_536
_IDENTITY_COLUMNS = ("file_id", "neuron_id", "subject")


class MaskQueryCancelled(RuntimeError):
    """A cancelled search must not publish a partial cohort."""


def _check_cancelled(cancel_check):
    if cancel_check is not None and cancel_check():
        raise MaskQueryCancelled(
            "Mask search cancelled; the Data table was not changed."
        )


def prepare_query_mask(volumes, shape, *, cancel_check=None, progress_callback=None):
    """Snapshot and union masks in slabs, then find bounds with axis reductions."""
    if not volumes:
        raise ValueError("Select at least one mask.")
    if len(shape) != 3:
        raise ValueError("Expected a 3D atlas mask.")
    arrays = [np.asarray(volume) for volume in volumes]
    if any(array.shape != tuple(shape) for array in arrays):
        raise ValueError("Every mask must match the loaded atlas's 3D grid shape.")
    _check_cancelled(cancel_check)
    mask = np.empty(shape, dtype=bool)
    for start in range(0, shape[0], 8):
        _check_cancelled(cancel_check)
        slab = slice(start, start + 8)
        np.greater(arrays[0][slab], 0, out=mask[slab])
        for array in arrays[1:]:
            np.logical_or(mask[slab], array[slab] > 0, out=mask[slab])
        if progress_callback:
            progress_callback("Preparing mask", min(start + 8, shape[0]), shape[0])
    occupied = []
    for axis in range(3):
        _check_cancelled(cancel_check)
        occupied.append(
            np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        )
    if any(len(indices) == 0 for indices in occupied):
        return mask, None
    return mask, (
        np.array([indices[0] for indices in occupied]),
        np.array([indices[-1] + 1 for indices in occupied]),
    )


def _stat_bounds(column):
    stats = column.statistics
    if stats is not None and stats.has_min_max:
        try:
            return stats.min, stats.max
        except (pa.ArrowNotImplementedError, TypeError, ValueError):
            pass
    return None


def _skip_group(group, indices, lower, upper, types, matched, allowed, excluded):
    """Conservative row-group pruning; missing statistics never exclude rows."""
    for axis, name in enumerate(("x", "y", "z")):
        limits = _stat_bounds(group.column(indices[name]))
        if limits is not None and (limits[1] < lower[axis] or limits[0] >= upper[axis]):
            return True
    if types is not None:
        limits = _stat_bounds(group.column(indices["type"]))
        if limits is not None and not any(
            limits[0] <= value <= limits[1] for value in types
        ):
            return True
    limits = _stat_bounds(group.column(indices["file_id"]))
    if limits is not None and limits[0] == limits[1]:
        file_id = str(limits[0])
        if file_id in matched or file_id in excluded:
            return True
        if allowed is not None and file_id not in allowed:
            return True
    return False


def query_neurons_in_masks(
    parquet_path: str | Path,
    volumes: Sequence[np.ndarray],
    *,
    atlas_shape: tuple[int, int, int],
    resolution_um: Sequence[float],
    file_ids=None,
    exclude_file_ids=None,
    node_types=None,
    batch_size: int = MASK_QUERY_BATCH_SIZE,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """Return unique matching file_ids using bounded, cancellable Arrow batches."""
    started = perf_counter()
    resolution = np.asarray(resolution_um, dtype=float)
    if (
        resolution.shape != (3,)
        or not np.isfinite(resolution).all()
        or (resolution <= 0).any()
    ):
        raise ValueError("Atlas resolution must contain three positive finite values.")
    if batch_size <= 0:
        raise ValueError("Mask-query batch size must be positive.")
    types = normalize_node_types(node_types)
    allowed = None if file_ids is None else set(map(str, file_ids))
    excluded = set(map(str, exclude_file_ids or ()))
    result = pd.DataFrame(columns=list(_IDENTITY_COLUMNS))
    _check_cancelled(cancel_check)
    if allowed == set() or types == ():
        return result
    mask, bounds = prepare_query_mask(
        volumes,
        atlas_shape,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    if bounds is None:
        result.attrs["empty_mask"] = True
        return result
    lower, upper = (bound * resolution for bound in bounds)
    matched = {}
    completed = scanned = skipped = 0
    with pq.ParquetFile(
        parquet_path, read_dictionary=list(_IDENTITY_COLUMNS), pre_buffer=False
    ) as source:
        names = source.schema_arrow.names
        columns = [*_IDENTITY_COLUMNS, "x", "y", "z"]
        if types is not None:
            columns.append("type")
        missing = set(columns) - set(names)
        if missing:
            raise ValueError(
                f"Mask queries require columns: {', '.join(sorted(missing))}"
            )
        indices = {name: source.schema.names.index(name) for name in columns}
        total = source.metadata.num_rows
        if progress_callback:
            progress_callback("Searching nodes", 0, total)
        for group_index in range(source.metadata.num_row_groups):
            _check_cancelled(cancel_check)
            group = source.metadata.row_group(group_index)
            if _skip_group(
                group, indices, lower, upper, types, matched, allowed, excluded
            ):
                completed += group.num_rows
                skipped += group.num_rows
                if progress_callback:
                    progress_callback(
                        f"Searching nodes; {len(matched):,} neurons found",
                        completed,
                        total,
                    )
                continue
            for batch in source.iter_batches(
                batch_size=batch_size,
                row_groups=[group_index],
                columns=columns,
                use_threads=False,
            ):
                _check_cancelled(cancel_check)
                identities = batch.column("file_id")
                if not pa.types.is_dictionary(identities.type):
                    identities = identities.dictionary_encode()
                codes = pc.fill_null(identities.indices, -1).to_numpy(
                    zero_copy_only=False
                )
                dictionary = identities.dictionary.to_pylist()
                eligible = np.array(
                    [
                        value is not None
                        and str(value) not in matched
                        and str(value) not in excluded
                        and (allowed is None or str(value) in allowed)
                        for value in dictionary
                    ]
                    + [False]
                )
                keep = eligible[codes]
                if types is not None:
                    keep &= np.isin(
                        batch.column("type").to_numpy(zero_copy_only=False), types
                    )
                if keep.any():
                    xyz = [
                        batch.column(name)
                        .cast(pa.float64())
                        .to_numpy(zero_copy_only=False)
                        for name in ("x", "y", "z")
                    ]
                    for axis, values in enumerate(xyz):
                        keep &= (
                            np.isfinite(values)
                            & (values >= lower[axis])
                            & (values < upper[axis])
                        )
                    rows = np.flatnonzero(keep)
                    if len(rows):
                        voxels = [
                            np.floor(values[rows] / resolution[axis]).astype(np.int64)
                            for axis, values in enumerate(xyz)
                        ]
                        rows = rows[mask[voxels[0], voxels[1], voxels[2]]]
                        if len(rows):
                            # Decode one metadata row per hit file, never per hit node.
                            _, first = np.unique(codes[rows], return_index=True)
                            selected = pa.array(rows[first])
                            values = [
                                batch.column(name).take(selected).to_pylist()
                                for name in _IDENTITY_COLUMNS
                            ]
                            for record in zip(*values, strict=True):
                                matched.setdefault(str(record[0]), record)
                completed += batch.num_rows
                scanned += batch.num_rows
                if progress_callback:
                    progress_callback(
                        f"Searching nodes; {len(matched):,} neurons found",
                        completed,
                        total,
                    )
        _check_cancelled(cancel_check)
    if matched:
        result = (
            pd.DataFrame(matched.values(), columns=list(_IDENTITY_COLUMNS))
            .sort_values("file_id")
            .reset_index(drop=True)
        )
    result.attrs["mask_query"] = {
        "total_rows": completed,
        "scanned_rows": scanned,
        "skipped_rows": skipped,
        "elapsed_seconds": perf_counter() - started,
        "index_bytes": 0,
    }
    return result

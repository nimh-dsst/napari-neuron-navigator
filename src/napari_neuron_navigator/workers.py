"""Background workers for long-running analysis tasks.

Each worker runs in a QThread and emits progress/finished/error signals
so the napari UI stays responsive. Workers create their own DuckDB
connections since DuckDB connections are not thread-safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QObject, Signal

from .analysis.flatmap_correlation import (
    DEFAULT_FLATMAP_DEPTH_SCALE,
    DEFAULT_FLATMAP_SOMA_DBSCAN_EPS,
)
from .logging_utils import startup_timing

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from .analysis.clustering import ClusterRegionSelection, ClusterResult
    from .isocortex_layers import AllenIsocortexLayerMap

logger = logging.getLogger(__name__)


def cached_brainglobe_atlas_dir(
    atlas_name: str,
    *,
    brainglobe_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> Path | None:
    """Return the single local cache directory for an atlas, if available."""
    if brainglobe_dir is None:
        try:
            from brainglobe_atlasapi import config

            conf = config.read_config(config_dir)
            brainglobe_dir = conf["default_dirs"]["brainglobe_dir"]
        except Exception:
            logger.debug(
                "Failed to resolve BrainGlobe cache directory.",
                exc_info=True,
            )
            return None

    root = Path(brainglobe_dir).expanduser()
    candidates = sorted(path for path in root.glob(f"{atlas_name}_v*") if path.is_dir())
    if len(candidates) != 1:
        return None
    return candidates[0]


def resolve_brainglobe_dirs(
    *,
    brainglobe_dir: str | Path | None = None,
    interm_download_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return BrainGlobe atlas and intermediate download directories."""
    if brainglobe_dir is None or interm_download_dir is None:
        from brainglobe_atlasapi import config

        conf = config.read_config(config_dir)
        default_dirs = conf["default_dirs"]
        if brainglobe_dir is None:
            brainglobe_dir = default_dirs["brainglobe_dir"]
        if interm_download_dir is None:
            interm_download_dir = default_dirs.get(
                "interm_download_dir", brainglobe_dir
            )

    return Path(brainglobe_dir).expanduser(), Path(interm_download_dir).expanduser()


def load_brainglobe_atlas(
    atlas_name: str,
    *,
    brainglobe_dir: str | Path,
    interm_download_dir: str | Path,
    config_dir: str | Path | None = None,
    fn_update=None,
):
    """Load a BrainGlobe atlas with explicit dirs and no latest-version check."""
    from brainglobe_atlasapi import BrainGlobeAtlas

    return BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=brainglobe_dir,
        interm_download_dir=interm_download_dir,
        check_latest=False,
        config_dir=config_dir,
        fn_update=fn_update,
    )


def load_cached_brainglobe_atlas(atlas_name: str, atlas_dir: str | Path):
    """Load an already-cached BrainGlobe atlas without remote checks/downloads."""
    from brainglobe_atlasapi.core import Atlas

    atlas = Atlas(Path(atlas_dir))
    atlas.atlas_name = atlas_name
    return atlas


class CachedAtlasLoadWorker(QObject):
    """Load a locally cached BrainGlobe atlas in the background."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, atlas_name: str, atlas_dir: str | Path):
        super().__init__()
        self._atlas_name = str(atlas_name)
        self._atlas_dir = Path(atlas_dir)

    def run(self) -> None:
        """Load the cached atlas and emit it."""
        try:
            with startup_timing(
                logger,
                "cached_atlas_load_worker",
                atlas=self._atlas_name,
                atlas_dir=self._atlas_dir,
            ):
                atlas = load_cached_brainglobe_atlas(
                    self._atlas_name,
                    self._atlas_dir,
                )
            self.finished.emit(atlas)
        except Exception as e:
            logger.exception("Cached atlas load failed")
            self.error.emit(str(e))


class AtlasLoadWorker(QObject):
    """Load a BrainGlobe atlas in the background, downloading it if missing."""

    status = Signal(str)
    progress = Signal(int, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        atlas_name: str,
        *,
        brainglobe_dir: str | Path | None = None,
        interm_download_dir: str | Path | None = None,
        config_dir: str | Path | None = None,
    ):
        super().__init__()
        self._atlas_name = str(atlas_name)
        self._brainglobe_dir = brainglobe_dir
        self._interm_download_dir = interm_download_dir
        self._config_dir = config_dir
        self._installing_status_emitted = False

    def _emit_busy_progress(self) -> None:
        """Emit an indeterminate progress state."""
        self.progress.emit(0, 0, 0)

    def _emit_download_progress(self, completed: int, total: int) -> None:
        """Forward BrainGlobe byte progress as a 0-100 progress value."""
        if total <= 0:
            self._emit_busy_progress()
            return

        percentage = int(min(max(completed / total, 0.0), 1.0) * 100)
        self.progress.emit(0, 100, percentage)
        if completed >= total and not self._installing_status_emitted:
            self._installing_status_emitted = True
            self.status.emit(
                f"Atlas: Download complete. Installing {self._atlas_name} "
                f"into {self._resolved_brainglobe_dir}..."
            )
            self._emit_busy_progress()

    def run(self) -> None:
        """Load the atlas and emit progress/status updates."""
        try:
            with startup_timing(
                logger,
                "atlas_load_worker",
                atlas=self._atlas_name,
            ) as timing:
                self.status.emit(
                    f"Atlas: Checking BrainGlobe cache for {self._atlas_name}..."
                )
                self._emit_busy_progress()
                brainglobe_dir, interm_download_dir = resolve_brainglobe_dirs(
                    brainglobe_dir=self._brainglobe_dir,
                    interm_download_dir=self._interm_download_dir,
                    config_dir=self._config_dir,
                )
                self._resolved_brainglobe_dir = brainglobe_dir

                candidates = sorted(
                    path
                    for path in brainglobe_dir.glob(f"{self._atlas_name}_v*")
                    if path.is_dir()
                )
                timing.set(
                    cached_candidates=len(candidates),
                    brainglobe_dir=brainglobe_dir,
                    interm_download_dir=interm_download_dir,
                )

                if len(candidates) == 1:
                    self.status.emit(
                        f"Atlas: Found cached {self._atlas_name} in "
                        f"{candidates[0]}. Loading..."
                    )
                elif not candidates:
                    self.status.emit(
                        f"Atlas: {self._atlas_name} was not found in the "
                        "local BrainGlobe cache. Downloading via BrainGlobe "
                        f"to {brainglobe_dir}..."
                    )
                else:
                    self.status.emit(
                        f"Atlas: Found multiple cached versions of "
                        f"{self._atlas_name} in {brainglobe_dir}. "
                        "BrainGlobe will choose or report an error..."
                    )

                atlas = load_brainglobe_atlas(
                    self._atlas_name,
                    brainglobe_dir=brainglobe_dir,
                    interm_download_dir=interm_download_dir,
                    config_dir=self._config_dir,
                    fn_update=self._emit_download_progress,
                )
                structure_count = len(getattr(atlas, "structures", {}))
                timing.set(structures=structure_count)
                self.progress.emit(0, 100, 100)
                self.status.emit(
                    f"Atlas: Loaded {self._atlas_name} ({structure_count} structures)."
                )
            self.finished.emit(atlas)
        except Exception as e:
            logger.exception("Atlas load failed")
            self.error.emit(str(e))


def _attach_cluster_run_metadata(
    result,
    *,
    atlas,
    parquet_path: str,
    region_selection,
    analysis_method: str,
    clustering_algorithm: str,
    distance_metric: str,
    clustering_linkage: str | None,
    dendrogram_linkage: str | None,
    dilation_fraction: float,
    requested_cluster_count: int | None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int | None = None,
    extra_metadata: dict[str, object] | None = None,
):
    """Populate the cluster result with reproducibility metadata."""
    from .analysis.clustering import ClusterRegionSelection, ClusterRunMetadata

    if region_selection is None:
        region_selection = ClusterRegionSelection()

    result.metadata = ClusterRunMetadata.from_region_selection(
        region_selection=region_selection,
        analysis_method=analysis_method,
        clustering_algorithm=clustering_algorithm,
        distance_metric=distance_metric,
        clustering_linkage=clustering_linkage,
        dendrogram_linkage=dendrogram_linkage,
        dilation_fraction=dilation_fraction,
        requested_cluster_count=requested_cluster_count,
        actual_cluster_count=len(np.unique(result.labels)),
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        atlas_name=getattr(atlas, "atlas_name", None),
        atlas_resolution_um=tuple(
            float(value) for value in getattr(atlas, "resolution", ()) or ()
        ),
        source_parquet_path=str(Path(parquet_path)),
        dendrogram_leaf_order=[int(value) for value in result.reorder_indices.tolist()],
        extra_metadata=extra_metadata,
    )
    return result


@dataclass(frozen=True)
class ClusteringPreflightResult:
    """Exact clustering input count plus reusable CCF region preparation."""

    node_count: int
    voxel_id_map: np.ndarray | None = None


class ClusteringPreflightWorker(QObject):
    """Count the exact node rows that will contribute to a clustering run."""

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        *,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        coordinate_space: str,
        clustering_method: str,
        region_selection: ClusterRegionSelection | None = None,
        dilation_fraction: float = 0.0,
        file_ids: list[str] | None = None,
        flatmap_style: str | None = None,
        flatmap_y_bins: int = 0,
        flatmap_depth_bin_um: float = 0.0,
        flatmap_include_depth_minus_one: bool = True,
        flatmap_collapse_depth: bool = False,
    ) -> None:
        super().__init__()
        self._parquet_path = str(parquet_path)
        self._atlas = atlas
        self._coordinate_space = str(coordinate_space)
        self._clustering_method = str(clustering_method)
        self._region_selection = region_selection
        self._dilation_fraction = float(dilation_fraction)
        self._file_ids = (
            None if file_ids is None else [str(file_id) for file_id in file_ids]
        )
        self._flatmap_style = flatmap_style
        self._flatmap_y_bins = int(flatmap_y_bins)
        self._flatmap_depth_bin_um = float(flatmap_depth_bin_um)
        self._flatmap_include_depth_minus_one = bool(flatmap_include_depth_minus_one)
        # Collapsing does not change the node count, but it shrinks the voxel
        # grid the size guard checks, so the preflight must agree with the run.
        self._flatmap_collapse_depth = bool(flatmap_collapse_depth)

    def run(self) -> None:
        """Prepare an optional region map and count method-specific node rows."""
        try:
            voxel_id_map = None
            resolution = float(self._atlas.resolution[0])
            if self._coordinate_space == "ccf" and self._region_selection is not None:
                from .analysis.mask import get_expanded_region_voxel_ids_for_regions

                self.progress.emit("Preparing target-region mask...", 1, 2)
                voxel_id_map = get_expanded_region_voxel_ids_for_regions(
                    self._atlas,
                    self._region_selection.selected_region_acronyms,
                    self._dilation_fraction,
                )
            else:
                self.progress.emit("Preparing clustering input...", 1, 2)

            self.progress.emit("Counting clustering nodes...", 2, 2)
            if self._coordinate_space == "ccf":
                if self._clustering_method == "voxel":
                    import duckdb

                    from .analysis.correlation import count_correlation_input_nodes

                    conn = duckdb.connect()
                    try:
                        node_count = count_correlation_input_nodes(
                            conn,
                            self._parquet_path,
                            voxel_id_map,
                            resolution,
                            file_ids=self._file_ids,
                        )
                    finally:
                        conn.close()
                else:
                    from .analysis.clustering import query_ccf_soma_coordinates

                    _ids, _coords, node_count = query_ccf_soma_coordinates(
                        self._parquet_path,
                        resolution=resolution,
                        file_ids=self._file_ids,
                        voxel_id_map=voxel_id_map,
                    )
            elif self._clustering_method == "voxel":
                from .analysis.flatmap_correlation import (
                    count_flatmap_voxel_correlation_nodes,
                )

                if self._flatmap_style is None:
                    raise ValueError("No flatmap style is available for clustering.")
                node_count = count_flatmap_voxel_correlation_nodes(
                    self._parquet_path,
                    style=self._flatmap_style,
                    y_bins=self._flatmap_y_bins,
                    depth_bin_um=self._flatmap_depth_bin_um,
                    include_depth_minus_one=(self._flatmap_include_depth_minus_one),
                    file_ids=self._file_ids,
                    collapse_depth=self._flatmap_collapse_depth,
                )
            else:
                from .analysis.flatmap_correlation import (
                    query_flatmap_soma_coordinates_and_count,
                )

                if self._flatmap_style is None:
                    raise ValueError("No flatmap style is available for clustering.")
                _ids, _coords, node_count = query_flatmap_soma_coordinates_and_count(
                    self._parquet_path,
                    style=self._flatmap_style,
                    file_ids=self._file_ids,
                )

            self.finished.emit(
                ClusteringPreflightResult(
                    node_count=int(node_count),
                    voxel_id_map=voxel_id_map,
                )
            )
        except Exception as e:
            logger.exception("Clustering preflight failed")
            self.error.emit(str(e))


class ConvertWorker(QObject):
    """Convert SWC files to annotated Parquet in the background.

    Signals
    -------
    progress(str, int, int)
        (message, files_processed, total_files)
    finished(str, object)
        (output_path, BatchParquetConversionSummary)
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(
        self,
        swc_paths: str | Path | list[str | Path],
        output_path: str,
        resolution: int = 25,
        hemisphere: str | None = None,
        atlas_name: str = "allen_mouse_25um",
        coord_axis: int = 2,
        recursive: bool = False,
        n_workers: int = 1,
        source_mode: str | None = None,
        cached_atlas: BrainGlobeAtlas | None = None,
        use_cached_annotation: bool = False,
        flatmap_lookup_dir: str | Path | None = None,
        flatmap_lookup_resolution_um: float | None = None,
    ):
        super().__init__()
        self._swc_source = (
            Path(swc_paths)
            if isinstance(swc_paths, (str, Path))
            else [Path(p) for p in swc_paths]
        )
        self._output_path = Path(output_path)
        self._resolution = resolution
        self._hemisphere = hemisphere
        self._atlas_name = atlas_name
        self._coord_axis = coord_axis
        self._recursive = recursive
        self._n_workers = n_workers
        self._source_mode = source_mode or (
            "files"
            if isinstance(self._swc_source, list)
            else "directory"
            if recursive
            else "path"
        )
        self._cached_atlas = cached_atlas
        self._use_cached_annotation = use_cached_annotation
        self._flatmap_lookup_dir = (
            Path(flatmap_lookup_dir) if flatmap_lookup_dir is not None else None
        )
        self._flatmap_lookup_resolution_um = flatmap_lookup_resolution_um
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cooperative cancellation at the next pipeline checkpoint."""
        self._cancel_requested = True

    def _check_cancelled(self) -> None:
        if self._cancel_requested:
            raise RuntimeError("SWC conversion cancelled.")

    def _report_progress(self, message: str, current: int, total: int) -> None:
        self._check_cancelled()
        self.progress.emit(str(message), int(current), int(total))

    def _source_log_value(self) -> str:
        """Return a compact source description for conversion timing logs."""
        if isinstance(self._swc_source, list):
            count = len(self._swc_source)
            if count <= 50:
                return str([str(path) for path in self._swc_source])
            sample = self._swc_source[:10] + self._swc_source[-10:]
            return f"{count} files sample={[str(path) for path in sample]}"
        return str(self._swc_source)

    def _cached_atlas_conversion_inputs(self):
        """Return optional cached atlas-derived conversion inputs."""
        if self._cached_atlas is None:
            return None, None, None

        from .hemisphere import get_atlas_midline
        from .region import BrainGlobeStructureTree, build_region_lookup

        cached_start = perf_counter()
        cached_midline = None
        if self._hemisphere is not None:
            cached_midline = get_atlas_midline(
                self._cached_atlas,
                self._coord_axis,
            )

        cached_annotation_volume = None
        cached_region_lookup = None
        if self._use_cached_annotation:
            cached_annotation_volume = self._cached_atlas.annotation
            cached_region_lookup = build_region_lookup(
                BrainGlobeStructureTree(self._cached_atlas)
            )

        logger.debug(
            (
                "swc_conversion_worker_cached_atlas_ready source_mode=%s "
                "atlas=%s midline=%s annotation=%s region_lookup=%s "
                "elapsed_s=%.6f"
            ),
            self._source_mode,
            getattr(self._cached_atlas, "atlas_name", None),
            cached_midline,
            cached_annotation_volume is not None,
            len(cached_region_lookup) if cached_region_lookup is not None else 0,
            perf_counter() - cached_start,
        )
        return cached_midline, cached_annotation_volume, cached_region_lookup

    def run(self) -> None:
        """Execute the conversion pipeline."""
        run_start = perf_counter()
        staged_conversion_path: Path | None = None
        try:
            from .parquet import batch_convert_swc_to_parquet

            total = len(self._swc_source) if isinstance(self._swc_source, list) else 0
            logger.debug(
                (
                    "swc_conversion_worker_start source_mode=%s source=%s "
                    "file_count=%s recursive=%s output=%s resolution=%s "
                    "hemisphere=%s atlas=%s coord_axis=%d n_workers=%d "
                    "cached_atlas=%s cached_annotation=%s"
                ),
                self._source_mode,
                self._source_log_value(),
                total if isinstance(self._swc_source, list) else "unknown",
                self._recursive,
                self._output_path,
                self._resolution,
                self._hemisphere,
                self._atlas_name,
                self._coord_axis,
                self._n_workers,
                self._cached_atlas is not None,
                self._use_cached_annotation,
            )
            initial_message = (
                "Searching for SWC files..."
                if not isinstance(self._swc_source, list)
                else "Preparing SWC-to-Parquet conversion..."
            )
            self._report_progress(
                initial_message,
                0,
                total,
            )
            conversion_output_path = self._output_path
            if self._flatmap_lookup_dir is not None:
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, staged_name = tempfile.mkstemp(
                    prefix=f".{self._output_path.stem}.swc-",
                    suffix=".parquet",
                    dir=self._output_path.parent,
                )
                os.close(descriptor)
                staged_conversion_path = Path(staged_name)
                staged_conversion_path.unlink()
                conversion_output_path = staged_conversion_path
            batch_start = perf_counter()
            cached_midline, cached_annotation_volume, cached_region_lookup = (
                self._cached_atlas_conversion_inputs()
            )
            summary = batch_convert_swc_to_parquet(
                self._swc_source,
                conversion_output_path,
                recursive=self._recursive,
                hemisphere=self._hemisphere,
                atlas_name=self._atlas_name,
                coord_axis=self._coord_axis,
                midline=cached_midline,
                annotate_regions=True,
                resolution=self._resolution,
                n_workers=self._n_workers,
                annotation_volume=cached_annotation_volume,
                region_lookup=cached_region_lookup,
                source_mode=self._source_mode,
                progress_callback=self._report_progress,
            )
            batch_elapsed_s = perf_counter() - batch_start
            logger.debug(
                (
                    "swc_conversion_worker_batch_ok source_mode=%s "
                    "elapsed_s=%.6f discovered=%d processed=%d failed=%d "
                    "rows=%d flipped=%d"
                ),
                self._source_mode,
                batch_elapsed_s,
                summary.discovered_files,
                summary.processed_files,
                summary.failed_files,
                summary.rows_written,
                summary.flipped_files,
            )
            for failed_path, failure_message in summary.failures:
                logger.debug(
                    (
                        "swc_conversion_worker_file_failure source_mode=%s "
                        "file=%s error=%s"
                    ),
                    self._source_mode,
                    failed_path,
                    failure_message,
                )
            if summary.discovered_files == 0:
                raise ValueError("No SWC files found")
            if summary.processed_files == 0:
                raise ValueError("No SWC files were successfully processed")

            total = summary.discovered_files
            if self._flatmap_lookup_dir is not None:
                self._check_cancelled()
                from .flatmap_parquet import augment_neuron_parquet_with_flatmaps
                from .flatmap_profiles import discover_flatmap_lookup_set

                lookup_cache_dir = self._output_path.parent / ".flatmap-lookup-arrays"
                self._report_progress(
                    "Adding bilateral flatmap/depth columns...",
                    total,
                    total,
                )
                lookup_set = discover_flatmap_lookup_set(
                    self._flatmap_lookup_dir,
                    lookup_resolution_um=self._flatmap_lookup_resolution_um,
                    npy_cache_dir=lookup_cache_dir,
                    progress_callback=self._report_progress,
                    cancel_callback=lambda: self._cancel_requested,
                )
                augment_neuron_parquet_with_flatmaps(
                    conversion_output_path,
                    self._output_path,
                    lookup_set,
                    npy_cache_dir=lookup_cache_dir,
                    progress_callback=self._report_progress,
                    cancel_callback=lambda: self._cancel_requested,
                )
            # Returning from the final writer is the atomic publication
            # boundary. A later cancel request must not turn success into an
            # error after the destination already exists.
            self.progress.emit("Finalizing Parquet...", total, total)
            logger.debug(
                "swc_conversion_worker_finished source_mode=%s elapsed_s=%.6f",
                self._source_mode,
                perf_counter() - run_start,
            )
            self.finished.emit(str(self._output_path), summary)

        except Exception as e:
            logger.exception(
                "SWC-to-Parquet conversion failed source_mode=%s elapsed_s=%.6f",
                self._source_mode,
                perf_counter() - run_start,
            )
            self.error.emit(str(e))
        finally:
            if staged_conversion_path is not None:
                try:
                    staged_conversion_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Failed to remove staged SWC conversion %s",
                        staged_conversion_path,
                        exc_info=True,
                    )


class FlatmapParquetPreparationWorker(QObject):
    """Prepare every row of a neuron Parquet with v3 bilateral coordinates."""

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        source_path: str | Path,
        output_path: str | Path,
        lookup_dir: str | Path,
        *,
        lookup_resolution_um: float | None = None,
    ) -> None:
        super().__init__()
        self._source_path = Path(source_path)
        self._output_path = Path(output_path)
        self._lookup_dir = Path(lookup_dir)
        self._lookup_resolution_um = lookup_resolution_um
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _progress(self, message: str, current: int, total: int) -> None:
        if self._cancel_requested:
            raise RuntimeError("Flatmap Parquet preparation cancelled.")
        self.progress.emit(str(message), int(current), int(total))

    def run(self) -> None:
        try:
            from .flatmap_parquet import augment_neuron_parquet_with_flatmaps
            from .flatmap_profiles import discover_flatmap_lookup_set

            self._progress("Validating bilateral lookup files...", 0, 0)
            lookup_cache_dir = self._output_path.parent / ".flatmap-lookup-arrays"
            lookup_set = discover_flatmap_lookup_set(
                self._lookup_dir,
                lookup_resolution_um=self._lookup_resolution_um,
                npy_cache_dir=lookup_cache_dir,
                progress_callback=self._progress,
                cancel_callback=lambda: self._cancel_requested,
            )
            summary = augment_neuron_parquet_with_flatmaps(
                self._source_path,
                self._output_path,
                lookup_set,
                npy_cache_dir=lookup_cache_dir,
                progress_callback=self._progress,
                cancel_callback=lambda: self._cancel_requested,
            )
            self.finished.emit(summary)
        except Exception as exc:
            logger.exception("Flatmap Parquet preparation failed")
            self.error.emit(str(exc))


class RegionCacheOpenWorker(QObject):
    """Open and fully validate an existing flatmap region cache."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, cache_dir: str | Path) -> None:
        super().__init__()
        self._cache_dir = Path(cache_dir)

    def run(self) -> None:
        """Validate the cache away from the Qt/VisPy rendering thread."""
        try:
            from .flatmap_region_cache import open_region_cache

            with startup_timing(
                logger,
                "flatmap_region_cache_open_worker",
                cache_dir=self._cache_dir,
            ):
                cache = open_region_cache(self._cache_dir)
            self.finished.emit(cache)
        except Exception as exc:
            logger.exception(
                "Flatmap region-cache open failed: %s",
                self._cache_dir,
            )
            self.error.emit(str(exc))


class RegionCacheBuildWorker(QObject):
    """Build one shaped/square fixed-grid region-cache profile."""

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        lookup_dir: str | Path,
        annotation_path: str | Path,
        atlas_name: str,
        atlas_version: str | None,
        atlas_resolution_um,
        atlas_structures,
        y_bins: int,
        depth_bin_um: float,
        lookup_resolution_um: float | None = None,
    ) -> None:
        super().__init__()
        self._cache_dir = Path(cache_dir)
        self._lookup_dir = Path(lookup_dir)
        self._annotation_path = Path(annotation_path)
        self._atlas_name = str(atlas_name)
        self._atlas_version = atlas_version
        self._atlas_resolution_um = atlas_resolution_um
        self._atlas_structures = atlas_structures
        self._y_bins = int(y_bins)
        self._depth_bin_um = float(depth_bin_um)
        self._lookup_resolution_um = lookup_resolution_um
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _progress(self, message: str, current: int, total: int) -> None:
        if self._cancel_requested:
            raise RuntimeError("Flatmap region-cache build cancelled.")
        self.progress.emit(str(message), int(current), int(total))

    def run(self) -> None:
        try:
            from .flatmap_profiles import discover_flatmap_lookup_set
            from .flatmap_region_cache import build_region_cache_profile

            self._progress("Validating lookup set and atlas grid...", 0, 0)
            lookup_set = discover_flatmap_lookup_set(
                self._lookup_dir,
                lookup_resolution_um=self._lookup_resolution_um,
                npy_cache_dir=self._cache_dir / "lookup-arrays",
                progress_callback=self._progress,
                cancel_callback=lambda: self._cancel_requested,
            )
            bounds_by_style = {
                style: {
                    "x_bounds": grid.x_bounds,
                    "y_bounds": grid.y_bounds,
                    "depth_bounds_um": grid.depth_bounds_um,
                }
                for style, grid in (
                    ("shaped", lookup_set.shaped_grid),
                    ("square", lookup_set.square_grid),
                )
            }
            profile = build_region_cache_profile(
                self._cache_dir,
                lookup_set,
                annotation_path=self._annotation_path,
                atlas_name=self._atlas_name,
                atlas_version=self._atlas_version,
                atlas_resolution_um=self._atlas_resolution_um,
                atlas_structures=self._atlas_structures,
                y_bins=self._y_bins,
                depth_bin_um=self._depth_bin_um,
                bounds_by_style=bounds_by_style,
                cancel_callback=lambda: self._cancel_requested,
                progress_callback=self._progress,
            )
            self.finished.emit(profile)
        except Exception as exc:
            logger.exception("Flatmap region-cache build failed")
            self.error.emit(str(exc))


class AppendPointFileWorker(QObject):
    """Append a point CSV or point Parquet into an existing point Parquet."""

    progress = Signal(str, int, int)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(
        self,
        input_path: str,
        mapping_path: str | None,
        parquet_path: str,
        output_path: str | None = None,
    ):
        super().__init__()
        self._input_path = Path(input_path)
        self._mapping_path = Path(mapping_path) if mapping_path is not None else None
        self._parquet_path = Path(parquet_path)
        self._output_path = (
            Path(output_path) if output_path is not None else self._parquet_path
        )

    def run(self) -> None:
        """Execute the append pipeline."""
        try:
            from .point_import import (
                append_point_csv_to_parquet,
                append_point_parquet_to_parquet,
            )

            if self._input_path.suffix.lower() == ".parquet":
                self.progress.emit("Validating point Parquet schemas...", 1, 3)
                summary = append_point_parquet_to_parquet(
                    self._input_path,
                    self._parquet_path,
                    self._output_path,
                )
            else:
                self.progress.emit("Validating point CSV and target Parquet...", 1, 3)
                summary = append_point_csv_to_parquet(
                    self._input_path,
                    self._mapping_path,
                    self._parquet_path,
                    self._output_path,
                )
            self.progress.emit("Refreshing saved point Parquet...", 2, 3)
            self.progress.emit("Done", 3, 3)
            self.finished.emit(str(self._output_path), summary)

        except Exception as e:
            logger.exception("Point file append failed")
            self.error.emit(str(e))


AppendPointCSVWorker = AppendPointFileWorker


class ConvertPointCSVWorker(QObject):
    """Convert one or more point CSV files into a point Parquet in the background."""

    progress = Signal(str, int, int)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(
        self,
        csv_paths: list[str],
        output_path: str,
        mapping_path: str | None = None,
    ):
        super().__init__()
        self._csv_paths = [Path(path) for path in csv_paths]
        self._output_path = Path(output_path)
        self._mapping_path = Path(mapping_path) if mapping_path is not None else None

    def run(self) -> None:
        """Execute the point-CSV conversion pipeline."""
        try:
            from .point_import import convert_point_csv_files_to_parquet

            summary = convert_point_csv_files_to_parquet(
                self._csv_paths,
                self._output_path,
                self._mapping_path,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(str(self._output_path), summary)

        except Exception as e:
            logger.exception("Point CSV conversion failed")
            self.error.emit(str(e))


class CorrelationWorker(QObject):
    """Compute the full correlation + clustering pipeline in background.

    Steps:
    1. Extract and dilate region mask
    2. Build voxel ID map
    3. Compute pairwise Pearson correlations via DuckDB
    4. Build correlation matrix
    5. Hierarchical clustering

    Signals
    -------
    progress(str, int, int)
        (step_name, current_step, total_steps)
    finished(ClusterResult)
        Emitted with the clustering result on success.
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        region_selection: ClusterRegionSelection | None,
        dilation_fraction: float = 0.2,
        linkage_method: str = "average",
        n_clusters: int = 5,
        file_ids: list[str] | None = None,
        voxel_id_map: np.ndarray | None = None,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_selection = region_selection
        self._dilation_fraction = dilation_fraction
        self._linkage_method = linkage_method
        self._n_clusters = n_clusters
        self._file_ids = (
            None if file_ids is None else [str(file_id) for file_id in file_ids]
        )
        self._voxel_id_map = voxel_id_map

    def run(self) -> None:
        """Execute the full pipeline."""
        try:
            import duckdb

            from .analysis.clustering import compute_clustermap_data
            from .analysis.correlation import (
                compute_pearson_correlation_matrix,
                correlation_long_to_matrix,
            )

            total = 5
            voxel_id_map = self._voxel_id_map
            if voxel_id_map is None and self._region_selection is not None:
                from .analysis.mask import get_expanded_region_voxel_ids_for_regions

                self.progress.emit("Extracting and dilating region mask...", 1, total)
                voxel_id_map = get_expanded_region_voxel_ids_for_regions(
                    self._atlas,
                    self._region_selection.selected_region_acronyms,
                    self._dilation_fraction,
                )
            else:
                self.progress.emit("Using prepared clustering input...", 1, total)

            self.progress.emit("Computing pairwise correlations...", 2, total)
            conn = duckdb.connect()
            try:
                resolution = float(self._atlas.resolution[0])
                corr_df = compute_pearson_correlation_matrix(
                    conn,
                    self._parquet_path,
                    voxel_id_map,
                    resolution=resolution,
                    file_ids=self._file_ids,
                )
            finally:
                conn.close()

            self.progress.emit("Building correlation matrix...", 3, total)
            mat_df, mat = correlation_long_to_matrix(corr_df)

            self.progress.emit("Clustering...", 4, total)
            result = compute_clustermap_data(
                mat,
                list(mat_df.columns),
                method=self._linkage_method,
                n_clusters=self._n_clusters,
            )
            _attach_cluster_run_metadata(
                result,
                atlas=self._atlas,
                parquet_path=self._parquet_path,
                region_selection=self._region_selection,
                analysis_method="voxel_correlation",
                clustering_algorithm="hierarchical",
                distance_metric="one_minus_pearson_r",
                clustering_linkage=self._linkage_method,
                dendrogram_linkage=self._linkage_method,
                dilation_fraction=(
                    self._dilation_fraction
                    if self._region_selection is not None
                    else 0.0
                ),
                requested_cluster_count=self._n_clusters,
            )

            self.progress.emit("Done", 5, total)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Correlation pipeline failed")
            self.error.emit(str(e))


class FlatmapCorrelationWorker(QObject):
    """Compute flatmap-space voxel correlation + clustering in background."""

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        *,
        source,
        atlas: BrainGlobeAtlas,
        parquet_path: str,
        region_selection: ClusterRegionSelection | None = None,
        linkage_method: str = "average",
        n_clusters: int = 5,
    ):
        super().__init__()
        self._source = source
        self._atlas = atlas
        self._parquet_path = parquet_path
        self._region_selection = region_selection
        self._linkage_method = linkage_method
        self._n_clusters = n_clusters

    def _build_region_mask(self):
        """Return a flatmap-space boolean mask for the selected region."""
        if self._region_selection is None:
            return None, {}
        represented_ids = list(self._region_selection.represented_region_ids)
        if not represented_ids:
            return None, {}
        mask_region_ids = self._expanded_selected_region_ids(represented_ids)

        source = self._source
        cache_dir = getattr(source, "cache_dir", None)
        cache_profile_id = getattr(source, "cache_profile_id", None)
        if cache_dir and cache_profile_id:
            from .flatmap_region_cache import (
                materialize_region_selection,
                open_region_cache,
            )

            cache = open_region_cache(cache_dir)
            try:
                profile = cache.profile(cache_profile_id)
                result = materialize_region_selection(
                    profile,
                    mask_region_ids,
                    style=(
                        getattr(source, "cache_style", None)
                        or source.flatmap_style
                        or "both_shaped"
                    ),
                    include_surfaces=False,
                    include_outlines=False,
                )
                mask = np.array(result.labels > 0, dtype=bool, copy=True)
                source_shape = tuple(int(size) for size in source.volume_shape)
                if (
                    source.include_depth_minus_one
                    and source_shape[0] == mask.shape[0] + 1
                ):
                    padded = np.zeros(source_shape, dtype=bool)
                    padded[1:, :, :] = mask
                    mask = padded
                elif mask.shape != source_shape:
                    raise RuntimeError(
                        "Cached region mask shape does not match the flatmap heatmap; "
                        f"got mask {mask.shape} and heatmap {source_shape}."
                    )
                return mask, {
                    "flatmap_region_source": "precomputed_cache",
                    "flatmap_region_cache_path": str(cache_dir),
                    "flatmap_region_cache_profile_id": str(cache_profile_id),
                    "flatmap_region_labeled_voxels": int(result.summary.labeled_bins),
                    "flatmap_region_collision_voxels": int(
                        result.summary.collision_bins
                    ),
                    "flatmap_region_source_voxel_count": int(
                        result.summary.source_voxel_count
                    ),
                    "flatmap_region_represented_region_ids": [
                        int(region_id) for region_id in result.represented_region_ids
                    ],
                }
            finally:
                close = getattr(cache, "close", None)
                if callable(close):
                    close()
        if getattr(source, "coordinate_mode", None) == "parquet_columns":
            raise RuntimeError(
                "Region-filtered precomputed flatmap clustering requires a "
                "compatible cache profile; NRRD recomputation is never automatic."
            )
        if not source.flatmap_path or not source.depth_path:
            raise RuntimeError(
                "Region-filtered flatmap clustering requires the flatmap and "
                "depth NRRD files used to render the heatmap."
            )

        from .flatmap_labels import build_flatmap_region_label_volume
        from .flatmap_loader import load_flatmap_volume_set

        volume_set = load_flatmap_volume_set(source.flatmap_path, source.depth_path)
        result = build_flatmap_region_label_volume(
            np.asarray(self._atlas.annotation),
            volume_set.flatmap,
            volume_set.depth,
            selected_region_ids=mask_region_ids,
            # Both counts come from the render this mask is projected onto, so
            # the mask cannot end up on a different grid than the heatmap.
            y_bins=source.y_bins,
            x_bins=source.x_bins,
            depth_bin_um=source.depth_bin_um,
            invalid_zero_sentinel=source.invalid_zero_sentinel,
            invalid_negative_one_sentinel=source.invalid_negative_one_sentinel,
            lookup_stats=source.lookup_stats,
            mirror_depth_fallback=getattr(source, "mirror_depth_fallback", True),
            mirror_coord_axis=getattr(source, "mirror_coord_axis", 2),
        )
        mask = np.asarray(result.labels > 0, dtype=bool)
        source_shape = tuple(int(size) for size in source.volume_shape)
        if source.include_depth_minus_one and source_shape[0] == mask.shape[0] + 1:
            padded = np.zeros(source_shape, dtype=bool)
            padded[1:, :, :] = mask
            mask = padded
        elif mask.shape != source_shape:
            raise RuntimeError(
                "Projected region mask shape does not match the flatmap heatmap; "
                f"got mask {mask.shape} and heatmap {source_shape}."
            )

        return mask, {
            "flatmap_region_labeled_voxels": int(result.summary.labeled_voxels),
            "flatmap_region_valid_source_voxels": int(
                result.summary.valid_source_voxels
            ),
            "flatmap_region_mirrored_depth_source_voxels": int(
                getattr(result.summary, "mirrored_depth_source_voxels", 0)
            ),
            "flatmap_region_collision_voxels": int(result.summary.collision_voxels),
            "flatmap_region_represented_region_ids": [
                int(region_id) for region_id in result.represented_region_ids
            ],
        }

    def _expanded_selected_region_ids(
        self,
        fallback_region_ids: list[int],
    ) -> list[int]:
        """Expand direct selections through the atlas structure catalog only."""
        direct_ids = {
            int(value)
            for value in getattr(
                self._region_selection,
                "selected_region_ids",
                (),
            )
            if int(value) > 0
        }
        structures = getattr(self._atlas, "structures", None)
        if not direct_ids or not isinstance(structures, Mapping):
            return sorted({int(value) for value in fallback_region_ids})

        expanded = set(direct_ids)
        for key, structure in structures.items():
            if not isinstance(structure, Mapping):
                continue
            try:
                region_id = int(structure.get("id", key))
            except (TypeError, ValueError):
                continue
            raw_path = structure.get("structure_id_path", ()) or ()
            if isinstance(raw_path, str):
                try:
                    path_ids = {
                        int(part) for part in raw_path.strip("/").split("/") if part
                    }
                except ValueError:
                    continue
            else:
                try:
                    path_ids = {int(value) for value in raw_path}
                except (TypeError, ValueError):
                    continue
            if region_id in direct_ids or direct_ids.intersection(path_ids):
                expanded.add(region_id)
        return sorted(expanded)

    @staticmethod
    def _lookup_mode_counts(projected_nodes) -> dict[str, int]:
        if projected_nodes is None:
            return {
                "flatmap_direct_lookup_node_count": 0,
                "flatmap_mirrored_depth_lookup_node_count": 0,
                "flatmap_mirrored_lookup_node_count": 0,
                "flatmap_unmapped_lookup_node_count": 0,
            }
        if "flatmap_lookup_mode" in projected_nodes.columns:
            modes = (
                projected_nodes["flatmap_lookup_mode"].fillna("").astype(str).to_numpy()
            )
        else:
            valid = projected_nodes.get("valid")
            if valid is None:
                valid_mask = np.zeros(len(projected_nodes), dtype=bool)
            else:
                valid_mask = valid.fillna(False).astype(bool).to_numpy()
            modes = np.where(valid_mask, "direct", "unmapped")
        direct = int(np.count_nonzero(modes == "direct"))
        mirrored_depth = int(np.count_nonzero(modes == "mirrored_depth"))
        mirrored = int(np.count_nonzero(modes == "mirrored"))
        unmapped = int(np.count_nonzero(modes == "unmapped"))
        return {
            "flatmap_direct_lookup_node_count": direct,
            "flatmap_mirrored_depth_lookup_node_count": mirrored_depth,
            "flatmap_mirrored_lookup_node_count": mirrored,
            "flatmap_unmapped_lookup_node_count": unmapped,
        }

    def run(self) -> None:
        """Execute the flatmap correlation pipeline."""
        try:
            from .analysis.flatmap_correlation import (
                compute_flatmap_voxel_correlation_result,
            )

            total = 4
            self.progress.emit("Preparing flatmap voxel source...", 1, total)
            region_mask, region_metadata = self._build_region_mask()

            self.progress.emit("Computing flatmap voxel correlations...", 2, total)
            result, count_data = compute_flatmap_voxel_correlation_result(
                self._source,
                method=self._linkage_method,
                n_clusters=self._n_clusters,
                region_mask=region_mask,
            )

            self.progress.emit("Recording flatmap clustering metadata...", 3, total)
            source = self._source
            extra_metadata = {
                "flatmap_style": source.flatmap_style,
                "flatmap_coordinate_mode": source.coordinate_mode,
                "flatmap_y_bins": int(source.y_bins),
                "flatmap_x_bins": (
                    None if source.x_bins is None else int(source.x_bins)
                ),
                "flatmap_depth_bin_um": float(source.depth_bin_um),
                "flatmap_include_depth_minus_one": bool(source.include_depth_minus_one),
                "flatmap_path": source.flatmap_path,
                "depth_path": source.depth_path,
                "flatmap_cache_path": getattr(source, "cache_dir", None),
                "flatmap_cache_profile_id": getattr(source, "cache_profile_id", None),
                "flatmap_input_neuron_count": int(len(source.input_file_ids)),
                "flatmap_clustered_neuron_count": int(len(result.neuron_ids)),
                "flatmap_unassigned_neuron_count": int(
                    len(result.unassigned_neuron_ids)
                ),
                "flatmap_rendered_node_count": int(count_data.rendered_node_count),
                "flatmap_occupied_voxel_count": int(len(count_data.voxel_ids)),
            }
            extra_metadata.update(self._lookup_mode_counts(source.projected_nodes))
            extra_metadata.update(region_metadata)
            _attach_cluster_run_metadata(
                result,
                atlas=self._atlas,
                parquet_path=self._parquet_path,
                region_selection=self._region_selection,
                analysis_method="flatmap_voxel_correlation",
                clustering_algorithm="hierarchical",
                distance_metric="one_minus_pearson_r",
                clustering_linkage=self._linkage_method,
                dendrogram_linkage=self._linkage_method,
                dilation_fraction=0.0,
                requested_cluster_count=self._n_clusters,
                extra_metadata=extra_metadata,
            )

            self.progress.emit("Done", 4, total)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Flatmap correlation pipeline failed")
            self.error.emit(str(e))


class FlatmapParquetCorrelationWorker(QObject):
    """Flatmap-space voxel correlation clustering straight from Parquet.

    Unlike :class:`FlatmapCorrelationWorker`, this worker does not require a
    rendered heatmap source.  It bins the precomputed flatmap/depth Parquet
    columns via DuckDB, so it is available whenever the coordinates exist in
    the loaded Parquet.

    ``collapse_depth`` removes the depth axis from the voxel grid so neurons
    correlate on flat map footprint alone.  There is deliberately no depth
    *weight* here: the correlation compares voxel occupancy, which has no notion
    of distance between voxels, and scaling depth would merely duplicate
    ``depth_bin_um``.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        *,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        style: str,
        y_bins: int,
        depth_bin_um: float,
        include_depth_minus_one: bool = True,
        linkage_method: str = "average",
        n_clusters: int = 5,
        file_ids: list[str] | None = None,
        collapse_depth: bool = False,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._style = style
        self._y_bins = int(y_bins)
        self._depth_bin_um = float(depth_bin_um)
        self._include_depth_minus_one = bool(include_depth_minus_one)
        self._linkage_method = linkage_method
        self._n_clusters = int(n_clusters)
        self._file_ids = (
            None if file_ids is None else [str(file_id) for file_id in file_ids]
        )
        self._collapse_depth = bool(collapse_depth)

    def run(self) -> None:
        """Execute the parquet-driven flatmap correlation pipeline."""
        try:
            from .analysis.flatmap_correlation import (
                compute_flatmap_voxel_correlation_from_parquet,
            )

            total = 3
            self.progress.emit("Binning flatmap coordinates in DuckDB...", 1, total)
            result, count_data, provenance = (
                compute_flatmap_voxel_correlation_from_parquet(
                    self._parquet_path,
                    style=self._style,
                    y_bins=self._y_bins,
                    depth_bin_um=self._depth_bin_um,
                    include_depth_minus_one=self._include_depth_minus_one,
                    method=self._linkage_method,
                    n_clusters=self._n_clusters,
                    file_ids=self._file_ids,
                    collapse_depth=self._collapse_depth,
                )
            )

            self.progress.emit("Recording flatmap clustering metadata...", 2, total)
            extra_metadata = {
                "flatmap_style": provenance.style,
                "flatmap_coordinate_mode": "parquet_columns",
                "flatmap_y_bins": int(provenance.y_bins),
                "flatmap_x_bins": int(provenance.x_bins),
                "flatmap_depth_bin_um": float(provenance.depth_bin_um),
                "flatmap_include_depth_minus_one": bool(
                    provenance.include_depth_minus_one
                ),
                "flatmap_volume_shape": [int(size) for size in provenance.volume_shape],
                "flatmap_collapse_depth": bool(provenance.collapse_depth),
                "flatmap_input_neuron_count": int(
                    len(result.neuron_ids) + len(result.unassigned_neuron_ids)
                ),
                "flatmap_clustered_neuron_count": int(len(result.neuron_ids)),
                "flatmap_unassigned_neuron_count": int(
                    len(result.unassigned_neuron_ids)
                ),
                "flatmap_rendered_node_count": int(count_data.rendered_node_count),
                "flatmap_occupied_voxel_count": int(len(count_data.voxel_ids)),
            }
            _attach_cluster_run_metadata(
                result,
                atlas=self._atlas,
                parquet_path=self._parquet_path,
                region_selection=None,
                analysis_method="flatmap_voxel_correlation",
                clustering_algorithm="hierarchical",
                distance_metric=(
                    "one_minus_pearson_r_flatmap_xy"
                    if provenance.collapse_depth
                    else "one_minus_pearson_r"
                ),
                clustering_linkage=self._linkage_method,
                dendrogram_linkage=self._linkage_method,
                dilation_fraction=0.0,
                requested_cluster_count=self._n_clusters,
                extra_metadata=extra_metadata,
            )

            self.progress.emit("Done", 3, total)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Flatmap parquet correlation pipeline failed")
            self.error.emit(str(e))


class FlatmapSomaClusterWorker(QObject):
    """Cluster neurons by soma location in flatmap + depth space.

    Mirrors :class:`SomaClusterWorker` but computes Euclidean distances in
    flatmap ``(x_flat, y_flat, depth_um)`` space using the precomputed
    Parquet columns.  Region filtering is intentionally not applied here; it
    is handled separately for flatmap space.

    The raw columns mix units — ``x_flat``/``y_flat`` are normalized floats
    while ``depth_um`` is microns — so coordinates are rescaled before any
    distance is computed.  Both flat map axes are divided by one shared
    divisor, the ``y`` span, so the flat map plane is scaled without being
    distorted; ``depth_um`` gets its own divisor because it is a different
    physical quantity.  ``depth_scale`` then weights the depth axis relative to
    the flat map, and ``include_depth=False`` drops depth entirely for a purely
    tangential clustering.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        *,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        style: str,
        algorithm: str = "hierarchical",
        linkage_method: str = "ward",
        n_clusters: int = 5,
        eps: float = DEFAULT_FLATMAP_SOMA_DBSCAN_EPS,
        min_samples: int = 5,
        file_ids: list[str] | None = None,
        depth_scale: float = DEFAULT_FLATMAP_DEPTH_SCALE,
        include_depth: bool = True,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._style = style
        self._algorithm = algorithm
        self._linkage_method = linkage_method
        self._n_clusters = int(n_clusters)
        self._eps = float(eps)
        self._min_samples = int(min_samples)
        self._file_ids = (
            None if file_ids is None else [str(file_id) for file_id in file_ids]
        )
        self._depth_scale = float(depth_scale)
        self._include_depth = bool(include_depth)

    def run(self) -> None:
        """Execute the flatmap-space soma clustering pipeline."""
        try:
            from .analysis.clustering import (
                cluster_somas_dbscan,
                cluster_somas_hierarchical,
                cluster_somas_kmeans,
            )
            from .analysis.flatmap_correlation import (
                normalize_flatmap_soma_coordinates,
                query_flatmap_soma_coordinates,
                resolve_flatmap_depth_normalization,
            )

            total = 3
            self.progress.emit("Querying soma flatmap coordinates...", 1, total)
            filtered_ids, raw_coords = query_flatmap_soma_coordinates(
                self._parquet_path,
                style=self._style,
                file_ids=self._file_ids,
            )

            if len(filtered_ids) < 2:
                self.error.emit(
                    f"Only {len(filtered_ids)} soma(s) have valid flatmap/depth "
                    "coordinates — need at least 2 for clustering."
                )
                return

            # The raw columns mix normalized flat map units with microns, so
            # depth would otherwise supply >99.99% of the Euclidean variance and
            # reduce the run to a laminar partition.
            normalization = resolve_flatmap_depth_normalization(
                self._parquet_path,
                style=self._style,
                depth_scale=self._depth_scale,
                include_depth=self._include_depth,
            )
            filtered_coords = normalize_flatmap_soma_coordinates(
                raw_coords,
                normalization,
            )

            self.progress.emit(
                f"Clustering {len(filtered_ids)} somas ({self._algorithm})...",
                2,
                total,
            )

            if self._algorithm == "hierarchical":
                result = cluster_somas_hierarchical(
                    filtered_coords,
                    filtered_ids,
                    method=self._linkage_method,
                    n_clusters=self._n_clusters,
                )
                clustering_linkage = self._linkage_method
                dendrogram_linkage = self._linkage_method
                requested_cluster_count = self._n_clusters
            elif self._algorithm == "kmeans":
                result = cluster_somas_kmeans(
                    filtered_coords,
                    filtered_ids,
                    n_clusters=self._n_clusters,
                )
                clustering_linkage = None
                dendrogram_linkage = "average"
                requested_cluster_count = self._n_clusters
            elif self._algorithm == "dbscan":
                result = cluster_somas_dbscan(
                    filtered_coords,
                    filtered_ids,
                    eps=self._eps,
                    min_samples=self._min_samples,
                )
                clustering_linkage = None
                dendrogram_linkage = "average"
                requested_cluster_count = None
            else:
                self.error.emit(f"Unknown algorithm: {self._algorithm}")
                return

            _attach_cluster_run_metadata(
                result,
                atlas=self._atlas,
                parquet_path=self._parquet_path,
                region_selection=None,
                analysis_method="flatmap_soma_location",
                clustering_algorithm=self._algorithm,
                # Renamed away from "unit_hemisphere" when the two flat map axes
                # started sharing one divisor: a hemisphere is 1.0 tall but no
                # longer forced to 1.0 wide, and the rename keeps results from
                # before that change from being compared as if they matched.
                distance_metric=(
                    "euclidean_flatmap_isotropic_plus_depth"
                    if normalization.include_depth
                    else "euclidean_flatmap_isotropic"
                ),
                clustering_linkage=clustering_linkage,
                dendrogram_linkage=dendrogram_linkage,
                dilation_fraction=0.0,
                requested_cluster_count=requested_cluster_count,
                dbscan_eps=self._eps if self._algorithm == "dbscan" else None,
                dbscan_min_samples=self._min_samples
                if self._algorithm == "dbscan"
                else None,
                extra_metadata={
                    "flatmap_style": self._style,
                    "flatmap_normalization": normalization.to_dict(),
                    # eps is in normalized flat map units here, not the
                    # microns the CCF soma clustering uses.
                    "dbscan_eps_units": (
                        "normalized_hemisphere" if self._algorithm == "dbscan" else None
                    ),
                },
            )

            self.progress.emit("Done", 3, total)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Flatmap soma clustering pipeline failed")
            self.error.emit(str(e))


class SomaClusterWorker(QObject):
    """Cluster neurons by soma location in a background thread.

    Steps:
    1. Build expanded region mask
    2. Query soma locations from parquet
    3. Filter somas to those inside the expanded region
    4. Cluster using the chosen algorithm

    Signals
    -------
    progress(str, int, int)
        (step_name, current_step, total_steps)
    finished(ClusterResult)
        Emitted with the clustering result on success.
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        region_selection: ClusterRegionSelection | None,
        dilation_fraction: float = 0.2,
        algorithm: str = "hierarchical",
        linkage_method: str = "ward",
        n_clusters: int = 5,
        eps: float = 100.0,
        min_samples: int = 5,
        file_ids: list[str] | None = None,
        voxel_id_map: np.ndarray | None = None,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_selection = region_selection
        self._dilation_fraction = dilation_fraction
        self._algorithm = algorithm
        self._linkage_method = linkage_method
        self._n_clusters = n_clusters
        self._eps = eps
        self._min_samples = min_samples
        self._file_ids = (
            None if file_ids is None else [str(file_id) for file_id in file_ids]
        )
        self._voxel_id_map = voxel_id_map

    def run(self) -> None:
        """Execute the soma clustering pipeline."""
        try:
            from .analysis.clustering import (
                cluster_somas_dbscan,
                cluster_somas_hierarchical,
                cluster_somas_kmeans,
                query_ccf_soma_coordinates,
            )

            total = 4
            run_start = perf_counter()
            resolution = float(self._atlas.resolution[0])
            region_label = (
                ",".join(self._region_selection.selected_region_acronyms)
                if self._region_selection is not None
                else "all scoped CCF coordinates"
            )
            logger.debug(
                "SomaClusterWorker start: algorithm=%s linkage=%s n_clusters=%d eps=%s min_samples=%d region=%s dilation_fraction=%.3f resolution=%.3f parquet=%s",
                self._algorithm,
                self._linkage_method,
                self._n_clusters,
                self._eps,
                self._min_samples,
                region_label,
                self._dilation_fraction,
                resolution,
                self._parquet_path,
            )
            voxel_id_map = self._voxel_id_map
            if voxel_id_map is None and self._region_selection is not None:
                from .analysis.mask import get_expanded_region_voxel_ids_for_regions

                self.progress.emit("Extracting and dilating region mask...", 1, total)
                mask_start = perf_counter()
                voxel_id_map = get_expanded_region_voxel_ids_for_regions(
                    self._atlas,
                    self._region_selection.selected_region_acronyms,
                    self._dilation_fraction,
                )
                logger.debug(
                    "SomaClusterWorker mask ready: shape=%s dtype=%s elapsed=%.3fs",
                    voxel_id_map.shape,
                    voxel_id_map.dtype,
                    perf_counter() - mask_start,
                )
            else:
                self.progress.emit("Using prepared clustering input...", 1, total)

            self.progress.emit("Querying soma locations...", 2, total)
            query_start = perf_counter()
            filtered_ids, filtered_coords, _node_count = query_ccf_soma_coordinates(
                self._parquet_path,
                resolution=resolution,
                file_ids=self._file_ids,
                voxel_id_map=voxel_id_map,
            )
            logger.debug(
                "SomaClusterWorker soma query complete: rows=%d elapsed=%.3fs",
                len(filtered_ids),
                perf_counter() - query_start,
            )

            if not filtered_ids:
                self.error.emit("No soma nodes found in the dataset.")
                return

            logger.info(
                "Soma filtering: %d somas retained for %s",
                len(filtered_ids),
                region_label,
            )

            if len(filtered_ids) < 2:
                self.error.emit(
                    f"Only {len(filtered_ids)} soma(s) found in '{region_label}' "
                    "— need at least 2 for clustering."
                )
                return

            self.progress.emit(
                f"Clustering {len(filtered_ids)} somas ({self._algorithm})...", 3, total
            )

            if self._algorithm == "hierarchical":
                result = cluster_somas_hierarchical(
                    filtered_coords,
                    filtered_ids,
                    method=self._linkage_method,
                    n_clusters=self._n_clusters,
                )
                clustering_linkage = self._linkage_method
                dendrogram_linkage = self._linkage_method
                requested_cluster_count = self._n_clusters
            elif self._algorithm == "kmeans":
                result = cluster_somas_kmeans(
                    filtered_coords,
                    filtered_ids,
                    n_clusters=self._n_clusters,
                )
                clustering_linkage = None
                dendrogram_linkage = "average"
                requested_cluster_count = self._n_clusters
            elif self._algorithm == "dbscan":
                result = cluster_somas_dbscan(
                    filtered_coords,
                    filtered_ids,
                    eps=self._eps,
                    min_samples=self._min_samples,
                )
                clustering_linkage = None
                dendrogram_linkage = "average"
                requested_cluster_count = None
            else:
                self.error.emit(f"Unknown algorithm: {self._algorithm}")
                return

            _attach_cluster_run_metadata(
                result,
                atlas=self._atlas,
                parquet_path=self._parquet_path,
                region_selection=self._region_selection,
                analysis_method="soma_location",
                clustering_algorithm=self._algorithm,
                distance_metric="euclidean_um",
                clustering_linkage=clustering_linkage,
                dendrogram_linkage=dendrogram_linkage,
                dilation_fraction=(
                    self._dilation_fraction
                    if self._region_selection is not None
                    else 0.0
                ),
                requested_cluster_count=requested_cluster_count,
                dbscan_eps=self._eps if self._algorithm == "dbscan" else None,
                dbscan_min_samples=self._min_samples
                if self._algorithm == "dbscan"
                else None,
            )

            self.progress.emit("Done", 4, total)
            logger.debug(
                "SomaClusterWorker finished: total_elapsed=%.3fs neuron_count=%d",
                perf_counter() - run_start,
                len(result.neuron_ids),
            )
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Soma clustering pipeline failed")
            self.error.emit(str(e))


class HeatmapWorker(QObject):
    """Build a node-count heatmap volume in the background.

    Signals
    -------
    progress(str, int, int)
        (step_name, current_step, total_steps)
    finished(NDArray)
        Emitted with the 3D volume on success.
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        region_ids: list[int] | None = None,
        file_ids: list[str] | None = None,
        node_types: list[int] | None = None,
        soma_radius_um: float | None = None,
        depth_bin_factor: int = 1,
        depth_axis: int = 0,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_ids = region_ids
        self._file_ids = file_ids
        self._node_types = node_types
        self._soma_radius_um = soma_radius_um
        self._depth_bin_factor = depth_bin_factor
        self._depth_axis = depth_axis

    def run(self) -> None:
        """Execute the heatmap pipeline."""
        try:
            import duckdb

            from .analysis.heatmap import build_node_counts_volume

            self.progress.emit("Building heatmap volume...", 1, 2)
            conn = duckdb.connect()
            try:
                volume = build_node_counts_volume(
                    conn,
                    self._parquet_path,
                    self._atlas,
                    region_ids=self._region_ids,
                    file_ids=self._file_ids,
                    node_types=self._node_types,
                    soma_radius_um=self._soma_radius_um,
                    depth_bin_factor=self._depth_bin_factor,
                    depth_axis=self._depth_axis,
                )
            finally:
                conn.close()

            self.progress.emit("Done", 2, 2)
            self.finished.emit(volume)

        except Exception as e:
            logger.exception("Heatmap pipeline failed")
            self.error.emit(str(e))


class FlatmapHeatmapWorker(QObject):
    """Build a precomputed flatmap heatmap volume in the background.

    Mirrors :class:`HeatmapWorker`: it opens its own DuckDB connection and lets
    DuckDB bin the precomputed flatmap columns with a ``GROUP BY``, reading only
    the coordinate/validity columns rather than materializing every node in
    pandas.

    Signals
    -------
    progress(str, int, int)
        (step_name, current_step, total_steps)
    finished(object)
        Emitted with a ``FlatmapHeatmapVolumeResult`` on success.
    error(str)
        Emitted with an error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        parquet_path: str,
        *,
        style_key: str,
        color_mode: str,
        x_bounds: tuple[float, float] | None,
        y_bounds: tuple[float, float] | None,
        depth_range_um: tuple[float, float] | None,
        y_bins: int,
        x_bins: int | None = None,
        depth_bin_um: float,
        include_depth_minus_one: bool,
        file_ids: list[object] | None = None,
        cluster_map: dict[object, int | None] | None = None,
        plane_mode: str = "depth",
        allen_layer_map: AllenIsocortexLayerMap | None = None,
        allen_layer_indices: tuple[int, ...] | None = None,
        allen_layer_output_mode: str = "stack",
    ):
        super().__init__()
        self._parquet_path = str(parquet_path)
        self._style_key = str(style_key)
        self._color_mode = str(color_mode)
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        self._depth_range_um = depth_range_um
        self._y_bins = int(y_bins)
        # ``None`` lets the builder derive the square-bin x count.  A cache-backed
        # render passes the profile's stored count so the render grid matches the
        # cached mask exactly.
        self._x_bins = None if x_bins is None else int(x_bins)
        self._depth_bin_um = float(depth_bin_um)
        self._include_depth_minus_one = bool(include_depth_minus_one)
        self._file_ids = file_ids
        self._cluster_map = cluster_map
        self._plane_mode = str(plane_mode)
        self._allen_layer_map = allen_layer_map
        self._allen_layer_indices = allen_layer_indices
        self._allen_layer_output_mode = str(allen_layer_output_mode)

    def run(self) -> None:
        """Execute the DuckDB flatmap heatmap pipeline."""
        try:
            import duckdb

            from .flatmap_heatmap import (
                FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION,
                FLATMAP_PLANE_MODE_ALLEN_LAYERS,
                FLATMAP_PLANE_MODE_FLAT,
                build_allen_layer_heatmap_volume_result,
                build_flatmap_heatmap_volume_result,
                compute_flatmap_bounds_from_parquet,
            )

            total_steps = 3
            conn = duckdb.connect()
            try:
                x_bounds = self._x_bounds
                y_bounds = self._y_bounds
                depth_range_um = self._depth_range_um
                needs_depth_bounds = self._plane_mode not in {
                    FLATMAP_PLANE_MODE_ALLEN_LAYERS,
                    FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION,
                }
                if (
                    x_bounds is None
                    or y_bounds is None
                    or (needs_depth_bounds and depth_range_um is None)
                ):
                    self.progress.emit("Deriving flatmap bounds...", 1, total_steps)
                    bounds = compute_flatmap_bounds_from_parquet(
                        conn,
                        self._parquet_path,
                        style_key=self._style_key,
                        file_ids=self._file_ids,
                    )
                    x_bounds = bounds["x_bounds"]
                    y_bounds = bounds["y_bounds"]
                    depth_range_um = bounds["depth_range_um"]

                if self._plane_mode in {
                    FLATMAP_PLANE_MODE_ALLEN_LAYERS,
                    FLATMAP_PLANE_MODE_ALLEN_LAYER_PROJECTION,
                }:
                    if self._allen_layer_map is None:
                        raise ValueError(
                            "Allen layer rendering requires an atlas-derived "
                            "Isocortex layer mapping."
                        )
                    result = build_allen_layer_heatmap_volume_result(
                        conn,
                        self._parquet_path,
                        style_key=self._style_key,
                        color_mode=self._color_mode,
                        layer_map=self._allen_layer_map,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        y_bins=self._y_bins,
                        x_bins=self._x_bins,
                        file_ids=self._file_ids,
                        cluster_map=self._cluster_map,
                        selected_layer_indices=self._allen_layer_indices,
                        output_mode=self._allen_layer_output_mode,
                        progress_callback=self.progress.emit,
                        progress_total=total_steps,
                    )
                else:
                    # A flat render collapses the depth axis but keeps depth's
                    # say over which nodes are counted, so it takes the same
                    # builder with the same bounds.
                    result = build_flatmap_heatmap_volume_result(
                        conn,
                        self._parquet_path,
                        style_key=self._style_key,
                        color_mode=self._color_mode,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        depth_range_um=depth_range_um,
                        y_bins=self._y_bins,
                        x_bins=self._x_bins,
                        depth_bin_um=self._depth_bin_um,
                        include_depth_minus_one=self._include_depth_minus_one,
                        file_ids=self._file_ids,
                        cluster_map=self._cluster_map,
                        progress_callback=self.progress.emit,
                        progress_total=total_steps,
                        collapse_depth=(
                            self._plane_mode == FLATMAP_PLANE_MODE_FLAT
                        ),
                    )
            finally:
                conn.close()

            self.progress.emit("Done", total_steps, total_steps)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Flatmap heatmap pipeline failed")
            self.error.emit(str(e))


class AnalysisExportWorker(QObject):
    """Export analysis outputs that benefit from background execution."""

    progress = Signal(str, int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        export_kind: str,
        output_path: str,
        cluster_result: ClusterResult,
        cluster_color_map: dict[str, list[float]] | None = None,
        *,
        figure_title: str = "",
        x_label: str = "",
        y_label: str = "",
    ):
        super().__init__()
        self._export_kind = str(export_kind)
        self._output_path = Path(output_path)
        self._cluster_result = cluster_result
        self._cluster_color_map = cluster_color_map or {}
        self._figure_title = figure_title
        self._x_label = x_label
        self._y_label = y_label

    def run(self) -> None:
        """Execute one analysis export."""
        try:
            from .analysis.export import (
                export_cluster_workbook,
                export_distance_workbook,
                export_extended_parquet,
            )

            if self._export_kind == "cluster_workbook":
                export_cluster_workbook(
                    self._output_path,
                    self._cluster_result,
                    self._cluster_color_map,
                    figure_title=self._figure_title,
                    x_label=self._x_label,
                    y_label=self._y_label,
                    progress_callback=self.progress.emit,
                )
            elif self._export_kind == "distance_workbook":
                export_distance_workbook(
                    self._output_path,
                    self._cluster_result,
                    self._cluster_color_map,
                    figure_title=self._figure_title,
                    x_label=self._x_label,
                    y_label=self._y_label,
                    progress_callback=self.progress.emit,
                )
            elif self._export_kind == "extended_parquet":
                export_extended_parquet(
                    self._output_path,
                    self._cluster_result,
                    progress_callback=self.progress.emit,
                )
            else:
                raise ValueError(f"Unknown analysis export kind: {self._export_kind}")

            self.finished.emit(str(self._output_path))

        except Exception as e:
            logger.exception("Analysis export failed")
            self.error.emit(str(e))


class TerminusWorker(QObject):
    """Find termini (childless nodes) of the selected node types in the background.

    Signals
    -------
    progress(str, int, int)
        (step_name, current_step, total_steps)
    finished(object, object)
        Emitted with ``(DataFrame, TerminusCoverage)`` on success.
    error(str)
        Emitted with an error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(
        self,
        parquet_path: str,
        file_ids: list[str] | None,
        node_types: list[int] | tuple[int, ...] | None,
    ):
        """Store the query parameters.

        Parameters
        ----------
        parquet_path : str
            Source Parquet file.
        file_ids : list[str] or None
            Neurons to cover; ``None`` covers every neuron in the source.
        node_types : list[int] or None
            SWC types to report; ``None`` reports every childless node. Stated
            explicitly rather than defaulted, so "all types" is never implied by
            omission.
        """
        super().__init__()
        self._parquet_path = str(parquet_path)
        self._file_ids = file_ids
        self._node_types = node_types

    def run(self) -> None:
        """Query termini and emit them with their coverage report."""
        try:
            import duckdb

            from .terminals import query_termini

            self.progress.emit("Finding termini...", 0, 1)
            source_path = self._parquet_path.replace("\\", "/")

            def report(done: int, total: int) -> None:
                self.progress.emit(
                    f"Scanning neurons for termini ({done:,}/{total:,})",
                    done,
                    max(total, 1),
                )

            conn = duckdb.connect()
            try:
                frame, coverage = query_termini(
                    conn,
                    f"read_parquet('{source_path}')",
                    node_types=self._node_types,
                    file_ids=self._file_ids,
                    progress_callback=report,
                )
            finally:
                conn.close()

            self.progress.emit("Done", 1, 1)
            self.finished.emit(frame, coverage)

        except Exception as e:
            logger.exception("Terminus detection failed")
            self.error.emit(str(e))

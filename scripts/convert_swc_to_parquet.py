#!/usr/bin/env python
"""Convert SWC files into a single Parquet file with optional alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from napari_neuron_navigator.hemisphere import Hemisphere
from napari_neuron_navigator.parquet import (
    BatchParquetConversionSummary,
    batch_convert_swc_to_parquet,
)


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _worker_count(value: str) -> int | None:
    """Parse worker count where ``auto`` maps to the library default."""
    if value.lower() == "auto":
        return None
    return _positive_int(value)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a directory of SWC files into one Parquet file, with "
            "optional hemisphere alignment before writing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pfc_complete neurons.parquet
      Convert all SWCs under pfc_complete into one fast raw-node Parquet

  %(prog)s pfc_complete neurons_right.parquet --hemisphere right
      Align files to the right hemisphere before writing

  %(prog)s pfc_complete neurons_annotated.parquet --annotate-regions --resolution 25
      Write Allen-region annotated output for registered SWCs
""",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input SWC directory or single SWC file",
    )
    parser.add_argument(
        "output_parquet",
        type=Path,
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--hemisphere",
        choices=[Hemisphere.LEFT.value, Hemisphere.RIGHT.value],
        default=None,
        help="Optional target hemisphere for alignment",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search subdirectories for SWC files (default: enabled)",
    )
    parser.add_argument(
        "--atlas",
        default="allen_mouse_10um",
        help="BrainGlobe atlas name for hemisphere detection",
    )
    parser.add_argument(
        "--coord-axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Coordinate axis for left-right orientation (default: 2)",
    )
    parser.add_argument(
        "--midline",
        type=float,
        default=None,
        help="Custom left-right midline in microns; skips atlas loading when set",
    )
    parser.add_argument(
        "--annotate-regions",
        action="store_true",
        help="Populate Allen region columns using the annotation volume",
    )
    parser.add_argument(
        "--build-regional-profile",
        action="store_true",
        help=(
            "Also build a validated .region_profile.parquet sidecar; enables "
            "region annotation"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=_positive_int,
        default=25,
        help="Allen annotation resolution in microns (default: 25)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Allen SDK cache directory",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=25,
        help="Files per serial flush or parallel worker chunk (default: 25)",
    )
    parser.add_argument(
        "--workers",
        type=_worker_count,
        default=None,
        help="Parallel worker count or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Optional directory for temporary shard and memmap files",
    )
    return parser.parse_args(args)


def _print_summary(summary: BatchParquetConversionSummary, output_path: Path) -> None:
    """Print the batch conversion summary."""
    print(f"Discovered files: {summary.discovered_files}")
    print(f"Processed files: {summary.processed_files}")
    print(f"Flipped files: {summary.flipped_files}")
    print(f"Already target: {summary.already_target_files}")
    print(f"Midline files: {summary.midline_files}")
    print(f"Failed/skipped: {summary.failed_files}")
    print(f"Total node rows written: {summary.rows_written}")
    print(f"Output path: {output_path}")
    if summary.region_profile_path:
        print(f"Regional profile: {summary.region_profile_path}")


def main(args: list[str] | None = None) -> int:
    """Run the batch SWC-to-Parquet conversion CLI."""
    parsed = parse_args(args)

    if not parsed.input_path.exists():
        print(f"Error: Input path not found: {parsed.input_path}", file=sys.stderr)
        return 1

    try:
        summary = batch_convert_swc_to_parquet(
            parsed.input_path,
            parsed.output_parquet,
            recursive=parsed.recursive,
            hemisphere=parsed.hemisphere,
            atlas_name=parsed.atlas,
            coord_axis=parsed.coord_axis,
            midline=parsed.midline,
            annotate_regions=(
                parsed.annotate_regions or parsed.build_regional_profile
            ),
            resolution=parsed.resolution,
            cache_dir=parsed.cache_dir,
            batch_size=parsed.batch_size,
            n_workers=parsed.workers,
            temp_dir=parsed.temp_dir,
            build_regional_profile=parsed.build_regional_profile,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary, parsed.output_parquet)

    if summary.failures:
        for swc_path, message in summary.failures:
            print(f"Skipped {Path(swc_path).name}: {message}", file=sys.stderr)

    if summary.discovered_files == 0:
        print(f"Error: No SWC files found in {parsed.input_path}", file=sys.stderr)
        return 1

    if summary.processed_files == 0:
        print("Error: No SWC files were successfully processed", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

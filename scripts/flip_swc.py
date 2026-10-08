#!/usr/bin/env python
"""CLI tool to detect hemisphere and flip SWC file coordinates.

This script reads an SWC file, determines which hemisphere the soma is in
using a BrainGlobe atlas, and produces a flipped version of the SWC file
with coordinates mirrored across the atlas midline.

Example usage:
    python scripts/flip_swc.py input.swc -o output_flipped.swc
    python scripts/flip_swc.py input.swc --atlas allen_mouse_10um
    python scripts/flip_swc.py input.swc --info-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brainglobe_atlasapi import BrainGlobeAtlas

from napari_neuron_navigator import (
    detect_soma_hemisphere,
    flip_swc,
    get_atlas_midline,
    parse_swc,
    write_swc,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect hemisphere and flip SWC file coordinates using BrainGlobe atlas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.swc -o flipped.swc
      Flip input.swc and save to flipped.swc

  %(prog)s input.swc --atlas allen_mouse_10um
      Use 10um resolution Allen Mouse Brain Atlas

  %(prog)s input.swc --info-only
      Only report hemisphere, don't create flipped file

  %(prog)s input.swc --coord-axis 2
      Flip along Z axis instead of X axis
""",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input SWC file path",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output SWC file path. If not specified, appends '_flipped' to input filename.",
    )

    parser.add_argument(
        "-a",
        "--atlas",
        type=str,
        default="allen_mouse_10um",
        help="BrainGlobe atlas name (default: allen_mouse_10um). "
        "Common options: allen_mouse_10um, allen_mouse_25um, allen_mouse_50um",
    )

    parser.add_argument(
        "--coord-axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Coordinate axis for left-right (0=X, 1=Y, 2=Z). Default: 2 (Z axis)",
    )

    parser.add_argument(
        "--midline",
        type=float,
        default=None,
        help="Override atlas midline with custom value (in microns)",
    )

    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only report hemisphere information, don't create flipped file",
    )

    parser.add_argument(
        "--list-atlases",
        action="store_true",
        help="List available BrainGlobe atlases and exit",
    )

    return parser.parse_args(args)


def list_available_atlases() -> None:
    """Print list of available BrainGlobe atlases."""
    from brainglobe_atlasapi import show_atlases

    print("Available BrainGlobe atlases:")
    print("-" * 40)
    show_atlases()


def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parsed = parse_args(args)

    # Handle --list-atlases
    if parsed.list_atlases:
        list_available_atlases()
        return 0

    # Validate input file
    input_path: Path = parsed.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not input_path.suffix.lower() == ".swc":
        print(f"Warning: Input file does not have .swc extension: {input_path}")

    # Determine output path
    if parsed.output is None:
        output_path = input_path.with_stem(f"{input_path.stem}_flipped")
    else:
        output_path = parsed.output

    # Load atlas
    print(f"Loading atlas: {parsed.atlas}")
    try:
        atlas = BrainGlobeAtlas(parsed.atlas)
    except Exception as e:
        print(f"Error loading atlas '{parsed.atlas}': {e}", file=sys.stderr)
        print("Use --list-atlases to see available atlases.", file=sys.stderr)
        return 1

    # Get midline
    if parsed.midline is not None:
        midline = parsed.midline
        print(f"Using custom midline: {midline:.2f} um")
    else:
        midline = get_atlas_midline(atlas)
        print(f"Atlas midline: {midline:.2f} um")

    # Parse SWC file
    print(f"Reading SWC file: {input_path}")
    try:
        swc_data = parse_swc(input_path)
    except Exception as e:
        print(f"Error parsing SWC file: {e}", file=sys.stderr)
        return 1

    print(f"  Nodes: {swc_data.n_nodes}")
    soma_coords = swc_data.soma_coords
    if len(soma_coords) > 0:
        print(f"  Soma nodes: {len(soma_coords)}")
        print(f"  Soma center: ({soma_coords[:, 0].mean():.2f}, "
              f"{soma_coords[:, 1].mean():.2f}, {soma_coords[:, 2].mean():.2f})")

    # Detect hemisphere
    try:
        hemisphere = detect_soma_hemisphere(
            swc_data,
            atlas=atlas,
            midline=midline,
            coord_axis=parsed.coord_axis,
        )
        print(f"  Soma hemisphere: {hemisphere.value.upper()}")
    except ValueError as e:
        print(f"Warning: {e}", file=sys.stderr)
        hemisphere = None

    # Exit early if info-only
    if parsed.info_only:
        print("\n--info-only specified, skipping flip operation.")
        return 0

    # Flip coordinates
    print(f"\nFlipping coordinates across midline (axis={parsed.coord_axis})...")
    flipped_swc = flip_swc(
        swc_data,
        atlas=atlas,
        midline=midline,
        coord_axis=parsed.coord_axis,
    )

    # Report new hemisphere
    if hemisphere is not None:
        new_hemisphere = detect_soma_hemisphere(
            flipped_swc,
            atlas=atlas,
            midline=midline,
            coord_axis=parsed.coord_axis,
        )
        print(f"  New soma hemisphere: {new_hemisphere.value.upper()}")

    # Write output
    print(f"\nWriting flipped SWC file: {output_path}")
    try:
        write_swc(flipped_swc, output_path)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return 1

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

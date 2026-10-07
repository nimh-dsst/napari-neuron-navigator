"""Display-only color limits for completed, symmetric distance matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ClustermapContrast:
    """Resolved limits shared verbatim by the preview and image export."""

    minimum: float
    maximum: float
    mode: Literal["auto", "full", "manual"] = "manual"

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.minimum)
            or not np.isfinite(self.maximum)
            or self.minimum >= self.maximum
        ):
            raise ValueError("Color limits must be finite, with minimum below maximum.")
        if self.mode not in {"auto", "full", "manual"}:
            raise ValueError(f"Unknown clustermap contrast mode: {self.mode}")

    @property
    def label(self) -> str:
        return {
            "auto": "Auto contrast",
            "full": "Full range",
            "manual": "Manual contrast",
        }[self.mode]

    def format_value(self, value: float) -> str:
        """Keep even very close manual limits and colorbar ticks distinct."""
        precision = 10
        magnitude = max(abs(self.minimum), abs(self.maximum))
        span = self.maximum - self.minimum
        if magnitude > 0 and np.isfinite(span):
            precision = max(
                precision,
                min(17, int(np.ceil(np.log10(magnitude) - np.log10(span))) + 2),
            )
        return format(value, f".{precision}g")


def _nonzero_range(lower: float, upper: float) -> tuple[float, float]:
    """Pad a constant range without suggesting variation in the matrix."""
    if lower == upper:
        padding = max(abs(lower), 1.0) * 0.01
        return lower - padding, upper + padding
    return lower, upper


@dataclass(frozen=True)
class ClustermapContrastStatistics:
    """Small cached summary; no extra matrix is retained."""

    full_minimum: float
    full_maximum: float
    auto_minimum: float
    auto_maximum: float
    message: str = ""
    percentile_fallback: bool = False

    def explanation(self, contrast: ClustermapContrast) -> str:
        if self.message:
            return self.message
        if self.percentile_fallback and contrast.mode == "auto":
            return "Percentile limits coincide; using the off-diagonal range."
        return ""

    def resolve(self, mode: Literal["auto", "full"] = "auto") -> ClustermapContrast:
        if mode == "auto":
            limits = (self.auto_minimum, self.auto_maximum)
        elif mode == "full":
            limits = (self.full_minimum, self.full_maximum)
        else:
            raise ValueError(f"Unknown automatic contrast mode: {mode}")
        return ClustermapContrast(*_nonzero_range(*limits), mode=mode)

    def extension(self, contrast: ClustermapContrast) -> str:
        """Colorbar extensions reflect the full matrix, including its diagonal."""
        below = self.full_minimum < contrast.minimum
        above = self.full_maximum > contrast.maximum
        return (
            "both"
            if below and above
            else "min"
            if below
            else "max"
            if above
            else "neither"
        )


def compute_clustermap_contrast_statistics(
    matrix: np.ndarray,
) -> ClustermapContrastStatistics:
    """Use each finite off-diagonal pair once; keep self-distances out of quantiles.

    Completed clustering matrices are symmetric. Copy their upper triangle
    without allocating two quadratic integer-index arrays, and discard it
    after computing the summary. Percentile partitioning never mutates the
    original matrix.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Clustermap distances must be a square matrix.")
    size = matrix.shape[0]
    pairs = (
        np.concatenate([matrix[index, index + 1 :] for index in range(size - 1)])
        if size > 1
        else np.array([], dtype=float)
    )
    pairs = pairs[np.isfinite(pairs)]
    diagonal = np.diag(matrix)
    diagonal = diagonal[np.isfinite(diagonal)]
    message = ""
    percentile_fallback = False
    if pairs.size:
        pair_minimum, pair_maximum = float(pairs.min()), float(pairs.max())
        lower, upper = map(float, np.percentile(pairs, [1, 99], overwrite_input=True))
        if lower == upper:
            lower, upper = pair_minimum, pair_maximum
            percentile_fallback = True
        if pair_minimum == pair_maximum:
            message = f"All finite off-diagonal distances equal {pair_minimum:.10g}; no contrast to stretch."
        full_minimum, full_maximum = pair_minimum, pair_maximum
        if diagonal.size:
            full_minimum = min(full_minimum, float(diagonal.min()))
            full_maximum = max(full_maximum, float(diagonal.max()))
    else:
        message = "No finite off-diagonal distances to stretch."
        full_minimum, full_maximum = (
            (float(diagonal.min()), float(diagonal.max()))
            if diagonal.size
            else (0.0, 1.0)
        )
        lower, upper = full_minimum, full_maximum
    return ClustermapContrastStatistics(
        full_minimum, full_maximum, lower, upper, message, percentile_fallback
    )

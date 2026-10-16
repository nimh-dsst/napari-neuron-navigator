"""Shared CCF Pearson policy provenance and user-facing descriptions."""

from collections.abc import Mapping

CORRECTED_PEARSON_LABEL = (
    "Nonoverlapping neurons use zero cross-product; zero-variance neurons excluded"
)
CORRECTED_PEARSON_TOOLTIP = (
    "Checked: zero cross-product for nonoverlapping neurons; exclude zero-variance neurons.\n"
    "Unchecked: assign r = -1 to nonoverlapping pairs or pairs containing a zero-variance neuron."
)


def ccf_pearson_policy_metadata(use_corrected_pearson: bool) -> dict[str, object]:
    """Describe the effective policy independently of any live UI state."""
    return {
        "use_corrected_pearson": bool(use_corrected_pearson),
        "implementation": (
            "ccf_pearson_complete_pairs_v2"
            if use_corrected_pearson
            else "ccf_pearson_legacy_v1"
        ),
        "no_overlap_policy": (
            "zero_cross_product" if use_corrected_pearson else "pearson_r_minus_one"
        ),
        "zero_variance_policy": (
            "unclustered" if use_corrected_pearson else "pearson_r_minus_one"
        ),
    }


def ccf_pearson_mode_text(metadata: Mapping[str, object]) -> str:
    """Label a recorded CCF policy; absent or unrecognized provenance stays unlabeled."""
    corrected = metadata.get("use_corrected_pearson")
    if corrected is True:
        return "CCF Pearson: corrected."
    if corrected is False:
        return "CCF Pearson: legacy."
    # Older results can explicitly record a policy without the new boolean.
    implementation = metadata.get("implementation")
    policy = metadata.get("missing_correlation_policy")
    if (
        implementation == "ccf_pearson_complete_pairs_v2"
        or policy == "zero_cross_product_omit_zero_variance"
    ):
        return "CCF Pearson: corrected."
    if implementation == "ccf_pearson_legacy_v1" or policy == "pearson_r_minus_one":
        return "CCF Pearson: legacy."
    return ""

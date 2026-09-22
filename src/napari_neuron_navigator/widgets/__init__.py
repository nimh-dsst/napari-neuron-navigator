"""napari widgets for Neuron Navigator."""

import logging

from ..logging_utils import configure_debug_logging, startup_timing

configure_debug_logging()
logger = logging.getLogger(__name__)

with startup_timing(logger, "widget_package_import", module="reference_layers"):
    from .reference_layers import (
        add_allen_template,
        add_region_id_segmentation,
        add_region_mesh,
        add_region_mesh_group,
        add_region_segmentation,
    )

with startup_timing(logger, "widget_package_import", module="region_selector"):
    from .region_selector import RegionSelectorWidget

with startup_timing(logger, "widget_package_import", module="custom_region_selector"):
    from .custom_region_selector import CustomRegionSelectorWidget

with startup_timing(logger, "widget_package_import", module="slice_projection"):
    from .slice_projection import NeuronSliceProjector, SomaSliceProjector

with startup_timing(logger, "widget_package_import", module="neuron_table"):
    from .neuron_table import NeuronTableWidget

with startup_timing(logger, "widget_package_import", module="analysis_tab"):
    from .analysis_tab import AnalysisTabWidget

with startup_timing(logger, "widget_package_import", module="search_tab"):
    from .search_tab import SearchTabWidget

with startup_timing(logger, "widget_package_import", module="neuron_viewer"):
    from .neuron_viewer import NeuronViewerWidget

__all__ = [
    "AnalysisTabWidget",
    "CustomRegionSelectorWidget",
    "NeuronSliceProjector",
    "NeuronTableWidget",
    "NeuronViewerWidget",
    "RegionSelectorWidget",
    "SearchTabWidget",
    "SomaSliceProjector",
    "add_allen_template",
    "add_region_id_segmentation",
    "add_region_mesh",
    "add_region_mesh_group",
    "add_region_segmentation",
]

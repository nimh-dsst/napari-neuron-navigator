"""Register loaded TIFF volumes for mask queries without another volume copy."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import tifffile

from tests.test_mask_query_widget import _wait
from tests.test_mask_query_widget import mask_owner as mask_owner  # noqa: PLC0414

# napari/npe2's Pydantic-v1 compatibility declarations warn during layer imports.
pytestmark = [
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning:tifffile.*"
    ),
]


@pytest.fixture
def import_owner(mask_owner):
    from napari.components import LayerList
    from qtpy.QtWidgets import QPushButton

    from napari_neuron_navigator.widgets.mask_layer_selector import (
        MaskLayerSelectorWidget,
    )

    app, owner, started, release, ticks, _old_layer = mask_owner
    owner._atlas.atlas_name = "fixture"
    owner.viewer = SimpleNamespace(layers=LayerList())
    owner._use_layer_as_mask_btn = QPushButton("Use Active Layer as Mask", owner)
    owner._use_layer_as_mask_btn.clicked.connect(owner._use_active_layer_as_mask)
    owner._mask_layer_selector = MaskLayerSelectorWidget(owner)
    owner._mask_layer_selector.selection_changed.connect(
        owner._on_mask_layer_selection_changed
    )
    owner._selected_mask_query_layers = MethodType(
        type(owner)._selected_mask_query_layers, owner
    )
    from qtpy.QtWidgets import QLabel

    owner._mask_query_hint_label = QLabel(owner)
    return app, owner, started, release, ticks


@pytest.mark.parametrize("kind", ["image", "labels"])
def test_loaded_tif_registers_and_queries_as_mask(import_owner, tmp_path, kind):
    from napari.layers import Image, Labels

    app, owner, _started, release, _ticks = import_owner
    data = np.zeros((3, 3, 3), dtype=np.float32 if kind == "image" else np.uint32)
    data[1, 1, 1] = 7
    path = tmp_path / "external.tif"
    tifffile.imwrite(path, data, photometric="minisblack")
    layer = (
        Image(tifffile.imread(path), rgb=False, name="External mask")
        if kind == "image"
        else Labels(tifffile.imread(path), name="External mask")
    )
    owner.viewer.layers.append(layer)
    owner.viewer.layers.selection.active = layer
    original = layer.data
    owner._use_layer_as_mask_btn.click()
    assert layer.data is original
    assert len(owner.viewer.layers) == 1
    assert layer.metadata["mask_query_source"]
    assert layer.metadata["mask_imported"]
    assert layer.metadata["atlas_name"] == "fixture"
    assert "source_file_ids" not in layer.metadata
    assert owner._mask_layer_selector.get_selected_layer_names() == [layer.name]
    assert (
        "source-neuron exclusion cannot be applied"
        in owner._mask_query_hint_label.text()
    )
    assert "Registered" in owner._regions_status_label.text()
    owner._query_neurons_by_mask()
    release.set()
    _wait(app, lambda: owner._regional_query_thread is None)
    assert owner._populate_neuron_table.call_args.args[0].file_id.tolist() == [
        "a/shared",
        "b/shared",
    ]
    # Registration is repeatable and does not duplicate the mask or its data.
    owner._use_layer_as_mask_btn.click()
    assert owner._mask_layer_selector.get_selected_layer_names() == [layer.name]
    assert layer.data is original


@pytest.mark.parametrize(
    "case",
    [
        "no_atlas",
        "no_layer",
        "shape",
        "2d",
        "4d",
        "scale",
        "translate",
        "atlas",
        "points",
    ],
)
def test_incompatible_import_does_not_mark_layer(import_owner, case):
    from napari.layers import Image, Points

    _app, owner, _started, _release, _ticks = import_owner
    layer = Image(np.zeros((2, 3, 3) if case == "shape" else (3, 3, 3)), rgb=False)
    if case == "2d":
        layer = Image(np.zeros((3, 3)), rgb=False)
    elif case == "4d":
        layer = Image(np.zeros((2, 3, 3, 3)), rgb=False)
    elif case == "points":
        layer = Points(np.array([[0, 0, 0]]))
    elif case == "scale":
        layer.scale = (2, 1, 1)
    elif case == "translate":
        layer.translate = (1, 0, 0)
    elif case == "atlas":
        layer.metadata["atlas_name"] = "different"
    owner.viewer.layers.append(layer)
    owner.viewer.layers.selection.active = layer
    if case == "no_atlas":
        owner._atlas = None
    elif case == "no_layer":
        owner.viewer.layers.selection.clear()
    owner._use_layer_as_mask_btn.click()
    assert not layer.metadata.get("mask_query_source")
    assert "Could not register mask" in owner._regions_status_label.text()


def test_existing_generated_mask_keeps_its_provenance(import_owner):
    from napari.layers import Labels

    _app, owner, _started, _release, _ticks = import_owner
    metadata = {
        "mask_query_source": True,
        "atlas_name": "fixture",
        "source_file_ids": ["a/shared"],
        "sigma": 1.0,
        "lower_threshold": 0.25,
        "neuron_set": {"set_id": "original"},
    }
    layer = Labels(np.zeros((3, 3, 3), dtype=np.uint8), metadata=dict(metadata))
    owner.viewer.layers.append(layer)
    owner.viewer.layers.selection.active = layer
    owner._use_layer_as_mask_btn.click()
    assert layer.metadata == metadata


def test_registered_float_image_persists_as_an_image_mask(import_owner, tmp_path):
    from napari.layers import Image

    from napari_neuron_navigator.project_io import (
        load_project_bundle,
        save_project_bundle,
    )

    _app, owner, _started, _release, _ticks = import_owner
    data = np.zeros((3, 3, 3), dtype=np.float32)
    data[1, 1, 1] = 0.5
    layer = Image(data, rgb=False, name="Float mask")
    owner.viewer.layers.append(layer)
    owner.viewer.layers.selection.active = layer
    owner._use_layer_as_mask_btn.click()
    path = tmp_path / "imported.nnproj"
    save_project_bundle(
        path,
        source_parquet_path=owner._db.parquet_path,
        table_state={"entries": [{"file_id": "a/shared"}]},
        layers=[layer],
        atlas_name="fixture",
    )
    bundle = load_project_bundle(path)
    assert bundle.layers[0].metadata["layer_type"] == "image"
    assert bundle.layers[0].metadata["metadata"]["mask_query_source"]
    assert bundle.layers[0].metadata["metadata"]["mask_imported"]
    assert bundle.layers[0].data.dtype == data.dtype
    np.testing.assert_array_equal(bundle.layers[0].data, data)

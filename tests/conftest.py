"""Shared Qt fixtures that keep widget interaction tests independent of OpenGL."""

import pytest


@pytest.fixture(scope="module")
def qt_app():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def viewer_model(qt_app):
    """Use real napari layers and events without constructing a VisPy canvas."""
    from napari.components import ViewerModel

    viewer = ViewerModel()
    yield viewer
    viewer.layers.clear()
    qt_app.processEvents()

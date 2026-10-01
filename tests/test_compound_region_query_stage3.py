"""Stage-3 structured regional-query editor model tests.

Qt is stubbed because these tests exercise immutable editor values and AST
conversion; napari UI behavior is covered by the Regions integration tests.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from napari_neuron_navigator.analysis.region_profile import Compartment
from napari_neuron_navigator.analysis.region_query import (
    AllOf,
    AnyOf,
    NeuriteIntersects,
    Not,
    QueryLaterality,
    RegionalMeasurement,
    RegionalThreshold,
    RegionQueryValidationError,
    RegionSelection,
    SomaIn,
    ThresholdComparison,
)


class _Signal:
    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _QtObject:
    def __init__(self, *_args, **_kwargs) -> None:
        return None


def _import_editor_module():
    qtcore = types.ModuleType("qtpy.QtCore")
    qtcore.Signal = _Signal
    qtwidgets = types.ModuleType("qtpy.QtWidgets")
    for name in (
        "QCheckBox",
        "QComboBox",
        "QDialog",
        "QDialogButtonBox",
        "QDoubleSpinBox",
        "QFrame",
        "QGridLayout",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QProgressBar",
        "QPushButton",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(qtwidgets, name, _QtObject)

    qtpy = types.ModuleType("qtpy")
    qtpy.QtCore = qtcore
    qtpy.QtWidgets = qtwidgets

    widgets_package = types.ModuleType("napari_neuron_navigator.widgets")
    widgets_package.__path__ = [
        str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "napari_neuron_navigator"
            / "widgets"
        )
    ]
    region_selector = types.ModuleType(
        "napari_neuron_navigator.widgets.region_selector"
    )
    region_selector.RegionSelectorWidget = _QtObject

    replacements = {
        "qtpy": qtpy,
        "qtpy.QtCore": qtcore,
        "qtpy.QtWidgets": qtwidgets,
        "napari_neuron_navigator.widgets": widgets_package,
        "napari_neuron_navigator.widgets.region_selector": region_selector,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    module_name = "napari_neuron_navigator.widgets.compound_region_query_test_module"
    previous_module = sys.modules.get(module_name)
    try:
        sys.modules.update(replacements)
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "napari_neuron_navigator"
            / "widgets"
            / "compound_region_query.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture
def editor():
    return _import_editor_module()


@pytest.fixture
def structures() -> dict[int, dict[str, object]]:
    return {
        10: {"id": 10, "acronym": "MO", "name": "Motor areas"},
        11: {"id": 11, "acronym": "MOp5", "name": "Primary motor layer 5"},
        12: {"id": 12, "acronym": "MOs5", "name": "Secondary motor layer 5"},
        30: {"id": 30, "acronym": "CP", "name": "Caudoputamen"},
        40: {"id": 40, "acronym": "pons", "name": "Pons"},
    }


def _selection(
    ids: list[int],
    structures: dict[int, dict[str, object]],
    *,
    descendants: bool = False,
) -> RegionSelection:
    return RegionSelection.from_ids(
        ids,
        structures=structures,
        include_descendants=descendants,
    )


def test_structured_values_build_both_motivating_queries(editor, structures):
    motor = _selection([11, 12], structures)
    cp = _selection([30], structures, descendants=True)
    pons = _selection([40], structures, descendants=True)

    contralateral = editor.StructuredRegionGroup(
        children=(
            editor.StructuredRegionCondition(
                subject=editor.SUBJECT_SOMA,
                condition=editor.CONDITION_IN_REGION,
                region_selection=motor,
            ),
            editor.StructuredRegionCondition(
                subject=editor.SUBJECT_NEURITE,
                condition=editor.CONDITION_INTERSECTS,
                laterality=QueryLaterality.CONTRALATERAL,
                region_selection=cp,
            ),
        )
    )
    ipsilateral = editor.StructuredRegionGroup(
        children=(
            contralateral.children[0],
            editor.StructuredRegionCondition(
                subject=editor.SUBJECT_NEURITE,
                condition=editor.CONDITION_INTERSECTS,
                laterality=QueryLaterality.IPSILATERAL,
                region_selection=pons,
            ),
        )
    )

    first = editor.structured_query_from_value(contralateral)
    second = editor.structured_query_from_value(ipsilateral)

    assert isinstance(first, AllOf)
    assert isinstance(first.children[0], SomaIn)
    assert isinstance(first.children[1], NeuriteIntersects)
    assert first.children[1].laterality is QueryLaterality.CONTRALATERAL
    assert isinstance(second, AllOf)
    assert second.children[1].laterality is QueryLaterality.IPSILATERAL
    assert editor.format_structured_query(first) == (
        "(SOMA IN ANY(MOp5, MOs5) AND "
        "NEURITE INTERSECTS CONTRALATERAL CP + descendants)"
    )


def test_nested_groups_thresholds_and_not_convert_without_losing_semantics(
    editor,
    structures,
):
    motor = _selection([10], structures, descendants=True)
    cp = _selection([30], structures, descendants=True)
    value = editor.StructuredRegionGroup(
        operator=editor.GROUP_AND,
        children=(
            editor.StructuredRegionCondition(
                region_selection=motor,
                negated=True,
            ),
            editor.StructuredRegionGroup(
                operator=editor.GROUP_OR,
                children=(
                    editor.StructuredRegionCondition(
                        subject=editor.SUBJECT_NEURITE,
                        condition=editor.CONDITION_CABLE_LENGTH,
                        region_selection=cp,
                        laterality=QueryLaterality.IPSILATERAL,
                        comparison=ThresholdComparison.GREATER_THAN_OR_EQUAL,
                        value=125.0,
                    ),
                    editor.StructuredRegionCondition(
                        subject=editor.SUBJECT_NEURITE,
                        condition=editor.CONDITION_TERMINUS_COUNT,
                        region_selection=cp,
                        comparison=ThresholdComparison.GREATER_THAN,
                        value=0,
                    ),
                ),
            ),
        ),
    )

    query = editor.structured_query_from_value(value)

    assert isinstance(query, AllOf)
    assert isinstance(query.children[0], Not)
    assert isinstance(query.children[1], AnyOf)
    length = query.children[1].children[0]
    assert isinstance(length, RegionalThreshold)
    assert length.compartment is Compartment.NEURITE
    assert length.measurement is RegionalMeasurement.CABLE_LENGTH_UM
    assert length.laterality is QueryLaterality.IPSILATERAL
    assert length.value == 125.0


def test_subject_options_expose_only_applicable_controls(editor):
    assert editor.structured_condition_options(editor.SUBJECT_SOMA) == (
        editor.CONDITION_IN_REGION,
        editor.CONDITION_NODE_COUNT,
    )
    assert editor.structured_condition_options(editor.SUBJECT_NEURITE) == (
        editor.CONDITION_INTERSECTS,
        editor.CONDITION_CABLE_LENGTH,
        editor.CONDITION_NODE_COUNT,
        editor.CONDITION_TERMINUS_COUNT,
    )
    with pytest.raises(RegionQueryValidationError, match="subject"):
        editor.structured_condition_options("axon")
    with pytest.raises(RegionQueryValidationError, match="not valid"):
        editor.StructuredRegionCondition(
            subject=editor.SUBJECT_SOMA,
            condition=editor.CONDITION_TERMINUS_COUNT,
        )


def test_independent_clause_selections_survive_remap_and_foreign_ids_clear(
    editor,
    structures,
):
    first = _selection([11], structures)
    second = _selection([30], structures, descendants=True)
    value = editor.StructuredRegionGroup(
        children=(
            editor.StructuredRegionCondition(region_selection=first),
            editor.StructuredRegionCondition(
                subject=editor.SUBJECT_NEURITE,
                condition=editor.CONDITION_INTERSECTS,
                region_selection=second,
            ),
        )
    )
    renamed = {key: dict(item) for key, item in structures.items()}
    renamed[11]["acronym"] = "MOp5-new"
    remapped, invalidated = editor.remap_structured_query_regions(value, renamed)

    assert invalidated == 0
    assert remapped.children[0].region_selection.regions[0].acronym == "MOp5-new"
    assert remapped.children[1].region_selection.region_ids == (30,)
    assert remapped.children[1].region_selection.include_descendants is True

    incompatible, invalidated = editor.remap_structured_query_regions(
        value,
        {11: structures[11]},
    )
    assert invalidated == 1
    assert incompatible.children[0].region_selection is not None
    assert incompatible.children[1].region_selection is None


def test_incomplete_or_empty_editor_values_fail_before_execution(editor):
    with pytest.raises(RegionQueryValidationError, match="select at least one"):
        editor.structured_query_from_value(
            editor.StructuredRegionGroup(children=(editor.StructuredRegionCondition(),))
        )
    with pytest.raises(RegionQueryValidationError, match="at least one condition"):
        editor.structured_query_from_value(editor.StructuredRegionGroup())

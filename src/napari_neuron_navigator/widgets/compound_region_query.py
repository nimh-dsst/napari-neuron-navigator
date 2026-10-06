"""Structured editor for Boolean regional-profile queries.

The value objects in this module are independent of widget state.  Every
successful editor snapshot is converted to the immutable query AST from
``analysis.region_query`` before it can leave the widget.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TypeAlias

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..analysis.region_profile import Compartment
from ..analysis.region_query import (
    AllOf,
    AnyOf,
    NeuriteIntersects,
    Not,
    QueryLaterality,
    RegionalMeasurement,
    RegionalThreshold,
    RegionQuery,
    RegionQueryValidationError,
    RegionSelection,
    SomaIn,
    ThresholdComparison,
)
from .region_selector import RegionSelectorWidget

SUBJECT_SOMA = "soma"
SUBJECT_NEURITE = "neurite"

CONDITION_IN_REGION = "in_region"
CONDITION_INTERSECTS = "intersects"
CONDITION_CABLE_LENGTH = RegionalMeasurement.CABLE_LENGTH_UM.value
CONDITION_NODE_COUNT = RegionalMeasurement.NODE_COUNT.value
CONDITION_TERMINUS_COUNT = RegionalMeasurement.TERMINUS_COUNT.value

GROUP_AND = "and"
GROUP_OR = "or"

_SUBJECT_LABELS = {
    SUBJECT_SOMA: "Soma",
    SUBJECT_NEURITE: "Projection (all non-soma neurites)",
}
_CONDITION_LABELS = {
    CONDITION_IN_REGION: "In region",
    CONDITION_INTERSECTS: "Intersects",
    CONDITION_CABLE_LENGTH: "Cable length",
    CONDITION_NODE_COUNT: "Node count",
    CONDITION_TERMINUS_COUNT: "Termini",
}
_SIDE_LABELS = {
    QueryLaterality.EITHER: "Either",
    QueryLaterality.IPSILATERAL: "Ipsilateral",
    QueryLaterality.CONTRALATERAL: "Contralateral",
}
_COMPARISON_LABELS = {
    ThresholdComparison.GREATER_THAN: ">",
    ThresholdComparison.GREATER_THAN_OR_EQUAL: ">=",
    ThresholdComparison.LESS_THAN: "<",
    ThresholdComparison.LESS_THAN_OR_EQUAL: "<=",
    ThresholdComparison.EQUAL: "=",
}


def _conditions_for_subject(subject: str) -> tuple[str, ...]:
    if subject == SUBJECT_SOMA:
        return (
            CONDITION_IN_REGION,
            CONDITION_NODE_COUNT,
        )
    return (
        CONDITION_INTERSECTS,
        CONDITION_CABLE_LENGTH,
        CONDITION_NODE_COUNT,
        CONDITION_TERMINUS_COUNT,
    )


def structured_condition_options(subject: str) -> tuple[str, ...]:
    """Return condition keys applicable to a structured-query subject."""
    normalized = str(subject)
    if normalized not in _SUBJECT_LABELS:
        raise RegionQueryValidationError(
            f"unsupported structured-query subject: {normalized!r}"
        )
    return _conditions_for_subject(normalized)


@dataclass(frozen=True)
class StructuredRegionCondition:
    """Editable value for one regional-profile condition."""

    subject: str = SUBJECT_SOMA
    condition: str = CONDITION_IN_REGION
    region_selection: RegionSelection | None = None
    laterality: QueryLaterality = QueryLaterality.EITHER
    comparison: ThresholdComparison = ThresholdComparison.GREATER_THAN
    value: float = 0.0
    negated: bool = False

    def __post_init__(self) -> None:
        subject = str(self.subject)
        condition = str(self.condition)
        if subject not in _SUBJECT_LABELS:
            raise RegionQueryValidationError(
                f"unsupported structured-query subject: {subject!r}"
            )
        if condition not in _conditions_for_subject(subject):
            raise RegionQueryValidationError(
                f"condition {condition!r} is not valid for {subject}"
            )
        try:
            laterality = QueryLaterality(self.laterality)
            comparison = ThresholdComparison(self.comparison)
            value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise RegionQueryValidationError(
                "structured regional condition contains an unsupported value"
            ) from exc
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "laterality", laterality)
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "negated", bool(self.negated))


@dataclass(frozen=True)
class StructuredRegionGroup:
    """Editable Boolean group containing conditions or nested groups."""

    operator: str = GROUP_AND
    children: tuple[StructuredRegionNode, ...] = ()
    negated: bool = False

    def __post_init__(self) -> None:
        operator = str(self.operator)
        if operator not in {GROUP_AND, GROUP_OR}:
            raise RegionQueryValidationError(
                f"unsupported structured-query operator: {operator!r}"
            )
        children = tuple(self.children)
        if any(
            not isinstance(child, (StructuredRegionCondition, StructuredRegionGroup))
            for child in children
        ):
            raise RegionQueryValidationError(
                "structured-query groups may contain only conditions or groups"
            )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "negated", bool(self.negated))


StructuredRegionNode: TypeAlias = StructuredRegionCondition | StructuredRegionGroup


def structured_query_from_value(value: StructuredRegionNode) -> RegionQuery:
    """Convert a fully specified structured value to the typed query AST."""
    if isinstance(value, StructuredRegionCondition):
        selection = value.region_selection
        if selection is None:
            raise RegionQueryValidationError(
                "each compound-query condition must select at least one region"
            )
        if value.condition == CONDITION_IN_REGION:
            query: RegionQuery = SomaIn(selection)
        elif value.condition == CONDITION_INTERSECTS:
            query = NeuriteIntersects(selection, value.laterality)
        else:
            query = RegionalThreshold(
                compartment=(
                    Compartment.SOMA
                    if value.subject == SUBJECT_SOMA
                    else Compartment.NEURITE
                ),
                measurement=RegionalMeasurement(value.condition),
                comparison=value.comparison,
                value=value.value,
                region_selection=selection,
                laterality=(
                    QueryLaterality.EITHER
                    if value.subject == SUBJECT_SOMA
                    else value.laterality
                ),
            )
        return Not(query) if value.negated else query

    if not value.children:
        raise RegionQueryValidationError(
            "compound region query must contain at least one condition"
        )
    children = tuple(structured_query_from_value(child) for child in value.children)
    if len(children) == 1:
        query = children[0]
    elif value.operator == GROUP_AND:
        query = AllOf(children)
    else:
        query = AnyOf(children)
    return Not(query) if value.negated else query


def remap_structured_query_regions(
    value: StructuredRegionGroup,
    structures: Mapping[int, Mapping[str, object]],
) -> tuple[StructuredRegionGroup, int]:
    """Refresh selection labels and clear clauses foreign to an atlas."""
    normalized_structures = {
        int(region_id): structure for region_id, structure in structures.items()
    }
    invalidated = 0

    def remap(node: StructuredRegionNode) -> StructuredRegionNode:
        nonlocal invalidated
        if isinstance(node, StructuredRegionGroup):
            return replace(node, children=tuple(remap(c) for c in node.children))
        selection = node.region_selection
        if selection is None:
            return node
        if not set(selection.region_ids).issubset(normalized_structures):
            invalidated += 1
            return replace(node, region_selection=None)
        return replace(
            node,
            region_selection=RegionSelection.from_ids(
                selection.region_ids,
                structures=normalized_structures,
                include_descendants=selection.include_descendants,
            ),
        )

    remapped = remap(value)
    assert isinstance(remapped, StructuredRegionGroup)
    return remapped, invalidated


def _selection_text(selection: RegionSelection) -> str:
    labels = tuple(
        region.acronym or region.name or str(region.region_id)
        for region in selection.regions
    )
    text = labels[0] if len(labels) == 1 else f"ANY({', '.join(labels)})"
    if selection.include_descendants:
        text += " + descendants"
    return text


def format_structured_query(query: RegionQuery) -> str:
    """Return a stable, human-readable representation of a typed query."""
    if isinstance(query, AllOf):
        return (
            "(" + " AND ".join(format_structured_query(c) for c in query.children) + ")"
        )
    if isinstance(query, AnyOf):
        return (
            "(" + " OR ".join(format_structured_query(c) for c in query.children) + ")"
        )
    if isinstance(query, Not):
        return f"NOT ({format_structured_query(query.child)})"
    if isinstance(query, SomaIn):
        return f"SOMA IN {_selection_text(query.region_selection)}"
    if isinstance(query, NeuriteIntersects):
        side = (
            ""
            if query.laterality is QueryLaterality.EITHER
            else f"{query.laterality.value.upper()} "
        )
        return f"NEURITE INTERSECTS {side}{_selection_text(query.region_selection)}"
    if isinstance(query, RegionalThreshold):
        subject = "SOMA" if query.compartment is Compartment.SOMA else "NEURITE"
        measurement = {
            RegionalMeasurement.CABLE_LENGTH_UM: "CABLE LENGTH",
            RegionalMeasurement.NODE_COUNT: "NODE COUNT",
            RegionalMeasurement.TERMINUS_COUNT: "TERMINI",
        }[query.measurement]
        side = (
            ""
            if query.laterality is QueryLaterality.EITHER
            else f"{query.laterality.value.upper()} "
        )
        return (
            f"{subject} {measurement} {query.comparison.value} {query.value:g} "
            f"IN {side}{_selection_text(query.region_selection)}"
        )
    raise RegionQueryValidationError(
        f"unsupported regional-query node: {type(query).__name__}"
    )


class RegionPickerDialog(QDialog):
    """One reusable atlas hierarchy picker for every condition row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Regions for Condition")
        self.resize(620, 680)
        layout = QVBoxLayout(self)
        self._selector = RegionSelectorWidget()
        layout.addWidget(self._selector)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_atlas(self, atlas: object) -> None:
        self._selector.set_atlas(atlas)

    def set_selection(self, selection: RegionSelection | None) -> None:
        ids = () if selection is None else selection.region_ids
        self._selector.set_selected_ids(ids)
        self._selector.set_include_children_enabled(
            False if selection is None else selection.include_descendants
        )

    def selection(
        self, structures: Mapping[int, Mapping[str, object]]
    ) -> RegionSelection | None:
        region_ids = self._selector.get_selected_ids(include_children=False)
        if not region_ids:
            return None
        return RegionSelection.from_ids(
            region_ids,
            structures=structures,
            include_descendants=self._selector.include_children_enabled(),
        )


class _ConditionRow(QFrame):
    """Visible controls for one leaf condition."""

    def __init__(
        self,
        value: StructuredRegionCondition,
        *,
        on_change: Callable[[StructuredRegionCondition, bool], None],
        on_action: Callable[[str], None],
        combine_operator: str | None,
        on_combine: Callable[[str], None],
        custom_selection_available: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = value
        self._on_change = on_change
        self._on_action = on_action
        self.setFrameShape(QFrame.StyledPanel)
        layout = QGridLayout(self)

        layout.addWidget(QLabel("Subject:"), 0, 0)
        self._subject = QComboBox()
        for key, label in _SUBJECT_LABELS.items():
            self._subject.addItem(label, key)
        self._set_combo_data(self._subject, value.subject)
        layout.addWidget(self._subject, 0, 1, 1, 3)

        layout.addWidget(QLabel("Condition:"), 1, 0)
        self._condition = QComboBox()
        for key in _conditions_for_subject(value.subject):
            self._condition.addItem(_CONDITION_LABELS[key], key)
        self._set_combo_data(self._condition, value.condition)
        layout.addWidget(self._condition, 1, 1)

        self._side_label = QLabel("Side:")
        layout.addWidget(self._side_label, 1, 2)
        self._side = QComboBox()
        for key, label in _SIDE_LABELS.items():
            self._side.addItem(label, key.value)
        self._set_combo_data(self._side, value.laterality.value)
        layout.addWidget(self._side, 1, 3)

        layout.addWidget(QLabel("Regions:"), 2, 0)
        summary = (
            "Select one or more regions"
            if value.region_selection is None
            else _selection_text(value.region_selection)
        )
        self._regions = QLabel(summary)
        self._regions.setWordWrap(True)
        layout.addWidget(self._regions, 2, 1)
        edit = QPushButton("Edit Regions...")
        edit.clicked.connect(lambda: self._on_action("edit"))
        layout.addWidget(edit, 2, 2)
        self._custom = QPushButton("Use Custom Selection")
        self._custom.setVisible(custom_selection_available)
        self._custom.clicked.connect(lambda: self._on_action("custom"))
        layout.addWidget(self._custom, 2, 3)

        self._comparison_label = QLabel("Comparison:")
        layout.addWidget(self._comparison_label, 3, 0)
        self._comparison = QComboBox()
        for key, label in _COMPARISON_LABELS.items():
            self._comparison.addItem(label, key.value)
        self._set_combo_data(self._comparison, value.comparison.value)
        layout.addWidget(self._comparison, 3, 1)
        self._threshold_label = QLabel("Value:")
        layout.addWidget(self._threshold_label, 3, 2)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0, 1.0e12)
        self._threshold.setDecimals(3)
        self._threshold.setValue(value.value)
        layout.addWidget(self._threshold, 3, 3)

        actions = QHBoxLayout()
        if combine_operator is not None:
            actions.addWidget(QLabel("Combine with:"))
            combine = QComboBox()
            combine.addItem("AND", GROUP_AND)
            combine.addItem("OR", GROUP_OR)
            self._set_combo_data(combine, combine_operator)
            combine.currentIndexChanged.connect(
                lambda _index, combo=combine: on_combine(str(combo.currentData()))
            )
            actions.addWidget(combine)
        self._negated = QCheckBox("NOT")
        self._negated.setChecked(value.negated)
        actions.addWidget(self._negated)
        for label, action in (
            ("Add", "add"),
            ("Duplicate", "duplicate"),
            ("Group", "group"),
            ("Remove", "remove"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, name=action: self._on_action(name)
            )
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions, 4, 0, 1, 4)

        self._update_control_visibility()
        self._subject.currentIndexChanged.connect(self._subject_changed)
        self._condition.currentIndexChanged.connect(
            lambda _index: self._emit_value(rebuild=True)
        )
        self._side.currentIndexChanged.connect(
            lambda _index: self._emit_value(rebuild=False)
        )
        self._comparison.currentIndexChanged.connect(
            lambda _index: self._emit_value(rebuild=False)
        )
        self._threshold.valueChanged.connect(
            lambda _value: self._emit_value(rebuild=False)
        )
        self._negated.toggled.connect(lambda _checked: self._emit_value(rebuild=False))

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _subject_changed(self, _index: int) -> None:
        self._emit_value(rebuild=True)

    def _read_value(self) -> StructuredRegionCondition:
        subject = str(self._subject.currentData())
        condition = str(self._condition.currentData())
        if condition not in _conditions_for_subject(subject):
            condition = (
                CONDITION_IN_REGION if subject == SUBJECT_SOMA else CONDITION_INTERSECTS
            )
        return StructuredRegionCondition(
            subject=subject,
            condition=condition,
            region_selection=self._value.region_selection,
            laterality=(
                QueryLaterality.EITHER
                if subject == SUBJECT_SOMA
                else QueryLaterality(str(self._side.currentData()))
            ),
            comparison=ThresholdComparison(str(self._comparison.currentData())),
            value=self._threshold.value(),
            negated=self._negated.isChecked(),
        )

    def _emit_value(self, *, rebuild: bool) -> None:
        self._value = self._read_value()
        self._on_change(self._value, rebuild)

    def _update_control_visibility(self) -> None:
        side_visible = self._value.subject == SUBJECT_NEURITE
        self._side_label.setVisible(side_visible)
        self._side.setVisible(side_visible)
        threshold = self._value.condition not in {
            CONDITION_IN_REGION,
            CONDITION_INTERSECTS,
        }
        for widget in (
            self._comparison_label,
            self._comparison,
            self._threshold_label,
            self._threshold,
        ):
            widget.setVisible(threshold)


def _node_at(root: StructuredRegionNode, path: tuple[int, ...]) -> StructuredRegionNode:
    node = root
    for index in path:
        if not isinstance(node, StructuredRegionGroup):
            raise IndexError("structured-query path enters a condition")
        node = node.children[index]
    return node


def _replace_node(
    root: StructuredRegionNode,
    path: tuple[int, ...],
    replacement: StructuredRegionNode,
) -> StructuredRegionNode:
    if not path:
        return replacement
    if not isinstance(root, StructuredRegionGroup):
        raise IndexError("structured-query path enters a condition")
    index = path[0]
    children = list(root.children)
    children[index] = _replace_node(children[index], path[1:], replacement)
    return replace(root, children=tuple(children))


def _remove_node(
    root: StructuredRegionGroup,
    path: tuple[int, ...],
) -> StructuredRegionGroup:
    if not path:
        raise ValueError("the root query group cannot be removed")
    parent_path = path[:-1]
    parent = _node_at(root, parent_path)
    if not isinstance(parent, StructuredRegionGroup):
        raise IndexError("structured-query parent is not a group")
    children = list(parent.children)
    children.pop(path[-1])
    replacement = replace(parent, children=tuple(children))
    updated = _replace_node(root, parent_path, replacement)
    assert isinstance(updated, StructuredRegionGroup)
    return updated


class CompoundRegionQueryWidget(QWidget):
    """Nested structured query builder for the Regions tab."""

    query_changed = Signal(object)
    active_selection_changed = Signal(object)
    build_profile_requested = Signal()
    cancel_profile_requested = Signal()

    def __init__(
        self,
        *,
        custom_selection_provider: Callable[[], RegionSelection | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._atlas: object | None = None
        self._structures: Mapping[int, Mapping[str, object]] = {}
        self._picker = RegionPickerDialog(self)
        self._custom_selection_provider = custom_selection_provider
        self._value = StructuredRegionGroup(children=(StructuredRegionCondition(),))
        self._active_path: tuple[int, ...] = (0,)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Compound queries use Soma versus Projection (all non-soma neurites) "
            "from the regional profile. The raw Node types control does not apply."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self._editor = QWidget()
        self._editor_layout = QVBoxLayout(self._editor)
        self._editor_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._editor)

        layout.addWidget(QLabel("Canonical structured query:"))
        self._canonical = QLabel("")
        self._canonical.setWordWrap(True)
        self._canonical.setTextInteractionFlags(self._canonical.textInteractionFlags())
        layout.addWidget(self._canonical)

        self._editor_status = QLabel("")
        self._editor_status.setWordWrap(True)
        layout.addWidget(self._editor_status)

        profile_group = QGroupBox("Regional Profile")
        profile_layout = QVBoxLayout(profile_group)
        self._profile_status = QLabel(
            "Load a Parquet and compatible atlas to inspect its regional profile."
        )
        self._profile_status.setWordWrap(True)
        profile_layout.addWidget(self._profile_status)
        self._profile_progress = QProgressBar()
        self._profile_progress.setVisible(False)
        profile_layout.addWidget(self._profile_progress)
        profile_actions = QHBoxLayout()
        self._build_profile = QPushButton("Build Regional Profile")
        self._build_profile.clicked.connect(self.build_profile_requested)
        profile_actions.addWidget(self._build_profile)
        self._cancel_profile = QPushButton("Cancel Build")
        self._cancel_profile.clicked.connect(self.cancel_profile_requested)
        self._cancel_profile.setVisible(False)
        profile_actions.addWidget(self._cancel_profile)
        profile_actions.addStretch()
        profile_layout.addLayout(profile_actions)
        layout.addWidget(profile_group)

        self._rebuild()

    @property
    def value(self) -> StructuredRegionGroup:
        return self._value

    def set_value(self, value: StructuredRegionGroup) -> None:
        if not isinstance(value, StructuredRegionGroup):
            raise TypeError("compound query value must be a StructuredRegionGroup")
        self._value = value
        self._active_path = self._first_condition_path(value) or ()
        self._rebuild()

    def set_atlas(self, atlas: object | None) -> int:
        """Set the picker atlas and clear selections absent from the new atlas."""
        self._atlas = atlas
        structures = getattr(atlas, "structures", {}) if atlas is not None else {}
        self._structures = {
            int(region_id): structure for region_id, structure in structures.items()
        }
        if atlas is not None:
            self._picker.set_atlas(atlas)
        self._value, invalidated = remap_structured_query_regions(
            self._value,
            self._structures,
        )
        self._editor_status.setText(
            (
                f"Cleared {invalidated} condition selection(s) that are not "
                "available in the loaded atlas."
            )
            if invalidated
            else ""
        )
        self._rebuild()
        return invalidated

    def query(self) -> RegionQuery:
        if self._atlas is None:
            raise RegionQueryValidationError(
                "load a compatible Allen atlas before running a compound query"
            )
        return structured_query_from_value(self._value)

    def canonical_query(self) -> str:
        return format_structured_query(self.query())

    def active_region_selection(self) -> RegionSelection | None:
        try:
            node = _node_at(self._value, self._active_path)
        except (IndexError, TypeError):
            return None
        return (
            node.region_selection
            if isinstance(node, StructuredRegionCondition)
            else None
        )

    def set_profile_status(self, message: str, *, build_enabled: bool = True) -> None:
        self._profile_status.setText(str(message))
        self._build_profile.setEnabled(bool(build_enabled))

    def set_profile_progress(
        self,
        message: str,
        completed: int,
        total: int,
    ) -> None:
        self._profile_status.setText(str(message))
        self._profile_progress.setVisible(True)
        if total <= 0:
            self._profile_progress.setRange(0, 0)
        else:
            self._profile_progress.setRange(0, int(total))
            self._profile_progress.setValue(int(completed))

    def set_busy(self, busy: bool, *, building: bool = False) -> None:
        self._editor.setEnabled(not busy)
        self._build_profile.setEnabled(not busy)
        self._cancel_profile.setVisible(bool(busy and building))
        self._cancel_profile.setEnabled(bool(busy and building))
        if not busy:
            self._profile_progress.setVisible(False)

    @staticmethod
    def _first_condition_path(
        node: StructuredRegionNode,
        prefix: tuple[int, ...] = (),
    ) -> tuple[int, ...] | None:
        if isinstance(node, StructuredRegionCondition):
            return prefix
        for index, child in enumerate(node.children):
            path = CompoundRegionQueryWidget._first_condition_path(
                child, prefix + (index,)
            )
            if path is not None:
                return path
        return None

    def _clear_editor(self) -> None:
        while self._editor_layout.count():
            item = self._editor_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild(self) -> None:
        self._clear_editor()
        self._editor_layout.addWidget(self._build_group_widget(self._value, ()))
        self._update_canonical()
        self.active_selection_changed.emit(self.active_region_selection())

    def _build_group_widget(
        self,
        group: StructuredRegionGroup,
        path: tuple[int, ...],
    ) -> QWidget:
        box = QGroupBox("Compound Query" if not path else "Nested Group")
        layout = QVBoxLayout(box)
        header = QHBoxLayout()
        header.addWidget(QLabel("Combine conditions with:"))
        operator = QComboBox()
        operator.addItem("AND", GROUP_AND)
        operator.addItem("OR", GROUP_OR)
        operator.setCurrentIndex(0 if group.operator == GROUP_AND else 1)
        operator.currentIndexChanged.connect(
            lambda _index, p=path, combo=operator: self._set_group_value(
                p, operator=str(combo.currentData())
            )
        )
        header.addWidget(operator)
        negated = QCheckBox("NOT group")
        negated.setChecked(group.negated)
        negated.toggled.connect(
            lambda checked, p=path: self._set_group_value(p, negated=checked)
        )
        header.addWidget(negated)
        header.addStretch()
        if path:
            duplicate = QPushButton("Duplicate Group")
            duplicate.clicked.connect(lambda _checked=False, p=path: self._duplicate(p))
            header.addWidget(duplicate)
            remove = QPushButton("Remove Group")
            remove.clicked.connect(lambda _checked=False, p=path: self._remove(p))
            header.addWidget(remove)
        layout.addLayout(header)

        for index, child in enumerate(group.children):
            child_path = path + (index,)
            if index:
                join = QLabel("AND" if group.operator == GROUP_AND else "OR")
                layout.addWidget(join)
            if isinstance(child, StructuredRegionGroup):
                layout.addWidget(self._build_group_widget(child, child_path))
            else:
                layout.addWidget(
                    _ConditionRow(
                        child,
                        on_change=lambda value, rebuild, p=child_path: (
                            self._set_condition(p, value, rebuild=rebuild)
                        ),
                        on_action=lambda action, p=child_path: self._condition_action(
                            p, action
                        ),
                        combine_operator=(group.operator if index else None),
                        on_combine=lambda value, p=path: self._set_group_value(
                            p,
                            operator=value,
                        ),
                        custom_selection_available=(
                            self._custom_selection_provider is not None
                        ),
                    )
                )

        footer = QHBoxLayout()
        add_condition = QPushButton("Add Condition")
        add_condition.clicked.connect(
            lambda _checked=False, p=path: self._append_child(
                p, StructuredRegionCondition()
            )
        )
        footer.addWidget(add_condition)
        add_group = QPushButton("Add Group")
        add_group.clicked.connect(
            lambda _checked=False, p=path: self._append_child(
                p,
                StructuredRegionGroup(children=(StructuredRegionCondition(),)),
            )
        )
        footer.addWidget(add_group)
        footer.addStretch()
        layout.addLayout(footer)
        return box

    def _set_condition(
        self,
        path: tuple[int, ...],
        value: StructuredRegionCondition,
        *,
        rebuild: bool,
    ) -> None:
        updated = _replace_node(self._value, path, value)
        assert isinstance(updated, StructuredRegionGroup)
        self._value = updated
        self._active_path = path
        if rebuild:
            self._rebuild()
        else:
            self._update_canonical()
            self.active_selection_changed.emit(value.region_selection)

    def _set_group_value(self, path: tuple[int, ...], **changes: object) -> None:
        group = _node_at(self._value, path)
        if not isinstance(group, StructuredRegionGroup):
            raise TypeError("structured-query path is not a group")
        updated_group = replace(group, **changes)
        updated = _replace_node(self._value, path, updated_group)
        assert isinstance(updated, StructuredRegionGroup)
        self._value = updated
        self._rebuild()

    def _append_child(self, path: tuple[int, ...], child: StructuredRegionNode) -> None:
        group = _node_at(self._value, path)
        if not isinstance(group, StructuredRegionGroup):
            raise TypeError("structured-query path is not a group")
        updated_group = replace(group, children=(*group.children, child))
        updated = _replace_node(self._value, path, updated_group)
        assert isinstance(updated, StructuredRegionGroup)
        self._value = updated
        self._active_path = path + (len(group.children),)
        self._rebuild()

    def _insert_after(self, path: tuple[int, ...], child: StructuredRegionNode) -> None:
        parent_path = path[:-1]
        parent = _node_at(self._value, parent_path)
        if not isinstance(parent, StructuredRegionGroup):
            raise TypeError("structured-query parent is not a group")
        children = list(parent.children)
        children.insert(path[-1] + 1, child)
        updated_parent = replace(parent, children=tuple(children))
        updated = _replace_node(self._value, parent_path, updated_parent)
        assert isinstance(updated, StructuredRegionGroup)
        self._value = updated
        self._active_path = parent_path + (path[-1] + 1,)
        self._rebuild()

    def _duplicate(self, path: tuple[int, ...]) -> None:
        self._insert_after(path, _node_at(self._value, path))

    def _remove(self, path: tuple[int, ...]) -> None:
        self._value = _remove_node(self._value, path)
        self._active_path = self._first_condition_path(self._value) or ()
        self._rebuild()

    def _condition_action(self, path: tuple[int, ...], action: str) -> None:
        self._active_path = path
        node = _node_at(self._value, path)
        if not isinstance(node, StructuredRegionCondition):
            return
        if action == "edit":
            self._edit_regions(path, node)
        elif action == "custom":
            self._use_custom_selection(path, node)
        elif action == "add":
            self._insert_after(
                path,
                StructuredRegionCondition(
                    subject=node.subject, condition=node.condition
                ),
            )
        elif action == "duplicate":
            self._duplicate(path)
        elif action == "group":
            updated = _replace_node(
                self._value,
                path,
                StructuredRegionGroup(children=(node,)),
            )
            assert isinstance(updated, StructuredRegionGroup)
            self._value = updated
            self._active_path = path + (0,)
            self._rebuild()
        elif action == "remove":
            self._remove(path)

    def _edit_regions(
        self,
        path: tuple[int, ...],
        condition: StructuredRegionCondition,
    ) -> None:
        if self._atlas is None:
            self._editor_status.setText(
                "Load a compatible Allen atlas before selecting regions."
            )
            return
        self._picker.set_selection(condition.region_selection)
        if not self._picker.exec():
            return
        selection = self._picker.selection(self._structures)
        self._set_condition(
            path,
            replace(condition, region_selection=selection),
            rebuild=True,
        )
        if selection is None:
            self._editor_status.setText(
                "The condition still needs at least one selected region."
            )
        else:
            self._editor_status.setText("")

    def _use_custom_selection(
        self,
        path: tuple[int, ...],
        condition: StructuredRegionCondition,
    ) -> None:
        provider = self._custom_selection_provider
        selection = provider() if provider is not None else None
        if selection is None:
            self._editor_status.setText(
                "Select one or more terminal Custom Regions first."
            )
            return
        self._set_condition(
            path,
            replace(condition, region_selection=selection),
            rebuild=True,
        )
        self._editor_status.setText("")

    def _update_canonical(self) -> None:
        try:
            query = structured_query_from_value(self._value)
        except RegionQueryValidationError as exc:
            self._canonical.setText(f"Incomplete query: {exc}")
            self.query_changed.emit(None)
            return
        self._canonical.setText(format_structured_query(query))
        self.query_changed.emit(query)


__all__ = [
    "CONDITION_CABLE_LENGTH",
    "CONDITION_INTERSECTS",
    "CONDITION_IN_REGION",
    "CONDITION_NODE_COUNT",
    "CONDITION_TERMINUS_COUNT",
    "GROUP_AND",
    "GROUP_OR",
    "SUBJECT_NEURITE",
    "SUBJECT_SOMA",
    "CompoundRegionQueryWidget",
    "RegionPickerDialog",
    "StructuredRegionCondition",
    "StructuredRegionGroup",
    "format_structured_query",
    "remap_structured_query_regions",
    "structured_condition_options",
    "structured_query_from_value",
]

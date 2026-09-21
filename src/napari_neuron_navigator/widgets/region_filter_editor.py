"""Qt editor for per-region clustering include and exclude rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - fallback supports lightweight import-only tests
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QHeaderView,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
    )
except ImportError:  # pragma: no cover
    QAbstractItemView = QHeaderView = QTableWidget = QTableWidgetItem = QTabWidget = (
        None
    )

from ..analysis.clustering import (
    ClusterExclusionRule,
    ClusterRegionFilter,
    ClusterRegionRule,
)
from ..swc import NodeType
from .node_type_selector import NodeTypeSelectorComboBox, node_type_options
from .region_selector import RegionSelectorWidget


@dataclass
class _RuleSettings:
    dilation_percent: int = 0
    node_types: tuple[int, ...] | None = (NodeType.SOMA,)
    minimum_node_count: int = 1


_node_type_options = node_type_options


class RegionFilterEditorWidget(QWidget):
    """Edit include/exclude region rules for one input-neuron scope."""

    rules_changed = Signal()

    def __init__(
        self,
        *,
        node_types: Iterable[int] = (NodeType.SOMA,),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._node_types = tuple(sorted({int(value) for value in node_types}))
        if NodeType.SOMA not in self._node_types:
            self._node_types = (NodeType.SOMA, *self._node_types)
        self._include_settings: dict[int, _RuleSettings] = {}
        self._exclude_settings: dict[int, _RuleSettings] = {}
        self._syncing = False
        self._soma_mode = False
        self._exclude_soma_widgets: list[QWidget] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.include_selector = RegionSelectorWidget(
            single_select=False,
            show_include_children=False,
            force_include_children=True,
        )
        self.exclude_selector = RegionSelectorWidget(
            single_select=False,
            show_include_children=False,
            force_include_children=True,
        )
        self.include_default_dilation = self._default_dilation_control()
        self.exclude_default_dilation = self._default_dilation_control()
        self.include_table = None
        self.exclude_table = None

        if QTabWidget is None:
            layout.addWidget(QLabel("Include regions"))
            layout.addWidget(self.include_selector)
            layout.addWidget(QLabel("Exclude regions"))
            layout.addWidget(self.exclude_selector)
        else:
            tabs = QTabWidget()
            tabs.addTab(
                self._build_tab(excluded=False),
                "Include",
            )
            tabs.addTab(
                self._build_tab(excluded=True),
                "Exclude",
            )
            layout.addWidget(tabs)
            self.tabs = tabs

        self.include_selector.selection_changed.connect(
            lambda _values: self._on_selection_changed(False)
        )
        self.exclude_selector.selection_changed.connect(
            lambda _values: self._on_selection_changed(True)
        )

    @staticmethod
    def _default_dilation_control() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(0)
        spin.setSuffix("%")
        return spin

    def _build_tab(self, *, excluded: bool) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.addWidget(QLabel("Default dilation for new regions:"))
        row.addWidget(
            self.exclude_default_dilation if excluded else self.include_default_dilation
        )
        layout.addLayout(row)
        layout.addWidget(self.exclude_selector if excluded else self.include_selector)

        table = QTableWidget()
        headers = (
            ["Region", "Dilation %", "Soma node types", "Minimum nodes"]
            if excluded
            else ["Region", "Dilation %"]
        )
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(table)
        if excluded:
            self.exclude_table = table
            caution = QLabel(
                "Soma Location only: a neuron is excluded when this many selected-"
                "type nodes enter the region. Axon-typed labels come from source "
                "annotations and may include mislabelled dendrites."
            )
            caution.setWordWrap(True)
            layout.addWidget(caution)
        else:
            self.include_table = table
        return page

    def set_node_types(self, values: Iterable[int]) -> None:
        """Replace the node-type catalog while retaining valid rule choices."""
        values_tuple = tuple(sorted({int(value) for value in values}))
        if NodeType.SOMA not in values_tuple:
            values_tuple = (NodeType.SOMA, *values_tuple)
        self._node_types = values_tuple
        available = set(values_tuple)
        for settings in self._exclude_settings.values():
            if settings.node_types is not None:
                retained = tuple(
                    value for value in settings.node_types if value in available
                )
                settings.node_types = retained or (NodeType.SOMA,)
        self._rebuild_tables()

    def set_soma_mode(self, enabled: bool) -> None:
        """Enable exclusion eligibility columns only for Soma Location."""
        self._soma_mode = bool(enabled)
        for widget in self._exclude_soma_widgets:
            widget.setEnabled(self._soma_mode)

    def set_atlas_and_allowed_ids(self, atlas, allowed_ids: set[int]) -> None:
        """Refresh both trees without losing settings for retained regions."""
        self._syncing = True
        try:
            for selector, settings in (
                (self.include_selector, self._include_settings),
                (self.exclude_selector, self._exclude_settings),
            ):
                retained_ids = [
                    region_id for region_id in settings if region_id in allowed_ids
                ]
                selector.set_allowed_structure_ids(allowed_ids)
                selector.set_atlas(atlas)
                acronyms = [
                    str(atlas.structures[region_id].get("acronym", ""))
                    for region_id in retained_ids
                    if region_id in atlas.structures
                ]
                selector.select_regions([value for value in acronyms if value])
                for region_id in list(settings):
                    if region_id not in allowed_ids:
                        settings.pop(region_id, None)
        finally:
            self._syncing = False
        self._rebuild_tables()

    def clear(self) -> None:
        """Clear selectors and all rule settings."""
        self._syncing = True
        try:
            self.include_selector.clear()
            self.exclude_selector.clear()
            self._include_settings.clear()
            self._exclude_settings.clear()
        finally:
            self._syncing = False
        self._rebuild_tables()

    def _on_selection_changed(self, excluded: bool) -> None:
        if self._syncing:
            return
        selector = self.exclude_selector if excluded else self.include_selector
        settings = self._exclude_settings if excluded else self._include_settings
        selected = {
            int(value) for value in selector.get_selected_ids(include_children=False)
        }
        default = (
            self.exclude_default_dilation.value()
            if excluded
            else self.include_default_dilation.value()
        )
        for region_id in sorted(selected):
            settings.setdefault(
                region_id,
                _RuleSettings(dilation_percent=int(default)),
            )
        for region_id in list(settings):
            if region_id not in selected:
                settings.pop(region_id, None)
        self._rebuild_tables()
        self.rules_changed.emit()

    def _region_label(self, selector: RegionSelectorWidget, region_id: int) -> str:
        structure = selector._structure_map.get(int(region_id), {})
        acronym = str(structure.get("acronym", f"Region {region_id}"))
        name = str(structure.get("name", "")).strip()
        return f"{acronym} ({name})" if name and name != acronym else acronym

    def _rebuild_tables(self) -> None:
        self._exclude_soma_widgets.clear()
        if self.include_table is not None:
            self._populate_table(False)
        if self.exclude_table is not None:
            self._populate_table(True)

    def _populate_table(self, excluded: bool) -> None:
        table = self.exclude_table if excluded else self.include_table
        if table is None:
            return
        settings_map = self._exclude_settings if excluded else self._include_settings
        selector = self.exclude_selector if excluded else self.include_selector
        table.setRowCount(len(settings_map))
        for row, region_id in enumerate(sorted(settings_map)):
            settings = settings_map[region_id]
            table.setItem(
                row, 0, QTableWidgetItem(self._region_label(selector, region_id))
            )

            dilation = QSpinBox()
            dilation.setRange(0, 100)
            dilation.setSuffix("%")
            dilation.setValue(int(settings.dilation_percent))
            dilation.valueChanged.connect(
                lambda value, region_id=region_id, excluded=excluded: (
                    self._set_dilation(excluded, region_id, value)
                )
            )
            table.setCellWidget(row, 1, dilation)

            if not excluded:
                continue
            types = NodeTypeSelectorComboBox(
                options=_node_type_options(self._node_types)
            )
            types.set_selected_node_types(settings.node_types)
            types.selection_changed.connect(
                lambda values, region_id=region_id: self._set_node_types(
                    region_id,
                    values,
                )
            )
            minimum = QSpinBox()
            minimum.setRange(1, 2_147_483_647)
            minimum.setValue(int(settings.minimum_node_count))
            minimum.valueChanged.connect(
                lambda value, region_id=region_id: self._set_minimum(
                    region_id,
                    value,
                )
            )
            types.setEnabled(self._soma_mode)
            minimum.setEnabled(self._soma_mode)
            self._exclude_soma_widgets.extend((types, minimum))
            table.setCellWidget(row, 2, types)
            table.setCellWidget(row, 3, minimum)

    def _set_dilation(self, excluded: bool, region_id: int, value: int) -> None:
        settings = self._exclude_settings if excluded else self._include_settings
        if region_id in settings:
            settings[region_id].dilation_percent = int(value)
            self.rules_changed.emit()

    def _set_node_types(self, region_id: int, values) -> None:
        if region_id in self._exclude_settings:
            self._exclude_settings[region_id].node_types = (
                None if values is None else tuple(int(value) for value in values)
            )
            self.rules_changed.emit()

    def _set_minimum(self, region_id: int, value: int) -> None:
        if region_id in self._exclude_settings:
            self._exclude_settings[region_id].minimum_node_count = int(value)
            self.rules_changed.emit()

    def _rule_context(self, selector, region_id, represented_provider):
        structure = selector._structure_map.get(int(region_id), {})
        acronym = str(structure.get("acronym", "")).strip()
        represented = represented_provider([int(region_id)])
        represented_ids = tuple(int(value[0]) for value in represented)
        represented_acronyms = tuple(str(value[1]) for value in represented)
        return acronym, represented_ids, represented_acronyms

    def region_filter(self, represented_provider) -> ClusterRegionFilter | None:
        """Return immutable rules using a dataset-descendant resolver callback."""
        includes: list[ClusterRegionRule] = []
        for region_id in sorted(self._include_settings):
            settings = self._include_settings[region_id]
            acronym, represented_ids, represented_acronyms = self._rule_context(
                self.include_selector,
                region_id,
                represented_provider,
            )
            if acronym:
                includes.append(
                    ClusterRegionRule(
                        region_id=region_id,
                        acronym=acronym,
                        represented_region_ids=represented_ids,
                        represented_region_acronyms=represented_acronyms,
                        dilation_fraction=settings.dilation_percent / 100.0,
                    )
                )

        excludes: list[ClusterExclusionRule] = []
        for region_id in sorted(self._exclude_settings):
            settings = self._exclude_settings[region_id]
            acronym, represented_ids, represented_acronyms = self._rule_context(
                self.exclude_selector,
                region_id,
                represented_provider,
            )
            if acronym:
                excludes.append(
                    ClusterExclusionRule(
                        region_id=region_id,
                        acronym=acronym,
                        represented_region_ids=represented_ids,
                        represented_region_acronyms=represented_acronyms,
                        dilation_fraction=settings.dilation_percent / 100.0,
                        node_types=settings.node_types,
                        minimum_node_count=settings.minimum_node_count,
                    )
                )
        result = ClusterRegionFilter(
            include_rules=tuple(includes),
            exclude_rules=tuple(excludes),
        )
        return None if result.is_empty else result

    def summary(self, *, excluded: bool) -> str:
        """Return a compact summary of one rule side."""
        settings = self._exclude_settings if excluded else self._include_settings
        selector = self.exclude_selector if excluded else self.include_selector
        if not settings:
            return "None"
        labels = [
            f"{self._region_label(selector, region_id).split(' (', 1)[0]} "
            f"(+{settings[region_id].dilation_percent}%)"
            for region_id in sorted(settings)[:2]
        ]
        if len(settings) > 2:
            labels.append(f"+{len(settings) - 2} more")
        return ", ".join(labels)

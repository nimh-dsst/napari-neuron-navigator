"""Compare live neuron sets and their thresholded spatial regions."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..analysis.set_overlap import build_set_intersection_mask, compare_set_regions
from ..neuron_sets import membership_overlap


class SetOverlapWidget(QGroupBox):
    def __init__(
        self,
        table,
        layers_provider,
        atlas_name_provider,
        parent=None,
        *,
        intersection_mask_creator=None,
    ):
        super().__init__("Neuron Set Overlap", parent)
        self.table = table
        self.layers_provider = layers_provider
        self.atlas_name_provider = atlas_name_provider
        self.intersection_mask_creator = intersection_mask_creator
        self._signature = None
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Compare live set membership and spatial regions. Create each set's "
            "heatmap in Data → Table view: Sets, blur it in Tools, then use "
            "Histogram to threshold each blurred layer into a separate mask. "
            "Choose those masks below; spatial results depend on sigma and thresholds."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.set_combos = []
        self.region_combos = []
        for label in ("A", "B"):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Set {label}:"))
            combo = QComboBox()
            combo.currentIndexChanged.connect(self.refresh_regions)
            self.set_combos.append(combo)
            row.addWidget(combo, 1)
            row.addWidget(QLabel("Region:"))
            regions = QComboBox()
            regions.currentIndexChanged.connect(self.invalidate)
            self.region_combos.append(regions)
            row.addWidget(regions, 1)
            layout.addLayout(row)
        self.compare_button = QPushButton("Compare Sets")
        self.compare_button.clicked.connect(self.compare)
        layout.addWidget(self.compare_button)
        self.create_intersection_button = QPushButton("Create Intersection Mask")
        self.create_intersection_button.setToolTip(
            "Create a new binary mask containing only voxels inside both selected regions."
        )
        self.create_intersection_button.clicked.connect(self.create_intersection_mask)
        layout.addWidget(self.create_intersection_button)
        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(["Measure", "Neurons", "Spatial voxels"])
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.results)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh()

    def invalidate(self, *_args):
        self.results.setRowCount(0)
        self.status.setText("Compare Sets to calculate current overlap.")
        self.create_intersection_button.setEnabled(
            self.intersection_mask_creator is not None
            and len(self.region_combos) == 2
            and all(combo.currentData() is not None for combo in self.region_combos)
        )

    def refresh(self):
        items = self.table.neuron_sets.sets()
        signature = tuple(
            (s.set_id, s.name, s.rule.members(self.table._entries)) for s in items
        )
        if signature != self._signature:
            self._signature = signature
            self.invalidate()
        for index, combo in enumerate(self.set_combos):
            previous = combo.currentData()
            blocked = combo.blockSignals(True)
            combo.clear()
            for item in items:
                combo.addItem(item.name, item.set_id)
            selected = combo.findData(previous)
            combo.setCurrentIndex(
                selected if selected >= 0 else min(index, len(items) - 1)
            )
            combo.blockSignals(blocked)
        self.refresh_regions()

    def refresh_regions(self, *_args):
        if len(self.region_combos) != 2:
            return
        layers = list(self.layers_provider())
        for set_combo, regions in zip(self.set_combos, self.region_combos, strict=True):
            previous = regions.currentData()
            blocked = regions.blockSignals(True)
            regions.clear()
            regions.addItem("Membership only", None)
            for layer in layers:
                snapshot = layer.metadata.get("neuron_set") or {}
                if snapshot.get("set_id") == set_combo.currentData():
                    regions.addItem(layer.name, layer)
            match = next(
                (i for i in range(regions.count()) if regions.itemData(i) is previous),
                0,
            )
            regions.setCurrentIndex(match)
            regions.blockSignals(blocked)
        self.invalidate()

    def _current_snapshots(self):
        definitions = [
            self.table.neuron_sets.get(c.currentData()) for c in self.set_combos
        ]
        if not all(definitions):
            raise ValueError("Define two sets in Data → Table view: Sets first.")
        return [
            {**s.to_state(), "file_ids": list(s.rule.members(self.table._entries))}
            for s in definitions
        ]

    def create_intersection_mask(self):
        try:
            snapshots = self._current_snapshots()
            layers = [combo.currentData() for combo in self.region_combos]
            if any(layer is None for layer in layers):
                raise ValueError("Choose a region mask for each set first.")
            if self.intersection_mask_creator is None:
                raise ValueError("Mask creation is unavailable in this view.")
            mask = build_set_intersection_mask(
                *layers,
                set_a=snapshots[0],
                set_b=snapshots[1],
                atlas_name=self.atlas_name_provider(),
            )
            count = int(mask.sum())
            layer = self.intersection_mask_creator(mask, layers)
            self.status.setText(
                f"Created {layer.name}: {count:,} intersecting voxels."
                + (" The intersection is empty." if count == 0 else "")
            )
        except (ValueError, RuntimeError, MemoryError) as exc:
            self.status.setText(f"Intersection mask unavailable: {exc}")

    def compare(self):
        try:
            snapshots = self._current_snapshots()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        neurons = membership_overlap(*(s["file_ids"] for s in snapshots))
        layers = [combo.currentData() for combo in self.region_combos]
        spatial = None
        message = "Live membership compared. Choose a region for each set for spatial overlap."
        if all(layer is not None for layer in layers):
            try:
                spatial = compare_set_regions(
                    *layers,
                    set_a=snapshots[0],
                    set_b=snapshots[1],
                    atlas_name=self.atlas_name_provider(),
                )
                details = []
                for layer in layers:
                    m = layer.metadata
                    details.append(
                        f"{layer.name}: sigma={m.get('sigma')}, "
                        f"lower={m.get('lower_threshold')}, upper={m.get('upper_threshold')}"
                    )
                message = "; ".join(details)
            except ValueError as exc:
                message = f"Spatial comparison unavailable: {exc}"
        metrics = (
            ("a", "Set A"),
            ("b", "Set B"),
            ("intersection", "Intersection"),
            ("union", "Union"),
            ("a_only", "A only"),
            ("b_only", "B only"),
            ("fraction_a", "Intersection / A"),
            ("fraction_b", "Intersection / B"),
            ("jaccard", "Jaccard (intersection / union)"),
            ("dice", "Dice"),
        )
        self.results.setRowCount(len(metrics))
        for row, (key, title) in enumerate(metrics):
            self.results.setItem(row, 0, QTableWidgetItem(title))
            for column, data in ((1, neurons), (2, spatial)):
                value = data.get(key) if data else None
                text = (
                    "N/A"
                    if value is None
                    else f"{value:.2%}"
                    if isinstance(value, float)
                    else f"{value:,}"
                )
                self.results.setItem(row, column, QTableWidgetItem(text))
        self.status.setText(message)

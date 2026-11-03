"""Regional-profile clustering controls, feature heatmap, and cohort handoff."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.regional_clustering import (
    MODE_LABELS,
    RegionalClusteringSettings,
    RegionalMode,
    estimate_regional_memory,
    save_regional_cluster_result,
    save_regional_neuron_group,
)
from .collapsible_section import CollapsibleSection
from .region_filter_editor import RegionFilterEditorWidget

_LIVE_THREADS: set[QThread] = set()


class RegionalClusteringWidget(QWidget):
    """The three independent measurement modes share a fixed average linkage."""

    result_ready = Signal(object)
    cluster_selected = Signal(object)
    neuron_group_saved = Signal(object, str)

    def __init__(
        self, cohort_provider, grouping_provider, annotation_provider, parent=None
    ):
        super().__init__(parent)
        self._cohort_provider = cohort_provider
        self._grouping_provider = grouping_provider
        self._annotation_provider = annotation_provider
        self._source_path = None
        self._atlas = None
        self._generation = 0
        self._thread = self._worker = None
        self._running_settings = None
        self._running = False
        self._closing = False
        self.result = None
        self.assignment_id = None
        self._heat_axis = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.section = CollapsibleSection("Regional-Profile Clustering", expanded=False)
        layout.addWidget(self.section)
        content = self.section.content_layout()
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        self.scope = QComboBox()
        self.scope.addItem("Current table", "table")
        self.scope.addItem("Selected rows", "selected")
        form.addRow("Cohort scope:", self.scope)
        self.mode = QComboBox()
        for mode, label in MODE_LABELS.items():
            self.mode.addItem(label, mode.value)
        form.addRow("Measurement:", self.mode)
        self.grouping = QComboBox()
        self.grouping.addItem("Direct atlas regions", "direct")
        self.grouping.addItem("Collapsed Region Report groups", "report")
        self.grouping.setToolTip(
            "Collapse branches in Region Report to combine their retained direct "
            "regions. Other regions remain direct. Outermost collapsed groups "
            "take precedence, consistently across lateralities."
        )
        form.addRow("Anatomical grouping:", self.grouping)
        self.laterality = QComboBox()
        for label, value in (
            ("Separate sides (including midline / unknown)", "separate"),
            ("Combine all sides", "combined"),
            ("Ipsilateral only", "ipsilateral"),
            ("Contralateral only", "contralateral"),
        ):
            self.laterality.addItem(label, value)
        form.addRow("Laterality:", self.laterality)
        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(1, 1000)
        self.cluster_count.setValue(5)
        form.addRow("Requested clusters:", self.cluster_count)
        self.memory_limit = QSpinBox()
        self.memory_limit.setRange(128, 4096)
        self.memory_limit.setValue(512)
        self.memory_limit.setSuffix(" MiB")
        form.addRow("Working memory limit:", self.memory_limit)
        self.exclusion_section = CollapsibleSection("Exclude Regions", expanded=False)
        self.exclusions = RegionFilterEditorWidget(anatomical_exclusions_only=True)
        self.exclusion_section.content_layout().addWidget(self.exclusions)
        form.addRow(self.exclusion_section)
        content.addWidget(self.controls)
        self.definition = QLabel(
            "Neurite cable and reconstructed termini include all non-soma types. "
            "Cable/termini use cosine distance; regions reached uses Jaccard. "
            "All modes use average linkage. Zero-feature neurons are excluded."
        )
        self.definition.setWordWrap(True)
        content.addWidget(self.definition)
        self.memory_label = QLabel()
        self.memory_label.setWordWrap(True)
        self.memory_label.setToolTip(
            "The estimate covers distance buffers, dense features, and a linkage "
            "process allowance. Existing application/atlas memory is additional."
        )
        content.addWidget(self.memory_label)
        actions = QHBoxLayout()
        self.run_button = QPushButton("Run Regional Clustering")
        self.cancel_button = QPushButton("Cancel Clustering")
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        content.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        content.addWidget(self.progress)
        self.status = QLabel("Load a neuron Parquet and compatible atlas.")
        self.status.setWordWrap(True)
        content.addWidget(self.status)
        self.zero_section = CollapsibleSection(
            "Zero-Feature Neurons (Excluded)", expanded=False
        )
        self.zero_table = QTableWidget(0, 1)
        self.zero_table.setHorizontalHeaderLabels(["file_id"])
        self.zero_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.zero_table.setMinimumHeight(130)
        self.zero_section.content_layout().addWidget(self.zero_table)
        content.addWidget(self.zero_section)
        inspect = QHBoxLayout()
        inspect.addWidget(QLabel("Inspect cluster:"))
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self._select_cluster)
        inspect.addWidget(self.cluster_combo)
        content.addLayout(inspect)
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Overlay Data annotations:"))
        self.overlay = QComboBox()
        for label, value in (
            ("None", ""),
            ("Label", "label"),
            ("Group", "group"),
            ("Tags", "tags"),
        ):
            self.overlay.addItem(label, value)
        self.overlay.currentIndexChanged.connect(self.draw)
        overlay_row.addWidget(self.overlay)
        refresh = QPushButton("Refresh Overlay")
        refresh.clicked.connect(self.draw)
        overlay_row.addWidget(refresh)
        self.normalize = QCheckBox("Normalize heatmap rows (display only)")
        self.normalize.setChecked(True)
        self.normalize.toggled.connect(self.draw)
        overlay_row.addWidget(self.normalize)
        content.addLayout(overlay_row)
        self.figure = Figure(figsize=(9, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(360)
        self.canvas.mpl_connect("button_press_event", self._heatmap_clicked)
        content.addWidget(self.canvas)
        self.overlay_counts = QTableWidget()
        self.overlay_counts.setEditTriggers(QTableWidget.NoEditTriggers)
        self.overlay_counts.setVisible(False)
        content.addWidget(self.overlay_counts)
        note = QLabel(
            "Annotation overlays are post hoc consistency checks. They are never "
            "clustering inputs and do not provide independent biological validation."
        )
        note.setWordWrap(True)
        content.addWidget(note)
        save_row = QHBoxLayout()
        self.group_name = QLineEdit()
        self.group_name.setPlaceholderText("Neuron group name")
        save_row.addWidget(self.group_name)
        self.save_group = QPushButton("Save Neuron Group...")
        self.save_group.clicked.connect(self._save_group)
        save_row.addWidget(self.save_group)
        self.save_result = QPushButton("Save Clustering Result...")
        self.save_result.clicked.connect(self._save_result)
        save_row.addWidget(self.save_result)
        content.addLayout(save_row)
        for combo in (self.scope, self.mode, self.grouping, self.laterality):
            combo.currentIndexChanged.connect(self.notify_cohort_changed)
        self._set_busy(False)
        self.notify_cohort_changed()

    def set_context(self, source_path, atlas):
        source = None if source_path is None else str(source_path)
        if source == self._source_path and atlas is self._atlas:
            return
        self.cancel()
        self._running = False
        self._generation += 1
        self._source_path, self._atlas = source, atlas
        self.result = self.assignment_id = None
        self.cluster_combo.clear()
        self.group_name.clear()
        self.exclusions.clear()
        if atlas is not None:
            structures = dict(atlas.structures)
            structures[0] = {
                "id": 0,
                "name": "Outside / unmapped",
                "acronym": "unmapped",
                "structure_id_path": [0],
            }
            self.exclusions.set_atlas_and_allowed_ids(
                SimpleNamespace(
                    atlas_name=getattr(atlas, "atlas_name", ""), structures=structures
                ),
                set(structures),
            )
        self.zero_table.setRowCount(0)
        self.figure.clear()
        self.overlay_counts.setVisible(False)
        self._heat_axis = None
        self.canvas.draw_idle()
        self._set_busy(self._thread is not None)
        self.status.setText(
            "Choose a cohort and run regional-profile clustering."
            if source is not None and atlas is not None
            else "Load a neuron Parquet and compatible atlas."
        )
        self.notify_cohort_changed()

    def notify_cohort_changed(self, *_args):
        ids = tuple(
            sorted({str(v) for v in self._cohort_provider(self.scope.currentData())})
        )
        if (
            self._running
            and self._running_settings is not None
            and ids != self._running_settings.file_ids
        ):
            self.cancel()
            self._generation += 1
            self.status.setText(
                "Cohort changed; cancelled the previous clustering snapshot."
            )
        estimate = estimate_regional_memory(len(ids))
        self.memory_label.setText(
            f"Next-run cohort: {len(ids):,} neurons. Pairwise distances: up to "
            f"{estimate.condensed_bytes / 1024**2:.1f} MiB; working estimate "
            f"{estimate.estimated_peak_bytes / 1024**2:.1f} MiB plus features. "
            "The guard uses the nonzero cohort before allocating distances."
        )

    def _settings(self):
        return RegionalClusteringSettings(
            file_ids=tuple(self._cohort_provider(self.scope.currentData())),
            mode=self.mode.currentData(),
            scope=self.scope.currentData(),
            group_region_ids=tuple(self._grouping_provider())
            if self.grouping.currentData() == "report"
            else (),
            excluded_region_ids=self.exclusions.excluded_region_ids(),
            laterality=self.laterality.currentData(),
            n_clusters=self.cluster_count.value(),
            memory_limit_bytes=self.memory_limit.value() * 1024**2,
        )

    def _set_busy(self, busy):
        self.controls.setEnabled(not busy)
        self.run_button.setEnabled(
            not busy and self._source_path is not None and self._atlas is not None
        )
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)
        self.canvas.setEnabled(not busy)
        self.cluster_combo.setEnabled(not busy and self.result is not None)
        self.save_result.setEnabled(not busy and self.result is not None)
        self.save_group.setEnabled(
            not busy
            and self.result is not None
            and self.cluster_combo.currentData() is not None
        )

    def run(self):
        if self._thread is not None or self._source_path is None or self._atlas is None:
            return
        from ..workers import RegionalClusteringWorker

        try:
            settings = self._settings()
            if not settings.file_ids:
                raise ValueError(
                    "Choose a nonempty current-table or selected-row cohort."
                )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._generation += 1
        self.result = self.assignment_id = None
        self.cluster_combo.clear()
        self.figure.clear()
        self._heat_axis = None
        self.overlay_counts.setVisible(False)
        self.canvas.draw_idle()
        worker = RegionalClusteringWorker(
            self._generation, self._source_path, self._atlas, settings
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.prepared.connect(self._prepared)
        worker.progress.connect(self._progress)
        worker.finished.connect(self._finished)
        worker.error.connect(self._error)
        worker.cancelled.connect(self._error)
        worker.done.connect(worker.deleteLater)
        worker.done.connect(thread.quit)
        thread.finished.connect(lambda: self._cleanup(thread))
        thread.finished.connect(thread.deleteLater)
        self._worker, self._thread, self._running_settings = worker, thread, settings
        self._running = True
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.status.setText("Validating regional profile...")
        self.zero_table.setRowCount(0)
        _LIVE_THREADS.add(thread)
        thread.start()

    def cancel(self):
        if self._worker is not None:
            # Invalidate even a completion already queued on the GUI thread.
            self._generation += 1
            self._worker.cancel()
            self._running = False
            if not self._closing:
                self.status.setText("Cancelling regional clustering...")
                self.cancel_button.setEnabled(False)

    def _prepared(self, generation, features):
        if self._closing or generation != self._generation:
            return
        self.zero_table.setRowCount(len(features.zero_feature_file_ids))
        for row, file_id in enumerate(features.zero_feature_file_ids):
            self.zero_table.setItem(row, 0, QTableWidgetItem(file_id))
        estimate = estimate_regional_memory(
            len(features.retained_file_ids), features.matrix.size * 8
        )
        self.status.setText(
            f"{len(features.retained_file_ids):,} nonzero neurons; "
            f"{len(features.zero_feature_file_ids):,} zero-feature neurons excluded "
            f"(listed below); {len(features.columns):,} features. "
            f"Estimated peak: {estimate.estimated_peak_bytes / 1024**2:.1f} MiB."
        )

    def _progress(self, message, current, total):
        if not self._closing and self._running:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
            self.status.setText(message)

    def _finished(self, generation, result):
        if self._closing or generation != self._generation:
            return
        self._running = False
        self.result = result
        self.assignment_id = None
        self.result_ready.emit(result)
        blocked = self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        self.cluster_combo.addItem("Choose a cluster...", None)
        for label in sorted(set(result.labels.tolist())):
            self.cluster_combo.addItem(
                f"Cluster {label} ({len(result.file_ids_for_cluster(label)):,} neurons)",
                label,
            )
        self.cluster_combo.blockSignals(blocked)
        self.status.setText(
            f"Clustered {len(result.file_ids):,} neurons into {len(set(result.labels))} clusters; "
            f"excluded {len(result.features.zero_feature_file_ids):,} zero-feature neurons."
        )
        self._set_busy(False)
        self.draw()

    def _error(self, generation, message):
        if not self._closing and generation == self._generation:
            self._running = False
            self.status.setText(message)

    def _cleanup(self, thread):
        _LIVE_THREADS.discard(thread)
        if self._thread is thread:
            self._thread = self._worker = None
            self._running = False
        if not self._closing:
            self._set_busy(False)
            if self.status.text() == "Cancelling regional clustering...":
                self.status.setText("Regional-profile clustering was cancelled.")

    def _select_cluster(self, *_args):
        label = self.cluster_combo.currentData()
        self.save_group.setEnabled(
            self.result is not None and label is not None and not self._running
        )
        if self.result is None or label is None or self._running:
            return
        self.group_name.setText(
            f"Regional {self.result.features.settings.mode.value} cluster {label}"
        )
        self.cluster_selected.emit(
            (
                self.assignment_id,
                self.result.file_ids_for_cluster(label),
                self.result.features.settings,
            )
        )

    def draw(self, *_args):
        self.figure.clear()
        self._heat_axis = None
        self.overlay_counts.setVisible(False)
        if self.result is None:
            self.canvas.draw_idle()
            return
        result = self.result
        ordered = result.matrix[result.leaf_order].astype(float)
        normalize = (
            self.normalize.isChecked()
            and result.features.settings.mode is not RegionalMode.REACHED
        )
        if normalize:
            ordered /= ordered.sum(axis=1, keepdims=True)
        ids = [result.file_ids[i] for i in result.leaf_order]
        labels = result.labels[result.leaf_order]
        field = self.overlay.currentData()
        annotations = self._annotation_provider(ids, field) if field else {}
        values = [annotations.get(file_id) or "Unlabelled" for file_id in ids]
        categories = sorted(set(values)) if field else []
        widths = [0.03, 1] if not field else [0.03, 0.03, 1]
        axes = self.figure.subplots(
            1, len(widths), gridspec_kw={"width_ratios": widths}, squeeze=False
        )[0]
        axes[0].imshow(
            labels[:, None], aspect="auto", interpolation="nearest", cmap="tab20"
        )
        axes[0].set_title("Cluster", fontsize=8)
        axes[0].set_xticks([])
        clusters = sorted(set(labels.tolist()))
        axes[0].set_yticks(
            [float(np.flatnonzero(labels == label).mean()) for label in clusters],
            [str(label) for label in clusters],
            fontsize=7,
        )
        if field:
            codes = {v: i for i, v in enumerate(categories)}
            axes[1].imshow(
                np.asarray([codes[v] for v in values])[:, None],
                aspect="auto",
                interpolation="nearest",
                cmap="tab20",
            )
            axes[1].set_title(self.overlay.currentText(), fontsize=8)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            self.overlay_counts.setColumnCount(len(categories) + 1)
            self.overlay_counts.setHorizontalHeaderLabels(["Cluster", *categories])
            clusters = sorted(set(labels.tolist()))
            self.overlay_counts.setRowCount(len(clusters))
            for row, label in enumerate(clusters):
                self.overlay_counts.setItem(row, 0, QTableWidgetItem(str(label)))
                for col, category in enumerate(categories, start=1):
                    count = sum(
                        c == label and a == category
                        for c, a in zip(labels, values, strict=True)
                    )
                    self.overlay_counts.setItem(row, col, QTableWidgetItem(str(count)))
            self.overlay_counts.setVisible(True)
        axis = axes[-1]
        self._heat_axis = axis
        image = axis.imshow(
            ordered,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=1
            if normalize or result.features.settings.mode is RegionalMode.REACHED
            else None,
        )
        step = max(1, int(np.ceil(len(result.features.columns) / 60)))
        ticks = np.arange(0, len(result.features.columns), step)
        axis.set_xticks(
            ticks,
            [result.features.columns[i].label for i in ticks],
            rotation=90,
            fontsize=7,
        )
        if len(ids) <= 50:
            axis.set_yticks(np.arange(len(ids)), ids, fontsize=7)
        else:
            axis.set_yticks([])
        axis.set_ylabel("Neurons (file_id), clustering order")
        axis.set_xlabel("Nonoverlapping region / laterality features")
        title = MODE_LABELS[result.features.settings.mode]
        axis.set_title(
            f"{title}: {len(result.file_ids):,} clustered, "
            f"{len(result.features.zero_feature_file_ids):,} zero-feature excluded"
        )
        self.figure.colorbar(
            image,
            ax=axis,
            label="Within-neuron fraction"
            if normalize
            else result.metadata["measurement_units"],
        )
        self.canvas.draw_idle()

    def _heatmap_clicked(self, event):
        if (
            not self._running
            and self.result is not None
            and event.inaxes is self._heat_axis
            and event.ydata is not None
        ):
            index = round(event.ydata)
            if 0 <= index < len(self.result.leaf_order):
                label = int(self.result.labels[self.result.leaf_order[index]])
                self.cluster_combo.setCurrentIndex(self.cluster_combo.findData(label))

    def _save_result(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Regional Clustering Result",
            "regional_clustering.npz",
            "NumPy result (*.npz)",
        )
        if path:
            try:
                save_regional_cluster_result(path, self.result)
                self.status.setText(
                    f"Saved clustering features, assignments and provenance to {path}."
                )
            except (OSError, ValueError) as exc:
                self.status.setText(f"Could not save result: {exc}")

    def _save_group(self):
        label = self.cluster_combo.currentData()
        name = self.group_name.text().strip()
        if self.result is None or label is None or not name:
            self.status.setText("Choose a cluster and enter a neuron group name.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neuron Group",
            "regional_neuron_group.json",
            "Neuron group (*.json)",
        )
        if path:
            try:
                save_regional_neuron_group(path, self.result, label, name)
                self.neuron_group_saved.emit(
                    self.result.file_ids_for_cluster(label), name
                )
                self.status.setText(
                    f"Saved group {name} to {path}; Group metadata applied in Data."
                )
            except (OSError, ValueError) as exc:
                self.status.setText(f"Could not save neuron group: {exc}")

    def shutdown(self):
        self._closing = True
        self.cancel()

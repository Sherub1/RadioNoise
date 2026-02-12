# -*- coding: utf-8 -*-
"""
NIST SP 800-22 test widget.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QRadioButton, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QButtonGroup, QFileDialog
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor
import numpy as np

from radionoise.gui.workers.nist_worker import NistWorker
from radionoise.gui.widgets.worker_mixin import WorkerWidgetMixin


# Default storage directory
RADIONOISE_DIR = Path.home() / ".radionoise"
ENTROPY_DIR = RADIONOISE_DIR / "entropy"


class NistWidget(WorkerWidgetMixin, QWidget):
    """Widget for NIST SP 800-22 tests."""

    tests_completed = pyqtSignal(dict)
    test_error = pyqtSignal(str)
    entropy_saved = pyqtSignal(str)  # Emits saved file path

    def __init__(self):
        super().__init__()
        self._entropy: np.ndarray | None = None
        self._entropy_source: str = ""
        self._worker: NistWorker | None = None
        self._results: dict | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Options
        options_group = QGroupBox("Options des tests")
        options_layout = QVBoxLayout(options_group)

        self.fast_mode = QRadioButton("Mode rapide (9 tests)")
        self.fast_mode.setChecked(True)
        options_layout.addWidget(self.fast_mode)

        self.full_mode = QRadioButton("Mode complet (15 tests, ~30s)")
        options_layout.addWidget(self.full_mode)

        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.fast_mode)
        self._mode_group.addButton(self.full_mode)

        layout.addWidget(options_group)

        # Results table
        results_group = QGroupBox("Résultats")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Test", "P-value", "Résultat"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        results_layout.addWidget(self.results_table)

        # Summary
        summary_row = QHBoxLayout()
        self.summary_label = QLabel("En attente des tests...")
        self.summary_label.setStyleSheet("font-weight: bold;")
        summary_row.addWidget(self.summary_label)
        summary_row.addStretch()
        results_layout.addLayout(summary_row)

        layout.addWidget(results_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_row = QHBoxLayout()

        self.run_btn = QPushButton("Lancer les tests")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_tests)
        button_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_tests)
        button_row.addWidget(self.cancel_btn)

        self.export_btn = QPushButton("Exporter")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        button_row.addWidget(self.export_btn)

        layout.addLayout(button_row)

        layout.addStretch()

    def set_entropy(self, entropy: np.ndarray, source: str = ""):
        """Set entropy data for testing."""
        self._entropy = entropy
        self._entropy_source = source
        self.run_btn.setEnabled(True)
        self.summary_label.setText(f"Prêt à tester {len(entropy):,} octets")

    def _run_tests(self):
        """Start NIST tests."""
        if self._entropy is None:
            return

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.results_table.setRowCount(0)
        self.summary_label.setText("Tests en cours...")

        self._worker = NistWorker(
            self._entropy,
            fast_mode=self.fast_mode.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.test_complete.connect(self._on_single_test_complete)
        self._worker.result_ready.connect(self._on_tests_finished)
        self._worker.error.connect(self._on_test_error)
        self._worker.start()

    def _cancel_tests(self):
        """Cancel ongoing tests."""
        if self._worker:
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait()
            self._worker = None

        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.summary_label.setText("Tests annulés")

    def _on_progress(self, test_name: str, percent: int):
        """Handle progress update."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percent)
        self.summary_label.setText(f"Test: {test_name}")

    def _on_single_test_complete(self, test_data: dict):
        """Handle a single test completing — add row to table incrementally."""
        row = self.results_table.rowCount()
        self.results_table.setRowCount(row + 1)

        name_item = QTableWidgetItem(test_data.get('name', 'Unknown'))
        self.results_table.setItem(row, 0, name_item)

        p_value = test_data.get('p_value', 0)
        p_item = QTableWidgetItem(f"{p_value:.4f}")
        p_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_table.setItem(row, 1, p_item)

        passed = test_data.get('passed', False)
        result_item = QTableWidgetItem("OK" if passed else "ÉCHEC")
        result_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if passed:
            result_item.setBackground(QColor(200, 255, 200))
        else:
            result_item.setBackground(QColor(255, 200, 200))
        self.results_table.setItem(row, 2, result_item)

    def _on_tests_finished(self, results: dict):
        """Handle test completion — table already populated incrementally."""
        self._results = results

        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        # Summary
        pass_rate = results.get('pass_rate', 0) * 100
        passed = results.get('passed', 0)
        total = results.get('total', 0)

        if pass_rate >= 95:
            color = "green"
            verdict = "EXCELLENT"
        elif pass_rate >= 80:
            color = "orange"
            verdict = "ACCEPTABLE"
        else:
            color = "red"
            verdict = "INSUFFISANT"

        self.summary_label.setText(
            f"{verdict}: {passed}/{total} tests réussis ({pass_rate:.0f}%)"
        )
        self.summary_label.setStyleSheet(f"font-weight: bold; color: {color};")

        self.export_btn.setEnabled(True)
        self.tests_completed.emit(results)

        # Clean up worker
        self._cleanup_worker()

        # Propose save if validation passed
        if pass_rate >= 95 and self._entropy is not None:
            self._propose_save_entropy(pass_rate)

    def _on_test_error(self, error: str):
        """Handle test error."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.summary_label.setText(f"ERREUR: {error}")
        self.summary_label.setStyleSheet("font-weight: bold; color: red;")
        self.test_error.emit(error)

        # Clean up worker
        self._cleanup_worker()

    def get_results(self) -> dict | None:
        """Get test results."""
        return self._results

    def load_config(self, config):
        """Load settings from Config object."""
        fast = config.get("nist", "fast_mode")
        if fast is not None:
            self.fast_mode.setChecked(fast)
            self.full_mode.setChecked(not fast)

    def save_config(self, config):
        """Save settings to Config object."""
        config.set("nist", "fast_mode", self.fast_mode.isChecked())

    def _export_results(self):
        """Export NIST test results to JSON or CSV."""
        if not self._results:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Exporter les résultats NIST", "nist_results",
            "JSON (*.json);;CSV (*.csv)"
        )
        if not path:
            return

        try:
            tests = self._results.get('tests', [])

            if path.endswith('.csv') or 'CSV' in selected_filter:
                if not path.endswith('.csv'):
                    path += '.csv'
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Test Name", "P-value", "Passed"])
                    for t in tests:
                        writer.writerow([t['name'], f"{t['p_value']:.6f}", t['passed']])
            else:
                if not path.endswith('.json'):
                    path += '.json'
                with open(path, 'w') as f:
                    json.dump(self._results, f, indent=2)

            QMessageBox.information(
                self, "Export réussi",
                f"Résultats exportés:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Erreur d'export", str(e))

    def _propose_save_entropy(self, pass_rate: float):
        """Propose to save validated entropy."""
        reply = QMessageBox.question(
            self,
            "Entropie validée",
            f"Tests NIST réussis ({pass_rate:.0f}%).\n\n"
            f"Voulez-vous sauvegarder cette entropie validée ?\n"
            f"({len(self._entropy):,} octets)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._save_entropy()

    def _save_entropy(self):
        """Save validated entropy to ~/.radionoise/entropy/"""
        try:
            # Create directory if needed
            ENTROPY_DIR.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp and source
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            source = self._entropy_source.replace("/", "-") if self._entropy_source else "unknown"
            filename = f"entropy_{timestamp}_{source}.bin"
            filepath = ENTROPY_DIR / filename

            # Save entropy
            self._entropy.tofile(filepath)

            QMessageBox.information(
                self,
                "Sauvegarde réussie",
                f"Entropie sauvegardée:\n{filepath}"
            )

            self.entropy_saved.emit(str(filepath))

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur de sauvegarde",
                f"Impossible de sauvegarder:\n{e}"
            )

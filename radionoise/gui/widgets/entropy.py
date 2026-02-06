# -*- coding: utf-8 -*-
"""
Entropy capture and configuration widget.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QCheckBox, QProgressBar, QTextEdit
)
from PyQt6.QtCore import pyqtSignal, Qt
import numpy as np

from radionoise.gui.workers import CaptureWorker
from radionoise.gui.widgets.worker_mixin import WorkerWidgetMixin


class EntropyWidget(WorkerWidgetMixin, QWidget):
    """Widget for entropy capture and configuration."""

    entropy_captured = pyqtSignal(object, str, object)  # (entropy, source, raw_iq)
    capture_started = pyqtSignal()
    capture_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._entropy: np.ndarray | None = None
        self._raw_iq_data: np.ndarray | None = None
        self._worker: CaptureWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # RTL-SDR Configuration
        sdr_group = QGroupBox("Configuration RTL-SDR")
        sdr_layout = QVBoxLayout(sdr_group)

        # Frequency
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Fréquence (MHz):"))
        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(24, 1766)
        self.frequency_spin.setValue(100.0)
        self.frequency_spin.setDecimals(1)
        self.frequency_spin.setSuffix(" MHz")
        freq_row.addWidget(self.frequency_spin)
        freq_row.addStretch()
        sdr_layout.addLayout(freq_row)

        # Samples
        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel("Échantillons:"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100000, 10000000)
        self.samples_spin.setValue(500000)
        self.samples_spin.setSingleStep(100000)
        samples_row.addWidget(self.samples_spin)
        samples_row.addStretch()
        sdr_layout.addLayout(samples_row)

        layout.addWidget(sdr_group)

        # Fallback options
        fallback_group = QGroupBox("Options de fallback")
        fallback_layout = QVBoxLayout(fallback_group)

        self.allow_fallback = QCheckBox("Autoriser fallback si RTL-SDR indisponible")
        self.allow_fallback.setChecked(True)
        fallback_layout.addWidget(self.allow_fallback)

        self.use_rdseed = QCheckBox("Utiliser RDSEED au lieu de RDRAND (plus lent, entropie directe)")
        fallback_layout.addWidget(self.use_rdseed)

        layout.addWidget(fallback_group)

        # Traceability options
        trace_group = QGroupBox("Traçabilité")
        trace_layout = QVBoxLayout(trace_group)

        self.capture_raw = QCheckBox("Conserver les données IQ brutes (pour preuves cryptographiques)")
        self.capture_raw.setChecked(True)  # Activé par défaut
        self.capture_raw.setToolTip(
            "Conserve les données IQ brutes pour permettre la création\n"
            "de preuves cryptographiques et de backups.\n"
            "Nécessite RTL-SDR. Désactiver pour économiser la mémoire."
        )
        trace_layout.addWidget(self.capture_raw)

        layout.addWidget(trace_group)

        # Status
        status_group = QGroupBox("État")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlaceholderText("En attente de capture...")
        status_layout.addWidget(self.status_text)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        layout.addWidget(status_group)

        # Buttons
        button_row = QHBoxLayout()

        self.capture_btn = QPushButton("Capturer l'entropie")
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.clicked.connect(self._start_capture)
        button_row.addWidget(self.capture_btn)

        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_capture)
        button_row.addWidget(self.cancel_btn)

        layout.addLayout(button_row)

        # Spacer
        layout.addStretch()

        # Initial status check
        self._check_rtl_sdr_status()

    def _check_rtl_sdr_status(self):
        """Check RTL-SDR availability."""
        from radionoise import is_rtl_sdr_available

        if is_rtl_sdr_available():
            self._log("RTL-SDR détecté et disponible.")
        else:
            self._log("RTL-SDR non disponible. Fallback activé.")
            self._log("Sources disponibles: RDRAND, RDSEED, CSPRNG")

    def _log(self, message: str):
        """Add message to status log."""
        self.status_text.append(message)

    def _start_capture(self):
        """Start entropy capture."""
        self.capture_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        self._log(f"Démarrage de la capture...")
        self._log(f"   Fréquence: {self.frequency_spin.value()} MHz")
        self._log(f"   Échantillons: {self.samples_spin.value():,}")

        self.capture_started.emit()

        self._worker = CaptureWorker(
            samples=self.samples_spin.value(),
            frequency=self.frequency_spin.value() * 1e6,
            allow_fallback=self.allow_fallback.isChecked(),
            use_rdseed=self.use_rdseed.isChecked(),
            capture_raw=self.capture_raw.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.result_ready.connect(self._on_capture_finished)
        self._worker.error.connect(self._on_capture_error)
        self._worker.start()

    def _cancel_capture(self):
        """Cancel ongoing capture."""
        if self._worker:
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait()
            self._worker = None

        self.capture_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._log("Capture annulée.")

    def _on_progress(self, message: str, percent: int = 0):
        """Handle progress update."""
        self._log(message)
        if percent > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)

    def _on_status(self, message: str):
        """Handle status update."""
        self._log(message)

    def _on_capture_finished(self, entropy: np.ndarray, source: str,
                             raw_iq: np.ndarray | None):
        """Handle capture completion."""
        self._entropy = entropy
        self._raw_iq_data = raw_iq

        self.capture_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self._log(f"Capture terminée!")
        self._log(f"   Source: {source}")
        self._log(f"   Octets obtenus: {len(entropy):,}")
        if raw_iq is not None:
            self._log(f"   Données IQ brutes: {len(raw_iq):,} octets")

        self.entropy_captured.emit(entropy, source, raw_iq)

        # Clean up worker after thread finishes
        self._cleanup_worker()

    def _on_capture_error(self, error: str):
        """Handle capture error."""
        self.capture_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self._log(f"ERREUR: {error}")
        self.capture_error.emit(error)

        # Clean up worker after thread finishes
        self._cleanup_worker()

    def load_from_file(self, path: str):
        """Load entropy from file."""
        try:
            from radionoise import load_entropy_from_file

            self._log(f"Chargement de {path}...")
            entropy = load_entropy_from_file(path)
            self._entropy = entropy

            self._log(f"Chargé: {len(entropy):,} octets")
            self.entropy_captured.emit(entropy, "file", None)

        except Exception as e:
            self._log(f"ERREUR: {e}")
            self.capture_error.emit(str(e))

    def get_entropy(self) -> np.ndarray | None:
        """Get captured entropy."""
        return self._entropy

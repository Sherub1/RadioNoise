# -*- coding: utf-8 -*-
"""
Password/passphrase generation widget.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSpinBox, QComboBox, QPushButton,
    QRadioButton, QButtonGroup, QProgressBar, QListWidget,
    QListWidgetItem, QApplication, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QFontDatabase
import numpy as np

from radionoise.gui.workers.generator_worker import GeneratorWorker
from radionoise.gui.widgets.worker_mixin import WorkerWidgetMixin


class GeneratorWidget(WorkerWidgetMixin, QWidget):
    """Widget for password/passphrase generation."""

    passwords_generated = pyqtSignal(list)
    generation_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._entropy: np.ndarray | None = None
        self._worker: GeneratorWorker | None = None
        self._passwords: list[str] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Type selection
        type_group = QGroupBox("Type de génération")
        type_layout = QVBoxLayout(type_group)

        self.password_radio = QRadioButton("Mots de passe")
        self.password_radio.setChecked(True)
        type_layout.addWidget(self.password_radio)

        self.passphrase_radio = QRadioButton("Passphrases (mots)")
        type_layout.addWidget(self.passphrase_radio)

        self._type_group = QButtonGroup(self)
        self._type_group.addButton(self.password_radio)
        self._type_group.addButton(self.passphrase_radio)

        self.password_radio.toggled.connect(self._update_ui_for_type)

        layout.addWidget(type_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        # Length
        length_row = QHBoxLayout()
        self.length_label = QLabel("Longueur:")
        length_row.addWidget(self.length_label)
        self.length_spin = QSpinBox()
        self.length_spin.setRange(4, 128)
        self.length_spin.setValue(16)
        self.length_spin.valueChanged.connect(self._update_entropy_estimate)
        length_row.addWidget(self.length_spin)
        length_row.addStretch()
        options_layout.addLayout(length_row)

        # Count
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Nombre:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(5)
        count_row.addWidget(self.count_spin)
        count_row.addStretch()
        options_layout.addLayout(count_row)

        # Charset
        charset_row = QHBoxLayout()
        self.charset_label = QLabel("Jeu de caractères:")
        charset_row.addWidget(self.charset_label)
        self.charset_combo = QComboBox()
        self.charset_combo.addItems([
            "safe - Sûr (sans ambigus)",
            "full - Complet (tous caractères)",
            "alpha - Alphabétique",
            "alphanum - Alphanumérique",
            "numeric - Numérique",
            "hex - Hexadécimal"
        ])
        self.charset_combo.currentIndexChanged.connect(self._update_entropy_estimate)
        charset_row.addWidget(self.charset_combo)
        charset_row.addStretch()
        options_layout.addLayout(charset_row)

        # Entropy estimate
        self.entropy_estimate = QLabel("Entropie estimée: ~0 bits")
        self.entropy_estimate.setStyleSheet("color: gray; font-style: italic;")
        options_layout.addWidget(self.entropy_estimate)

        layout.addWidget(options_group)

        # Results
        results_group = QGroupBox("Résultats")
        results_layout = QVBoxLayout(results_group)

        self.results_list = QListWidget()
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed.setPointSize(12)
        self.results_list.setFont(fixed)
        self.results_list.setMinimumHeight(200)
        results_layout.addWidget(self.results_list)

        # Copy button
        copy_row = QHBoxLayout()
        self.copy_btn = QPushButton("Copier la sélection")
        self.copy_btn.clicked.connect(self._copy_selected)
        self.copy_btn.setEnabled(False)
        copy_row.addWidget(self.copy_btn)

        self.copy_all_btn = QPushButton("Copier tout")
        self.copy_all_btn.clicked.connect(self._copy_all)
        self.copy_all_btn.setEnabled(False)
        copy_row.addWidget(self.copy_all_btn)

        copy_row.addStretch()
        results_layout.addLayout(copy_row)

        layout.addWidget(results_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Generate button
        button_row = QHBoxLayout()

        self.generate_btn = QPushButton("Générer")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._generate)
        button_row.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        button_row.addWidget(self.cancel_btn)

        layout.addLayout(button_row)

        # Initial update
        self._update_ui_for_type()
        self._update_entropy_estimate()

    def _update_ui_for_type(self):
        """Update UI based on password/passphrase selection."""
        is_password = self.password_radio.isChecked()

        if is_password:
            self.length_label.setText("Longueur (caractères):")
            self.length_spin.setRange(4, 128)
            self.length_spin.setValue(16)
            self.charset_label.setVisible(True)
            self.charset_combo.setVisible(True)
        else:
            self.length_label.setText("Nombre de mots:")
            self.length_spin.setRange(3, 20)
            self.length_spin.setValue(6)
            self.charset_label.setVisible(False)
            self.charset_combo.setVisible(False)

        self._update_entropy_estimate()

    def _update_entropy_estimate(self):
        """Update entropy estimate display."""
        if self.password_radio.isChecked():
            # Password entropy
            charset_map = {
                0: 70,   # safe
                1: 94,   # full
                2: 52,   # alpha
                3: 62,   # alphanum
                4: 10,   # numeric
                5: 16,   # hex
            }
            charset_size = charset_map.get(self.charset_combo.currentIndex(), 70)
            import math
            bits = self.length_spin.value() * math.log2(charset_size)
        else:
            # Passphrase entropy (~12.925 bits per word with EFF 7776-word list)
            bits = self.length_spin.value() * 12.925

        self.entropy_estimate.setText(f"Entropie estimée: ~{bits:.0f} bits")

    def _get_charset_name(self) -> str:
        """Get charset name from combo selection."""
        # Must match CHARSETS keys in radionoise.core.generator
        charset_map = {
            0: "safe",
            1: "full",
            2: "alpha",
            3: "alnum",    # Was "alphanum" - core uses "alnum"
            4: "digits",   # Was "numeric" - core uses "digits"
            5: "hex",
        }
        return charset_map.get(self.charset_combo.currentIndex(), "safe")

    def set_entropy(self, entropy: np.ndarray):
        """Set entropy data for generation."""
        self._entropy = entropy
        self.generate_btn.setEnabled(True)

    def _generate(self):
        """Start generation."""
        if self._entropy is None:
            return

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.count_spin.value())
        self.progress_bar.setValue(0)

        self.results_list.clear()
        self._passwords = []

        self._worker = GeneratorWorker(
            entropy=self._entropy,
            count=self.count_spin.value(),
            length=self.length_spin.value(),
            charset=self._get_charset_name(),
            passphrase=self.passphrase_radio.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.password_generated.connect(self._on_password_generated)
        self._worker.result_ready.connect(self._on_generation_finished)
        self._worker.error.connect(self._on_generation_error)
        self._worker.start()

    def _cancel(self):
        """Cancel generation."""
        if self._worker:
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait()
            self._worker = None

        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def _on_progress(self, current: int, total: int):
        """Handle progress update."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _on_password_generated(self, password: str, entropy_bits: float):
        """Handle single password generation."""
        item = QListWidgetItem(f"{password}  ({entropy_bits:.0f} bits)")
        self.results_list.addItem(item)
        self._passwords.append(password)

    def _on_generation_finished(self, passwords: list[str]):
        """Handle generation completion."""
        self._passwords = passwords

        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.copy_btn.setEnabled(bool(passwords))
        self.copy_all_btn.setEnabled(bool(passwords))

        self.passwords_generated.emit(passwords)

        # Clean up worker
        self._cleanup_worker()

    def _on_generation_error(self, error: str):
        """Handle generation error."""
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.generation_error.emit(error)

        # Clean up worker
        self._cleanup_worker()

    def _copy_selected(self):
        """Copy selected password to clipboard."""
        current = self.results_list.currentItem()
        if current:
            # Extract just the password (before the entropy info)
            text = current.text().split("  (")[0]
            QApplication.clipboard().setText(text)

    def _copy_all(self):
        """Copy all passwords to clipboard."""
        if self._passwords:
            QApplication.clipboard().setText("\n".join(self._passwords))

    def get_passwords(self) -> list[str]:
        """Get generated passwords."""
        return self._passwords

    def get_settings(self) -> dict:
        """Get current generation settings."""
        return {
            "length": self.length_spin.value(),
            "charset": self._get_charset_name(),
            "passphrase": self.passphrase_radio.isChecked()
        }

    def load_config(self, config):
        """Load settings from Config object."""
        gen_type = config.get("generator", "type")
        if gen_type == "passphrase":
            self.passphrase_radio.setChecked(True)
        else:
            self.password_radio.setChecked(True)
        length = config.get("generator", "length")
        if length is not None:
            self.length_spin.setValue(length)
        count = config.get("generator", "count")
        if count is not None:
            self.count_spin.setValue(count)
        charset = config.get("generator", "charset")
        if charset is not None:
            charset_map = {"safe": 0, "full": 1, "alpha": 2, "alnum": 3, "digits": 4, "hex": 5}
            idx = charset_map.get(charset, 0)
            self.charset_combo.setCurrentIndex(idx)

    def save_config(self, config):
        """Save settings to Config object."""
        config.set("generator", "type", "passphrase" if self.passphrase_radio.isChecked() else "password")
        config.set("generator", "length", self.length_spin.value())
        config.set("generator", "count", self.count_spin.value())
        config.set("generator", "charset", self._get_charset_name())

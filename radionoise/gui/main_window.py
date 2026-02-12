# -*- coding: utf-8 -*-
"""
Main window for RadioNoise GUI.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStatusBar, QLabel, QProgressBar,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QAction
import numpy as np

from radionoise.gui.widgets.generator import GeneratorWidget
from radionoise.gui.widgets.entropy import EntropyWidget
from radionoise.gui.widgets.nist import NistWidget
from radionoise.gui.widgets.traceability import TraceabilityWidget
from radionoise.config import Config


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("RadioNoise - Générateur TRNG")
        self.setMinimumSize(800, 600)

        # State
        self._entropy: np.ndarray | None = None
        self._entropy_source: str = ""
        self._raw_iq_data: np.ndarray | None = None
        self._nist_results: dict | None = None
        self._passwords: list[str] = []

        self._config = Config()

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        self._load_config()

    def _setup_ui(self):
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header = QLabel("RadioNoise")
        header_font = QFont()
        header_font.setPointSize(24)
        header_font.setWeight(QFont.Weight.Bold)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel("Générateur de nombres aléatoires par bruit radio")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Widgets
        self.entropy_widget = EntropyWidget()
        self.nist_widget = NistWidget()
        self.generator_widget = GeneratorWidget()
        self.traceability_widget = TraceabilityWidget()

        self.tabs.addTab(self.entropy_widget, "1. Entropie")
        self.tabs.addTab(self.nist_widget, "2. Tests NIST")
        self.tabs.addTab(self.generator_widget, "3. Génération")
        self.tabs.addTab(self.traceability_widget, "4. Traçabilité")

        self.tabs.setTabEnabled(1, False)  # NIST
        self.tabs.setTabEnabled(2, False)  # Generator
        self.tabs.setTabEnabled(3, False)  # Traceability

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.info_label = QLabel("En attente de capture...")
        self.status_bar.addWidget(self.info_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&Fichier")

        load_entropy = QAction("&Charger entropie...", self)
        load_entropy.triggered.connect(self._load_entropy_file)
        file_menu.addAction(load_entropy)

        save_entropy = QAction("&Sauvegarder entropie...", self)
        save_entropy.triggered.connect(self._save_entropy_file)
        file_menu.addAction(save_entropy)

        file_menu.addSeparator()

        quit_action = QAction("&Quitter", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Help menu
        help_menu = menubar.addMenu("&Aide")

        about = QAction("À &propos", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _connect_signals(self):
        """Connect widget signals."""
        # Entropy widget
        self.entropy_widget.entropy_captured.connect(self._on_entropy_captured)
        self.entropy_widget.capture_started.connect(self._on_capture_started)
        self.entropy_widget.capture_error.connect(self._on_capture_error)

        # NIST widget
        self.nist_widget.tests_completed.connect(self._on_nist_completed)
        self.nist_widget.test_error.connect(self._on_nist_error)

        # Generator widget
        self.generator_widget.passwords_generated.connect(self._on_passwords_generated)
        self.generator_widget.generation_error.connect(self._on_generation_error)

        # Traceability
        self.traceability_widget.proof_created.connect(self._on_proof_created)


    def _on_entropy_captured(self, entropy: np.ndarray, source: str,
                             raw_iq: np.ndarray | None):
        """Handle entropy capture completion."""
        self._entropy = entropy
        self._entropy_source = source
        self._raw_iq_data = raw_iq

        self.progress_bar.setVisible(False)

        # Enable NIST and generator
        self.nist_widget.set_entropy(entropy, source)
        self.generator_widget.set_entropy(entropy)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)

        self._update_status_info()
        self.status_bar.showMessage("Entropie capturée avec succès", 3000)

    def _on_capture_started(self):
        """Handle capture start."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Capture en cours...")

    def _on_capture_error(self, error: str):
        """Handle capture error."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erreur de capture", error)
        self.status_bar.showMessage("Erreur de capture", 3000)

    def _on_nist_completed(self, results: dict):
        """Handle NIST test completion."""
        self._nist_results = results
        self.progress_bar.setVisible(False)
        self._update_status_info()
        self.status_bar.showMessage("Tests NIST terminés", 3000)

    def _on_nist_error(self, error: str):
        """Handle NIST test error."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erreur NIST", error)

    def _on_passwords_generated(self, passwords: list[str]):
        """Handle password generation completion."""
        self._passwords = passwords
        # Get charset from generator widget
        settings = self.generator_widget.get_settings()
        charset = settings.get('charset', 'safe')
        self.traceability_widget.set_passwords(
            passwords,
            self._raw_iq_data,
            charset
        )
        self.progress_bar.setVisible(False)
        self.tabs.setTabEnabled(3, True)
        self._update_status_info()
        self.status_bar.showMessage(f"{len(passwords)} mot(s) de passe généré(s)", 3000)

    def _on_generation_error(self, error: str):
        """Handle generation error."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erreur de génération", error)

    def _on_proof_created(self, proof_path: str):
        """Handle proof creation."""
        self.status_bar.showMessage(f"Preuve sauvegardée: {proof_path}", 5000)

    def _update_status_info(self):
        """Update the status bar info label with current state summary."""
        parts = []
        if self._entropy is not None:
            parts.append(f"Entropie: {len(self._entropy):,} octets")
        if self._entropy_source:
            parts.append(f"Source: {self._entropy_source}")
        if self._nist_results:
            rate = self._nist_results.get('pass_rate', 0) * 100
            parts.append(f"NIST: {rate:.0f}%")
        if parts:
            self.info_label.setText(" · ".join(parts))
        else:
            self.info_label.setText("En attente de capture...")

    def _load_entropy_file(self):
        """Load entropy from file."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Charger entropie", "",
            "Fichiers binaires (*.bin);;Tous (*)"
        )
        if path:
            self.entropy_widget.load_from_file(path)

    def _save_entropy_file(self):
        """Save entropy to file."""
        if self._entropy is None:
            QMessageBox.warning(self, "Attention", "Pas d'entropie à sauvegarder.")
            return

        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder entropie", "entropy.bin",
            "Fichiers binaires (*.bin);;Tous (*)"
        )
        if path:
            self._entropy.tofile(path)
            self.status_bar.showMessage(f"Entropie sauvegardée: {path}", 3000)

    def _show_about(self):
        """Show about dialog."""
        from radionoise import __version__

        QMessageBox.about(
            self, "À propos de RadioNoise",
            f"<h2>RadioNoise v{__version__}</h2>"
            "<p>Générateur de nombres aléatoires (TRNG) utilisant le bruit radio "
            "capturé par RTL-SDR.</p>"
            "<p><b>Sources d'entropie:</b></p>"
            "<ul>"
            "<li>RTL-SDR (bruit thermique + atmosphérique)</li>"
            "<li>Intel/AMD RDRAND/RDSEED</li>"
            "<li>CSPRNG système (fallback)</li>"
            "</ul>"
            "<p>Validation NIST SP 800-22</p>"
        )

    def _load_config(self):
        """Load config into widgets."""
        self.entropy_widget.load_config(self._config)
        self.nist_widget.load_config(self._config)
        self.generator_widget.load_config(self._config)

    def _save_config(self):
        """Save widget settings to config."""
        self.entropy_widget.save_config(self._config)
        self.nist_widget.save_config(self._config)
        self.generator_widget.save_config(self._config)
        self._config.save()

    def closeEvent(self, event):
        """Handle window close."""
        # Check for active workers
        widgets = [self.entropy_widget, self.nist_widget, self.generator_widget]
        if any(w._worker is not None for w in widgets):
            reply = QMessageBox.question(
                self, "Opération en cours",
                "Une opération est en cours. Quitter quand même ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            # Cancel all active workers
            for w in widgets:
                if hasattr(w, '_cancel_capture'):
                    w._cancel_capture()
                elif hasattr(w, '_cancel_tests'):
                    w._cancel_tests()
                elif hasattr(w, '_cancel'):
                    w._cancel()

        # Save config
        self._save_config()

        # Cleanup sensitive data
        if self._entropy is not None:
            from radionoise import secure_zero
            secure_zero(self._entropy)

        if self._raw_iq_data is not None:
            from radionoise import secure_zero
            secure_zero(self._raw_iq_data)

        event.accept()


def run():
    """Run the GUI application."""
    import sys

    app = QApplication(sys.argv)

    # Apply dark theme
    from radionoise.gui.styles import apply_dark_theme
    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run()

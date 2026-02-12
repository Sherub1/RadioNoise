# -*- coding: utf-8 -*-
"""
Traceability widget for proofs, audit trail, and backups.
"""

from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QTabWidget, QComboBox
)
from PyQt6.QtCore import pyqtSignal
import numpy as np


# Default storage directories
RADIONOISE_DIR = Path.home() / ".radionoise"
PROOFS_DIR = RADIONOISE_DIR / "proofs"
BACKUPS_DIR = RADIONOISE_DIR / "backups"
AUDIT_DB = RADIONOISE_DIR / "audit.db"


class TraceabilityWidget(QWidget):
    """Widget for traceability features."""

    proof_created = pyqtSignal(str)  # proof path
    backup_created = pyqtSignal(str)  # backup id

    def __init__(self):
        super().__init__()
        self._password: str = ""
        self._passwords: list[str] = []
        self._raw_iq_data: np.ndarray | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Sub-tabs for different features
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Proof tab
        proof_widget = self._create_proof_tab()
        tabs.addTab(proof_widget, "Preuves")

        # Verify tab
        verify_widget = self._create_verify_tab()
        tabs.addTab(verify_widget, "Vérification")

        # Audit tab
        audit_widget = self._create_audit_tab()
        tabs.addTab(audit_widget, "Audit Trail")

        # Backup tab
        backup_widget = self._create_backup_tab()
        tabs.addTab(backup_widget, "Backup")

    def _create_proof_tab(self) -> QWidget:
        """Create proof generation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info
        info_group = QGroupBox("Génération de preuve")
        info_layout = QVBoxLayout(info_group)

        info_text = QLabel(
            "Une preuve cryptographique permet de vérifier qu'un mot de passe "
            "a bien été généré à partir d'une capture RTL-SDR spécifique.\n\n"
            "La preuve contient les hashes de chaque étape du pipeline "
            "sans révéler le mot de passe lui-même."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)

        # Password selector
        pwd_group = QGroupBox("Mot de passe")
        pwd_layout = QHBoxLayout(pwd_group)
        pwd_layout.addWidget(QLabel("Sélection:"))
        self.password_combo = QComboBox()
        self.password_combo.currentIndexChanged.connect(self._on_password_selected)
        pwd_layout.addWidget(self.password_combo)
        layout.addWidget(pwd_group)

        # Output path
        path_group = QGroupBox("Fichier de sortie")
        path_layout = QHBoxLayout(path_group)

        # Default to ~/.radionoise/proofs/ with timestamp
        default_proof = PROOFS_DIR / f"proof_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
        self.proof_path_edit = QLineEdit(str(default_proof))
        path_layout.addWidget(self.proof_path_edit)

        browse_btn = QPushButton("Parcourir...")
        browse_btn.clicked.connect(self._browse_proof_path)
        path_layout.addWidget(browse_btn)

        layout.addWidget(path_group)

        # Status
        self.proof_status = QLabel("En attente d'un mot de passe généré...")
        self.proof_status.setStyleSheet("color: gray;")
        layout.addWidget(self.proof_status)

        # Generate button
        self.create_proof_btn = QPushButton("Créer la preuve")
        self.create_proof_btn.setMinimumHeight(40)
        self.create_proof_btn.setEnabled(False)
        self.create_proof_btn.clicked.connect(self._create_proof)
        layout.addWidget(self.create_proof_btn)

        layout.addStretch()
        return widget

    def _create_verify_tab(self) -> QWidget:
        """Create verification tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Load proof
        load_group = QGroupBox("Charger une preuve")
        load_layout = QHBoxLayout(load_group)

        self.verify_path_edit = QLineEdit()
        self.verify_path_edit.setPlaceholderText("Chemin vers proof.json...")
        load_layout.addWidget(self.verify_path_edit)

        browse_btn = QPushButton("Parcourir...")
        browse_btn.clicked.connect(self._browse_verify_path)
        load_layout.addWidget(browse_btn)

        layout.addWidget(load_group)

        # Password input
        password_group = QGroupBox("Mot de passe à vérifier")
        password_layout = QHBoxLayout(password_group)

        self.verify_password_edit = QLineEdit()
        self.verify_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.verify_password_edit.setPlaceholderText("Entrez le mot de passe...")
        password_layout.addWidget(self.verify_password_edit)

        show_btn = QPushButton("Afficher")
        show_btn.setCheckable(True)
        show_btn.toggled.connect(
            lambda checked: self.verify_password_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        password_layout.addWidget(show_btn)

        layout.addWidget(password_group)

        # Verify button
        verify_btn = QPushButton("Vérifier")
        verify_btn.setMinimumHeight(40)
        verify_btn.clicked.connect(self._verify_proof)
        layout.addWidget(verify_btn)

        # Result
        self.verify_result = QTextEdit()
        self.verify_result.setReadOnly(True)
        self.verify_result.setMaximumHeight(150)
        layout.addWidget(self.verify_result)

        layout.addStretch()
        return widget

    def _create_audit_tab(self) -> QWidget:
        """Create audit trail tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Database path
        db_group = QGroupBox("Base de données d'audit")
        db_layout = QHBoxLayout(db_group)

        self.audit_db_edit = QLineEdit(str(AUDIT_DB))
        db_layout.addWidget(self.audit_db_edit)

        browse_btn = QPushButton("Parcourir...")
        browse_btn.clicked.connect(self._browse_audit_db)
        db_layout.addWidget(browse_btn)

        layout.addWidget(db_group)

        # Add to audit
        add_group = QGroupBox("Ajouter à l'audit")
        add_layout = QVBoxLayout(add_group)

        add_info = QLabel(
            "Ajoute la génération actuelle (mot de passe + preuve) à l'audit trail.\n"
            "La base sera créée automatiquement si elle n'existe pas."
        )
        add_info.setWordWrap(True)
        add_layout.addWidget(add_info)

        self.add_audit_btn = QPushButton("Ajouter la génération actuelle")
        self.add_audit_btn.setEnabled(False)
        self.add_audit_btn.clicked.connect(self._add_to_audit)
        add_layout.addWidget(self.add_audit_btn)

        layout.addWidget(add_group)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        verify_chain_btn = QPushButton("Vérifier l'intégrité de la chaîne")
        verify_chain_btn.clicked.connect(self._verify_chain)
        actions_layout.addWidget(verify_chain_btn)

        export_btn = QPushButton("Exporter le rapport d'audit")
        export_btn.clicked.connect(self._export_audit)
        actions_layout.addWidget(export_btn)

        layout.addWidget(actions_group)

        # Result
        self.audit_result = QTextEdit()
        self.audit_result.setReadOnly(True)
        layout.addWidget(self.audit_result)

        layout.addStretch()
        return widget

    def _create_backup_tab(self) -> QWidget:
        """Create backup tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info
        info_group = QGroupBox("Backup chiffré")
        info_layout = QVBoxLayout(info_group)

        info_text = QLabel(
            "Crée un backup chiffré (AES-256-GCM) des données IQ brutes "
            "permettant de régénérer le mot de passe.\n\n"
            "ATTENTION: Le master password est CRITIQUE. Sans lui, "
            "la récupération est impossible."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)

        # Backup directory
        dir_group = QGroupBox("Répertoire de backup")
        dir_layout = QHBoxLayout(dir_group)

        self.backup_dir_edit = QLineEdit(str(BACKUPS_DIR))
        dir_layout.addWidget(self.backup_dir_edit)

        browse_btn = QPushButton("Parcourir...")
        browse_btn.clicked.connect(self._browse_backup_dir)
        dir_layout.addWidget(browse_btn)

        layout.addWidget(dir_group)

        # Master password
        password_group = QGroupBox("Master Password")
        password_layout = QVBoxLayout(password_group)

        self.master_password_edit = QLineEdit()
        self.master_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.master_password_edit.setPlaceholderText("Master password...")
        password_layout.addWidget(self.master_password_edit)

        self.master_password_confirm = QLineEdit()
        self.master_password_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self.master_password_confirm.setPlaceholderText("Confirmer...")
        password_layout.addWidget(self.master_password_confirm)

        layout.addWidget(password_group)

        # Buttons
        button_row = QHBoxLayout()

        self.create_backup_btn = QPushButton("Créer le backup")
        self.create_backup_btn.setEnabled(False)
        self.create_backup_btn.clicked.connect(self._create_backup)
        button_row.addWidget(self.create_backup_btn)

        recover_btn = QPushButton("Récupérer...")
        recover_btn.clicked.connect(self._recover_backup)
        button_row.addWidget(recover_btn)

        layout.addLayout(button_row)

        # Status
        self.backup_status = QLabel("")
        layout.addWidget(self.backup_status)

        layout.addStretch()
        return widget

    def set_password(self, password: str, raw_iq_data: np.ndarray | None,
                     charset: str = "safe"):
        """Set a single generated password for proof/backup."""
        self.set_passwords([password] if password else [], raw_iq_data, charset)

    def set_passwords(self, passwords: list[str], raw_iq_data: np.ndarray | None,
                      charset: str = "safe"):
        """Set multiple generated passwords for proof/backup."""
        self._passwords = passwords
        self._raw_iq_data = raw_iq_data
        self._password_charset = charset

        # Update password selector combo
        self.password_combo.blockSignals(True)
        self.password_combo.clear()
        for i, pwd in enumerate(passwords):
            # Masked display: "Pwd 1 (T***...3!)"
            if len(pwd) > 4:
                masked = f"{pwd[0]}{'*' * min(len(pwd) - 3, 6)}...{pwd[-2:]}"
            else:
                masked = "****"
            self.password_combo.addItem(f"Pwd {i+1} ({masked})")
        self.password_combo.blockSignals(False)

        if passwords:
            self._password = passwords[0]
        else:
            self._password = ""

        self._update_buttons_state()

    def _on_password_selected(self, index: int):
        """Handle password selection from combo."""
        if 0 <= index < len(self._passwords):
            self._password = self._passwords[index]
            self._update_buttons_state()

    def _update_buttons_state(self):
        """Update button enabled state based on current data."""
        has_password = bool(self._password)
        has_iq = self._raw_iq_data is not None

        self.create_proof_btn.setEnabled(has_password and has_iq)
        self.create_backup_btn.setEnabled(has_password and has_iq)
        self.add_audit_btn.setEnabled(has_password)

        if has_password:
            if has_iq:
                self.proof_status.setText("Prêt à créer une preuve")
                self.proof_status.setStyleSheet("color: green;")
            else:
                self.proof_status.setText(
                    "Données IQ non disponibles (source non-RTL-SDR)"
                )
                self.proof_status.setStyleSheet("color: orange;")
        else:
            self.proof_status.setText("En attente d'un mot de passe généré...")
            self.proof_status.setStyleSheet("color: gray;")

    def _browse_proof_path(self):
        """Browse for proof output path."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder la preuve", str(PROOFS_DIR / "proof.json"),
            "JSON (*.json)"
        )
        if path:
            self.proof_path_edit.setText(path)

    def _browse_verify_path(self):
        """Browse for proof file to verify."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une preuve", "",
            "JSON (*.json)"
        )
        if path:
            self.verify_path_edit.setText(path)

    def _browse_audit_db(self):
        """Browse for audit database."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Base d'audit", str(AUDIT_DB),
            "SQLite (*.db)"
        )
        if path:
            self.audit_db_edit.setText(path)

    def _browse_backup_dir(self):
        """Browse for backup directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Répertoire de backup"
        )
        if path:
            self.backup_dir_edit.setText(path)

    def _create_proof(self):
        """Create cryptographic proof."""
        if not self._password or self._raw_iq_data is None:
            return

        try:
            import hashlib
            import json
            from datetime import datetime
            from radionoise import von_neumann_extract, hash_entropy

            timestamp = datetime.utcnow().isoformat() + 'Z'
            capture_hash = hashlib.sha256(self._raw_iq_data.tobytes()).hexdigest()

            entropy = von_neumann_extract(self._raw_iq_data)
            entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()

            processed = hash_entropy(entropy)
            processed_hash = hashlib.sha256(processed.tobytes()).hexdigest()

            password_hash = hashlib.sha256(self._password.encode()).hexdigest()

            signature_data = timestamp + capture_hash + entropy_hash + password_hash
            signature = hashlib.sha256(signature_data.encode()).hexdigest()

            proof = {
                "timestamp": timestamp,
                "capture_hash": capture_hash,
                "capture_size": len(self._raw_iq_data),
                "entropy_hash": entropy_hash,
                "processed_hash": processed_hash,
                "password_hash": password_hash,
                "signature": signature,
            }

            path = Path(self.proof_path_edit.text())
            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                json.dump(proof, f, indent=2)

            self.proof_status.setText(f"Preuve créée: {path}")
            self.proof_status.setStyleSheet("color: green;")

            self.proof_created.emit(str(path))

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur: {e}")

    def _verify_proof(self):
        """Verify a proof file using ProofOfGeneration.verify_proof()."""
        path = self.verify_path_edit.text()
        password = self.verify_password_edit.text()

        if not path or not password:
            self.verify_result.setText("Veuillez fournir le fichier et le mot de passe.")
            return

        try:
            import json
            from radionoise.traceability.proof import ProofOfGeneration

            with open(path) as f:
                proof = json.load(f)

            pog = ProofOfGeneration()
            valid, msg = pog.verify_proof(password, proof)

            if valid:
                self.verify_result.setStyleSheet("color: green;")
                self.verify_result.setText(
                    "VALIDE\n\n"
                    f"Timestamp: {proof.get('timestamp')}\n"
                    f"Signature: {proof.get('signature', '')[:32]}...\n"
                    f"Vérification: hash mot de passe + signature OK"
                )
            else:
                self.verify_result.setStyleSheet("color: red;")
                self.verify_result.setText(
                    f"INVALIDE\n\n{msg}"
                )

        except FileNotFoundError:
            self.verify_result.setText(f"Fichier non trouvé: {path}")
        except json.JSONDecodeError:
            self.verify_result.setText("Fichier JSON invalide")
        except Exception as e:
            self.verify_result.setText(f"Erreur: {e}")

    def _add_to_audit(self):
        """Add current generation to audit trail."""
        if not self._password:
            return

        try:
            import hashlib
            from datetime import timezone
            from radionoise.traceability.audit import ForensicAuditTrail

            db_path = Path(self.audit_db_edit.text())
            # Create parent directory if needed
            db_path.parent.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).isoformat()
            password_hash = hashlib.sha256(self._password.encode()).hexdigest()

            # Create proof data
            if self._raw_iq_data is not None:
                from radionoise import von_neumann_extract, hash_entropy
                capture_hash = hashlib.sha256(self._raw_iq_data.tobytes()).hexdigest()
                entropy = von_neumann_extract(self._raw_iq_data)
                entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()
            else:
                capture_hash = "N/A (source non-RTL-SDR)"
                entropy_hash = "N/A"

            signature = hashlib.sha256(
                (timestamp + capture_hash + entropy_hash + password_hash).encode()
            ).hexdigest()

            proof = {
                "timestamp": timestamp,
                "capture_hash": capture_hash,
                "entropy_hash": entropy_hash,
                "password_hash": password_hash,
                "signature": signature,
                "metadata": {
                    "frequency": 100000000,  # Default 100 MHz
                    "sample_rate": 2400000,
                    "samples": len(self._raw_iq_data) if self._raw_iq_data is not None else 0,
                }
            }

            # Add to audit trail
            audit = ForensicAuditTrail(str(db_path))
            gen_id, chain_hash = audit.add_generation(proof)
            audit.close()

            self.audit_result.setStyleSheet("color: green;")
            self.audit_result.setText(
                f"Ajouté à l'audit trail!\n\n"
                f"Base: {db_path}\n"
                f"ID génération: {gen_id}\n"
                f"Hash chaîne: {chain_hash[:32]}..."
            )

            QMessageBox.information(
                self,
                "Audit ajouté",
                f"Génération ajoutée à l'audit trail.\nID: {gen_id}"
            )

        except Exception as e:
            self.audit_result.setStyleSheet("color: red;")
            self.audit_result.setText(f"Erreur: {e}")
            QMessageBox.critical(self, "Erreur", f"Impossible d'ajouter à l'audit:\n{e}")

    def _verify_chain(self):
        """Verify audit chain integrity."""
        import os
        db_path = self.audit_db_edit.text()

        if not os.path.exists(db_path):
            self.audit_result.setStyleSheet("color: orange;")
            self.audit_result.setText(
                f"Base non trouvée: {db_path}\n\n"
                "L'audit trail n'existe pas encore.\n"
                "Utilisez 'Ajouter la génération actuelle' pour créer la base."
            )
            return

        try:
            from radionoise.traceability.audit import ForensicAuditTrail

            audit = ForensicAuditTrail(db_path)
            valid, errors = audit.verify_chain_integrity()
            audit.close()

            if valid:
                self.audit_result.setStyleSheet("color: green;")
                self.audit_result.setText("Intégrité de la chaîne: OK")
            else:
                self.audit_result.setStyleSheet("color: red;")
                self.audit_result.setText(
                    "CORRUPTED\n\n" + "\n".join(errors)
                )

        except Exception as e:
            self.audit_result.setText(f"Erreur: {e}")

    def _export_audit(self):
        """Export audit report."""
        import os
        db_path = self.audit_db_edit.text()

        if not os.path.exists(db_path):
            self.audit_result.setStyleSheet("color: orange;")
            self.audit_result.setText(
                f"Base non trouvée: {db_path}\n\n"
                "L'audit trail n'existe pas encore.\n"
                "Utilisez 'Ajouter la génération actuelle' pour créer la base."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter le rapport", "audit_report.json",
            "JSON (*.json)"
        )
        if not path:
            return

        try:
            from radionoise.traceability.audit import ForensicAuditTrail

            audit = ForensicAuditTrail(db_path)
            audit.export_audit_report(path)
            audit.close()

            self.audit_result.setText(f"Rapport exporté: {path}")

        except Exception as e:
            self.audit_result.setText(f"Erreur: {e}")

    def _create_backup(self):
        """Create encrypted backup."""
        if not self._password or self._raw_iq_data is None:
            return

        master = self.master_password_edit.text()
        confirm = self.master_password_confirm.text()

        if not master:
            self.backup_status.setText("Master password requis")
            return

        if master != confirm:
            self.backup_status.setText("Les mots de passe ne correspondent pas")
            return

        try:
            from radionoise.traceability.backup import SecureBackupSystem
            import hashlib
            from datetime import datetime
            from radionoise import von_neumann_extract, hash_entropy

            # Create proof
            timestamp = datetime.utcnow().isoformat() + 'Z'
            capture_hash = hashlib.sha256(self._raw_iq_data.tobytes()).hexdigest()
            entropy = von_neumann_extract(self._raw_iq_data)
            entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()
            password_hash = hashlib.sha256(self._password.encode()).hexdigest()
            signature = hashlib.sha256(
                (timestamp + capture_hash + entropy_hash + password_hash).encode()
            ).hexdigest()

            proof = {
                "timestamp": timestamp,
                "capture_hash": capture_hash,
                "entropy_hash": entropy_hash,
                "password_hash": password_hash,
                "signature": signature,
                "password_length": len(self._password),
                "charset": self._password_charset if hasattr(self, '_password_charset') else "safe",
            }

            backup = SecureBackupSystem(self.backup_dir_edit.text())
            backup_id = backup.backup_password(
                self._password, proof, self._raw_iq_data, master
            )

            self.backup_status.setText(f"Backup créé: {backup_id}")
            self.backup_status.setStyleSheet("color: green;")

            self.backup_created.emit(backup_id)

        except Exception as e:
            self.backup_status.setText(f"Erreur: {e}")
            self.backup_status.setStyleSheet("color: red;")

    def _recover_backup(self):
        """Recover password from backup."""
        from radionoise.traceability.backup import SecureBackupSystem
        import os

        backup_dir = self.backup_dir_edit.text()
        if not os.path.exists(backup_dir):
            QMessageBox.warning(self, "Erreur", f"Répertoire non trouvé: {backup_dir}")
            return

        # List available backups
        backup = SecureBackupSystem(backup_dir)
        backups_dict = backup.get_backup_list()

        if not backups_dict:
            QMessageBox.information(self, "Info", "Aucun backup trouvé dans ce répertoire")
            return

        # Create list of backup IDs with timestamps for display
        backup_items = []
        for backup_id, meta in backups_dict.items():
            timestamp = meta.get('timestamp', 'N/A')[:19]  # Truncate to datetime
            backup_items.append(f"{backup_id} ({timestamp})")

        from PyQt6.QtWidgets import QInputDialog

        selected, ok = QInputDialog.getItem(
            self, "Sélectionner un backup",
            "Backup:", backup_items, 0, False
        )
        if not ok:
            return

        # Extract backup_id from selection
        backup_id = selected.split(" (")[0]

        master = self.master_password_edit.text()
        if not master:
            QMessageBox.warning(self, "Erreur", "Master password requis")
            return

        try:
            password, proof = backup.recover_password(backup_id, master)
            if password:
                QMessageBox.information(
                    self, "Récupération réussie",
                    f"Mot de passe récupéré:\n\n{password}"
                )
            else:
                QMessageBox.warning(self, "Échec", "Récupération échouée")

        except ValueError as e:
            QMessageBox.critical(self, "Erreur", str(e))

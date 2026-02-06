"""
Security-focused widgets for sensitive data display.
"""

import hashlib

from PyQt6.QtWidgets import (
    QLineEdit, QWidget, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QApplication
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont


class SecurePasswordDisplay(QLineEdit):
    """
    Password display with security features:
    - Hidden by default
    - Temporary reveal with auto-hide
    - Secure clipboard handling
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setEchoMode(QLineEdit.EchoMode.Password)
        self.setFont(QFont("Monospace", 12))

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._auto_hide)

        self._reveal_duration = 5000  # 5 seconds default
        self._clipboard_hash = None  # Hash of copied password

    def set_reveal_duration(self, ms: int):
        """Set auto-hide duration in milliseconds."""
        self._reveal_duration = ms

    def reveal_temporarily(self, duration_ms: int = None):
        """Reveal password for a limited time."""
        if duration_ms is None:
            duration_ms = self._reveal_duration

        self.setEchoMode(QLineEdit.EchoMode.Normal)
        self._timer.start(duration_ms)

    def _auto_hide(self):
        """Auto-hide the password."""
        self.setEchoMode(QLineEdit.EchoMode.Password)

    def hide_now(self):
        """Immediately hide the password."""
        self._timer.stop()
        self.setEchoMode(QLineEdit.EchoMode.Password)

    def copy_to_clipboard(self, clear_after_ms: int = 30000):
        """Copy to clipboard with auto-clear."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text())

        # Store hash of copied password for comparison (avoid plaintext in memory)
        self._clipboard_hash = hashlib.sha256(self.text().encode()).hexdigest()

        # Auto-clear clipboard after delay
        QTimer.singleShot(clear_after_ms, lambda: self._clear_clipboard_if_same())

    def _clear_clipboard_if_same(self):
        """Clear clipboard if it still contains our password (hash-based comparison)."""
        clipboard = QApplication.clipboard()
        current_hash = hashlib.sha256(clipboard.text().encode()).hexdigest()
        if self._clipboard_hash and current_hash == self._clipboard_hash:
            clipboard.clear()
            self._clipboard_hash = None

    def secure_clear(self):
        """Securely clear the password from widget."""
        text = self.text()
        # Overwrite with asterisks before clearing
        self.setText('*' * len(text))
        self.clear()


class SecurePasswordInput(QWidget):
    """
    Password input widget with reveal button.
    """

    text_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._line_edit = QLineEdit()
        self._line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._line_edit.textChanged.connect(self.text_changed.emit)
        layout.addWidget(self._line_edit)

        self._reveal_btn = QPushButton("Afficher")
        self._reveal_btn.setCheckable(True)
        self._reveal_btn.setFixedWidth(80)
        self._reveal_btn.toggled.connect(self._toggle_reveal)
        layout.addWidget(self._reveal_btn)

    def _toggle_reveal(self, checked: bool):
        if checked:
            self._line_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._reveal_btn.setText("Masquer")
        else:
            self._line_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._reveal_btn.setText("Afficher")

    def text(self) -> str:
        return self._line_edit.text()

    def setText(self, text: str):
        self._line_edit.setText(text)

    def setPlaceholderText(self, text: str):
        self._line_edit.setPlaceholderText(text)

    def clear(self):
        self._line_edit.clear()

    def secure_clear(self):
        """Securely clear the input."""
        text = self._line_edit.text()
        self._line_edit.setText('*' * len(text))
        self._line_edit.clear()


class SecurePasswordList(QListWidget):
    """
    List widget for displaying multiple passwords securely.
    """

    password_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Monospace", 11))
        self._passwords: list[str] = []
        self._revealed = False

        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def add_password(self, password: str, entropy_bits: float = 0):
        """Add a password to the list."""
        self._passwords.append(password)

        if self._revealed:
            display = password
        else:
            display = '*' * len(password)

        if entropy_bits > 0:
            display += f"  ({entropy_bits:.0f} bits)"

        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, len(self._passwords) - 1)
        self.addItem(item)

    def reveal_all(self):
        """Reveal all passwords."""
        self._revealed = True
        self._update_display()

    def hide_all(self):
        """Hide all passwords."""
        self._revealed = False
        self._update_display()

    def _update_display(self):
        """Update display based on reveal state."""
        for i in range(self.count()):
            item = self.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is not None and idx < len(self._passwords):
                password = self._passwords[idx]
                if self._revealed:
                    display = password
                else:
                    display = '*' * len(password)
                # Preserve entropy info if present
                text = item.text()
                if "(" in text:
                    entropy_part = text[text.rfind("("):]
                    display += f"  {entropy_part}"
                item.setText(display)

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is not None and idx < len(self._passwords):
            self.password_selected.emit(self._passwords[idx])

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click to copy."""
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is not None and idx < len(self._passwords):
            clipboard = QApplication.clipboard()
            clipboard.setText(self._passwords[idx])

    def get_passwords(self) -> list[str]:
        """Get all passwords."""
        return self._passwords.copy()

    def get_selected_password(self) -> str | None:
        """Get currently selected password."""
        item = self.currentItem()
        if item:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is not None and idx < len(self._passwords):
                return self._passwords[idx]
        return None

    def secure_clear(self):
        """Securely clear all passwords."""
        # Overwrite passwords in memory
        for i, pwd in enumerate(self._passwords):
            self._passwords[i] = '*' * len(pwd)
        self._passwords.clear()
        self.clear()


def secure_clear_widget(widget):
    """
    Securely clear sensitive data from a widget.
    """
    if hasattr(widget, 'secure_clear'):
        widget.secure_clear()
    elif isinstance(widget, QLineEdit):
        text = widget.text()
        widget.setText('*' * len(text))
        widget.clear()
    elif isinstance(widget, QListWidget):
        widget.clear()

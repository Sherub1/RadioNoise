"""
Dark theme for RadioNoise GUI - Security-focused design.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


class DarkTheme:
    """Dark theme color definitions."""

    # Background colors
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#1f3460"

    # Accent colors
    ACCENT_PRIMARY = "#0f3460"
    ACCENT_SUCCESS = "#2ecc71"
    ACCENT_WARNING = "#f39c12"
    ACCENT_DANGER = "#e74c3c"
    ACCENT_INFO = "#3498db"

    # Text colors
    TEXT_PRIMARY = "#ecf0f1"
    TEXT_SECONDARY = "#bdc3c7"
    TEXT_MUTED = "#7f8c8d"

    # Border colors
    BORDER = "#34495e"
    BORDER_FOCUS = "#3498db"

    @classmethod
    def get_stylesheet(cls) -> str:
        """Get the complete stylesheet."""
        return f"""
            /* Main window */
            QMainWindow {{
                background-color: {cls.BG_DARK};
            }}

            /* Central widget */
            QWidget {{
                background-color: {cls.BG_DARK};
                color: {cls.TEXT_PRIMARY};
                font-family: "Segoe UI", "Ubuntu", sans-serif;
            }}

            /* Group boxes */
            QGroupBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {cls.TEXT_PRIMARY};
            }}

            /* Tabs */
            QTabWidget::pane {{
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                background-color: {cls.BG_MEDIUM};
            }}

            QTabBar::tab {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_SECONDARY};
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}

            QTabBar::tab:selected {{
                background-color: {cls.ACCENT_PRIMARY};
                color: {cls.TEXT_PRIMARY};
            }}

            QTabBar::tab:hover:!selected {{
                background-color: {cls.BG_MEDIUM};
            }}

            /* Buttons */
            QPushButton {{
                background-color: {cls.ACCENT_PRIMARY};
                color: {cls.TEXT_PRIMARY};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background-color: {cls.ACCENT_INFO};
            }}

            QPushButton:pressed {{
                background-color: {cls.BG_LIGHT};
            }}

            QPushButton:disabled {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_MUTED};
            }}

            /* Success button variant */
            QPushButton[variant="success"] {{
                background-color: {cls.ACCENT_SUCCESS};
            }}

            /* Danger button variant */
            QPushButton[variant="danger"] {{
                background-color: {cls.ACCENT_DANGER};
            }}

            /* Input fields */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                padding: 6px;
            }}

            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {cls.BORDER_FOCUS};
            }}

            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
            }}

            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {cls.TEXT_SECONDARY};
            }}

            /* Text areas */
            QTextEdit {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
            }}

            /* Lists */
            QListWidget {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
            }}

            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {cls.BORDER};
            }}

            QListWidget::item:selected {{
                background-color: {cls.ACCENT_PRIMARY};
            }}

            /* Tables */
            QTableWidget {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                gridline-color: {cls.BORDER};
            }}

            QTableWidget::item {{
                padding: 5px;
            }}

            QHeaderView::section {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                padding: 8px;
                border: none;
                border-bottom: 1px solid {cls.BORDER};
                font-weight: bold;
            }}

            /* Progress bar */
            QProgressBar {{
                background-color: {cls.BG_LIGHT};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                text-align: center;
                color: {cls.TEXT_PRIMARY};
            }}

            QProgressBar::chunk {{
                background-color: {cls.ACCENT_SUCCESS};
                border-radius: 3px;
            }}

            /* Checkboxes */
            QCheckBox {{
                spacing: 8px;
            }}

            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {cls.BORDER};
                border-radius: 3px;
                background-color: {cls.BG_LIGHT};
            }}

            QCheckBox::indicator:checked {{
                background-color: {cls.ACCENT_SUCCESS};
                border-color: {cls.ACCENT_SUCCESS};
            }}

            /* Scrollbars */
            QScrollBar:vertical {{
                background-color: {cls.BG_DARK};
                width: 12px;
                border-radius: 6px;
            }}

            QScrollBar::handle:vertical {{
                background-color: {cls.BG_LIGHT};
                border-radius: 6px;
                min-height: 20px;
            }}

            QScrollBar::handle:vertical:hover {{
                background-color: {cls.BORDER};
            }}

            /* Status bar */
            QStatusBar {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_SECONDARY};
            }}

            /* Menu bar */
            QMenuBar {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
            }}

            QMenuBar::item:selected {{
                background-color: {cls.ACCENT_PRIMARY};
            }}

            QMenu {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
            }}

            QMenu::item:selected {{
                background-color: {cls.ACCENT_PRIMARY};
            }}

            /* Labels */
            QLabel {{
                color: {cls.TEXT_PRIMARY};
            }}

            QLabel[type="header"] {{
                font-size: 24px;
                font-weight: bold;
            }}

            QLabel[type="subtitle"] {{
                color: {cls.TEXT_SECONDARY};
            }}

            QLabel[type="success"] {{
                color: {cls.ACCENT_SUCCESS};
            }}

            QLabel[type="warning"] {{
                color: {cls.ACCENT_WARNING};
            }}

            QLabel[type="danger"] {{
                color: {cls.ACCENT_DANGER};
            }}

            /* Tooltips */
            QToolTip {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                padding: 5px;
            }}
        """


def apply_dark_theme(app: QApplication):
    """Apply dark theme to the application."""
    app.setStyle("Fusion")

    # Set palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(DarkTheme.BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(DarkTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(DarkTheme.BG_LIGHT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(DarkTheme.BG_MEDIUM))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(DarkTheme.BG_MEDIUM))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(DarkTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(DarkTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(DarkTheme.ACCENT_PRIMARY))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(DarkTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(DarkTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Link, QColor(DarkTheme.ACCENT_INFO))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(DarkTheme.ACCENT_PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(DarkTheme.TEXT_PRIMARY))

    app.setPalette(palette)
    app.setStyleSheet(DarkTheme.get_stylesheet())

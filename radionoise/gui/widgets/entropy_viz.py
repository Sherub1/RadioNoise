# -*- coding: utf-8 -*-
"""
Entropy visualization widget using pure QPainter.

Displays a byte histogram (256 bars), statistics, and quality indicator.
No matplotlib dependency.
"""

import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QFontDatabase


class EntropyVizWidget(QWidget):
    """Custom widget for entropy visualization."""

    def __init__(self):
        super().__init__()
        self._data: np.ndarray | None = None
        self._histogram: np.ndarray | None = None
        self._stats: dict = {}
        self.setMinimumHeight(200)
        self.setMinimumWidth(300)

    def set_data(self, data: np.ndarray):
        """Set entropy data and compute statistics."""
        self._data = data
        self._histogram = np.bincount(data, minlength=256).astype(float)

        bits = np.unpackbits(data)
        ones = int(np.sum(bits))
        total_bits = len(bits)

        self._stats = {
            "bytes": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "ones_ratio": ones / total_bits if total_bits > 0 else 0,
        }
        self.update()

    def paintEvent(self, event):
        """Draw histogram and stats."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        if self._histogram is None or self._data is None:
            painter.setPen(QColor(128, 128, 128))
            placeholder_font = QFont()
            placeholder_font.setPointSize(10)
            painter.setFont(placeholder_font)
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                             "En attente d'entropie...")
            painter.end()
            return

        # Layout
        margin = 10
        stats_height = 50
        hist_left = margin
        hist_top = margin
        hist_width = w - 2 * margin
        hist_height = h - stats_height - 2 * margin

        # Draw histogram
        max_val = float(np.max(self._histogram)) if np.max(self._histogram) > 0 else 1.0
        bar_width = hist_width / 256.0

        for i in range(256):
            bar_height = (self._histogram[i] / max_val) * hist_height
            x = hist_left + i * bar_width
            y = hist_top + hist_height - bar_height

            # Color based on deviation from expected
            expected = len(self._data) / 256.0
            ratio = self._histogram[i] / expected if expected > 0 else 1.0
            if 0.8 <= ratio <= 1.2:
                color = QColor(100, 200, 100)  # Green - good
            elif 0.6 <= ratio <= 1.4:
                color = QColor(200, 200, 100)  # Yellow - ok
            else:
                color = QColor(200, 100, 100)  # Red - bad

            painter.fillRect(QRectF(x, y, max(bar_width - 0.5, 0.5), bar_height), color)

        # Draw stats
        stats_y = hist_top + hist_height + 5
        stats_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        stats_font.setPointSize(9)
        painter.setFont(stats_font)

        mean = self._stats["mean"]
        std = self._stats["std"]
        ones_ratio = self._stats["ones_ratio"]

        # Quality indicator
        quality_color, quality_text = self._assess_quality()
        painter.setPen(quality_color)
        painter.drawText(QRectF(margin, stats_y, hist_width / 3, stats_height),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"Qualité: {quality_text}")

        painter.setPen(QColor(200, 200, 200))
        painter.drawText(QRectF(margin + hist_width / 3, stats_y, hist_width / 3, stats_height),
                         Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter,
                         f"μ={mean:.1f}  σ={std:.1f}")

        painter.drawText(QRectF(margin + 2 * hist_width / 3, stats_y, hist_width / 3, stats_height),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f"1s={ones_ratio:.3f}  n={self._stats['bytes']:,}")

        painter.end()

    def _assess_quality(self) -> tuple:
        """Assess entropy quality from statistics."""
        if not self._stats:
            return QColor(128, 128, 128), "N/A"

        mean = self._stats["mean"]
        std = self._stats["std"]
        ones_ratio = self._stats["ones_ratio"]

        # Expected: mean~127.5, std~73.9, ones~0.5
        mean_ok = abs(mean - 127.5) < 15
        std_ok = abs(std - 73.9) < 15
        ones_ok = abs(ones_ratio - 0.5) < 0.05

        if mean_ok and std_ok and ones_ok:
            return QColor(100, 255, 100), "Excellent"
        elif mean_ok and ones_ok:
            return QColor(255, 200, 50), "Acceptable"
        else:
            return QColor(255, 80, 80), "Faible"

# -*- coding: utf-8 -*-
"""
QThread worker for NIST SP 800-22 tests.
"""

from PyQt6.QtCore import pyqtSignal
import numpy as np

from radionoise.gui.workers.base_worker import BaseWorker


class NistWorker(BaseWorker):
    """Worker thread for NIST statistical tests."""

    # Signals
    progress = pyqtSignal(str, int)  # (test_name, percent)
    test_complete = pyqtSignal(dict)  # Single test result
    result_ready = pyqtSignal(dict)  # Full results

    def __init__(self, entropy: np.ndarray, fast_mode: bool = True):
        super().__init__()
        self.entropy = entropy
        self.fast_mode = fast_mode

    def run(self):
        try:
            with self._suppress_stdout():
                from radionoise import NISTTests

                self.progress.emit("Initialisation...", 0)

                results = NISTTests.run_all_tests(
                    self.entropy,
                    verbose=False,
                    fast_mode=self.fast_mode
                )

                if self.is_cancelled:
                    return

                self.progress.emit("Terminé", 100)
                self.result_ready.emit(results)

        except Exception as e:
            self.error.emit(str(e))

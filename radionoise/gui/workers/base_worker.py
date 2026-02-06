# -*- coding: utf-8 -*-
"""
Base worker class for QThread workers with thread-safe cancellation.
"""

import contextlib
import io
import threading

from PyQt6.QtCore import QThread, pyqtSignal


class BaseWorker(QThread):
    """
    Base class for worker threads.

    Provides:
    - Thread-safe cancellation via threading.Event
    - stdout suppression via contextlib.redirect_stdout
    """

    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cancel_event = threading.Event()

    def cancel(self):
        """Request cancellation (thread-safe)."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_event.is_set()

    @contextlib.contextmanager
    def _suppress_stdout(self):
        """Suppress stdout from core library print() calls."""
        with contextlib.redirect_stdout(io.StringIO()):
            yield

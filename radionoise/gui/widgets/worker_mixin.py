# -*- coding: utf-8 -*-
"""
Mixin for widgets that manage a QThread worker.
"""

from radionoise.gui.workers.base_worker import BaseWorker


class WorkerWidgetMixin:
    """
    Mixin providing _cleanup_worker() for widgets that own a BaseWorker.

    Expects the host widget to have a `_worker` attribute.
    """

    _worker: BaseWorker | None

    def _cleanup_worker(self) -> None:
        """Safely clean up the worker thread."""
        if self._worker:
            self._worker.quit()
            self._worker.wait()
            self._worker = None

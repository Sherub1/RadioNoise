# -*- coding: utf-8 -*-
"""
QThread worker for RTL-SDR entropy capture.
"""

from PyQt6.QtCore import pyqtSignal
import numpy as np

from radionoise.gui.workers.base_worker import BaseWorker


class CaptureWorker(BaseWorker):
    """Worker thread for entropy capture with progress reporting."""

    # Signals
    progress = pyqtSignal(str, int)  # (status_message, percent)
    status = pyqtSignal(str)  # Status update
    result_ready = pyqtSignal(object, str, object)  # (entropy, source, raw_iq)

    def __init__(self, samples: int = 500000, frequency: float = 100e6,
                 allow_fallback: bool = True, use_rdseed: bool = False,
                 capture_raw: bool = False):
        super().__init__()
        self.samples = samples
        self.frequency = frequency
        self.allow_fallback = allow_fallback
        self.use_rdseed = use_rdseed
        self.capture_raw = capture_raw

    def run(self):
        try:
            with self._suppress_stdout():
                from radionoise import capture_entropy, get_last_entropy_source

                self.status.emit("Initialisation de la capture...")
                self.progress.emit("Initialisation...", 0)

                raw_iq_data = None

                # If we need raw IQ for proof/backup
                if self.capture_raw:
                    from radionoise.core.entropy import (
                        capture_entropy_raw,
                        is_rtl_sdr_available,
                        von_neumann_extract,
                        hash_entropy
                    )

                    if is_rtl_sdr_available():
                        self.status.emit("Capture RTL-SDR en cours...")
                        self.progress.emit("Capture RTL-SDR...", 20)

                        raw_iq_data = capture_entropy_raw(
                            samples=self.samples,
                            frequency=self.frequency
                        )

                        if self.is_cancelled:
                            return

                        self.progress.emit("Extraction Von Neumann...", 60)
                        extracted = von_neumann_extract(raw_iq_data)

                        self.progress.emit("Hachage SHA-512...", 80)
                        entropy = hash_entropy(extracted)
                    else:
                        self.status.emit("RTL-SDR indisponible, fallback...")
                        entropy = capture_entropy(
                            samples=self.samples,
                            frequency=self.frequency,
                            allow_fallback=self.allow_fallback,
                            use_rdseed=self.use_rdseed
                        )
                else:
                    self.status.emit("Capture d'entropie...")
                    self.progress.emit("Capture...", 30)

                    entropy = capture_entropy(
                        samples=self.samples,
                        frequency=self.frequency,
                        allow_fallback=self.allow_fallback,
                        use_rdseed=self.use_rdseed
                    )

                if self.is_cancelled:
                    return

                self.progress.emit("Terminé", 100)
                source = get_last_entropy_source()
                self.result_ready.emit(entropy, source, raw_iq_data)

        except Exception as e:
            self.error.emit(str(e))

# -*- coding: utf-8 -*-
"""
QThread worker for password generation.
"""

from PyQt6.QtCore import pyqtSignal
import numpy as np

from radionoise.gui.workers.base_worker import BaseWorker


class GeneratorWorker(BaseWorker):
    """Worker thread for password/passphrase generation."""

    # Signals
    progress = pyqtSignal(int, int)  # (current, total)
    password_generated = pyqtSignal(str, float)  # (password, entropy_bits)
    result_ready = pyqtSignal(list)  # All passwords

    def __init__(self, entropy: np.ndarray, count: int = 5, length: int = 16,
                 charset: str = "safe", passphrase: bool = False):
        super().__init__()
        self.entropy = entropy
        self.count = count
        self.length = length
        self.charset = charset
        self.passphrase = passphrase

    def run(self):
        try:
            from radionoise import generate_password, generate_passphrase
            from radionoise.core.generator import (
                calculate_password_entropy,
                calculate_passphrase_entropy
            )

            passwords = []
            offset = 0

            for i in range(self.count):
                if self.is_cancelled:
                    return

                self.progress.emit(i + 1, self.count)

                if self.passphrase:
                    # EFF wordlist uses rejection sampling: ~11.9% acceptance rate
                    # Need ~17 bytes/word average, use 20 for safety margin
                    chunk_size = self.length * 20
                    if offset + chunk_size > len(self.entropy):
                        break

                    chunk = self.entropy[offset:offset + chunk_size]
                    offset += chunk_size

                    password = generate_passphrase(chunk, words=self.length)
                    entropy_bits = calculate_passphrase_entropy(self.length)
                else:
                    chunk_size = self.length * 3
                    if offset + chunk_size > len(self.entropy):
                        break

                    chunk = self.entropy[offset:offset + chunk_size]
                    offset += chunk_size

                    password = generate_password(
                        chunk, length=self.length, charset=self.charset
                    )
                    entropy_bits = calculate_password_entropy(self.length, self.charset)

                passwords.append(password)
                self.password_generated.emit(password, entropy_bits)

            self.result_ready.emit(passwords)

        except Exception as e:
            self.error.emit(str(e))

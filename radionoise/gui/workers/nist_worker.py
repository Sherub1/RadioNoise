# -*- coding: utf-8 -*-
"""
QThread worker for NIST SP 800-22 tests with granular per-test progress.
"""

from PyQt6.QtCore import pyqtSignal
import numpy as np

from radionoise.gui.workers.base_worker import BaseWorker


class NistWorker(BaseWorker):
    """Worker thread for NIST statistical tests with per-test reporting."""

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
                from radionoise.core.nist import NISTTests

                self.progress.emit("Initialisation...", 0)

                bits = NISTTests.bytes_to_bits(self.entropy)

                tests_to_run = [
                    ("Frequency (Monobit)", lambda: NISTTests.frequency_monobit_test(bits)),
                    ("Block Frequency", lambda: NISTTests.frequency_block_test(bits)),
                    ("Runs", lambda: NISTTests.runs_test(bits)),
                    ("Longest Run", lambda: NISTTests.longest_run_test(bits)),
                    ("Spectral (DFT)", lambda: NISTTests.spectral_test(bits)),
                    ("Serial (m=2)", lambda: NISTTests.serial_test(bits, 2)),
                    ("Approximate Entropy", lambda: NISTTests.approximate_entropy_test(bits, 2)),
                    ("Cumulative Sums (forward)", lambda: NISTTests.cumulative_sums_test(bits, 'forward')),
                    ("Cumulative Sums (backward)", lambda: NISTTests.cumulative_sums_test(bits, 'backward')),
                ]

                if not self.fast_mode:
                    tests_to_run.extend([
                        ("Binary Matrix Rank", lambda: NISTTests.binary_matrix_rank_test(bits)),
                        ("Non-overlapping Template", lambda: NISTTests.non_overlapping_template_test(bits)),
                        ("Overlapping Template", lambda: NISTTests.overlapping_template_test(bits)),
                        ("Maurer's Universal", lambda: NISTTests.maurers_universal_test(bits)),
                        ("Linear Complexity", lambda: NISTTests.linear_complexity_test(bits)),
                        ("Random Excursions", lambda: NISTTests.random_excursions_test(bits)),
                        ("Random Excursions Variant", lambda: NISTTests.random_excursions_variant_test(bits)),
                    ])

                total = len(tests_to_run)
                results = []

                for i, (name, fn) in enumerate(tests_to_run):
                    if self.is_cancelled:
                        return

                    self.progress.emit(name, int((i / total) * 100))
                    result = fn()

                    if result is not None:
                        results.append(result)
                        self.test_complete.emit(result)

                if self.is_cancelled:
                    return

                passed = sum(1 for t in results if t['passed'])
                full_results = {
                    'tests': results,
                    'passed': passed,
                    'total': len(results),
                    'pass_rate': passed / len(results) if results else 0,
                }

                self.progress.emit("Terminé", 100)
                self.result_ready.emit(full_results)

        except Exception as e:
            self.error.emit(str(e))

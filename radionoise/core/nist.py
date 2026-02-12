"""
RadioNoise NIST - NIST SP 800-22 statistical tests for randomness validation.
"""

import numpy as np
from scipy import stats, special
from typing import Optional, Dict, Any

from radionoise.core.log import get_logger

logger = get_logger('nist')


def _count_patterns_vectorized(bits: np.ndarray, m: int) -> np.ndarray:
    """Count occurrences of all patterns of length m in a vectorized manner."""
    n = len(bits)
    if m == 0:
        return np.array([n])

    extended = np.concatenate([bits, bits[:m - 1]])
    windows = np.lib.stride_tricks.sliding_window_view(extended, m)[:n]
    powers = 2 ** np.arange(m - 1, -1, -1, dtype=np.int64)
    patterns = windows @ powers
    counts = np.bincount(patterns, minlength=2 ** m)

    return counts


def _find_longest_runs_vectorized(blocks: np.ndarray) -> np.ndarray:
    """Find the longest run of 1s in each block in a vectorized manner."""
    num_blocks, block_size = blocks.shape
    blocks_signed = blocks.astype(np.int8)
    padded = np.pad(blocks_signed, ((0, 0), (1, 1)), constant_values=0)
    diff = np.diff(padded, axis=1)

    max_runs = np.zeros(num_blocks, dtype=np.int64)

    for i in range(num_blocks):
        starts = np.where(diff[i] == 1)[0]
        ends = np.where(diff[i] == -1)[0]

        if len(starts) > 0 and len(ends) > 0:
            run_lengths = ends - starts
            max_runs[i] = np.max(run_lengths)

    return max_runs


class NISTTests:
    """Simplified NIST statistical tests for entropy validation."""

    @staticmethod
    def bytes_to_bits(data):
        """Convert bytes to bit array."""
        return np.unpackbits(data)

    @staticmethod
    def frequency_monobit_test(bits):
        """Test 1: Frequency (Monobit) Test"""
        n = len(bits)
        ones_count = np.sum(bits, dtype=np.int64)
        s = 2 * ones_count - n
        s_obs = abs(s) / np.sqrt(n)
        p_value = special.erfc(s_obs / np.sqrt(2))

        return {
            'name': 'Frequency (Monobit)',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': s_obs
        }

    @staticmethod
    def frequency_block_test(bits, block_size=128):
        """Test 2: Frequency Test within a Block"""
        n = len(bits)
        num_blocks = n // block_size

        if num_blocks < 1:
            return None

        blocks = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
        proportions = np.mean(blocks, axis=1)
        chi_squared = 4 * block_size * np.sum((proportions - 0.5) ** 2)
        p_value = special.gammaincc(num_blocks / 2, chi_squared / 2)

        return {
            'name': 'Block Frequency',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': chi_squared
        }

    @staticmethod
    def runs_test(bits):
        """Test 3: Runs Test"""
        n = len(bits)
        proportion = np.mean(bits)

        if abs(proportion - 0.5) >= 2 / np.sqrt(n):
            return {
                'name': 'Runs',
                'p_value': 0.0,
                'passed': False,
                'statistic': None,
                'note': 'Pre-test failed: proportion too far from 0.5'
            }

        runs = 1 + np.sum(bits[:-1] != bits[1:])
        expected_runs = 2 * n * proportion * (1 - proportion) + 1
        variance = 2 * n * proportion * (1 - proportion) * (2 * n * proportion * (1 - proportion) - 1) / (n - 1)

        if variance == 0:
            return {
                'name': 'Runs',
                'p_value': 0.0,
                'passed': False,
                'statistic': None
            }

        v_obs = (runs - expected_runs) / np.sqrt(variance)
        p_value = special.erfc(abs(v_obs) / np.sqrt(2))

        return {
            'name': 'Runs',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': v_obs,
            'runs': runs,
            'expected': expected_runs
        }

    @staticmethod
    def longest_run_test(bits: np.ndarray) -> Optional[Dict[str, Any]]:
        """Test 4: Test for the Longest Run of Ones"""
        n = len(bits)

        if n < 128:
            return None

        if n < 6272:
            M, K = 8, 3
            v_values = [1, 2, 3, 4]
            pi = [0.2148, 0.3672, 0.2305, 0.1875]
        elif n < 750000:
            M, K = 128, 5
            v_values = [4, 5, 6, 7, 8, 9]
            pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        else:
            M, K = 10000, 6
            v_values = [10, 11, 12, 13, 14, 15, 16]
            pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

        num_blocks = n // M
        blocks = bits[:num_blocks * M].reshape(num_blocks, M)
        max_runs = _find_longest_runs_vectorized(blocks)

        frequencies = np.zeros(len(v_values))
        for i, v in enumerate(v_values[:-1]):
            frequencies[i] = np.sum(max_runs == v)
        frequencies[-1] = np.sum(max_runs >= v_values[-1])

        chi_squared = np.sum(((frequencies - num_blocks * np.array(pi)) ** 2) / (num_blocks * np.array(pi)))
        p_value = special.gammaincc((K - 1) / 2, chi_squared / 2)

        return {
            'name': 'Longest Run',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': chi_squared
        }

    @staticmethod
    def spectral_test(bits):
        """Test DFT: Discrete Fourier Transform (Spectral) Test"""
        n = len(bits)
        x = 2 * bits.astype(float) - 1
        fft = np.fft.fft(x)
        modulus = np.abs(fft[:n//2])
        threshold = np.sqrt(np.log(1/0.05) * n)
        n0 = 0.95 * n / 2
        n1 = np.sum(modulus < threshold)
        d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4)
        p_value = special.erfc(abs(d) / np.sqrt(2))

        return {
            'name': 'Spectral (DFT)',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': d,
            'peaks': n1,
            'expected': n0
        }

    @staticmethod
    def serial_test(bits: np.ndarray, pattern_length: int = 2) -> Optional[Dict[str, Any]]:
        """Test Serial: checks bit independence"""
        n = len(bits)

        if n < pattern_length ** 3:
            return None

        def psi_squared(m: int) -> float:
            if m == 0:
                return 0.0
            counts = _count_patterns_vectorized(bits, m)
            num_patterns = 2 ** m
            psi = np.sum(counts.astype(np.float64) ** 2)
            psi = (psi * num_patterns / n) - n
            return psi

        psi2_m = psi_squared(pattern_length)
        psi2_m1 = psi_squared(pattern_length - 1)
        psi2_m2 = psi_squared(pattern_length - 2) if pattern_length > 1 else 0

        delta1 = psi2_m - psi2_m1
        delta2 = psi2_m - 2 * psi2_m1 + psi2_m2

        p_value1 = special.gammaincc(2 ** (pattern_length - 2), delta1 / 2)
        p_value2 = special.gammaincc(2 ** (pattern_length - 3), delta2 / 2)

        return {
            'name': f'Serial (m={pattern_length})',
            'p_value': min(p_value1, p_value2),
            'passed': min(p_value1, p_value2) >= 0.01,
            'p_value1': p_value1,
            'p_value2': p_value2
        }

    @staticmethod
    def approximate_entropy_test(bits: np.ndarray, pattern_length: int = 2) -> Optional[Dict[str, Any]]:
        """Test Approximate Entropy"""
        n = len(bits)

        if n < pattern_length ** 2:
            return None

        def phi(m: int) -> float:
            counts = _count_patterns_vectorized(bits, m)
            nonzero_counts = counts[counts > 0].astype(np.float64)
            proportions = nonzero_counts / n
            phi_m = np.sum(proportions * np.log(proportions))
            return phi_m

        phi_m = phi(pattern_length)
        phi_m1 = phi(pattern_length + 1)

        apen = phi_m - phi_m1
        chi_squared = 2 * n * (np.log(2) - apen)
        p_value = special.gammaincc(2 ** (pattern_length - 1), chi_squared / 2)

        return {
            'name': f'Approximate Entropy (m={pattern_length})',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': chi_squared,
            'apen': apen
        }

    @staticmethod
    def binary_matrix_rank_test(bits, M=32, Q=32):
        """Test 5: Binary Matrix Rank Test - OPTIMIZED: reduces M,Q if low data"""
        n = len(bits)
        num_matrices = n // (M * Q)

        # OPTIMIZATION: Reduce matrix size if low data
        if num_matrices < 38:
            # Try with smaller matrices
            M, Q = 16, 16
            num_matrices = n // (M * Q)
            if num_matrices < 38:
                return None

        def compute_rank(matrix):
            m = matrix.copy().astype(np.int8)
            rows, cols = m.shape
            rank = 0

            for col in range(min(rows, cols)):
                pivot_row = None
                for row in range(rank, rows):
                    if m[row, col] == 1:
                        pivot_row = row
                        break

                if pivot_row is None:
                    continue

                m[[rank, pivot_row]] = m[[pivot_row, rank]]

                for row in range(rows):
                    if row != rank and m[row, col] == 1:
                        m[row] = (m[row] + m[rank]) % 2

                rank += 1

            return rank

        # OPTIMIZATION: Limit number of tested matrices if too many
        max_matrices = min(num_matrices, 100)  # Max 100 matrices for performance
        ranks = []
        for i in range(max_matrices):
            start = i * M * Q
            matrix = bits[start:start + M * Q].reshape(M, Q)
            ranks.append(compute_rank(matrix))

        full_rank = sum(1 for r in ranks if r == M)
        rank_m1 = sum(1 for r in ranks if r == M - 1)
        other = max_matrices - full_rank - rank_m1

        # Theoretical probabilities for M=Q (16 or 32)
        if M == 16:
            p_full = 0.2888
            p_m1 = 0.5776
            p_other = 0.1336
        else:
            p_full = 0.2888
            p_m1 = 0.5776
            p_other = 0.1336

        chi_squared = ((full_rank - max_matrices * p_full) ** 2 / (max_matrices * p_full) +
                       (rank_m1 - max_matrices * p_m1) ** 2 / (max_matrices * p_m1) +
                       (other - max_matrices * p_other) ** 2 / (max_matrices * p_other))

        p_value = np.exp(-chi_squared / 2)

        return {
            'name': 'Binary Matrix Rank',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': chi_squared,
            'full_rank': full_rank,
            'rank_m1': rank_m1
        }

    @staticmethod
    def non_overlapping_template_test(bits, template_length=9):
        """Test 7: Non-overlapping Template Matching Test"""
        n = len(bits)
        m = template_length

        template = np.ones(m, dtype=np.uint8)

        M = 1032
        N = n // M

        if N < 8:
            return None

        counts = []
        for i in range(N):
            block = bits[i * M:(i + 1) * M]
            count = 0
            j = 0
            while j <= M - m:
                if np.array_equal(block[j:j + m], template):
                    count += 1
                    j += m
                else:
                    j += 1
            counts.append(count)

        mu = (M - m + 1) / (2 ** m)
        sigma_sq = M * (1 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))

        chi_squared = sum((c - mu) ** 2 / sigma_sq for c in counts)
        p_value = special.gammaincc(N / 2, chi_squared / 2)

        return {
            'name': 'Non-overlapping Template',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': chi_squared,
            'mean_count': np.mean(counts)
        }

    @staticmethod
    def overlapping_template_test(bits, template_length=9):
        """Test 8: Overlapping Template Matching Test"""
        n = len(bits)
        m = template_length

        template = np.ones(m, dtype=np.uint8)

        M = 1032
        N = n // M

        if N < 8:
            return None

        pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
        K = 5

        v = np.zeros(K + 1)
        for i in range(N):
            block = bits[i * M:(i + 1) * M]
            count = 0
            for j in range(M - m + 1):
                if np.array_equal(block[j:j + m], template):
                    count += 1
            if count >= K:
                v[K] += 1
            else:
                v[count] += 1

        chi_squared = 0.0
        for i in range(K + 1):
            expected = N * pi[i]
            if expected > 0:
                chi_squared += (v[i] - expected) ** 2 / expected

        p_value = special.gammaincc(K / 2, chi_squared / 2)

        return {
            'name': 'Overlapping Template',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': chi_squared
        }

    @staticmethod
    def maurers_universal_test(bits, L=7, Q=1280):
        """Test 9: Maurer's "Universal Statistical" Test"""
        n = len(bits)

        expected_values = {
            6: (5.2177052, 2.954),
            7: (6.1962507, 3.125),
            8: (7.1836656, 3.238),
        }

        # OPTIMIZATION: Use L=6 if low data (faster)
        if n < 100000:
            L = 6
            Q = 640

        if L not in expected_values:
            return None

        expected_value, variance = expected_values[L]

        K = n // L - Q
        if K < 1000:
            return None

        table = np.zeros(2 ** L, dtype=np.int64)

        for i in range(Q):
            block_val = 0
            for j in range(L):
                block_val = (block_val << 1) | bits[i * L + j]
            table[block_val] = i + 1

        total = 0.0
        for i in range(Q, Q + K):
            block_val = 0
            for j in range(L):
                block_val = (block_val << 1) | bits[i * L + j]
            total += np.log2(i + 1 - table[block_val])
            table[block_val] = i + 1

        fn = total / K

        c = 0.7 - 0.8 / L + (4 + 32 / L) * (K ** (-3 / L)) / 15
        sigma = c * np.sqrt(variance / K)

        p_value = special.erfc(abs(fn - expected_value) / (np.sqrt(2) * sigma))

        return {
            'name': "Maurer's Universal",
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': fn,
            'expected': expected_value
        }

    @staticmethod
    def linear_complexity_test(bits, M=500):
        """Test 10: Linear Complexity Test - OPTIMIZED: reduces M if needed"""
        n = len(bits)

        # OPTIMIZATION: Reduce M for speed (fewer blocks = less Berlekamp-Massey)
        if n < 100000:
            M = 200  # Smaller blocks = less computation

        N = n // M

        if N < 200:
            return None

        def berlekamp_massey(sequence):
            n = len(sequence)
            c = np.zeros(n, dtype=np.int8)
            b = np.zeros(n, dtype=np.int8)
            c[0] = 1
            b[0] = 1
            L = 0
            m = -1

            for i in range(n):
                d = sequence[i]
                for j in range(1, L + 1):
                    d ^= c[j] & sequence[i - j]

                if d == 1:
                    t = c.copy()
                    for j in range(n - i + m):
                        c[i - m + j] ^= b[j]

                    if L <= i // 2:
                        L = i + 1 - L
                        m = i
                        b = t

            return L

        # OPTIMIZATION: Limit number of tested blocks
        max_blocks = min(N, 200)  # Max 200 blocks
        complexities = []
        for i in range(max_blocks):
            block = bits[i * M:(i + 1) * M]
            L_i = berlekamp_massey(block)
            complexities.append(L_i)

        mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - (M / 3 + 2 / 9) / (2 ** M)

        K = 6
        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

        v = np.zeros(K + 1)
        for L_i in complexities:
            T = (-1) ** M * (L_i - mu) + 2 / 9
            if T <= -2.5:
                v[0] += 1
            elif T <= -1.5:
                v[1] += 1
            elif T <= -0.5:
                v[2] += 1
            elif T <= 0.5:
                v[3] += 1
            elif T <= 1.5:
                v[4] += 1
            elif T <= 2.5:
                v[5] += 1
            else:
                v[6] += 1

        chi_squared = sum((v[i] - max_blocks * pi[i]) ** 2 / (max_blocks * pi[i]) for i in range(K + 1))
        p_value = special.gammaincc(K / 2, chi_squared / 2)

        return {
            'name': 'Linear Complexity',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': chi_squared,
            'mean_complexity': np.mean(complexities)
        }

    @staticmethod
    def cumulative_sums_test(bits, mode='forward'):
        """Test 13: Cumulative Sums (Cusum) Test"""
        n = len(bits)

        x = 2 * bits.astype(np.int64) - 1

        if mode == 'backward':
            x = x[::-1]

        S = np.cumsum(x)
        z = max(abs(S))

        sum1 = 0.0
        sum2 = 0.0

        for k in range(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1):
            sum1 += (stats.norm.cdf((4 * k + 1) * z / np.sqrt(n)) -
                     stats.norm.cdf((4 * k - 1) * z / np.sqrt(n)))
            sum2 += (stats.norm.cdf((4 * k + 3) * z / np.sqrt(n)) -
                     stats.norm.cdf((4 * k + 1) * z / np.sqrt(n)))

        p_value = 1 - sum1 + sum2

        return {
            'name': f'Cumulative Sums ({mode})',
            'p_value': p_value,
            'passed': p_value >= 0.01,
            'statistic': z,
            'mode': mode
        }

    @staticmethod
    def random_excursions_test(bits):
        """Test 14: Random Excursions Test"""
        n = len(bits)

        x = 2 * bits.astype(np.int64) - 1

        S = np.concatenate([[0], np.cumsum(x)])

        zero_crossings = np.where(S == 0)[0]
        J = len(zero_crossings) - 1

        if J < 500:
            return None

        states = [-4, -3, -2, -1, 1, 2, 3, 4]

        pi_table = {
            1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
            2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
            3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
            4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0733],
        }

        results = []

        for state in states:
            abs_state = abs(state)
            pi = pi_table[abs_state]

            v = np.zeros(6)
            for i in range(J):
                start = zero_crossings[i]
                end = zero_crossings[i + 1]
                cycle = S[start:end + 1]
                count = np.sum(cycle == state)
                if count >= 5:
                    v[5] += 1
                else:
                    v[count] += 1

            chi_squared = sum((v[k] - J * pi[k]) ** 2 / (J * pi[k])
                              for k in range(6) if pi[k] > 0)
            p_value = special.gammaincc(5 / 2, chi_squared / 2)

            results.append({
                'state': state,
                'p_value': p_value,
                'passed': p_value >= 0.01
            })

        min_p = min(r['p_value'] for r in results)

        return {
            'name': 'Random Excursions',
            'p_value': min_p,
            'passed': min_p >= 0.01,
            'cycles': J,
            'results': results
        }

    @staticmethod
    def random_excursions_variant_test(bits):
        """Test 15: Random Excursions Variant Test"""
        n = len(bits)

        x = 2 * bits.astype(np.int64) - 1

        S = np.concatenate([[0], np.cumsum(x)])

        J = np.sum(S == 0) - 1

        if J < 500:
            return None

        states = list(range(-9, 0)) + list(range(1, 10))

        results = []

        for state in states:
            xi = np.sum(S == state)

            p_value = special.erfc(abs(xi - J) / np.sqrt(2 * J * (4 * abs(state) - 2)))

            results.append({
                'state': state,
                'p_value': p_value,
                'passed': p_value >= 0.01,
                'visits': xi
            })

        min_p = min(r['p_value'] for r in results)

        return {
            'name': 'Random Excursions Variant',
            'p_value': min_p,
            'passed': min_p >= 0.01,
            'cycles': J,
            'results': results
        }

    @classmethod
    def run_all_tests(cls, data, verbose=True, fast_mode=False):
        """
        Run all 15 NIST SP 800-22 tests.

        Args:
            data: Entropy data as numpy uint8 array
            verbose: Print results
            fast_mode: If True, run only fast tests (1-4, 6, 11-12, 13)
                      and limit heavy tests (5, 10) to reduced samples.

        Returns:
            Dictionary with test results
        """
        bits = cls.bytes_to_bits(data)

        # Tests always executed (fast)
        tests = [
            cls.frequency_monobit_test(bits),           # Test 1
            cls.frequency_block_test(bits),             # Test 2
            cls.runs_test(bits),                        # Test 3
            cls.longest_run_test(bits),                 # Test 4
            cls.spectral_test(bits),                    # Test 6
            cls.serial_test(bits, 2),                   # Test 11
            cls.approximate_entropy_test(bits, 2),      # Test 12
            cls.cumulative_sums_test(bits, 'forward'),  # Test 13a
            cls.cumulative_sums_test(bits, 'backward'), # Test 13b
        ]

        # Heavy optional tests or with reduced samples
        if not fast_mode:
            tests.extend([
                cls.binary_matrix_rank_test(bits),
                cls.non_overlapping_template_test(bits),
                cls.overlapping_template_test(bits),
                cls.maurers_universal_test(bits),
                cls.linear_complexity_test(bits),
                cls.random_excursions_test(bits),
                cls.random_excursions_variant_test(bits),
            ])

        tests = [t for t in tests if t is not None]

        if verbose:
            mode = "FAST MODE (9 essential tests)" if fast_mode else "FULL SUITE"
            logger.info("NIST SP 800-22 TESTS - %s", mode)
            logger.info("Data: %s bytes (%s bits)", f"{len(data):,}", f"{len(bits):,}")
            logger.info("p-value threshold: 0.01 (99%% confidence level)")

            passed = 0
            total = len(tests)

            for test in tests:
                status = "PASS" if test['passed'] else "FAIL"
                logger.info("%s  %-30s p-value: %.6f", status, test['name'], test['p_value'])
                if test['passed']:
                    passed += 1

            logger.info("Result: %d/%d tests passed (%.1f%%)", passed, total, passed/total*100)

            if fast_mode:
                logger.info("Fast mode: Heavy tests skipped. Use --full-test for complete suite.")

            if passed == total:
                logger.info("Entropy of CRYPTOGRAPHIC QUALITY")
            elif passed >= total * 0.95:
                logger.info("Acceptable entropy with minor weaknesses")
            else:
                logger.warning("LOW QUALITY entropy - NOT recommended for crypto")

        return {
            'tests': tests,
            'passed': sum(1 for t in tests if t['passed']),
            'total': len(tests),
            'pass_rate': sum(1 for t in tests if t['passed']) / len(tests) if tests else 0
        }

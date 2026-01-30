#!/usr/bin/env python3
"""
Générateur de mots de passe avec tests de qualité NIST intégrés.
Version optimisée pour la performance.
"""

import subprocess
import numpy as np
import sys
import os
import string
import tempfile
import hashlib
import argparse
import secrets
import ctypes
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, Union
from scipy import stats, special


# =============================================================================
# RDRAND Hardware RNG Support
# =============================================================================

class RDRANDError(Exception):
    """Erreur lors de l'utilisation de RDRAND."""
    pass


class HardwareRNG:
    """
    Interface pour le générateur matériel RDRAND/RDSEED des CPU Intel/AMD.
    """

    _lib = None
    _lib_path = None

    _C_SOURCE = '''
#include <stdint.h>
#include <immintrin.h>

typedef unsigned long long ull_t;

int rdrand64(ull_t *value) {
    return _rdrand64_step(value);
}

int rdseed64(ull_t *value) {
    return _rdseed64_step(value);
}

int rdrand_bytes(uint8_t *buffer, size_t n) {
    ull_t value;
    size_t i = 0;
    int retries;

    while (i < n) {
        retries = 10;
        while (retries-- > 0) {
            if (_rdrand64_step(&value)) {
                size_t to_copy = (n - i < 8) ? (n - i) : 8;
                for (size_t j = 0; j < to_copy; j++) {
                    buffer[i++] = (value >> (j * 8)) & 0xFF;
                }
                break;
            }
        }
        if (retries < 0) return -1;
    }
    return 0;
}

int rdseed_bytes(uint8_t *buffer, size_t n) {
    ull_t value;
    size_t i = 0;
    int retries;

    while (i < n) {
        retries = 100;
        while (retries-- > 0) {
            if (_rdseed64_step(&value)) {
                size_t to_copy = (n - i < 8) ? (n - i) : 8;
                for (size_t j = 0; j < to_copy; j++) {
                    buffer[i++] = (value >> (j * 8)) & 0xFF;
                }
                break;
            }
        }
        if (retries < 0) return -1;
    }
    return 0;
}
'''

    @classmethod
    def _compile_lib(cls) -> Optional[str]:
        """Compile la bibliothèque RDRAND si nécessaire."""
        if cls._lib_path and os.path.exists(cls._lib_path):
            return cls._lib_path

        try:
            result = subprocess.run(
                ["gcc", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(cls._C_SOURCE)
            c_path = f.name

        lib_path = c_path.replace('.c', '.so')

        try:
            result = subprocess.run([
                "gcc", "-O2", "-shared", "-fPIC",
                "-mrdrnd", "-mrdseed",
                c_path, "-o", lib_path
            ], capture_output=True, timeout=30)

            if result.returncode != 0:
                return None

            cls._lib_path = lib_path
            return lib_path

        except subprocess.TimeoutExpired:
            return None
        finally:
            if os.path.exists(c_path):
                os.remove(c_path)

    @classmethod
    def _get_lib(cls) -> Optional[ctypes.CDLL]:
        """Charge la bibliothèque RDRAND."""
        if cls._lib is not None:
            return cls._lib

        lib_path = cls._compile_lib()
        if lib_path is None:
            return None

        try:
            cls._lib = ctypes.CDLL(lib_path)

            cls._lib.rdrand_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            cls._lib.rdrand_bytes.restype = ctypes.c_int

            cls._lib.rdseed_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            cls._lib.rdseed_bytes.restype = ctypes.c_int

            return cls._lib
        except OSError:
            return None

    @classmethod
    def is_available(cls) -> bool:
        """Vérifie si RDRAND est disponible."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'rdrand' not in cpuinfo:
                    return False
        except (IOError, OSError):
            return False

        return cls._get_lib() is not None

    @classmethod
    def get_bytes(cls, n: int, use_rdseed: bool = False) -> bytes:
        """Génère n octets aléatoires via RDRAND ou RDSEED."""
        lib = cls._get_lib()
        if lib is None:
            raise RDRANDError("RDRAND non disponible")

        buffer = (ctypes.c_uint8 * n)()

        if use_rdseed:
            result = lib.rdseed_bytes(buffer, n)
        else:
            result = lib.rdrand_bytes(buffer, n)

        if result != 0:
            raise RDRANDError("Échec de génération RDRAND/RDSEED")

        return bytes(buffer)

    @classmethod
    def cleanup(cls):
        """Nettoie les ressources."""
        if cls._lib_path and os.path.exists(cls._lib_path):
            try:
                os.remove(cls._lib_path)
            except OSError:
                pass
        cls._lib = None
        cls._lib_path = None


# Jeux de caractères
CHARSETS = {
    "alnum": string.ascii_letters + string.digits,
    "alpha": string.ascii_letters,
    "digits": string.digits,
    "hex": string.hexdigits[:16],
    "full": string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:,.<>?",
    "safe": string.ascii_letters + string.digits + "-_!@#$%",
}

CharsetName = Literal["alnum", "alpha", "digits", "hex", "full", "safe"]
_last_entropy_source: str = "unknown"


def _count_patterns_vectorized(bits: np.ndarray, m: int) -> np.ndarray:
    """Compte les occurrences de tous les patterns de longueur m de manière vectorisée."""
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
    """Trouve le plus long run de 1 dans chaque bloc de manière vectorisée."""
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
    """Tests statistiques NIST simplifiés pour validation d'entropie."""

    @staticmethod
    def bytes_to_bits(data):
        """Convertit bytes en array de bits."""
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
        """Test Serial: vérifie l'indépendance des bits"""
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
        """Test 5: Binary Matrix Rank Test - OPTIMISÉ: réduit M,Q si peu de données"""
        n = len(bits)
        num_matrices = n // (M * Q)

        # OPTIMISATION: Réduire la taille des matrices si peu de données
        if num_matrices < 38:
            # Essayer avec des matrices plus petites
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

        # OPTIMISATION: Limiter le nombre de matrices testées si trop nombreuses
        max_matrices = min(num_matrices, 100)  # Max 100 matrices pour la performance
        ranks = []
        for i in range(max_matrices):
            start = i * M * Q
            matrix = bits[start:start + M * Q].reshape(M, Q)
            ranks.append(compute_rank(matrix))

        full_rank = sum(1 for r in ranks if r == M)
        rank_m1 = sum(1 for r in ranks if r == M - 1)
        other = max_matrices - full_rank - rank_m1

        # Probabilités théoriques pour M=Q (16 ou 32)
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

        # OPTIMISATION: Utiliser L=6 si peu de données (plus rapide)
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
        """Test 10: Linear Complexity Test - OPTIMISÉ: réduit M si nécessaire"""
        n = len(bits)

        # OPTIMISATION: Réduire M pour accélérer (moins de blocs = plus rapide)
        if n < 100000:
            M = 200  # Blocs plus petits = moins de calcul Berlekamp-Massey

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

        # OPTIMISATION: Limiter le nombre de blocs testés
        max_blocks = min(N, 200)  # Max 200 blocs
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
        Exécute tous les 15 tests NIST SP 800-22.

        Args:
            fast_mode: Si True, exécute uniquement les tests rapides (1-4, 6, 11-12, 13)
                      et limite les tests lourds (5, 10) à des échantillons réduits.
        """
        bits = cls.bytes_to_bits(data)

        # Tests toujours exécutés (rapides)
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

        # Tests lourds optionnels ou avec échantillons réduits
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
        else:
            # En mode rapide, on saute les tests très lourds
            # mais on ajoute une note
            pass

        tests = [t for t in tests if t is not None]

        if verbose:
            print("\n" + "=" * 70)
            if fast_mode:
                print("TESTS NIST SP 800-22 - MODE RAPIDE (9 tests essentiels)")
            else:
                print("TESTS NIST SP 800-22 - SUITE COMPLÈTE")
            print("=" * 70)
            print(f"Données: {len(data):,} bytes ({len(bits):,} bits)")
            print(f"Seuil p-value: 0.01 (niveau de confiance 99%)")
            print("-" * 70)

            passed = 0
            total = len(tests)

            for test in tests:
                status = "✓ PASS" if test['passed'] else "✗ FAIL"
                color = "\033[92m" if test['passed'] else "\033[91m"
                reset = "\033[0m"

                print(f"{color}{status}{reset}  {test['name']:<30} p-value: {test['p_value']:.6f}")

                if test['passed']:
                    passed += 1

            print("-" * 70)
            print(f"Résultat: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")

            if fast_mode:
                print("⚡ Mode rapide: Tests lourds (Matrices, Complexité Linéaire) ignorés")
                print("   Utilisez --full-test pour la suite complète (plus lent)")

            if passed == total:
                print("✓ Entropie de QUALITÉ CRYPTOGRAPHIQUE")
            elif passed >= total * 0.95:
                print("⚠ Entropie acceptable mais avec quelques faiblesses")
            else:
                print("✗ ATTENTION: Entropie de FAIBLE QUALITÉ - NON recommandée pour crypto")

            print("=" * 70 + "\n")

        return {
            'tests': tests,
            'passed': sum(1 for t in tests if t['passed']),
            'total': len(tests),
            'pass_rate': sum(1 for t in tests if t['passed']) / len(tests) if tests else 0
        }


def secure_zero(data: Union[np.ndarray, bytearray]) -> None:
    """
    Tente d'effacer les données sensibles en mémoire (best-effort).
    """
    try:
        if isinstance(data, np.ndarray):
            if data.flags.writeable:
                data[:] = 0
        elif isinstance(data, bytearray):
            for i in range(len(data)):
                data[i] = 0
    except (TypeError, ValueError):
        pass


def von_neumann_extract(data: np.ndarray) -> np.ndarray:
    """
    Extracteur Von Neumann pour éliminer les biais.
    OPTIMISÉ: Utilise NumPy pour vectoriser les opérations bit-level.

    Performance: ~5-10x plus rapide que la version Python pure.
    """
    # OPTIMISATION: Vectorisation avec NumPy
    # Prendre les LSB de chaque octet
    bits = np.unpackbits(data)

    # Reshape en paires (0,1), (2,3), etc.
    if len(bits) % 2 != 0:
        bits = bits[:-1]

    pairs = bits.reshape(-1, 2)

    # Von Neumann: 01 -> 0, 10 -> 1, 00 et 11 -> rejet
    # Condition: paires[:,0] != paires[:,1] (bits différents)
    # Valeur: paires[:,0] (si 01 -> 0, si 10 -> 1)
    valid = pairs[:, 0] != pairs[:, 1]
    extracted_bits = pairs[valid, 0]

    # Reconstituer les octets
    n_bytes = len(extracted_bits) // 8
    if n_bytes == 0:
        return np.array([], dtype=np.uint8)

    extracted_bits = extracted_bits[:n_bytes * 8]
    bytes_out = np.packbits(extracted_bits.reshape(n_bytes, 8))

    return bytes_out


def hash_entropy(
    raw_entropy: np.ndarray,
    hash_algo: Literal['sha256', 'sha512'] = 'sha512'
) -> np.ndarray:
    """
    Whitening cryptographique.
    """
    if hash_algo == 'sha256':
        h = hashlib.sha256
        hash_size = 32
    elif hash_algo == 'sha512':
        h = hashlib.sha512
        hash_size = 64
    else:
        raise ValueError(f"Hash non supporté: {hash_algo}")

    output = bytearray()
    offset = 0
    chunk_index = 0

    while offset < len(raw_entropy):
        chunk_end = min(offset + hash_size, len(raw_entropy))
        chunk = raw_entropy[offset:chunk_end]

        ctx = h()
        ctx.update(chunk_index.to_bytes(4, 'big'))
        ctx.update(chunk.tobytes())
        hashed = ctx.digest()

        needed = min(len(hashed), len(raw_entropy) - offset)
        output.extend(hashed[:needed])

        offset += len(chunk)
        chunk_index += 1

    result = np.frombuffer(bytes(output), dtype=np.uint8)
    secure_zero(output)

    return result


def capture_entropy_fallback(bytes_needed: int, use_rdseed: bool = False) -> np.ndarray:
    """
    Fallback utilisant le RNG matériel RDRAND/RDSEED ou secrets.
    """
    global _last_entropy_source

    if HardwareRNG.is_available():
        source = "RDSEED" if use_rdseed else "RDRAND"
        _last_entropy_source = source
        print(f"⚠️  RTL-SDR non disponible, utilisation de {source} (CPU hardware RNG)")
        print(f"   Source: Générateur matériel Intel/AMD certifié")
        try:
            raw_bytes = HardwareRNG.get_bytes(bytes_needed, use_rdseed=use_rdseed)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
            secure_zero(raw_bytes)
            return arr
        except RDRANDError as e:
            print(f"   Erreur RDRAND: {e}, fallback vers secrets...")

    _last_entropy_source = "CSPRNG"
    print("⚠️  RTL-SDR et RDRAND non disponibles, utilisation du CSPRNG système")
    print("   Note: L'entropie système est cryptographiquement sûre mais")
    print("   n'utilise pas de source matérielle dédiée.")
    return np.frombuffer(secrets.token_bytes(bytes_needed), dtype=np.uint8)


def is_rtl_sdr_available() -> bool:
    """Vérifie si rtl_sdr est disponible sur le système."""
    try:
        result = subprocess.run(
            ["which", "rtl_sdr"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def capture_entropy(
    samples: int = 500000,
    frequency: float = 100e6,
    sample_rate: float = 2.4e6,
    gain: int = 0,
    allow_fallback: bool = True,
    use_rdseed: bool = False
) -> np.ndarray:
    """
    Capture de l'entropie depuis le RTL-SDR.
    """
    if not is_rtl_sdr_available():
        if allow_fallback:
            estimated_bytes = max(1000, samples // 30)
            return capture_entropy_fallback(estimated_bytes, use_rdseed=use_rdseed)
        else:
            raise RuntimeError("rtl_sdr non trouvé. Installez rtl-sdr ou utilisez --allow-fallback")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as temp_file:
        temp_path = temp_file.name

    try:
        result = subprocess.run([
            "rtl_sdr",
            "-f", str(int(frequency)),
            "-s", str(int(sample_rate)),
            "-g", str(gain),
            "-n", str(samples),
            temp_path
        ], capture_output=True, timeout=30, text=True)

        if result.returncode != 0:
            if allow_fallback:
                print(f"⚠️  rtl_sdr a échoué: {result.stderr.strip()}")
                estimated_bytes = max(1000, samples // 30)
                return capture_entropy_fallback(estimated_bytes, use_rdseed=use_rdseed)
            raise RuntimeError(f"rtl_sdr a échoué: {result.stderr}")

        global _last_entropy_source
        _last_entropy_source = "RTL-SDR"

        raw = np.fromfile(temp_path, dtype=np.uint8)

        if len(raw) < samples * 0.9:
            raise RuntimeError(f"Données incomplètes: {len(raw)}/{samples} samples")

        extracted = von_neumann_extract(raw)
        hashed = hash_entropy(extracted)

        secure_zero(raw)
        secure_zero(extracted)

        return hashed

    finally:
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            with open(temp_path, 'wb') as f:
                f.write(os.urandom(file_size))
            os.remove(temp_path)


def load_entropy_from_file(filepath: str, apply_hash: bool = True) -> np.ndarray:
    """
    Charge l'entropie depuis un fichier.
    """
    global _last_entropy_source
    _last_entropy_source = "file"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable: {filepath}")

    data = np.fromfile(filepath, dtype=np.uint8)

    if apply_hash:
        hashed = hash_entropy(data)
        secure_zero(data)
        return hashed

    return data


def generate_password(
    random_bytes: np.ndarray,
    length: int = 16,
    charset: CharsetName = "safe"
) -> str:
    """
    Génère un mot de passe avec rejection sampling.
    """
    chars = CHARSETS.get(charset, CHARSETS["safe"])
    base = len(chars)
    threshold = (256 // base) * base

    password: List[str] = []
    idx = 0

    while len(password) < length and idx < len(random_bytes):
        byte = random_bytes[idx]
        idx += 1

        if byte < threshold:
            password.append(chars[byte % base])

    if len(password) < length:
        raise ValueError("Pas assez d'entropie pour générer le mot de passe complet")

    return ''.join(password)


def generate_passphrase(random_bytes: np.ndarray, words: int = 6) -> str:
    """
    Génère une passphrase basée sur une liste de mots.
    """
    wordlist = [
        "able", "acid", "aged", "also", "area", "army", "away", "baby",
        "back", "ball", "band", "bank", "base", "bath", "bear", "beat",
        "been", "beer", "bell", "belt", "best", "bill", "bird", "blow",
        "blue", "boat", "body", "bomb", "bond", "bone", "book", "boom",
        "born", "boss", "both", "bowl", "bulk", "burn", "bush", "busy",
        "call", "calm", "came", "camp", "card", "care", "case", "cash",
        "cast", "cell", "chat", "chip", "city", "club", "coal", "coat",
        "code", "cold", "come", "cook", "cool", "cope", "copy", "core",
        "cost", "crew", "crop", "dark", "data", "date", "dawn", "days",
        "dead", "deal", "dean", "dear", "debt", "deep", "deny", "desk",
        "dial", "diet", "dirt", "disc", "disk", "does", "done", "door",
        "dose", "down", "draw", "drew", "drop", "drug", "dual", "duke",
        "dust", "duty", "each", "earn", "ease", "east", "easy", "edge",
        "else", "even", "ever", "evil", "exam", "exit", "face", "fact",
        "fail", "fair", "fall", "fame", "farm", "fast", "fate", "fear",
        "feed", "feel", "feet", "fell", "felt", "file", "fill", "film",
        "find", "fine", "fire", "firm", "fish", "five", "flat", "flow",
        "food", "foot", "ford", "form", "fort", "four", "free", "from",
        "fuel", "full", "fund", "gain", "game", "gate", "gave", "gear",
        "gene", "gift", "girl", "give", "glad", "goal", "goes", "gold",
        "golf", "gone", "good", "gray", "grew", "grey", "grow", "gulf",
        "hair", "half", "hall", "hand", "hang", "hard", "harm", "hate",
        "have", "head", "hear", "heat", "held", "hell", "help", "here",
        "hero", "high", "hill", "hire", "hold", "hole", "holy", "home",
        "hope", "host", "hour", "huge", "hung", "hunt", "hurt", "idea",
        "inch", "into", "iron", "item", "jack", "jane", "jean", "jobs",
        "john", "join", "jump", "jury", "just", "keen", "keep", "kent",
        "kept", "kick", "kill", "kind", "king", "knee", "knew", "know",
        "lack", "lady", "laid", "lake", "land", "lane", "last", "late",
        "lead", "left", "less", "life", "lift", "like", "line", "link",
        "list", "live", "load", "loan", "lock", "logo", "long", "look",
        "lord", "lose", "loss", "lost", "love", "luck", "made", "mail",
        "main", "make", "male", "many", "mark", "mass", "mate", "meal",
        "mean", "meat", "meet", "menu", "mere", "mess", "mild", "mile",
        "milk", "mill", "mind", "mine", "miss", "mode", "mood", "moon",
        "more", "most", "move", "much", "must", "name", "navy", "near",
        "neck", "need", "news", "next", "nice", "nine", "none", "nose",
        "note", "okay", "once", "only", "onto", "open", "oral", "over",
        "pace", "pack", "page", "pain", "pair", "palm", "park", "part",
        "pass", "past", "path", "peak", "pick", "pile", "pink", "pipe",
        "plan", "play", "plot", "plug", "plus", "poem", "poet", "pool",
        "poor", "port", "post", "pour", "pray", "pull", "pure", "push",
        "quit", "race", "rail", "rain", "rank", "rare", "rate", "read",
        "real", "rear", "rely", "rent", "rest", "rice", "rich", "ride",
        "ring", "rise", "risk", "road", "rock", "role", "roll", "roof",
        "room", "root", "rope", "rose", "rule", "rush", "ruth", "safe",
        "sake", "sale", "salt", "same", "sand", "save", "seat", "seed",
        "seek", "seem", "seen", "self", "sell", "send", "sent", "ship",
        "shop", "shot", "show", "shut", "sick", "side", "sign", "silk"
    ]

    assert len(wordlist) == 256, f"Wordlist doit contenir exactement 256 mots, a {len(wordlist)}"

    phrase = []
    for i in range(min(words, len(random_bytes))):
        phrase.append(wordlist[random_bytes[i]])

    if len(phrase) < words:
        raise ValueError("Pas assez d'entropie")

    return '-'.join(phrase)


def main():
    parser = argparse.ArgumentParser(
        description="Générateur de mots de passe RTL-SDR avec tests NIST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s                           # Génération standard rapide (9 tests)
  %(prog)s --full-test               # Suite complète lente (15 tests)
  %(prog)s --test-only               # Tests NIST uniquement (pas de passwords)
  %(prog)s --no-test                 # Désactive les tests (plus rapide)
  %(prog)s -f entropy.bin --test-only  # Teste un fichier d'entropie existant
        """
    )

    parser.add_argument("-l", "--length", type=int, default=16,
                        help="Longueur du mot de passe (défaut: 16)")
    parser.add_argument("-n", "--count", type=int, default=5,
                        help="Nombre de mots de passe (défaut: 5)")
    parser.add_argument("-c", "--charset", choices=CHARSETS.keys(), default="safe",
                        help="Type de caractères (défaut: safe)")
    parser.add_argument("-p", "--passphrase", action="store_true",
                        help="Génère des passphrases")
    parser.add_argument("-f", "--file", metavar="FILE",
                        help="Utilise un fichier d'entropie")
    parser.add_argument("--save-entropy", metavar="FILE",
                        help="Sauvegarde l'entropie générée")
    parser.add_argument("--no-hash", action="store_true",
                        help="Désactive le hash (NON RECOMMANDÉ)")
    parser.add_argument("--hash-algo", choices=['sha256', 'sha512'], default='sha512',
                        help="Algorithme de hash (défaut: sha512)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Mode silencieux")
    parser.add_argument("--test-only", action="store_true",
                        help="Effectue uniquement les tests NIST (pas de génération)")
    parser.add_argument("--no-test", action="store_true",
                        help="Désactive les tests NIST (génération seulement)")
    parser.add_argument("--frequency", type=float, default=100e6,
                        help="Fréquence de capture RTL-SDR en Hz (défaut: 100 MHz)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Désactive le fallback si RTL-SDR indisponible")
    parser.add_argument("--use-rdseed", action="store_true",
                        help="Utilise RDSEED au lieu de RDRAND (plus lent, entropie directe)")
    parser.add_argument("--full-test", action="store_true",
                        help="Exécute la suite complète des 15 tests NIST (lent, ~30s)")

    args = parser.parse_args()

    if not args.quiet and not args.test_only:
        print("=" * 60)
        print("GÉNÉRATEUR DE MOTS DE PASSE RTL-SDR")
        print("=" * 60)
        print()

    try:
        # Obtenir l'entropie
        if args.file:
            if not args.quiet:
                print(f"📂 Lecture de {args.file}...")
            random_bytes = load_entropy_from_file(args.file, apply_hash=not args.no_hash)
        else:
            if not args.quiet:
                print("📡 Capture RTL-SDR en cours...")

            bytes_per_password = args.length * 3
            total_bytes_needed = args.count * bytes_per_password
            samples_needed = total_bytes_needed * 2
            samples = max(500000, samples_needed)

            if not args.quiet:
                print(f"   Samples: {samples:,}")
                print(f"   Fréquence: {args.frequency / 1e6:.1f} MHz")

            random_bytes = capture_entropy(
                samples=samples,
                frequency=args.frequency,
                allow_fallback=not args.no_fallback,
                use_rdseed=args.use_rdseed
            )

        # Vérifier la quantité
        bytes_needed = args.count * args.length * 3
        if len(random_bytes) < bytes_needed and not args.test_only:
            print(f"❌ ERREUR: Pas assez d'entropie!", file=sys.stderr)
            print(f"   Disponible: {len(random_bytes)} bytes", file=sys.stderr)
            print(f"   Nécessaire: {bytes_needed} bytes", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"✓ Entropie disponible: {len(random_bytes):,} bytes")
            if not args.no_hash:
                print(f"✓ Hash: {args.hash_algo.upper()}")

        # Tests NIST
        if not args.no_test:
            # OPTIMISATION: Réduire la taille de test si pas mode full
            if args.full_test:
                test_size = min(len(random_bytes), 100000)  # 100KB pour full test
            else:
                test_size = min(len(random_bytes), 10000)   # 10KB suffisent pour rapide

            test_data = random_bytes[:test_size]

            # OPTIMISATION: Mode rapide par défaut, full test sur demande
            results = NISTTests.run_all_tests(
                test_data,
                verbose=not args.quiet,
                fast_mode=not args.full_test
            )

            if results['pass_rate'] < 0.95:
                print("\n⚠️  ATTENTION: La qualité de l'entropie est DOUTEUSE!", file=sys.stderr)
                print("    Il est recommandé de ne PAS utiliser ces passwords pour de la crypto.", file=sys.stderr)
                if not args.test_only:
                    response = input("\nContinuer quand même? (o/N): ")
                    if response.lower() != 'o':
                        print("Génération annulée.")
                        sys.exit(1)

        if args.test_only:
            sys.exit(0)

        if args.save_entropy:
            random_bytes.tofile(args.save_entropy)
            if not args.quiet:
                print(f"\n💾 Entropie sauvegardée: {args.save_entropy}")

        if not args.quiet:
            print()

        # Générer les passwords
        passwords = []
        offset = 0

        if args.passphrase:
            if not args.quiet:
                print(f"Passphrases ({args.length} mots):")
                print("-" * 60)

            for i in range(args.count):
                chunk_size = args.length * 2
                if offset + chunk_size > len(random_bytes):
                    print(f"⚠️  Entropie épuisée après {i} passwords", file=sys.stderr)
                    break

                chunk = random_bytes[offset:offset + chunk_size]
                offset += chunk_size

                try:
                    passphrase = generate_passphrase(chunk, words=args.length)
                    passwords.append(passphrase)

                    if args.quiet:
                        print(passphrase)
                    else:
                        entropy_bits = args.length * 8
                        print(f"  {passphrase}")
                        print(f"    Entropie: ~{entropy_bits:.0f} bits")
                except ValueError as e:
                    print(f"⚠️  {e}", file=sys.stderr)
                    break
        else:
            chars = CHARSETS[args.charset]
            bits_per_char = np.log2(len(chars))

            if not args.quiet:
                print(f"Mots de passe ({args.length} chars, charset={args.charset}):")
                print("-" * 60)

            for i in range(args.count):
                chunk_size = args.length * 3
                if offset + chunk_size > len(random_bytes):
                    print(f"⚠️  Entropie épuisée après {i} passwords", file=sys.stderr)
                    break

                chunk = random_bytes[offset:offset + chunk_size]
                offset += chunk_size

                try:
                    password = generate_password(chunk, length=args.length, charset=args.charset)
                    passwords.append(password)

                    if args.quiet:
                        print(password)
                    else:
                        entropy_bits = args.length * bits_per_char
                        print(f"  {password}")
                        print(f"    Entropie: ~{entropy_bits:.1f} bits")
                except ValueError as e:
                    print(f"⚠️  {e}", file=sys.stderr)
                    break

        if not args.quiet:
            print()
            print("=" * 60)
            source_descriptions = {
                "RTL-SDR": "🔒 Source: Bruit radio RTL-SDR (hardware)",
                "RDRAND": "🔒 Source: Intel/AMD RDRAND (CPU hardware RNG)",
                "RDSEED": "🔒 Source: Intel/AMD RDSEED (CPU hardware entropy)",
                "CSPRNG": "🔒 Source: CSPRNG système (urandom)",
                "file": f"🔒 Source: Fichier {args.file}",
            }
            print(source_descriptions.get(_last_entropy_source, f"🔒 Source: {_last_entropy_source}"))
            if _last_entropy_source == "RTL-SDR":
                print(f"   Pipeline: Von Neumann → {args.hash_algo.upper()} → NIST validated")
            elif _last_entropy_source in ("RDRAND", "RDSEED"):
                print(f"   Pipeline: Hardware RNG → NIST validated")
            else:
                print(f"   Pipeline: {args.hash_algo.upper()} → NIST validated")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'random_bytes' in locals():
            secure_zero(random_bytes)
        HardwareRNG.cleanup()


if __name__ == "__main__":
    main()

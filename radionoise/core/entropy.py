"""
RadioNoise Entropy - Hardware RNG and entropy capture/processing.
"""

import subprocess
import numpy as np
import os
import tempfile
import hashlib
import ctypes
import secrets
from typing import Optional, Literal

from radionoise.core.security import secure_zero
from radionoise.core.log import get_logger

logger = get_logger('entropy')


# =============================================================================
# Module state
# =============================================================================

_last_entropy_source: str = "unknown"


def get_last_entropy_source() -> str:
    """Returns the last entropy source used."""
    return _last_entropy_source


# =============================================================================
# RDRAND Hardware RNG Support
# =============================================================================

class RDRANDError(Exception):
    """Error when using RDRAND."""
    pass


class RTLSDRError(Exception):
    """Base error for RTL-SDR device issues."""
    pass


class DeviceNotFoundError(RTLSDRError):
    """RTL-SDR device not found / not connected."""
    pass


class DeviceDisconnectedError(RTLSDRError):
    """RTL-SDR device disconnected during operation."""
    pass


def _parse_rtlsdr_error(stderr: str) -> RTLSDRError:
    """Parse rtl_sdr stderr and return an appropriate exception."""
    stderr_lower = stderr.lower()
    if "no supported devices" in stderr_lower or "no devices" in stderr_lower:
        return DeviceNotFoundError(
            "Aucun périphérique RTL-SDR trouvé. Vérifiez la connexion USB."
        )
    if "usb_claim_interface" in stderr_lower or "resource busy" in stderr_lower:
        return DeviceDisconnectedError(
            "Périphérique RTL-SDR inaccessible. "
            "Vérifiez qu'aucun autre programme ne l'utilise."
        )
    if "disconnect" in stderr_lower or "lost" in stderr_lower:
        return DeviceDisconnectedError(
            "Périphérique RTL-SDR déconnecté pendant la capture. "
            "Reconnectez le périphérique et réessayez."
        )
    return RTLSDRError(f"Erreur RTL-SDR: {stderr.strip()}")


class HardwareRNG:
    """
    Interface for Intel/AMD RDRAND/RDSEED hardware random number generators.
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
        """Compile the RDRAND library if needed."""
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

        # Use mkstemp for secure temp files with restrictive permissions
        c_fd, c_path = tempfile.mkstemp(suffix='.c')
        try:
            with os.fdopen(c_fd, 'w') as f:
                f.write(cls._C_SOURCE)
        except Exception:
            os.close(c_fd)
            os.remove(c_path)
            return None

        so_fd, lib_path = tempfile.mkstemp(suffix='.so')
        os.close(so_fd)

        try:
            result = subprocess.run([
                "gcc", "-O2", "-shared", "-fPIC",
                "-mrdrnd", "-mrdseed",
                c_path, "-o", lib_path
            ], capture_output=True, timeout=30)

            if result.returncode != 0:
                os.remove(lib_path)
                return None

            # Restrict .so permissions to owner read-only before loading
            os.chmod(lib_path, 0o500)

            cls._lib_path = lib_path
            return lib_path

        except subprocess.TimeoutExpired:
            if os.path.exists(lib_path):
                os.remove(lib_path)
            return None
        finally:
            if os.path.exists(c_path):
                os.remove(c_path)

    @classmethod
    def _get_lib(cls) -> Optional[ctypes.CDLL]:
        """Load the RDRAND library."""
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
        """Check if RDRAND is available."""
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
        """Generate n random bytes via RDRAND or RDSEED."""
        lib = cls._get_lib()
        if lib is None:
            raise RDRANDError("RDRAND not available")

        buffer = (ctypes.c_uint8 * n)()

        if use_rdseed:
            result = lib.rdseed_bytes(buffer, n)
        else:
            result = lib.rdrand_bytes(buffer, n)

        if result != 0:
            raise RDRANDError("RDRAND/RDSEED generation failed")

        return bytes(buffer)

    @classmethod
    def cleanup(cls):
        """Clean up resources."""
        if cls._lib_path and os.path.exists(cls._lib_path):
            try:
                os.remove(cls._lib_path)
            except OSError:
                pass
        cls._lib = None
        cls._lib_path = None


# =============================================================================
# Entropy extraction functions
# =============================================================================

def von_neumann_extract(data: np.ndarray) -> np.ndarray:
    """
    Von Neumann extractor to eliminate bias.
    OPTIMIZED: Uses NumPy for vectorized bit-level operations.

    Performance: ~5-10x faster than pure Python version.

    Args:
        data: Raw entropy data as numpy uint8 array

    Returns:
        Unbiased entropy as numpy uint8 array
    """
    # Vectorization with NumPy
    # Take LSB of each byte
    bits = np.unpackbits(data)

    # Reshape into pairs (0,1), (2,3), etc.
    if len(bits) % 2 != 0:
        bits = bits[:-1]

    pairs = bits.reshape(-1, 2)

    # Von Neumann: 01 -> 0, 10 -> 1, 00 and 11 -> reject
    # Condition: pairs[:,0] != pairs[:,1] (different bits)
    # Value: pairs[:,0] (if 01 -> 0, if 10 -> 1)
    valid = pairs[:, 0] != pairs[:, 1]
    extracted_bits = pairs[valid, 0]

    # Reconstruct bytes
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
    Cryptographic whitening using hash function.

    Args:
        raw_entropy: Raw entropy data
        hash_algo: Hash algorithm to use ('sha256' or 'sha512')

    Returns:
        Whitened entropy as numpy uint8 array
    """
    if hash_algo == 'sha256':
        h = hashlib.sha256
        hash_size = 32
    elif hash_algo == 'sha512':
        h = hashlib.sha512
        hash_size = 64
    else:
        raise ValueError(f"Unsupported hash: {hash_algo}")

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


# =============================================================================
# RTL-SDR functions
# =============================================================================

def is_rtl_sdr_available() -> bool:
    """Check if rtl_sdr is available AND a device is connected."""
    try:
        # Check if command exists
        result = subprocess.run(
            ["which", "rtl_sdr"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return False

        # Check if a device is connected
        result = subprocess.run(
            ["rtl_test", "-t"],
            capture_output=True,
            timeout=5,
            text=True
        )
        # rtl_test returns 0 even without device, but displays "No supported devices found"
        return "Found" in result.stderr and "No supported" not in result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def capture_entropy_fallback(bytes_needed: int, use_rdseed: bool = False) -> np.ndarray:
    """
    Fallback using RDRAND/RDSEED hardware RNG or secrets.

    Args:
        bytes_needed: Number of bytes to generate
        use_rdseed: Use RDSEED instead of RDRAND

    Returns:
        Entropy as numpy uint8 array
    """
    global _last_entropy_source

    if HardwareRNG.is_available():
        source = "RDSEED" if use_rdseed else "RDRAND"
        _last_entropy_source = source
        logger.info("RTL-SDR unavailable, using %s (CPU hardware RNG)", source)
        logger.info("Source: Intel/AMD certified hardware generator")
        try:
            raw_bytes = HardwareRNG.get_bytes(bytes_needed, use_rdseed=use_rdseed)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
            secure_zero(bytearray(raw_bytes))
            return arr
        except RDRANDError as e:
            logger.warning("RDRAND error: %s, falling back to secrets...", e)

    _last_entropy_source = "CSPRNG"
    logger.info("RTL-SDR and RDRAND unavailable, using system CSPRNG")
    logger.info("Note: System entropy is cryptographically secure but does not use a dedicated hardware source.")
    return np.frombuffer(secrets.token_bytes(bytes_needed), dtype=np.uint8)


def capture_entropy(
    samples: int = 500000,
    frequency: float = 100e6,
    sample_rate: float = 2.4e6,
    gain: int = 0,
    allow_fallback: bool = True,
    use_rdseed: bool = False
) -> np.ndarray:
    """
    Capture entropy from RTL-SDR.

    Args:
        samples: Number of samples to capture
        frequency: Capture frequency in Hz
        sample_rate: Sample rate in Hz
        gain: RTL-SDR gain (0 = auto)
        allow_fallback: Allow fallback to RDRAND/CSPRNG
        use_rdseed: Use RDSEED for fallback instead of RDRAND

    Returns:
        Processed entropy as numpy uint8 array
    """
    global _last_entropy_source

    if not is_rtl_sdr_available():
        if allow_fallback:
            estimated_bytes = max(1000, samples // 30)
            return capture_entropy_fallback(estimated_bytes, use_rdseed=use_rdseed)
        else:
            raise RuntimeError("rtl_sdr not found. Install rtl-sdr or use --allow-fallback")

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
                logger.warning("rtl_sdr failed: %s", result.stderr.strip())
                estimated_bytes = max(1000, samples // 30)
                return capture_entropy_fallback(estimated_bytes, use_rdseed=use_rdseed)
            raise _parse_rtlsdr_error(result.stderr)

        _last_entropy_source = "RTL-SDR"

        raw = np.fromfile(temp_path, dtype=np.uint8)

        if len(raw) < samples * 0.9:
            raise DeviceDisconnectedError(
                f"Capture incomplète: {len(raw)}/{samples} échantillons. "
                "Le périphérique a peut-être été déconnecté."
            )

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


def capture_entropy_raw(
    samples: int = 500000,
    frequency: float = 100e6,
    sample_rate: float = 2.4e6,
    gain: int = 0,
) -> np.ndarray:
    """
    Capture raw IQ samples from RTL-SDR without processing.
    Used for traceability when proof of the raw capture is needed.

    Args:
        samples: Number of samples to capture
        frequency: Capture frequency in Hz
        sample_rate: Sample rate in Hz
        gain: RTL-SDR gain (0 = auto)

    Returns:
        Raw IQ samples as numpy uint8 array
    """
    global _last_entropy_source

    if not is_rtl_sdr_available():
        raise RuntimeError("RTL-SDR required for raw capture (no fallback)")

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
            raise _parse_rtlsdr_error(result.stderr)

        _last_entropy_source = "RTL-SDR"

        raw = np.fromfile(temp_path, dtype=np.uint8)

        if len(raw) < samples * 0.9:
            raise DeviceDisconnectedError(
                f"Capture incomplète: {len(raw)}/{samples} échantillons. "
                "Le périphérique a peut-être été déconnecté."
            )

        return raw

    finally:
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            with open(temp_path, 'wb') as f:
                f.write(os.urandom(file_size))
            os.remove(temp_path)


def load_entropy_from_file(filepath: str, apply_hash: bool = True) -> np.ndarray:
    """
    Load entropy from a file.

    Args:
        filepath: Path to the entropy file
        apply_hash: Apply hash whitening

    Returns:
        Entropy as numpy uint8 array
    """
    global _last_entropy_source
    _last_entropy_source = "file"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.fromfile(filepath, dtype=np.uint8)

    if apply_hash:
        hashed = hash_entropy(data)
        secure_zero(data)
        return hashed

    return data

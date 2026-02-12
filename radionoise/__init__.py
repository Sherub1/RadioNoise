"""
RadioNoise - True Random Number Generator using RTL-SDR radio noise.

This package provides cryptographic-quality entropy from RTL-SDR captures,
with NIST SP 800-22 validation and traceability features.
"""

__version__ = "2.0.0"

from radionoise.core.entropy import (
    HardwareRNG,
    RDRANDError,
    capture_entropy,
    capture_entropy_fallback,
    capture_entropy_raw,
    von_neumann_extract,
    hash_entropy,
    load_entropy_from_file,
    is_rtl_sdr_available,
    get_last_entropy_source,
)

from radionoise.core.nist import NISTTests

from radionoise.core.generator import (
    generate_password,
    generate_passphrase,
    calculate_password_entropy,
    calculate_passphrase_entropy,
    CHARSETS,
    CharsetName,
)

from radionoise.core.security import secure_zero

__all__ = [
    # Version
    "__version__",
    # Entropy
    "HardwareRNG",
    "RDRANDError",
    "capture_entropy",
    "capture_entropy_fallback",
    "capture_entropy_raw",
    "von_neumann_extract",
    "hash_entropy",
    "load_entropy_from_file",
    "is_rtl_sdr_available",
    "get_last_entropy_source",
    # NIST
    "NISTTests",
    # Generator
    "generate_password",
    "generate_passphrase",
    "calculate_password_entropy",
    "calculate_passphrase_entropy",
    "CHARSETS",
    "CharsetName",
    # Security
    "secure_zero",
]

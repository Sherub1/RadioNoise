# -*- coding: utf-8 -*-
"""
RadioNoise Generator - Password and passphrase generation.
"""

import string
import numpy as np
from typing import List, Literal

from radionoise.core.wordlist import EFF_WORDLIST, WORDLIST_SIZE, BITS_PER_WORD

# Character sets
CHARSETS = {
    "alnum": string.ascii_letters + string.digits,
    "alpha": string.ascii_letters,
    "digits": string.digits,
    "hex": string.hexdigits[:16],
    "full": string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:,.<>?",
    "safe": string.ascii_letters + string.digits + "-_!@#$%",
}

CharsetName = Literal["alnum", "alpha", "digits", "hex", "full", "safe"]

# Backward compatibility alias
WORDLIST = EFF_WORDLIST


def generate_password(
    random_bytes: np.ndarray,
    length: int = 16,
    charset: CharsetName = "safe"
) -> str:
    """
    Generate a password with rejection sampling for uniform distribution.

    Args:
        random_bytes: Random entropy as numpy uint8 array
        length: Password length
        charset: Character set to use

    Returns:
        Generated password string

    Raises:
        ValueError: If not enough entropy to generate password
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
        raise ValueError("Not enough entropy to generate complete password")

    return ''.join(password)


def generate_passphrase(random_bytes: np.ndarray, words: int = 6) -> str:
    """
    Generate a passphrase from EFF large wordlist using rejection sampling.

    Uses 2 bytes per word attempt (16 bits) with rejection sampling to ensure
    uniform distribution over 7776 words (~12.925 bits/word entropy).

    Args:
        random_bytes: Random entropy as numpy uint8 array
        words: Number of words in passphrase (default: 6 = ~77.5 bits)

    Returns:
        Generated passphrase with words separated by hyphens

    Raises:
        ValueError: If not enough entropy to generate complete passphrase
    """
    # EFF large wordlist: 7776 words = 6^5 (for 5 dice rolls)
    # We use 2 bytes (16 bits, max 65535) and reject values >= 7776
    # Acceptance rate: 7776/65536 ≈ 11.9% per attempt
    # Expected bytes per word: 2 / 0.119 ≈ 17 bytes

    phrase: List[str] = []
    idx = 0

    while len(phrase) < words:
        # Need 2 bytes for each attempt
        if idx + 2 > len(random_bytes):
            raise ValueError(
                f"Not enough entropy: generated {len(phrase)}/{words} words, "
                f"used {idx} bytes. Need more entropy (try ~{words * 20} bytes)."
            )

        # Combine 2 bytes into 16-bit value (big-endian)
        value = (int(random_bytes[idx]) << 8) | int(random_bytes[idx + 1])
        idx += 2

        # Rejection sampling: only accept if value < 7776
        if value < WORDLIST_SIZE:
            phrase.append(EFF_WORDLIST[value])

    return '-'.join(phrase)


def calculate_password_entropy(length: int, charset: CharsetName = "safe") -> float:
    """
    Calculate theoretical entropy of a password.

    Args:
        length: Password length
        charset: Character set used

    Returns:
        Entropy in bits
    """
    chars = CHARSETS.get(charset, CHARSETS["safe"])
    return length * np.log2(len(chars))


def calculate_passphrase_entropy(words: int) -> float:
    """
    Calculate theoretical entropy of a passphrase.

    Args:
        words: Number of words

    Returns:
        Entropy in bits (EFF wordlist: ~12.925 bits/word)
    """
    return words * BITS_PER_WORD

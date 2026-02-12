"""
RadioNoise Security - Secure memory handling utilities.
"""

from typing import Union
import numpy as np


def secure_zero(data: Union[np.ndarray, bytearray, bytes]) -> None:
    """
    Attempt to securely erase sensitive data from memory (best-effort).

    Note: This is best-effort only. Python's memory management doesn't
    guarantee that memory is actually cleared, and the GC may have already
    copied the data elsewhere.

    Args:
        data: The data to attempt to zero out. Must be a writable numpy array
              or bytearray.
    """
    try:
        if isinstance(data, np.ndarray):
            if data.flags.writeable:
                data[:] = 0
        elif isinstance(data, bytearray):
            for i in range(len(data)):
                data[i] = 0
        # bytes objects are immutable, cannot be zeroed
    except (TypeError, ValueError):
        pass

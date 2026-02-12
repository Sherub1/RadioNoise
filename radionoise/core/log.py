"""
RadioNoise structured logging.

Provides a consistent logging interface for all core and traceability modules.
CLI output (cli.py) intentionally uses print() and is not modified.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger namespaced under 'radionoise'."""
    return logging.getLogger(f'radionoise.{name}')


def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure the radionoise root logger.

    Args:
        level: Logging level (default INFO)
        log_file: Optional file path for file logging
    """
    logger = logging.getLogger('radionoise')
    logger.setLevel(level)

    fmt = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

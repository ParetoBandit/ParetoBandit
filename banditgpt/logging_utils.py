from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str = "INFO", fmt: Optional[str] = None) -> logging.Logger:
    """
    Configure root logging with a sensible default format.

    Args:
        level: Logging level name (e.g., "INFO", "DEBUG").
        fmt: Optional logging format override.

    Returns:
        The root logger (configured).
    """
    fmt = fmt or "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    return logging.getLogger()

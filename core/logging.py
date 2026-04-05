from __future__ import annotations

import logging
from pathlib import Path


def build_file_logger(
    name: str,
    log_file: str | Path,
    level: int | str = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(f"{name}:{Path(log_file).expanduser()}")
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_exists = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not stream_exists:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    file_exists = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved
        for h in logger.handlers
    )
    if not file_exists:
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    for handler in logger.handlers:
        handler.setLevel(level)

    return logger

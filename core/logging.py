from __future__ import annotations
import logging
from pathlib import Path

_DEFAULT_SHUGA_LOGGER: logging.Logger | None = None

class ClassNameFilter(logging.Filter):
    """
    Ensure custom formatter fields always exist.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        cls = getattr(record, "cls", "")
        if cls is None:
            cls = ""
        record.cls = str(cls)
        record.cls_prefix = f"{record.cls}." if record.cls else ""
        return True

def set_default_logger(logger: logging.Logger | logging.LoggerAdapter | None) -> None:
    global _DEFAULT_SHUGA_LOGGER
    if logger is None:
        _DEFAULT_SHUGA_LOGGER = None
    elif isinstance(logger, logging.LoggerAdapter):
        _DEFAULT_SHUGA_LOGGER = logger.logger
    else:
        _DEFAULT_SHUGA_LOGGER = logger

def get_default_logger() -> logging.Logger | None:
    return _DEFAULT_SHUGA_LOGGER

def resolve_logger(logger: logging.Logger | logging.LoggerAdapter | None = None, name: str | None = None) -> logging.Logger | logging.LoggerAdapter:
    """
    Resolve a usable logger.

    Precedence:
    1. explicitly supplied logger
    2. child of globally registered default logger
    3. standard library logger by name
    """
    if logger is not None:
        return logger
    default = get_default_logger()
    if default is not None:
        return default.getChild(name) if name else default
    return logging.getLogger(name or "shuga")

def build_file_logger(name: str, log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    log_path = Path(log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # notebook-safe rebuild
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    fmt = "%(asctime)s - %(levelname)s - [%(name)s.%(cls_prefix)s%(funcName)s:%(lineno)d] %(message)s"
    formatter = logging.Formatter(fmt)
    class_filter = ClassNameFilter()
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    fh.addFilter(class_filter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    sh.addFilter(class_filter)
    logger.addHandler(sh)
    set_default_logger(logger)
    return logger

def get_class_logger(logger: logging.Logger | logging.LoggerAdapter | None, cls_name: str, name: str | None = None) -> logging.LoggerAdapter:
    base = resolve_logger(logger, name=name)
    if isinstance(base, logging.LoggerAdapter):
        base = base.logger
    return logging.LoggerAdapter(base, {"cls": cls_name})


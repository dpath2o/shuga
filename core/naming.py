from __future__ import annotations
import re

_METHOD_ALIASES = {"raw"          : "raw",
                   "daily"        : "raw",
                   "day"          : "raw",
                   "bin-days"     : "binary-days",
                   "binary-days"  : "binary-days",
                   "binary_days"  : "binary-days",
                   "binary"       : "binary-days",
                   "bin"          : "binary-days",
                   "roll-mean"    : "rolling-mean",
                   "rolling-mean" : "rolling-mean",
                   "rolling_mean" : "rolling-mean",
                   "rolling"      : "rolling-mean",
                   "roll"         : "rolling-mean"}

def normalize_method(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(f"Unsupported method={value!r}. Use one of raw, binary-days, rolling-mean.")
    return _METHOD_ALIASES[key]

def _format_sci(value: float, decimals: int = 1) -> str:
    s = f"{float(value):.{decimals}e}"
    s = re.sub(r"e([+-])0*(\d+)$", r"e\1\2", s)
    s = s.replace("e+", "e")
    return s

def threshold_tag_dir(value: float) -> str:
    return _format_sci(value, decimals=1)

def threshold_tag_compact(value: float) -> str:
    s = f"{float(value):.0e}"
    s = re.sub(r"e([+-])0*(\d+)$", r"e\1\2", s)
    s = s.replace("e+", "e")
    return s

def method_dirname(method: str, *, bin_window: int, bin_min_days: int, roll_window: int) -> str:
    norm = normalize_method(method)
    if norm == "raw":
        return "raw"
    if norm == "binary-days":
        return f"bin-win-{int(bin_window):02d}_bin-min-{int(bin_min_days):02d}"
    return f"roll-days-{int(roll_window)}"

def method_slug(method: str) -> str:
    return normalize_method(method).replace("-", "_")

def filename_token(value: str) -> str:
    """Return a filesystem-safe filename token."""
    s = str(value).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

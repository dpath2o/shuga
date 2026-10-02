from __future__ import annotations
import shutil
from datetime import datetime
from pathlib import Path
import xarray as xr
from shuga.core.naming import normalize_method
from shuga.core.paths import ShugaPaths

def output_chunk_map(ds: xr.Dataset) -> dict[str, int]:
    """
    Default chunking policy for metrics stores.
    """
    chunk_map: dict[str, int] = {}

    if "time" in ds.dims:
        chunk_map["time"] = min(31, ds.sizes["time"])
    if "nj" in ds.dims:
        chunk_map["nj"] = min(128, ds.sizes["nj"])
    if "ni" in ds.dims:
        chunk_map["ni"] = min(128, ds.sizes["ni"])
    if "region" in ds.dims:
        chunk_map["region"] = ds.sizes["region"]
    return chunk_map

def open_existing_metrics(*, pth_cfg: ShugaPaths, cache: dict[str, xr.Dataset], method: str) -> xr.Dataset | None:
    """
    Open an existing method-specific metrics store, using a caller-owned cache.
    """
    norm = normalize_method(method)
    if norm in cache:
        return cache[norm]
    store = pth_cfg.metrics_store(norm)
    if not store.exists():
        return None
    ds = xr.open_zarr(store, consolidated = False)
    cache[norm] = ds
    return ds

def backup_legacy_store(store: Path, *, logger = None, suffix: str = "legacy_badcoords") -> Path:
    """
    Move an existing legacy/problematic store aside.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = store.with_name(f"{store.name}.{suffix}_{stamp}")
    if backup.exists():
        shutil.rmtree(backup)
    shutil.move(str(store), str(backup))
    if logger is not None:
        logger.warning("Backed up legacy metrics store to: %s", backup)
    return backup

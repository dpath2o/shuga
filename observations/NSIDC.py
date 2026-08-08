from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.logging import build_file_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import ObservationSpec, RunSpec

class NSIDCObservations:
    """
    NSIDC sea-ice concentration loader and SIA/SIE calculator.

    This is the NSIDC-specific portion extracted from the older, ambiguously named
    ``shuga.observations.cice.SeaIceObservations`` module.
    """
    def __init__(self,
                 run_cfg: RunSpec,
                 obs_cfg: ObservationSpec | None = None,
                 pth_cfg: ShugaPaths | None = None, *,
                 chunks : dict | None = None,
                 logger = None) -> None:
        self.run_cfg = run_cfg
        self.obs_cfg = obs_cfg or ObservationSpec()
        self.pth_cfg = pth_cfg or ShugaPaths(run_cfg = run_cfg, cls_cfg = None, obs_cfg = self.obs_cfg)  # type: ignore[arg-type]
        self.chunks  = chunks or {"time": 31}
        self.logger  = logger or build_file_logger("shuga.obs_cfg.NSIDC", Path.home() / "logs" / "observations" / "shuga_NSIDC.log")
        self._cache: dict[tuple[str, str, str], xr.Dataset] = {}

    @staticmethod
    def canonical_hemisphere(value: str) -> str:
        token = str(value).strip().lower()
        return "south" if token in {"s", "sh", "south", "southern"} else "north"

    def _aux_suffix(self, hemisphere: str) -> str:
        hemi = "S" if hemisphere == "south" else "N"
        return f"{hemi}25km_v1.1.nc"

    def area_file(self, hemisphere: str) -> Path:
        return self.pth_cfg.nsidc_aux_root_path / f"NSIDC0771_CellArea_PS_{self._aux_suffix(hemisphere)}"

    def latlon_file(self, hemisphere: str) -> Path:
        return self.pth_cfg.nsidc_aux_root_path / f"NSIDC0771_LatLon_PS_{self._aux_suffix(hemisphere)}"

    def daily_files(self, start_date: str, end_date: str, hemisphere: str) -> list[Path]:
        hemi = self.canonical_hemisphere(hemisphere)
        root = self.pth_cfg.nsidc_root_path / hemi / "daily"
        if not root.exists():
            raise FileNotFoundError(f"NSIDC daily directory does not exist: {root}")
        dates = pd.date_range(start_date, end_date, freq="D")
        files: list[Path] = []
        for dt in dates:
            patt = f"sic_ps{'s' if hemi == 'south' else 'n'}25_{dt:%Y%m%d}_*_v06r00.nc"
            matches = sorted(root.glob(patt))
            if matches:
                files.append(matches[0])
        if not files:
            raise FileNotFoundError(f"No NSIDC daily files found in {root} between {start_date} and {end_date}")
        return files

    def load_daily(self, start_date: str | None = None, end_date: str | None = None, hemisphere: str | None = None) -> xr.Dataset:
        start_date = start_date or self.run_cfg.start_date
        end_date   = end_date or self.run_cfg.end_date
        hemi       = self.canonical_hemisphere(hemisphere or self.run_cfg.hemisphere)
        key        = (start_date, end_date, hemi)
        if key in self._cache:
            return self._cache[key]
        files = self.daily_files(start_date, end_date, hemi)
        self.logger.info("Opening %s NSIDC daily files for %s hemisphere", len(files), hemi)
        def _prep(ds: xr.Dataset) -> xr.Dataset:
            keep = [v for v in (self.obs_cfg.nsidc_sic_var,) if v in ds]
            return ds[keep] if keep else ds
        ds     = xr.open_mfdataset(files, combine="by_coords", parallel=True, preprocess=_prep, chunks=self.chunks)
        latlon = xr.open_dataset(self.latlon_file(hemi))[["latitude", "longitude"]]
        area   = xr.open_dataset(self.area_file(hemi))[["cell_area"]]
        ds     = xr.merge([ds, latlon, area], compat="override", combine_attrs="drop_conflicts")
        self._cache[key] = ds
        return ds

    def compute_sia_sie(self, start_date: str | None = None, end_date: str | None = None, hemisphere: str | None = None, threshold: float | None = None) -> xr.Dataset:
        ds   = self.load_daily(start_date=start_date, end_date=end_date, hemisphere=hemisphere)
        sic  = ds[self.obs_cfg.nsidc_sic_var].astype("float32")
        mask = sic >= float(threshold if threshold is not None else self.obs_cfg.nsidc_threshold)
        area = ds["cell_area"].astype("float64")
        sia  = (sic.where(mask, 0.0) * area).sum(dim=("y", "x")) / 1e12
        sie  = (mask.astype("float32") * area).sum(dim=("y", "x")) / 1e12
        out  = xr.Dataset({"SIA": sia, "SIE": sie})
        out["SIA"].attrs.update(long_name="Sea Ice Area", units="10^6 km^2")
        out["SIE"].attrs.update(long_name="Sea Ice Extent", units="10^6 km^2")
        return out

SeaIceNSIDC = NSIDCObservations 

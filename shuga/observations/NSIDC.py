from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from shuga.core.logging import build_file_logger
from shuga.core.paths import ShugaPaths
from shuga.core.types import ObservationSpec, RunSpec

class NSIDCObservations:
    """NSIDC sea-ice concentration loader and SIA/SIE processor."""

    def __init__(self,
                 run_cfg: RunSpec,
                 obs_cfg: ObservationSpec | None = None,
                 pth_cfg: ShugaPaths | None = None, *,
                 chunks : dict | None = None,
                 logger = None) -> None:
        self.run_cfg = run_cfg
        self.obs_cfg = obs_cfg or ObservationSpec()
        self.pth_cfg = pth_cfg or ShugaPaths(run_cfg=run_cfg, cls_cfg=None, obs_cfg=self.obs_cfg)  # type: ignore[arg-type]
        self.chunks  = chunks or {"time": 31}
        self.logger  = logger or build_file_logger("shuga.obs_cfg.NSIDC", Path.home() / "logs" / "observations" / "shuga_NSIDC.log")
        self._cache: dict[tuple[str, str, str], xr.Dataset] = {}

    @staticmethod
    def canonical_hemisphere(value: str) -> str:
        token = str(value).strip().lower()
        return "south" if token in {"s", "sh", "south", "southern"} else "north"

    @staticmethod
    def hemisphere_token(value: str) -> str:
        return "SH" if NSIDCObservations.canonical_hemisphere(value) == "south" else "NH"

    def _aux_suffix(self, hemisphere: str) -> str:
        hemi = "S" if hemisphere == "south" else "N"
        return f"{hemi}25km_v1.1.nc"

    def area_file(self, hemisphere: str) -> Path:
        return self.pth_cfg.nsidc_aux_root_path / f"NSIDC0771_CellArea_PS_{self._aux_suffix(hemisphere)}"

    def latlon_file(self, hemisphere: str) -> Path:
        return self.pth_cfg.nsidc_aux_root_path / f"NSIDC0771_LatLon_PS_{self._aux_suffix(hemisphere)}"

    def processed_sia_sie_store(self, hemisphere: str | None = None) -> Path:
        hemi = self.hemisphere_token(hemisphere or self.run_cfg.hemisphere)
        return self.pth_cfg.seaice_root_path / "NSIDC" / "processed" / f"NSIDC_{self.obs_cfg.nsidc_version}_{hemi}_SIA_SIE.zarr"

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

    def compute_sia_sie(self,
                        start_date: str | None = None,
                        end_date: str | None = None,
                        hemisphere: str | None = None,
                        threshold: float | None = None) -> xr.Dataset:
        ds   = self.load_daily(start_date=start_date, end_date=end_date, hemisphere=hemisphere)
        sic  = ds[self.obs_cfg.nsidc_sic_var].astype("float32")
        thr  = float(threshold if threshold is not None else self.obs_cfg.nsidc_threshold)
        mask = sic >= thr
        area = ds["cell_area"].astype("float64")
        sia  = (sic.where(mask, 0.0) * area).sum(dim=("y", "x")) / 1e12
        sie  = (mask.astype("float32") * area).sum(dim=("y", "x")) / 1e12
        out  = xr.Dataset({"SIA": sia, "SIE": sie})
        out["SIA"].attrs.update(long_name="Sea Ice Area", units="10^6 km^2")
        out["SIE"].attrs.update(long_name="Sea Ice Extent", units="10^6 km^2")
        out.attrs.update(source="NSIDC Sea Ice Concentration CDR", nsidc_version=self.obs_cfg.nsidc_version,
                         concentration_threshold=thr, hemisphere=self.hemisphere_token(hemisphere or self.run_cfg.hemisphere))
        return out

    def process_sia_sie(self,
                        start_date: str | None = None,
                        end_date: str | None = None,
                        hemisphere: str | None = None,
                        threshold: float | None = None,
                        output_store: str | Path | None = None,
                        overwrite: bool = False,
                        chunk_time: int = 365) -> Path:
        """Compute NSIDC SIA/SIE once and persist the one-dimensional product."""
        start_date = start_date or self.run_cfg.start_date
        end_date   = end_date or self.run_cfg.end_date
        hemi       = hemisphere or self.run_cfg.hemisphere
        path       = Path(output_store).expanduser() if output_store else self.processed_sia_sie_store(hemi)
        if path.exists() and not overwrite:
            self.logger.info("NSIDC processed SIA/SIE store exists: %s", path)
            return path
        if path.exists():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = self.compute_sia_sie(start_date=start_date, end_date=end_date, hemisphere=hemi, threshold=threshold).compute()
        out.attrs.update(time_start=start_date, time_end=end_date)
        for var in out.variables:
            out[var].encoding.clear()
        nt = max(1, min(int(chunk_time), out.sizes.get("time", 1)))
        encoding = {name: {"chunks": (nt,)} for name in ("SIA", "SIE", "time") if name in out.variables}
        out.to_zarr(path, mode="w", consolidated=True, zarr_format=2, encoding=encoding)
        self.logger.info("Wrote NSIDC processed SIA/SIE: %s", path)
        return path

    def load_sia_sie(self,
                     start_date: str | None = None,
                     end_date: str | None = None,
                     hemisphere: str | None = None,
                     store: str | Path | None = None,
                     chunks: dict | None = None) -> xr.Dataset:
        """Load the precomputed NSIDC SIA/SIE product; no concentration processing occurs."""
        path = Path(store).expanduser() if store else self.processed_sia_sie_store(hemisphere or self.run_cfg.hemisphere)
        if not path.exists():
            raise FileNotFoundError(f"Processed NSIDC SIA/SIE store not found: {path}. Run process_NSIDC_SIA_SIE_pbs_wrapper.sh first.")
        try:
            ds = xr.open_zarr(path, consolidated=True, chunks=chunks or self.chunks)
        except (KeyError, ValueError, FileNotFoundError):
            ds = xr.open_zarr(path, consolidated=False, chunks=chunks or self.chunks)
        dt0 = start_date or self.run_cfg.start_date
        dtN = end_date or self.run_cfg.end_date
        return ds.sel(time=slice(dt0, dtN))

SeaIceNSIDC = NSIDCObservations

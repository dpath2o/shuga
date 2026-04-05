from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .naming import method_dirname, method_slug, threshold_tag_compact, threshold_tag_dir
from .types import ClassificationSpec, RunSpec


@dataclass(slots=True)
class ShugaPaths:
    run: RunSpec
    classify: ClassificationSpec
    afim_output_root: str | Path | None = None
    graphics_root: str | Path | None = None
    logs_root: str | Path | None = None
    cice_store: str | Path | None = None
    static_store: str | Path | None = None
    classification_root: str | Path | None = None

    @staticmethod
    def canonical_hemisphere(value: str) -> str:
        token = str(value).strip().lower()
        mapping = {
            "s": "SH",
            "sh": "SH",
            "south": "SH",
            "southern": "SH",
            "n": "NH",
            "nh": "NH",
            "north": "NH",
            "northern": "NH",
        }
        if token not in mapping:
            raise ValueError(f"Unsupported hemisphere={value!r}. Use SH/NH or south/north.")
        return mapping[token]

    @property
    def hemisphere(self) -> str:
        return self.canonical_hemisphere(self.run.hemisphere)

    @property
    def output_root(self) -> Path:
        if self.afim_output_root is not None:
            return Path(self.afim_output_root).expanduser()
        return Path(f"/g/data/{self.run.project}/{self.run.user}/afim_output/{self.run.sim_name}")

    @property
    def zarr_root(self) -> Path:
        return self.output_root / "zarr"

    @property
    def classification_root_path(self) -> Path:
        if self.classification_root is not None:
            return Path(self.classification_root).expanduser()
        return (
            self.zarr_root
            / self.hemisphere
            / f"ispd_thresh_{threshold_tag_dir(self.classify.ispd_thresh)}"
            / self.classify.ice_type
            / self.classify.grid_type
        )

    @property
    def graphics_root_path(self) -> Path:
        if self.graphics_root is not None:
            return Path(self.graphics_root).expanduser()
        return Path(f"/g/data/{self.run.project}/{self.run.user}/GRAPHICAL/AFIM")

    @property
    def logs_root_path(self) -> Path:
        if self.logs_root is not None:
            return Path(self.logs_root).expanduser()
        return Path.home() / "logs"

    def resolve_cice_store(self) -> Path:
        if self.cice_store is not None:
            path = Path(self.cice_store).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Explicit CICE store does not exist: {path}")
            return path
        candidates = [
            self.zarr_root / "iceh_daily.zarr",
            self.zarr_root / "history" / "iceh_daily.zarr",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not infer iceh_daily.zarr from the default AFIM layout. "
            f"Tried: {', '.join(str(c) for c in candidates)}"
        )

    def resolve_static_store(self) -> Path | None:
        if self.static_store is not None:
            path = Path(self.static_store).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Explicit static store does not exist: {path}")
            return path
        candidates = [
            self.zarr_root / "iceh_static.zarr",
            self.zarr_root / "static" / "iceh_static.zarr",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def classification_store(self, method: str) -> Path:
        return self.classification_root_path / method_dirname(
            method,
            bin_window=self.classify.bin_window,
            bin_min_days=self.classify.bin_min_days,
            roll_window=self.classify.roll_window,
        ) / "data.zarr"

    def metrics_store(self, method: str) -> Path:
        return self.classification_root_path / method_dirname(
            method,
            bin_window=self.classify.bin_window,
            bin_min_days=self.classify.bin_min_days,
            roll_window=self.classify.roll_window,
        ) / "mets.zarr"

    def classification_log_path(self) -> Path:
        stem = (
            f"classify_{self.run.sim_name}_{self.classify.ice_type}_{self.classify.grid_type}"
            f"_ispd_thresh{threshold_tag_compact(self.classify.ispd_thresh)}"
            f"_BW{self.classify.bin_window}_BM{self.classify.bin_min_days}_roll{self.classify.roll_window}.log"
        )
        return self.logs_root_path / "classification" / stem

    def metrics_log_path(self) -> Path:
        stem = (
            f"metrics_{self.run.sim_name}_{self.classify.ice_type}_{self.classify.grid_type}"
            f"_ispd_thresh{threshold_tag_compact(self.classify.ispd_thresh)}"
            f"_BW{self.classify.bin_window}_BM{self.classify.bin_min_days}_roll{self.classify.roll_window}.log"
        )
        return self.logs_root_path / "metrics" / stem

    def fip_plot_path(self, method: str) -> Path:
        method_part = method_slug(method)
        name = f"{self.run.start_date}_{self.run.end_date}_FIP_{method_part}.png"
        return self.graphics_root_path / self.run.sim_name / "FIP" / name

    def timeseries_plot_path(self, variable: str, method: str, region: str = "total") -> Path:
        method_part = method_slug(method)
        region_part = str(region).lower()
        name = (
            f"{self.run.start_date}_{self.run.end_date}_{self.run.sim_name}_"
            f"{variable}_{method_part}_{region_part}.png"
        )
        return self.graphics_root_path / "timeseries" / name

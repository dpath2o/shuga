from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .naming import filename_token, method_dirname, method_slug, normalize_method, threshold_tag_compact, threshold_tag_dir
from .types import (
    ClassificationSpec,
    MetricsSpec,
    ObservationSpec,
    PlottingSpec,
    RunSpec,
)


@dataclass(slots=True)
class ShugaPaths:
    run: RunSpec
    classify: ClassificationSpec
    metrics: MetricsSpec | None = None
    plotting: PlottingSpec | None = None
    observations: ObservationSpec | None = None

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

    @property
    def seaice_root_path(self) -> Path:
        obs = self.observations or ObservationSpec()
        if obs.seaice_root is not None:
            return Path(obs.seaice_root).expanduser()
        return Path(f"/g/data/{self.run.project}/{self.run.user}/SeaIce")

    @property
    def fi_obs_root_path(self) -> Path:
        obs = self.observations or ObservationSpec()
        if obs.af2020_root is not None:
            return Path(obs.af2020_root).expanduser()
        return self.seaice_root_path / "FI_obs"

    @property
    def nsidc_root_path(self) -> Path:
        obs = self.observations or ObservationSpec()
        if obs.nsidc_root is not None:
            return Path(obs.nsidc_root).expanduser()
        return self.seaice_root_path / "NSIDC" / obs.nsidc_version

    @property
    def nsidc_aux_root_path(self) -> Path:
        obs = self.observations or ObservationSpec()
        if obs.nsidc_cellarea_root is not None:
            return Path(obs.nsidc_cellarea_root).expanduser()
        return self.seaice_root_path / "NSIDC" / obs.nsidc_cellarea_product

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

    # backward-compatible aliases for newer callers
    def resolve_class_store(self, method: str) -> Path:
        return self.classification_store(method)

    def resolve_metrics_store(self, method: str) -> Path:
        return self.metrics_store(method)

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

    def figure_root(self, region: str | None = None) -> Path:
        parts = [self.graphics_root_path, self.run.sim_name]
        if region is not None:
            parts.append(str(region))
        path = parts[0]
        for part in parts[1:]:
            path = path / part
        return path

    def fip_plot_path(self, method: str, region: str = "TOTAL") -> Path:
        method_part = method_slug(method)
        name = f"{self.run.start_date}_{self.run.end_date}_{self.run.sim_name}_FIP_{method_part}.png"
        return self.figure_root(region=region) / "FIP" / name

    def timeseries_plot_path(self, variable: str, method: str, region: str = "total") -> Path:
        method_part = method_slug(method)
        name = f"{self.run.start_date}_{self.run.end_date}_{self.run.sim_name}_{variable}_{method_part}.png"
        return self.figure_root(region=region) / "timeseries" / name


    def multi_timeseries_plot_path(
        self,
        variable: str,
        method: str,
        simulations,
        *,
        region: str = "total",
        dt0_str: str | None = None,
        dtN_str: str | None = None,
    ) -> Path:
        var = filename_token(str(variable).upper())
        norm = filename_token(normalize_method(method))
        region_key = filename_token("total" if str(region).strip().lower() == "total" else str(region))
        dt0 = filename_token(dt0_str or self.run.start_date)
        dtN = filename_token(dtN_str or self.run.end_date)
        sim_tokens: list[str] = []
        for spec in simulations:
            if isinstance(spec, str):
                sim_name = spec
            elif isinstance(spec, dict):
                sim_name = spec.get("sim_name")
            else:
                sim_name = getattr(spec, "sim_name", None)
            if not sim_name:
                raise ValueError(f"Could not resolve sim_name from simulation spec: {spec!r}")
            sim_tokens.append(filename_token(sim_name))
        sim_part = "_".join(sim_tokens)
        name = f"{var}_{sim_part}_{dt0}_{dtN}_{norm}_{region_key}.png"
        return self.graphics_root_path / "timeseries" / name

    def split_hemisphere_plot_path(self, variable: str, date_str: str) -> Path:
        return self.figure_root() / variable / f"{date_str}.png"

    def regional_var_plot_path(self, variable: str, date_str: str, region: str) -> Path:
        return self.figure_root(region=region) / variable / f"{date_str}.png"

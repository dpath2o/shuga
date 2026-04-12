from __future__  import annotations
import re
from dataclasses import dataclass
from pathlib     import Path
from .naming     import filename_token, method_dirname, method_slug, normalize_method, threshold_tag_compact, threshold_tag_dir
from .types      import (ClassificationSpec,
                         MetricsSpec,
                         ObservationSpec,
                         PlottingSpec,
                         RunSpec,
                         CICEGridSpec,
                         WaveForcingSpec,
                         LateralDragSpec)

@dataclass(slots=True)
class ShugaPaths:
    # run                 : RunSpec
    # classify            : ClassificationSpec
    # metrics             : MetricsSpec | None     = None
    # plotting            : PlottingSpec | None    = None
    # observations        : ObservationSpec | None = None
    # wave_forcing        : WaveForcingSpec | None = None
    # afim_output_root    : str | Path | None      = None
    # graphics_root       : str | Path | None      = None
    # logs_root           : str | Path | None      = None
    # cice_store          : str | Path | None      = None
    # static_store        : str | Path | None      = None
    # classification_root : str | Path | None      = None
    run: RunSpec
    classify: ClassificationSpec
    metrics: MetricsSpec | None = None
    plotting: PlottingSpec | None = None
    observations: ObservationSpec | None = None
    wave_forcing: WaveForcingSpec | None = None
    cice_grid: CICEGridSpec | None = None
    lateral_drag: LateralDragSpec | None = None
    afim_output_root: str | Path | None = None
    graphics_root: str | Path | None = None
    logs_root: str | Path | None = None
    cice_store: str | Path | None = None
    static_store: str | Path | None = None
    classification_root: str | Path | None = None

    @staticmethod
    def canonical_hemisphere(value: str) -> str:
        token   = str(value).strip().lower()
        mapping = {"s"        : "SH",
                   "sh"       : "SH",
                   "south"    : "SH",
                   "southern" : "SH",
                   "n"        : "NH",
                   "nh"       : "NH",
                   "north"    : "NH",
                   "northern" : "NH"}
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
        return (self.zarr_root
                / self.hemisphere
                / f"ispd_thresh_{threshold_tag_dir(self.classify.ispd_thresh)}"
                / self.classify.ice_type
                / self.classify.grid_type)

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

    def resolve_cice_store(self) -> Path:
        if self.cice_store is not None:
            path = Path(self.cice_store).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Explicit CICE store does not exist: {path}")
            return path
        candidates = [self.zarr_root / "iceh_daily.zarr",
                      self.zarr_root / "history" / "iceh_daily.zarr"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not infer iceh_daily.zarr from the default AFIM layout. "
                                f"Tried: {', '.join(str(c) for c in candidates)}")

    def resolve_static_store(self) -> Path | None:
        if self.static_store is not None:
            path = Path(self.static_store).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Explicit static store does not exist: {path}")
            return path
        candidates = [self.zarr_root / "iceh_static.zarr",
                      self.zarr_root / "static" / "iceh_static.zarr"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def classification_store(self, method: str) -> Path:
        return self.classification_root_path / method_dirname(method,
                                                              bin_window   = self.classify.bin_window,
                                                              bin_min_days = self.classify.bin_min_days,
                                                              roll_window  = self.classify.roll_window) / "data.zarr"

    def metrics_store(self, method: str) -> Path:
        return self.classification_root_path / method_dirname(method,
                                                              bin_window   = self.classify.bin_window,
                                                              bin_min_days = self.classify.bin_min_days,
                                                              roll_window  = self.classify.roll_window) / "mets.zarr"

    # backward-compatible aliases for newer callers
    def resolve_class_store(self, method: str) -> Path:
        return self.classification_store(method)

    def resolve_metrics_store(self, method: str) -> Path:
        return self.metrics_store(method)

    def classification_log_path(self) -> Path:
        stem = (f"classify_{self.run.sim_name}_{self.classify.ice_type}_{self.classify.grid_type}"
                f"_ispd_thresh{threshold_tag_compact(self.classify.ispd_thresh)}"
                f"_BW{self.classify.bin_window}_BM{self.classify.bin_min_days}_roll{self.classify.roll_window}.log")
        return self.logs_root_path / "classification" / stem

    def metrics_log_path(self) -> Path:
        stem = (f"metrics_{self.run.sim_name}_{self.classify.ice_type}_{self.classify.grid_type}"
                f"_ispd_thresh{threshold_tag_compact(self.classify.ispd_thresh)}"
                f"_BW{self.classify.bin_window}_BM{self.classify.bin_min_days}_roll{self.classify.roll_window}.log")
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


    def multi_timeseries_plot_path(self, variable: str, method: str, simulations, *,
                                   region  : str        = "total",
                                   dt0_str : str | None = None,
                                   dtN_str : str | None = None) -> Path:
        var        = filename_token(str(variable).upper())
        norm       = filename_token(normalize_method(method))
        region_key = filename_token("total" if str(region).strip().lower() == "total" else str(region))
        dt0        = filename_token(dt0_str or self.run.start_date)
        dtN        = filename_token(dtN_str or self.run.end_date)
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
        name     = f"{var}_{sim_part}_{dt0}_{dtN}_{norm}_{region_key}.png"
        return self.graphics_root_path / "timeseries" / name

    def split_hemisphere_plot_path(self, variable: str, date_str: str) -> Path:
        return self.figure_root() / variable / f"{date_str}.png"

    def regional_var_plot_path(self, variable: str, date_str: str, region: str) -> Path:
        return self.figure_root(region=region) / variable / f"{date_str}.png"

    #-----------------------------------------------
    # OBSERVATIONS
    #----------------------------------------------
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
    def nsidc2cice_weight_file(self) -> Path:
        return self.wave_weights_root_path / "nsidc2cice_nearest.npz"

    @property
    def nsidc_aux_root_path(self) -> Path:
        obs = self.observations or ObservationSpec()
        if obs.nsidc_cellarea_root is not None:
            return Path(obs.nsidc_cellarea_root).expanduser()
        return self.seaice_root_path / "NSIDC" / obs.nsidc_cellarea_product

    #-----------------------------------------------
    # CAWCR
    #----------------------------------------------
    @property
    def cawcr_root_path(self) -> Path:
        return Path(f"/g/data/{self.run.project}/{self.run.user}/afim_input/CAWCR")

    @property
    def cawcr_org_root_path(self) -> Path:
        obs = self.observations or ObservationSpec()
        return self.cawcr_root_path / obs.cawcr_org_subdir

    @property
    def wave_weights_root_path(self) -> Path:
        return Path(f"/g/data/{self.run.project}/{self.run.user}/grids/weights")

    def cawcr_file(self, year: int, month: int) -> Path:
        return self.cawcr_root_path / "org" / f"ww3.{year:04d}{month:02d}_spec.nc"

    def cawcr_regridded_file(self, year: int, month: int) -> Path:
        return self.cawcr_root_path / f"CAWCR_efreq_for_CICE6_{year:04d}{month:02d}.nc"

    def cawcr2cice_weight_file(self, year: int, month: int) -> Path:
        return self.wave_weights_root_path / f"cawcr2cice_{year:04d}{month:02d}.npz"

    @property
    def regridded_wave_root_path(self) -> Path:
        wf = self.wave_forcing or WaveForcingSpec()
        if wf.regridded_wave_root is not None:
            return Path(wf.regridded_wave_root).expanduser()
        return self.cawcr_root_path

    def cawcr_figure_dir(self, year: int, month: int) -> Path:
        wf = self.wave_forcing or WaveForcingSpec()
        return self.graphics_root_path / wf.figure_subdir / f"{year:04d}{month:02d}"

    # @property
    # def cawcr_root_path(self) -> Path:
    #     obs = self.observations or ObservationSpec()
    #     if obs.cawcr_root is not None:
    #         return Path(obs.cawcr_root).expanduser()
    #     return Path(f"/g/data/{self.run.project}/{self.run.user}/afim_input/CAWCR")

    # def cawcr_file(self, year: int, month: int) -> Path:
    #     obs = self.observations or ObservationSpec()
    #     name = obs.cawcr_filename_template.format(year=year, month=month)
    #     return self.cawcr_org_root_path / name

    # @property
    # def wave_weights_root_path(self) -> Path:
    #     wf = self.wave_forcing or WaveForcingSpec()
    #     if wf.weights_root is not None:
    #         return Path(wf.weights_root).expanduser()
    #     return Path(f"/g/data/{self.run.project}/{self.run.user}/grids/weights")

    # def cawcr_regridded_file(self, year: int, month: int) -> Path:
    #     wf = self.wave_forcing or WaveForcingSpec()
    #     name = wf.regridded_wave_filename_template.format(year=year, month=month)
    #     return self.regridded_wave_root_path / name

    # def cawcr2cice_weight_file(self, year: int, month: int) -> Path:
    #     wf = self.wave_forcing or WaveForcingSpec()
    #     name = wf.cawcr2cice_weight_template.format(year=year, month=month)
    #     return self.wave_weights_root_path / name

    # @property
    # def nsidc2cice_weight_file(self) -> Path:
    #     wf = self.wave_forcing or WaveForcingSpec()
    #     return self.wave_weights_root_path / wf.nsidc2cice_weight_name

    # ------------------------------
    # CICE grid / ice_in resolution
    # ------------------------------
    @property
    def grids_root_path(self) -> Path:
        return Path(f"/g/data/{self.run.project}/{self.run.user}/grids")

    @property
    def cice_defaults(self) -> dict[str, Path]:
        spec = self.cice_grid or CICEGridSpec()
        return {
            "grid_file": Path(spec.default_grid_file).expanduser() if spec.default_grid_file is not None else self.grids_root_path / "ACCESS-OM3-025_Cgrid.nc",
            "kmt_file": Path(spec.default_kmt_file).expanduser() if spec.default_kmt_file is not None else self.grids_root_path / "ACCESS-OM3-025_kmt.nc",
            "bathymetry_file": Path(spec.default_bathymetry_file).expanduser() if spec.default_bathymetry_file is not None else self.grids_root_path / "unknown_bathymetry_file",
            "f2_file": Path(spec.default_f2_file).expanduser() if spec.default_f2_file is not None else self.form_factors_root_path / "combined.nc",
        }

    def resolve_ice_in_file(self) -> Path | None:
        spec = self.cice_grid or CICEGridSpec()
        if spec.ice_in_file is not None:
            path = Path(spec.ice_in_file).expanduser()
            return path if path.exists() else None

        roots = []
        if spec.experiment_root is not None:
            roots.append(Path(spec.experiment_root).expanduser())
        roots.extend(
            [
                self.output_root,
                Path(f"/g/data/{self.run.project}/{self.run.user}/simulations/{self.run.sim_name}"),
                Path(f"/g/data/{self.run.project}/{self.run.user}/experiments/{self.run.sim_name}"),
                Path.home() / self.run.sim_name,
            ]
        )

        candidates = []
        for root in roots:
            candidates.extend(
                [
                    root / "ice_in",
                    root / "run" / "ice_in",
                    root / "config" / "ice_in",
                    root / "work" / "ice_in",
                    root / "history" / "ice_in",
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _parse_ice_in_scalar_lines(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            if "=" not in s:
                continue
            key, value = s.split("=", 1)
            key = key.strip()
            value = value.strip().rstrip(",")
            if not key:
                continue
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            out[key] = value
        return out

    def resolve_cice_grid_assets(self) -> dict[str, Path | None]:
        spec = self.cice_grid or CICEGridSpec()
        defaults = self.cice_defaults
        resolved: dict[str, Path | None] = {
            "grid_file": Path(spec.grid_file).expanduser() if spec.grid_file is not None else None,
            "kmt_file": Path(spec.kmt_file).expanduser() if spec.kmt_file is not None else None,
            "bathymetry_file": Path(spec.bathymetry_file).expanduser() if spec.bathymetry_file is not None else None,
            "f2_file": Path(spec.f2_file).expanduser() if spec.f2_file is not None else None,
            "gridcpl_file": Path(spec.gridcpl_file).expanduser() if spec.gridcpl_file is not None else None,
            "ice_in_file": self.resolve_ice_in_file(),
        }
        ice_in = resolved["ice_in_file"]
        if ice_in is not None and ice_in.exists():
            parsed = self._parse_ice_in_scalar_lines(ice_in.read_text())
            mapping = {
                "grid_file": "grid_file",
                "kmt_file": "kmt_file",
                "bathymetry_file": "bathymetry_file",
                "F2_file": "f2_file",
                "gridcpl_file": "gridcpl_file",
            }
            for namelist_key, out_key in mapping.items():
                val = parsed.get(namelist_key)
                if val and not str(val).startswith("unknown_") and resolved[out_key] is None:
                    resolved[out_key] = Path(val).expanduser()

        for key in ("grid_file", "kmt_file", "bathymetry_file", "f2_file"):
            if resolved[key] is None:
                resolved[key] = defaults[key]
        return resolved

    @property
    def cice_grid_path(self) -> Path:
        path = self.resolve_cice_grid_assets()["grid_file"]
        assert path is not None
        return path

    @property
    def cice_kmt_path(self) -> Path:
        path = self.resolve_cice_grid_assets()["kmt_file"]
        assert path is not None
        return path

    @property
    def cice_bathymetry_path(self) -> Path | None:
        return self.resolve_cice_grid_assets()["bathymetry_file"]

    @property
    def cice_f2_path(self) -> Path | None:
        return self.resolve_cice_grid_assets()["f2_file"]

    @property
    def cice_gridcpl_path(self) -> Path | None:
        return self.resolve_cice_grid_assets()["gridcpl_file"]

    # ------------------------------
    # Lateral drag
    # ------------------------------
    @property
    def coastal_drag_root_path(self) -> Path:
        return Path(f"/g/data/{self.run.project}/{self.run.user}/coastal_drag")

    @property
    def form_factors_root_path(self) -> Path:
        return self.coastal_drag_root_path / "form_factors"

    @property
    def grounded_iceberg_file_path(self) -> Path:
        ld = self.lateral_drag or LateralDragSpec()
        if ld.grounded_iceberg_file is not None:
            return Path(ld.grounded_iceberg_file).expanduser()
        return Path(
            f"/g/data/{self.run.project}/{self.run.user}/grounded_icebergs/"
            "Kaihong_Jiao/Grounded_Icebergs_Full_Merged.gpkg"
        )

    @property
    def high_res_coast_file_path(self) -> Path:
        ld = self.lateral_drag or LateralDragSpec()
        if ld.high_res_coast_file is not None:
            return Path(ld.high_res_coast_file).expanduser()
        return Path(
            f"/g/data/{self.run.project}/{self.run.user}/coastlines/high_res_coast/"
            "add_coastline_high_res_polygon_v7_9.shp"
        )

    @property
    def coast_form_factors_path(self) -> Path:
        ld = self.lateral_drag or LateralDragSpec()
        if ld.coast_form_factors_file is not None:
            return Path(ld.coast_form_factors_file).expanduser()
        return self.form_factors_root_path / "coast.nc"

    @property
    def grounded_iceberg_form_factors_path(self) -> Path:
        ld = self.lateral_drag or LateralDragSpec()
        if ld.grounded_iceberg_form_factors_file is not None:
            return Path(ld.grounded_iceberg_form_factors_file).expanduser()
        return self.form_factors_root_path / "grounded_icebergs.nc"

    @property
    def combined_form_factors_path(self) -> Path:
        ld = self.lateral_drag or LateralDragSpec()
        if ld.combined_form_factors_file is not None:
            return Path(ld.combined_form_factors_file).expanduser()
        return self.form_factors_root_path / "combined.nc"

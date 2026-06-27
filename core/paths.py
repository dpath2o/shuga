from __future__  import annotations
import re, json
from dataclasses import dataclass, replace
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
    run_cfg            : RunSpec            | None = None
    cls_cfg            : ClassificationSpec | None = None
    met_cfg            : MetricsSpec        | None = None
    plt_cfg            : PlottingSpec       | None = None
    obs_cfg            : ObservationSpec    | None = None
    wave_frcg_cfg      : WaveForcingSpec    | None = None
    G_cice_cfg         : CICEGridSpec       | None = None
    LD_cfg             : LateralDragSpec    | None = None
    default_project    : str = "gv90"
    default_user       : str = "da1339"
    default_hemisphere : str = "SH"
    gdata_root         : str | Path | None = None
    afim_output_root   : str | Path | None = None
    graphics_root      : str | Path | None = None
    logs_root          : str | Path | None = None
    cice_store         : str | Path | None = None
    static_store       : str | Path | None = None
    classification_root: str | Path | None = None
    archive_root       : str | Path | None = None

    def __post_init__(self) -> None:
        project = self.run_cfg.project if self.run_cfg is not None else self.default_project
        user    = self.run_cfg.user    if self.run_cfg is not None else self.default_user
        if self.gdata_root is None:
            self.gdata_root = Path("/g/data") / project / user
        else:
            self.gdata_root = Path(self.gdata_root).expanduser()
        if self.afim_output_root is None:
            self.afim_output_root = self.gdata_root / "afim_output"
        else:
            self.afim_output_root = Path(self.afim_output_root).expanduser()
        if self.graphics_root is None:
            self.graphics_root = self.gdata_root / "GRAPHICAL"
        else:
            self.graphics_root = Path(self.graphics_root).expanduser()
        if self.logs_root is None:
            self.logs_root = self.gdata_root / "logs"
        else:
            self.logs_root = Path(self.logs_root).expanduser()
        if self.archive_root is None:
            self.archive_root = Path.home() / "AFIM_archive"
        else:
            self.archive_root = Path(self.archive_root).expanduser()
        if self.static_store is not None:
            self.static_store = Path(self.static_store).expanduser()
        if self.classification_root is not None:
            self.classification_root = Path(self.classification_root).expanduser()
        if self.cice_store is not None:
            self.cice_store = Path(self.cice_store).expanduser()

    #--------------------------------------------------------------------------------------------------
    # class properties
    #-------------------------------------------------------------------------------------------------
    @property
    def _project(self) -> str:
        return self.run_cfg.project if self.run_cfg is not None else self.default_project

    @property
    def _user(self) -> str:
        return self.run_cfg.user if self.run_cfg is not None else self.default_user

    @property
    def analysis_zarr_root_path(self) -> Path:
        archive_target = self.archive_zarr_root_path
        archive_history = self.archive_root_path / "history"
        if archive_target.exists() or archive_history.exists():
            return archive_target
        return self.zarr_root

    @property
    def ice_domain(self) -> str:
        cls_cfg = self._require_classify("ice_domain")
        token = str(cls_cfg.ice_type).strip().upper()
        if token not in {"FI", "PI", "SI"}:
            raise ValueError(f"Unsupported cls_cfg.ice_type={cls_cfg.ice_type!r}. Use 'FI', 'PI', or 'SI'.")
        return token

    @property
    def classification_root_path(self) -> Path:
        """
        Root for classification/metrics products for the active ice domain.

        SI is not speed-threshold or method dependent, so it lives directly under:
            zarr/HEMISPHERE/SI

        FI and PI are derived from speed-threshold methods and live under:
            zarr/HEMISPHERE/ispd_thresh_*/FI|PI/grid_type
        """
        if self.classification_root is not None:
            return Path(self.classification_root).expanduser()
        base   = self.analysis_zarr_root_path / self.hemisphere
        domain = self.ice_domain
        if domain == "SI":
            return base / "SI"
        cls_cfg = self._require_classify("classification_root_path")
        return (base / f"ispd_thresh_{threshold_tag_dir(cls_cfg.ispd_thresh)}" / domain / str(cls_cfg.grid_type))

    @property
    def hemisphere(self) -> str:
        value = self.run_cfg.hemisphere if self.run_cfg is not None else self.default_hemisphere
        return self.canonical_hemisphere(value)

    @property
    def output_root(self) -> Path:
        base = Path(self.afim_output_root).expanduser() if self.afim_output_root is not None else Path(f"/g/data/{self._project}/{self._user}/afim_output")
        if self.run_cfg is None:
            return base
        if base.name == self.run_cfg.sim_name:
            return base
        return base / self.run_cfg.sim_name

    @property
    def zarr_root(self) -> Path:
        return self.output_root / "zarr"

    @property
    def cice_grid_assets_config_path(self) -> Path:
        return self.output_root / "config" / "cice_grid_assets.json"

    @property
    def archive_root_path(self) -> Path:
        base = Path(self.archive_root).expanduser() if self.archive_root is not None else Path.home() / "AFIM_archive"
        if self.run_cfg is None:
            return base
        if base.name == self.run_cfg.sim_name:
            return base
        return base / self.run_cfg.sim_name

    @property
    def default_cice_static_store_path(self) -> Path:
        """
        Default persistent CICE static-coordinate store.

        This store is grid-level, not simulation-level, and is intended to be shared
        by AF2020 comparison, plotting, regridding, and other workflows.

        Expected default:
            ~/AFIM_archive/CICE_0p25_Cgrid_coords.zarr
        """
        return Path.home() / "AFIM_archive" / "CICE_0p25_Cgrid_coords.zarr"


    @property
    def cice_static_store_path(self) -> Path:
        """
        Return an existing CICE static-coordinate store, or raise if none exists.
        """
        path = self.resolve_static_store()
        if path is None:
            raise FileNotFoundError("Could not find a CICE static-coordinate zarr store. "
                                    f"Tried default={self.default_cice_static_store_path}, "
                                    f"static_store={self.static_store}, "
                                    f"zarr_root={self.zarr_root / 'iceh_static.zarr'}")
        return path

    @property
    def graphics_root_path(self) -> Path:
        if self.graphics_root is not None:
            return Path(self.graphics_root).expanduser()
        return Path(f"/g/data/{self._project}/{self._user}/GRAPHICAL")

    @property
    def logs_root_path(self) -> Path:
        if self.logs_root is not None:
            return Path(self.logs_root).expanduser()
        return Path.home() / "logs"

    @property
    def seaice_root_path(self) -> Path:
        obs = self.obs_cfg or ObservationSpec()
        if obs.seaice_root is not None:
            return Path(obs.seaice_root).expanduser()
        return Path(f"/g/data/{self._project}/{self._user}/SeaIce")

    @property
    def iceh_frequency(self) -> str:
        value = getattr(self.run_cfg, "iceh_frequency", "daily") if self.run_cfg is not None else "daily"
        token = str(value).strip().lower()
        if token not in {"daily", "hourly"}:
            raise ValueError(f"Unsupported iceh_frequency={token!r}")
        return token

    @property
    def archive_zarr_root_path(self) -> Path:
        return self.archive_root_path / "zarr"

    @property
    def iceh_store_name(self) -> str:
        if self.iceh_frequency == "hourly":
            return "iceh_hourly.zarr"
        return "iceh_daily.zarr"

    @property
    def fip_cmap(self) -> Path:
        return (self.graphics_root_path / "CPTs" / "AF2020_YlGnBu.cpt")

    @property
    def fip_diff_con_cmap(self) -> Path:
        return (self.graphics_root_path / "CPTs" / "FIPdiff_green_orange_blue.cpt")

    @property
    def fip_diff_cat_cmap(self) -> Path:
        return (self.graphics_root_path / "CPTs" / "FIP_diff_cat.cpt")

    @property
    def fi_obs_root_path(self) -> Path:
        obs = self.obs_cfg or ObservationSpec()
        if obs.af2020_root is not None:
            return Path(obs.af2020_root).expanduser()
        return self.seaice_root_path / "FI_obs"

    @property
    def af2020_fia_native_store_path(self) -> Path:
        obs = self.obs_cfg or ObservationSpec()
        return self.fi_obs_root_path / obs.af2020_fia_native_store

    @property
    def ld_pub_tables_root_path(self) -> Path:
        return Path(f"/g/data/{self._project}/{self._user}/GRAPHICAL/LD-pub-workspace/tables")

    @property
    def nsidc_root_path(self) -> Path:
        obs = self.obs_cfg or ObservationSpec()
        if obs.nsidc_root is not None:
            return Path(obs.nsidc_root).expanduser()
        return self.seaice_root_path / "NSIDC" / obs.nsidc_version

    @property
    def nsidc2cice_weight_file(self) -> Path:
        return self.wave_weights_root_path / "nsidc2cice_nearest.npz"

    @property
    def nsidc_aux_root_path(self) -> Path:
        obs = self.obs_cfg or ObservationSpec()
        if obs.nsidc_cellarea_root is not None:
            return Path(obs.nsidc_cellarea_root).expanduser()
        return self.seaice_root_path / "NSIDC" / obs.nsidc_cellarea_product

    @property
    def cawcr_root_path(self) -> Path:
        return Path(f"/g/data/{self._project}/{self._user}/afim_input/CAWCR")

    @property
    def cawcr_org_root_path(self) -> Path:
        obs = self.obs_cfg or ObservationSpec()
        return self.cawcr_root_path / obs.cawcr_org_subdir

    @property
    def wave_weights_root_path(self) -> Path:
        return Path(f"/g/data/{self._project}/{self._user}/grids/weights")

    @property
    def grids_root_path(self) -> Path:
        return Path(f"/g/data/{self._project}/{self._user}/grids")

    @property
    def cice_defaults(self) -> dict[str, Path]:
        spec = self.G_cice_cfg or CICEGridSpec()
        return {"grid_file"      : Path(spec.default_grid_file).expanduser() if spec.default_grid_file is not None else self.grids_root_path / "ACCESS-OM3-025_Cgrid.nc",
                "kmt_file"       : Path(spec.default_kmt_file).expanduser() if spec.default_kmt_file is not None else self.grids_root_path / "ACCESS-OM3-025_kmt.nc",
                "bathymetry_file": Path(spec.default_bathymetry_file).expanduser() if spec.default_bathymetry_file is not None else self.grids_root_path / "unknown_bathymetry_file",
                "f2_file"        : Path(spec.default_f2_file).expanduser() if spec.default_f2_file is not None else self.form_factors_root_path / "combined.nc"}

    @property
    def default_cice_grid_file_path(self) -> Path:
        """
        Generic default CICE/ACCESS-OM3 grid file.

        This is safe to use without run_cfg=RunSpec(...). It is intended for
        preprocessing workflows, such as ERA5 -> CICE regridding, that need
        the standard destination grid but are not tied to a specific CICE run.
        """
        return self.cice_defaults["grid_file"]

    @property
    def regridded_wave_root_path(self) -> Path:
        wf = self.wave_frcg_cfg or WaveForcingSpec()
        if wf.regridded_wave_root is not None:
            return Path(wf.regridded_wave_root).expanduser()
        return self.cawcr_root_path

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

    @property
    def coastal_drag_root_path(self) -> Path:
        return Path(f"/g/data/{self._project}/{self._user}/coastal_drag")

    @property
    def form_factors_root_path(self) -> Path:
        return self.coastal_drag_root_path / "form_factors"

    @property
    def grounded_iceberg_file_path(self) -> Path:
        ld = self.LD_cfg or LateralDragSpec()
        if ld.grounded_iceberg_file is not None:
            return Path(ld.grounded_iceberg_file).expanduser()
        return Path(f"/g/data/{self._project}/{self._user}/grounded_icebergs/Kaihong_Jiao/Grounded_Icebergs_Full_Merged.gpkg")

    @property
    def high_res_coast_file_path(self) -> Path:
        ld = self.LD_cfg or LateralDragSpec()
        if ld.high_res_coast_file is not None:
            return Path(ld.high_res_coast_file).expanduser()
        return Path(f"/g/data/{self._project}/{self._user}/coastlines/high_res_coast/add_coastline_high_res_polygon_v7_9.shp")

    @property
    def coast_form_factors_path(self) -> Path:
        ld = self.LD_cfg or LateralDragSpec()
        if ld.coast_form_factors_file is not None:
            return Path(ld.coast_form_factors_file).expanduser()
        return self.form_factors_root_path / "coast.nc"

    @property
    def grounded_iceberg_form_factors_path(self) -> Path:
        ld = self.LD_cfg or LateralDragSpec()
        if ld.grounded_iceberg_form_factors_file is not None:
            return Path(ld.grounded_iceberg_form_factors_file).expanduser()
        return self.form_factors_root_path / "grounded_icebergs.nc"

    @property
    def combined_form_factors_path(self) -> Path:
        ld = self.LD_cfg or LateralDragSpec()
        if ld.combined_form_factors_file is not None:
            return Path(ld.combined_form_factors_file).expanduser()
        return self.form_factors_root_path / "combined.nc"

    #--------------------------------------------------------------------------------------------------
    # helper functions
    #-------------------------------------------------------------------------------------------------
    def _require_run(self, caller: str) -> RunSpec:
        if self.run_cfg is None:
            raise ValueError(f"ShugaPaths.{caller} requires run_cfg=RunSpec(...). "
                             "Instantiate ShugaPaths with a RunSpec for simulation-specific paths.")
        return self.run_cfg

    def _require_classify(self, caller: str) -> ClassificationSpec:
        if self.cls_cfg is None:
            raise ValueError(f"ShugaPaths.{caller} requires cls_cfg=ClassificationSpec(...). "
                             "Instantiate ShugaPaths with a ClassificationSpec for classification/metrics paths.")
        return self.cls_cfg

    def with_ice_type(self, ice_type: str) -> "ShugaPaths":
        """
        Return a copy of this path bundle with a different classification/metrics
        ice-domain selector.
        """
        cls_cfg = self._require_classify("with_ice_type")
        return replace(self, cls_cfg = replace(cls_cfg, ice_type = str(ice_type).strip().upper()))

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

    @staticmethod
    def _parse_ice_in_scalar_lines(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            # Strip simple inline comments.
            for marker in ("!", "#"):
                if marker in s:
                    s = s.split(marker, 1)[0].strip()
            if "=" not in s:
                continue
            key, value = s.split("=", 1)
            key        = key.strip().lower()
            value      = value.strip().rstrip(",").strip()
            if not key:
                continue
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            out[key] = value
        return out

    @staticmethod
    def _grid_asset_keys() -> tuple[str, ...]:
        return("grid_file", "kmt_file", "bathymetry_file", "f2_file", "gridcpl_file", "ice_in_file", "ice_diag_file")
    #--------------------------------------------------------------------------------------------------
    # APIs
    #-------------------------------------------------------------------------------------------------
    def classification_store(self, method: str) -> Path:
        if self.ice_domain == "SI":
            return self.classification_root_path / "data.zarr"
        cls_cfg = self._require_classify("classification_store")
        return (self.classification_root_path / method_dirname(method,
                                                               bin_window   = cls_cfg.bin_window,
                                                               bin_min_days = cls_cfg.bin_min_days,
                                                               roll_window  = cls_cfg.roll_window) / "data.zarr")

    def metrics_store(self, method: str) -> Path:
        if self.ice_domain == "SI":
            return self.classification_root_path / "mets.zarr"
        cls_cfg = self._require_classify("metrics_store")
        return (self.classification_root_path / method_dirname(method,
                                                               bin_window   = cls_cfg.bin_window,
                                                               bin_min_days = cls_cfg.bin_min_days,
                                                               roll_window  = cls_cfg.roll_window) / "mets.zarr")

    def resolve_daily_iceh_root(self, daily_root: str | Path | None = None) -> Path:
        if daily_root is not None:
            path = Path(daily_root).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Explicit daily CICE NetCDF root does not exist: {path}")
            return path
        candidates = [self.archive_root_path / "history" / "daily", self.output_root / "history" / "daily"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not infer daily CICE NetCDF root from the default AFIM layout. Tried: {', '.join(str(c) for c in candidates)}")

    def resolve_iceh_history_root(self, history_root: str | Path | None = None, *, frequency: str | None = None) -> Path:
        freq = frequency or self.iceh_frequency
        if history_root is not None:
            path = Path(history_root).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Explicit CICE NetCDF root does not exist: {path}")
            return path
        candidates = [self.archive_root_path / "history" / freq, self.output_root / "history" / freq]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not infer {freq} CICE NetCDF root from the default AFIM layout. Tried: {', '.join(str(c) for c in candidates)}")

    def resolve_hourly_iceh_root(self, hourly_root: str | Path | None = None) -> Path:
        return self.resolve_iceh_history_root(hourly_root, frequency="hourly")

    def resolve_cice_store_target(self) -> Path:
        if self.cice_store is not None:
            return Path(self.cice_store).expanduser()
        store_name = self.iceh_store_name
        archive_target = self.archive_zarr_root_path / store_name
        output_target  = self.zarr_root / store_name
        # For archive-driven workflows, write beside ~/AFIM_archive/SIM_NAME/history/*
        # when that source tree exists. Otherwise preserve the /g/data output default.
        archive_history = self.archive_root_path / "history" / self.iceh_frequency
        if archive_history.exists() or archive_target.exists():
            return archive_target
        return output_target

    def resolve_cice_store(self) -> Path:
        store_name = self.iceh_store_name
        if self.cice_store is not None:
            candidates = [Path(self.cice_store).expanduser()]
        elif self.iceh_frequency == "hourly":
            candidates = [self.archive_zarr_root_path / store_name,
                          self.zarr_root / store_name,
                          self.archive_zarr_root_path / "history" / store_name,
                          self.zarr_root / "history" / store_name]
        else:
            candidates = [self.zarr_root / store_name,
                          self.archive_zarr_root_path / store_name,
                          self.zarr_root / "history" / store_name,
                          self.archive_zarr_root_path / "history" / store_name]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not infer {store_name} from the default AFIM layout. "
                                f"Tried: {', '.join(str(c) for c in candidates)}")

    def resolve_cice_grid_file(self, grid_file: str | Path | None = None) -> Path:
        """
        Resolve a CICE grid file for generic preprocessing workflows.

        If grid_file is supplied, it must be a full path. If omitted, use the
        shuga default CICE grid file from cice_defaults.
        """
        if grid_file is not None:
            return Path(grid_file).expanduser()
        return self.default_cice_grid_file_path

    def resolve_static_store_target(self) -> Path:
        """
        Return the target path for writing a persistent CICE static-coordinate store.
        """
        if self.static_store is not None:
            return Path(self.static_store).expanduser()
        return self.default_cice_static_store_path


    def resolve_static_store(self) -> Path | None:
        """
        Resolve an existing CICE static-coordinate zarr store.

        Search order:
          1. explicit/default self.static_store,
          2. shared archive static store,
          3. legacy /g/data afim_output shared store,
          4. simulation-local iceh_static.zarr fallbacks.
        """
        candidates: list[Path] = []
        if self.static_store is not None:
            candidates.append(Path(self.static_store).expanduser())
        candidates.extend([self.default_cice_static_store_path,
                           Path.home() / "AFIM_archive" / "CICE_0p25_Cgrid_coords.zarr",
                           self.afim_output_root / "CICE_0p25_Cgrid_coords.zarr",
                           self.zarr_root / "iceh_static.zarr",
                           self.zarr_root / "static" / "iceh_static.zarr",
                           self.archive_zarr_root_path / "iceh_static.zarr",
                           self.archive_zarr_root_path / "static" / "iceh_static.zarr"])
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique: list[Path] = []
        for candidate in candidates:
            p = Path(candidate).expanduser()
            key = str(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        for candidate in unique:
            if candidate.exists():
                return candidate
        return None

    # backward-compatible aliases for newer callers
    def resolve_class_store(self, method: str) -> Path:
        return self.classification_store(method)

    def resolve_metrics_store(self, method: str) -> Path:
        return self.metrics_store(method)

    def classification_log_path(self) -> Path:
        run_cfg = self._require_run("classification_log_path")
        cls_cfg = self._require_classify("classification_log_path")
        stem = (f"classify_{run_cfg.sim_name}_{cls_cfg.ice_type}_{cls_cfg.grid_type}"
                f"_ispd_thresh{threshold_tag_compact(cls_cfg.ispd_thresh)}"
                f"_BW{cls_cfg.bin_window}_BM{cls_cfg.bin_min_days}_roll{cls_cfg.roll_window}.log")
        return self.logs_root_path / "classification" / stem

    def metrics_log_path(self) -> Path:
        run_cfg = self._require_run("metrics_log_path")
        cls_cfg = self._require_classify("metrics_log_path")
        stem = (f"metrics_{run_cfg.sim_name}_{cls_cfg.ice_type}_{cls_cfg.grid_type}"
                f"_ispd_thresh{threshold_tag_compact(cls_cfg.ispd_thresh)}"
                f"_BW{cls_cfg.bin_window}_BM{cls_cfg.bin_min_days}_roll{cls_cfg.roll_window}.log")
        return self.logs_root_path / "metrics" / stem

    def lateral_drag_log_path(self) -> Path:
        return self.logs_root_path / "LD-form-factor" / "lateral_drag_form_factor_construction.log"

    def figure_root(self, region: str | None = None, *, sim_name: str | None = None) -> Path:
        sim   = sim_name or self._require_run("figure_root").sim_name
        parts = [self.graphics_root_path, sim]
        if region is not None:
            parts.append(str(region))
        path = parts[0]
        for part in parts[1:]:
            path = path / part
        return path

    def fip_plot_path(self, classification: str, *,
                      region    : str = "total",
                      sim_name  : str | None = None,
                      start_date: str | None = None,
                      end_date  : str | None = None) -> Path:
        """
        Return output path for a fast-ice persistence (FIP) plot.

        Parameters
        ----------
        classification : str
            Classification / method name, e.g. 'binary-days' or 'rolling-mean'.
        region : str, default 'total'
            Region key used in the graphical output tree.
        sim_name : str, optional
            Simulation name override. Defaults to self.run_cfg.sim_name.
        start_date : str, optional
            Plot start date override. Defaults to self.run_cfg.start_date.
        end_date : str, optional
            Plot end date override. Defaults to self.run_cfg.end_date.
        """
        norm   = normalize_method(classification)
        run_cfg    = self.run_cfg
        sim    = sim_name or (self._require_run("fip_plot_path").sim_name if run_cfg is None else run_cfg.sim_name)
        dt0    = start_date or (self._require_run("fip_plot_path").start_date if run_cfg is None else run_cfg.start_date)
        dtN    = end_date or (self._require_run("fip_plot_path").end_date if run_cfg is None else run_cfg.end_date)
        return (self.graphics_root_path / sim / region / "FIP" / f"{dt0}_{dtN}_{sim}_FIP_{norm.replace('-', '_')}.png")

    def timeseries_plot_path(self, variable: str, method: str, region: str = "total") -> Path:
        run_cfg     = self._require_run("timeseries_plot_path")
        method_part = method_slug(method)
        name        = f"{run_cfg.start_date}_{run_cfg.end_date}_{run_cfg.sim_name}_{variable}_{method_part}.png"
        P_          = self.figure_root(region=region) / "timeseries" / name
        P_.parent.mkdir(parents = True, exist_ok = True)
        return P_

    def multi_timeseries_plot_path(self, variable: str, method: str, simulations, *,
                                   region  : str        = "total",
                                   dt0_str : str | None = None,
                                   dtN_str : str | None = None) -> Path:
        var        = filename_token(str(variable).upper())
        norm       = filename_token(normalize_method(method))
        region_key = filename_token("total" if str(region).strip().lower() == "total" else str(region))
        run_cfg    = self.run_cfg
        if dt0_str is None or dtN_str is None:
            run_cfg = self._require_run("multi_timeseries_plot_path")
        dt0 = filename_token(dt0_str or run_cfg.start_date)
        dtN = filename_token(dtN_str or run_cfg.end_date)
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
    # waves
    #----------------------------------------------
    def cawcr_file(self, year: int, month: int) -> Path:
        return self.cawcr_root_path / "org" / f"ww3.{year:04d}{month:02d}_spec.nc"

    def cawcr_regridded_file(self, year: int, month: int) -> Path:
        return self.cawcr_root_path / f"CAWCR_efreq_for_CICE6_{year:04d}{month:02d}.nc"

    def cawcr2cice_weight_file(self, year: int, month: int) -> Path:
        return self.wave_weights_root_path / f"cawcr2cice_{year:04d}{month:02d}.npz"

    def cawcr_figure_dir(self, year: int, month: int) -> Path:
        wf = self.wave_frcg_cfg or WaveForcingSpec()
        return self.graphics_root_path / wf.figure_subdir / f"{year:04d}{month:02d}"

    # ------------------------------
    # CICE grid / ice_in resolution
    # ------------------------------
    def resolve_ice_in_file(self) -> Path | None:
        spec = self.G_cice_cfg or CICEGridSpec()
        if spec.ice_in_file is not None:
            path = Path(spec.ice_in_file).expanduser()
            return path if path.exists() else None
        roots = []
        if spec.experiment_root is not None:
            roots.append(Path(spec.experiment_root).expanduser())
        roots.extend([self.output_root,
                      Path(f"/g/data/{self.run_cfg.project}/{self.run_cfg.user}/simulations/{self.run_cfg.sim_name}"),
                      Path(f"/g/data/{self.run_cfg.project}/{self.run_cfg.user}/experiments/{self.run_cfg.sim_name}"),
                      Path.home() / self.run_cfg.sim_name])
        candidates = []
        for root in roots:
            candidates.extend([root / "ice_in",
                               root / "run" / "ice_in",
                               root / "config" / "ice_in",
                               root / "work" / "ice_in",
                               root / "history" / "ice_in"])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def resolve_ice_diag_file(self) -> Path | None:
        """
        Return the first available CICE diagnostic file for this simulation.

        This is intentionally searched under the simulation output root first,
        because AFIM commonly preserves ice_diag.d beside the simulation archive:

            /g/data/<project>/<user>/afim_output/<sim_name>/ice_diag.d
        """
        spec = self.G_cice_cfg or CICEGridSpec()
        explicit = getattr(spec, "ice_diag_file", None)
        if explicit is not None:
            path = Path(explicit).expanduser()
            return path if path.exists() else None
        roots = []
        if spec.experiment_root is not None:
            roots.append(Path(spec.experiment_root).expanduser())
        roots.extend([self.output_root,
                      self.archive_root_path,
                      Path(f"/g/data/{self.run_cfg.project}/{self.run_cfg.user}/simulations/{self.run_cfg.sim_name}"),
                      Path(f"/g/data/{self.run_cfg.project}/{self.run_cfg.user}/experiments/{self.run_cfg.sim_name}"),
                      Path.home() / self.run_cfg.sim_name])
        candidates: list[Path] = []
        for root in roots:
            candidates.extend([root / "ice_diag.d",
                               root / "run" / "ice_diag.d",
                               root / "config" / "ice_diag.d",
                               root / "work" / "ice_diag.d",
                               root / "history" / "ice_diag.d"])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def load_persisted_cice_grid_assets(self) -> dict[str, Path | None]:
        if self.run_cfg is None:
            return {key: None for key in self._grid_asset_keys()}
        cfg = self.cice_grid_assets_config_path
        if not cfg.exists():
            return {key: None for key in self._grid_asset_keys()}
        raw = json.loads(cfg.read_text())
        out: dict[str, Path | None] = {}
        for key in self._grid_asset_keys():
            value = raw.get(key)
            out[key] = Path(value).expanduser() if value not in (None, "", "null") else None
        return out

    def persist_cice_grid_assets(self, *, grid_spec: CICEGridSpec | None = None, overwrite: bool = True) -> Path:
        spec = grid_spec or self.G_cice_cfg or CICEGridSpec()
        payload: dict[str, str | None] = {}
        for key in self._grid_asset_keys():
            value = getattr(spec, key, None)
            payload[key] = str(Path(value).expanduser()) if value is not None else None
        target = self.cice_grid_assets_config_path
        target.parent.mkdir(parents = True, exist_ok = True)
        if target.exists() and not overwrite:
            return target
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return target

    def resolve_cice_grid_assets(self) -> dict[str, Path | None]:
        spec      = self.G_cice_cfg or CICEGridSpec()
        defaults  = self.cice_defaults
        persisted = self.load_persisted_cice_grid_assets()
        resolved: dict[str, Path | None] = {"grid_file"      : Path(spec.grid_file).expanduser() if spec.grid_file is not None else None,
                                            "kmt_file"       : Path(spec.kmt_file).expanduser() if spec.kmt_file is not None else None,
                                            "bathymetry_file": Path(spec.bathymetry_file).expanduser() if spec.bathymetry_file is not None else None,
                                            "f2_file"        : Path(spec.f2_file).expanduser() if spec.f2_file is not None else None,
                                            "gridcpl_file"   : Path(spec.gridcpl_file).expanduser() if spec.gridcpl_file is not None else None,
                                            "ice_in_file"    : Path(spec.ice_in_file).expanduser() if spec.ice_in_file is not None else None,
                                            "ice_diag_file"  : Path(getattr(spec, "ice_diag_file", "")).expanduser() if getattr(spec, "ice_diag_file", None) is not None else None}
        # Persisted config fills only missing explicit values.
        for key, value in persisted.items():
            if key in resolved and resolved[key] is None and value is not None:
                resolved[key] = value
        if self.run_cfg is not None:
            if resolved["ice_in_file"] is None:
                resolved["ice_in_file"] = self.resolve_ice_in_file()
            if resolved["ice_diag_file"] is None:
                resolved["ice_diag_file"] = self.resolve_ice_diag_file()
        def _path_from_metadata(parsed: dict[str, str], key: str) -> Path | None:
            raw = parsed.get(key.lower())
            if raw in (None, "", "none", "None", "unknown", "unknown_file"):
                return None
            return Path(raw).expanduser()
        # Prefer ice_in; use ice_diag.d only to fill gaps.
        for meta_key in ("ice_in_file", "ice_diag_file"):
            meta_file = resolved.get(meta_key)
            if meta_file is None or not meta_file.exists():
                continue
            parsed  = self._parse_ice_in_scalar_lines(meta_file.read_text(errors="ignore"))
            aliases = {"grid_file"      : ("grid_file",),
                       "kmt_file"       : ("kmt_file", "mask_file"),
                       "bathymetry_file": ("bathymetry_file", "bathy_file", "topography_file", "topog_file"),
                       "f2_file"        : ("f2_file", "F2_file".lower()),
                       "gridcpl_file"   : ("gridcpl_file",)}
            for out_key, candidate_keys in aliases.items():
                if resolved[out_key] is not None:
                    continue
                for candidate_key in candidate_keys:
                    value = _path_from_metadata(parsed, candidate_key)
                    if value is not None:
                        resolved[out_key] = value
                        break
        # Defaults are allowed only after explicit/persisted/metadata resolution.
        # This fixes the current dict/getattr mismatch in this method.
        for key in ("grid_file", "kmt_file", "bathymetry_file", "f2_file", "gridcpl_file"):
            if resolved.get(key) is None:
                value = defaults.get(key)
                if value is not None:
                    resolved[key] = Path(value).expanduser()
        return resolved


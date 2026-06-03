from __future__  import annotations
from dataclasses import dataclass
from pathlib     import Path
from typing      import Iterable, Sequence, Literal

IcehFrequency = Literal["daily", "hourly"]

@dataclass(slots=True)
class RunSpec:
    sim_name      : str
    start_date    : str
    end_date      : str
    hemisphere    : str                 = "SH"
    project       : str                 = "gv90"
    user          : str                 = "da1339"
    iceh_frequency: IcehFrequency | str = "daily"
    def __post_init__(self) -> None:
        token   = str(self.iceh_frequency).strip().lower().replace("_", "-")
        aliases = {"d"            : "daily",
                   "day"          : "daily",
                   "daily"        : "daily",
                   "h"            : "hourly",
                   "hour"         : "hourly",
                   "hours"        : "hourly",
                   "hourly"       : "hourly",
                   "inst"         : "hourly",
                   "instantaneous": "hourly"}
        if token not in aliases:
            raise ValueError(f"Unsupported iceh_frequency={self.iceh_frequency!r}. "
                             "Use 'daily' or 'hourly'.")
        self.iceh_frequency = aliases[token]

@dataclass(slots=True)
class ClassificationSpec:
    ice_type     : str = "FI"
    grid_type    : str | Sequence[str] = "Tc"
    ispd_thresh  : float = 5e-4
    methods      : str | Sequence[str] = ("raw", "binary-days", "rolling-mean")
    bin_window   : int = 11
    bin_min_days : int = 9
    roll_window  : int = 15
    speed_var_u  : str = "uvel"
    speed_var_v  : str = "vvel"
    uvelE_var    : str = "uvelE"
    uvelN_var    : str = "uvelN"
    vvelE_var    : str = "vvelE"
    vvelN_var    : str = "vvelN"
    aice_var     : str = "aice"
    aice_thresh  : float = 0.15
    wrap_x       : bool = True
    cgrid_combine: str = "mean"
    def __post_init__(self) -> None:
        if isinstance(self.methods, str):
            self.methods = (self.methods,)
        else:
            self.methods = tuple(self.methods)

    @classmethod
    def from_methods(cls, methods: Iterable[str] | None = None, **kwargs) -> "ClassificationSpec":
        obj = cls(**kwargs)
        if methods is not None:
            obj.methods = (methods,) if isinstance(methods, str) else tuple(methods)
        return obj

@dataclass(slots=True)
class MetricsSpec:
    methods           : str | Sequence[str] = ("binary-days", "rolling-mean")
    obs_metrics_store : str | None = None
    obs_fia_var       : str = "FIA"
    obs_fit_var       : str = "FIT"
    coast_distance_var: str | None = None
    area_scale        : float = 1e9
    volume_scale      : float = 1e12

@dataclass(slots=True)
class PlottingSpec:
    fig_size          : float             = 20.0
    fip_fig_size      : float             = 20.0
    split_fig_size    : float             = 14.0
    region_fig_size   : float             = 20.0
    shorelines        : str               = "0.25p,black"
    land              : str               = "lightgray"
    water             : str               = "white"
    grid_style        : str               = "s0.1c"
    nsidc_pen         : str               = "1p,green"
    fip_cmap          : str | Path | None = None
    colorbar_position : str               = "JMB+w8c/0.4c+o0.8c/0c"
    colorbar_xlabel   : str | None        = None
    colorbar_ylabel   : str | None        = None

@dataclass(slots=True)
class ObservationSpec:
    seaice_root              : str | Path | None = None
    nsidc_root               : str | Path | None = None
    nsidc_version            : str = "G02202_V6"
    nsidc_cellarea_root      : str | Path | None = None
    nsidc_cellarea_product   : str = "NSIDC0771"
    nsidc_resolution_km      : float = 25.0
    nsidc_sic_var            : str = "cdr_seaice_conc"
    nsidc_threshold          : float = 0.15
    af2020_root              : str | Path | None = None
    af2020_fia_daily_file    : str = "FIA_AF2020_daily.nc"
    af2020_fia_daily_var     : str = "AF2020"
    af2020_regridded_store   : str = "AF-FI-2020db_org-timestep_reG.zarr"
    af2020_climatology_store : str = "AF-FI-2020db_18yrAvg_gridded.zarr"
    af2020_regridded_var     : str = "FI"
    af2020_climatology_var   : str = "FI_OBS_GRD"
    af2020_time_var          : str = "t_FI_obs"
    af2020_doy_var           : str = "doy"
    # CAWCR raw observational input
    cawcr_root               : str | Path | None = None
    cawcr_org_subdir         : str = "org"
    cawcr_filename_template  : str = "ww3.{year:04d}{month:02d}_spec.nc"
    # CAWCR variable / dimension names
    cawcr_spectrum_var       : str = "Efth"
    cawcr_lon_var            : str = "longitude"
    cawcr_lat_var            : str = "latitude"
    cawcr_time_var           : str = "time"
    cawcr_station_dim        : str = "station"
    cawcr_frequency_dim      : str = "frequency"
    cawcr_direction_dim      : str = "direction"
    cawcr_frequency_var      : str = "frequency"
    cawcr_frequency_lower_var: str = "frequency1"
    cawcr_frequency_upper_var: str = "frequency2"
# optional alias if you want the plural spelling in new code
ObservationsSpec = ObservationSpec

@dataclass(slots=True)
class CICEGridSpec:
    grid_file              : str | Path | None = None
    kmt_file               : str | Path | None = None
    bathymetry_file        : str | Path | None = None
    f2_file                : str | Path | None = None
    gridcpl_file           : str | Path | None = None
    ice_in_file            : str | Path | None = None
    experiment_root        : str | Path | None = None
    grid_format            : str = "nc"
    grid_type              : str | None = None
    lon_type               : str = "-180-180"
    default_grid_file      : str | Path | None = None
    default_kmt_file       : str | Path | None = None
    default_bathymetry_file: str | Path | None = None
    default_f2_file        : str | Path | None = None

@dataclass(slots=True)
class WaveForcingSpec:
    regridded_wave_root             : str | Path | None = None
    weights_root                    : str | Path | None = None
    regridded_wave_filename_template: str = "CAWCR_efreq_for_CICE6_{year:04d}{month:02d}.nc"
    cawcr2cice_weight_template      : str = "cawcr2cice_{year:04d}{month:02d}.npz"
    nsidc2cice_weight_name          : str = "nsidc2cice_nearest.npz"
    figure_subdir                   : str = "LD-waves/CAWCR"

@dataclass(slots=True)
class LateralDragSpec:
    f2_map_method                     : str = "max"
    lat_subset_max                    : float = -30.0
    proj_crs                          : str = "EPSG:3031"
    max_assign_km                     : float = 50.0
    coast_buffer_cells                : int = 1
    use_coastal_ocean_kdtree          : bool = True
    chunk_segments                    : int = 2_000_000
    netcdf_compression                : int = 4
    grounded_iceberg_file             : str | Path | None = None
    high_res_coast_file               : str | Path | None = None
    coast_form_factors_file           : str | Path | None = None
    grounded_iceberg_form_factors_file: str | Path | None = None
    combined_form_factors_file        : str | Path | None = "/g/data/gv90/da1339/coastal_drag/form_factors/ADD_high-res_cstln_v7p9_GI_CICE_free-slip.nc"

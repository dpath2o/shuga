#!/usr/bin/env python3
from __future__ import annotations
import os, sys, warnings, argparse, pygmt
import numpy             as np
import pandas            as pd
import xarray            as xr
from dataclasses         import dataclass, replace 
from pathlib             import Path
from math                import ceil
xr.set_options(keep_attrs=True)
warnings.filterwarnings("default")
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import shuga
from shuga.plotting.cice import CICEPlotter
from shuga               import configs
from shuga               import loaders
from shuga               import regions
START        = "2000-01-01"
END          = "2005-12-31"
ICE_TYPE     = "FI"
METHOD       = "binary-days"
FIG_SIZE     = 15
GRD_STY      = "c0.15c"
pth_cfg      = configs.ShugaPaths(LD_cfg = configs.LateralDragSpec())
P_CPT_FIP    = pth_cfg.fip_cmap
P_F2         = pth_cfg.combined_form_factors_path
D_pub        = pth_cfg.graphics_root_path / "LD-pub-workspace"
EXPS_STATIC  = ["Cs-high", "Cs-high-ktens-mid", "Cs-high-ktens-high", "Cs-high-eDef", "Cs-mid", "Cs-low"]
EXPS_QUAD    = ["Cq-high", "Cq-mid", "Cq-low"]
EXPS_LINEAR  = ["Cl-mid", "Cl-low"]
EXPS_BLEND   = ["blend-strain-high", "blend-strain-mid", "blend-strain-low"]
EXPS_NO_SLIP = ["no-slip-LFI"]
EXPS_ALL     = EXPS_STATIC + EXPS_QUAD + EXPS_LINEAR + EXPS_BLEND + EXPS_NO_SLIP
DYN_VARS     = ["aice", "hi", "tarea", "TLON", "TLAT", "uvel", "vvel", "uocn", "vocn",
                "divu", "shear", "sig1", "sig2", "sigP", "strength",
                "strairx", "strairy", "strocnx", "strocny", "strintx", "strinty", "strcorx", "strcory", "strtltx", "strtlty",
                "KuxE", "KuyE", "KuxN", "KuyN", "F2E", "F2N"]
FLD_NAME     = "FIP-weighted-FI-thickness_and_LD-tau"
tau_mnmx     = [0, 3]
met_mnmx     = [0, 5]
cmap_tau     = "cmocean/turbid"
cmap_met     = "cmocean/matter"
cbar_tau     = ["xaf", "x+lstress", "y+l(N m@+-2@+)"]
cbar_met     = ["xaf", "x+lFIP-weighted thickness", "y+lm"]
overwrite    = True
FF           = xr.open_dataset(P_F2)
# Use GICB-only form-factor for visual overlay; coastal part is redundant with fig.coast().
F2_mag       = xr.apply_ufunc(np.hypot, FF["F2x_gi"], FF["F2y_gi"], dask = "allowed")
# Explicit binary mask; keeps non-F2 cells as 0.0, not NaN, so GMT can contour the 0/1 transition.
F2_bin       = xr.where(np.isfinite(F2_mag) & (F2_mag > 0.0), 1.0, 0.0)
for sim_name in EXPS_ALL:
    run_cfg  = configs.RunSpec(sim_name = sim_name, start_date = START, end_date = END)
    cls_cfg  = configs.ClassificationSpec(ice_type = ICE_TYPE, methods = METHOD)
    met_cfg  = configs.MetricsSpec(methods = METHOD)
    plt_cfg  = configs.PlottingSpec()
    obs_cfg  = configs.ObservationSpec()
    pltr     = CICEPlotter(run_cfg = run_cfg, cls_cfg = cls_cfg, met_cfg = met_cfg, plt_cfg = plt_cfg, obs_cfg = obs_cfg)
    coords   = pltr._load_static_lonlat()
    lon      = coords["TLON"]
    lat      = coords["TLAT"]
    # ------------------------------------------------------------------
    # Persistence-weighted fast-ice thickness
    #
    # FIHI is conditional mean thickness over FI days.
    # FIP is persistence, stored as percent in the metrics store.
    #
    # FIHI_p = FIHI * FIP_fraction
    #        = mean(h | FI) * P(FI)
    #        = mean(h * I_FI)
    # ------------------------------------------------------------------
    mets     = shuga.load_metrics(sim_name = sim_name, classification = METHOD)
    FIHI     = mets["FIHI"]
    FIP      = mets["FIP"] / 100.0
    FIHI_p   = FIHI * FIP
    # Optional: do not show cells with zero persistence
    FIHI_p   = FIHI_p.where(FIP > 0.0)
    # ------------------------------------------------------------------
    # Lateral-drag stress
    # ------------------------------------------------------------------
    cice_dly = shuga.load_cice(sim_name = sim_name, variables = ["KuxE", "KuyE", "KuxN", "KuyN"], dt0_str = START, dtN_str = END)
    cls_ds   = shuga.load_classified(sim_name = sim_name, classification = METHOD, variables = ["FI_mask"], dt0_str = START, dtN_str = END)
    FI_mask  = cls_ds["FI_mask"].astype(bool)
    KuxE, KuyE, KuxN, KuyN, FI_mask = xr.align(cice_dly["KuxE"], cice_dly["KuyE"], cice_dly["KuxN"], cice_dly["KuyN"], FI_mask, join = "inner")
    KuE      = np.hypot(KuxE, KuyE)
    KuN      = np.hypot(KuxN, KuyN)
    Ku       = 0.5 * (KuE + KuN)
    # 90th percentile LD stress during FI conditions
    Ku       = Ku.where(FI_mask)
    Ku       = Ku.quantile(0.90, dim = "time", skipna = True)
    Ku       = Ku.drop_vars("quantile", errors = "ignore")
    # ------------------------------------------------------------------
    # Plot regions
    # ------------------------------------------------------------------
    for REG_NAME in regions.ANTARCTIC_8_REGIONS.keys():
        D_s_f_r = D_pub / sim_name / FLD_NAME / REG_NAME
        D_s_f_r.mkdir(parents = True, exist_ok = True)
        P_png   = D_s_f_r / f"{START}_{END}.png"
        if P_png.exists() and not overwrite:
            print(f"{P_png} exists and not overwriting")
            continue
        REG_PLOT = regions.ANTARCTIC_8_REGIONS[REG_NAME]["plot_region"]
        plt_FIHI = pltr.pygmt_da_prep(FIHI_p, lon = lon, lat = lat, mask_zero = False, region = REG_PLOT)
        plt_tau  = pltr.pygmt_da_prep(Ku    , lon = lon, lat = lat, mask_zero = False, region = REG_PLOT)
        plt_F2   = pltr.pygmt_da_prep(F2_bin, lon = lon, lat = lat, mask_zero = False, region = REG_PLOT)
        fig      = pygmt.Figure()
        proj     = pltr.projection_from_region(REG_PLOT, fig_size = FIG_SIZE)
        fig.basemap(region = REG_PLOT, projection = proj, frame = ["af", r"+t@[Q_{0.90,t}(|\boldsymbol{\tau}_{\mathrm{LD},u}|)@["])
        fig.coast(shorelines = "0.25p,black", land = "lightgray", water = "white")
        pygmt.makecpt(cmap = cmap_tau, series = tau_mnmx, background = True)
        fig.plot(x = plt_tau["lon"], y = plt_tau["lat"], style = GRD_STY, fill = plt_tau["z"], cmap = True, pen = None)
        fig.contour(x = plt_F2["lon"], y = plt_F2["lat"], z = plt_F2["z"], levels = [0.5], pen = "0.45p,blue")
        fig.colorbar(position = "JBC+w6.5c+o0c/-0.1c+h", frame = cbar_tau)
        fig.shift_origin(xshift = "w+10c", yshift = "0c")
        fig.basemap(region = REG_PLOT, projection = proj, frame = ["af", r"+t@[\overline{h I_{\mathrm{FI}}}_{t}@["])
        fig.coast(shorelines = "0.25p,black", land = "lightgray", water = "white")
        pygmt.makecpt(cmap = cmap_met, series = met_mnmx, background = True)
        fig.plot(x = plt_FIHI["lon"], y = plt_FIHI["lat"], style = GRD_STY, fill = plt_FIHI["z"], cmap = True, pen = None)
        fig.colorbar(position = "JBC+w6.5c+o0c/-0.1c+h", frame = cbar_met)
        if overwrite:
            print(f"writing to {P_png}")
            fig.savefig(P_png)

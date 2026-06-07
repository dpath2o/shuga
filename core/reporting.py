from __future__             import annotations
import json, re
import pandas               as pd
import xarray               as xr
from calendar               import monthrange
from dataclasses            import asdict, dataclass, field
from pathlib                import Path
from typing                 import Any
from .naming                import normalize_method
from .paths                 import ShugaPaths
from .store_selection       import StoreSelection
from .types                 import ClassificationSpec, RunSpec
from shuga.io.store_locator import CICEStoreLocator

_MONTH_RE  = re.compile(r"^(\d{4})-(\d{2})$")
_THRESH_RE = re.compile(r"^ispd_thresh_(.+)$")
_BIN_RE    = re.compile(r"^bin-win-(\d+)_bin-min-(\d+)$")
_ROLL_RE   = re.compile(r"^roll-days-(\d+)$")

@dataclass(slots=True)
class StoreHealth:
    status           : str
    reasons          : list[str] = field(default_factory=list)
    variables_present: list[str] = field(default_factory=list)
    variables_missing: list[str] = field(default_factory=list)

@dataclass(slots=True)
class TimeCoverage:
    start_date: str | None = None
    end_date  : str | None = None
    n_time    : int | None = None
    n_groups  : int | None = None

@dataclass(slots=True)
class CICEStoreStatus:
    path         : str
    exists       : bool
    coverage     : TimeCoverage | None = None
    static_store : str | None = None
    static_exists: bool | None = None

@dataclass(slots=True)
class ClassificationStatus:
    method             : str
    grid_type          : str
    classification_path: str
    metrics_path       : str
    ice_type           : str | None = None
    ispd_thresh        : str | None = None
    aice_thresh        : float | str | None = None
    bin_window         : int | None = None
    bin_min_days       : int | None = None
    roll_window        : int | None = None
    coverage           : TimeCoverage | None = None
    metrics_exists     : bool = False
    metrics_health     : StoreHealth | None = None

@dataclass(slots=True)
class IceInStatus:
    path    : str | None = None
    exists  : bool = False
    keys    : list[str] = field(default_factory=list)
    preview : dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class SimulationStatusReport:
    sim_name       : str
    hemisphere     : str
    output_root    : str
    zarr_root      : str
    cice           : CICEStoreStatus
    classifications: list[ClassificationStatus] = field(default_factory=list)
    ice_in_json    : IceInStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Simulation: {self.sim_name}")
        lines.append(f"Hemisphere: {self.hemisphere}")
        lines.append(f"Output root: {self.output_root}")
        lines.append(f"Zarr root: {self.zarr_root}")
        lines.append("")
        lines.append("CICE history store")
        lines.append(f"  path: {self.cice.path}")
        lines.append(f"  exists: {self.cice.exists}")
        if self.cice.coverage is not None:
            lines.append(f"  coverage: {self.cice.coverage.start_date} -> {self.cice.coverage.end_date}"
                         f" ({self.cice.coverage.n_groups} monthly groups, n_time={self.cice.coverage.n_time})")
        if self.cice.static_store is not None:
            lines.append(f"  static store: {self.cice.static_store}")
            lines.append(f"  static exists: {self.cice.static_exists}")
        lines.append("")
        if self.classifications:
            lines.append("Classifications")
            for i, cls in enumerate(self.classifications, start=1):
                lines.append(f"  {i}. {cls.method} [{cls.grid_type}]")
                if cls.ice_type is not None:
                    lines.append(f"     ice_type: {cls.ice_type}")
                if cls.ispd_thresh is not None:
                    lines.append(f"     ispd_thresh: {cls.ispd_thresh}")
                if cls.aice_thresh is not None:
                    lines.append(f"     aice_thresh: {cls.aice_thresh}")
                if cls.bin_window is not None:
                    lines.append(f"     binary-days: window={cls.bin_window}, min_days={cls.bin_min_days}")
                if cls.roll_window is not None:
                    lines.append(f"     rolling-mean: window={cls.roll_window}")
                lines.append(f"     class store: {cls.classification_path}")
                if cls.coverage is not None:
                    lines.append(f"     coverage: {cls.coverage.start_date} -> {cls.coverage.end_date}"
                                 f" (n_time={cls.coverage.n_time})")
                lines.append(f"     metrics: {'yes' if cls.metrics_exists else 'no'}")
                if cls.metrics_health is not None:
                    lines.append(f"     metrics health: {cls.metrics_health.status}")
                    if cls.metrics_health.reasons:
                        for reason in cls.metrics_health.reasons:
                            lines.append(f"       - {reason}")
        else:
            lines.append("Classifications")
            lines.append("  none discovered")
        lines.append("")
        if self.ice_in_json is not None:
            lines.append("ice_in JSON")
            lines.append(f"  exists: {self.ice_in_json.exists}")
            if self.ice_in_json.path:
                lines.append(f"  path: {self.ice_in_json.path}")
            if self.ice_in_json.keys:
                lines.append(f"  keys: {', '.join(self.ice_in_json.keys[:12])}")
            if self.ice_in_json.preview:
                lines.append("  preview:")
                for k, v in self.ice_in_json.preview.items():
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

#-----------------------------------------------------------------------------------
# primary function/API of this module:
#-----------------------------------------------------------------------------------
def report_sim_status(sim_name        : str | None = None, *,
                      run_cfg             : RunSpec | None = None,
                      cls_cfg        : ClassificationSpec | None = None,
                      hemisphere      : str | None = None,
                      project         : str | None = None,
                      user            : str | None = None,
                      afim_output_root: str | Path | None = None,
                      echo            : bool = True,
                      logger                 = None) -> SimulationStatusReport:
    """
    Inspect the AFIM/Shuga on-disk layout for one simulation and report:
      - iceh_daily.zarr coverage
      - discovered classification products and settings
      - metrics presence and a simple health check
      - ice_in_AFIM_subset_[SIM_NAME].json if present

    This is intended as a lightweight Jupyter-friendly diagnostic entry point.
    """
    if run_cfg is None and not sim_name:
        raise ValueError("report_sim_status() requires sim_name=... or run_cfg=RunSpec(...).")
    if run_cfg is None:
        run_cfg = RunSpec(sim_name   = str(sim_name),
                          start_date = "1900-01-01",
                          end_date   = "1900-01-01",
                          hemisphere = hemisphere or "SH",
                          project    = project or RunSpec.__dataclass_fields__["project"].default,
                          user       = user or RunSpec.__dataclass_fields__["user"].default)
    else:
        if sim_name is not None and sim_name != run_cfg.sim_name:
            run_cfg = RunSpec(sim_name   = sim_name,
                              start_date = run_cfg.start_date,
                              end_date   = run_cfg.end_date,
                              hemisphere = hemisphere or run_cfg.hemisphere,
                              project    = project or run_cfg.project,
                              user       = user or run_cfg.user)
        elif hemisphere is not None or project is not None or user is not None:
            run_cfg = RunSpec(sim_name   = run_cfg.sim_name,
                              start_date = run_cfg.start_date,
                              end_date   = run_cfg.end_date,
                              hemisphere = hemisphere or run_cfg.hemisphere,
                              project    = project or run_cfg.project,
                              user       = user or run_cfg.user)
    classify_eff = cls_cfg or ClassificationSpec()
    pth_cfg        = ShugaPaths(run_cfg = run_cfg, cls_cfg = classify_eff, afim_output_root = afim_output_root)
    try:
        cice_store = pth_cfg.resolve_cice_store()
        cice_exists = cice_store.exists()
    except FileNotFoundError:
        cice_store = pth_cfg.zarr_root / "iceh_daily.zarr"
        cice_exists = False
    static_store    = pth_cfg.resolve_static_store()
    cice_status     = CICEStoreStatus(path          = str(cice_store),
                                      exists        = cice_exists,
                                      coverage      = _infer_grouped_store_coverage(cice_store) if cice_exists else None,
                                      static_store  = str(static_store) if static_store is not None else None,
                                      static_exists = static_store.exists() if static_store is not None else None)
    classifications = _discover_classifications(pth_cfg, cls_cfg=classify_eff, logger=logger)
    ice_in          = _discover_ice_in_json(pth_cfg, run_cfg.sim_name)
    report          = SimulationStatusReport(sim_name        = run_cfg.sim_name,
                                             hemisphere      = pth_cfg.hemisphere,
                                             output_root     = str(pth_cfg.output_root),
                                             zarr_root       = str(pth_cfg.zarr_root),
                                             cice            = cice_status,
                                             classifications = classifications,
                                             ice_in_json     = ice_in)
    if echo:
        print(report.to_text())
    return report


#-----------------------------------------------------------------------------------
# helper functions outside any class/module
#-----------------------------------------------------------------------------------
def _maybe_open_time_coverage(store: Path, *, group: str | None = None) -> TimeCoverage | None:
    try:
        ds = xr.open_zarr(store, group=group, consolidated=False)
    except Exception:
        return None
    if "time" not in ds.coords:
        return TimeCoverage(start_date=None, end_date=None, n_time=None, n_groups=1 if group else None)
    n_time = int(ds.sizes.get("time", 0))
    if n_time == 0:
        return TimeCoverage(start_date=None, end_date=None, n_time=0, n_groups=1 if group else None)
    try:
        t0 = pd.to_datetime(ds["time"].isel(time=0).values).strftime("%Y-%m-%d")
        tN = pd.to_datetime(ds["time"].isel(time=-1).values).strftime("%Y-%m-%d")
    except Exception:
        t0 = tN = None
    return TimeCoverage(start_date=t0, end_date=tN, n_time=n_time, n_groups=1 if group else None)

def _infer_grouped_store_coverage(zarr_root: Path) -> TimeCoverage | None:
    if not zarr_root.exists():
        return None
    groups = sorted(p.name for p in zarr_root.iterdir() if p.is_dir() and _MONTH_RE.match(p.name))
    if not groups:
        return _maybe_open_time_coverage(zarr_root)
    first_group = groups[0]
    last_group  = groups[-1]
    cov0        = _maybe_open_time_coverage(zarr_root, group=first_group)
    covN        = _maybe_open_time_coverage(zarr_root, group=last_group)
    if cov0 is not None and covN is not None and cov0.start_date and covN.end_date:
        n_time = None
        if cov0.n_time is not None and covN.n_time is not None:
            # exact total time would require opening all groups; omit by default
            n_time = None
        return TimeCoverage(start_date = cov0.start_date,
                            end_date   = covN.end_date,
                            n_time     = n_time,
                            n_groups   = len(groups))
    y0, m0 = map(int, first_group.split("-"))
    yN, mN = map(int, last_group.split("-"))
    return TimeCoverage(start_date=f"{y0:04d}-{m0:02d}-01",
                        end_date=f"{yN:04d}-{mN:02d}-{monthrange(yN, mN)[1]:02d}",
                        n_time=None,
                        n_groups=len(groups))

def _discover_ice_in_json(pth_cfg: ShugaPaths, sim_name: str) -> IceInStatus:
    filename   = f"ice_in_AFIM_subset_{sim_name}.json"
    candidates = [pth_cfg.output_root / filename,
                  pth_cfg.output_root / "config" / filename,
                  pth_cfg.output_root / "configs" / filename,
                  pth_cfg.output_root.parent / filename,
                  Path.home() / "AFIM" / filename]
    found      = next((p for p in candidates if p.exists()), None)
    if found is None:
        return IceInStatus(path=str(candidates[0]), exists=False)
    try:
        payload = json.loads(found.read_text())
    except Exception:
        return IceInStatus(path=str(found), exists=True)
    preview_keys = ["dt0_str", "dtN_str", "grid_type", "ice_type", "ispd_thresh",
                    "aice_thresh", "bin_window", "bin_min_days", "roll_window"]
    preview = {k: payload.get(k) for k in preview_keys if k in payload}
    return IceInStatus(path    = str(found),
                       exists  = True,
                       keys    = sorted(payload.keys()),
                       preview = preview)

def _parse_store_metadata_from_path(path: Path, *, fallback_classify: ClassificationSpec | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"ice_type": None,
                           "grid_type": None,
                           "ispd_thresh": None,
                           "aice_thresh": getattr(fallback_classify, "aice_thresh", None),
                           "bin_window": None,
                           "bin_min_days": None,
                           "roll_window": None}
    parts = path.parts
    for i, token in enumerate(parts):
        m = _THRESH_RE.match(token)
        if m:
            out["ispd_thresh"] = m.group(1)
            if i + 2 < len(parts):
                out["ice_type"] = parts[i + 1]
                out["grid_type"] = parts[i + 2]
            break
    method_dir = path.parent.name
    if method_dir == "raw":
        return out
    m_bin = _BIN_RE.match(method_dir)
    if m_bin:
        out["bin_window"] = int(m_bin.group(1))
        out["bin_min_days"] = int(m_bin.group(2))
        return out
    m_roll = _ROLL_RE.match(method_dir)
    if m_roll:
        out["roll_window"] = int(m_roll.group(1))
    return out

def _metrics_health(metrics_store: Path) -> StoreHealth:
    expected = ["FIP", "FIPSI", "FIA", "FIT", "FIV"]
    try:
        ds = xr.open_zarr(metrics_store, consolidated=False)
    except Exception as e:
        return StoreHealth(status="broken", reasons=[f"open failed: {e}"])
    present = [v for v in expected if v in ds.data_vars or v in ds.coords]
    missing = [v for v in expected if v not in present]
    reasons: list[str] = []
    status = "healthy"
    if missing:
        status = "warning"
        reasons.append(f"missing expected variables: {', '.join(missing)}")
    if "time" in ds.coords and int(ds.sizes.get("time", 0)) == 0:
        status = "broken"
        reasons.append("time dimension exists but has length 0")
    for var in [v for v in ["FIP", "FIA", "FIT", "FIV"] if v in ds.data_vars]:
        try:
            all_nan = bool(ds[var].isnull().all().item())
        except Exception:
            all_nan = False
        if all_nan:
            status = "warning" if status != "broken" else status
            reasons.append(f"{var} is entirely NaN")
    return StoreHealth(status=status,
                       reasons=reasons,
                       variables_present=present,
                       variables_missing=missing)

def _discover_classifications(pth_cfg: ShugaPaths, cls_cfg: ClassificationSpec | None = None, logger=None) -> list[ClassificationStatus]:
    hemi_root = pth_cfg.zarr_root / pth_cfg.hemisphere
    if not hemi_root.exists():
        return []
    statuses: list[ClassificationStatus] = []
    seen: set[tuple[str, str, str]] = set()
    for thresh_dir in sorted(p for p in hemi_root.iterdir() if p.is_dir() and _THRESH_RE.match(p.name)):
        for ice_type_dir in sorted(p for p in thresh_dir.iterdir() if p.is_dir()):
            for grid_dir in sorted(p for p in ice_type_dir.iterdir() if p.is_dir()):
                for method_dir in sorted(p for p in grid_dir.iterdir() if p.is_dir()):
                    class_store = method_dir / "data.zarr"
                    if not class_store.exists():
                        continue
                    method_name: str | None = None
                    if method_dir.name == "raw":
                        method_name = "raw"
                    elif _BIN_RE.match(method_dir.name):
                        method_name = "binary-days"
                    elif _ROLL_RE.match(method_dir.name):
                        method_name = "rolling-mean"
                    if method_name is None:
                        continue
                    key = (grid_dir.name, method_dir.name, str(class_store))
                    if key in seen:
                        continue
                    seen.add(key)
                    meta = _parse_store_metadata_from_path(class_store, fallback_classify=cls_cfg)
                    coverage = _maybe_open_time_coverage(class_store)
                    metrics_store = method_dir / "mets.zarr"
                    health = _metrics_health(metrics_store) if metrics_store.exists() else None
                    statuses.append(
                        ClassificationStatus(
                            method=normalize_method(method_name),
                            grid_type=meta["grid_type"] or grid_dir.name,
                            classification_path=str(class_store),
                            metrics_path=str(metrics_store),
                            ice_type=meta["ice_type"] or ice_type_dir.name,
                            ispd_thresh=meta["ispd_thresh"],
                            aice_thresh=meta["aice_thresh"],
                            bin_window=meta["bin_window"],
                            bin_min_days=meta["bin_min_days"],
                            roll_window=meta["roll_window"],
                            coverage=coverage,
                            metrics_exists=metrics_store.exists(),
                            metrics_health=health,
                        )
                    )
    statuses.sort(key=lambda x: (x.grid_type, x.method, x.ispd_thresh or ""))
    return statuses

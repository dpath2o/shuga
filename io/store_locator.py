from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import xarray as xr

from shuga.core.paths import ShugaPaths
from shuga.core.naming import method_dirname, normalize_method
from shuga.core.store_selection import ResolvedStore, StoreSelection


class CICEStoreLocator:
    """
    Resolve and optionally open classification/metrics stores for a simulation,
    while accounting for mixed classification branches such as Tb and Tc.
    """

    def __init__(self, run, classify, metrics=None, plotting=None, observations=None, logger=None):
        self.run = run
        self.classify = classify
        self.metrics = metrics
        self.plotting = plotting
        self.observations = observations
        self.logger = logger

    def _paths_for_sim(self, sim_name: str, project: str | None = None, user: str | None = None) -> ShugaPaths:
        run_other = replace(
            self.run,
            sim_name=sim_name,
            project=project or self.run.project,
            user=user or self.run.user,
        )
        return ShugaPaths(
            run=run_other,
            classify=self.classify,
            metrics=self.metrics,
            plotting=self.plotting,
            observations=self.observations,
        )

    def _method_dir(self, method: str) -> str:
        norm = normalize_method(method)
        return method_dirname(
            norm,
            bin_window=self.classify.bin_window,
            bin_min_days=self.classify.bin_min_days,
            roll_window=self.classify.roll_window,
        )

    def _classification_parent(self, paths: ShugaPaths) -> Path:
        return paths.classification_root_path.parent

    def _candidate_grid_types(
        self,
        paths: ShugaPaths,
        method: str,
        store_name: str,
        search_order: tuple[str, ...],
    ) -> list[tuple[str, Path]]:
        parent = self._classification_parent(paths)
        method_dir = self._method_dir(method)
        found: list[tuple[str, Path]] = []

        if not parent.exists():
            return found

        requested = list(dict.fromkeys(search_order))
        existing = [p.name for p in parent.iterdir() if p.is_dir()]
        for g in existing:
            if g not in requested:
                requested.append(g)

        for grid_type in requested:
            candidate = parent / grid_type / method_dir / store_name
            if candidate.exists():
                found.append((grid_type, candidate))
        return found

    def resolve_store(
        self,
        sim_name: str,
        method: str,
        *,
        store_kind: str,
        selection: StoreSelection | None = None,
        project: str | None = None,
        user: str | None = None,
    ) -> ResolvedStore:
        if store_kind not in {"metrics", "classification"}:
            raise ValueError(f"Unsupported store_kind={store_kind!r}")

        selection = selection or StoreSelection()
        paths = self._paths_for_sim(sim_name=sim_name, project=project, user=user)
        norm = normalize_method(method)
        store_name = "mets.zarr" if store_kind == "metrics" else "data.zarr"
        method_dir = self._method_dir(norm)

        requested_grid_type = selection.requested_grid_type(sim_name)
        if requested_grid_type is not None:
            candidate = self._classification_parent(paths) / requested_grid_type / method_dir / store_name
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Requested {store_kind} store does not exist for sim={sim_name!r}, "
                    f"method={norm!r}, grid_type={requested_grid_type!r}: {candidate}"
                )
            resolved = ResolvedStore(
                sim_name=sim_name,
                method=norm,
                grid_type=requested_grid_type,
                store_kind=store_kind,
                path=candidate,
            )
            if self.logger is not None:
                self.logger.info(
                    "Resolved %s store for %s [%s/%s]: %s",
                    store_kind,
                    sim_name,
                    requested_grid_type,
                    norm,
                    candidate,
                )
            return resolved

        found = self._candidate_grid_types(
            paths=paths,
            method=norm,
            store_name=store_name,
            search_order=selection.search_order,
        )

        if len(found) == 1:
            grid_type, candidate = found[0]
            resolved = ResolvedStore(
                sim_name=sim_name,
                method=norm,
                grid_type=grid_type,
                store_kind=store_kind,
                path=candidate,
            )
            if self.logger is not None:
                self.logger.info(
                    "Resolved %s store for %s [%s/%s]: %s",
                    store_kind,
                    sim_name,
                    grid_type,
                    norm,
                    candidate,
                )
            return resolved

        if not found:
            parent = self._classification_parent(paths)
            raise FileNotFoundError(
                f"Could not find any {store_kind} store for sim={sim_name!r}, method={norm!r}. "
                f"Looked under: {parent}/*/{method_dir}/{store_name}"
            )

        if selection.require_unique:
            matches = "\n".join(f"- {grid_type}: {path}" for grid_type, path in found)
            raise ValueError(
                f"Multiple candidate {store_kind} stores found for sim={sim_name!r}, method={norm!r}. "
                f"Specify StoreSelection(grid_type=...) or grid_type_map={{...}}.\n{matches}"
            )

        grid_type, candidate = found[0]
        resolved = ResolvedStore(
            sim_name=sim_name,
            method=norm,
            grid_type=grid_type,
            store_kind=store_kind,
            path=candidate,
        )
        if self.logger is not None:
            self.logger.info(
                "Resolved %s store for %s [%s/%s] by ordered fallback: %s",
                store_kind,
                sim_name,
                grid_type,
                norm,
                candidate,
            )
        return resolved

    def resolve_metrics_store(self, sim_name: str, method: str, **kwargs) -> ResolvedStore:
        return self.resolve_store(sim_name=sim_name, method=method, store_kind="metrics", **kwargs)

    def resolve_classification_store(self, sim_name: str, method: str, **kwargs) -> ResolvedStore:
        return self.resolve_store(sim_name=sim_name, method=method, store_kind="classification", **kwargs)

    def open_metrics_dataset(self, sim_name: str, method: str, **kwargs):
        resolved = self.resolve_metrics_store(sim_name=sim_name, method=method, **kwargs)
        return xr.open_zarr(resolved.path, consolidated=False), resolved

    def open_classification_dataset(self, sim_name: str, method: str, **kwargs):
        resolved = self.resolve_classification_store(sim_name=sim_name, method=method, **kwargs)
        return xr.open_zarr(resolved.path, consolidated=False), resolved

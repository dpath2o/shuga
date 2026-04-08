from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(slots=True, frozen=True)
class StoreSelection:
    """
    Selection policy for locating classification/metrics stores across mixed
    classification branches (for example Tb vs Tc).

    Parameters
    ----------
    grid_type
        Global explicit classification branch to use for all simulations.
    grid_type_map
        Per-simulation override, e.g. {"ndte-max": "Tb", "LD-waves-exp01": "Tc"}.
    search_order
        Ordered fallback search when the branch is not explicitly specified.
    require_unique
        If True, raise when multiple valid branches are found and no explicit
        branch was supplied.
    """

    grid_type: str | None = None
    grid_type_map: Mapping[str, str] | None = None
    search_order: tuple[str, ...] = ("Tb", "Tc")
    require_unique: bool = True

    def requested_grid_type(self, sim_name: str) -> str | None:
        if self.grid_type_map is not None and sim_name in self.grid_type_map:
            return self.grid_type_map[sim_name]
        return self.grid_type


@dataclass(slots=True, frozen=True)
class ResolvedStore:
    """Concrete result returned by store resolution."""

    sim_name: str
    method: str
    grid_type: str
    store_kind: str
    path: Path

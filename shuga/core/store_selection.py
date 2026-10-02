from __future__  import annotations
from dataclasses import dataclass
from pathlib     import Path
from typing      import Mapping

@dataclass(slots=True, frozen=True)
class StoreSelection:
    """
    Policy for resolving classification or metrics stores when a simulation may
    have outputs under more than one classification branch, such as ``Tb`` and
    ``Tc``.

    This object tells the store locator how to choose a grid-type branch:

    1. use a per-simulation override from ``grid_type_map`` if present,
    2. otherwise use the global ``grid_type`` if supplied,
    3. otherwise search branches in ``search_order``.

    If more than one valid branch exists and no explicit branch has been
    selected, ``require_unique`` controls whether resolution should fail or
    fall back to the first match in the configured search order.

    Parameters
    ----------
    grid_type : str | None, optional
        Global grid-type branch to use for all simulations unless overridden by
        ``grid_type_map``. Typical examples are ``"Tb"`` and ``"Tc"``.
    grid_type_map : Mapping[str, str] | None, optional
        Per-simulation grid-type override. This takes precedence over
        ``grid_type`` when the requested simulation name is present in the
        mapping. For example:
        ``{"ndte-max": "Tb", "LD-waves-exp01": "Tc"}``.
    search_order : tuple[str, ...], optional
        Ordered list of grid-type branches to try when no explicit branch is
        provided. The default is ``("Tb", "Tc")``.
    require_unique : bool, optional
        If ``True``, raise an error when multiple valid branches are found and
        no explicit branch has been selected. If ``False``, the first matching
        branch in ``search_order`` may be used as a fallback.

    Notes
    -----
    - ``grid_type_map`` has highest priority.
    - ``grid_type`` acts as a global default.
    - ``search_order`` is only used when neither of the above chooses a branch.
    """

    grid_type     : str | None               = None
    grid_type_map : Mapping[str, str] | None = None
    search_order  : tuple[str, ...]          = ("Tb", "Tc")
    require_unique: bool                     = True

    def requested_grid_type(self, sim_name: str) -> str | None:
        """
        Return the explicitly requested grid type for a simulation, if any.

        Resolution priority is:

        1. ``grid_type_map[sim_name]`` when available,
        2. otherwise the global ``grid_type``,
        3. otherwise ``None``.

        Parameters
        ----------
        sim_name : str
            Simulation name for which an explicit grid type is being queried.

        Returns
        -------
        str | None
            The selected grid type for this simulation, or ``None`` if no
            explicit selection has been configured.
        """
        if self.grid_type_map is not None and sim_name in self.grid_type_map:
            return self.grid_type_map[sim_name]
        return self.grid_type

@dataclass(slots=True, frozen=True)
class ResolvedStore:
    """
    Fully resolved store descriptor returned by store-location logic.

    This dataclass records the exact store selected for a given simulation,
    classification method, and store kind. It is typically returned by a store
    locator after ambiguity has been resolved and a concrete Zarr path has been
    identified.

    Attributes
    ----------
    sim_name : str
        Simulation name associated with the resolved store.
    method : str
        Normalised classification method name used during resolution.
    grid_type : str
        Resolved classification branch, such as ``"Tb"`` or ``"Tc"``.
    store_kind : str
        Store category, typically ``"classification"`` or ``"metrics"``.
    path : Path
        Absolute or fully resolved filesystem path to the selected Zarr store.
    """
    sim_name   : str
    method     : str
    grid_type  : str
    store_kind : str
    path       : Path

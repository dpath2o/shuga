from .AF2020 import (
    AF2020Spec,
    AF2020Observations,
    SeaIceAF2020,
)

from .NSIDC import (
    NSIDCObservations,
    SeaIceNSIDC,
)

from .sea_ice_thickness import (
    SITSourceSpec,
    build_source_store,
    discover_awi_l3cp,
    discover_esa_l3c,
)

from .legacy import SeaIceObservations


# ---------------------------------------------------------------------
# Short class-name aliases
# ---------------------------------------------------------------------
AF2020Obs = AF2020Observations
NSIDCObs = NSIDCObservations

# Optional shorter legacy alias, if useful
SeaIceObs = SeaIceObservations


__all__ = [
    "AF2020Spec",
    "AF2020Observations",
    "AF2020Obs",
    "SeaIceAF2020",

    "NSIDCObservations",
    "NSIDCObs",
    "SeaIceNSIDC",

    "SITSourceSpec",
    "build_source_store",
    "discover_awi_l3cp",
    "discover_esa_l3c",

    "SeaIceObservations",
    "SeaIceObs",
]

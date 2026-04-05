ANTARCTIC_8_REGIONS: dict[str, dict[str, list[float] | str]] = {
    "DML": {"geo_region": [-19, 18, -80, -60], "plot_region": [-30, 10, -75, -68], "projection": "S{MC}/-90/{fig_size}c"},
    "WIO": {"geo_region": [27, 71, -80, -60], "plot_region": [10, 52.5, -72, -64], "projection": "S{MC}/-90/{fig_size}c"},
    "EIO": {"geo_region": [74, 103, -80, -60], "plot_region": [52.5, 97.5, -70.5, -64], "projection": "S{MC}/-90/{fig_size}c"},
    "Aus": {"geo_region": [103, 146, -80, -60], "plot_region": [97.5, 142.5, -67.5, -63], "projection": "S{MC}/-90/{fig_size}c"},
    "VOL": {"geo_region": [146, 172, -80, -60], "plot_region": [140, 180, -78, -65], "projection": "S{MC}/-90/{fig_size}c"},
    "AS": {"geo_region": [-158, -102, -80, -60], "plot_region": [-180, -115, -79, -72], "projection": "S{MC}/-90/{fig_size}c"},
    "BS": {"geo_region": [-102, -60, -80, -60], "plot_region": [-115, -70, -76, -67], "projection": "S{MC}/-90/{fig_size}c"},
    "WS": {"geo_region": [-60, -27, -80, -60], "plot_region": [-75, -30, -78, -62], "projection": "S{MC}/-90/{fig_size}c"},
}

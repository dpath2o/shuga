from __future__ import annotations
import numpy as np

def skill_stats(mod_data: np.ndarray, obs_data: np.ndarray) -> dict[str, float]:
    """
    Return Bias, RMSE, MAE, and Pearson correlation for paired model/obs data.
    """
    good = np.isfinite(mod_data) & np.isfinite(obs_data)
    if good.sum() == 0:
        return {"Bias": np.nan, "RMSE": np.nan, "MAE": np.nan, "Corr": np.nan}
    m    = mod_data[good]
    o    = obs_data[good]
    corr = np.corrcoef(m, o)[0, 1] if good.sum() > 1 else np.nan
    return {"Bias": float(np.mean(m - o)),
            "RMSE": float(np.sqrt(np.mean((m - o) ** 2))),
            "MAE" : float(np.mean(np.abs(m - o))),
            "Corr": float(corr)}

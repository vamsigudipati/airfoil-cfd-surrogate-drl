"""Metrics shared by the gust-control notebooks (Stages 2 and 4) and Stage 3."""

import numpy as np


def episode_metrics(data):
    """Summary of one flown episode (dict of arrays with keys cl, cl_target, flap_deg, cd, reward)."""
    error = data["cl"] - data["cl_target"]

    return {
        "CL_RMSE": np.sqrt(np.mean(error**2)),
        "mean_abs_flap_deg": np.mean(np.abs(data["flap_deg"])),
        "max_abs_flap_deg": np.max(np.abs(data["flap_deg"])),
        "mean_CD": np.mean(data["cd"]),
        "episode_return": np.sum(data["reward"]),
    }


def regression_metrics(y_true, y_hat):
    """MAE, RMSE and R2 of one output, as printed by Stage 3 (``print_metrics``)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MAE": float(mean_absolute_error(y_true, y_hat)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_hat))),
        "R2": float(r2_score(y_true, y_hat)),
    }

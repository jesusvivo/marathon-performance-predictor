"""Personal velocity-duration race-time curves (Riegel)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_riegel(distance_km: float, a: float, b: float) -> float:
    """Riegel model: estimated time in seconds for a distance, T = a * D^b."""
    return float(a * (distance_km**b))


def fit_riegel(points: pd.DataFrame) -> tuple[float, float]:
    """Least-squares fit of T = a * D^b in log-log space; returns (a, b)."""
    log_d = np.log(points["distance_km"])
    log_t = np.log(points["duration_s"])

    slope, intercept = np.polyfit(log_d, log_t, 1)

    return float(np.exp(intercept)), float(slope)

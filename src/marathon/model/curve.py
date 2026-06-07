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

def fit_cs(points: pd.DataFrame) -> tuple[float, float]:
    """Critical-speed fit distance = D' + CS*time; returns (cs, d_prime)."""
    d = points["distance_km"] * 1000
    t = points["duration_s"]

    slope, intercept = np.polyfit(d, t, 1)

    cs = 1 / slope
    d_prime = intercept * -cs

    return float(cs), float(d_prime)

def predict_cs(distance_km: float, cs: float, d_prime: float) -> float:
    """Critical-speed model: time in seconds for a distance."""

    return float(((distance_km * 1000) - d_prime) / cs)
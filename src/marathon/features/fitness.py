"""Fitness state (CTL / ATL / TSB) from a daily training-load series."""

from __future__ import annotations

import numpy as np
import pandas as pd

CTL_DAYS = 28
ATL_DAYS = 7


def _ewma(load: pd.Series, days: int) -> pd.Series:
    """Impulse-response EWMA with time constant `days` (TrainingPeaks form)."""
    alpha = 1.0 - np.exp(-1.0 / days)
    return load.ewm(alpha=alpha, adjust=False).mean()


def fitness_state(
    load: pd.Series, ctl_days: int = CTL_DAYS, atl_days: int = ATL_DAYS
) -> pd.DataFrame:
    """CTL, ATL, and TSB (prior-day CTL - ATL) from a daily load series."""
    ctl = _ewma(load, ctl_days)
    atl = _ewma(load, atl_days)
    tsb = (ctl - atl).shift(1)
    return pd.DataFrame({"load": load, "ctl": ctl, "atl": atl, "tsb": tsb})

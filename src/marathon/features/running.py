"""Running-specific daily features from parsed activities."""

from __future__ import annotations

import numpy as np
import pandas as pd

LONG_RUN_KM = 15.0
WINDOW_DAYS = 28


def running_features(activities: pd.DataFrame) -> pd.DataFrame:
    """Daily running volume, rolling load, long-run recency, pace, efficiency, VO2max."""
    # calendar spans all non-parent activities (matches the load spine), runs are aggregated onto it
    kept = activities.loc[~activities["is_parent"]]
    span = kept["start_time_local"].dt.normalize()
    full = pd.date_range(span.min(), span.max(), freq="D")

    runs = kept.loc[kept["discipline"] == "run"]
    day = runs["start_time_local"].dt.normalize()

    km = runs.groupby(day)["distance_km"].sum().reindex(full, fill_value=0.0)
    dur = runs.groupby(day)["duration_s"].sum().reindex(full, fill_value=0.0)

    # distance-weighted average pace over the trailing window: total time / total distance
    dur_window = dur.rolling(WINDOW_DAYS, min_periods=1).sum()
    km_window = km.rolling(WINDOW_DAYS, min_periods=1).sum()

    # efficiency factor (speed per heartbeat), averaged over runs in the trailing window
    ef = (runs["avg_speed_ms"] / runs["avg_hr"]).groupby(day).mean().reindex(full)

    # days since the last long run (NaN before the first one)
    is_long = (runs["distance_km"] >= LONG_RUN_KM).groupby(day).max()
    long_day = is_long.reindex(full, fill_value=False).astype(bool)
    pos = pd.Series(np.arange(len(full)), index=full)
    days_since_long_run = pos - pos.where(long_day).ffill()

    # most recent running VO2max, carried forward
    vo2 = runs.dropna(subset=["vo2max"]).groupby(day)["vo2max"].last().reindex(full).ffill()

    return pd.DataFrame(
        {
            "run_km": km,
            "volume_7d_km": km.rolling(7, min_periods=1).sum(),
            "volume_28d_km": km_window,
            "avg_pace_s_per_km_28d": (dur_window / km_window).where(km_window > 0),
            "efficiency_28d": ef.rolling(WINDOW_DAYS, min_periods=1).mean(),
            "days_since_long_run": days_since_long_run,
            "vo2max": vo2,
        }
    ).rename_axis("date")

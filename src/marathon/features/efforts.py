"""Velocity-duration effort points and the best-effort frontier (running)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def effort_points(activities: pd.DataFrame) -> pd.DataFrame:
    """Per-run velocity-duration points: duration, distance, speed, race flag, date."""
    runs = activities.loc[
        (activities["discipline"] == "run")
        & ~activities["is_parent"]
        & (activities["distance_km"] > 0)
        & (activities["duration_s"] > 0)
    ]
    return (
        pd.DataFrame(
            {
                "date": runs["start_time_local"].dt.normalize().to_numpy(),
                "duration_s": runs["duration_s"].to_numpy(),
                "distance_km": runs["distance_km"].to_numpy(),
                "speed_ms": runs["avg_speed_ms"].to_numpy(),
                "is_race": runs["is_race"].to_numpy(),
            }
        )
        .sort_values("duration_s")
        .reset_index(drop=True)
    )


def velocity_duration_frontier(points: pd.DataFrame) -> pd.DataFrame:
    """Best-effort upper envelope: an effort is kept only if it is faster than every
    longer-duration effort, giving the monotonic speed-vs-duration curve to fit."""
    ordered = points.sort_values("duration_s", ascending=False)
    keep: list[int] = []
    best_speed = -np.inf
    for idx, speed in zip(ordered.index, ordered["speed_ms"], strict=True):
        if speed > best_speed:
            keep.append(int(idx))
            best_speed = speed
    return points.loc[keep].sort_values("duration_s").reset_index(drop=True)

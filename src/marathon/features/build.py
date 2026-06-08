"""Assemble the daily feature matrix that feeds the model and the feature store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from marathon.features.fitness import fitness_state
from marathon.features.load import daily_load
from marathon.features.running import running_features
from marathon.features.wellness import daily_wellness


def daily_features(
    activities: pd.DataFrame, wellness_dir: str | Path, metrics_dir: str | Path
) -> pd.DataFrame:
    """One row per day: fitness state, running load/volume, and wellness, aligned on date.

    The training-load calendar is the spine (every day from the first to the last activity);
    running and wellness signals are left-joined onto it.
    """
    fitness = fitness_state(daily_load(activities))
    running = running_features(activities)
    wellness = daily_wellness(wellness_dir, metrics_dir)
    final = fitness.join(running, how="left").join(wellness, how="left")
    final["athlete_id"] = 1
    return final

"""Daily multisport training load from parsed activities."""

from __future__ import annotations

import pandas as pd


def daily_load(activities: pd.DataFrame) -> pd.Series:
    """Total training load per local calendar day across all disciplines,
    excluding multisport parents, on a gap-free daily index (rest days = 0)."""
    kept = activities.loc[~activities["is_parent"]]
    day = kept["start_time_local"].dt.normalize()
    per_day = kept.groupby(day)["training_load"].sum()
    full = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
    return per_day.reindex(full, fill_value=0.0).rename_axis("date").rename("load")

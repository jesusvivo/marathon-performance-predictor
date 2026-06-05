"""Unit tests for the daily training-load feature."""

from __future__ import annotations

import pandas as pd

from marathon.features.load import daily_load


def test_sums_within_day_and_fills_rest_days() -> None:
    df = pd.DataFrame(
        {
            "start_time_local": pd.to_datetime(
                ["2024-05-01 07:00", "2024-05-01 18:00", "2024-05-03 09:00"]
            ),
            "training_load": [50.0, 30.0, 40.0],
            "is_parent": [False, False, False],
        }
    )
    s = daily_load(df)
    assert s[pd.Timestamp("2024-05-01")] == 80.0  # two activities summed
    assert s[pd.Timestamp("2024-05-02")] == 0.0  # rest day filled
    assert s[pd.Timestamp("2024-05-03")] == 40.0
    assert s.name == "load"
    assert s.index.name == "date"


def test_excludes_parents() -> None:
    df = pd.DataFrame(
        {
            "start_time_local": pd.to_datetime(["2024-05-01 07:00", "2024-05-01 08:00"]),
            "training_load": [100.0, 60.0],
            "is_parent": [True, False],
        }
    )
    s = daily_load(df)
    assert s[pd.Timestamp("2024-05-01")] == 60.0  # parent's 100 excluded


def test_index_is_contiguous_daily() -> None:
    df = pd.DataFrame(
        {
            "start_time_local": pd.to_datetime(["2024-05-01 07:00", "2024-05-05 09:00"]),
            "training_load": [10.0, 20.0],
            "is_parent": [False, False],
        }
    )
    s = daily_load(df)
    assert list(s.index) == list(pd.date_range("2024-05-01", "2024-05-05", freq="D"))
    assert len(s) == 5

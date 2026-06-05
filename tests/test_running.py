"""Unit tests for running-specific daily features."""

from __future__ import annotations

import pandas as pd

from marathon.features.running import running_features


def _fixture() -> pd.DataFrame:
    # runs on Jan 1/3/5, a strength session on Jan 8 (extends the calendar, not a run),
    # and a multisport parent on Jan 3 that must be excluded.
    return pd.DataFrame(
        {
            "discipline": ["run", "run", "run", "strength", "run"],
            "is_parent": [False, False, False, False, True],
            "start_time_local": pd.to_datetime(
                [
                    "2024-01-01 07:00",
                    "2024-01-03 07:00",
                    "2024-01-05 07:00",
                    "2024-01-08 18:00",
                    "2024-01-03 09:00",
                ]
            ),
            "distance_km": [10.0, 16.0, 5.0, 0.0, 99.0],
            "duration_s": [3000.0, 5760.0, 1500.0, 1800.0, 9999.0],
            "avg_speed_ms": [3.33, 2.78, 3.33, 0.0, 5.0],
            "avg_hr": [150.0, 150.0, 150.0, 100.0, 140.0],
            "vo2max": [50.0, 51.0, None, None, None],
            "is_race": [False, False, False, False, False],
        }
    )


def test_run_km_and_calendar_span() -> None:
    fs = running_features(_fixture())
    assert fs.loc["2024-01-01", "run_km"] == 10.0
    assert fs.loc["2024-01-02", "run_km"] == 0.0  # rest day
    assert fs.index[-1] == pd.Timestamp("2024-01-08")  # spans to last activity, not last run


def test_rolling_volume() -> None:
    fs = running_features(_fixture())
    assert fs.loc["2024-01-05", "volume_7d_km"] == 31.0  # 10 + 16 + 5


def test_days_since_long_run() -> None:
    fs = running_features(_fixture())
    assert pd.isna(fs.loc["2024-01-02", "days_since_long_run"])  # before first long run
    assert fs.loc["2024-01-03", "days_since_long_run"] == 0.0  # the 16 km long run
    assert fs.loc["2024-01-08", "days_since_long_run"] == 5.0


def test_vo2max_forward_filled() -> None:
    fs = running_features(_fixture())
    assert fs.loc["2024-01-02", "vo2max"] == 50.0  # carried from Jan 1
    assert fs.loc["2024-01-08", "vo2max"] == 51.0  # carried from Jan 3


def test_parent_excluded_from_distance() -> None:
    fs = running_features(_fixture())
    assert fs.loc["2024-01-03", "run_km"] == 16.0  # parent's 99 km not added

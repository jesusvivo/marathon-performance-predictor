"""Unit tests for the assembled daily feature matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from marathon.features.build import daily_features


def _activities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "discipline": ["run", "run"],
            "is_parent": [False, False],
            "start_time_local": pd.to_datetime(["2024-01-01 07:00", "2024-01-03 07:00"]),
            "training_load": [50.0, 60.0],
            "distance_km": [10.0, 12.0],
            "duration_s": [3000.0, 3600.0],
            "avg_speed_ms": [3.33, 3.33],
            "avg_hr": [150.0, 150.0],
            "vo2max": [50.0, 51.0],
            "is_race": [False, False],
        }
    )


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    well = tmp_path / "wellness"
    met = tmp_path / "metrics"
    well.mkdir()
    met.mkdir()
    sleep = [
        {
            "calendarDate": "2024-01-01",
            "deepSleepSeconds": 3600,
            "lightSleepSeconds": 3600,
            "remSleepSeconds": 1800,
            "sleepScores": {"overallScore": 75},
        }
    ]
    readiness = [
        {"calendarDate": "2024-01-01", "score": 70, "recoveryTime": 12, "hrvWeeklyAverage": 68}
    ]
    (well / "x_sleepData.json").write_text(json.dumps(sleep))
    (met / "TrainingReadinessDTO_x.json").write_text(json.dumps(readiness))
    return well, met


def test_daily_features_columns_and_alignment(tmp_path: Path) -> None:
    well, met = _dirs(tmp_path)
    feat = daily_features(_activities(), well, met)
    expected = {"ctl", "atl", "tsb", "run_km", "vo2max", "sleep_hours", "readiness_score"}
    assert expected <= set(feat.columns)
    assert feat.index.is_monotonic_increasing and not feat.index.has_duplicates
    assert list(feat.index) == list(pd.date_range("2024-01-01", "2024-01-03", freq="D"))


def test_wellness_joined_on_date(tmp_path: Path) -> None:
    well, met = _dirs(tmp_path)
    feat = daily_features(_activities(), well, met)
    assert feat.loc["2024-01-01", "sleep_hours"] == pytest.approx((3600 + 3600 + 1800) / 3600)
    assert feat.loc["2024-01-01", "readiness_score"] == 70
    assert pd.isna(feat.loc["2024-01-02", "sleep_hours"])  # no wellness that day

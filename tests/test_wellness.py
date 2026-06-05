"""Unit tests for sleep and readiness wellness features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marathon.features.wellness import load_readiness, load_sleep


def test_load_sleep_hours_score_and_skips_unmeasured(tmp_path: Path) -> None:
    recs = [
        {
            "calendarDate": "2024-01-01",
            "deepSleepSeconds": 3600,
            "lightSleepSeconds": 7200,
            "remSleepSeconds": 1800,
            "sleepScores": {"overallScore": 80},
        },
        {"calendarDate": "2024-01-02", "deepSleepSeconds": None, "sleepScores": {}},
    ]
    (tmp_path / "x_sleepData.json").write_text(json.dumps(recs))
    df = load_sleep(tmp_path)
    assert list(df.columns) == ["sleep_hours", "sleep_score"]
    assert df.loc["2024-01-01", "sleep_hours"] == (3600 + 7200 + 1800) / 3600
    assert df.loc["2024-01-01", "sleep_score"] == 80
    assert pd.Timestamp("2024-01-02") not in df.index  # unmeasured night dropped


def test_load_readiness_filters_zero_and_dedupes(tmp_path: Path) -> None:
    recs = [
        {"calendarDate": "2024-01-01", "score": 0},
        {"calendarDate": "2024-01-02", "score": 60, "recoveryTime": 12, "hrvWeeklyAverage": 70},
        {"calendarDate": "2024-01-02", "score": 65, "recoveryTime": 10, "hrvWeeklyAverage": 72},
    ]
    (tmp_path / "TrainingReadinessDTO_x.json").write_text(json.dumps(recs))
    df = load_readiness(tmp_path)
    assert pd.Timestamp("2024-01-01") not in df.index  # zero score dropped
    assert df.loc["2024-01-02", "readiness_score"] == 65  # later record wins
    assert df.loc["2024-01-02", "recovery_time_h"] == 10

"""Unit tests for the Garmin reference loader and correlation report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from marathon.features.garmin import (
    correlation_report,
    load_race_predictions,
    load_reference,
)


def _write_metrics(tmp_path: Path) -> Path:
    recs = [
        {"calendarDate": d, "dailyTrainingLoadAcute": a, "dailyTrainingLoadChronic": c}
        for d, a, c in [(1_714_348_800_000, 30, 200), (1_714_435_200_000, 40, 210)]
    ]
    p = tmp_path / "MetricsAcuteTrainingLoad_x.json"
    p.write_text(json.dumps(recs))
    return tmp_path


def test_load_reference_parses_and_renames(tmp_path: Path) -> None:
    ref = load_reference(_write_metrics(tmp_path))
    assert list(ref.columns) == ["garmin_acute", "garmin_chronic"]
    assert isinstance(ref.index, pd.DatetimeIndex)
    assert ref["garmin_acute"].tolist() == [30, 40]


def test_load_race_predictions_renames_and_dedupes(tmp_path: Path) -> None:
    recs = [
        {"calendarDate": "2024-01-01", "raceTime5K": 1300, "raceTime10K": 2700,
         "raceTimeHalf": 6000, "raceTimeMarathon": 12600},
        {"calendarDate": "2024-01-01", "raceTime5K": 1290, "raceTime10K": 2690,
         "raceTimeHalf": 5990, "raceTimeMarathon": 12500},
    ]
    (tmp_path / "RunRacePredictions_x.json").write_text(json.dumps(recs))
    df = load_race_predictions(tmp_path)
    assert list(df.columns) == ["pred_5k_s", "pred_10k_s", "pred_half_s", "pred_marathon_s"]
    assert len(df) == 1  # one prediction per day
    assert df.loc["2024-01-01", "pred_5k_s"] == 1290  # later record wins


def test_correlation_report_perfect_match() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    fitness = pd.DataFrame({"ctl": [1.0, 2, 3, 4, 5], "atl": [5.0, 4, 3, 2, 1]}, index=idx)
    # reference is a linear rescale of our series; correlation should be 1.0
    reference = pd.DataFrame(
        {"garmin_chronic": fitness["ctl"] * 7 + 10, "garmin_acute": fitness["atl"] * 5}, index=idx
    )
    rep = correlation_report(fitness, reference)
    assert np.isclose(rep["ctl_vs_chronic"], 1.0)
    assert np.isclose(rep["atl_vs_acute"], 1.0)
    assert rep["n_days"] == 5.0

"""Unit tests for the Garmin activity parser, against hand-checked fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from marathon.parse import load_export, normalize_activity, parse_activities

RUN_RAW: dict[str, Any] = {
    "activityId": 111,
    "name": "Morning Run",
    "activityType": "running",
    "eventTypeId": 9,
    "startTimeGmt": 1_780_317_224_000,
    "startTimeLocal": 1_780_324_424_000,
    "duration": 2_490_000.0,  # ms -> 2490 s
    "movingDuration": 2_400_000.0,
    "distance": 708_000.0,  # cm -> 7.08 km
    "avgSpeed": 0.284,  # *10 -> 2.84 m/s
    "avgHr": 150,
    "maxHr": 170,
    "minHr": 95,
    "vO2MaxValue": 52.0,
    "activityTrainingLoad": 120.5,
    "elevationGain": 4448.0,  # cm -> 44.48 m
    "avgRunCadence": 84.0,  # per-leg rev/min *2 -> 168 spm
    "avgStrideLength": 107.35,  # cm -> 1.0735 m
    "hrTimeInZone_0": 90_000.0,  # ms -> 90 s
    "parent": False,
}

RACE_RAW: dict[str, Any] = {
    "activityId": 222,
    "name": "Madrid Half",
    "activityType": "running",
    "eventTypeId": 1,
    "startTimeGmt": 1_780_400_000_000,
    "distance": 2_131_000.0,
    "duration": 5_400_000.0,
    "avgSpeed": 0.394,
}

BIKE_RAW: dict[str, Any] = {
    "activityId": 333,
    "name": "Road Ride",
    "activityType": "road_biking",
    "eventTypeId": 9,
    "startTimeGmt": 1_780_500_000_000,
    "distance": 4_000_000.0,
    "duration": 5_000_000.0,
    "trainingStressScore": 53.6,
    "intensityFactor": 0.732,
    "avgPower": 154.0,
}


def test_normalize_units() -> None:
    row = normalize_activity(RUN_RAW)
    assert row["distance_km"] == pytest.approx(7.08)
    assert row["duration_s"] == pytest.approx(2490.0)
    assert row["avg_speed_ms"] == pytest.approx(2.84)
    assert row["elevation_gain_m"] == pytest.approx(44.48)
    assert row["avg_stride_length_m"] == pytest.approx(1.0735)
    assert row["avg_run_cadence_spm"] == pytest.approx(168.0)
    assert row["hr_zone_0_s"] == pytest.approx(90.0)


def test_discipline_and_race_flags() -> None:
    assert normalize_activity(RUN_RAW)["discipline"] == "run"
    assert normalize_activity(BIKE_RAW)["discipline"] == "bike"
    assert normalize_activity(RUN_RAW)["is_race"] is False
    assert normalize_activity(RACE_RAW)["is_race"] is True


def test_unknown_type_maps_to_other() -> None:
    raw = {**RUN_RAW, "activityType": "kitesurfing"}
    assert normalize_activity(raw)["discipline"] == "other"


def test_activity_type_dict_shape() -> None:
    raw = {**RUN_RAW, "activityType": {"typeKey": "running"}}
    assert normalize_activity(raw)["activity_type"] == "running"
    assert normalize_activity(raw)["discipline"] == "run"


def test_missing_optional_fields_are_none() -> None:
    row = normalize_activity(RACE_RAW)
    assert row["avg_hr"] is None
    assert row["vo2max"] is None
    assert row["elevation_gain_m"] is None


def test_timestamps_parsed() -> None:
    row = normalize_activity(RUN_RAW)
    assert isinstance(row["start_time_utc"], pd.Timestamp)
    assert row["start_time_utc"].tz is not None
    assert row["start_time_local"].tz is None


def test_parse_activities_dedupes_and_sorts() -> None:
    early = {**RUN_RAW, "activityId": 111, "startTimeGmt": 1_000}
    late = {**RUN_RAW, "activityId": 111, "startTimeGmt": 2_000, "distance": 999_000.0}
    df = parse_activities([RACE_RAW, early, late])
    assert len(df) == 2  # 111 deduped, 222 kept
    dup = df[df["activity_id"] == 111].iloc[0]
    assert dup["distance_km"] == pytest.approx(9.99)  # latest start wins
    assert df["start_time_utc"].is_monotonic_increasing


def test_parse_empty_returns_empty_frame() -> None:
    assert parse_activities([]).empty


def test_load_export_unwraps_payload(tmp_path: Path) -> None:
    payload = [{"summarizedActivitiesExport": [RUN_RAW, BIKE_RAW]}]
    p = tmp_path / "export.json"
    p.write_text(json.dumps(payload))
    df = load_export(p)
    assert len(df) == 2
    assert set(df["discipline"]) == {"run", "bike"}


def test_load_export_rejects_bad_shape(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"nope": 1}))
    with pytest.raises(ValueError, match="Unrecognized"):
        load_export(p)

"""Parse a Garmin GDPR export's summarizedActivities into a tidy multisport frame.

Units in the raw export: distance in cm, duration in ms, avgSpeed in (m/s)/10,
elevation and stride length in cm, HR time-in-zone in ms, timestamps in ms epoch,
avgRunCadence in per-leg rev/min (doubled here to full strides/min).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

CM_PER_KM = 100_000.0
CM_PER_M = 100.0
MS_PER_S = 1000.0
SPEED_RAW_TO_MS = 10.0
CADENCE_TO_SPM = 2.0
RACE_EVENT_TYPE_ID = 1
HR_ZONE_COUNT = 7

DISCIPLINE_BY_TYPE: dict[str, str] = {
    "running": "run",
    "treadmill_running": "run",
    "road_biking": "bike",
    "virtual_ride": "bike",
    "indoor_cycling": "bike",
    "lap_swimming": "swim",
    "open_water_swimming": "swim",
    "indoor_cardio": "cardio",
    "indoor_rowing": "cardio",
    "strength_training": "strength",
    "pilates": "strength",
    "walking": "walk",
    "hiking": "walk",
    "multi_sport": "multisport",
    "transition_v2": "multisport",
}


def _activity_type_key(raw: dict[str, Any]) -> str:
    """activityType is a typeKey string in this export, a nested dict in others."""
    value = raw.get("activityType")
    if isinstance(value, dict):
        return str(value.get("typeKey", "unknown"))
    return str(value) if value is not None else "unknown"


def _div(value: Any, factor: float) -> float | None:
    return float(value) / factor if value is not None else None


def _mul(value: Any, factor: float) -> float | None:
    return float(value) * factor if value is not None else None


def _epoch_ms_to_utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True)


def _epoch_ms_to_local_naive(value: Any) -> pd.Timestamp | None:
    """startTimeLocal is ms epoch shifted to local wall time; drop tz to keep it."""
    if value is None:
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True).tz_localize(None)


def normalize_activity(raw: dict[str, Any]) -> dict[str, Any]:
    """One raw activity record to a flat, unit-normalized row."""
    type_key = _activity_type_key(raw)
    row: dict[str, Any] = {
        "activity_id": int(raw["activityId"]),
        "name": raw.get("name"),
        "discipline": DISCIPLINE_BY_TYPE.get(type_key, "other"),
        "activity_type": type_key,
        "start_time_utc": _epoch_ms_to_utc(raw.get("startTimeGmt") or raw.get("beginTimestamp")),
        "start_time_local": _epoch_ms_to_local_naive(raw.get("startTimeLocal")),
        "duration_s": _div(raw.get("duration"), MS_PER_S),
        "moving_duration_s": _div(raw.get("movingDuration"), MS_PER_S),
        "distance_km": _div(raw.get("distance"), CM_PER_KM),
        "avg_speed_ms": _mul(raw.get("avgSpeed"), SPEED_RAW_TO_MS),
        "avg_hr": raw.get("avgHr"),
        "max_hr": raw.get("maxHr"),
        "min_hr": raw.get("minHr"),
        "vo2max": raw.get("vO2MaxValue"),
        "training_load": raw.get("activityTrainingLoad"),
        "aerobic_te": raw.get("aerobicTrainingEffect"),
        "anaerobic_te": raw.get("anaerobicTrainingEffect"),
        "training_stress_score": raw.get("trainingStressScore"),
        "intensity_factor": raw.get("intensityFactor"),
        "avg_power": raw.get("avgPower"),
        "calories": raw.get("calories"),
        "elevation_gain_m": _div(raw.get("elevationGain"), CM_PER_M),
        "elevation_loss_m": _div(raw.get("elevationLoss"), CM_PER_M),
        "avg_run_cadence_spm": _mul(raw.get("avgRunCadence"), CADENCE_TO_SPM),
        "avg_stride_length_m": _div(raw.get("avgStrideLength"), CM_PER_M),
        "is_race": raw.get("eventTypeId") == RACE_EVENT_TYPE_ID,
        "is_parent": bool(raw.get("parent", False)),
        "parent_id": raw.get("parentId"),
    }
    for i in range(HR_ZONE_COUNT):
        row[f"hr_zone_{i}_s"] = _div(raw.get(f"hrTimeInZone_{i}"), MS_PER_S)
    return row


def parse_activities(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Raw activity records to a tidy frame, deduped by activity_id (latest start wins)."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(normalize_activity(r) for r in records)
    df = df.sort_values("start_time_utc").drop_duplicates("activity_id", keep="last")
    return df.sort_values("start_time_utc").reset_index(drop=True)


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    """The export wraps records as [{"summarizedActivitiesExport": [...]}]."""
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if "summarizedActivitiesExport" in payload[0]:
            return list(payload[0]["summarizedActivitiesExport"])
        return list(payload)
    raise ValueError("Unrecognized summarizedActivities payload shape")


def load_export(path: str | Path) -> pd.DataFrame:
    """Read a summarizedActivities.json file into a tidy multisport frame."""
    payload = json.loads(Path(path).read_text())
    return parse_activities(_extract_records(payload))

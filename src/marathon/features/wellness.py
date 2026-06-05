"""Daily wellness features: sleep and training readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SECONDS_PER_HOUR = 3600.0


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records += json.loads(path.read_text())
    return records


def _by_date(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()


def load_sleep(wellness_dir: str | Path) -> pd.DataFrame:
    """Daily sleep hours (deep + light + rem) and Garmin sleep score, by calendar date."""
    rows: list[dict[str, Any]] = []
    for r in _load_records(sorted(Path(wellness_dir).glob("*sleepData.json"))):
        deep = r.get("deepSleepSeconds")
        if deep is None:
            continue  # unmeasured night
        staged = deep + (r.get("lightSleepSeconds") or 0) + (r.get("remSleepSeconds") or 0)
        rows.append(
            {
                "date": r["calendarDate"],
                "sleep_hours": staged / SECONDS_PER_HOUR,
                "sleep_score": (r.get("sleepScores") or {}).get("overallScore"),
            }
        )
    return _by_date(rows)


def load_readiness(metrics_dir: str | Path) -> pd.DataFrame:
    """Daily training-readiness score, recovery time (h), and weekly HRV, by calendar date."""
    rows: list[dict[str, Any]] = []
    for r in _load_records(sorted(Path(metrics_dir).glob("TrainingReadinessDTO*.json"))):
        if not r.get("score"):
            continue  # pre-device / invalid days carry a zero score
        rows.append(
            {
                "date": r["calendarDate"],
                "readiness_score": r.get("score"),
                "recovery_time_h": r.get("recoveryTime"),
                "hrv_weekly_avg": r.get("hrvWeeklyAverage"),
            }
        )
    return _by_date(rows)


def daily_wellness(wellness_dir: str | Path, metrics_dir: str | Path) -> pd.DataFrame:
    """Sleep and readiness features joined on calendar date."""
    return load_sleep(wellness_dir).join(load_readiness(metrics_dir), how="outer")

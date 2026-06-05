"""Garmin's own daily acute/chronic training load, as a validation reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_reference(metrics_dir: str | Path) -> pd.DataFrame:
    """Garmin's daily acute (ATL-like) and chronic (CTL-like) load, deduped by date."""
    records: list[dict[str, Any]] = []
    for path in sorted(Path(metrics_dir).glob("*AcuteTrainingLoad*.json")):
        records += json.loads(path.read_text())
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["calendarDate"], unit="ms")
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df[["dailyTrainingLoadAcute", "dailyTrainingLoadChronic"]].rename(
        columns={
            "dailyTrainingLoadAcute": "garmin_acute",
            "dailyTrainingLoadChronic": "garmin_chronic",
        }
    )


def load_race_predictions(metrics_dir: str | Path) -> pd.DataFrame:
    """Garmin's daily predicted race times (seconds) for 5K / 10K / Half / Marathon.

    The baseline our model must beat; deduped to one prediction per calendar date.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(Path(metrics_dir).glob("RunRacePredictions*.json")):
        records += json.loads(path.read_text())
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["calendarDate"])
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return df[["raceTime5K", "raceTime10K", "raceTimeHalf", "raceTimeMarathon"]].rename(
        columns={
            "raceTime5K": "pred_5k_s",
            "raceTime10K": "pred_10k_s",
            "raceTimeHalf": "pred_half_s",
            "raceTimeMarathon": "pred_marathon_s",
        }
    )


def correlation_report(fitness: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Pearson r of our CTL/ATL against Garmin's chronic/acute on overlapping days.

    Correlation is scale-invariant, so it compares the shape of the series and is
    unaffected by Garmin's different load units (sum vs mean, device weighting).
    """
    joined = fitness.join(reference, how="inner").dropna(
        subset=["ctl", "atl", "garmin_chronic", "garmin_acute"]
    )
    return {
        "ctl_vs_chronic": float(joined["ctl"].corr(joined["garmin_chronic"])),
        "atl_vs_acute": float(joined["atl"].corr(joined["garmin_acute"])),
        "n_days": float(len(joined)),
    }

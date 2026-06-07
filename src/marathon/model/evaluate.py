"""Evaluation metrics for race-time predictions."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from marathon.features.efforts import velocity_duration_frontier
from marathon.model.curve import fit_cs, fit_riegel, predict_cs, predict_riegel


def mape(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Mean absolute percentage error. Assumes actual and predicted won't have any zeros."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(np.mean(np.abs((actual - predicted) / actual)))


def rmspe(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Root mean squared percentage error. Assumes actual and predicted won't have any zeros."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(np.sqrt(np.mean(((actual - predicted) / actual) ** 2)))


def race_results(activities: pd.DataFrame) -> pd.DataFrame:
    """Actual races: date, distance, finish time."""
    races = activities.loc[
        (activities["is_race"]) & (activities["discipline"] == "run"),
        ["start_time_local", "distance_km", "duration_s"],
    ]

    races = races.assign(start_time_local=races["start_time_local"].dt.normalize()).rename(
        columns={"start_time_local": "date", "duration_s": "actual_s"}
    )

    return races


def holdout_predict(
    races: pd.DataFrame,
    points: pd.DataFrame,
    fit_fn: Callable[..., tuple[float, float]],
    predict_fn: Callable[..., float],
) -> pd.Series:
    """Predicted finish time per race, fit only on efforts before each race."""
    predictions = []

    for race in races.itertuples():
        prior = points[points["date"] < race.date]
        frontier = velocity_duration_frontier(prior)

        if len(frontier) < 2:
            predictions.append(np.nan)
        else:
            params = fit_fn(frontier)
            predictions.append(predict_fn(race.distance_km, *params))

    return pd.Series(predictions, index=races.index)


def garmin_predictions(races: pd.DataFrame, garmin: pd.DataFrame) -> pd.Series:
    """Garmin's predicted finish time per race (as of each race date), or NaN where the
    race distance has no matching standard Garmin prediction."""
    garmin_preds: list[float] = []

    for race in races.itertuples():
        race_day_preds = garmin.asof(race.date)
        distance_col = _distance_to_column(cast(float, race.distance_km))

        if distance_col is None:
            garmin_preds.append(np.nan)
        else:
            garmin_preds.append(cast(float, race_day_preds[distance_col]))

    return pd.Series(garmin_preds, index=races.index)


def evaluation_table(
    races: pd.DataFrame, points: pd.DataFrame, garmin: pd.DataFrame
) -> pd.DataFrame:
    """Per-race comparison: actual time alongside Riegel, Critical Speed, and Garmin predictions."""
    riegel = holdout_predict(races, points, fit_riegel, predict_riegel)
    cs = holdout_predict(races, points, fit_cs, predict_cs)
    garmin_s = garmin_predictions(races, garmin)

    result: pd.DataFrame = races.assign(
        riegel_s=riegel.values, cs_s=cs.values, garmin_s=garmin_s.values
    )

    return result


def _distance_to_column(distance_km: float) -> str | None:
    """Garmin prediction column for a race distance, or None if not near a standard distance."""
    match distance_km:
        case d if 9.5 <= d <= 10.5:
            return "pred_10k_s"
        case d if 20.5 <= d <= 21.6:
            return "pred_half_s"
        case d if 4.5 <= d <= 5.5:
            return "pred_5k_s"
        case d if 41 <= d <= 43:
            return "pred_marathon_s"
        case _:
            return None

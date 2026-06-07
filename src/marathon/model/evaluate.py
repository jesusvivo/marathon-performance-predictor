"""Evaluation metrics for race-time predictions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


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
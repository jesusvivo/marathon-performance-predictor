"""ONNX-exportable Riegel model built with sklearn."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from skl2onnx import to_onnx
from sklearn.linear_model import LinearRegression


def fit_riegel_sklearn(frontier: pd.DataFrame) -> LinearRegression:
    """Riegel as a log-space linear regression: ln(T) = ln(a) + b * ln(D)."""
    log_distance = np.log(frontier["distance_km"].to_numpy()).reshape(-1, 1)
    log_time = np.log(frontier["duration_s"].to_numpy())
    return LinearRegression().fit(log_distance, log_time)


def to_onnx_bytes(model: LinearRegression, sample_X: npt.NDArray[np.float64]) -> bytes:
    """Convert the fitted regressor to a serialized ONNX model."""
    onnx_model = to_onnx(model, sample_X[:1].astype(np.float32))
    return bytes(onnx_model.SerializeToString())


def onnx_log_predict(onnx_bytes: bytes, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Run the ONNX model and return its log-space predictions."""
    import onnxruntime as rt

    session = rt.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: X.astype(np.float32)})[0]
    return np.asarray(output, dtype=np.float64).ravel()

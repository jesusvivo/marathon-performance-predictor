"""Parity test: the exported ONNX model must reproduce the trained sklearn model."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from marathon.model.export import (  # noqa: E402
    fit_riegel_sklearn,
    onnx_log_predict,
    to_onnx_bytes,
)


def _frontier() -> pd.DataFrame:
    distances = np.array([5.0, 10.0, 21.097, 42.195])
    return pd.DataFrame({"distance_km": distances, "duration_s": 200.0 * distances**1.06})


def test_onnx_matches_sklearn() -> None:
    """The ONNX runtime reproduces the trained sklearn model within float tolerance."""
    frontier = _frontier()
    model = fit_riegel_sklearn(frontier)
    X = np.log(frontier["distance_km"].to_numpy()).reshape(-1, 1)

    onnx_out = onnx_log_predict(to_onnx_bytes(model, X), X)

    assert np.allclose(onnx_out, model.predict(X), atol=1e-4)

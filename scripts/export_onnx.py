"""Export the sklearn model to an ONNX file.

Usage:
    uv run --group serving python scripts/export_onnx.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from marathon.features.efforts import effort_points, velocity_duration_frontier
from marathon.model.export import fit_riegel_sklearn, onnx_log_predict, to_onnx_bytes
from marathon.parse.activities import load_export

FITNESS_GLOB = "data/DI_CONNECT/DI-Connect-Fitness/*_summarizedActivities.json"
OUT = Path("artifacts")


def main() -> None:
    activities = load_export(glob.glob(FITNESS_GLOB)[0])
    points = effort_points(activities)
    frontier = velocity_duration_frontier(points)
    X = np.log(frontier["distance_km"].to_numpy()).reshape(-1, 1)

    model = fit_riegel_sklearn(frontier)
    onnx_bytes = to_onnx_bytes(model, X)

    path = OUT / "riegel.onnx"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_bytes)

    diff = np.abs(onnx_log_predict(onnx_bytes, X) - model.predict(X)).max()
    print(f"wrote {path}  (parity diff {diff:.2e})")


if __name__ == "__main__":
    main()

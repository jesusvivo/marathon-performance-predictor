"""Build the daily feature matrix and effort points from the Garmin export.

Writes two artifacts:
  artifacts/daily_features.parquet  one row per day (model input + feature-store source)
  artifacts/effort_points.parquet   per-run velocity-duration points

Usage:
    uv run python scripts/build_features.py
"""

from __future__ import annotations

import glob
from pathlib import Path

from marathon.features.build import daily_features
from marathon.features.efforts import effort_points
from marathon.parse import load_export

FITNESS_GLOB = "data/DI_CONNECT/DI-Connect-Fitness/*_summarizedActivities.json"
WELLNESS_DIR = "data/DI_CONNECT/DI-Connect-Wellness"
METRICS_DIR = "data/DI_CONNECT/DI-Connect-Metrics"
OUT = Path("artifacts")


def main() -> None:
    activities = load_export(glob.glob(FITNESS_GLOB)[0])
    features = daily_features(activities, WELLNESS_DIR, METRICS_DIR)
    points = effort_points(activities)

    OUT.mkdir(parents=True, exist_ok=True)
    features.reset_index().to_parquet(OUT / "daily_features.parquet", index=False)
    points.to_parquet(OUT / "effort_points.parquet", index=False)

    print(f"daily_features: {features.shape} -> {OUT / 'daily_features.parquet'}")
    print(f"effort_points:  {points.shape} -> {OUT / 'effort_points.parquet'}")


if __name__ == "__main__":
    main()

"""Demo the two Feast retrieval paths for the daily fitness features.

Online: the latest feature values for the athlete, the low-latency lookup the prediction
service uses at request time. Historical: a point-in-time join returning each date's features
as they were known that day, the leakage-safe pull used to build training sets.

Requires `feast apply` and `feast materialize-incremental` to have run first.

Usage:
    uv run --group feast python scripts/feast_demo.py
"""

from pathlib import Path

import pandas as pd
from feast import FeatureStore


def main() -> None:
    path = str(Path(__file__).parent.parent / "feature_repo")
    store = FeatureStore(repo_path=path)

    # 1) ONLINE  -> latest state, the request-time serving path
    online = store.get_online_features(
        features=[
            "daily_fitness:ctl",
            "daily_fitness:atl",
            "daily_fitness:tsb",
            "daily_fitness:readiness_score",
        ],
        entity_rows=[{"athlete_id": 1}],
    ).to_dict()
    print("ONLINE (latest):", online)

    # 2) HISTORICAL -> point-in-time, the leakage-safe training pull
    entity_df = pd.DataFrame(
        {
            "athlete_id": [1, 1],
            "event_timestamp": pd.to_datetime(["2025-04-06", "2025-11-09"], utc=True),
        }
    )
    hist = store.get_historical_features(
        entity_df=entity_df,
        features=["daily_fitness:ctl", "daily_fitness:atl", "daily_fitness:tsb"],
    ).to_df()
    print("HISTORICAL (point-in-time):")
    print(hist)


if __name__ == "__main__":
    main()

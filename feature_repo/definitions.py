from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64

PARQUET = str(Path(__file__).parent.parent / "artifacts" / "daily_features.parquet")

athlete = Entity(name="athlete", join_keys=["athlete_id"], value_type=ValueType.INT64)

source = FileSource(path=PARQUET, timestamp_field="date")

daily_fitness = FeatureView(
    name="daily_fitness",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="load", dtype=Float64),
        Field(name="ctl", dtype=Float64),
        Field(name="atl", dtype=Float64),
        Field(name="tsb", dtype=Float64),
        Field(name="run_km", dtype=Float64),
        Field(name="volume_7d_km", dtype=Float64),
        Field(name="volume_28d_km", dtype=Float64),
        Field(name="avg_pace_s_per_km_28d", dtype=Float64),
        Field(name="efficiency_28d", dtype=Float64),
        Field(name="days_since_long_run", dtype=Float64),
        Field(name="vo2max", dtype=Float64),
        Field(name="sleep_hours", dtype=Float64),
        Field(name="sleep_score", dtype=Float64),
        Field(name="readiness_score", dtype=Float64),
        Field(name="recovery_time_h", dtype=Float64),
        Field(name="hrv_weekly_avg", dtype=Float64),
    ],
    source=source,
)

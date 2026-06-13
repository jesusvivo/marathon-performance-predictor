# Marathon Performance Predictor

A personal multi-distance race predictor (5K / 10K / Half / Marathon) built on my own Garmin
training data and served end-to-end: BentoML + ONNX behind a Feast feature store, deployed to
GCP Cloud Run with scale-to-zero. The headline output is a marathon finish-time projection with
an uncertainty band, tracked across an 8-month training block and validated against my actual
marathon debut in February 2027.

**Live demo:** https://race-predictor-738965554321.europe-west1.run.app (open in a browser for the
interactive Swagger UI). It is scale-to-zero, so the first request cold-starts in a few seconds:
```
curl -X POST https://race-predictor-738965554321.europe-west1.run.app/predict_marathon \
  -H "Content-Type: application/json" -d '{}'
```

## Why this project

My existing portfolio projects all stop at the notebook. This one is the deployment piece: a
live service with a feature store, online and batch inference, monitoring, and a reproducible
ingestion path for new training data.

## Dataset

The training data is my own Garmin Connect GDPR export: a multisport activity history (running,
cycling, swimming, plus strength and other cardio) starting in **May 2024**. The history depth
matters more than the raw count: CTL/ATL fitness state is an exponentially-weighted load, so the
first ~42 days are a warm-up before the long-term (42-day) average is trustworthy. Only **running**
efforts are prediction labels; the other disciplines feed systemic fatigue.

The set grows with every export, so the live counts (activities ingested, date span, weeks to race)
are not pinned here. They are computed by the pipeline (`python -m marathon.parse`) and surfaced on
the monitoring dashboard, where the countdown to the **Feb 2027 debut marathon** is the through-line.

## Requirements clarification (decided before building)

The scope was pinned up front rather than assumed:

- **Target**: multi-distance race-time prediction with uncertainty, from a personalized
  critical-speed / Riegel velocity-duration curve conditioned on fitness state. Chosen over a
  per-run duration model (low stakes), readiness (weak validation), and injury risk (noisy labels).
- **Fitness state is multisport**: load is computed across running, cycling, and swimming so the
  model sees true systemic fatigue, not running-only load. Labels remain running-only.
- **Evaluation**: per-distance MAPE / RMSPE, never an absolute-minutes average across distances.
- **Serving / features / deploy**: BentoML + ONNX, Feast (offline Parquet, online SQLite), GCP
  Cloud Run scale-to-zero. See the FinOps note below.
- **Baseline to beat**: Garmin's own daily race predictions. **Capstone**: the real Feb 2027 race.

## Architecture

Code layout as it stands:

```
src/marathon/
  parse/         # Garmin summarizedActivities.json -> tidy multisport activity frame
  features/
    load.py      # per-activity training load -> gap-free daily load series
    fitness.py   # daily load -> CTL (fitness) / ATL (fatigue) / TSB (form)
    running.py   # running daily volume, rolling load, pace, efficiency, long-run recency, VO2max
    wellness.py  # sleep and training-readiness daily features
    efforts.py   # velocity-duration effort points + best-effort frontier (model anchors)
    garmin.py    # Garmin's acute/chronic load (validation ref) + race predictions (baseline)
    build.py     # assemble the daily feature matrix from the above
  model/
    curve.py     # Riegel + Critical Speed velocity-duration fits (fit + predict)
    evaluate.py  # per-distance MAPE/RMSPE, leakage-safe holdout, Garmin baseline
    export.py    # Riegel as sklearn log-linear regression -> ONNX
  service.py     # BentoML service: ONNX predictions + Feast online features
feature_repo/    # Feast: entity, feature views, offline (parquet) + online (sqlite) stores
scripts/
  build_features.py     # write daily_features.parquet + effort_points.parquet
  validate_fitness.py   # correlate recomputed CTL/ATL against Garmin's series
  evaluate_models.py    # compare Riegel/CS/Garmin per-distance MAPE
  export_onnx.py        # write artifacts/riegel.onnx + parity check
  feast_demo.py         # online + point-in-time historical retrieval
docs/
  fitness-model.md      # the sports science: training load, CTL/ATL/TSB, calibration
  features.md           # every column in the daily feature matrix
```

The feature pipeline is a chain of pure functions: `load_export` -> `daily_load` / `running_features`
/ `daily_wellness` -> `daily_features`, each independently tested. The serving layers, ONNX export,
the Feast feature store, and a BentoML service, are in place and deployed to GCP Cloud Run with
scale-to-zero.

See [docs/features.md](docs/features.md) for the feature columns and
[docs/fitness-model.md](docs/fitness-model.md) for the physiology and Garmin calibration.

## Running the service

```
uv run --group serving --group feast bentoml serve marathon.service:RacePredictor
```
Open `localhost:3000` for the interactive Swagger UI (the full API contract: endpoints, request/
response schemas, validation), with Prometheus metrics at `/metrics` and health at `/healthz`. Each
prediction returns the finish time plus the athlete's current fitness state, looked up from Feast's
online store at request time. A live example is in the demo curl at the top of this README.

## FinOps

Cloud Run scale-to-zero with an immutable SQLite online store baked into the container image,
chosen over App Runner + managed Redis (~$300/yr fixed) for a single-user, sporadically-queried
service. Ingestion runs offline and rebuilds the image, so the live runtime stays read-only and
idles at ~$0.

## Results

On the 4 real races available, the velocity-duration baselines (Riegel / Critical Speed) lose to
Garmin's own predictions: **MAPE ~0.11 vs Garmin's 0.034**. The model improvements that might close
that gap (within-run splits, lactate-threshold-filtered efforts) turned out unsupported by an N=1
dataset with no maximal long-duration efforts. So the headline isn't beating a tuned commercial
model, it's the **end-to-end engineering**: a calibrated feature pipeline, a leakage-safe eval, and
the model served live (ONNX + Feast + BentoML on Cloud Run). The real-marathon validation is the
Feb 2027 race.

## Build log

The full phase-by-phase build log, design decisions, calibration numbers, and rejected approaches
(kept as an honest record, including null results) live in [docs/build-log.md](docs/build-log.md).

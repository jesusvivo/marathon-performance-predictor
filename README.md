# Marathon Performance Predictor

A personal multi-distance race predictor (5K / 10K / Half / Marathon) built on my own Garmin
training data and served end-to-end: BentoML + ONNX behind a Feast feature store, deployed to
GCP Cloud Run with scale-to-zero. The headline output is a marathon finish-time projection with
an uncertainty band, tracked across an 8-month training block and validated against my actual
marathon debut in February 2027.

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
the Feast feature store, and a BentoML service, are in place; Cloud Run deployment is the remaining
build phase.

See [docs/features.md](docs/features.md) for the feature columns and
[docs/fitness-model.md](docs/fitness-model.md) for the physiology and Garmin calibration.

## Running the service

```
uv run --group serving --group feast bentoml serve marathon.service:RacePredictor
```
Then predict over HTTP (served on `localhost:3000`, with a Swagger UI at the root, Prometheus
metrics at `/metrics`, and health at `/healthz`):
```
curl -X POST localhost:3000/predict_marathon -H "Content-Type: application/json" -d '{}'
curl -X POST localhost:3000/predict_race -H "Content-Type: application/json" -d '{"distance_km": 10}'
```
Each response carries the predicted finish time plus the athlete's current fitness state, looked up
from Feast's online store at request time.

## FinOps

Cloud Run scale-to-zero with an immutable SQLite online store baked into the container image,
chosen over App Runner + managed Redis (~$300/yr fixed) for a single-user, sporadically-queried
service. Ingestion runs offline and rebuilds the image, so the live runtime stays read-only and
idles at ~$0.

## Experiment log

- **Phase 1 (scaffold)**: uv project on Python 3.11 (Feast/BentoML compat), ruff + mypy-strict +
  pytest, `src/marathon` layout, dependency groups dev / serving / feast.
- **Phase 2 (parser)**: `summarizedActivities.json` to a tidy multisport activity frame (`marathon.parse`).
  - Units normalized (distance cm, duration/HR-zones ms, speed (m/s)/10, elevation/stride cm), discipline-tagged, deduped by `activityId`.
  - Verified on the full export: 581 activities (272 run / 139 strength / 71 bike / 52 swim / 32 cardio), 6 races, May 2024 to Jun 2026. 11 tests.
- **Phase 3 (features)**: daily fitness state + running/wellness/effort features, assembled into a 754 x 16 matrix (`features/`, `scripts/build_features.py`).
  - CTL/ATL/TSB via impulse-response EWMA; **CTL calibrated to Garmin's chronic load** by window sweep: 28-day CTL gives r 0.899 (vs 0.870 at 42d), ATL 7d gives r 0.829.
  - Decision rule revised r >= 0.95 to ~0.90 (Garmin's per-activity load isn't observable). See docs/fitness-model.md / docs/features.md. 32 tests.
- **Phase 4 (baseline + eval harness)**: Riegel + Critical Speed velocity-duration curves, scored against Garmin with a leakage-safe holdout (`model/`, `scripts/evaluate_models.py`).
  - **Result on 4 races: Garmin wins (MAPE 0.034) vs Riegel 0.113 / CS 0.117.** Both curves run ~11% slow (fit on whole-run average speeds). Marathon unvalidated until Feb 2027. 47 tests.
  - Two improvement levers investigated and rejected as unsupported by the data: within-run splits (heterogeneous, no clean per-km bests) and hard-efforts-only (at LT 172 bpm only 6 efforts qualify, 4 are races). The slow bias is a fundamental N=1 limit (no maximal long efforts).
- **Phase 5 (ONNX export)**: Riegel reframed as a sklearn log-linear regression and exported to ONNX (`model/export`, `scripts/export_onnx.py`).
  - `ln T = ln a + b*ln D`, so it converts via skl2onnx; sklearn coefficients reproduce the polyfit `a`/`b` exactly.
  - **Parity: ONNX runtime matches the trained model to ~3.5e-7** (float32 vs float64). Test guarded by `pytest.importorskip`. Artifact `artifacts/riegel.onnx`.
- **Phase 6 (Feast feature store)**: `daily_features` registered in Feast with offline (parquet) + online (SQLite) stores (`feature_repo/`, `scripts/feast_demo.py`).
  - `athlete` entity, `FileSource` on `date`, `daily_fitness` view over the 16 features. SQLite online store (FinOps choice over managed Redis). `feast apply` + `materialize-incremental`.
  - **Both retrieval paths verified**: `get_online_features` (latest state, request-time serving) and `get_historical_features` (leakage-safe point-in-time, the training-set builder). Registry/online DB gitignored.
- **Phase 7 (BentoML service)**: a single live service (`marathon.service:RacePredictor`) that wires ONNX + Feast together over HTTP.
  - `__init__` loads the ONNX session and the Feast store once at startup; endpoints `predict_race` / `predict_marathon` run inference and attach the current fitness state from Feast's online store. Shared logic in an undecorated `_predict` helper (BentoML endpoints must not call each other).
  - Free `/metrics` (Prometheus), `/healthz`, and Swagger UI. Verified by HTTP smoke test (a 42.195 km request returns a finish-time prediction plus live ctl/atl/tsb/readiness); no unit test, since the service needs the ONNX artifact and a materialized online store (integration territory, re-verified when containerized).
- **Phase 8 (containerize, baked-in)**: `bentofile.yaml` -> `bentoml build` (Bento) -> `bentoml containerize` (Docker image), a fully self-contained, read-only runtime.
  - The ONNX model and the materialized Feast SQLite are **baked into the image** (`include`), so the running container needs no external store, the FinOps "stateless baked-in" pattern. Image targets x86_64 (Cloud Run), runs under emulation locally.
  - Three packaging gotchas solved: the `src/` layout puts the package at `src/src/marathon` in the Bento (fixed with `PYTHONPATH=/home/bentoml/bento/src/src`); the container's cwd is the bento root, so the ONNX path was anchored to `Path(__file__)` (cwd-independent, like the feature repo); and `python.packages` must list the serving deps explicitly (onnxruntime/feast/pandas).
  - **Verified in-container**: with no project directory present, the image serves marathon/10K predictions plus the baked fitness state, and exposes Prometheus metrics.

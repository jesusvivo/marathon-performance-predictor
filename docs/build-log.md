# Build log

The phase-by-phase record of how this project was built: what each step delivered, the design
decisions, the calibration results, and the approaches that were tried and rejected. Kept as an
honest engineering narrative (including null results), separate from the README so the front page
stays concise.

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
- **Phase 9 (Cloud Run deploy)**: the baked image deployed to GCP Cloud Run as a public, scale-to-zero service (live URL in the README).
  - `--min-instances=0` (idles at $0), `--max-instances=1` (caps worst-case spend on the public endpoint), plus a billing budget alert. First request cold-starts in a few seconds, then back to $0 when idle.
  - Arch gotcha: the locally built image was arm64 (Apple Silicon) but Cloud Run needs amd64. Rebuilt server-side with Cloud Build, which also sidestepped a slow ~1.75 GB image upload (only the 156 KB bento source goes up; GCP installs the deps and builds on its network).

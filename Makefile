.PHONY: features onnx materialize ingest deploy

# Rebuild the daily feature matrix + effort points from the current Garmin export.
features:
	uv run python scripts/build_features.py

# Refit Riegel as sklearn, export to ONNX, and parity-check.
onnx:
	uv run --group serving python scripts/export_onnx.py

# Register feature definitions and refresh the Feast online store up to now.
materialize:
	cd feature_repo && uv run --group feast feast apply && \
	  uv run --group feast feast materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)

# Full idempotent refresh after dropping a fresh Garmin export into data/.
ingest: features onnx materialize

# Build the image (Cloud Build, amd64) and deploy to Cloud Run.
deploy:
	bash scripts/deploy.sh

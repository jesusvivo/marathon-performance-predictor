#!/usr/bin/env bash
# Build the Bento, build an amd64 image in Cloud Build, and deploy to Cloud Run.
# Run after `make ingest` to ship a refreshed model + features (or to ship code changes).
# Cloud Build is used (not local `bentoml containerize`) so the image is amd64 for Cloud Run
# and only the small bento source uploads, not the multi-GB image.
set -euo pipefail

PROJECT=marathon-performance-predictor
REGION=europe-west1
IMAGE="$REGION-docker.pkg.dev/$PROJECT/marathon/race_predictor:latest"
SERVICE=race-predictor

echo "==> Building Bento"
uv run --group serving --group feast bentoml build

BENTO=$(uv run --group serving --group feast bentoml get race_predictor:latest -o path)
echo "==> Bento built at $BENTO"

echo "==> Writing Cloud Build config into the bento"
cat > "$BENTO/cloudbuild.yaml" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ['build', '-t', '$IMAGE', '-f', 'env/docker/Dockerfile', '.']
images:
  - $IMAGE
EOF

echo "==> Building amd64 image in Cloud Build"
gcloud builds submit --config="$BENTO/cloudbuild.yaml" "$BENTO"

echo "==> Deploying to Cloud Run"
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" --port=3000 --allow-unauthenticated \
  --min-instances=0 --max-instances=1

echo "==> Live at:"
gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)'

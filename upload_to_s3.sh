#!/bin/bash

# === VARIABLES ===
LOCAL_DIR="./training-runs"
S3_BUCKET="aws-gpu-monitoring-logs"
S3_PREFIX="stylegan2-ada-results"
GPU_TAG="nvidia-t4"

# === Sync all training run data to S3 ===
aws s3 sync "$LOCAL_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" --storage-class STANDARD_IA

# === Optionally tag the uploaded objects ===
# Note: S3 doesn't support direct object tagging on sync; you must apply tags after upload.
for file in $(find "$LOCAL_DIR" -type f); do
  key="${file#./training-runs/}"
  key_path="${S3_PREFIX}/${key}"

  aws s3api put-object-tagging \
    --bucket "$S3_BUCKET" \
    --key "$key_path" \
    --tagging "TagSet=[{Key=GPU,Value=$GPU_TAG}]" || true
done

echo "Upload complete with tagging."
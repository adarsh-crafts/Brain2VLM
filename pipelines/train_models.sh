#!/usr/bin/env bash
set -e

echo "Using RUN_ID=$RUN_ID"
echo "=== Training models ==="

python3 scripts/train_ridge_fmri_to_clip_image.py
python3 scripts/train_ridge_fmri_to_clip_text.py

echo "Models trained."

echo ""
echo "=== Generating predictions ==="
python3 scripts/predict_ridge_fmri_to_clip_multimodal.py

echo "Predictions generated."

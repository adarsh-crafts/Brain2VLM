#!/usr/bin/env bash
set -e

echo "=== Training models ==="

python scripts/train_ridge_fmri_to_clip_image.py
python scripts/train_ridge_fmri_to_clip_text.py

echo "Models trained."

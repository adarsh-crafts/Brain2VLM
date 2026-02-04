#!/usr/bin/env bash
set -e

echo "Using RUN_ID=$RUN_ID"
echo "=== fMRI preprocessing ==="

python3 scripts/extract_fmri_betas.py
python3 scripts/build_trial_image_mapping.py
python3 scripts/build_image_level_betas.py

echo "fMRI preprocessing done."

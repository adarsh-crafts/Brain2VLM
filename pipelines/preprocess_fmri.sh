#!/usr/bin/env bash
set -e

echo "=== fMRI preprocessing ==="

python scripts/extract_fmri_betas.py
python scripts/build_trial_image_mapping.py
python scripts/build_image_level_betas.py

echo "fMRI preprocessing done."

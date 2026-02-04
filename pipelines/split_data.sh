#!/usr/bin/env bash
set -e

echo "Using RUN_ID=$RUN_ID"
echo "=== Dataset splits ==="

python3 scripts/build_image_splits.py

echo "Splits done."

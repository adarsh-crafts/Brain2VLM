#!/usr/bin/env bash
set -e

echo "=== Dataset splits ==="

python scripts/build_image_splits.py

echo "Splits done."

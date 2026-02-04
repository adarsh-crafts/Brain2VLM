#!/usr/bin/env bash
set -e

echo "Using RUN_ID=$RUN_ID"
echo "=== CLIP embeddings ==="

echo ""
echo "[1/3] Building COCO → NSD mapping..."
python3 scripts/build_coco_nsd_mapping.py

echo ""
echo "[2/3] Extracting CLIP image embeddings..."
python3 scripts/extract_clip_image_embeddings.py

echo ""
echo "[3/3] Extracting CLIP text embeddings..."
python3 scripts/extract_clip_text_embeddings.py

echo ""
echo "CLIP embeddings done."

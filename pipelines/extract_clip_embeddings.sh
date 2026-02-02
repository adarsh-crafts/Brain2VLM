#!/usr/bin/env bash
set -e

echo "=== CLIP embeddings ==="

echo ""
echo "[1/3] Building COCO → NSD mapping..."
python scripts/build_coco_nsd_mapping.py

echo ""
echo "[2/3] Extracting CLIP image embeddings..."
python scripts/extract_clip_image_embeddings.py

echo ""
echo "[3/3] Extracting CLIP text embeddings..."
python scripts/extract_clip_text_embeddings.py

echo ""
echo "CLIP embeddings done."

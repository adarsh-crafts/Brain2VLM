#!/usr/bin/env bash
set -e

pipelines/preprocess_fmri.sh
pipelines/extract_clip_embeddings.sh
pipelines/split_data.sh
pipelines/train_models.sh

echo "========================================"
echo " Linear decoders trained successfully"
echo "========================================"

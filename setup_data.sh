#!/usr/bin/env bash
set -e

echo "=========================================="
echo " NSD / COCO Raw Feature Setup"
echo "=========================================="

# Optional: activate environment
# source .venv/bin/activate

############################################
# 1. Extract masked fMRI betas
############################################

echo ""
echo "[1/3] Extracting masked fMRI betas..."
python scripts/extract_fmri_betas.py


############################################
# 2. Extract CLIP image embeddings
############################################

echo ""
echo "[2/3] Extracting CLIP image embeddings..."

python src/models/clip/image_embeddings.py \
    --input_hdf5 data/nsddata/stimuli/nsd_stimuli.hdf5 \
    --output_hdf5 data/embeddings/clip/clip_img_embeddings.hdf5 \
    --batch_size 64


############################################
# 3. Extract CLIP text embeddings
############################################

echo ""
echo "[3/3] Extracting CLIP text embeddings..."

python src/models/clip/text_embeddings.py \
    --annotations_dir data/coco_annotations \
    --output_train data/embeddings/clip/clip_captions_train2017_embeddings.hdf5 \
    --output_val data/embeddings/clip/clip_captions_val2017_embeddings.hdf5 \
    --batch_size 128


echo ""
echo "=========================================="
echo " Raw feature setup complete"
echo "=========================================="

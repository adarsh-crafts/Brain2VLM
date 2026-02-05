"""
Prepare CLIP text embeddings for NSD images.

Aggregates caption embeddings per image and saves the image-ordered matrix.

Usage:
    python prepare_clip_text_embeddings.py
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def main():
    image_ids_path = Path("data/processed/fmri/subj01_image_ids.npy")
    train_h5_path = Path("data/embeddings/clip/clip_captions_train2017_embeddings.hdf5")
    val_h5_path = Path("data/embeddings/clip/clip_captions_val2017_embeddings.hdf5")
    output_path = Path("data/processed/clip/subj01_text_embeddings.npy")

    print("Loading subject image ids...")
    image_ids = np.load(image_ids_path)
    print(f"Count: {len(image_ids)}")

    # Load all caption embeddings and build mapping
    print("Loading caption embeddings from train2017...")
    caption_map = defaultdict(list)
    
    with h5py.File(train_h5_path, "r") as f:
        text_emb = f["text_embeddings"][:]
        assert text_emb.shape[1] == 768
        img_ids = f["image_ids"][:]
        print(f"Train captions: {len(text_emb)}")
        
        for i in range(len(img_ids)):
            caption_map[int(img_ids[i])].append(text_emb[i])
    
    print("Loading caption embeddings from val2017...")
    with h5py.File(val_h5_path, "r") as f:
        text_emb = f["text_embeddings"][:]
        assert text_emb.shape[1] == 768
        img_ids = f["image_ids"][:]
        print(f"Val captions: {len(text_emb)}")
        
        for i in range(len(img_ids)):
            caption_map[int(img_ids[i])].append(text_emb[i])
    
    print(f"Total unique images with captions: {len(caption_map)}")
    
    # Process each image
    print("Averaging captions per image...")
    text_embeddings = []
    caption_counts = []
    missing_images = []
    
    for img_id in tqdm(image_ids, desc="Processing"):
        img_id_int = int(img_id)
        
        if img_id_int not in caption_map:
            missing_images.append(img_id_int)
            # Use zero vector for missing images
            text_embeddings.append(np.zeros(768, dtype=np.float32))
            caption_counts.append(0)
        else:
            captions = caption_map[img_id_int]
            # Average all captions for this image
            avg_caption = np.mean(captions, axis=0).astype(np.float32)
            text_embeddings.append(avg_caption)
            caption_counts.append(len(captions))
    
    # Convert to array
    text_embeddings = np.array(text_embeddings, dtype=np.float32)
    assert text_embeddings.shape[1] == 768
    caption_counts = np.array(caption_counts)
    
    # Check for missing images
    if missing_images:
        raise ValueError(
            f"Found {len(missing_images)} images without captions. "
            f"First few missing IDs: {missing_images[:10]}"
        )
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, text_embeddings)
    
    print(f"\nSaved: {output_path}")
    print(f"Final shape: {text_embeddings.shape}")
    print(f"Min captions per image: {caption_counts.min()}")
    print(f"Max captions per image: {caption_counts.max()}")
    print(f"Mean captions per image: {caption_counts.mean():.2f}")


if __name__ == "__main__":
    main()

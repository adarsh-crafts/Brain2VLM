"""
Prepare CLIP image embeddings for NSD images.

Indexes CLIP image embeddings by NSD image IDs and saves the subject-ordered
embedding matrix.

Usage:
    python prepare_clip_image_embeddings.py
"""

import numpy as np
import h5py
from pathlib import Path


def main():
    nsd_ids_path = Path("data/processed/mappings/subj01_nsd_image_ids.npy")
    embeddings_h5_path = Path("data/embeddings/clip/clip_img_embeddings.hdf5")
    output_path = Path("data/processed/clip/subj01_image_embeddings.npy")

    print("Loading NSD image ids...")
    nsd_ids = np.load(nsd_ids_path)
    print("Count:", len(nsd_ids))

    print("Opening HDF5...")
    with h5py.File(embeddings_h5_path, "r") as f:
        emb = f["image_embeddings"]
        print("HDF5 shape:", emb.shape)

        print("Indexing embeddings...")

        order = np.argsort(nsd_ids)
        sorted_ids = nsd_ids[order]

        sorted_embeddings = emb[sorted_ids]

        inv_order = np.argsort(order)
        subj_embeddings = sorted_embeddings[inv_order]

    subj_embeddings = subj_embeddings.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, subj_embeddings)

    print("Saved:", output_path)
    print("Final shape:", subj_embeddings.shape)


if __name__ == "__main__":
    main()

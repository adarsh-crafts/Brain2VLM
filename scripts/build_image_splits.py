"""
Prepare image-level train/test splits.

Uses NSD metadata to split subject image IDs into train/test subsets.

Usage:
    python prepare_image_splits.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    csv_path = Path("data/nsddata/experiments/nsd/nsd_stim_info_merged.csv")
    image_ids_path = Path("data/processed/fmri/subj01_image_ids.npy")
    output_path = Path("data/processed/splits/subj01_image_splits.npz")

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    if "cocoId" not in df.columns or "shared1000" not in df.columns:
        raise RuntimeError("CSV must contain cocoId and shared1000")

    print("Loading subj01 image ids...")
    subj_image_ids = np.load(image_ids_path)

    print("Building cocoId → shared1000 mapping...")
    coco_to_shared = dict(zip(df["cocoId"].astype(int), df["shared1000"].astype(bool)))

    train_idx = []
    test_idx = []

    for i, coco in enumerate(subj_image_ids):
        if coco_to_shared[int(coco)]:
            test_idx.append(i)
        else:
            train_idx.append(i)

    train_idx = np.array(train_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(train_idx) + len(test_idx) == len(subj_image_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, train_idx=train_idx, test_idx=test_idx)

    print("\nSummary:")
    print("Train images:", len(train_idx))
    print("Test images:", len(test_idx))
    print("Total images:", len(subj_image_ids))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

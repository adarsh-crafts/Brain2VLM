"""
Prepare COCO-to-NSD image ID mapping.

Maps COCO image IDs to NSD image IDs for the subject image list.

Usage:
    python prepare_coco_nsd_mapping.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    csv_path = Path("data/nsddata/experiments/nsd/nsd_stim_info_merged.csv")
    coco_ids_path = Path("data/processed/fmri/subj01_image_ids.npy")   # cocoIds
    output_path = Path("data/processed/mappings/subj01_nsd_image_ids.npy")

    print("Loading CSV...")
    df = pd.read_csv(csv_path)

    print("Loading coco image ids...")
    coco_ids = np.load(coco_ids_path)

    print("Building cocoId → nsdId mapping...")
    coco_to_nsd = dict(zip(df["cocoId"].astype(int), df["nsdId"].astype(int)))

    nsd_ids = np.array([coco_to_nsd[int(c)] for c in coco_ids], dtype=np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, nsd_ids)

    print("Saved:", output_path)
    print("Min nsdId:", nsd_ids.min())
    print("Max nsdId:", nsd_ids.max())
    print("Count:", len(nsd_ids))


if __name__ == "__main__":
    main()

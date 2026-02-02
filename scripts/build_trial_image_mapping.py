"""
Prepare trial-to-image mapping for NSD stimuli.

Builds the mapping from trial indices to COCO image IDs and records keep masks.

Usage:
    python prepare_trial_image_mapping.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    csv_path = Path("data/nsddata/experiments/nsd/nsd_stim_info_merged.csv")
    output_map = Path("data/processed/mappings/subj01_trial_to_image.npy")
    output_keep = Path("data/processed/fmri/subj01_keep_trials.npy")

    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    if "cocoId" not in df.columns:
        raise RuntimeError("Missing cocoId")

    # subj01 repetition columns
    rep_cols = ["subject1_rep0", "subject1_rep1", "subject1_rep2"]

    for c in rep_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c}")

    N_TRIALS = 31125
    trial_to_image = np.full(N_TRIALS, -1, dtype=np.int32)

    print("Building trial → cocoId mapping...")

    for _, row in df.iterrows():
        coco = int(row["cocoId"])

        for c in rep_cols:
            t = int(row[c])
            if t > 0:              # IMPORTANT: 0 means absent
                trial_to_image[t] = coco

    keep = trial_to_image >= 0

    print(f"Keeping {keep.sum()} stimulus trials")
    print(f"Dropping {np.sum(~keep)} non-stimulus trials")

    output_map.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_map, trial_to_image[keep])
    np.save(output_keep, keep)

    print("Saved:")
    print(output_map)
    print(output_keep)
    print("Final mapped trials:", keep.sum())
    print("Unique images:", len(np.unique(trial_to_image[keep])))


if __name__ == "__main__":
    main()

"""
Prepare image-level fMRI betas.

Loads trial-level betas, applies trial filters and mappings, and averages
betas per image.

Usage:
    python prepare_image_level_betas.py
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path


def main():
    betas_path = Path("data/processed/fmri/subj01_masked_betas.npy")
    trial_to_image_path = Path("data/processed/mappings/subj01_trial_to_image.npy")
    keep_path = Path("data/processed/fmri/subj01_keep_trials.npy")

    output_betas_path = Path("data/processed/fmri/subj01_image_level_betas.npy")
    output_ids_path = Path("data/processed/fmri/subj01_image_ids.npy")

    print(f"Loading betas from raw binary file {betas_path}...")

    N_TRIALS = 31125
    betas = np.fromfile(betas_path, dtype=np.float32)
    betas = betas.reshape(N_TRIALS, -1)

    keep = np.load(keep_path)
    betas = betas[keep]

    print(f"Loading trial-to-image mapping from {trial_to_image_path}...")
    trial_to_image = np.load(trial_to_image_path)

    assert betas.shape[0] == trial_to_image.shape[0]

    print(f"Filtered betas shape: {betas.shape}")

    unique_images = np.unique(trial_to_image)
    n_unique = len(unique_images)

    print(f"Unique images: {n_unique}")

    image_betas = []

    for img in tqdm(unique_images, desc="Averaging betas"):
        idx = np.where(trial_to_image == img)[0]
        image_betas.append(betas[idx].mean(axis=0))

    image_betas = np.stack(image_betas)

    mean_reps = betas.shape[0] / n_unique

    output_betas_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_betas_path, image_betas)
    np.save(output_ids_path, unique_images)

    print("\nSummary:")
    print("Image betas:", image_betas.shape)
    print("Mean repetitions per image:", mean_reps)


if __name__ == "__main__":
    main()

"""
Extract fMRI betas from NSD volumes.

This script loads NSD beta volumes, applies an ROI mask, and saves the resulting
masked voxel vectors as a stacked numpy array.

Usage:
    python extract_fmri_betas.py
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


def load_roi_mask(mask_path):
    """
    Load and prepare the ROI mask from a NIfTI file.

    Parameters:
    -----------
    mask_path : str
        Path to the NIfTI mask file

    Returns:
    --------
    roi_mask : np.ndarray
        Boolean array of shape (num_voxels,) where True indicates mask > 0
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()  # Shape: (81, 104, 83) in nibabel convention

    # Transpose to match betas shape: (83, 104, 81)
    mask_data = mask_data.transpose(2, 1, 0)

    # Flatten the mask
    mask_flat = mask_data.flatten()

    # Create boolean mask for voxels > 0
    roi_mask = mask_flat > 0

    return roi_mask


def get_beta_files(betas_dir):
    """
    Get sorted list of beta NIfTI files from directory.

    Parameters:
    -----------
    betas_dir : str
        Path to directory containing beta files

    Returns:
    --------
    beta_files : list
        Sorted list of beta file paths (.nii or .nii.gz)
    """
    betas_path = Path(betas_dir)

    if not betas_path.exists():
        raise FileNotFoundError(f"Betas directory not found: {betas_dir}")

    # Find all .nii and .nii.gz files
    beta_files = sorted(betas_path.glob("*.nii.gz")) + sorted(betas_path.glob("*.nii"))

    if not beta_files:
        raise FileNotFoundError(f"No beta files (.nii/.nii.gz) found in {betas_dir}")

    return beta_files


def load_and_mask_beta(beta_path, roi_mask):
    """
    Load beta NIfTI and apply ROI mask.
    Handles both 3D and 4D beta files.

    Returns:
        list of masked voxel vectors (one per trial in file)
    """
    beta_img = nib.load(beta_path)
    beta_data = beta_img.get_fdata()

    # Fix orientation if needed (NSD uses 81,104,83 -> want 83,104,81)
    if beta_data.shape[:3] == (81, 104, 83):
        if beta_data.ndim == 3:
            beta_data = beta_data.transpose(2, 1, 0)
        elif beta_data.ndim == 4:
            beta_data = beta_data.transpose(2, 1, 0, 3)


    roi_flat = roi_mask.flatten()

    masked_list = []

    if beta_data.ndim == 3:
        beta_flat = beta_data.flatten()
        masked_list.append(beta_flat[roi_flat])

    elif beta_data.ndim == 4:
        for i in range(beta_data.shape[-1]):
            beta_flat = beta_data[..., i].flatten()
            masked_list.append(beta_flat[roi_flat])

    else:
        raise ValueError(f"Unexpected beta shape: {beta_data.shape}")

    return masked_list



def save_masked_betas(
    betas_dir,
    mask_path,
    output_dir="data/processed",
    subject="subj01",
):
    """
    Extract masked NSD betas and save to .npy files using streaming writes.
    
    Uses numpy memmap to write results incrementally instead of accumulating
    all trials in memory.

    Parameters:
    -----------
    betas_dir : str
        Path to directory with beta NIfTI files
    mask_path : str
        Path to ROI mask NIfTI file
    output_dir : str
        Directory to save output .npy files
    subject : str
        Subject identifier for output filenames

    Returns:
    --------
    num_trials : int
        Total number of trials processed
    num_voxels_in_roi : int
        Number of voxels in ROI
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading ROI mask from: {mask_path}")
    roi_mask = load_roi_mask(mask_path)
    num_voxels_in_roi = roi_mask.sum()
    print(f"  ROI contains {num_voxels_in_roi} voxels")

    print(f"\nLoading beta files from: {betas_dir}")
    beta_files = get_beta_files(betas_dir)
    print(f"  Found {len(beta_files)} beta files")

    # First pass: count total number of trials
    print("\nFirst pass: counting trials...")
    num_trials = 0
    for beta_file in tqdm(beta_files, desc="Counting trials"):
        masked_trials = load_and_mask_beta(beta_file, roi_mask)
        num_trials += len(masked_trials)
    
    print(f"  Total trials across all files: {num_trials}")

    # Preallocate memmap for output
    output_betas_path = output_path / f"{subject}_masked_betas.npy"
    
    print(f"\nPreallocating output array: shape=({num_trials}, {num_voxels_in_roi}), dtype=float32")
    masked_betas_mm = np.memmap(
        output_betas_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_trials, num_voxels_in_roi)
    )

    # Second pass: write to memmap in batches
    print("\nSecond pass: writing to disk...")
    row_idx = 0
    
    for beta_file in tqdm(beta_files, desc="Writing betas"):
        masked_trials = load_and_mask_beta(beta_file, roi_mask)
        n_trials = len(masked_trials)
        
        # Convert to array and write to memmap
        trials_array = np.array(masked_trials, dtype=np.float32)
        masked_betas_mm[row_idx:row_idx + n_trials] = trials_array
        
        row_idx += n_trials

    # Flush to disk
    print("\nFlushing to disk...")
    masked_betas_mm.flush()
    
    print(f"Saved masked betas to: {output_betas_path}")
    print(f"  Final shape: ({num_trials}, {num_voxels_in_roi})")
    print(f"  Data type: float32")

    # Save trial indices
    output_indices_path = output_path / f"{subject}_trial_indices.npy"
    trial_indices = np.arange(num_trials)
    np.save(output_indices_path, trial_indices)
    print(f"Saved trial indices to: {output_indices_path}")

    return num_trials, num_voxels_in_roi


def main():
    """Main entry point."""
    # Configuration
    BETAS_DIR = "data/nsddata/betas/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    ROI_PATH = "data/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz"
    OUTPUT_DIR = "data/processed/fmri"
    SUBJECT = "subj01"

    print("=" * 70)
    print("Extract Masked NSD Beta Responses")
    print("=" * 70)

    # Process and save
    num_trials, num_voxels = save_masked_betas(
        betas_dir=BETAS_DIR,
        mask_path=ROI_PATH,
        output_dir=OUTPUT_DIR,
        subject=SUBJECT,
    )

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Total trials: {num_trials}")
    print(f"  Voxels in ROI: {num_voxels}")
    print(f"  Output shape: ({num_trials}, {num_voxels})")
    print(f"  Data type: float32")
    print("=" * 70)


if __name__ == "__main__":
    main()

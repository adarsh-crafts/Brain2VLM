"""
Create an SD-IPC regression dataset from NSD fMRI and COCO metadata.

Inputs (CLI):
  --nsd_root: NSD dataset root (expects betas/, ppdata/, experiments/, stimuli/)
  --coco_annotations: COCO annotations directory (official annotations)
  --output_dir: base output directory (default: data/processed)
  --subject: subject identifier (default: subj01)
  --roi: ROI name (default: nsdgeneral)

Outputs (in output_dir):
  - fMRI masked betas and trial indices (.npy)
  - image-level betas and image ids (.npy)
  - trial-to-image mapping and keep mask (.npy)
  - COCO-to-NSD mapping (.npy)
  - train/test split file (.npz)

Consolidates logic from:
  scripts/build_trial_image_mapping.py
  scripts/build_coco_nsd_mapping.py
  scripts/extract_fmri_betas.py
  scripts/build_image_level_betas.py
  scripts/build_image_splits.py

Prepares aligned data for SD-IPC regression (fMRI -> CLIP -> SD).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from extract_fmri_betas import save_masked_betas
import stimuli_loader


N_TRIALS = 31125


def _subject_number(subject: str) -> int:
    try:
        return int(subject.replace("subj", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid subject format: {subject}") from exc


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_or_build_trial_mapping(csv_path: Path, output_map: Path, output_keep: Path, subject: str):
    # From scripts/build_trial_image_mapping.py
    if output_map.exists() and output_keep.exists():
        trial_to_image = np.load(output_map)
        keep = np.load(output_keep)
        return trial_to_image, keep

    df = pd.read_csv(csv_path)
    if "cocoId" not in df.columns:
        raise RuntimeError("Missing cocoId")

    subj_num = _subject_number(subject)
    rep_cols = [f"subject{subj_num}_rep0", f"subject{subj_num}_rep1", f"subject{subj_num}_rep2"]
    for c in rep_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c}")

    trial_to_image = np.full(N_TRIALS, -1, dtype=np.int32)

    for _, row in df.iterrows():
        coco = int(row["cocoId"])
        for c in rep_cols:
            t = int(row[c])
            if t > 0:  # IMPORTANT: 0 means absent
                trial_to_image[t] = coco

    keep = trial_to_image >= 0

    _ensure_parent(output_map)
    _ensure_parent(output_keep)
    np.save(output_map, trial_to_image[keep])
    np.save(output_keep, keep)

    return trial_to_image[keep], keep


def _load_or_build_coco_nsd_mapping(csv_path: Path, coco_ids_path: Path, output_path: Path):
    # From scripts/build_coco_nsd_mapping.py
    if output_path.exists():
        return np.load(output_path)

    df = pd.read_csv(csv_path)

    coco_ids = np.load(coco_ids_path)

    coco_to_nsd = dict(zip(df["cocoId"].astype(int), df["nsdId"].astype(int)))
    nsd_ids = np.array([coco_to_nsd[int(c)] for c in coco_ids], dtype=np.int32)

    _ensure_parent(output_path)
    np.save(output_path, nsd_ids)

    return nsd_ids


def _load_or_extract_masked_betas(
    betas_dir: Path,
    roi_path: Path,
    output_dir: Path,
    subject: str,
):
    # From scripts/extract_fmri_betas.py
    masked_betas_path = output_dir / f"{subject}_masked_betas.npy"
    trial_indices_path = output_dir / f"{subject}_trial_indices.npy"

    if masked_betas_path.exists() and trial_indices_path.exists():
        trial_indices = np.load(trial_indices_path)
        if len(trial_indices) == 0:
            raise RuntimeError("Trial indices file is empty")

        byte_size = masked_betas_path.stat().st_size
        num_trials = len(trial_indices)
        num_voxels = byte_size // (4 * num_trials)
        if num_trials * num_voxels * 4 != byte_size:
            raise RuntimeError("Masked betas file size does not match trial indices")

        return num_trials, num_voxels

    num_trials, num_voxels = save_masked_betas(
        betas_dir=str(betas_dir),
        mask_path=str(roi_path),
        output_dir=str(output_dir),
        subject=subject,
    )

    return num_trials, num_voxels


def _load_or_build_image_level_betas(
    betas_path: Path,
    trial_to_image_path: Path,
    keep_path: Path,
    output_betas_path: Path,
    output_ids_path: Path,
):
    # From scripts/build_image_level_betas.py
    if output_betas_path.exists() and output_ids_path.exists():
        image_betas = np.load(output_betas_path)
        image_ids = np.load(output_ids_path)
        return image_betas, image_ids

    betas = np.fromfile(betas_path, dtype=np.float32)
    betas = betas.reshape(N_TRIALS, -1)

    keep = np.load(keep_path)
    betas = betas[keep]

    trial_to_image = np.load(trial_to_image_path)

    if betas.shape[0] != trial_to_image.shape[0]:
        raise RuntimeError("Trial-to-image mapping does not align with filtered betas")

    unique_images = np.unique(trial_to_image)

    image_betas = []
    for img in tqdm(unique_images, desc="Averaging betas"):
        idx = np.where(trial_to_image == img)[0]
        image_betas.append(betas[idx].mean(axis=0))

    image_betas = np.stack(image_betas)

    _ensure_parent(output_betas_path)
    np.save(output_betas_path, image_betas)
    np.save(output_ids_path, unique_images)

    return image_betas, unique_images


def _load_or_build_splits(csv_path: Path, image_ids_path: Path, output_path: Path):
    # From scripts/build_image_splits.py
    if output_path.exists():
        splits = np.load(output_path)
        return splits["train_idx"], splits["test_idx"]

    df = pd.read_csv(csv_path)

    if "cocoId" not in df.columns or "shared1000" not in df.columns:
        raise RuntimeError("CSV must contain cocoId and shared1000")

    subj_image_ids = np.load(image_ids_path)
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

    _ensure_parent(output_path)
    np.savez(output_path, train_idx=train_idx, test_idx=test_idx)

    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SD-IPC regression dataset from NSD fMRI and COCO metadata"
    )
    parser.add_argument(
        "--nsd_root",
        type=str,
        required=True,
        help="NSD dataset root containing betas/, ppdata/, experiments/, stimuli/",
    )
    parser.add_argument(
        "--coco_annotations",
        type=str,
        required=True,
        help="Path to official COCO annotations directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Base output directory (default: data/processed)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="subj01",
        help="Subject identifier (default: subj01)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="nsdgeneral",
        help="ROI name (default: nsdgeneral)",
    )

    args = parser.parse_args()

    nsd_root = Path(args.nsd_root)
    coco_annotations = Path(args.coco_annotations)
    output_root = Path(args.output_dir)
    subject = args.subject
    roi = args.roi

    if not nsd_root.exists():
        raise FileNotFoundError(f"NSD root not found: {nsd_root}")
    if not coco_annotations.exists():
        raise FileNotFoundError(f"COCO annotations directory not found: {coco_annotations}")

    csv_path = nsd_root / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    stimuli_hdf5_path = nsd_root / "stimuli" / "nsd" / "nsd_stimuli.hdf5"

    output_fmri_dir = output_root / "fmri"
    output_mappings_dir = output_root / "mappings"
    output_splits_dir = output_root / "splits"

    trial_map_path = output_mappings_dir / f"{subject}_trial_to_image.npy"
    keep_path = output_fmri_dir / f"{subject}_keep_trials.npy"
    image_betas_path = output_fmri_dir / f"{subject}_image_level_betas.npy"
    image_ids_path = output_fmri_dir / f"{subject}_image_ids.npy"
    coco_nsd_map_path = output_mappings_dir / f"{subject}_nsd_image_ids.npy"
    splits_path = output_splits_dir / f"{subject}_image_splits.npz"

    betas_dir = nsd_root / "betas" / subject / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR"
    roi_path = nsd_root / "ppdata" / subject / "func1pt8mm" / "roi" / f"{roi}.nii.gz"

    print("=" * 70)
    print("Preparing SD-IPC dataset")
    print("=" * 70)

    print("\nStep 1: Trial -> image mapping")
    trial_to_image, keep = _load_or_build_trial_mapping(
        csv_path=csv_path,
        output_map=trial_map_path,
        output_keep=keep_path,
        subject=subject,
    )

    trial_indices = np.where(keep)[0].astype(np.int32)
    _ = stimuli_loader._normalize_indices(trial_indices.tolist())  # From src/data/stimuli_loader.py

    print("\nStep 2: Extract masked fMRI betas")
    num_trials, num_voxels = _load_or_extract_masked_betas(
        betas_dir=betas_dir,
        roi_path=roi_path,
        output_dir=output_fmri_dir,
        subject=subject,
    )

    print("\nStep 3: Image-level betas")
    image_betas, image_ids = _load_or_build_image_level_betas(
        betas_path=output_fmri_dir / f"{subject}_masked_betas.npy",
        trial_to_image_path=trial_map_path,
        keep_path=keep_path,
        output_betas_path=image_betas_path,
        output_ids_path=image_ids_path,
    )

    print("\nStep 4: COCO -> NSD mapping")
    nsd_ids = _load_or_build_coco_nsd_mapping(
        csv_path=csv_path,
        coco_ids_path=image_ids_path,
        output_path=coco_nsd_map_path,
    )

    print("\nStep 5: Train/test splits")
    train_idx, test_idx = _load_or_build_splits(
        csv_path=csv_path,
        image_ids_path=image_ids_path,
        output_path=splits_path,
    )

    print("\nSummary")
    print("- Trials (kept):", len(trial_to_image))
    print("- Unique images:", len(image_ids))
    print("- Masked betas shape:", (num_trials, num_voxels))
    print("- Image-level betas shape:", image_betas.shape)
    print("- Train images:", len(train_idx))
    print("- Test images:", len(test_idx))
    print("- Val images: 0 (validation is derived during training)")
    print("- Stimuli HDF5:", stimuli_hdf5_path)

    print("\nOutputs:")
    print("-", image_betas_path)
    print("-", image_ids_path)
    print("-", trial_map_path)
    print("-", keep_path)
    print("-", coco_nsd_map_path)
    print("-", splits_path)


if __name__ == "__main__":
    main()

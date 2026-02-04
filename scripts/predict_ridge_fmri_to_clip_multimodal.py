"""
Predict CLIP image and text embeddings from fMRI using trained ridge models.

Loads trained ridge regression models and generates predictions for both
CLIP image and text embeddings from fMRI betas. Outputs normalized embeddings
and metadata.

Usage:
    python predict_ridge_fmri_to_clip_multimodal.py --subject subj01
"""

import numpy as np
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def main(subject="subj01"):
    """
    Predict CLIP image and text embeddings from fMRI betas using trained models.
    
    Args:
        subject: Subject identifier (default: subj01)
    """
    print(f"Predicting CLIP embeddings from fMRI for {subject}")
    
    # Set deterministic seed
    np.random.seed(0)
    
    # Define paths
    betas_path = Path(f"data/processed/fmri/{subject}_masked_betas.npy")
    trial_indices_path = Path(f"data/processed/fmri/{subject}_trial_indices.npy")
    
    # Shared run_id mechanism: use RUN_ID env var or generate new one
    run_id = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/linear_decoders") / subject / f"run_{run_id}"
    models_dir = output_dir / "models"
    embeddings_dir = output_dir / "embeddings"
    
    # Load models from run directory
    image_model_path = models_dir / "fmri_to_clip_image.pkl"
    text_model_path = models_dir / "fmri_to_clip_text.pkl"
    norm_stats_image_path = models_dir / "fmri_norm_stats_image.npz"
    norm_stats_text_path = models_dir / "fmri_norm_stats_text.npz"
    
    # Save embeddings to run directory
    image_pred_path = embeddings_dir / "image.npy"
    text_pred_path = embeddings_dir / "text.npy"
    trial_indices_output_path = embeddings_dir / "trial_indices.npy"
    results_path = output_dir / "results.json"
    
    # Check that input files exist
    for path in [betas_path, trial_indices_path, image_model_path, text_model_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    
    # Load fMRI data
    print("\nLoading fMRI data...")
    # masked_betas is stored as raw binary float32 (not standard .npy format)
    # Load using fromfile and reshape using trial_indices
    betas_raw = np.fromfile(betas_path, dtype=np.float32)
    trial_indices = np.load(trial_indices_path)
    
    # Determine number of voxels from raw file size
    n_voxels = betas_raw.shape[0] // len(trial_indices)
    masked_betas = betas_raw.reshape(len(trial_indices), n_voxels)
    
    print(f"Masked betas shape: {masked_betas.shape}")
    print(f"Trial indices shape: {trial_indices.shape}")
    
    # Load trained models
    print("\nLoading trained ridge models...")
    ridge_image = joblib.load(image_model_path)
    ridge_text = joblib.load(text_model_path)
    
    # Load normalization statistics
    print("Loading normalization statistics...")
    norm_stats_image = np.load(norm_stats_image_path)
    norm_stats_text = np.load(norm_stats_text_path)
    
    X_mean_image = norm_stats_image['mean']
    X_std_image = norm_stats_image['std']
    nonzero_voxels_image = norm_stats_image['nonzero_voxels']
    
    X_mean_text = norm_stats_text['mean']
    X_std_text = norm_stats_text['std']
    nonzero_voxels_text = norm_stats_text['nonzero_voxels']
    
    # Apply voxel filtering and normalization for image model
    print("\nPreprocessing fMRI data for image predictions...")
    X_image = masked_betas[:, nonzero_voxels_image]
    X_image_z = (X_image - X_mean_image) / X_std_image
    
    # Apply voxel filtering and normalization for text model
    print("Preprocessing fMRI data for text predictions...")
    X_text = masked_betas[:, nonzero_voxels_text]
    X_text_z = (X_text - X_mean_text) / X_std_text
    
    print(f"X_image_z shape: {X_image_z.shape}")
    print(f"X_text_z shape: {X_text_z.shape}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    Y_image_pred = ridge_image.predict(X_image_z)
    Y_text_pred = ridge_text.predict(X_text_z)
    
    print(f"Raw image predictions shape: {Y_image_pred.shape}")
    print(f"Raw text predictions shape: {Y_text_pred.shape}")
    
    # Normalize to unit norm
    print("\nNormalizing embeddings to unit norm...")
    Y_image_pred_norm = normalize(Y_image_pred, norm='l2')
    Y_text_pred_norm = normalize(Y_text_pred, norm='l2')
    
    # Lightweight diagnostics
    print("\n=== Diagnostics ===")
    print(f"Predicted image embeddings shape: {Y_image_pred_norm.shape}")
    print(f"Predicted text embeddings shape: {Y_text_pred_norm.shape}")
    
    # Mean L2 norm before normalization
    image_l2_norm_mean = np.mean(np.linalg.norm(Y_image_pred, axis=1))
    text_l2_norm_mean = np.mean(np.linalg.norm(Y_text_pred, axis=1))
    print(f"Mean L2 norm (image, pre-norm): {image_l2_norm_mean:.6f}")
    print(f"Mean L2 norm (text, pre-norm): {text_l2_norm_mean:.6f}")
    
    # Cosine similarity between predicted image and text embeddings
    # cosine_sim_cross = cosine_similarity(Y_image_pred_norm, Y_text_pred_norm)
    # mean_cross_cosine = np.mean(np.diag(cosine_sim_cross))
    mean_cross_cosine = np.mean(np.sum(Y_image_pred_norm * Y_text_pred_norm, axis=1))

    print(f"Mean cosine similarity (image vs text): {mean_cross_cosine:.6f}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    print(f"\nSaving predictions...")
    np.save(image_pred_path, Y_image_pred_norm)
    print(f"Saved image embeddings to {image_pred_path}")
    
    np.save(text_pred_path, Y_text_pred_norm)
    print(f"Saved text embeddings to {text_pred_path}")

    np.save(trial_indices_output_path, trial_indices)
    print(f"Saved trial indices to {trial_indices_output_path}")
    
    # Prepare and save prediction metrics
    print(f"Saving results...")
    prediction_metrics = {
        "timestamp": datetime.now().isoformat(),
        "mean_cross_cosine": float(mean_cross_cosine),
        "num_samples": int(Y_image_pred_norm.shape[0]),
        "embedding_dim": int(Y_image_pred_norm.shape[1]),
    }
    
    # Load existing results.json if present, otherwise create new
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    # Add or update prediction metrics
    results["prediction"] = prediction_metrics

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict CLIP embeddings from fMRI using trained ridge models"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="subj01",
        help="Subject identifier (default: subj01)"
    )
    
    args = parser.parse_args()
    main(subject=args.subject)

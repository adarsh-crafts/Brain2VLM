"""
Train ridge regression from fMRI to CLIP text embeddings.

Fits a ridge regression model mapping image-level fMRI betas to CLIP text
embeddings with validation-based alpha selection.

Usage:
    python train_ridge_fmri_to_clip_text.py
"""

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def cosine_similarity_score(y_true, y_pred):
    """
    Compute mean diagonal cosine similarity between predicted and ground truth.
    
    Args:
        y_true: (N, D) array
        y_pred: (N, D) array
    
    Returns:
        Mean cosine similarity along diagonal
    """
    sim = cosine_similarity(y_pred, y_true)
    diag_sim = np.diag(sim)
    return diag_sim.mean()


def retrieval_at_k(sim, k=5):
    """
    Compute retrieval@k accuracy.
    
    For each prediction, check whether the true item appears in top-k
    most similar targets.
    
    Args:
        sim: (N, N) cosine similarity matrix (predictions × targets)
        k: retrieval cutoff
    
    Returns:
        Fraction of samples whose true match is in top-k
    """
    ranks = np.argsort(-sim, axis=1)
    correct = np.arange(sim.shape[0])[:, None]
    return np.mean(np.any(ranks[:, :k] == correct, axis=1))


def main():
    """
    Train ridge regression mapping averaged fMRI betas to CLIP text embeddings.
    
    Performs alpha cross-validation on a validation set derived from training data.
    """
    print("Training fMRI → CLIP text decoder")
    
    # Define paths
    betas_path = Path("data/processed/fmri/subj01_image_level_betas.npy")
    clip_text_path = Path("data/processed/clip/subj01_text_embeddings.npy")
    splits_path = Path("data/processed/splits/subj01_image_splits.npz")
    model_path = Path("models/linear_decoders/subj01_fmri_to_clip_text.pkl")
    norm_stats_path = Path("models/linear_decoders/subj01_fmri_norm_stats_text.npz")
    results_path = Path("results/linear_decoders/subj01_clip_text_metrics.json")
    
    # Check that input files exist
    for path in [betas_path, clip_text_path, splits_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    
    # Load data
    print("Loading data...")
    X = np.load(betas_path)
    Y = np.load(clip_text_path)
    splits = np.load(splits_path)
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")
    
    # Split data
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    
    # Remove zero-variance voxels
    print("\nRemoving zero-variance voxels...")
    voxel_var = np.var(X_train, axis=0)
    nonzero_voxels = voxel_var > 0
    n_removed = np.sum(~nonzero_voxels)
    print(f"Removed {n_removed} zero-variance voxels")
    
    X_train = X_train[:, nonzero_voxels]
    X_test = X_test[:, nonzero_voxels]
    
    print(f"X_train shape after filtering: {X_train.shape}")
    print(f"X_test shape after filtering: {X_test.shape}")
    
    # Z-score X using training set statistics
    print("\nZ-scoring X using training set statistics...")
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    
    X_train_z = (X_train - X_mean) / X_std
    X_test_z = (X_test - X_mean) / X_std
    
    print(f"X_train_z mean: {X_train_z.mean():.6f}, std: {X_train_z.std():.6f}")
    
    # Do NOT normalize Y before regression (normalize only for metrics)
    print("\nUsing raw CLIP embeddings for regression...")
    Y_train_norm = Y_train
    Y_test_norm = Y_test

    # Create validation set from training data (10%, deterministic)
    print("\nCreating validation set (10% of training data)...")
    rng = np.random.default_rng(0)
    n_train = len(X_train_z)
    n_val = int(0.1 * n_train)
    val_idx = rng.choice(n_train, n_val, replace=False)
    train_only_idx = np.setdiff1d(np.arange(n_train), val_idx)
    
    X_train_only = X_train_z[train_only_idx]
    Y_train_only = Y_train_norm[train_only_idx]
    X_val = X_train_z[val_idx]
    Y_val = Y_train_norm[val_idx]
    
    print(f"X_train_only: {X_train_only.shape}, X_val: {X_val.shape}")
    
    # Ridge regression with alpha sweep
    print("\nPerforming alpha sweep with validation set...")
    alphas = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    best_alpha = None
    best_score = -np.inf
    val_scores = {}
    
    for alpha in alphas:
        print(f"  Testing alpha={alpha}...")
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_train_only, Y_train_only)
        
        Y_val_pred = ridge.predict(X_val)
        Y_val_pred_norm = normalize(Y_val_pred, norm="l2")
        score = cosine_similarity_score(Y_val, Y_val_pred_norm)

        val_scores[str(alpha)] = float(score)
        print(f"    Validation cosine similarity: {score:.6f}")
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} (validation score: {best_score:.6f})")
    
    # Retrain ridge on full training set with best alpha
    print("\nRetraining ridge on full training set with best alpha...")
    ridge_final = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge_final.fit(X_train_z, Y_train_norm)
    
    # Predict on test set
    print("Predicting on test set...")
    Y_test_pred = ridge_final.predict(X_test_z)
    Y_test_pred_norm = normalize(Y_test_pred, norm="l2")
    
    # Compute metrics
    print("\nComputing metrics...")
    sim_test = cosine_similarity(
        Y_test_pred_norm,
        normalize(Y_test, norm="l2")
    )

    test_cosine = np.mean(np.diag(sim_test))
    r1 = retrieval_at_k(sim_test, k=1)
    r5 = retrieval_at_k(sim_test, k=5)
    r10 = retrieval_at_k(sim_test, k=10)

        
    print(f"Test cosine similarity: {test_cosine:.6f}")
    print(f"Retrieval@1: {r1:.6f}")
    print(f"Retrieval@5: {r5:.6f}")
    print(f"Retrieval@10: {r10:.6f}")
    
    # Create output directories if they don't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    joblib.dump(ridge_final, model_path)
    
    # Save normalization statistics
    print(f"Saving normalization stats to {norm_stats_path}...")
    np.savez(norm_stats_path, mean=X_mean, std=X_std, nonzero_voxels=nonzero_voxels)
    
    # Save metrics
    print(f"Saving metrics to {results_path}...")
    metrics = {
        "best_alpha": float(best_alpha),
        "validation_scores": val_scores,
        "test_cosine_similarity": float(test_cosine),
        "retrieval_at_1": float(r1),
        "retrieval_at_5": float(r5),
        "retrieval_at_10": float(r10),
        "n_voxels": int(X_train.shape[1]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

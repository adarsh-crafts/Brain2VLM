import argparse
import json
import os

import h5py
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def _load_fmri_inputs(processed_dir, subject):
    # Prefer the naming used in this repo; fall back to the requested name if present.
    image_level_path = os.path.join(processed_dir, "fmri", f"{subject}_image_level_betas.npy")
    legacy_path = os.path.join(processed_dir, "fmri", f"{subject}_image_betas.npy")

    if os.path.exists(image_level_path):
        fmri_path = image_level_path
    elif os.path.exists(legacy_path):
        fmri_path = legacy_path
    else:
        raise FileNotFoundError(
            "Missing fMRI image-level betas. Expected one of: "
            f"{image_level_path} or {legacy_path}"
        )

    fmri_data = np.load(fmri_path)
    print("\n==== Loading fMRI ====")
    print(f"Loaded fMRI from: {fmri_path}")
    print(f"fMRI shape: {fmri_data.shape}")
    print(f"fMRI dtype: {fmri_data.dtype}")
    return fmri_data


def _load_y_embeddings(embeddings_path, dataset_key):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

    with h5py.File(embeddings_path, "r") as h5_file:
        if dataset_key not in h5_file:
            raise KeyError(f"Expected dataset '{dataset_key}' in the embeddings file")
        embeddings = h5_file[dataset_key][...]

    print("\n==== Loading Pseudo Text Embeddings ====")
    print(f"Embeddings path: {embeddings_path}")
    print(f"Embeddings dataset key: {dataset_key}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings min: {float(np.min(embeddings))}")
    print(f"Embeddings max: {float(np.max(embeddings))}")
    print(f"Embeddings mean: {float(np.mean(embeddings))}")

    return embeddings


def _load_splits(processed_dir, subject):
    splits_path = os.path.join(processed_dir, "splits", f"{subject}_image_splits.npz")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Missing splits file: {splits_path}")

    splits = np.load(splits_path)
    if "train_idx" not in splits or "test_idx" not in splits:
        raise KeyError("Splits file must contain train_idx and test_idx")

    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]
    print("\n==== Loading Splits ====")
    print(f"Splits path: {splits_path}")
    print(f"Train size: {len(train_idx)}")
    print(f"Test size: {len(test_idx)}")
    print(f"Train idx (first 5): {train_idx[:5]}")
    print(f"Test idx (first 5): {test_idx[:5]}")

    return train_idx, test_idx


def _evaluate_ridge(model, x_data, y_data):
    # Predictions are computed in the same space as the model targets.
    predictions = model.predict(x_data)
    mse = mean_squared_error(y_data, predictions)
    r2 = r2_score(y_data, predictions)
    return mse, r2


def _cosine_per_sample(a, b):
    """
    a, b: (N, D)
    Returns mean cosine similarity across samples.
    """
    sims = cosine_similarity(a, b)
    return float(np.mean(np.diag(sims)))


def main():
    parser = argparse.ArgumentParser(description="Train a ridge mapper from fMRI to pseudo text embeddings")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="mappers")
    parser.add_argument("--standardize_y", action="store_true", help="Standardize Y before regression")
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help=(
            "Path to HDF5 embeddings file. Can be absolute or relative to processed_dir. "
            "Defaults to embeddings/pseudo_txt_embeddings.hdf5 under processed_dir."
        ),
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="pseudo_txt",
        help="HDF5 dataset key for target embeddings (default: pseudo_txt)",
    )

    args = parser.parse_args()

    # -------------------------
    # Step 1: load X, Y, splits
    # -------------------------
    fmri_data = _load_fmri_inputs(args.processed_dir, args.subject)
    if args.target_path is None:
        y_embeddings_path = os.path.join(args.processed_dir, "embeddings", "pseudo_txt_embeddings.hdf5")
    elif os.path.isabs(args.target_path):
        y_embeddings_path = args.target_path
    else:
        y_embeddings_path = os.path.join(args.processed_dir, args.target_path)

    embeddings = _load_y_embeddings(y_embeddings_path, args.target_key)

    fmri_data = fmri_data.astype(np.float32)
    embeddings = embeddings.astype(np.float32)

    print("\n==== Casting to float32 ====")
    print(f"X dtype after cast: {fmri_data.dtype}")
    print(f"Y dtype after cast: {embeddings.dtype}")

    train_idx, test_idx = _load_splits(args.processed_dir, args.subject)

    # -------------------------
    # Step 1: confirm shapes
    # -------------------------
    print(f"X.shape: {fmri_data.shape}")
    print(f"Y.shape: {embeddings.shape}")

    if fmri_data.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "X/Y sample count mismatch: "
            f"X has {fmri_data.shape[0]} samples, Y has {embeddings.shape[0]} samples"
        )

    # Flatten Y to 2D so Ridge can do multi-output regression directly.
    original_y_shape = embeddings.shape
    embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
    print("\n==== Flattening Y ====")
    print(f"Original Y shape: {original_y_shape}")
    print(f"Flattened Y shape: {embeddings_2d.shape}")

    # -------------------------
    # Step 3: preprocessing
    # -------------------------
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(fmri_data[train_idx])
    x_test = x_scaler.transform(fmri_data[test_idx])
    print("\n==== Scaling X ====")
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"x_train mean (global): {float(np.mean(x_train))}")
    print(f"x_train std (global): {float(np.std(x_train))}")

    if args.standardize_y:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(embeddings_2d[train_idx])
        y_test = y_scaler.transform(embeddings_2d[test_idx])
        print("\n==== Scaling Y ====")
        print(f"y_train mean (global): {float(np.mean(y_train))}")
        print(f"y_train std (global): {float(np.std(y_train))}")
        print(f"y_test mean (global): {float(np.mean(y_test))}")
        print(f"y_test std (global): {float(np.std(y_test))}")
    else:
        y_scaler = None
        y_train = embeddings_2d[train_idx]
        y_test = embeddings_2d[test_idx]
        print("\n==== Scaling Y ====")
        print("Y not standardized")

    # -------------------------
    # Step 4: ridge sweep
    # -------------------------
    alpha_candidates = [1e3, 1e4, 1e5, 1e6, 1e7]
    best_alpha = None
    best_test_cos = None
    best_test_mse = None
    best_train_mse = None
    best_train_r2 = None
    best_test_r2 = None

    print("\n================ Ridge Sweep ================")
    for alpha in alpha_candidates:
        print(f"Training ridge with alpha = {alpha}")
        ridge_model = Ridge(alpha=alpha, solver="svd")
        ridge_model.fit(x_train, y_train)

        print(f"Coef shape: {ridge_model.coef_.shape}")
        print(f"Coef norm: {float(np.linalg.norm(ridge_model.coef_))}")

        train_mse, train_r2 = _evaluate_ridge(ridge_model, x_train, y_train)
        test_mse, test_r2 = _evaluate_ridge(ridge_model, x_test, y_test)

        train_pred = ridge_model.predict(x_train)
        test_pred = ridge_model.predict(x_test)

        if y_scaler is not None:
            train_pred = y_scaler.inverse_transform(train_pred)
            test_pred = y_scaler.inverse_transform(test_pred)

        train_cos = _cosine_per_sample(train_pred, y_train)
        test_cos = _cosine_per_sample(test_pred, y_test)

        print(
            f"alpha={alpha} | "
            f"train MSE={train_mse:.6f} R2={train_r2:.6f} COS={train_cos:.4f} | "
            f"test MSE={test_mse:.6f} R2={test_r2:.6f} COS={test_cos:.4f}"
        )

        if best_test_cos is None or test_cos > best_test_cos:
            best_test_cos = test_cos
            best_test_mse = test_mse
            best_alpha = alpha
            best_train_mse = train_mse
            best_train_r2 = train_r2
            best_test_r2 = test_r2

    print("\n==== Best Alpha ====")
    print(f"Best alpha selected: {best_alpha}")
    print(f"Best test MSE: {best_test_mse}")
    print(f"Best test R2: {best_test_r2}")
    print(f"Best test COS: {best_test_cos}")
    print(f"Best alpha selected by cosine similarity: {best_alpha}")
    print(f"Best test cosine: {best_test_cos}")

    # -------------------------
    # Step 5: final training
    # -------------------------
    full_x_scaler = StandardScaler()
    full_x = full_x_scaler.fit_transform(fmri_data)

    if args.standardize_y:
        full_y_scaler = StandardScaler()
        full_y = full_y_scaler.fit_transform(embeddings_2d)
    else:
        full_y_scaler = None
        full_y = embeddings_2d

    final_ridge = Ridge(alpha=best_alpha, solver="svd")
    final_ridge.fit(full_x, full_y)

    print("\n==== Final Training ====")
    print(f"Final model coef shape: {final_ridge.coef_.shape}")
    print(f"Final coef norm: {float(np.linalg.norm(final_ridge.coef_))}")

    # -------------------------
    # Step 6: saving
    # -------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    model_path = os.path.join(args.out_dir, "ridge_fmri_to_pseudotxt.joblib")
    x_scaler_path = os.path.join(args.out_dir, "scaler_X.joblib")
    y_scaler_path = os.path.join(args.out_dir, "scaler_Y.joblib")
    metadata_path = os.path.join(args.out_dir, "ridge_metadata.json")

    joblib.dump(final_ridge, model_path)
    joblib.dump(full_x_scaler, x_scaler_path)

    if full_y_scaler is not None:
        joblib.dump(full_y_scaler, y_scaler_path)

    metadata = {
        "alpha": best_alpha,
        "train_MSE": float(best_train_mse),
        "test_MSE": float(best_test_mse),
        "train_R2": float(best_train_r2),
        "test_R2": float(best_test_r2),
        "test_cosine": float(best_test_cos),
        "X_shape": list(fmri_data.shape),
        "Y_shape": list(original_y_shape),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved model to", model_path)
    print("Saved X scaler to", x_scaler_path)
    if full_y_scaler is not None:
        print("Saved Y scaler to", y_scaler_path)
    print("Saved metadata to", metadata_path)

    # -------------------------
    # Step 7: verification
    # -------------------------
    loaded_model = joblib.load(model_path)
    loaded_x_scaler = joblib.load(x_scaler_path)
    loaded_y_scaler = joblib.load(y_scaler_path) if full_y_scaler is not None else None

    first_five = fmri_data[:5]
    first_five_scaled = loaded_x_scaler.transform(first_five)
    predictions = loaded_model.predict(first_five_scaled)

    gt_first_five = embeddings_2d[:5]
    pred_first_five = predictions

    if loaded_y_scaler is not None:
        predictions = loaded_y_scaler.inverse_transform(predictions)
        pred_first_five = loaded_y_scaler.inverse_transform(pred_first_five)

    verify_cos = _cosine_per_sample(pred_first_five, gt_first_five)
    print("verification cosine (first 5):", verify_cos)

    predictions = predictions.reshape((predictions.shape[0],) + original_y_shape[1:])

    print("prediction.shape:", predictions.shape)
    print("prediction mean:", float(np.mean(predictions)))
    print("prediction std:", float(np.std(predictions)))
    print("prediction min:", float(np.min(predictions)))
    print("prediction max:", float(np.max(predictions)))

    if np.isnan(predictions).any():
        raise ValueError("NaNs found in predictions")

    # -------------------------
    # Step 8: save predicted embeddings for decoder.py
    # -------------------------
    pred_h5_path = os.path.join(args.out_dir, "predicted_pseudo_txt_embeddings.hdf5")

    with h5py.File(pred_h5_path, "w") as f:
        f.create_dataset("pseudo_txt", data=predictions.astype(np.float32))

    print("Saved predicted embeddings to", pred_h5_path)

    with h5py.File(pred_h5_path, "r") as f:
        keys = list(f.keys())
        dataset = f["pseudo_txt"]
        print("\n==== Saved HDF5 Verification ====")
        print(f"Dataset keys: {keys}")
        print(f"Saved shape: {dataset.shape}")
        print(f"Saved dtype: {dataset.dtype}")


if __name__ == "__main__":
    main()

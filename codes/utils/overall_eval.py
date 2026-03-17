"""
overall_eval.py
===============
Evaluate brain-decoded image reconstructions using pre-extracted .npy
features (produced by extract_feats.py).

Directly comparable to brain-diffuser (Ozcelik & VanRullen, 2023):
  scripts/evaluate_reconstruction.py

Alignment changes vs previous version
--------------------------------------
1. PixCorr
   - Brain-diffuser: RGB (not grayscale), normalised to [0,1], flatten, corrcoef
   - GT image also resized to 425×425 (recon already is)
   - Fixed: was using grayscale + pearsonr

2. SSIM
   - Brain-diffuser: rgb2gray → ssim with
       gaussian_weights=True, sigma=1.5,
       use_sample_covariance=False, data_range=1.0
   - Fixed: was using data_range=255 on uint8 grayscale with no gaussian weights

3. Feature metric: pairwise_corr_all
   - Brain-diffuser: builds full N×N corrcoef matrix, reports fraction of
     off-diagonal entries per column that are LOWER than the diagonal
     (i.e. pairwise identification accuracy)
   - We extend this to 5 reconstructions by averaging features across reps
     before building the N×N matrix (matches identification script convention
     of r_true.mean() > r_fake.mean())
   - Fixed: was reporting mean Pearson r, not pairwise identification accuracy

4. CLIP model
   - Brain-diffuser: ViT-L/14 via openai/clip
   - Fixed: was using ViT-B/32

Metric table produced
---------------------
  Low-Level  : PixCorr↑  SSIM↑  PSNR↑  LPIPS↓
               AlexNet(5)↑  CLIP(6)↑  DINOv2(6)↑
  High-Level : AlexNet↑  CLIP↑  DINO↑  Inception↑
  Retrieval  : @1↑  @10↑

Usage
-----
  python overall_eval.py --subject subj01 --methods cvpr mlp --gpu 0
"""

import argparse
import os
import traceback

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from scipy.stats import binom

import torch
import torchvision.transforms as T
import lpips

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NIMAGE = 982
NREP   = 5
_EVAL_SIZE = 425   # brain-diffuser resizes reconstructions to 425×425

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",     type=str, default="subj01")
    parser.add_argument("--methods",     nargs="+", default=["cvpr", "mlp"])
    parser.add_argument("--output_csv",  type=str, default="eval_results.csv")
    parser.add_argument("--gpu",         type=int, default=0)
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# PAIRWISE IDENTIFICATION ACCURACY  (exact copy from brain-diffuser)
#
# brain-diffuser/scripts/evaluate_reconstruction.py :: pairwise_corr_all
#
# Builds the full N×N corrcoef matrix between GT features and predicted
# features.  For each prediction i, counts how many GT features j≠i have
# a LOWER correlation than the matching GT_i → fraction correct.
# Returns (mean_accuracy, p_value).
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_corr_all(ground_truth: np.ndarray,
                       predictions: np.ndarray) -> tuple:
    """
    Parameters
    ----------
    ground_truth : (N, D) float array — GT features
    predictions  : (N, D) float array — predicted/reconstructed features

    Returns
    -------
    perf : float  — mean pairwise identification accuracy (0–1)
    p    : float  — one-sided binomial p-value
    """
    r = np.corrcoef(ground_truth, predictions)   # (2N, 2N)
    r = r[:len(ground_truth), len(ground_truth):]  # (N, N) GT-rows × pred-cols

    congruents  = np.diag(r)                      # (N,) diagonal = correct pairs
    success     = r < congruents                  # broadcast: True where off-diag is lower
    success_cnt = np.sum(success, axis=0)         # (N,) per-prediction count

    n   = len(ground_truth)
    perf = np.mean(success_cnt) / (n - 1)
    p    = 1 - binom.cdf(perf * n * (n - 1), n * (n - 1), 0.5)
    return float(perf), float(p)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE LOADERS  (mirrors identification script)
# ─────────────────────────────────────────────────────────────────────────────

def load_feat_org(imgid: int, subject: str, method: str,
                  usefeat: str) -> np.ndarray:
    featdir = f'../../identification/{method}/{subject}'
    return np.load(f'{featdir}/{imgid:05}_org_{usefeat}.npy').flatten()


def load_feat_gen(imgid: int, subject: str, method: str,
                  usefeat: str) -> list:
    featdir = f'../../identification/{method}/{subject}'
    return [
        np.load(f'{featdir}/{imgid:05}_{rep:03}_{usefeat}.npy').flatten()
        for rep in range(NREP)
    ]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE METRIC  (pairwise identification accuracy, brain-diffuser protocol)
#
# For 5 reconstructions per image: average features across reps then
# run pairwise_corr_all.  This matches the identification script's convention
# of using r_true.mean() vs r_fake.mean().
# ─────────────────────────────────────────────────────────────────────────────

def compute_feat_metric(subject: str, method: str, usefeat: str) -> float:
    """
    Returns pairwise identification accuracy (0–1) for the given feature.
    """
    gt_feats   = []
    pred_feats = []

    for imgid in range(NIMAGE):
        gt_feats.append(load_feat_org(imgid, subject, method, usefeat))
        reps = load_feat_gen(imgid, subject, method, usefeat)
        pred_feats.append(np.mean(reps, axis=0))   # mean over 5 reps

    gt_arr   = np.stack(gt_feats,   axis=0).astype(np.float32)   # (N, D)
    pred_arr = np.stack(pred_feats, axis=0).astype(np.float32)   # (N, D)

    perf, _ = pairwise_corr_all(gt_arr, pred_arr)
    return perf

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL @1 / @10
# Full N×N cosine similarity, mean-pooled reconstructions vs GT.
# ─────────────────────────────────────────────────────────────────────────────

def compute_retrieval(subject: str, method: str,
                      usefeat: str = "clip") -> tuple:
    gt_feats   = []
    pred_feats = []

    print(f"    Loading {usefeat} features for retrieval …")
    for imgid in tqdm(range(NIMAGE), desc="    Loading feats"):
        gt_feats.append(load_feat_org(imgid, subject, method, usefeat))
        reps = load_feat_gen(imgid, subject, method, usefeat)
        pred_feats.append(np.mean(reps, axis=0))

    gt_arr   = np.stack(gt_feats,   axis=0).astype(np.float32)
    pred_arr = np.stack(pred_feats, axis=0).astype(np.float32)

    def _l2(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    sim = _l2(pred_arr) @ _l2(gt_arr).T   # (N, N)

    at1 = at10 = 0
    for i in range(NIMAGE):
        rank = int(np.where(np.argsort(sim[i])[::-1] == i)[0][0])
        at1  += int(rank == 0)
        at10 += int(rank <  10)

    return at1 / NIMAGE, at10 / NIMAGE

# ─────────────────────────────────────────────────────────────────────────────
# PIXEL-LEVEL METRICS  (aligned to brain-diffuser exactly)
# ─────────────────────────────────────────────────────────────────────────────

_lpips_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def compute_pixcorr(img_gt: Image.Image, img_pred: Image.Image) -> float:
    """
    Pearson correlation on flattened RGB pixels in [0, 1].

    Brain-diffuser protocol (evaluate_reconstruction.py line 82-83):
      gen_image = np.array(gen_image) / 255.0   (resized to 425×425)
      gt_image  = np.array(gt_image)  / 255.0   (original NSD size, ~425×425)
      pixcorr   = np.corrcoef(gt.reshape(1,-1), gen.reshape(1,-1))[0,1]

    Key difference from previous version:
      - RGB (not grayscale)
      - normalised to [0,1] (not uint8)
    """
    gt  = np.array(img_gt.resize((_EVAL_SIZE, _EVAL_SIZE),
                                  Image.LANCZOS)) / 255.0
    pr  = np.array(img_pred.resize((_EVAL_SIZE, _EVAL_SIZE),
                                   Image.LANCZOS)) / 255.0
    return float(np.corrcoef(gt.reshape(1, -1), pr.reshape(1, -1))[0, 1])


def compute_ssim(img_gt: Image.Image, img_pred: Image.Image) -> float:
    """
    SSIM on grayscale images.

    Brain-diffuser protocol (evaluate_reconstruction.py line 84-87):
      gen_image = rgb2gray(gen_image)   (already [0,1] float)
      gt_image  = rgb2gray(gt_image)
      ssim(..., multichannel=True,
               gaussian_weights=True, sigma=1.5,
               use_sample_covariance=False, data_range=1.0)

    Key differences from previous version:
      - data_range=1.0 (not 255)
      - gaussian_weights=True, sigma=1.5, use_sample_covariance=False
    """
    gt = rgb2gray(np.array(img_gt.resize((_EVAL_SIZE, _EVAL_SIZE),
                                          Image.LANCZOS)) / 255.0)
    pr = rgb2gray(np.array(img_pred.resize((_EVAL_SIZE, _EVAL_SIZE),
                                            Image.LANCZOS)) / 255.0)
    return float(ssim_fn(
        gt, pr,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=1.0,
    ))


def compute_psnr(img_gt: Image.Image, img_pred: Image.Image) -> float:
    """PSNR on RGB [0,1] images at 425×425."""
    gt  = np.array(img_gt.resize((_EVAL_SIZE, _EVAL_SIZE),
                                  Image.LANCZOS), dtype=np.float32) / 255.0
    pr  = np.array(img_pred.resize((_EVAL_SIZE, _EVAL_SIZE),
                                   Image.LANCZOS), dtype=np.float32) / 255.0
    mse = np.mean((gt - pr) ** 2)
    return float(20.0 * np.log10(1.0 / np.sqrt(mse))) if mse > 0 else float("inf")


@torch.no_grad()
def compute_lpips(lpips_model, img_gt: Image.Image, img_pred: Image.Image,
                  device) -> float:
    t_gt   = _lpips_tf(img_gt).unsqueeze(0).to(device)
    t_pred = _lpips_tf(img_pred).unsqueeze(0).to(device)
    return float(lpips_model(t_gt, t_pred).item())


def best_rep_by_clip(imgid: int, subject: str, method: str) -> int:
    """Pick the reconstruction rep with highest Pearson r to GT via CLIP feat."""
    gt = load_feat_org(imgid, subject, method, "clip")
    rs = [
        float(np.corrcoef(gt, r)[0, 1])
        for r in load_feat_gen(imgid, subject, method, "clip")
    ]
    return int(np.argmax(rs))


def compute_pixel_metrics(subject: str, method: str,
                           lpips_model, device) -> dict:
    samples_dir = f'../../decoded/image-{method}/{subject}/samples'
    scores = {"PixCorr": [], "SSIM": [], "PSNR": [], "LPIPS": []}

    for imgid in tqdm(range(NIMAGE), desc="    Pixel metrics"):
        org_path = os.path.join(samples_dir, f"{imgid:05}_org.png")
        if not os.path.isfile(org_path):
            continue

        img_gt    = load_pil(org_path)
        best_rep  = best_rep_by_clip(imgid, subject, method)
        img_recon = load_pil(os.path.join(
            samples_dir, f"{imgid:05}_{best_rep:03}.png"))

        scores["PixCorr"].append(compute_pixcorr(img_gt, img_recon))
        scores["SSIM"].append(compute_ssim(img_gt, img_recon))
        scores["PSNR"].append(compute_psnr(img_gt, img_recon))
        scores["LPIPS"].append(compute_lpips(lpips_model, img_gt, img_recon, device))

    return {k: float(np.mean(v)) for k, v in scores.items()}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE → TABLE METRIC MAPPING
# Maps table metric name → saved .npy feature suffix (from extract_feats.py)
# ─────────────────────────────────────────────────────────────────────────────

FEAT_METRICS = {
    # Low-level feature metrics
    "AlexNet(5)":  "alexnet5",    # AlexNet features[11]   ← matches brain-diffuser layer-5
    "CLIP(6)":     "clip_h6",     # CLIP ViT-L/14 block-6 CLS
    "DINOv2(6)":   "dino_h6",     # DINOv2 ViT-B/14 block-6 CLS
    # High-level feature metrics
    "AlexNet":     "alexnet18",   # AlexNet classifier[5] FC (4096-d)
    "CLIP":        "clip",        # CLIP ViT-L/14 final embed  ← matches brain-diffuser
    "DINO":        "dino",        # DINOv2 ViT-B/14 final CLS
    "Inception":   "inception",   # InceptionV3 avgpool        ← matches brain-diffuser
}

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-METHOD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_method(method: str, subject: str,
                    lpips_model, device) -> dict:
    print(f"\n{'='*65}")
    print(f"  Method : {method}   Subject : {subject}")
    print(f"{'='*65}")

    results = {}

    # ── Pixel-level ─────────────────────────────────────────────────────────
    print("  [1/3] Pixel-level metrics (PixCorr, SSIM, PSNR, LPIPS) …")
    results.update(compute_pixel_metrics(subject, method, lpips_model, device))

    # ── Feature-based (pairwise identification accuracy) ────────────────────
    print("  [2/3] Feature-based metrics (pairwise identification accuracy) …")
    for metric_name, feat_suffix in FEAT_METRICS.items():
        print(f"    {metric_name} ({feat_suffix}) …")
        results[metric_name] = compute_feat_metric(subject, method, feat_suffix)

    # ── Retrieval @1 / @10 ──────────────────────────────────────────────────
    print("  [3/3] Retrieval @1 / @10 (CLIP features) …")
    at1, at10 = compute_retrieval(subject, method, usefeat="clip")
    results["@1"]  = at1
    results["@10"] = at10

    return results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice : {device}")
    print(f"Subject: {args.subject}")
    print(f"Methods: {args.methods}")

    print("\nLoading LPIPS model …")
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    all_results: dict = {}
    for method in args.methods:
        try:
            all_results[method] = evaluate_method(
                method, args.subject, lpips_model, device)
        except Exception as exc:
            print(f"\n[ERROR] method={method}: {exc}")
            traceback.print_exc()

    # ── Results table ────────────────────────────────────────────────────────
    metric_order = [
        "PixCorr", "SSIM", "PSNR", "LPIPS",          # Low-level pixel
        "AlexNet(5)", "CLIP(6)", "DINOv2(6)",          # Low-level feature
        "AlexNet", "CLIP", "DINO", "Inception",        # High-level
        "@1", "@10",                                   # Retrieval
    ]

    rows = []
    for metric in metric_order:
        row = {"metric": metric}
        for method in args.methods:
            if method in all_results:
                row[method] = round(all_results[method].get(metric, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["metric"] + args.methods)
    df.to_csv(args.output_csv, index=False)

    print(f"\n{'='*65}")
    print(f"Results saved → {args.output_csv}")
    print()
    print(df.to_string(index=False))
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
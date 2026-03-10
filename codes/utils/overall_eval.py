"""
overall_eval.py
===============
CVPR-protocol evaluation of brain-decoded image reconstructions.

Methods  : cvpr, mlp
Subject  : subj01
Protocol : standard NSD brain-decoding evaluation suite

Metrics
-------
  Low-level fidelity   : PixCorr, SSIM
  Perceptual similarity: LPIPS (AlexNet backbone)
  Feature-space        : CLIP-cosine, Inception-cosine, SwAV-cosine,
                         AlexNet-L2-cosine, AlexNet-L5-cosine
  Semantic correctness : 50-way identification accuracy (best-of-N, CLIP)

50-way protocol
---------------
  Distractors sampled from full NSD 73k pool (excluding shared test images).
  For each test image, 49 distractors are drawn randomly (N_TRIALS times).
  A trial is correct if ANY of the 5 reconstructions ranks the GT image #1
  (best-of-N / oracle identification, matching the diffusion sampling setup).
  Final accuracy = mean over N_TRIALS trials.

Output
------
  eval_results.csv  — one row per metric, one column per method

Usage
-----
  python overall_eval.py [--subject subj01] [--n_trials 100] [--n_way 50]
                         [--cache_clip_feats]
"""

import os
import glob
import random
import warnings
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_fn
import lpips
import clip
import h5py
import scipy.io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",      type=str, default="subj01")
    parser.add_argument("--methods",      nargs="+", default=["cvpr", "mlp"])
    parser.add_argument("--n_way",        type=int, default=50)
    parser.add_argument("--n_trials",     type=int, default=100)
    parser.add_argument("--output_csv",   type=str, default="eval_results.csv")
    parser.add_argument("--sample_root",  type=str,
                        default="./decoded/image-{method}/{subject}/samples")
    parser.add_argument("--nsd_root",     type=str, default="../../nsd")
    # Optional: cache CLIP features for the 73k pool to disk
    parser.add_argument("--cache_clip_feats", action="store_true",
                        help="Cache/load CLIP features for 73k pool to speed up reruns")
    parser.add_argument("--clip_cache_path", type=str,
                        default="./nsd_clip_feats_73k.npy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def get_test_items(sample_dir: str):
    """
    Scan sample_dir for *_org.png files and pair them with their
    corresponding _000.png … _004.png reconstructions.

    Returns list of (imgidx: int, org_path: str, recon_paths: list[str]).
    """
    org_paths = sorted(glob.glob(os.path.join(sample_dir, "*_org.png")))
    if not org_paths:
        raise FileNotFoundError(f"No *_org.png files found in: {sample_dir}")

    items = []
    for org_path in org_paths:
        base = os.path.basename(org_path).replace("_org.png", "")
        recon_paths = sorted(
            glob.glob(os.path.join(sample_dir, f"{base}_[0-9][0-9][0-9].png"))
        )
        if not recon_paths:
            print(f"  [warn] No reconstructions found for {base}, skipping.")
            continue
        items.append((int(base), org_path, recon_paths))

    return items

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_lpips_model(device):
    """LPIPS with AlexNet backbone (standard in brain-decoding literature)."""
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def build_clip_model(device):
    """OpenAI CLIP ViT-B/32."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def build_inception_model(device):
    """
    InceptionV3 — pool3 (2048-d) feature extractor.
    We attach a forward hook on the avgpool layer to grab 2048-d features
    before the classifier, avoiding the auxiliary-logits complication.
    """
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.aux_logits = False
    model.eval().to(device)

    features = {}

    def hook(module, input, output):
        features["pool3"] = output.flatten(1)

    model.avgpool.register_forward_hook(hook)
    return model, features


def build_swav_model(device):
    """
    SwAV ResNet50 from facebookresearch/swav.
    We remove the projection head and keep only the trunk (avgpool output).
    """
    swav = torch.hub.load("facebookresearch/swav:main", "resnet50", verbose=False)
    # Remove the final fc layer to expose the 2048-d avgpool representation
    trunk = torch.nn.Sequential(*list(swav.children())[:-1])
    trunk.eval().to(device)
    return trunk


def build_alexnet_model(device):
    """
    AlexNet pretrained on ImageNet.
    Layer 2 = ReLU after 2nd conv block  (index 4 in .features)
    Layer 5 = ReLU after 5th conv block  (index 11 in .features)

    AlexNet .features layout:
      0  Conv2d(3,64,11,4,2)
      1  ReLU                <- after conv1
      2  MaxPool2d
      3  Conv2d(64,192,5,1,2)
      4  ReLU                <- after conv2  ← "layer 2"
      5  MaxPool2d
      6  Conv2d(192,384,3,1,1)
      7  ReLU                <- after conv3
      8  Conv2d(384,256,3,1,1)
      9  ReLU                <- after conv4
     10  Conv2d(256,256,3,1,1)
     11  ReLU                <- after conv5  ← "layer 5"
     12  MaxPool2d
    """
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval().to(device)
    return model

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

_imagenet_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_inception_tf = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_lpips_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL METRICS
# ─────────────────────────────────────────────────────────────────────────────

_PIXCORR_SIZE = 425  # standard size used in NSD evaluation papers

def compute_pixcorr(img_gt: Image.Image, img_pred: Image.Image) -> float:
    """
    Pearson correlation on flattened grayscale pixels.
    Images are resized to 425×425 following the NSD evaluation convention.
    """
    gt = np.array(img_gt.convert("L").resize((_PIXCORR_SIZE, _PIXCORR_SIZE),
                                              Image.LANCZOS)).flatten().astype(np.float32)
    pr = np.array(img_pred.convert("L").resize((_PIXCORR_SIZE, _PIXCORR_SIZE),
                                               Image.LANCZOS)).flatten().astype(np.float32)
    r, _ = pearsonr(gt, pr)
    return float(r)


def compute_ssim(img_gt: Image.Image, img_pred: Image.Image) -> float:
    """
    Structural Similarity Index (SSIM) on grayscale 425×425 images.
    """
    gt = np.array(img_gt.convert("L").resize((_PIXCORR_SIZE, _PIXCORR_SIZE),
                                              Image.LANCZOS))
    pr = np.array(img_pred.convert("L").resize((_PIXCORR_SIZE, _PIXCORR_SIZE),
                                               Image.LANCZOS))
    return float(ssim_fn(gt, pr, data_range=255))

# ─────────────────────────────────────────────────────────────────────────────
# PERCEPTUAL / FEATURE METRICS  (all per-image, return scalar similarity)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_lpips(model, img_gt: Image.Image, img_pred: Image.Image,
                  device) -> float:
    t_gt   = _lpips_tf(img_gt).unsqueeze(0).to(device)
    t_pred = _lpips_tf(img_pred).unsqueeze(0).to(device)
    return float(model(t_gt, t_pred).item())


@torch.no_grad()
def _clip_embed(model, preprocess, img: Image.Image, device) -> torch.Tensor:
    t = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(t).float()
    return F.normalize(feat, dim=-1)


def compute_clip_similarity(model, preprocess, img_gt, img_pred, device) -> float:
    f1 = _clip_embed(model, preprocess, img_gt,   device)
    f2 = _clip_embed(model, preprocess, img_pred, device)
    return float((f1 * f2).sum().item())


@torch.no_grad()
def compute_inception_similarity(model, feat_dict,
                                  img_gt, img_pred, device) -> float:
    def _embed(img):
        t = _inception_tf(img).unsqueeze(0).to(device)
        model(t)          # forward pass triggers hook
        return F.normalize(feat_dict["pool3"].clone(), dim=-1)

    f1 = _embed(img_gt)
    f2 = _embed(img_pred)
    return float((f1 * f2).sum().item())


@torch.no_grad()
def compute_swav_similarity(trunk, img_gt, img_pred, device) -> float:
    def _embed(img):
        t = _imagenet_tf(img).unsqueeze(0).to(device)
        feat = trunk(t).flatten(1)
        return F.normalize(feat, dim=-1)

    f1 = _embed(img_gt)
    f2 = _embed(img_pred)
    return float((f1 * f2).sum().item())


@torch.no_grad()
def compute_alexnet_similarity(model, img_gt, img_pred,
                                device, layer: int = 5) -> float:
    """
    layer=2 → features up to ReLU after conv2 (index 4, inclusive)
    layer=5 → features up to ReLU after conv5 (index 11, inclusive)
    """
    layer_end = {2: 5, 5: 12}   # slice end (exclusive) → up to & including relu
    submodel = model.features[:layer_end[layer]]

    def _embed(img):
        t = _imagenet_tf(img).unsqueeze(0).to(device)
        feat = submodel(t).flatten(1)
        return F.normalize(feat, dim=-1)

    f1 = _embed(img_gt)
    f2 = _embed(img_pred)
    return float((f1 * f2).sum().item())

# ─────────────────────────────────────────────────────────────────────────────
# 50-WAY IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _batch_clip_embed(clip_model, preprocess, images, device,
                      batch_size: int = 64) -> torch.Tensor:
    """
    Embed a list of PIL images with CLIP, returning L2-normalised feature matrix
    of shape (N, D) on CPU.
    """
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        feats = clip_model.encode_image(tensors).float()
        all_feats.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(all_feats, dim=0)


def precompute_pool_clip_features(clip_model, preprocess, sdataset,
                                   pool_indices, device,
                                   cache_path: str = None,
                                   batch_size: int = 128) -> torch.Tensor:
    """
    Pre-compute CLIP features for every image in pool_indices.
    If cache_path is provided and exists, loads from disk.
    Returns tensor of shape (len(pool_indices), D) on CPU, L2-normalised.
    """
    if cache_path and os.path.isfile(cache_path):
        print(f"  Loading cached pool CLIP features from {cache_path}")
        return torch.tensor(np.load(cache_path))

    print(f"  Computing CLIP features for {len(pool_indices):,} pool images …")
    all_feats = []
    for start in tqdm(range(0, len(pool_indices), batch_size),
                      desc="  Pool CLIP feats"):
        batch_idx = pool_indices[start : start + batch_size]
        imgs = [
            ndarray_to_pil(np.squeeze(sdataset[idx]))
            for idx in batch_idx
        ]
        tensors = torch.stack([preprocess(img) for img in imgs]).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(tensors).float()
        all_feats.append(F.normalize(feats, dim=-1).cpu())

    pool_feats = torch.cat(all_feats, dim=0)   # (N_pool, D)

    if cache_path:
        np.save(cache_path, pool_feats.numpy())
        print(f"  Saved pool CLIP features to {cache_path}")

    return pool_feats


def compute_50way_accuracy(clip_model, preprocess,
                            gt_images, recon_lists,
                            distractor_pool_indices, pool_feats,
                            device,
                            n_way: int = 50,
                            n_trials: int = 100,
                            rng: np.random.RandomState = None) -> tuple:
    """
    50-way identification accuracy with best-of-N protocol.

    For each test image i across n_trials:
      1. Sample (n_way-1) distractor indices from distractor_pool_indices.
      2. Retrieve their pre-computed CLIP features.
      3. Stack [gt_feat_i, distractor_feats] → candidate matrix (n_way, D).
      4. For each reconstruction r (up to 5):
           sim_r = recon_feat_r @ candidates.T  → (n_way,)
      5. Correct if argmax of ANY sim_r == 0 (i.e., gt is top-ranked).

    Returns (mean_accuracy, std_accuracy).
    """
    n_test = len(gt_images)
    print(f"\n  Pre-computing CLIP features for {n_test} GT images …")
    gt_feats = _batch_clip_embed(clip_model, preprocess, gt_images, device)  # (N_test, D)

    print(f"  Pre-computing CLIP features for reconstructions …")
    recon_feats_list = []   # list of (n_recons, D) tensors
    for recon_imgs in tqdm(recon_lists, desc="  Recon CLIP feats"):
        feats = _batch_clip_embed(clip_model, preprocess, recon_imgs, device)
        recon_feats_list.append(feats)   # (n_recons, D)

    if rng is None:
        rng = np.random.RandomState(42)

    # pool_feats[j] corresponds to distractor_pool_indices[j]
    # We need to quickly look up features by position in the pool list.
    pool_index_to_pos = {idx: pos for pos, idx in enumerate(distractor_pool_indices)}

    trial_accs = []
    print(f"\n  Running {n_trials} × {n_way}-way identification trials …")
    for _trial in tqdm(range(n_trials), desc="  50-way trials"):
        correct = 0
        for i in range(n_test):
            # Sample n_way-1 distractor positions in the pool
            dist_positions = rng.choice(len(distractor_pool_indices),
                                        size=n_way - 1, replace=False)
            dist_feats = pool_feats[dist_positions]   # (n_way-1, D)

            # Candidate matrix: row 0 = GT, rows 1..(n_way-1) = distractors
            candidates = torch.cat([gt_feats[i:i+1], dist_feats], dim=0)  # (n_way, D)

            # Reconstruction features for this test image
            recon_feats = recon_feats_list[i]   # (n_recons, D)

            # Similarity of each recon to all candidates
            sims = recon_feats @ candidates.T   # (n_recons, n_way)
            best_idx = sims.argmax(dim=1)       # (n_recons,)

            # Best-of-N: correct if ANY reconstruction identifies GT (index 0)
            if (best_idx == 0).any().item():
                correct += 1

        trial_accs.append(correct / n_test)

    return float(np.mean(trial_accs)), float(np.std(trial_accs))

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-METHOD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _best_recon_by_clip(clip_model, preprocess, img_gt, recons, device):
    """Return the reconstruction with the highest CLIP cosine similarity to GT."""
    sims = [
        compute_clip_similarity(clip_model, preprocess, img_gt, r, device)
        for r in recons
    ]
    return recons[int(np.argmax(sims))], sims


def evaluate_method(method: str, subject: str, args, models_dict: dict,
                    device) -> dict:
    sample_dir = args.sample_root.format(method=method, subject=subject)
    print(f"\n{'='*65}")
    print(f"  Method : {method}   Subject : {subject}")
    print(f"  Dir    : {sample_dir}")
    print(f"{'='*65}")

    items = get_test_items(sample_dir)
    print(f"  Test images found: {len(items)}")

    # Unpack models
    lpips_model    = models_dict["lpips"]
    clip_model     = models_dict["clip"]
    clip_pre       = models_dict["clip_preprocess"]
    inception_model= models_dict["inception"]
    inception_feats= models_dict["inception_feats"]
    swav_trunk     = models_dict["swav"]
    alex_model     = models_dict["alexnet"]
    pool_feats     = models_dict["pool_feats"]
    distractor_pool= models_dict["distractor_pool"]

    scores = {
        "PixCorr":        [],
        "SSIM":           [],
        "LPIPS":          [],
        "CLIP":           [],
        "Inception":      [],
        "SwAV":           [],
        "AlexNet-L2":     [],
        "AlexNet-L5":     [],
    }

    gt_images   = []
    recon_lists = []

    for imgidx, org_path, recon_paths in tqdm(items, desc=f"[{method}] per-image"):
        img_gt  = load_pil(org_path)
        recons  = [load_pil(p) for p in recon_paths]

        # Select best recon by CLIP for single-image metrics
        best_recon, clip_sims = _best_recon_by_clip(
            clip_model, clip_pre, img_gt, recons, device)

        # ── Low-level ──────────────────────────────────────────────────────
        scores["PixCorr"].append(compute_pixcorr(img_gt, best_recon))
        scores["SSIM"].append(compute_ssim(img_gt, best_recon))

        # ── Perceptual ─────────────────────────────────────────────────────
        scores["LPIPS"].append(
            compute_lpips(lpips_model, img_gt, best_recon, device))

        # ── Feature-space ──────────────────────────────────────────────────
        scores["CLIP"].append(float(max(clip_sims)))
        scores["Inception"].append(
            compute_inception_similarity(inception_model, inception_feats,
                                         img_gt, best_recon, device))
        scores["SwAV"].append(
            compute_swav_similarity(swav_trunk, img_gt, best_recon, device))
        scores["AlexNet-L2"].append(
            compute_alexnet_similarity(alex_model, img_gt, best_recon, device, layer=2))
        scores["AlexNet-L5"].append(
            compute_alexnet_similarity(alex_model, img_gt, best_recon, device, layer=5))

        gt_images.append(img_gt)
        recon_lists.append(recons)

    # ── 50-way identification ───────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    acc_mean, acc_std = compute_50way_accuracy(
        clip_model, clip_pre,
        gt_images, recon_lists,
        distractor_pool, pool_feats,
        device,
        n_way=args.n_way,
        n_trials=args.n_trials,
        rng=rng,
    )

    results = {metric: float(np.mean(vals)) for metric, vals in scores.items()}
    results[f"{args.n_way}-way-Acc"]     = acc_mean
    results[f"{args.n_way}-way-Acc-std"] = acc_std
    return results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Subject: {args.subject}")
    print(f"Methods: {args.methods}")
    print(f"N-way  : {args.n_way}  |  Trials: {args.n_trials}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading models …")

    print("  [1/5] LPIPS …")
    lpips_model = build_lpips_model(device)

    print("  [2/5] CLIP ViT-B/32 …")
    clip_model, clip_pre = build_clip_model(device)

    print("  [3/5] InceptionV3 …")
    inception_model, inception_feats = build_inception_model(device)

    print("  [4/5] SwAV ResNet50 …")
    swav_trunk = build_swav_model(device)

    print("  [5/5] AlexNet …")
    alex_model = build_alexnet_model(device)

    # ── Load NSD stimuli & build distractor pool ──────────────────────────
    nsd_hdf5      = os.path.join(args.nsd_root,
                                  "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    nsd_expdesign = os.path.join(args.nsd_root,
                                  "nsddata/experiments/nsd/nsd_expdesign.mat")

    print(f"\nOpening NSD stimuli: {nsd_hdf5}")
    sf       = h5py.File(nsd_hdf5, "r")
    sdataset = sf.get("imgBrick")

    print("Building distractor pool (excluding shared test set) …")
    expdesign = scipy.io.loadmat(nsd_expdesign)
    sharedix  = set((expdesign["sharedix"] - 1).flatten().tolist())   # 0-indexed
    n_total   = sdataset.shape[0]
    distractor_pool = [i for i in range(n_total) if i not in sharedix]
    print(f"  Pool size: {len(distractor_pool):,}  (from {n_total:,} total, "
          f"{len(sharedix):,} test images excluded)")

    # ── Pre-compute CLIP features for the entire distractor pool ──────────
    cache_path = args.clip_cache_path if args.cache_clip_feats else None
    pool_feats = precompute_pool_clip_features(
        clip_model, clip_pre, sdataset, distractor_pool, device,
        cache_path=cache_path
    )   # (N_pool, D) float32 CPU tensor

    models_dict = dict(
        lpips=lpips_model,
        clip=clip_model,
        clip_preprocess=clip_pre,
        inception=inception_model,
        inception_feats=inception_feats,
        swav=swav_trunk,
        alexnet=alex_model,
        pool_feats=pool_feats,
        distractor_pool=distractor_pool,
        sdataset=sdataset,
    )

    # ── Evaluate each method ─────────────────────────────────────────────
    all_results = {}
    for method in args.methods:
        try:
            all_results[method] = evaluate_method(
                method, args.subject, args, models_dict, device)
        except Exception as exc:
            print(f"\n[ERROR] method={method}: {exc}")
            traceback.print_exc()

    sf.close()

    # ── Build & save results CSV ─────────────────────────────────────────
    # One row per metric, one column per method
    metric_order = [
        "PixCorr", "SSIM", "LPIPS",
        "CLIP", "Inception", "SwAV", "AlexNet-L2", "AlexNet-L5",
        f"{args.n_way}-way-Acc", f"{args.n_way}-way-Acc-std",
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
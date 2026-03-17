"""
extract_feats.py
================
Extract and save per-image features for evaluation.

Aligned to brain-diffuser's eval_extract_features.py so that metrics are
directly comparable:
  - CLIP  : ViT-L/14 via openai/clip  (brain-diffuser uses ViT-L/14, NOT ViT-B/32)
  - AlexNet layer indices: features[4] = layer-2, features[11] = layer-5
               classifier[5] = FC / high-level
  - InceptionV3 avgpool hook
  - All images resized to 224x224 with ImageNet normalisation
  - CLIP normalisation kept separate (different mean/std)

Additional features beyond brain-diffuser (for our table):
  - DINOv2 ViT-B/14 final CLS         → dino
  - DINOv2 ViT-B/14 block-6 CLS       → dino_h6
  - CLIP ViT-L/14 block-6 CLS token   → clip_h6

Storage format (mirrors identification script):
  Per-image .npy files in ../../identification/{method}/{subject}/
    {imgid:05}_org_{feat}.npy          — GT image feature
    {imgid:05}_{rep:03}_{feat}.npy     — reconstruction feature (rep 0-4)

Usage
-----
  python extract_feats.py --gpu 0 --subject subj01 --method mlp
"""

import argparse
import os
import glob

import numpy as np
import torch
import torchvision
import torchvision.models as tvmodels
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import clip   # openai/clip  — ViT-L/14

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",     required=True, type=int)
    parser.add_argument("--subject", required=True, type=str, help="e.g. subj01")
    parser.add_argument("--method",  required=True, type=str, help="e.g. cvpr or mlp")
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# Brain-diffuser resizes everything to 224×224 before feature extraction.
# Two separate normalisation stats for CLIP vs ImageNet models.
# ─────────────────────────────────────────────────────────────────────────────

# Standard ImageNet normalisation (Inception, AlexNet, DINOv2)
_imagenet_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CLIP-specific normalisation (matches brain-diffuser exactly)
_clip_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    opt     = parse_args()
    subject = opt.subject
    method  = opt.method
    device  = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    imglist = sorted(glob.glob(f'../../decoded/image-{method}/{subject}/samples/*'))
    outdir  = f'../../identification/{method}/{subject}/'
    os.makedirs(outdir, exist_ok=True)

    # ── InceptionV3  (brain-diffuser: avgpool hook) ───────────────────────
    print("Loading InceptionV3 …")
    inception_store = {}
    model_inception = tvmodels.inception_v3(pretrained=True)
    model_inception.aux_logits = False
    model_inception.avgpool.register_forward_hook(
        lambda m, inp, out: inception_store.update({"feat": out.cpu()})
    )
    model_inception.eval().to(device)

    # ── AlexNet  (brain-diffuser: features[4]=layer2, features[11]=layer5) ─
    print("Loading AlexNet …")
    alexnet_store = {}
    model_alexnet = tvmodels.alexnet(pretrained=True)
    model_alexnet.features[4].register_forward_hook(
        lambda m, inp, out: alexnet_store.update({"alex2": out.cpu()})
    )
    model_alexnet.features[11].register_forward_hook(
        lambda m, inp, out: alexnet_store.update({"alex5": out.cpu()})
    )
    model_alexnet.classifier[5].register_forward_hook(
        lambda m, inp, out: alexnet_store.update({"alex18": out.cpu()})
    )
    model_alexnet.eval().to(device)

    # ── CLIP ViT-L/14  (brain-diffuser uses ViT-L/14 via openai/clip) ─────
    print("Loading CLIP ViT-L/14 …")
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    clip_visual = clip_model.visual.to(torch.float32)

    # ── DINOv2 ViT-B/14  (our addition for DINOv2(6) and DINO metrics) ────
    print("Loading DINOv2 ViT-B/14 …")
    model_dino = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14", verbose=False
    ).eval().to(device)

    # ── Feature extraction loop ───────────────────────────────────────────
    print(f"\nExtracting features for method={method}, subject={subject} …")

    for img_path in tqdm(imglist):
        imgname = os.path.splitext(os.path.basename(img_path))[0]
        image   = Image.open(img_path).convert("RGB")
        fname   = os.path.join(outdir, imgname)

        batch_imagenet = _imagenet_tf(image).unsqueeze(0).to(device)
        batch_clip     = _clip_tf(image).unsqueeze(0).to(device)

        with torch.no_grad():

            # ── InceptionV3 ──────────────────────────────────────────────
            model_inception(batch_imagenet)   # hook populates inception_store
            feat_inception = inception_store["feat"].numpy().flatten()

            # ── AlexNet ──────────────────────────────────────────────────
            model_alexnet(batch_imagenet)     # hooks populate alexnet_store
            feat_alexnet2  = alexnet_store["alex2"].numpy().flatten()
            feat_alexnet5  = alexnet_store["alex5"].numpy().flatten()
            feat_alexnet18 = alexnet_store["alex18"].numpy().flatten()

            # ── CLIP ViT-L/14 ─────────────────────────────────────────────
            # Final embedding — hook on the full visual encoder
            # (brain-diffuser: net.register_forward_hook(fn) where net=model.visual)
            clip_final_store = {}
            clip_h6_store    = {}

            def _clip_final_hook(m, inp, out):
                clip_final_store["feat"] = out.cpu().float()

            def _clip_h6_hook(m, inp, out):
                # out: (seq_len, batch, dim) — CLS at position 0
                clip_h6_store["feat"] = out[0].cpu().float()

            h_final = clip_visual.register_forward_hook(_clip_final_hook)
            # block index 5 = 6th transformer block (0-indexed) → CLIP(6)
            h_h6    = clip_visual.transformer.resblocks[5].register_forward_hook(_clip_h6_hook)

            clip_visual(batch_clip)

            h_final.remove()
            h_h6.remove()

            feat_clip    = clip_final_store["feat"].numpy().flatten()
            feat_clip_h6 = clip_h6_store["feat"].numpy().flatten()

            # ── DINOv2 ViT-B/14 ──────────────────────────────────────────
            dino_h6_store = {}

            def _dino_h6_hook(m, inp, out):
                # out: (batch, seq_len, dim) — CLS at position 0
                dino_h6_store["feat"] = out[:, 0, :].cpu().float()

            h_dino = model_dino.blocks[5].register_forward_hook(_dino_h6_hook)
            dino_final = model_dino(batch_imagenet)
            h_dino.remove()

            feat_dino    = dino_final.cpu().numpy().flatten()
            feat_dino_h6 = dino_h6_store["feat"].numpy().flatten()

        # ── Save ─────────────────────────────────────────────────────────
        np.save(f"{fname}_inception.npy",  feat_inception)
        np.save(f"{fname}_alexnet2.npy",   feat_alexnet2)
        np.save(f"{fname}_alexnet5.npy",   feat_alexnet5)
        np.save(f"{fname}_alexnet18.npy",  feat_alexnet18)
        np.save(f"{fname}_clip.npy",       feat_clip)
        np.save(f"{fname}_clip_h6.npy",    feat_clip_h6)
        np.save(f"{fname}_dino.npy",       feat_dino)
        np.save(f"{fname}_dino_h6.npy",    feat_dino_h6)

    print(f"\nDone. Features saved to: {outdir}")


if __name__ == "__main__":
    main()
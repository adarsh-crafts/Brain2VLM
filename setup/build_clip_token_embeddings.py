#!/usr/bin/env python3
"""Prepare CLIP token-level text embeddings for NSD images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

SEQ_LEN = 77


def _load_captions(annotation_paths: list[Path]) -> dict[int, list[str]]:
    """Load COCO captions and build image_id -> list[str] mapping."""
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for path in annotation_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            caption = ann["caption"]
            captions_by_image[image_id].append(caption)
    return captions_by_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CLIP token-level text embeddings for NSD images",
    )
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise NotADirectoryError(f"Invalid repo_root directory: {repo_root}")
    image_ids_path = repo_root / "data" / "processed" / "fmri" / "subj01_image_ids.npy"
    annotations_dir = repo_root / "data" / "coco_annotations"
    if args.output_path is None:
        output_path = repo_root / "data" / "processed" / "embeddings" / "subj01_text_token_embeddings.hdf5"
    else:
        output_path = Path(args.output_path)

    annotation_paths = [
        annotations_dir / "captions_train2017.json",
        annotations_dir / "captions_val2017.json",
    ]

    if not image_ids_path.exists():
        raise FileNotFoundError(f"Missing image IDs file: {image_ids_path}")
    for path in annotation_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing annotation file: {path}")

    image_ids = np.load(image_ids_path)
    image_ids = image_ids.astype(np.int64)
    num_images = int(image_ids.shape[0])

    captions_by_image = _load_captions(annotation_paths)

    missing_ids = [int(i) for i in image_ids if int(i) not in captions_by_image]
    if missing_ids:
        missing_preview = ", ".join(str(i) for i in missing_ids[:20])
        raise ValueError(
            "Missing captions for image IDs: "
            f"{missing_preview}{' ...' if len(missing_ids) > 20 else ''}"
        )

    caption_counts = np.array([len(captions_by_image[int(i)]) for i in image_ids], dtype=np.int64)
    total_captions = int(caption_counts.sum())

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_path,
        torch_dtype=torch.float32,
    ).to(device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    text_encoder.eval()

    if tokenizer.model_max_length != SEQ_LEN:
        raise ValueError(
            f"Unexpected tokenizer max length: {tokenizer.model_max_length} (expected {SEQ_LEN})"
        )

    hidden_dim = int(text_encoder.config.hidden_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix not in {".hdf5", ".h5"}:
        raise ValueError("output_path must end with .hdf5 or .h5")
    h5_file = h5py.File(output_path, "w")
    h5_dataset = h5_file.create_dataset(
        "text_token_embeddings", shape=(num_images, SEQ_LEN, hidden_dim), dtype="float32"
    )

    with torch.no_grad():
        for idx, image_id in enumerate(tqdm(image_ids, desc="Encoding captions")):
            captions = captions_by_image[int(image_id)]
            # Tokenized input shape: (num_captions, 77)
            encoded = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(device)
            # Token embeddings shape: (num_captions, 77, hidden_dim)
            token_embeddings = text_encoder(input_ids)[0]
            # Average across captions -> shape: (77, 768)
            mean_embedding = token_embeddings.mean(dim=0)
            mean_embedding_np = mean_embedding.detach().cpu().to(torch.float32).numpy()
            h5_dataset[idx] = mean_embedding_np

    h5_dataset.attrs["seq_len"] = SEQ_LEN
    h5_dataset.attrs["hidden_dim"] = hidden_dim
    h5_dataset.attrs["model"] = args.pretrained_path
    h5_file.close()

    print(f"Number of images: {num_images}")
    print(f"Total captions processed: {total_captions}")
    print(
        "Captions per image - "
        f"min: {caption_counts.min()}, max: {caption_counts.max()}, "
        f"mean: {caption_counts.mean():.2f}"
    )
    print(f"Saved embeddings shape: {(num_images, SEQ_LEN, hidden_dim)}")


if __name__ == "__main__":
    main()

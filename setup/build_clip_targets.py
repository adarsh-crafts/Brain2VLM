import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPImageProcessor

from prompt_clip import CLIPVisionModelWithPrompt


SEQ_LEN = 77
PINV_ATOL = 0.3
CONSTANT_NORM = 27.5


@torch.no_grad()
def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    processed_dir = Path(args.processed_dir)
    image_ids_path = processed_dir / "fmri" / f"{args.subject}_image_ids.npy"
    nsd_ids_path = processed_dir / "mappings" / f"{args.subject}_nsd_image_ids.npy"

    if not image_ids_path.exists():
        raise FileNotFoundError(f"Missing image ids: {image_ids_path}")
    if not nsd_ids_path.exists():
        raise FileNotFoundError(f"Missing NSD ids: {nsd_ids_path}")

    image_ids = np.load(image_ids_path)
    nsd_ids = np.load(nsd_ids_path)
    if len(image_ids) != len(nsd_ids):
        raise ValueError("image_ids and nsd_ids length mismatch")

    stimuli_path = Path(args.nsd_root) / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    if not stimuli_path.exists():
        raise FileNotFoundError(f"NSD stimuli HDF5 not found: {stimuli_path}")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_path,
        torch_dtype=torch.float32,
    ).to(device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    clip_for_inverse_matrix = CLIPModel.from_pretrained(args.clip_path).to(device)
    clip_for_inverse_matrix.eval()
    image_processer = CLIPImageProcessor.from_pretrained(args.clip_path)
    inv_text = torch.linalg.pinv(clip_for_inverse_matrix.text_projection.weight, atol=PINV_ATOL)
    visual_projection = clip_for_inverse_matrix.visual_projection.weight
    if os.path.exists(os.path.join(args.pretrained_path, "clip")):
        clip = CLIPVisionModelWithPrompt.from_pretrained(
            os.path.join(args.pretrained_path, "clip"),
            prompt_length=args.clip_prompt_length,
        ).to(device)
    else:
        clip = clip_for_inverse_matrix.vision_model

    inputs = tokenizer(
        args.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    text_input = inputs.input_ids.to(device)
    text_masks = inputs.attention_mask.to(device)
    text_embeddings = text_encoder(text_input)[0]

    num_images = len(nsd_ids)
    embed_dim = clip_for_inverse_matrix.text_projection.weight.shape[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    h5_file = None
    h5_dataset = None
    if output_path.suffix == ".npy":
        writer = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(num_images, SEQ_LEN, embed_dim)
        )
    elif output_path.suffix == ".hdf5":
        h5_file = h5py.File(output_path, "w")
        h5_dataset = h5_file.create_dataset(
            "pseudo_txt", shape=(num_images, SEQ_LEN, embed_dim), dtype="float32"
        )
    else:
        raise ValueError("output_path must end with .npy or .hdf5")

    with h5py.File(stimuli_path, "r") as stim_f:
        if "imgBrick" not in stim_f:
            raise KeyError("Dataset 'imgBrick' not found in the HDF5 file")
        dataset = stim_f["imgBrick"]

        for idx in tqdm(range(num_images), desc="Encoding"):
            image = Image.fromarray(dataset[int(nsd_ids[idx])]).convert("RGB")

            clip_image = image_processer(image, return_tensors="pt").pixel_values.to(device)

            image_emb = clip(pixel_values=clip_image).pooler_output

            image_emb_proj = image_emb @ visual_projection.T
            image_emb_proj = image_emb_proj @ inv_text.T
            image_emb_proj = image_emb_proj / image_emb_proj.norm(dim=1, keepdim=True)
            image_emb_proj = CONSTANT_NORM * image_emb_proj

            convert_text_embeddings = torch.zeros_like(text_embeddings)
            convert_text_embeddings[:, 0] = text_embeddings[:, 0]
            convert_text_embeddings[:, 1:] = image_emb_proj.unsqueeze(1)

            pseudo_np = convert_text_embeddings.float().cpu().numpy()[0]
            if writer is not None:
                writer[idx] = pseudo_np
            else:
                h5_dataset[idx] = pseudo_np

    if h5_file is not None:
        h5_dataset.attrs["clip_model"] = args.clip_path
        h5_dataset.attrs["constant_norm"] = CONSTANT_NORM
        h5_dataset.attrs["pinv_atol"] = PINV_ATOL
        h5_dataset.attrs["seq_len"] = SEQ_LEN
        h5_file.close()

    print("Saved pseudo text embeddings to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SD-IPC pseudo text embeddings for NSD images")
    parser.add_argument("--nsd_root", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--clip_prompt_length", type=int, default=50)
    parser.add_argument(
        "--clip_path",
        type=str,
        default="./clip-vit-large-patch14/",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    args = parser.parse_args()
    main(args)

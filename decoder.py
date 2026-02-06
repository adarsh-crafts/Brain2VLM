import argparse
import os

import h5py
import numpy as np
import torch
from diffusers import StableDiffusionPipeline


def load_embeddings(embeddings_path):
    if embeddings_path.endswith(".hdf5") or embeddings_path.endswith(".h5"):
        with h5py.File(embeddings_path, "r") as h5_file:
            if "pseudo_txt" not in h5_file:
                raise KeyError("Expected dataset 'pseudo_txt' in the .hdf5 file.")
            embeddings = torch.from_numpy(h5_file["pseudo_txt"][...]).float()
    elif embeddings_path.endswith(".npy"):
        embeddings = torch.from_numpy(np.load(embeddings_path)).float()
    else:
        raise ValueError("Unsupported embeddings file type. Use .hdf5/.h5 or .npy")

    return embeddings


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_path,
        torch_dtype=torch.float32,
    ).to(device)

    embeddings = load_embeddings(args.embeddings_path)

    if args.index < 0 or args.index >= embeddings.shape[0]:
        raise IndexError(f"Index {args.index} out of range for embeddings with shape {tuple(embeddings.shape)}")

    prompt_embeds = embeddings[args.index]
    prompt_embeds = prompt_embeds.unsqueeze(0).to(device)

    torch.manual_seed(args.seed)

    image = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    ).images[0]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.save_prefix}{args.index}.jpg")
    image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./decoded/")
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(args)

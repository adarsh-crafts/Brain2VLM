import os

import h5py
import numpy as np
import torch
from diffusers import StableDiffusionPipeline


def load_embeddings(path):
    if path.endswith(".hdf5") or path.endswith(".h5"):
        with h5py.File(path, "r") as h5_file:
            if "pseudo_txt" not in h5_file:
                raise KeyError("Expected dataset 'pseudo_txt' in the .hdf5 file.")
            embeddings = torch.from_numpy(h5_file["pseudo_txt"][...]).float()
    elif path.endswith(".npy"):
        embeddings = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError("Unsupported embeddings file type. Use .hdf5/.h5 or .npy")

    return embeddings


def init_decoder(pretrained_path, embeddings_path, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    embeddings = load_embeddings(embeddings_path).to(device)

    return pipe, embeddings


@torch.no_grad()
def decode_one(
    pipe,
    embeddings,
    idx,
    seed,
    guidance_scale,
    num_steps,
):
    images = decode_batch(
        pipe=pipe,
        embeddings=embeddings,
        indices=[idx],
        seed=seed,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
    )
    return images[0]

@torch.no_grad()
def decode_batch(
    pipe,
    embeddings,
    indices,
    seed,
    guidance_scale,
    num_steps,
):
    if torch.is_tensor(indices):
        if indices.dim() != 1:
            raise ValueError("indices must be a 1D tensor")
        indices_list = indices.tolist()
    elif isinstance(indices, (list, tuple, np.ndarray)):
        indices_list = list(indices)
    else:
        raise TypeError("indices must be a list, tuple, numpy array, or 1D tensor")

    if len(indices_list) == 0:
        raise ValueError("indices must be non-empty")

    max_index = embeddings.shape[0] - 1
    for idx in indices_list:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError("all indices must be integers")
        if idx < 0 or idx > max_index:
            raise IndexError(f"Index {idx} out of range for embeddings with shape {tuple(embeddings.shape)}")

    indices_tensor = torch.as_tensor(indices_list, device=embeddings.device)
    prompt_embeds = embeddings[indices_tensor]   # [B,77,768]

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    images = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        generator=generator,
    ).images

    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decode embeddings with Stable Diffusion.")

    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("--index", type=int)
    index_group.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./decoded")
    parser.add_argument("--save_prefix", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    pipe, embeddings = init_decoder(
        args.pretrained_path,
        args.embeddings_path,
        args.device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.index is not None:
        image = decode_one(
            pipe=pipe,
            embeddings=embeddings,
            idx=args.index,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
        )

        if args.save_prefix is None:
            save_prefix = f"{args.index}_seed{args.seed}"
        else:
            save_prefix = args.save_prefix

        output_path = os.path.join(args.output_dir, f"{save_prefix}.jpg")
        image.save(output_path)
    else:
        images = decode_batch(
            pipe=pipe,
            embeddings=embeddings,
            indices=args.indices,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
        )

        for idx, image in zip(args.indices, images):
            output_path = os.path.join(args.output_dir, f"{idx}_seed{args.seed}.jpg")
            image.save(output_path)

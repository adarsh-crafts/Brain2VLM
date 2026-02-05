"""
Extract CLIP text embeddings from COCO captions JSON and store them in HDF5.
"""

import argparse
import gc
import json
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.models.clip.encoder import CLIPEmbedder


def parse_args():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Extract CLIP text embeddings from COCO captions JSON"
	)
	parser.add_argument(
		"--captions_json",
		type=str,
		required=True,
		help="Path to COCO captions JSON file",
	)
	parser.add_argument(
		"--output_hdf5",
		type=str,
		required=True,
		help="Path to output HDF5 file",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=64,
		help="Batch size for text processing (default: 64)",
	)
	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Use FP16 precision if CUDA is available",
	)
	parser.add_argument(
		"--start_idx",
		type=int,
		default=0,
		help="Starting index for resume support (default: 0)",
	)
	return parser.parse_args()


def load_coco_captions(captions_json: str) -> Tuple[List[str], List[int], List[int]]:
	"""Load captions, caption ids, and image ids from a COCO captions JSON file."""
	with open(captions_json, "r") as f:
		data = json.load(f)

	annotations = data.get("annotations", [])
	captions: List[str] = []
	caption_ids: List[int] = []
	image_ids: List[int] = []

	for ann in annotations:
		captions.append(ann["caption"])
		caption_ids.append(int(ann["id"]))
		image_ids.append(int(ann["image_id"]))

	return captions, caption_ids, image_ids


def _ensure_dataset(f: h5py.File, name: str, shape: Tuple[int, ...], dtype, chunks: Tuple[int, ...]):
	"""Create or resize a dataset with given shape, dtype, and chunks."""
	shape = tuple(shape)
	maxshape = (None,) + shape[1:]

	if name not in f:
		return f.create_dataset(
			name,
			shape=shape,
			maxshape=maxshape,
			dtype=dtype,
			chunks=chunks,
		)

	ds = f[name]
	if ds.dtype != np.dtype(dtype):
		raise ValueError(f"Existing dataset '{name}' has dtype {ds.dtype}, expected {dtype}")
	if len(shape) > 1 and ds.shape[1:] != shape[1:]:
		raise ValueError(
			f"Existing dataset '{name}' has shape {ds.shape}, expected (*, {shape[1:]})"
		)
	if ds.shape[0] < shape[0]:
		new_shape = list(ds.shape)
		new_shape[0] = shape[0]
		ds.resize(tuple(new_shape))

	return ds


def initialize_output_hdf5(
	output_hdf5: str,
	total_captions: int,
	embedding_dim: int,
	model_name: str,
	source_json: str,
	mode: str = "w",
):
	"""Initialize or validate the output HDF5 file structure."""
	output_path = Path(output_hdf5)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if total_captions <= 0:
		raise ValueError("No captions found in the provided JSON file.")

	emb_chunk = min(256, max(1, total_captions))
	id_chunk = min(1024, max(1, total_captions))

	with h5py.File(output_hdf5, mode) as f:
		_ensure_dataset(
			f,
			name="text_embeddings",
			shape=(total_captions, embedding_dim),
			dtype=np.float32,
			chunks=(emb_chunk, embedding_dim),
		)
		_ensure_dataset(
			f,
			name="caption_ids",
			shape=(total_captions,),
			dtype=np.int64,
			chunks=(id_chunk,),
		)
		_ensure_dataset(
			f,
			name="image_ids",
			shape=(total_captions,),
			dtype=np.int64,
			chunks=(id_chunk,),
		)

		f.attrs["model_name"] = model_name
		f.attrs["embedding_dim"] = embedding_dim
		f.attrs["total_captions"] = total_captions
		f.attrs["source_json"] = str(Path(source_json).resolve())


def extract_text_embeddings(
	captions_json: str,
	output_hdf5: str,
	batch_size: int = 64,
	fp16: bool = False,
	start_idx: int = 0,
):
	"""Extract text embeddings from captions and write them to HDF5."""
	if start_idx < 0:
		raise ValueError("start_idx must be non-negative")

	captions, caption_ids, image_ids = load_coco_captions(captions_json)
	total_captions = len(captions)

	if start_idx >= total_captions:
		print("start_idx is greater than or equal to total captions; nothing to do.")
		return

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	print(f"Total captions: {total_captions}")
	print(f"Output path: {output_hdf5}")

	model_name = "openai/clip-vit-large-patch14"
	embedder = CLIPEmbedder(model_name=model_name, fp16=fp16)
	embedding_dim = embedder.model.config.projection_dim
	assert embedding_dim == 768

	mode = "a" if Path(output_hdf5).exists() else "w"
	initialize_output_hdf5(
		output_hdf5=output_hdf5,
		total_captions=total_captions,
		embedding_dim=embedding_dim,
		model_name=model_name,
		source_json=captions_json,
		mode=mode,
	)

	with h5py.File(output_hdf5, "r+") as out_f:
		embeddings_ds = out_f["text_embeddings"]
		caption_ids_ds = out_f["caption_ids"]
		image_ids_ds = out_f["image_ids"]

		for batch_start in tqdm(
			range(start_idx, total_captions, batch_size),
			desc="Extracting text embeddings",
		):
			batch_end = min(batch_start + batch_size, total_captions)

			batch_captions = captions[batch_start:batch_end]
			batch_caption_ids = caption_ids[batch_start:batch_end]
			batch_image_ids = image_ids[batch_start:batch_end]

			embeddings = embedder.encode_texts(batch_captions, batch_size=len(batch_captions))
			assert embeddings.shape[1] == 768

			embeddings_ds[batch_start:batch_end] = embeddings.astype(np.float32)
			caption_ids_ds[batch_start:batch_end] = np.asarray(batch_caption_ids, dtype=np.int64)
			image_ids_ds[batch_start:batch_end] = np.asarray(batch_image_ids, dtype=np.int64)

			tqdm.write(f"Batch {batch_start}-{batch_end}")
			torch.cuda.empty_cache()
			gc.collect()

	print(f"Embeddings saved to {output_hdf5}")


def main():
	args = parse_args()
	extract_text_embeddings(
		captions_json=args.captions_json,
		output_hdf5=args.output_hdf5,
		batch_size=args.batch_size,
		fp16=args.fp16,
		start_idx=args.start_idx,
	)


if __name__ == "__main__":
	main()

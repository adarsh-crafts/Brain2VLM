"""
Extract CLIP image embeddings from HDF5 file.

Loads images from input HDF5, computes CLIP embeddings, and saves to output HDF5.
Supports resume from checkpoint via start_idx.
"""

import argparse
import h5py
import numpy as np
import torch
import gc
from pathlib import Path
from tqdm import tqdm

from src.models.clip.encoder import CLIPEmbedder
from src.data.stimuli_loader import load_images


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract CLIP image embeddings from HDF5 file"
    )
    parser.add_argument(
        "--input_hdf5",
        type=str,
        required=True,
        help="Path to input HDF5 file containing imgBrick dataset"
    )
    parser.add_argument(
        "--output_hdf5",
        type=str,
        required=True,
        help="Path to output HDF5 file to write embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision for CLIP model"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for resume support (default: 0)"
    )
    return parser.parse_args()


def get_total_images(input_hdf5: str) -> int:
    """Get total number of images in input HDF5."""
    with h5py.File(input_hdf5, 'r') as f:
        if 'imgBrick' not in f:
            raise KeyError("Dataset 'imgBrick' not found in input HDF5")
        return f['imgBrick'].shape[0]


def initialize_output_hdf5(output_hdf5: str, total_images: int, embedding_dim: int, 
                          model_name: str, mode: str = 'w'):
    """Initialize output HDF5 file with datasets and attributes."""
    with h5py.File(output_hdf5, mode) as f:
        # Create datasets if they don't exist
        if 'image_embeddings' not in f:
            f.create_dataset(
                'image_embeddings',
                shape=(total_images, embedding_dim),
                dtype=np.float32,
                chunks=(256, embedding_dim)
            )
        
        if 'indices' not in f:
            f.create_dataset(
                'indices',
                shape=(total_images,),
                dtype=np.int64,
                chunks=(256,)
            )
        
        # Set attributes
        f.attrs['model_name'] = model_name
        f.attrs['embedding_dim'] = embedding_dim
        f.attrs['total_images'] = total_images


def extract_embeddings(input_hdf5: str, output_hdf5: str, batch_size: int = 32,
                      fp16: bool = False, start_idx: int = 0):
    """Extract CLIP embeddings and save to output HDF5."""
    
    # Get device and total images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    total_images = get_total_images(input_hdf5)
    print(f"Total images: {total_images}")
    
    # Initialize CLIP embedder
    model_name = "openai/clip-vit-large-patch14"
    embedder = CLIPEmbedder(model_name=model_name, fp16=fp16)
    embedding_dim = embedder.model.config.projection_dim
    assert embedding_dim == 768
    
    # Check if output file exists and has partial embeddings
    output_path = Path(output_hdf5)
    if output_path.exists():
        # Open in append mode
        print(f"Output file exists, appending from index {start_idx}")
        initialize_output_hdf5(output_hdf5, total_images, embedding_dim, model_name, mode='a')
    else:
        # Create new output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        initialize_output_hdf5(output_hdf5, total_images, embedding_dim, model_name, mode='w')
    
    # Process in batches
    with h5py.File(output_hdf5, 'r+') as out_f:
        embeddings_dataset = out_f['image_embeddings']
        indices_dataset = out_f['indices']
        
        # Process batches
        for batch_start in tqdm(range(start_idx, total_images, batch_size), 
                               desc="Extracting embeddings"):
            batch_end = min(batch_start + batch_size, total_images)
            batch_indices = range(batch_start, batch_end)
            
            # Load images for this batch
            images = load_images(batch_indices, input_hdf5)
            
            # Extract embeddings
            embeddings = embedder.encode_images(images, batch_size=len(images))
            assert embeddings.shape[1] == 768
            
            # Write to output file
            embeddings_dataset[batch_start:batch_end] = embeddings.astype(np.float32)
            indices_dataset[batch_start:batch_end] = np.arange(batch_start, batch_end, dtype=np.int64)
            
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
    
    print(f"Embeddings saved to {output_hdf5}")


def main():
    """Main entry point."""
    args = parse_args()
    
    extract_embeddings(
        input_hdf5=args.input_hdf5,
        output_hdf5=args.output_hdf5,
        batch_size=args.batch_size,
        fp16=args.fp16,
        start_idx=args.start_idx
    )


if __name__ == "__main__":
    main()

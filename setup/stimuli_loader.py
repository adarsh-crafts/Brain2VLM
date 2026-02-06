import h5py
import numpy as np
import os
import csv
import json
from pathlib import Path
from typing import Union, List, Optional, Dict, Any


def _normalize_indices(indices: Union[int, List[int], range]) -> List[int]:
    """
    Normalize input indices to a list of integers.

    Parameters:
    -----------
    indices : int, list of int, or range
        The indices to normalize.

    Returns:
    --------
    List[int]
        List of integer indices.

    Raises:
    -------
    ValueError
        If indices are invalid or range has step != 1.
    """
    if isinstance(indices, int):
        return [indices]
    elif isinstance(indices, range):
        if indices.step != 1:
            raise ValueError("Range step must be 1 for efficient loading.")
        return list(indices)
    elif isinstance(indices, list):
        if not all(isinstance(i, int) for i in indices):
            raise ValueError("All indices must be integers")
        return indices
    else:
        raise ValueError("indices must be int, list of ints, or range")


def load_images(indices: Union[int, List[int], range], hdf5_path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Load stimuli images from an NSD HDF5 file.

    Parameters:
    -----------
    indices : int, list of int, or range
        The indices of the images to load. Can be a single integer, a list of integers,
        or a range object (with step=1).
    hdf5_path : str or os.PathLike
        Path to the HDF5 file containing the 'imgBrick' dataset.

    Returns:
    --------
    np.ndarray
        Array of shape (N, H, W, C) where N is the number of images, H and W are height
        and width, and C is the number of channels (1 for grayscale, 3 for RGB).

    Raises:
    -------
    ValueError
        If indices are invalid, out of range, or if the range has step != 1.
    FileNotFoundError
        If the HDF5 file does not exist.
    KeyError
        If the 'imgBrick' dataset is not found in the file.

    Notes:
    ------
    - Images are assumed to be stored as 3D (H, W, C) or 2D (H, W) arrays.
    - 2D images are treated as grayscale and a channel dimension is added.
    - For efficiency, consecutive indices are loaded in a single slice.
    - Preserves the original order of requested indices, including duplicates.
    """
    indices_list = _normalize_indices(indices)

    # Check for empty list
    if not indices_list:
        return np.array([]).reshape(0, 0, 0, 0)  # Empty array with correct shape hint

    # Get unique indices in order of appearance, and sorted for checking consecutiveness
    unique_indices = list(dict.fromkeys(indices_list))
    sorted_unique = sorted(unique_indices)

    # Check if unique indices are consecutive
    is_consecutive = (len(sorted_unique) > 1 and
                      all(sorted_unique[i] + 1 == sorted_unique[i+1] for i in range(len(sorted_unique)-1)))

    with h5py.File(hdf5_path, 'r') as f:
        if 'imgBrick' not in f:
            raise KeyError("Dataset 'imgBrick' not found in the HDF5 file")
        dataset = f['imgBrick']
        num_images = dataset.shape[0]

        # Validate range
        min_idx, max_idx = min(unique_indices), max(unique_indices)
        if min_idx < 0 or max_idx >= num_images:
            raise ValueError(f"Indices out of range. Valid range: 0 to {num_images-1}")

        if is_consecutive:
            # Load consecutive images in one go
            imgs = dataset[min_idx:max_idx+1]
            processed_imgs = [_process_image(img) for img in imgs]
            # Map from sorted unique to processed_imgs index
            index_map = {idx: i for i, idx in enumerate(sorted_unique)}
            images = [processed_imgs[index_map[idx]] for idx in indices_list]
        else:
            # Load individually in order
            images = []
            for idx in indices_list:
                img = dataset[idx]
                processed_img = _process_image(img)
                images.append(processed_img)

        # Stack
        stacked = np.stack(images, axis=0)
        return stacked


def load_captions(indices: Union[int, List[int], range], captions_path: Union[str, os.PathLike]) -> List[str]:
    """
    Load captions corresponding to stimulus indices from a captions file.

    Parameters:
    -----------
    indices : int, list of int, or range
        The indices of the captions to load.
    captions_path : str or os.PathLike
        Path to the captions file (.csv, .tsv, or .json).

    Returns:
    --------
    List[str]
        List of captions in the same order as the input indices.

    Raises:
    -------
    ValueError
        If indices are invalid, or if a caption is missing for any index.
    FileNotFoundError
        If the captions file does not exist.

    Notes:
    ------
    - For CSV/TSV: Assumes columns 'index' (int) and 'caption' (str).
    - For JSON: Assumes a dict with integer keys and string values.
    """
    indices_list = _normalize_indices(indices)

    path = Path(captions_path)
    ext = path.suffix.lower()

    if ext in ['.csv', '.tsv']:
        delimiter = ',' if ext == '.csv' else '\t'
        captions_dict = {}
        with open(captions_path, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                idx = int(row['index'])
                captions_dict[idx] = row['caption']
    elif ext == '.json':
        with open(captions_path, 'r') as f:
            data = json.load(f)
        captions_dict = {int(k): v for k, v in data.items()}
    else:
        raise ValueError("Unsupported captions file format. Use .csv, .tsv, or .json")

    # Extract captions in order
    captions = []
    for idx in indices_list:
        if idx not in captions_dict:
            raise ValueError(f"No caption found for index {idx}")
        captions.append(captions_dict[idx])

    return captions


def load_stimuli(indices: Union[int, List[int], range],
                 hdf5_path: Union[str, os.PathLike],
                 captions_path: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    """
    Load stimuli images and optionally captions from NSD files.

    Parameters:
    -----------
    indices : int, list of int, or range
        The indices of the stimuli to load.
    hdf5_path : str or os.PathLike
        Path to the HDF5 file containing the 'imgBrick' dataset.
    captions_path : str or os.PathLike, optional
        Path to the captions file. If None, captions are not loaded.

    Returns:
    --------
    Dict[str, Any]
        Dictionary with keys:
        - "indices": List[int] of the requested indices
        - "images": np.ndarray of shape (N, H, W, C)
        - "captions": List[str] or None

    Raises:
    -------
    ValueError, FileNotFoundError, KeyError
        As raised by load_images and load_captions.
    """
    indices_list = _normalize_indices(indices)
    images = load_images(indices_list, hdf5_path)
    captions = None
    if captions_path is not None:
        captions = load_captions(indices_list, captions_path)

    return {
        "indices": indices_list,
        "images": images,
        "captions": captions
    }


def _process_image(img: np.ndarray) -> np.ndarray:
    """
    Process a single image to ensure it has shape (H, W, C).

    Parameters:
    -----------
    img : np.ndarray
        Image array, either 2D (H, W) or 3D (H, W, C).

    Returns:
    --------
    np.ndarray
        Processed image with shape (H, W, C).
    """
    if img.ndim == 3:
        return img
    elif img.ndim == 2:
        # Add channel dimension for grayscale
        return img[..., np.newaxis]
    else:
        raise ValueError(f"Unexpected image dimensions: {img.ndim}D array")


# Example usage (uncomment to test)
# if __name__ == "__main__":
#     hdf5_path = "data/stimuli/nsd/nsd_stimuli.hdf5"
#     captions_path = "data/stimuli/nsd/captions.csv"  # Example
#
#     # Load images only
#     imgs1 = load_images(5, hdf5_path)
#     imgs2 = load_images([1, 3, 7], hdf5_path)
#     imgs3 = load_images(range(10, 20), hdf5_path)
#
#     # Load captions only
#     caps1 = load_captions(5, captions_path)
#     caps2 = load_captions([1, 3, 7], captions_path)
#     caps3 = load_captions(range(10, 20), captions_path)
#
#     # Load both
#     data1 = load_stimuli(5, hdf5_path, captions_path)
#     data2 = load_stimuli([1, 3, 7], hdf5_path, captions_path)
#     data3 = load_stimuli(range(10, 20), hdf5_path, captions_path)
#
#     print(f"Shape of imgs1: {imgs1.shape}")
#     print(f"Length of caps1: {len(caps1)}")
#     print(f"Data1 keys: {list(data1.keys())}")
#     print(f"Data1 images shape: {data1['images'].shape}")
#     print(f"Data1 captions length: {len(data1['captions'])}")

"""
CLIP Embedding Extraction Module

A clean interface for extracting CLIP embeddings from images and texts.
Supports batch processing, GPU acceleration, and optional FP16 inference.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Union


class CLIPEmbedder:
    """CLIP embedding extractor for images and texts."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", fp16: bool = False):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.model.eval()

    def encode_images(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Extract CLIP embeddings for batch of images.

        Args:
            images: Array of shape (N, H, W, C) with uint8 dtype
            batch_size: Batch size for memory efficiency

        Returns:
            Embeddings of shape (N, D)
        """
        n_images = len(images)
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                batch = images[i:i + batch_size]
                pil_images = [Image.fromarray(img) for img in batch]

                inputs = self.processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                features = self.model.get_image_features(**inputs)
                assert features.shape[-1] == 768
                embeddings.append(features.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract CLIP embeddings for batch of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for memory efficiency

        Returns:
            Embeddings of shape (N, D)
        """
        n_texts = len(texts)
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_texts, batch_size):
                batch = texts[i:i + batch_size]

                inputs = self.processor(text=batch, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                features = self.model.get_text_features(**inputs)
                assert features.shape[-1] == 768
                embeddings.append(features.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


def get_clip_embedding(image: Union[str, np.ndarray],
                      model_name: str = "openai/clip-vit-large-patch14") -> np.ndarray:
    """Legacy function for single image embedding extraction."""
    embedder = CLIPEmbedder(model_name)

    if isinstance(image, str):
        pil_image = Image.open(image)
        img_array = np.array(pil_image)
    else:
        img_array = image

    # Ensure 3D array
    if img_array.ndim == 2:
        img_array = img_array[:, :, np.newaxis]

    embeddings = embedder.encode_images(img_array[np.newaxis], batch_size=1)
    return embeddings[0]

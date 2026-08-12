"""OpenCLIP image embeddings -- the 512-d vectors that route a frame to a model.

Deliberately local and image-native. OpenAI's embedding endpoints are text-only
and cannot embed a drone frame, so they play no part in routing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from gridsight.config import get_settings

log = logging.getLogger("gridsight.embed")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ClipEmbedder:
    """Wraps OpenCLIP ViT-B-32 (laion2b_s34b_b79k) -> L2-normalized 512-d vectors."""

    def __init__(self) -> None:
        settings = get_settings()
        self.device = pick_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.clip_model, pretrained=settings.clip_pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.dim = settings.embed_dim
        log.info(
            "OpenCLIP %s/%s ready on %s",
            settings.clip_model,
            settings.clip_pretrained,
            self.device,
        )

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed PIL images -> (N, 512) float32, each row L2-normalized."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = torch.stack([self.preprocess(im.convert("RGB")) for im in chunk])
            feats = self.model.encode_image(batch.to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def embed_paths(self, paths: Iterable[Path], batch_size: int = 32) -> np.ndarray:
        images: list[Image.Image] = []
        for p in paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB").copy())
        return self.embed_images(images, batch_size=batch_size)

    def embed_one(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]


@lru_cache(maxsize=1)
def get_embedder() -> ClipEmbedder:
    """Load the CLIP weights once per process."""
    return ClipEmbedder()


def centroid(vectors: np.ndarray) -> np.ndarray:
    """L2-normalized mean of a set of embeddings -- the registry routing vector."""
    if vectors.size == 0:
        raise ValueError("cannot take a centroid of zero embeddings")
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        raise ValueError("degenerate centroid: embeddings cancelled out")
    return (mean / norm).astype(np.float32)

"""Cold-start: fit a brand-new specialist from a handful of reference images.

This is the path the agent takes when the registry has nothing that covers an
asset class. There is no validation split to calibrate against -- only the
reference images the operator supplied -- so the decision threshold is derived
from the normal envelope those references describe, and that derivation is
recorded on the model document rather than hidden.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from gridsight.inference import inference_device, to_tensor
from gridsight.train.store import (
    AnomalyModule,
    PatchcoreConfig,
    build_padim,
    build_patchcore,
    post_processor,
)

log = logging.getLogger("gridsight.fewshot")

#: Below this many references a coreset is meaningless, so PaDiM's Gaussian is
#: the better-behaved estimator.
PADIM_FALLBACK_BELOW = 4

#: Few-shot keeps far more of the patch budget than the seeded models: with a
#: dozen images, 1% of patches would be a handful of vectors.
FEWSHOT_CORESET_RATIO = 0.25

PIXEL_THRESHOLD_PERCENTILE = 99.5


def choose_backbone(n_images: int) -> str:
    return "padim" if n_images < PADIM_FALLBACK_BELOW else "patchcore"


@torch.inference_mode()
def _collect(module: AnomalyModule, images: list[Image.Image], image_size: int) -> None:
    """Run the reference set through the model in training mode to accumulate embeddings."""
    module.model.train()
    for img in images:
        module.model(to_tensor(img, image_size).to(inference_device()))
    module.fit()
    module.model.eval()


@torch.inference_mode()
def _calibrate(module: AnomalyModule, images: list[Image.Image], image_size: int) -> dict[str, float]:
    """Derive thresholds from the reference set's own score distribution."""
    scores: list[float] = []
    pixels: list[np.ndarray] = []
    for img in images:
        out = module.model(to_tensor(img, image_size).to(inference_device()))
        scores.append(float(out.pred_score.squeeze().item()))
        pixels.append(out.anomaly_map.squeeze().detach().cpu().numpy().ravel())

    arr = np.asarray(scores, dtype=np.float64)
    flat = np.concatenate(pixels)

    # Every reference image is normal by construction, so the tightest honest
    # decision boundary is the worst score the references produced.
    image_threshold = float(arr.max())
    pixel_threshold = float(np.percentile(flat, PIXEL_THRESHOLD_PERCENTILE))
    return {
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "image_min": float(arr.min()),
        "image_max": float(arr.max()),
        "pixel_min": float(flat.min()),
        "pixel_max": float(flat.max()),
        "reference_score_mean": float(arr.mean()),
        "reference_score_std": float(arr.std()),
    }


@torch.inference_mode()
def calibrate_pixel_stats(
    module: AnomalyModule, images: list[Image.Image], image_size: int
) -> dict[str, float]:
    """Derive pixel-level stats from anomaly maps over known-good images.

    Anomalib can only learn a pixel threshold when the validation split carries
    ground-truth masks. None of these Hub datasets do, so it leaves
    `_pixel_threshold` as NaN -- which silently poisons every heatmap. The
    honest substitute is the normal envelope: the hottest pixels a *good* image
    produces are, by construction, not defects.
    """
    pixels: list[np.ndarray] = []
    module.model.eval()
    for img in images:
        out = module.model(to_tensor(img, image_size).to(inference_device()))
        pixels.append(out.anomaly_map.squeeze().detach().cpu().numpy().ravel())

    flat = np.concatenate(pixels)
    return {
        "pixel_threshold": float(np.percentile(flat, PIXEL_THRESHOLD_PERCENTILE)),
        "pixel_min": float(flat.min()),
        "pixel_max": float(flat.max()),
    }


def apply_pixel_stats(module: AnomalyModule, stats: dict[str, float]) -> None:
    """Write calibrated pixel stats into the post-processor buffers."""
    sd = post_processor(module).state_dict()
    for buf_name, stat_name in (
        ("_pixel_threshold", "pixel_threshold"),
        ("pixel_min", "pixel_min"),
        ("pixel_max", "pixel_max"),
    ):
        sd[buf_name] = torch.tensor(stats[stat_name], dtype=sd[buf_name].dtype)
    post_processor(module).load_state_dict(sd)


def _apply_thresholds(module: AnomalyModule, stats: dict[str, float]) -> None:
    sd = post_processor(module).state_dict()
    mapping = {
        "_image_threshold": "image_threshold",
        "_pixel_threshold": "pixel_threshold",
        "image_min": "image_min",
        "image_max": "image_max",
        "pixel_min": "pixel_min",
        "pixel_max": "pixel_max",
    }
    for buf_name, stat_name in mapping.items():
        sd[buf_name] = torch.tensor(stats[stat_name], dtype=sd[buf_name].dtype)
    post_processor(module).load_state_dict(sd)


def fit_fewshot(
    images: list[Image.Image], image_size: int = 256
) -> tuple[AnomalyModule, PatchcoreConfig, dict[str, Any]]:
    """Fit a specialist from reference images alone. Returns (module, cfg, info)."""
    if not images:
        raise ValueError("cold start requires at least one reference image")

    started = time.time()
    kind = choose_backbone(len(images))
    cfg = PatchcoreConfig(coreset_sampling_ratio=FEWSHOT_CORESET_RATIO, image_size=image_size)

    module: AnomalyModule
    if kind == "padim":
        log.info("cold start: %d references -> PaDiM fallback (too few for a coreset)", len(images))
        module = build_padim()
    else:
        log.info(
            "cold start: %d references -> PatchCore few-shot (coreset=%.2f)",
            len(images),
            cfg.coreset_sampling_ratio,
        )
        module = build_patchcore(cfg)

    module.to(inference_device())
    _collect(module, images, image_size)
    stats = _calibrate(module, images, image_size)
    _apply_thresholds(module, stats)

    elapsed = time.time() - started
    info: dict[str, Any] = {
        "backbone": kind,
        "reference_images": len(images),
        "seconds": round(elapsed, 2),
        "threshold_policy": (
            f"image_threshold = max score over the {len(images)} reference images "
            f"(normal envelope); pixel_threshold = p{PIXEL_THRESHOLD_PERCENTILE} of reference "
            "anomaly-map values"
        ),
        **{k: round(v, 6) for k, v in stats.items()},
    }
    log.info(
        "cold start fitted in %.1fs (%s, threshold=%.4f over %d refs)",
        elapsed,
        kind,
        stats["image_threshold"],
        len(images),
    )
    return module, cfg, info

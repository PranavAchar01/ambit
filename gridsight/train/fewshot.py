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


#: Folds used to score each reference against a bank it was not fitted on.
#:
#: Every k gives all n references an out-of-sample score; k only decides how big
#: each calibration bank is, and the cost is dominated by backbone feature
#: extraction -- k fits over n/k images each. k=2 is therefore the cheapest
#: setting that still holds anything out, and it is measurably cheaper than k=3
#: (calibration 9.8 s -> 5.9 s on an 8-reference cold start).
#:
#: Smaller calibration banks score their holdout *higher*, not lower, because a
#: half-size bank covers the normal manifold less well. So the low-k threshold
#: errs conservative -- towards more false alarms rather than missed defects --
#: which is the correct direction for a system whose whole claim is abstention.
CALIBRATION_FOLDS = 2

#: A calibration fold must retain at least this many references to be worth
#: fitting. Below it there is no out-of-sample estimate and the threshold falls
#: back to the in-sample maximum -- loudly, because that boundary is not one.
CALIBRATION_MIN_FIT = 2


@torch.inference_mode()
def _score_images(module: AnomalyModule, images: list[Image.Image], image_size: int) -> list[float]:
    scores: list[float] = []
    for img in images:
        out = module.model(to_tensor(img, image_size).to(inference_device()))
        scores.append(float(out.pred_score.squeeze().item()))
    return scores


def _out_of_sample_scores(
    images: list[Image.Image], cfg: PatchcoreConfig, folds: int = CALIBRATION_FOLDS
) -> list[float]:
    """Score every reference against a memory bank that does not contain it.

    PatchCore stores patches and scores by nearest neighbour, so a frame that is
    in the bank retrieves itself at near-zero distance. The maximum score over
    the fitted references is therefore an in-sample statistic and not a decision
    boundary for anything: measured on a 10-frame bench capture, every one of ten
    genuinely normal frames scored above it once withheld (24.5-35.5 against an
    in-sample maximum of 23.3).

    This is the same correction the routing gate already carries -- its
    percentile was moved off the training images for exactly this reason --
    applied to the other threshold, which never received it.

    Returns an empty list when the set is too small to hold anything out.
    """
    n = len(images)
    k = min(folds, n)
    if k < 2 or n - -(-n // k) < CALIBRATION_MIN_FIT:
        return []

    scores: list[float] = []
    for f in range(k):
        holdout = [i for i in range(n) if i % k == f]
        fit = [i for i in range(n) if i % k != f]
        if len(fit) < CALIBRATION_MIN_FIT or not holdout:
            return []
        module = build_patchcore(cfg)
        module.to(inference_device())
        _collect(module, [images[i] for i in fit], cfg.image_size)
        scores.extend(_score_images(module, [images[i] for i in holdout], cfg.image_size))
    return scores


@torch.inference_mode()
def _calibrate(
    module: AnomalyModule,
    images: list[Image.Image],
    image_size: int,
    out_of_sample: list[float] | None = None,
) -> dict[str, float]:
    """Derive thresholds from the reference set's score distribution."""
    scores: list[float] = []
    pixels: list[np.ndarray] = []
    for img in images:
        out = module.model(to_tensor(img, image_size).to(inference_device()))
        scores.append(float(out.pred_score.squeeze().item()))
        pixels.append(out.anomaly_map.squeeze().detach().cpu().numpy().ravel())

    arr = np.asarray(scores, dtype=np.float64)
    flat = np.concatenate(pixels)

    # Every reference image is normal by construction, so the tightest honest
    # decision boundary is the worst score a known-good frame produced -- but it
    # has to be a frame the bank was not fitted on, or the boundary is measured
    # on the one population that cannot cross it.
    in_sample_max = float(arr.max())
    if out_of_sample:
        oos = np.asarray(out_of_sample, dtype=np.float64)
        image_threshold = float(oos.max())
        image_max = max(in_sample_max, image_threshold)
    else:
        log.warning(
            "too few references (%d) to hold any out -- image threshold falls back to the in-sample "
            "maximum, which an unseen normal frame is expected to exceed. Treat nominal verdicts from "
            "this model as unproven and add references.",
            len(images),
        )
        image_threshold = in_sample_max
        image_max = in_sample_max

    pixel_threshold = float(np.percentile(flat, PIXEL_THRESHOLD_PERCENTILE))
    stats = {
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "image_min": float(arr.min()),
        "image_max": image_max,
        "pixel_min": float(flat.min()),
        "pixel_max": float(flat.max()),
        "reference_score_mean": float(arr.mean()),
        "reference_score_std": float(arr.std()),
        "image_threshold_in_sample": in_sample_max,
        "image_threshold_out_of_sample": float(len(out_of_sample or [])),
    }
    if out_of_sample:
        oos = np.asarray(out_of_sample, dtype=np.float64)
        stats["out_of_sample_mean"] = float(oos.mean())
        stats["out_of_sample_min"] = float(oos.min())
    return stats


@torch.inference_mode()
def calibrate_reference_stats(
    module: AnomalyModule, images: list[Image.Image], image_size: int
) -> dict[str, float]:
    """Public entry point for the normal-envelope threshold policy.

    Cold-start is not the only path with no labelled defects: a class captured
    from a prototype shop's own bench has none either, and anomalib's adaptive
    image threshold needs both classes present to mean anything. Both cases want
    the identical policy -- the boundary is the worst score a known-good frame
    produced -- so both call the same code rather than growing two definitions of
    "normal envelope" that could drift apart.
    """
    module.model.eval()
    return _calibrate(module, images, image_size)


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


def fit_normal_only(
    images: list[Image.Image], cfg: PatchcoreConfig
) -> tuple[AnomalyModule, dict[str, float]]:
    """Fit a PatchCore specialist on normals alone, with no labelled split.

    Cold-start already does this for images handed over at request time. A class
    ingested from a bench capture needs the identical thing at seed time: there
    is no defect set, so anomalib's Engine has nothing to validate or test
    against -- it raises on an empty `abnormal_dir`, and Lightning still demands
    a validation dataloader if the split is switched off. Rather than feed it a
    one-class split whose metrics would be meaningless, the normal-only path
    reuses the collection and calibration routines this module already owns.

    Returns the fitted module and the calibration it was given.
    """
    if not images:
        raise ValueError("normal-only fit requires at least one training image")

    out_of_sample = _out_of_sample_scores(images, cfg)
    module = build_patchcore(cfg)
    module.to(inference_device())
    _collect(module, images, cfg.image_size)
    stats = _calibrate(module, images, cfg.image_size, out_of_sample)
    _apply_thresholds(module, stats)
    log.info(
        "normal-only fit: %d frames, coreset=%.2f, image_threshold=%.4f (%s; in-sample max was %.4f)",
        len(images),
        cfg.coreset_sampling_ratio,
        stats["image_threshold"],
        f"worst of {len(out_of_sample)} out-of-sample normals" if out_of_sample else "IN-SAMPLE ONLY",
        stats["image_threshold_in_sample"],
    )
    return module, stats


def apply_reference_stats(module: AnomalyModule, stats: dict[str, float]) -> None:
    """Write a normal-envelope calibration into the post-processor buffers.

    The public counterpart to `calibrate_reference_stats`, for callers outside
    the cold-start path that also have no labelled defects.
    """
    _apply_thresholds(module, stats)


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

    # Calibrated before the final bank is built, and only for the coreset
    # backbone: the PaDiM fallback fits a Gaussian rather than a patch bank, and
    # at fewer than four references there is nothing to hold out anyway.
    out_of_sample = _out_of_sample_scores(images, cfg) if kind == "patchcore" else []

    module.to(inference_device())
    _collect(module, images, image_size)
    stats = _calibrate(module, images, image_size, out_of_sample)
    _apply_thresholds(module, stats)

    elapsed = time.time() - started
    policy = (
        (
            f"image_threshold = worst score over {len(out_of_sample)} references scored against a "
            f"bank fitted without them ({CALIBRATION_FOLDS}-fold); the in-sample maximum was "
            f"{stats['image_threshold_in_sample']:.4f} and is not a boundary, because PatchCore "
            "retrieves a fitted frame at near-zero distance"
        )
        if out_of_sample
        else (
            f"image_threshold = max score over the {len(images)} reference images IN SAMPLE -- too "
            "few references to hold any out. An unseen normal frame is expected to exceed it"
        )
    )
    info: dict[str, Any] = {
        "backbone": kind,
        "reference_images": len(images),
        "seconds": round(elapsed, 2),
        "threshold_policy": (
            f"{policy}; pixel_threshold = p{PIXEL_THRESHOLD_PERCENTILE} of reference anomaly-map values"
        ),
        **{k: round(v, 6) for k, v in stats.items()},
    }
    log.info(
        "cold start fitted in %.1fs (%s, threshold=%.4f over %d refs, %d out-of-sample)",
        elapsed,
        kind,
        stats["image_threshold"],
        len(images),
        len(out_of_sample),
    )
    return module, cfg, info

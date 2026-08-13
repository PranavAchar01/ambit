"""Runtime inference: score a frame, build a heatmap, derive defect regions.

Models are loaded straight out of GridFS and kept in a small in-process LRU so a
burst of frames for the same asset class does not re-download the memory bank.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from gridsight.train.store import (
    AnomalyModule,
    PatchcoreConfig,
    deserialize_patchcore,
    get_weights,
    post_processor,
)

log = logging.getLogger("gridsight.inference")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Regions smaller than this fraction of the frame are noise, not defects.
MIN_REGION_AREA_FRAC = 0.002
MAX_REGIONS = 8


def inference_device() -> torch.device:
    """CPU keeps scores bit-reproducible across processes, which the GridFS
    round-trip proof depends on."""
    return torch.device("cpu")


def build_transform(image_size: int) -> v2.Transform:
    return v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    return build_transform(image_size)(image.convert("RGB")).unsqueeze(0)


@dataclass
class BBoxRegion:
    x: int
    y: int
    w: int
    h: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": round(self.score, 4)}


@dataclass
class InferenceResult:
    raw_score: float
    normalized_score: float
    is_defect: bool
    image_threshold: float
    anomaly_map: np.ndarray = field(repr=False)
    regions: list[BBoxRegion] = field(default_factory=list)
    pixel_threshold_approximate: bool = False

    def severity(self) -> float:
        return float(np.clip(self.normalized_score, 0.0, 1.0))


def _normalize(values: np.ndarray, lo: float, hi: float, threshold: float) -> np.ndarray:
    """Anomalib's min-max normalisation: the threshold maps to exactly 0.5."""
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-12:
        return np.clip(values - threshold + 0.5, 0.0, 1.0)
    return np.clip((values - threshold) / span + 0.5, 0.0, 1.0)


def _resolve_pixel_threshold(stored: float, amap: np.ndarray) -> tuple[float, bool]:
    """Fall back to the frame's own distribution if the stored threshold is unusable.

    Anomalib leaves `_pixel_threshold` as NaN when the validation split had no
    ground-truth masks. Models registered by GridSight calibrate it at training
    time; this guard exists so a stale or externally-produced model degrades to a
    visible, logged approximation instead of an all-NaN heatmap.
    """
    if np.isfinite(stored):
        return stored, False
    fallback = float(np.percentile(amap, 99.0))
    log.warning(
        "stored pixel threshold is not finite; falling back to this frame's p99 (%.4f). "
        "Region boxes are approximate for this model.",
        fallback,
    )
    return fallback, True


def _stat(module: AnomalyModule, key: str) -> float:
    return float(post_processor(module).state_dict()[key].item())


@torch.inference_mode()
def run_inference(
    module: AnomalyModule,
    cfg: PatchcoreConfig,
    image: Image.Image,
) -> InferenceResult:
    """Score one frame and localise the defect regions."""
    module.eval()
    device = inference_device()
    module.to(device)

    tensor = to_tensor(image, cfg.image_size).to(device)
    out = module.model(tensor)

    raw = float(out.pred_score.squeeze().item())
    amap = out.anomaly_map.squeeze().detach().cpu().numpy().astype(np.float32)

    img_thr = _stat(module, "_image_threshold")
    img_lo, img_hi = _stat(module, "image_min"), _stat(module, "image_max")
    pix_lo, pix_hi = _stat(module, "pixel_min"), _stat(module, "pixel_max")
    pix_thr, approximate = _resolve_pixel_threshold(_stat(module, "_pixel_threshold"), amap)

    norm_score = float(_normalize(np.array([raw], dtype=np.float32), img_lo, img_hi, img_thr)[0])
    norm_map = _normalize(amap, pix_lo, pix_hi, pix_thr)

    regions = extract_regions(norm_map, image.size)
    return InferenceResult(
        pixel_threshold_approximate=approximate,
        raw_score=raw,
        normalized_score=norm_score,
        is_defect=raw >= img_thr,
        image_threshold=img_thr,
        anomaly_map=norm_map,
        regions=regions,
    )


def extract_regions(
    norm_map: np.ndarray, target_size: tuple[int, int], threshold: float = 0.5
) -> list[BBoxRegion]:
    """Threshold the normalised heatmap and derive boxes via connected components."""
    h_map, w_map = norm_map.shape
    mask = (norm_map >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        return []

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    tw, th = target_size
    sx, sy = tw / w_map, th / h_map
    min_area = MIN_REGION_AREA_FRAC * w_map * h_map

    regions: list[BBoxRegion] = []
    for i in range(1, count):
        x, y, w, h, area = (int(stats[i, j]) for j in range(5))
        if area < min_area:
            continue
        peak = float(norm_map[labels == i].max())
        regions.append(
            BBoxRegion(
                x=int(round(x * sx)),
                y=int(round(y * sy)),
                w=max(1, int(round(w * sx))),
                h=max(1, int(round(h * sy))),
                score=peak,
            )
        )
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions[:MAX_REGIONS]


#: Below this normalised value the heatmap is fully transparent, so the frame
#: underneath stays readable instead of being buried under a wash of colour.
HEATMAP_FLOOR = 0.5
HEATMAP_MAX_ALPHA = 0.78

#: The overlay is single-hue by design. The UI is black and red, where red means
#: attention and nothing else, so a multi-hue colormap like TURBO would introduce
#: greens and blues that read as *categories* rather than as intensity. Anomaly is
#: one scalar; it gets one colour, from a dim red at the threshold to a bright red
#: at the peak. BGR order, because that is what cv2.imencode writes.
HEATMAP_COLD_BGR = (20, 20, 193)  # #c11414 -- just over the pixel threshold
HEATMAP_HOT_BGR = (45, 45, 255)  # #ff2d2d -- peak anomaly


def heatmap_png(norm_map: np.ndarray, target_size: tuple[int, int]) -> bytes:
    """Render the normalised heatmap as an alpha-ramped single-hue RGBA PNG.

    Alpha rises with anomaly, so only the regions that actually exceeded the
    pixel threshold tint the frame. A flat opaque overlay would hide the very
    defect the operator is being asked to look at.
    """
    tw, th = target_size
    resized = cv2.resize(norm_map, (tw, th), interpolation=cv2.INTER_LINEAR)
    resized = np.nan_to_num(resized, nan=0.0, posinf=1.0, neginf=0.0)

    # One ramp drives both the red intensity and the opacity, so a hotter pixel is
    # both brighter and more opaque rather than merely a different colour.
    ramp = np.clip((resized - HEATMAP_FLOOR) / max(1e-6, 1.0 - HEATMAP_FLOOR), 0.0, 1.0)
    cold = np.array(HEATMAP_COLD_BGR, dtype=np.float32)
    hot = np.array(HEATMAP_HOT_BGR, dtype=np.float32)
    coloured = (cold + ramp[..., None] * (hot - cold)).astype(np.uint8)

    alpha = (ramp * HEATMAP_MAX_ALPHA * 255.0).astype(np.uint8)

    rgba = np.dstack([coloured, alpha])
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("failed to encode heatmap PNG")
    return bytes(buf.tobytes())


# ---------------------------------------------------------------------------
# LRU cache of GridFS-resident models
# ---------------------------------------------------------------------------


class ModelCache:
    """Bounded in-process cache keyed by GridFS file id."""

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, tuple[AnomalyModule, PatchcoreConfig]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, weights_file_id: str) -> tuple[AnomalyModule, PatchcoreConfig]:
        key = str(weights_file_id)
        if key in self._items:
            self._items.move_to_end(key)
            self.hits += 1
            return self._items[key]

        self.misses += 1
        log.info("loading weights %s from GridFS", key)
        module, cfg = deserialize_patchcore(get_weights(key))
        module.to(inference_device())
        self._items[key] = (module, cfg)
        if len(self._items) > self.capacity:
            evicted, _ = self._items.popitem(last=False)
            log.info("evicted %s from model cache", evicted)
        return module, cfg

    def invalidate(self, weights_file_id: str) -> None:
        self._items.pop(str(weights_file_id), None)


MODEL_CACHE = ModelCache()

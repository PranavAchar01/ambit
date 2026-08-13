"""Measure a candidate definition of the `vegetation` asset class.

The shipped definition -- "a vegetation box that intersects a conductor box" --
has a supervised CLIP-probe ceiling of 0.591, i.e. the label is barely visible in
the pixels, so PatchCore's 0.490 is a data-definition problem rather than a model
problem. This harness makes competing definitions comparable on the same footing:

  * supervised ceiling  -- 5-fold CLIP linear probe. If this is low, the label is
                           not learnable from the imagery and nothing downstream
                           will save it.
  * patchcore auroc     -- what actually ships, fitted on good-only.
  * routing separability -- can the redefined class still be told apart from the
                           four other registered specialists?

Nothing here writes to MongoDB or to `data/`; it only reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from gridsight.embed import centroid, get_embedder

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s | %(message)s")
for _n in ("httpx", "urllib3", "pymongo", "root", "timm", "lightning", "anomalib", "datasets"):
    logging.getLogger(_n).setLevel(logging.ERROR)

Box = list[float]

TRAIN_GOOD_CAP = 120
TEST_GOOD_CAP = 30
TEST_DEFECT_CAP = 40
UPSCALE_TO = 160


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def inter_area(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def iou(a: Box, b: Box) -> float:
    i = inter_area(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(union, 1e-6)


def overlap_of_smaller(a: Box, b: Box) -> float:
    """Intersection over the area of the smaller box -- robust when a thin cable
    box and a big canopy box overlap."""
    i = inter_area(a, b)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return i / max(smaller, 1e-6)


def gap(a: Box, b: Box) -> float:
    """Edge-to-edge gap in px; 0 when the boxes touch or overlap."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def union_box(a: Box, b: Box) -> Box:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def square_crop(img: Image.Image, box: Box, margin: float) -> Image.Image | None:
    x0, y0, x1, y1 = box
    if x1 - x0 <= 4 or y1 - y0 <= 4:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(max(x1 - x0, y1 - y0) * (1 + margin), float(min(img.width, img.height)))
    nx0 = int(min(max(cx - side / 2, 0), img.width - side))
    ny0 = int(min(max(cy - side / 2, 0), img.height - side))
    crop = img.convert("RGB").crop((nx0, ny0, int(nx0 + side), int(ny0 + side)))
    if crop.width < UPSCALE_TO:
        crop = crop.resize((UPSCALE_TO, UPSCALE_TO), Image.LANCZOS)
    return crop


# ---------------------------------------------------------------------------
# variant spec
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    #: what counts as encroachment
    metric: str = "intersect"  # intersect | iou | overlap_smaller | gap
    threshold: float = 0.0  # iou/overlap: minimum; gap: maximum px
    #: a "good" canopy must be at least this far from any conductor (px)
    clear_gap_min: float = 0.0
    #: what the crop is centred on
    anchor: str = "vegetation"  # vegetation | union | conductor
    margin: float = 0.0
    min_side: float = 32.0
    max_side: float = 640.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def encroaching(v: Box, c: Box, spec: Variant) -> bool:
    if spec.metric == "intersect":
        return inter_area(v, c) > 0
    if spec.metric == "iou":
        return iou(v, c) >= spec.threshold
    if spec.metric == "overlap_smaller":
        return overlap_of_smaller(v, c) >= spec.threshold
    if spec.metric == "gap":
        return gap(v, c) <= spec.threshold
    raise ValueError(f"unknown metric {spec.metric!r}")


def build_pools(
    spec: Variant, limit_frames: int | None = None
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Extract (good, defect) crops for this variant from the powerline corpus."""
    from gridsight.ingest.hf_scrape import _powerline_frames

    good: list[Image.Image] = []
    defect: list[Image.Image] = []
    seen = 0

    for img, grouped in _powerline_frames():
        seen += 1
        if limit_frames and seen > limit_frames:
            break
        conductors = grouped.get("Cable", []) + grouped.get("Broken Cable", [])
        for v in grouped.get("Vegetation", []):
            side = max(v[2] - v[0], v[3] - v[1])
            if side < spec.min_side or side > spec.max_side:
                continue

            hits = [c for c in conductors if encroaching(v, c, spec)]
            if hits:
                # crop so the conductor that defines the encroachment stays visible
                partner = max(hits, key=lambda c: inter_area(v, c))
                if spec.anchor == "union":
                    box = union_box(v, partner)
                elif spec.anchor == "conductor":
                    box = partner
                else:
                    box = v
                crop = square_crop(img, box, spec.margin)
                if crop is not None:
                    defect.append(crop)
                continue

            # a "good" canopy must be genuinely clear of every conductor, and is
            # framed the same way so the classes differ by content, not framing
            if (
                conductors
                and spec.clear_gap_min > 0
                and min(gap(v, c) for c in conductors) < spec.clear_gap_min
            ):
                continue
            if spec.anchor == "conductor":
                continue  # handled below
            crop = square_crop(img, v, spec.margin)
            if crop is not None:
                good.append(crop)

        if spec.anchor == "conductor":
            # right-of-way framing: a conductor with a clear corridor is normal
            vegs = grouped.get("Vegetation", [])
            for c in conductors:
                side = max(c[2] - c[0], c[3] - c[1])
                if side < spec.min_side or side > spec.max_side:
                    continue
                if any(encroaching(v, c, spec) for v in vegs):
                    continue
                if vegs and spec.clear_gap_min > 0 and min(gap(v, c) for v in vegs) < spec.clear_gap_min:
                    continue
                crop = square_crop(img, c, spec.margin)
                if crop is not None:
                    good.append(crop)

    return good, defect


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def supervised_ceiling(good: list[Image.Image], defect: list[Image.Image]) -> float:
    """5-fold CLIP linear probe -- how learnable the label is at all."""
    emb = get_embedder()
    X = np.concatenate([emb.embed_images(good), emb.embed_images(defect)])
    y = np.array([0] * len(good) + [1] * len(defect))
    if len(set(y)) < 2 or min((y == 0).sum(), (y == 1).sum()) < 5:
        return float("nan")
    probs = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, probs))


def patchcore_auroc(good: list[Image.Image], defect: list[Image.Image], seed: int = 1234) -> dict[str, Any]:
    """Fit PatchCore on good-only in a temp Anomalib folder and score the test split."""
    from anomalib.data import Folder
    from anomalib.engine import Engine

    from gridsight.train.store import PatchcoreConfig, build_patchcore

    rng = random.Random(seed)
    good = list(good)
    defect = list(defect)
    rng.shuffle(good)
    rng.shuffle(defect)

    n_train = min(TRAIN_GOOD_CAP, max(1, int(len(good) * 0.8)))
    train_good = good[:n_train]
    test_good = good[n_train : n_train + TEST_GOOD_CAP]
    test_defect = defect[:TEST_DEFECT_CAP]
    if len(train_good) < 20 or len(test_defect) < 5 or len(test_good) < 5:
        return {
            "image_auroc": None,
            "train_good": len(train_good),
            "test_good": len(test_good),
            "test_defect": len(test_defect),
            "note": "insufficient samples",
        }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, images in (
            ("train/good", train_good),
            ("test/good", test_good),
            ("test/defect", test_defect),
        ):
            d = root / name
            d.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(images):
                im.save(d / f"{i:05d}.png")

        dm = Folder(
            name="variant",
            root=root,
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir="test/defect",
            train_batch_size=8,
            eval_batch_size=8,
            num_workers=0,
            val_split_mode="from_test",
            val_split_ratio=0.5,
            seed=seed,
        )
        module = build_patchcore(PatchcoreConfig())
        engine = Engine(
            default_root_dir=str(root / "artifacts"),
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        engine.fit(model=module, datamodule=dm)
        results = engine.test(model=module, datamodule=dm, verbose=False)

    auroc = None
    if results:
        for k, v in dict(results[0]).items():
            if "image_auroc" in k.lower():
                auroc = round(float(v), 4)
    return {
        "image_auroc": auroc,
        "train_good": len(train_good),
        "test_good": len(test_good),
        "test_defect": len(test_defect),
    }


def routing_separability(good: list[Image.Image]) -> dict[str, Any]:
    """Would this redefined class still be routable against the live registry?"""
    from gridsight.db.mongo import models_col

    others = [
        d
        for d in models_col().find({}, {"asset_class": 1, "embedding": 1, "name": 1})
        if d["asset_class"] != "vegetation"
    ]
    if not others or len(good) < 40:
        return {"top1_vs_registry": None, "note": "registry empty or too few samples"}

    emb = get_embedder()
    vecs = emb.embed_images(good)
    fit, probe = vecs[: int(len(vecs) * 0.7)], vecs[int(len(vecs) * 0.7) :]
    if len(probe) < 5:
        return {"top1_vs_registry": None, "note": "too few probe samples"}

    c = centroid(fit)
    mat = np.stack([c] + [np.asarray(d["embedding"], dtype=np.float32) for d in others])
    names = ["vegetation(variant)"] + [d["asset_class"] for d in others]
    sims = probe @ mat.T
    idx = sims.argmax(axis=1)
    top_scores = (1 + sims[np.arange(len(probe)), idx]) / 2
    return {
        "top1_vs_registry": round(float((idx == 0).mean()), 4),
        "mean_top_score": round(float(top_scores.mean()), 4),
        "n_probe": int(len(probe)),
        "competitors": names[1:],
    }


VARIANTS: dict[str, Variant] = {
    "shipped": Variant(name="shipped", metric="intersect", anchor="vegetation", margin=0.0),
    "iou15_union": Variant(
        name="iou15_union", metric="iou", threshold=0.15, anchor="union", margin=0.15, clear_gap_min=40.0
    ),
    "ovl30_union": Variant(
        name="ovl30_union",
        metric="overlap_smaller",
        threshold=0.30,
        anchor="union",
        margin=0.15,
        clear_gap_min=40.0,
    ),
    "ovl50_union_wide": Variant(
        name="ovl50_union_wide",
        metric="overlap_smaller",
        threshold=0.50,
        anchor="union",
        margin=0.35,
        clear_gap_min=60.0,
    ),
    "corridor": Variant(
        name="corridor",
        metric="overlap_smaller",
        threshold=0.30,
        anchor="conductor",
        margin=0.25,
        clear_gap_min=40.0,
        min_side=60.0,
    ),
    "veg_margin": Variant(
        name="veg_margin",
        metric="overlap_smaller",
        threshold=0.30,
        anchor="vegetation",
        margin=0.40,
        clear_gap_min=40.0,
    ),
}


def evaluate(spec: Variant, skip_patchcore: bool = False) -> dict[str, Any]:
    good, defect = build_pools(spec)
    out: dict[str, Any] = {
        "variant": spec.name,
        "spec": spec.to_dict(),
        "pool_good": len(good),
        "pool_defect": len(defect),
    }
    if len(good) < 25 or len(defect) < 5:
        out["note"] = "pools too small to evaluate"
        return out

    out["supervised_ceiling"] = round(supervised_ceiling(good, defect), 4)
    out["routing"] = routing_separability(good)
    if not skip_patchcore:
        out["patchcore"] = patchcore_auroc(good, defect)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a candidate vegetation class definition")
    ap.add_argument("--variant", default="shipped", help=f"one of {sorted(VARIANTS)} or 'custom'")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--clear-gap-min", type=float, default=None)
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--min-side", type=float, default=None)
    ap.add_argument("--skip-patchcore", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    spec = VARIANTS.get(args.variant, Variant(name=args.variant))
    for field in ("metric", "threshold", "clear_gap_min", "anchor", "margin", "min_side"):
        val = getattr(args, field)
        if val is not None:
            spec = Variant(**{**spec.to_dict(), "name": args.variant, field: val})

    result = evaluate(spec, skip_patchcore=args.skip_patchcore)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

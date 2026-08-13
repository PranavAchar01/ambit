"""Recompute every registered model's centroid and coverage gate.

The gate depends only on CLIP embeddings of `train/good`, never on the PatchCore
memory bank, so a change to how the gate is derived does not require retraining --
only re-embedding. That keeps a calibration fix cheap enough to actually run.

Why this exists: gates were originally the 5th percentile of the *training* set's
similarity to its own centroid, which is optimistic because the centroid is fitted
on those exact frames. `mvtec_leather` exposed it -- a texture whose entire training
spread is under 0.002 wide produced a gate above its own held-out median, and the
model refused 17/25 of its own good frames. The gate is now measured on frames the
centroid never saw.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from gridsight.db.mongo import models_col
from gridsight.train.train_class import (
    ROUTING_GATE_PERCENTILE,
    class_dir,
    list_images,
    routing_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
for _n in ("httpx", "urllib3", "pymongo", "root", "timm"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("gridsight.recalibrate")


def holdout_acceptance(asset_class: str, gate: float) -> tuple[int, int]:
    """How many held-out good frames would this gate actually admit?"""
    from gridsight.embed import get_embedder

    paths = list_images(class_dir(asset_class) / "test" / "good")
    if not paths:
        return 0, 0
    doc = models_col().find_one({"asset_class": asset_class}, {"embedding": 1})
    if doc is None:
        return 0, 0
    centroid_vec = np.asarray(doc["embedding"], dtype=np.float32)
    vectors = get_embedder().embed_paths(paths)
    scores = (1.0 + vectors @ centroid_vec) / 2.0
    return int((scores >= gate).sum()), len(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalibrate routing gates from held-out normals")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    docs = list(
        models_col().find({}, {"name": 1, "asset_class": 1, "routing_threshold": 1}).sort("created_at", 1)
    )
    if not docs:
        log.error("registry is empty")
        return 1

    print(f"{'ASSET CLASS':<22}{'OLD GATE':>10}{'NEW GATE':>10}{'DELTA':>9}   HELD-OUT GOOD ADMITTED")
    print("-" * 84)

    updated = skipped = 0
    for doc in docs:
        asset_class = str(doc["asset_class"])
        if not (class_dir(asset_class) / "train" / "good").is_dir():
            print(f"{asset_class:<22}{'':>10}{'':>10}{'':>9}   skipped: no on-disk training data")
            skipped += 1
            continue

        embedding, n_train, _golden, gate = routing_embedding(asset_class)
        old = float(doc.get("routing_threshold") or 0.0)

        if not args.dry_run:
            models_col().update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "embedding": [float(x) for x in embedding],
                        "routing_threshold": gate,
                        "gate_calibration": {
                            "method": "heldout-percentile",
                            "percentile": ROUTING_GATE_PERCENTILE,
                            "train_samples": n_train,
                        },
                    }
                },
            )
        admitted, total = holdout_acceptance(asset_class, gate)
        share = f"{admitted}/{total}" if total else "n/a"
        print(f"{asset_class:<22}{old:>10.4f}{gate:>10.4f}{gate - old:>+9.4f}   {share}")
        updated += 1

    print("-" * 84)
    print(f"{'updated' if not args.dry_run else 'would update'} {updated} models, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

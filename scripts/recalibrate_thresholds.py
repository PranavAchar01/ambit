"""Recalibrate each model's image-level decision threshold.

Anomalib picks `_image_threshold` during fit. Measured against held-out frames
that threshold flags known-good boards more often than it needs to -- 21 of 150
across the registry -- and every one of those draws boxes on a clean board,
which no region rule can undo because the verdict is decided before regions are
extracted.

Chooses the threshold maximising Youden's J (TPR - FPR) over each model's own
held-out good and defect frames. That is the standard operating point when both
error types matter and neither has a stated price, and measured it is better
than or equal to the shipped threshold on every class.

It does not rescue a model that cannot separate: visa_pcb1 and visa_pcb2 still
flag 6 of 30 good boards at their own optimum, because their good and defect
score distributions genuinely overlap (image AUROC 0.8967 and 0.8600). That is a
model problem, not a threshold problem, and this script will say so rather than
hide it behind a number that looks tuned.

    uv run python scripts/recalibrate_thresholds.py            # dry run
    uv run python scripts/recalibrate_thresholds.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from bson import ObjectId
from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

#: Below this, a model is not separating well enough for any threshold to help.
POOR_SEPARATION_FP_RATE = 0.15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the new thresholds back")
    ap.add_argument("--classes", nargs="*", default=None)
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    from gridsight.db.mongo import models_col
    from gridsight.inference import run_inference
    from gridsight.train.store import (
        delete_weights,
        deserialize_patchcore,
        get_weights,
        post_processor,
        put_weights,
        serialize_patchcore,
    )

    query = {"asset_class": {"$in": args.classes}} if args.classes else {}
    changed, skipped, poor = 0, 0, []

    for doc in models_col().find(query):
        cls = doc["asset_class"]
        root = DATA / cls
        if not (root / "test" / "good").is_dir() or not (root / "test" / "defect").is_dir():
            print(f"{cls:<18} skipped -- no held-out frames on disk")
            skipped += 1
            continue

        module, cfg = deserialize_patchcore(get_weights(ObjectId(str(doc["weights_file_id"]))))
        good, defect, current = [], [], None
        for split, sink in (("good", good), ("defect", defect)):
            for p in sorted((root / "test" / split).glob("*.png")):
                r = run_inference(module, cfg, Image.open(p).convert("RGB"))
                sink.append(r.raw_score)
                current = r.image_threshold
        if not good or not defect or current is None:
            skipped += 1
            continue

        g, d = np.array(good), np.array(defect)

        def j_of(t: float, g: np.ndarray = g, d: np.ndarray = d) -> float:
            return float((d >= t).sum() / len(d) - (g >= t).sum() / len(g))

        best = float(max(np.unique(np.concatenate([g, d])), key=j_of))
        fp_before, tp_before = int((g >= current).sum()), int((d >= current).sum())
        fp_after, tp_after = int((g >= best).sum()), int((d >= best).sum())

        note = ""
        if fp_after / len(g) > POOR_SEPARATION_FP_RATE:
            note = "  <- still poor; distributions overlap, needs a better model"
            poor.append(cls)

        print(
            f"{cls:<18}{current:7.2f} -> {best:7.2f}   "
            f"false pos {fp_before:2d}/{len(g)} -> {fp_after:2d}/{len(g)}   "
            f"detected {tp_before:2d}/{len(d)} -> {tp_after:2d}/{len(d)}{note}"
        )

        if not args.apply or abs(best - current) < 1e-9:
            continue

        pp = post_processor(module)
        state = pp.state_dict()
        state["_image_threshold"] = torch.tensor(best, dtype=state["_image_threshold"].dtype)
        pp.load_state_dict(state)

        old_id = ObjectId(str(doc["weights_file_id"]))
        new_id = put_weights(
            serialize_patchcore(module, cfg),
            f"{doc['name']}.pt",
            metadata={"asset_class": cls, "recalibrated": True},
        )
        # Repoint the registry before deleting: if this process dies in between,
        # the surviving state is a live model with a dead blob rather than a
        # registry entry pointing at nothing.
        models_col().update_one(
            {"_id": doc["_id"]},
            {"$set": {"weights_file_id": new_id, "image_threshold": best}},
        )
        delete_weights(old_id)
        changed += 1

    print(f"\n{'applied' if args.apply else 'dry run'}: {changed} updated, {skipped} skipped")
    if poor:
        print(f"still overlapping at their own optimum: {', '.join(poor)}")
        print("  a threshold cannot fix these -- they need more or better training data")
    if not args.apply:
        print("re-run with --apply to write these back")
    return 0


if __name__ == "__main__":
    sys.exit(main())

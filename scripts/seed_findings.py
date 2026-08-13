"""Populate the `findings` collection so the trends dashboard has a fleet to show.

Every finding written here is a REAL inference: a real held-out frame, routed by a
real $vectorSearch, scored by the real specialist that won. Nothing is invented.

`--spread-days` only moves the *timestamp* so a fleet's inspection history spans
more than the minute this script runs in. Findings written that way carry
`backfilled: true` and their original wall-clock time, so a backfilled timeline is
never mistaken for one Ambit actually observed over time.
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import UTC, datetime, timedelta

from bson import ObjectId
from PIL import Image

from gridsight.agent.graph import inspect_frame, store_pil
from gridsight.config import DATA_ROOT
from gridsight.db.mongo import findings_col, models_col

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("gridsight.seed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--spread-days", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    classes = sorted(models_col().distinct("asset_class"))
    if not classes:
        log.error("registry is empty; train models first")
        return 1

    if args.spread_days:
        print(
            f"NOTE: timestamps will be spread across the last {args.spread_days} days so the "
            "trends view has a timeline. Scores and verdicts are real inference results; only "
            "the timestamp is shifted, and every such document is flagged backfilled=true.\n"
        )

    written = 0
    for asset_class in classes:
        for split, count in (("test/defect", args.per_class), ("test/good", args.per_class)):
            paths = sorted((DATA_ROOT / asset_class / split).glob("*.png"))
            if not paths:
                continue
            for path in paths[:count]:
                with Image.open(path) as im:
                    image_id = store_pil(im.convert("RGB"), path.name, kind="upload")
                state = inspect_frame(image_id, thread_id=f"seed-{asset_class}-{path.stem}-{split[5:]}")
                written += 1

                if args.spread_days and state.get("finding_id"):
                    real = datetime.now(UTC)
                    shifted = real - timedelta(
                        days=rng.uniform(0, args.spread_days), hours=rng.uniform(0, 23)
                    )
                    findings_col().update_one(
                        {"_id": ObjectId(state["finding_id"])},
                        {
                            "$set": {
                                "timestamp": shifted,
                                "backfilled": True,
                                "observed_at": real,
                            }
                        },
                    )
                print(
                    f"  {asset_class:<20} {path.name:<12} {split:<12} -> "
                    f"{state.get('verdict'):<11} score={state.get('raw_anomaly_score')} "
                    f"routed={state.get('routed_model_name')}",
                    flush=True,
                )

    total = findings_col().count_documents({})
    backfilled = findings_col().count_documents({"backfilled": True})
    print(f"\nwrote {written} findings ({total} total, {backfilled} with shifted timestamps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

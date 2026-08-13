"""Narrow the registry to electronics inspection only.

Ambit began on powerline assets and picked up MVTec's general object set on
the way. The product is now scoped to semiconductors, PCBs and dev boards, so
every specialist outside that scope is removed -- registry document, GridFS
weights, GridFS reference image, and the findings that pointed at it.

Findings go too, deliberately. A finding is an episodic record of a routing
decision; keeping ones whose routed model no longer exists would leave the
trends view and `finding_recall_idx` describing a fleet that is not there.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from contextlib import suppress

from gridfs.errors import NoFile

from gridsight.config import DATA_ROOT
from gridsight.db.mongo import (
    datasets_col,
    findings_col,
    images_bucket,
    models_col,
    weights_bucket,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
for _n in ("httpx", "urllib3", "pymongo", "root", "timm"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("gridsight.purge")

#: Asset classes that survive: semiconductors, PCBs, and dev-board hardware.
IN_SCOPE_PREFIXES = ("pcb", "visa_pcb", "arduino", "semiconductor")
IN_SCOPE_EXACT = {"mvtec_transistor", "mvtec_cable"}


def in_scope(asset_class: str) -> bool:
    name = asset_class.lower()
    return name in IN_SCOPE_EXACT or any(p in name for p in IN_SCOPE_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Drop every non-electronics specialist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-data", action="store_true", help="leave data/<class>/ on disk")
    args = parser.parse_args()

    docs = list(
        models_col().find({}, {"name": 1, "asset_class": 1, "weights_file_id": 1, "reference_image_id": 1})
    )
    keep = [d for d in docs if in_scope(str(d["asset_class"]))]
    drop = [d for d in docs if not in_scope(str(d["asset_class"]))]

    print(f"{'ASSET CLASS':<24}{'VERDICT':<10}  MODEL")
    print("-" * 76)
    for d in sorted(docs, key=lambda x: str(x["asset_class"])):
        verdict = "KEEP" if in_scope(str(d["asset_class"])) else "DROP"
        print(f"{str(d['asset_class']):<24}{verdict:<10}  {d['name']}")
    print("-" * 76)
    print(f"keeping {len(keep)}, dropping {len(drop)}")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0

    dropped_classes = {str(d["asset_class"]) for d in drop}
    for d in drop:
        with suppress(NoFile, Exception):
            weights_bucket().delete(d["weights_file_id"])
        with suppress(NoFile, Exception):
            images_bucket().delete(d["reference_image_id"])
        models_col().delete_one({"_id": d["_id"]})
        log.info("dropped %s", d["name"])

    # Findings that routed to a model that no longer exists describe a fleet that
    # does not exist; drop them so trends and recall stay truthful.
    removed_findings = (
        findings_col().delete_many({"asset_class": {"$in": list(dropped_classes)}}).deleted_count
    )
    orphaned = (
        findings_col()
        .delete_many({"routed_model_id": {"$nin": [d["_id"] for d in keep], "$ne": None}})
        .deleted_count
    )
    datasets_removed = (
        datasets_col().delete_many({"asset_class": {"$in": list(dropped_classes)}}).deleted_count
    )

    if not args.keep_data:
        for cls in dropped_classes:
            target = DATA_ROOT / cls
            if target.is_dir():
                shutil.rmtree(target)
                log.info("removed data/%s", cls)

    print(f"\nmodels dropped        : {len(drop)}")
    print(f"findings removed      : {removed_findings + orphaned}")
    print(f"provenance rows removed: {datasets_removed}")
    print(f"models remaining      : {models_col().count_documents({})}")
    print(f"findings remaining    : {findings_col().count_documents({})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ingest six MVTec AD categories as a second registry source.

Why a second source at all: the powerline classes are visually *nested* -- a
tower crop contains insulators and cables -- so their CLIP centroids overlap and
top-1 routing sits at 79%. MVTec categories are genuinely distinct objects and
textures, which is the condition routing-by-training-distribution is supposed to
exploit. Adding them also unblocks a real `pixel_auroc`, because MVTec is the
only verified-loadable source here that ships ground-truth masks.

Category selection spans three visual regimes rather than six samples of one, so
the registry is a real test of routing rather than six near-neighbours:

  rigid object, fixed pose   bottle (dark, top-down glass rim), screw (thin
                             specular thread on black), transistor (mounted part
                             on a green board)
  organic object, free pose  hazelnut (brown shell, arbitrary rotation)
  pure texture, no object    leather (fine brown grain), tile (coarse grey marbling)

Run::

    .venv/bin/python scripts/ingest_mvtec.py
    .venv/bin/python scripts/ingest_mvtec.py --categories bottle screw --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gridsight.config import DATA_ROOT, HF_CACHE  # noqa: E402
from gridsight.db.mongo import datasets_col, ensure_collections  # noqa: E402
from gridsight.ingest.corpus import class_counts, is_populated, mask_count  # noqa: E402
from gridsight.ingest.folder_corpus import FolderCategorySpec, ingest_category  # noqa: E402

log = logging.getLogger("gridsight.ingest.mvtec")

DATASET_ID = "foersben/mvtec-ad"

#: CC BY-NC-SA 4.0 is a hard constraint, not a formality: NonCommercial blocks
#: shipping MVTec-derived weights in a commercial product, and ShareAlike
#: propagates to anything derived from them. Recorded per class so the registry
#: UI can state it rather than an operator having to remember it.
LICENSE = "CC BY-NC-SA 4.0"
LICENSE_RESTRICTION = (
    "NON-COMMERCIAL. ShareAlike propagates to derived weights. Fine for research, "
    "internal evaluation and this demo; not fine for shipping GridSight commercially "
    "with MVTec-derived specialists baked in."
)

SELECTION_RATIONALE = (
    "MVTec AD is the reference benchmark PatchCore was published on, it is the only "
    "verified-loadable Hub mirror that preserves per-category folders AND ground-truth "
    "masks, and its categories are visually disjoint -- exactly the property the "
    "powerline corpus lacks and that vector routing needs to be measurable."
)

#: (category, visual regime) -- the regime is why this category earned a slot.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("bottle", "rigid object, fixed top-down pose: a dark frame with a bright circular glass rim"),
    ("hazelnut", "organic object at arbitrary rotation: warm brown shell on black, no fixed geometry"),
    ("screw", "thin high-frequency specular metal thread on black -- the hardest MVTec class"),
    ("leather", "pure texture, no object: fine uniform brown grain filling the frame"),
    ("tile", "pure texture at the opposite scale: coarse grey/blue marbled surface"),
    ("transistor", "mounted electronic part on a green board; anomalies are positional, not just surface"),
)

SPECS: tuple[FolderCategorySpec, ...] = tuple(
    FolderCategorySpec(
        dataset_id=DATASET_ID,
        category=category,
        asset_class=f"mvtec_{category}",
        license=LICENSE,
        license_restriction=LICENSE_RESTRICTION,
        rationale=f"{SELECTION_RATIONALE} Chosen for its visual regime: {regime}.",
        notes=(
            "Canonical MVTec layout used verbatim: <category>/train/good -> train/good, "
            "<category>/test/good -> test/good, every other <category>/test/<defect_type> -> "
            "test/defect (round-robin across defect types so the 40-frame cap keeps all types), "
            "and <category>/ground_truth/<defect_type>/<stem>_mask.png -> ground_truth/, "
            "renamed to match the defect frame's stem so anomalib's Folder(mask_dir=...) pairs them."
        ),
    )
    for category, regime in CATEGORIES
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MVTec AD categories into the GridSight corpus")
    parser.add_argument("--categories", nargs="*", help="restrict to these MVTec categories")
    parser.add_argument("--force", action="store_true", help="re-extract even if a class is populated")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    wanted = set(args.categories or [])
    targets = [s for s in SPECS if not wanted or s.category in wanted]
    if not targets:
        log.error("no matching categories; known: %s", [s.category for s in SPECS])
        return 1

    ensure_collections()
    records: list[dict[str, Any]] = []
    for spec in targets:
        class_dir = DATA_ROOT / spec.asset_class
        if is_populated(class_dir) and not args.force:
            existing = datasets_col().find_one({"_id": spec.asset_class})
            if existing:
                log.info("[%s] already populated, skipping (idempotent)", spec.asset_class)
                records.append(existing)
                continue
        record = ingest_category(spec)
        # Persisted per category rather than in one batch at the end: the Hub
        # rate-limits (429) mid-run, and a batched write would throw away the
        # provenance for every category that had already succeeded, leaving
        # populated directories with no row to prove where they came from.
        datasets_col().replace_one({"_id": record["_id"]}, record, upsert=True)
        records.append(record)
    log.info("provenance rows in the `datasets` collection for %d categories", len(records))

    print("\n" + "=" * 100)
    print(
        f"{'ASSET CLASS':<22}{'TRAIN/GOOD':>11}{'TEST/GOOD':>11}{'TEST/DEFECT':>13}"
        f"{'MASKS':>8}   DEFECT TYPES"
    )
    print("=" * 100)
    failures: list[str] = []
    for rec in sorted(records, key=lambda r: str(r["asset_class"])):
        cls = str(rec["asset_class"])
        counts = class_counts(DATA_ROOT / cls)
        masks = mask_count(DATA_ROOT / cls)
        types = ",".join(rec.get("defect_types", []))
        print(
            f"{cls:<22}{counts['train_good']:>11}{counts['test_good']:>11}"
            f"{counts['test_defect']:>13}{masks:>8}   {types}"
        )
        if counts["train_good"] < 20:
            failures.append(f"{cls}: train/good={counts['train_good']} < 20")
        if counts["test_defect"] < 5:
            failures.append(f"{cls}: test/defect={counts['test_defect']} < 5")
        if masks != counts["test_defect"]:
            failures.append(f"{cls}: {masks} masks != {counts['test_defect']} defect frames")
    print("=" * 100)
    print(f"source: {DATASET_ID} | licence: {LICENSE} | {LICENSE_RESTRICTION}")

    if failures:
        print("\nFAILED ASSERTIONS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: every category has >= 20 train/good, >= 5 test/defect, and one mask per defect frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

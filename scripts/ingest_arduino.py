"""Ingest the Arduino Uno class from original photography on a fixed rig.

Every other class in this registry came off the Hugging Face Hub, and every one
of them carries somebody else's licence: VisA is CC BY 4.0, MVTec is CC BY-NC-SA
4.0 and its NonCommercial term propagates into any weights derived from it. This
class carries neither restriction, because the photographs are the authors' own.

It is also the only class that matches the beachhead ICP exactly. A prototype
shop has no published corpus, no CAD, no AOI recipe and no defect set -- it has
a board on a bench and a phone. The class is therefore normal-only by
construction: `test/defect` is created and left empty, `image_auroc` is recorded
as null rather than computed from an absent defect set, and the specialist
learns what this board looks like when nothing is wrong.

Four frames are withheld from every split and written to `data/arduino_uno/heldout/`,
which nothing in the training path reads. `scripts/validate_arduino.py` feeds
those four through the live agent, so the validation measures generalisation to
frames the specialist has never seen rather than recall of its own training set.

Run::

    .venv/bin/python scripts/ingest_arduino.py --source ~/Downloads/arudino_normal
    .venv/bin/python scripts/ingest_arduino.py --source <dir> --held-out IMG_1969.png IMG_1984.png --force
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

from gridsight.config import DATA_ROOT  # noqa: E402
from gridsight.db.mongo import datasets_col, ensure_collections  # noqa: E402
from gridsight.ingest.corpus import class_counts  # noqa: E402
from gridsight.ingest.local_corpus import (  # noqa: E402
    HELDOUT_DIRNAME,
    LocalCategorySpec,
    ingest_local_category,
    list_source_images,
)

log = logging.getLogger("gridsight.ingest.arduino")

ASSET_CLASS = "arduino_uno"
DATASET_ID = "arduino_uno_local"

LICENSE = "Original photography -- owned by the authors, no third-party restriction"
LICENSE_RESTRICTION = (
    "UNRESTRICTED. These frames were shot by the authors for this project, so the "
    "specialist derived from them carries no upstream licence at all -- neither VisA's "
    "attribution term nor MVTec's NonCommercial one. It is the only class in the "
    "registry whose weights could ship commercially with no third-party condition "
    "attached, which is also the point: the ICP's own boards are the ICP's own data."
)

RATIONALE = (
    "The registry's other classes are published corpora, which is exactly what the "
    "beachhead customer does not have. An Arduino Uno photographed on a bench rig is "
    "the honest shape of the problem: one board, one revision, a handful of known-good "
    "frames, no defect set and no barcode telling the system what it is looking at. "
    "It is also the board used in the live demo, so the class doubles as the pre-registered "
    "half of the refusal-then-cold-start arc."
)

NOTES = (
    "Local directory ingest via gridsight.ingest.local_corpus -- no Hub download. Frames "
    "are re-encoded to PNG and resized to a 512 px longest edge by the same shared "
    "resize/write path the Hub adapters use. phash_distance is 0 rather than the scraped-"
    "imagery default of 4: a fixed rig photographing one board produces frames that are "
    "meant to be near-identical, and at distance 4 the whole class reads as one duplicate "
    "and is deleted. test/good and test/defect exist but are empty -- there is no defect "
    "set, so no image AUROC is computable and none is recorded. Held-out frames are written "
    "to heldout/, which no training code reads."
)

CAPTURE_NOTES = (
    "Fixed rig, single lighting condition, phone camera (3024x4032 portrait PNG, "
    "~10.7 MB per frame at source). One Arduino Uno, one board revision, top-down single "
    "view. Frames were captured in one continuous session, so the held-out split is spread "
    "evenly across capture order rather than taken from the end: a contiguous tail would "
    "confound generalisation with whatever drifted late in the session."
)

#: Held out by default: every 4th frame in filename order, starting with the
#: second. Filename order is capture order here, so this spreads the validation
#: frames across the session instead of clustering them.
DEFAULT_HELD_OUT: tuple[str, ...] = (
    "IMG_1969.png",
    "IMG_1974.png",
    "IMG_1979.png",
    "IMG_1984.png",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest the Arduino Uno class from a local directory of photographs"
    )
    parser.add_argument(
        "--source", required=True, type=Path, help="directory holding the captured frames"
    )
    parser.add_argument(
        "--held-out",
        nargs="*",
        default=list(DEFAULT_HELD_OUT),
        help="filenames withheld from every split (default: the 4 evenly spaced frames)",
    )
    parser.add_argument("--force", action="store_true", help="re-extract even if the class is populated")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    source = args.source.expanduser().resolve()
    class_dir = DATA_ROOT / ASSET_CLASS

    spec = LocalCategorySpec(
        source_dir=source,
        asset_class=ASSET_CLASS,
        license=LICENSE,
        license_restriction=LICENSE_RESTRICTION,
        rationale=RATIONALE,
        notes=NOTES,
        capture_notes=CAPTURE_NOTES,
        held_out=tuple(args.held_out),
    )

    ensure_collections()
    existing = datasets_col().find_one({"_id": ASSET_CLASS})
    if class_counts(class_dir)["train_good"] > 0 and existing and not args.force:
        log.info("[%s] already populated, skipping (idempotent) -- pass --force to re-extract", ASSET_CLASS)
        record: dict[str, Any] = existing
    else:
        log.info("[%s] reading %d frames from %s", ASSET_CLASS, len(list_source_images(source)), source)
        record = ingest_local_category(spec)
        datasets_col().replace_one({"_id": record["_id"]}, record, upsert=True)
        log.info("[%s] provenance row written to the `datasets` collection", ASSET_CLASS)

    counts = class_counts(class_dir)
    held = sorted(p.name for p in (class_dir / HELDOUT_DIRNAME).glob("*.png"))
    split = record.get("split_by_filename", {})

    print("\n" + "=" * 104)
    print(
        f"{'ASSET CLASS':<16}{'TRAIN/GOOD':>11}{'TEST/GOOD':>11}{'TEST/DEFECT':>13}"
        f"{'HELD OUT':>10}  LICENCE"
    )
    print("=" * 104)
    print(
        f"{ASSET_CLASS:<16}{counts['train_good']:>11}{counts['test_good']:>11}"
        f"{counts['test_defect']:>13}{len(held):>10}  {LICENSE}"
    )
    print("=" * 104)
    print(f"source directory : {source}")
    print(f"dataset id       : {DATASET_ID}")
    print(f"trained on       : {', '.join(split.get('train_good', [])) or '(unchanged)'}")
    print(f"held out         : {', '.join(split.get('held_out', [])) or '(unchanged)'}")
    if record.get("dropped_as_duplicate"):
        print(f"dropped as dupe  : {', '.join(record['dropped_as_duplicate'])}")

    failures: list[str] = []
    if counts["train_good"] < 8:
        failures.append(f"train/good={counts['train_good']} < 8 -- too few frames to fit a specialist")
    if counts["test_defect"] != 0:
        failures.append(f"test/defect={counts['test_defect']} -- this class is normal-only by construction")
    if len(held) != len(args.held_out):
        failures.append(f"{len(held)} frames in heldout/ but {len(args.held_out)} were requested")
    overlap = set(split.get("train_good", [])) & set(split.get("held_out", []))
    if overlap:
        failures.append(f"held-out frames also in train/good: {sorted(overlap)}")
    if not record.get("license"):
        failures.append("provenance row records no licence")

    if failures:
        print("\nFAILED ASSERTIONS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        f"\nOK: {counts['train_good']} training frames, {len(held)} held out and absent from every "
        "split, no defect set, licence recorded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

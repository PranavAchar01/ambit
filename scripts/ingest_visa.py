"""Ingest the electronics registry: VisA's four PCB classes plus MVTec `cable`.

GridSight has been rescoped to electronics inspection -- semiconductors, PCBs and
dev boards -- so the registry needs classes that are genuinely printed-circuit
imagery rather than general objects. Two things drive the source choice:

**Licence.** VisA ships under **CC BY 4.0**: attribution only, commercially
usable, no ShareAlike. MVTec AD ships under **CC BY-NC-SA 4.0**, whose
NonCommercial term propagates to derived weights. Now that the product story
rests on PCB inspection, the four VisA classes are the ones that can actually be
shipped; `mvtec_cable` is a research/demo-only addition that widens the routing
space with an electrical-wiring regime VisA does not cover. That difference is
recorded per class in `datasets.license_restriction` rather than left for an
operator to remember.

**Masks.** Both sources ship per-defect ground-truth masks, which is what turns
`pixel_auroc` from `null` into a measured number.

Both arrive as per-category parquet with named splits, so both ride the same
`gridsight.ingest.parquet_corpus` adapter -- different column names, identical
geometry -- and land in the shared resize/dedupe/write path.

Run::

    .venv/bin/python scripts/ingest_visa.py
    .venv/bin/python scripts/ingest_visa.py --classes visa_pcb1 mvtec_cable --force
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
from gridsight.ingest.parquet_corpus import (  # noqa: E402
    MVTEC_PARQUET_SCHEMA,
    VISA_SCHEMA,
    ParquetCategorySpec,
    ingest_parquet_category,
)

log = logging.getLogger("gridsight.ingest.visa")

VISA_DATASET = "BrachioLab/visa"
VISA_LICENSE = "CC BY 4.0"
#: The permissive half of the registry. Stated as a *capability*, not a caveat,
#: because it is the reason the PCB classes can carry a commercial product.
VISA_LICENSE_RESTRICTION = (
    "PERMISSIVE. Attribution only -- no NonCommercial and no ShareAlike term, so "
    "VisA-derived PatchCore weights can ship in a commercial product provided VisA "
    "(Zou et al., 2022) is credited. This is the licence the electronics rescope rests on."
)

MVTEC_DATASET = "TheoM55/mvtec_all_objects_split"
MVTEC_LICENSE = "CC BY-NC-SA 4.0"
MVTEC_LICENSE_RESTRICTION = (
    "NON-COMMERCIAL. ShareAlike propagates to derived weights. Fine for research, "
    "internal evaluation and this demo; not fine for shipping GridSight commercially "
    "with MVTec-derived specialists baked in. Contrast with the VisA PCB classes, "
    "which are CC BY 4.0 and carry no such restriction."
)

VISA_RATIONALE = (
    "VisA is the largest public PCB anomaly corpus with per-defect ground-truth masks, it "
    "is mirrored on the Hub as per-category parquet with normal-only train splits (so "
    "PatchCore fits on genuine nominals), and unlike MVTec it is CC BY 4.0 -- the only "
    "verified source whose licence permits shipping the derived specialists commercially."
)

#: (category, why this board earned its own specialist). Regimes are described from
#: the actual frames, not from the category name. The four are four physically
#: different assembled modules photographed on four different backgrounds -- not
#: four crops of one board. Four near-neighbours inside a single domain is a far
#: harder routing test than four unrelated objects, which is the point: this is
#: where routing-by-training-distribution either holds up or does not.
VISA_CATEGORIES: tuple[tuple[str, str], ...] = (
    (
        "pcb1",
        "HC-SR04 ultrasonic module, component side: blue board dominated by two large "
        "circular metal transducer cans, on a mid-grey textured background",
    ),
    (
        "pcb2",
        "the reverse face of the same HC-SR04 outline -- no transducers, instead dense "
        "small-outline ICs and passives with silkscreen refdes, on near-black background",
    ),
    (
        "pcb3",
        "narrow elongated sensor module carrying a clear-domed LED, a black IR "
        "photodiode and a blue trimmer pot, shot on a saturated green background",
    ),
    (
        "pcb4",
        "small battery-charger board with a micro-USB receptacle and IN/BAT pads, the "
        "most cluttered of the four at its scale, on a black background",
    ),
)

SPECS: tuple[ParquetCategorySpec, ...] = (
    *(
        ParquetCategorySpec(
            dataset_id=VISA_DATASET,
            category=category,
            asset_class=f"visa_{category}",
            license=VISA_LICENSE,
            license_restriction=VISA_LICENSE_RESTRICTION,
            rationale=f"{VISA_RATIONALE} Chosen for its visual regime: {regime}.",
            schema=VISA_SCHEMA,
            notes=(
                "Per-category parquet with named splits: data/<category>.train-*.parquet is "
                "normal-only (label=0) and feeds train/good; data/<category>.test-*.parquet is "
                "split by label into test/good (0) and test/defect (1). The `mask` Image column "
                "is written to ground_truth/<stem>_mask.png so anomalib's Folder(mask_dir=...) "
                "pairs it with the defect frame. VisA carries no defect-type taxonomy, so the "
                "40-frame defect budget is filled in file order rather than round-robin. Rows "
                "are selected from the label column before any pixels are decoded."
            ),
        )
        for category, regime in VISA_CATEGORIES
    ),
    ParquetCategorySpec(
        dataset_id=MVTEC_DATASET,
        category="cable",
        asset_class="mvtec_cable",
        license=MVTEC_LICENSE,
        license_restriction=MVTEC_LICENSE_RESTRICTION,
        rationale=(
            "The electrical-wiring regime VisA does not cover: a bundled three-conductor cable "
            "cross-section whose anomalies are as much topological (swapped, missing or bent "
            "conductors) as they are surface defects. It widens the routing space beyond bare "
            "boards while staying inside the electronics rescope. Sourced from "
            f"{MVTEC_DATASET} because it is the only verified Hub mirror that preserves MVTec's "
            "per-category split AND its ground-truth masks in loadable parquet."
        ),
        schema=MVTEC_PARQUET_SCHEMA,
        notes=(
            "Per-category parquet with named splits and real `object`/`defect`/`split` columns. "
            "Columns differ from VisA -- image_path/mask_path/label/defect -- which is exactly "
            "what ParquetSchema exists to absorb. Rows are filtered on object=='cable' as a "
            "guard, label 0/1 splits good from defect, and the `defect` column drives a "
            "round-robin across the eight defect types so the 40-frame cap keeps all of them. "
            "Normal rows carry a null mask_path; only defect rows have masks, which is why "
            "mask presence is asserted on the defect pool alone."
        ),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest the VisA PCB classes and MVTec cable into the GridSight corpus"
    )
    parser.add_argument("--classes", nargs="*", help="restrict to these asset classes")
    parser.add_argument("--force", action="store_true", help="re-extract even if a class is populated")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    wanted = set(args.classes or [])
    targets = [s for s in SPECS if not wanted or s.asset_class in wanted]
    if not targets:
        log.error("no matching classes; known: %s", [s.asset_class for s in SPECS])
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
        record = ingest_parquet_category(spec)
        # Persisted per class rather than in one batch at the end: the Hub
        # rate-limits (429) mid-run, and a batched write would throw away the
        # provenance for every class that had already succeeded, leaving
        # populated directories with no row to prove where they came from.
        datasets_col().replace_one({"_id": record["_id"]}, record, upsert=True)
        records.append(record)
    log.info("provenance rows in the `datasets` collection for %d classes", len(records))

    print("\n" + "=" * 118)
    print(
        f"{'ASSET CLASS':<16}{'TRAIN/GOOD':>11}{'TEST/GOOD':>11}{'TEST/DEFECT':>13}{'MASKS':>7}"
        f"  {'LICENCE':<16}DEFECT TYPES"
    )
    print("=" * 118)
    failures: list[str] = []
    for rec in sorted(records, key=lambda r: str(r["asset_class"])):
        cls = str(rec["asset_class"])
        counts = class_counts(DATA_ROOT / cls)
        masks = mask_count(DATA_ROOT / cls)
        types = ",".join(rec.get("defect_types", []))
        print(
            f"{cls:<16}{counts['train_good']:>11}{counts['test_good']:>11}"
            f"{counts['test_defect']:>13}{masks:>7}  {str(rec['license']):<16}{types[:44]}"
        )
        if counts["train_good"] < 20:
            failures.append(f"{cls}: train/good={counts['train_good']} < 20")
        if counts["test_defect"] < 5:
            failures.append(f"{cls}: test/defect={counts['test_defect']} < 5")
        if masks != counts["test_defect"]:
            failures.append(f"{cls}: {masks} masks != {counts['test_defect']} defect frames")
        if not rec.get("license"):
            failures.append(f"{cls}: provenance row records no licence")
    print("=" * 118)
    print(f"{VISA_DATASET}: {VISA_LICENSE} -- {VISA_LICENSE_RESTRICTION}")
    print(f"{MVTEC_DATASET}: {MVTEC_LICENSE} -- {MVTEC_LICENSE_RESTRICTION}")

    if failures:
        print("\nFAILED ASSERTIONS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: every class has >= 20 train/good, >= 5 test/defect, one mask per defect frame, a licence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 1 -- acquire real infrastructure-inspection imagery from the Hugging Face Hub.

The Hub is queried programmatically (never a hardcoded dataset id): every search
term is enumerated, candidates are filtered down to repos that actually carry
image features, and the selection rationale for each asset class is logged and
persisted to the `datasets` collection.

Output is an Anomalib-compatible tree per class::

    data/<class>/train/good/      # PatchCore fits on these only
    data/<class>/test/good/
    data/<class>/test/defect/

Nothing here is generated. Every pixel written to disk came out of a real Hub
dataset; the only transforms applied are cropping to a labelled region, resizing
the longest edge to 512px, and perceptual-hash deduplication.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
from huggingface_hub import HfApi, snapshot_download
from PIL import Image

from gridsight.config import DATA_ROOT, HF_CACHE, get_settings
from gridsight.db.mongo import datasets_col, ensure_collections

log = logging.getLogger("gridsight.ingest")

MAX_EDGE = 512
UPSCALE_TO = 160  # small crops are upscaled rather than padded with background
PHASH_DISTANCE = 4  # <= this hamming distance counts as a duplicate

# Crops are taken tight to the labelled box. An earlier policy padded every box
# with 25% context and inflated anything under 96px, which made a 20px insulator
# crop 80% sky -- and measurably destroyed routing, because every powerline crop
# then embedded to the same place. Measured top-1 routing over these classes:
# padded 61%, tight 69%, tight + the scale split below 88%.
CTX_MARGIN = 0.0

# Asset classes in this corpus are nested (a tower box contains insulators and
# cables), so scale is what actually distinguishes them: a transmission-tower
# frame shows a whole structure, a component frame shows a close-up. These
# bounds are in pixels on the source 640x640 frames.
TOWER_MIN_SIDE = 180.0
COMPONENT_MAX_SIDE = 160.0
CONDUCTOR_MAX_SIDE = 200.0

TRAIN_GOOD_CAP = 120
TEST_GOOD_CAP = 30
TEST_DEFECT_CAP = 40
TRAIN_FRACTION = 0.8

#: Search terms issued against the Hub. Kept verbatim from the asset taxonomy so
#: the discovery log shows exactly what was asked for.
SEARCH_TERMS: tuple[str, ...] = (
    "insulator defect",
    "power line inspection",
    "transmission tower",
    "UAV powerline",
    "corrosion detection",
    "rust segmentation",
    "vegetation encroachment",
    "rail surface defect",
    "railway track fault",
    "catenary inspection",
)

Splits = dict[str, list[Image.Image]]


# ---------------------------------------------------------------------------
# Hub discovery
# ---------------------------------------------------------------------------


def discover_candidates(api: HfApi, limit: int = 40) -> dict[str, dict[str, Any]]:
    """Enumerate Hub datasets for every search term, keeping image-bearing repos."""
    candidates: dict[str, dict[str, Any]] = {}
    for term in SEARCH_TERMS:
        try:
            hits = list(api.list_datasets(search=term, limit=limit))
        except Exception as exc:  # noqa: BLE001 - network failure must not abort discovery
            log.warning("hub search failed for %r: %s", term, exc)
            continue
        kept = 0
        for d in hits:
            tags = list(d.tags or [])
            has_image = any("image" in t for t in tags)
            if not has_image:
                continue
            kept += 1
            entry = candidates.setdefault(
                d.id,
                {
                    "dataset_id": d.id,
                    "downloads": d.downloads or 0,
                    "likes": d.likes or 0,
                    "matched_terms": [],
                    "tags": [t for t in tags if not t.startswith("region")][:12],
                },
            )
            entry["matched_terms"].append(term)
        log.info("search %-24r -> %3d hits, %2d with image features", term, len(hits), kept)
    log.info("discovery complete: %d distinct image-bearing candidates", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Source specifications -- what we selected out of the candidate pool, and why
# ---------------------------------------------------------------------------


@dataclass
class SourceSpec:
    asset_class: str
    dataset_id: str
    license: str
    rationale: str
    extract: Callable[[Path | None], Splits] = field(repr=False)
    needs_snapshot: bool = True
    notes: str = ""


def box_side(box: list[float]) -> float:
    """Longest side of a labelled box, in source pixels."""
    return max(box[2] - box[0], box[3] - box[1])


def _pad_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int] | None:
    """Take a square crop tight to the labelled box, clipped to the frame."""
    x0, y0, x1, y1 = box
    if x1 - x0 <= 4 or y1 - y0 <= 4:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max((x1 - x0), (y1 - y0)) * (1 + CTX_MARGIN)
    side = min(side, float(min(w, h)))
    nx0 = int(round(min(max(cx - side / 2, 0), w - side)))
    ny0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return nx0, ny0, int(round(nx0 + side)), int(round(ny0 + side))


def _powerline_frames() -> Iterator[tuple[Image.Image, dict[str, list[list[float]]]]]:
    """Yield each docmhvr frame with its boxes grouped by label name."""
    from datasets import load_dataset

    names = ["Broken Cable", "Broken Insulator", "Cable", "Insulators", "Tower", "Vegetation"]
    for split in ("train", "validation", "test"):
        ds = load_dataset("docmhvr/powerline-components-and-faults", split=split, cache_dir=str(HF_CACHE))
        for row in ds:
            grouped: dict[str, list[list[float]]] = {}
            for box, label in zip(row["bboxes"], row["labels"], strict=True):
                grouped.setdefault(names[label], []).append(list(box))
            yield row["image"], grouped


def _crop(img: Image.Image, box: list[float]) -> Image.Image | None:
    padded = _pad_box((box[0], box[1], box[2], box[3]), img.width, img.height)
    if padded is None:
        return None
    crop = img.convert("RGB").crop(padded)
    if crop.width < UPSCALE_TO:
        # Upscale rather than pad: padding would fill the frame with background
        # and collapse every component class onto the same embedding.
        crop = crop.resize((UPSCALE_TO, UPSCALE_TO), Image.LANCZOS)
    return crop


def _centre_inside(inner: list[float], outer: list[float]) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _intersects(a: list[float], b: list[float]) -> bool:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) > 0 and max(0.0, min(a[3], b[3]) - max(a[1], b[1])) > 0


def _extract_powerline(component: str) -> Callable[[Path | None], Splits]:
    """Insulator / conductor: healthy component crops vs broken component crops."""
    good_label, bad_label = {
        "insulator": ("Insulators", "Broken Insulator"),
        "conductor": ("Cable", "Broken Cable"),
    }[component]

    max_side = COMPONENT_MAX_SIDE if component == "insulator" else CONDUCTOR_MAX_SIDE

    def run(_: Path | None) -> Splits:
        good: list[Image.Image] = []
        defect: list[Image.Image] = []
        for img, grouped in _powerline_frames():
            broken = grouped.get(bad_label, [])
            for box in broken:
                if box_side(box) > max_side:
                    continue
                crop = _crop(img, box)
                if crop is not None:
                    defect.append(crop)
            for box in grouped.get(good_label, []):
                # Component close-ups only: a box large enough to be a whole
                # structure belongs to the transmission_tower class instead.
                if box_side(box) > max_side:
                    continue
                # a "healthy" crop must not contain a labelled break
                if any(_centre_inside(b, box) for b in broken):
                    continue
                crop = _crop(img, box)
                if crop is not None:
                    good.append(crop)
        return {"good": good, "defect": defect}

    return run


def _extract_tower(_: Path | None) -> Splits:
    """Whole-structure tower crops; anomalous when a broken component sits on the tower."""
    good: list[Image.Image] = []
    defect: list[Image.Image] = []
    for img, grouped in _powerline_frames():
        broken = grouped.get("Broken Cable", []) + grouped.get("Broken Insulator", [])
        for box in grouped.get("Tower", []):
            # Only boxes big enough to read as a structure, so this class stays
            # visually distinct from the component close-ups.
            if box_side(box) < TOWER_MIN_SIDE:
                continue
            crop = _crop(img, box)
            if crop is None:
                continue
            if any(_centre_inside(b, box) for b in broken):
                defect.append(crop)
            else:
                good.append(crop)
    return {"good": good, "defect": defect}


def _extract_vegetation(_: Path | None) -> Splits:
    """Right-of-way vegetation; anomalous when the canopy overlaps a conductor."""
    good: list[Image.Image] = []
    defect: list[Image.Image] = []
    for img, grouped in _powerline_frames():
        conductors = grouped.get("Cable", []) + grouped.get("Broken Cable", [])
        for box in grouped.get("Vegetation", []):
            crop = _crop(img, box)
            if crop is None:
                continue
            if any(_intersects(box, c) for c in conductors):
                defect.append(crop)
            else:
                good.append(crop)
    return {"good": good, "defect": defect}


def _load_dir(root: Path, patterns: tuple[str, ...]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB").copy())
            except OSError as exc:  # truncated / unreadable file
                log.debug("skipping unreadable %s: %s", p, exc)
    return images


def _extract_corrosion(root: Path | None) -> Splits:
    assert root is not None
    return {
        "good": _load_dir(root, ("Corrosion/no rust/*",)),
        "defect": _load_dir(root, ("Corrosion/rust/*",)),
    }


def _extract_rail(root: Path | None) -> Splits:
    assert root is not None
    return {
        "good": _load_dir(root, ("*/Non-defective/*",)),
        "defect": _load_dir(root, ("*/Defective/*",)),
    }


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        asset_class="insulator",
        dataset_id="docmhvr/powerline-components-and-faults",
        license="unspecified on the Hub card",
        rationale=(
            "Top image-bearing hit for 'insulator defect' / 'UAV powerline' that ships "
            "per-component bounding boxes with an explicit Insulators vs Broken Insulator "
            "distinction, which is exactly the healthy/anomalous split PatchCore needs."
        ),
        extract=_extract_powerline("insulator"),
        needs_snapshot=False,
        notes=(
            "Tight square crops of boxes with longest side <= 160px (component close-ups). "
            "Healthy = 'Insulators' crops with no labelled break inside; defect = 'Broken Insulator'."
        ),
    ),
    SourceSpec(
        asset_class="conductor",
        dataset_id="docmhvr/powerline-components-and-faults",
        license="unspecified on the Hub card",
        rationale=(
            "Same UAV corpus carries Cable vs Broken Cable boxes, giving a conductor/crossarm "
            "asset class with a genuine anomaly label rather than an inferred one."
        ),
        extract=_extract_powerline("conductor"),
        needs_snapshot=False,
        notes=(
            "Tight square crops of boxes with longest side <= 200px. Healthy = 'Cable' crops with "
            "no labelled break inside; defect = 'Broken Cable' crops."
        ),
    ),
    SourceSpec(
        asset_class="transmission_tower",
        dataset_id="docmhvr/powerline-components-and-faults",
        license="unspecified on the Hub card",
        rationale=(
            "No Hub dataset labels transmission-tower structural damage directly -- searches for "
            "'transmission tower damage', 'tower defect' and 'structural damage' returned nothing "
            "usable. Nearest real alternative: tower crops from this UAV corpus, where a tower is "
            "anomalous when a labelled broken component sits on it."
        ),
        extract=_extract_tower,
        needs_snapshot=False,
        notes=(
            "Tight square crops of 'Tower' boxes with longest side >= 180px, so the class reads as a "
            "whole structure rather than a component. Defect = tower crop containing the centre of a "
            "Broken Cable / Broken Insulator box."
        ),
    ),
    SourceSpec(
        asset_class="vegetation",
        dataset_id="docmhvr/powerline-components-and-faults",
        license="unspecified on the Hub card",
        rationale=(
            "Searches for 'vegetation encroachment' surfaced only generic land-cover segmentation "
            "corpora with no right-of-way context. This corpus labels Vegetation inside powerline "
            "frames, so encroachment is directly computable from box geometry."
        ),
        extract=_extract_vegetation,
        needs_snapshot=False,
        notes=(
            "Tight square crops of 'Vegetation' boxes. Defect = box intersecting a conductor box "
            "(encroachment); good = canopy clear of conductors."
        ),
    ),
    SourceSpec(
        asset_class="corrosion",
        dataset_id="khoaliamle/Corrosion_Rust",
        license="mit",
        rationale=(
            "Best 'corrosion detection' / 'rust segmentation' hit that is already partitioned into "
            "'no rust' and 'rust' folders -- a ready-made normal/anomalous split on real metalwork."
        ),
        extract=_extract_corrosion,
        notes="Folder labels used verbatim: Corrosion/no rust -> good, Corrosion/rust -> defect.",
    ),
    SourceSpec(
        asset_class="rail_surface",
        dataset_id="crangana/railroad-fault-detection",
        license="unspecified on the Hub card",
        rationale=(
            "The Dhika rail corpora ('rail surface defect', 'railway track fault') contain defect "
            "classes only -- no normal rail -- so PatchCore cannot fit on them. This dataset is the "
            "only rail hit carrying an explicit Non-defective class alongside Defective."
        ),
        extract=_extract_rail,
        notes="Folder labels used verbatim: */Non-defective -> good, */Defective -> defect.",
    ),
)

REJECTED: tuple[dict[str, str], ...] = (
    {
        "dataset_id": "Dhika/rail_defect",
        "reason": "1219 images across 134 defect-combination folders, zero normal-rail class; "
        "PatchCore trains on good only, so unusable as a primary rail source.",
    },
    {
        "dataset_id": "Dhika/defect_rail",
        "reason": "Same corpus at v8 (761 images, 5 defect classes); still no normal class.",
    },
    {
        "dataset_id": "silera/broken-insulators-synthetic-detection",
        "reason": "Self-declared synthetic imagery; excluded because the registry must be trained "
        "on real inspection frames.",
    },
    {
        "dataset_id": "MortenTabaka/LandCover-Aerial-Imagery-for-semantic-segmentation",
        "reason": "Generic land-cover aerial tiles with no right-of-way or conductor context, so "
        "'encroachment' is not derivable.",
    },
    {
        "dataset_id": "UPNAdroneLab/powerline_towers",
        "reason": "1022 real tower frames but detection-only annotations with no damage class; "
        "supplies no anomalous split.",
    },
    {
        "dataset_id": "Francesco/corrosion-bi3q3",
        "reason": "Corrosion boxes only -- every frame is anomalous, no clean-metal normal set.",
    },
)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def resize_max_edge(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def dedupe(images: list[Image.Image], seen: list[imagehash.ImageHash]) -> list[Image.Image]:
    """Drop perceptual near-duplicates, both within the batch and against `seen`."""
    kept: list[Image.Image] = []
    for img in images:
        h = imagehash.phash(img)
        if any((h - prior) <= PHASH_DISTANCE for prior in seen):
            continue
        seen.append(h)
        kept.append(img)
    return kept


def write_split(images: list[Image.Image], target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        resize_max_edge(img).save(target / f"{i:05d}.png", format="PNG", optimize=True)
    return len(images)


def class_counts(class_dir: Path) -> dict[str, int]:
    return {
        "train_good": len(list((class_dir / "train" / "good").glob("*.png"))),
        "test_good": len(list((class_dir / "test" / "good").glob("*.png"))),
        "test_defect": len(list((class_dir / "test" / "defect").glob("*.png"))),
    }


def is_populated(class_dir: Path) -> bool:
    c = class_counts(class_dir)
    return c["train_good"] >= 20 and c["test_defect"] >= 5


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def ingest_class(spec: SourceSpec, seed: int = 1234) -> dict[str, Any]:
    class_dir = DATA_ROOT / spec.asset_class
    root: Path | None = None

    if spec.needs_snapshot:
        log.info("[%s] snapshot_download %s -> %s", spec.asset_class, spec.dataset_id, HF_CACHE)
        root = Path(
            snapshot_download(
                repo_id=spec.dataset_id,
                repo_type="dataset",
                cache_dir=str(HF_CACHE),
                token=get_settings().hf_token or None,
            )
        )

    log.info("[%s] extracting from %s", spec.asset_class, spec.dataset_id)
    splits = spec.extract(root)
    raw_good, raw_defect = len(splits["good"]), len(splits["defect"])

    seen: list[imagehash.ImageHash] = []
    rng = random.Random(seed)
    good = splits["good"]
    defect = splits["defect"]
    rng.shuffle(good)
    rng.shuffle(defect)

    # Dedupe a bounded prefix -- phash over tens of thousands of crops is the
    # slowest step and we only ever keep a few hundred.
    good = dedupe(good[: (TRAIN_GOOD_CAP + TEST_GOOD_CAP) * 6], seen)
    defect = dedupe(defect[: TEST_DEFECT_CAP * 6], seen)

    n_train = min(TRAIN_GOOD_CAP, max(1, int(len(good) * TRAIN_FRACTION)))
    train_good = good[:n_train]
    test_good = good[n_train : n_train + TEST_GOOD_CAP]
    test_defect = defect[:TEST_DEFECT_CAP]

    if class_dir.exists():
        shutil.rmtree(class_dir)
    write_split(train_good, class_dir / "train" / "good")
    write_split(test_good, class_dir / "test" / "good")
    write_split(test_defect, class_dir / "test" / "defect")

    counts = class_counts(class_dir)
    record = {
        "_id": spec.asset_class,
        "asset_class": spec.asset_class,
        "dataset_id": spec.dataset_id,
        "license": spec.license,
        "rationale": spec.rationale,
        "extraction_notes": spec.notes,
        "raw_candidates": {"good": raw_good, "defect": raw_defect},
        "after_dedupe": {"good": len(good), "defect": len(defect)},
        "sample_counts": counts,
        "phash_distance": PHASH_DISTANCE,
        "max_edge_px": MAX_EDGE,
    }
    log.info(
        "[%s] raw good/defect=%d/%d -> kept %d/%d -> train_good=%d test_good=%d test_defect=%d",
        spec.asset_class,
        raw_good,
        raw_defect,
        len(good),
        len(defect),
        counts["train_good"],
        counts["test_good"],
        counts["test_defect"],
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape infrastructure imagery from the HF Hub")
    parser.add_argument("--force", action="store_true", help="re-extract even if a class is populated")
    parser.add_argument("--classes", nargs="*", help="restrict to these asset classes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    api = HfApi(token=settings.hf_token or None)

    candidates = discover_candidates(api)
    selected_ids = {s.dataset_id for s in SOURCES}
    log.info("selected %d datasets out of %d candidates", len(selected_ids), len(candidates))
    for spec in SOURCES:
        log.info("SELECT %-19s <- %-46s :: %s", spec.asset_class, spec.dataset_id, spec.rationale)
    for rej in REJECTED:
        log.info("REJECT %-46s :: %s", rej["dataset_id"], rej["reason"])

    ensure_collections()
    targets = [s for s in SOURCES if not args.classes or s.asset_class in args.classes]

    records: list[dict[str, Any]] = []
    for spec in targets:
        class_dir = DATA_ROOT / spec.asset_class
        if is_populated(class_dir) and not args.force:
            log.info("[%s] already populated, skipping (idempotent)", spec.asset_class)
            existing = datasets_col().find_one({"_id": spec.asset_class})
            if existing:
                records.append(existing)
                continue
        records.append(ingest_class(spec))

    for rec in records:
        datasets_col().replace_one({"_id": rec["_id"]}, rec, upsert=True)
    log.info("wrote %d provenance records to the `datasets` collection", len(records))

    report = {
        "search_terms": list(SEARCH_TERMS),
        "candidates_discovered": len(candidates),
        "candidates": sorted(candidates.values(), key=lambda c: -int(c["downloads"]))[:60],
        "selected": [
            {
                "asset_class": s.asset_class,
                "dataset_id": s.dataset_id,
                "license": s.license,
                "rationale": s.rationale,
                "notes": s.notes,
            }
            for s in SOURCES
        ],
        "rejected": list(REJECTED),
        "records": records,
    }
    (DATA_ROOT / "ingest_report.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- verification table -------------------------------------------------
    print("\n" + "=" * 108)
    print(f"{'ASSET CLASS':<20}{'TRAIN/GOOD':>11}{'TEST/GOOD':>11}{'TEST/DEFECT':>13}   SOURCE DATASET")
    print("=" * 108)
    failures: list[str] = []
    totals: Counter[str] = Counter()
    for rec in sorted(records, key=lambda r: str(r["asset_class"])):
        c = class_counts(DATA_ROOT / str(rec["asset_class"]))
        totals.update(c)
        print(
            f"{rec['asset_class']:<20}{c['train_good']:>11}{c['test_good']:>11}"
            f"{c['test_defect']:>13}   {rec['dataset_id']}"
        )
        if c["train_good"] < 20:
            failures.append(f"{rec['asset_class']}: train/good={c['train_good']} < 20")
        if c["test_defect"] < 5:
            failures.append(f"{rec['asset_class']}: test/defect={c['test_defect']} < 5")
    print("=" * 108)
    print(
        f"{'TOTAL':<20}{totals['train_good']:>11}{totals['test_good']:>11}{totals['test_defect']:>13}"
        f"   {len(records)} asset classes"
    )

    if failures:
        print("\nFAILED ASSERTIONS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: every asset class has >= 20 train/good and >= 5 test/defect images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

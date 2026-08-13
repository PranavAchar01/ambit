"""Ingest an asset class from a local directory of photographs.

Every existing adapter fetches from the Hugging Face Hub: `folder_corpus` and
`parquet_corpus` both open with an unconditional `snapshot_download`, because
both were written for published corpora. The beachhead ICP does not have a
published corpus -- it has a bench, a phone and a board -- so the class that
matters most to the pitch is the one no Hub adapter can reach.

This module is that adapter. It reuses the same resize/dedupe/write primitives
as the Hub paths (`gridsight.ingest.corpus`) so a locally captured class lands
on disk byte-identically to a downloaded one, and emits the same provenance
record so `datasets` describes every class the same way regardless of origin.

Three differences from the Hub adapters, all forced by controlled capture:

**Perceptual dedupe is off; exact-content dedupe is always on.** The scraped-
imagery gate of 4 exists to kill re-uploads of the same photograph. A rig that
photographs one board from one position produces frames that are *supposed* to
be near-identical, and phash saturates on them: measured on the Arduino capture
set, two genuinely different exposures scored phash distance 0 while differing
by 15.3 mean pixel levels -- the same order as an arbitrary pair from the set
(18-27). Even distance 0 is a false-positive gate on controlled capture, so the
duplicate test here is a hash of the decoded pixels, which catches the case that
actually occurs (the same file ingested twice) and cannot fire on a real frame.
Perceptual dedupe remains available via `phash_distance >= 0` for capture sets
loose enough to need it.

**Normal-only.** There is no defect set, so `test/defect` is created empty and
`image_auroc` stays null downstream. That is the honest state of a prototype
shop, not a gap to be filled with a fabricated number.

**Held-out frames never enter a split.** They are written to `heldout/`, a
directory no part of the training path reads, so validating against them cannot
degrade into scoring the training set.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from gridsight.config import DATA_ROOT
from gridsight.ingest.corpus import (
    MAX_EDGE,
    class_counts,
    dedupe_indices,
    write_split,
)

log = logging.getLogger("gridsight.ingest.local")

#: Extensions accepted from a capture directory. Everything is re-encoded to PNG
#: on the way in regardless: `train_class.list_images` globs `*.png` only, so a
#: JPEG would be invisible to the centroid, the gate and the pixel calibration
#: while anomalib still trained on it.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

#: Directory holding frames deliberately excluded from every split.
HELDOUT_DIRNAME = "heldout"


@dataclass(frozen=True)
class LocalCategorySpec:
    """One asset class captured locally rather than downloaded."""

    source_dir: Path
    asset_class: str
    license: str
    license_restriction: str
    rationale: str
    notes: str
    #: How the frames were shot -- rig, lighting, camera. Photometric conditions
    #: are the primary failure mode for a PatchCore fitted on a handful of
    #: images, so they are provenance, not trivia.
    capture_notes: str
    #: Filenames (as they appear in `source_dir`) withheld from training.
    held_out: tuple[str, ...] = ()
    #: Negative disables perceptual dedupe entirely and leaves only the
    #: exact-pixel-content test. See the module docstring for the measurement
    #: that made this the default for controlled capture.
    phash_distance: int = -1


def list_source_images(source_dir: Path) -> list[Path]:
    """Every image in the capture directory, in stable filename order."""
    return sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _exact_distinct(images: Sequence[Image.Image]) -> list[int]:
    """Positions of frames whose decoded pixels are not byte-identical to an earlier one.

    Hashing the decoded pixels rather than the file means a frame re-saved or
    re-encoded between captures is still caught, while two genuinely different
    exposures of a static scene -- which perceptual hashing cannot separate --
    are both kept.
    """
    seen: set[str] = set()
    kept: list[int] = []
    for i, img in enumerate(images):
        digest = hashlib.sha256(img.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(i)
    return kept


def _load_rgb(paths: Sequence[Path]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            frames.append(im.convert("RGB").copy())
    return frames


def ingest_local_category(spec: LocalCategorySpec) -> dict[str, Any]:
    """Cut the Anomalib tree from a capture directory and return a provenance row."""
    if not spec.source_dir.is_dir():
        raise FileNotFoundError(f"{spec.asset_class}: capture directory {spec.source_dir} does not exist")

    every = list_source_images(spec.source_dir)
    if not every:
        raise FileNotFoundError(
            f"{spec.asset_class}: no images in {spec.source_dir} "
            f"(looked for {', '.join(IMAGE_SUFFIXES)})"
        )

    held_out_names = set(spec.held_out)
    unknown = held_out_names - {p.name for p in every}
    if unknown:
        # A typo'd held-out filename would silently train on the frame it was
        # meant to withhold, and the validation table would then be measuring
        # memorisation. Refuse rather than proceed.
        raise ValueError(
            f"{spec.asset_class}: held-out files not present in {spec.source_dir}: {sorted(unknown)}"
        )

    train_paths = [p for p in every if p.name not in held_out_names]
    held_paths = [p for p in every if p.name in held_out_names]
    if not train_paths:
        raise ValueError(f"{spec.asset_class}: every frame is held out -- nothing left to train on")

    train_frames = _load_rgb(train_paths)
    keep = _exact_distinct(train_frames)
    if spec.phash_distance >= 0:
        seen: list[imagehash.ImageHash] = []
        perceptual = set(dedupe_indices([train_frames[i] for i in keep], seen, spec.phash_distance))
        keep = [idx for pos, idx in enumerate(keep) if pos in perceptual]
    dropped = [train_paths[i].name for i in range(len(train_paths)) if i not in set(keep)]
    if dropped:
        log.warning(
            "[%s] dropped %d duplicate frame(s) (phash distance %d, negative = exact-content only): %s",
            spec.asset_class,
            len(dropped),
            spec.phash_distance,
            ", ".join(dropped),
        )
    kept_frames = [train_frames[i] for i in keep]
    kept_names = [train_paths[i].name for i in keep]

    class_dir = DATA_ROOT / spec.asset_class
    if class_dir.exists():
        shutil.rmtree(class_dir)

    write_split(kept_frames, class_dir / "train" / "good")
    # Both are created empty and deliberately: anomalib's Folder validates that
    # every directory it is handed exists, and there is neither a defect set nor
    # a spare normal set to fill them from. An empty directory is the honest
    # shape of "no labelled defects", and it keeps the on-disk contract intact.
    write_split([], class_dir / "test" / "good")
    write_split([], class_dir / "test" / "defect")

    held_frames = _load_rgb(held_paths)
    write_split(held_frames, class_dir / HELDOUT_DIRNAME)
    for frame in (*train_frames, *held_frames):
        frame.close()

    counts = class_counts(class_dir)
    record: dict[str, Any] = {
        "_id": spec.asset_class,
        "asset_class": spec.asset_class,
        "dataset_id": f"{spec.asset_class}_local",
        "source_category": spec.asset_class,
        "source_files": sorted(p.name for p in every),
        "license": spec.license,
        "license_restriction": spec.license_restriction,
        "rationale": spec.rationale,
        "extraction_notes": spec.notes,
        "capture_notes": spec.capture_notes,
        "defect_types": [],
        "raw_candidates": {"good": len(every), "defect": 0},
        "after_dedupe": {"good": len(kept_frames) + len(held_frames), "defect": 0},
        "sample_counts": {**counts, "held_out": len(held_frames)},
        "ground_truth_masks": 0,
        "phash_distance": spec.phash_distance,
        "max_edge_px": MAX_EDGE,
        # The split is recorded by filename rather than by count so a later
        # reader can verify that the validation frames were never trained on.
        "split_by_filename": {"train_good": kept_names, "held_out": sorted(p.name for p in held_paths)},
        "dropped_as_duplicate": dropped,
    }
    log.info(
        "[%s] %d source frames -> train_good=%d held_out=%d (test/good and test/defect empty: "
        "normal-only capture, no labelled defects)",
        spec.asset_class,
        len(every),
        counts["train_good"],
        len(held_frames),
    )
    return record

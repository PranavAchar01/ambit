"""Source-agnostic corpus mechanics shared by every ingest adapter.

`hf_scrape` was written against one corpus and owned both halves of the job:
*how to find the pixels* (bounding-box geometry, powerline label names) and
*what to do with them once found* (dedupe, resize, cap, write an Anomalib tree).
Only the first half is corpus-specific. This module is the second half, lifted
out verbatim so a second source -- MVTec AD, which arrives as folders rather
than boxes -- reuses the identical resize/dedupe/write path instead of growing
a parallel one that could silently drift from it.

The on-disk contract every adapter must produce::

    data/<class>/train/good/      # PatchCore fits on these only
    data/<class>/test/good/
    data/<class>/test/defect/
    data/<class>/ground_truth/    # optional; only sources that ship masks
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imagehash
from PIL import Image

MAX_EDGE = 512
PHASH_DISTANCE = 4  # <= this hamming distance counts as a duplicate

TRAIN_GOOD_CAP = 120
TEST_GOOD_CAP = 30
TEST_DEFECT_CAP = 40
TRAIN_FRACTION = 0.8

#: Ground-truth masks are written next to the defect frames under this name, so
#: anomalib's `Folder(mask_dir=...)` pairs them positionally by sorted stem.
MASK_SUFFIX = "_mask"


def resize_max_edge(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def dedupe_indices(
    images: Sequence[Image.Image], seen: list[imagehash.ImageHash], distance: int = PHASH_DISTANCE
) -> list[int]:
    """Positions of the perceptually distinct frames, against `seen` and each other.

    Index-returning rather than image-returning because a masked corpus has to
    keep each surviving frame welded to its own mask; dropping an image without
    dropping its mask would misalign every pair after it.

    `distance` is per-source rather than global: scraped web imagery needs a loose
    gate to kill re-uploads, while a controlled-capture corpus photographs the
    same part in the same rig every time and the same gate would erase the class.
    """
    kept: list[int] = []
    for i, img in enumerate(images):
        h = imagehash.phash(img)
        if any((h - prior) <= distance for prior in seen):
            continue
        seen.append(h)
        kept.append(i)
    return kept


def dedupe(images: list[Image.Image], seen: list[imagehash.ImageHash]) -> list[Image.Image]:
    """Drop perceptual near-duplicates, both within the batch and against `seen`."""
    return [images[i] for i in dedupe_indices(images, seen)]


def write_split(images: list[Image.Image], target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        resize_max_edge(img).save(target / f"{i:05d}.png", format="PNG", optimize=True)
    return len(images)


def write_masked_split(
    pairs: Sequence[tuple[Image.Image, Image.Image]], image_dir: Path, mask_dir: Path
) -> int:
    """Write defect frames and their ground-truth masks under matching stems.

    Each mask is resampled with NEAREST to the *post-resize* size of its own
    frame: anomalib loads image and mask as one sample, so a mask that is even
    one pixel off would be resized against a different aspect ratio and the
    pixel AUROC would be measured against a misaligned target.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, mask) in enumerate(pairs):
        resized = resize_max_edge(img)
        resized.save(image_dir / f"{i:05d}.png", format="PNG", optimize=True)
        binary = mask.convert("L").resize(resized.size, Image.NEAREST).point(lambda v: 255 if v else 0)
        binary.save(mask_dir / f"{i:05d}{MASK_SUFFIX}.png", format="PNG", optimize=True)
    return len(pairs)


def class_counts(class_dir: Path) -> dict[str, int]:
    return {
        "train_good": len(list((class_dir / "train" / "good").glob("*.png"))),
        "test_good": len(list((class_dir / "test" / "good").glob("*.png"))),
        "test_defect": len(list((class_dir / "test" / "defect").glob("*.png"))),
    }


def mask_count(class_dir: Path) -> int:
    return len(list((class_dir / "ground_truth").glob("*.png")))


def is_populated(class_dir: Path) -> bool:
    c = class_counts(class_dir)
    return c["train_good"] >= 20 and c["test_defect"] >= 5

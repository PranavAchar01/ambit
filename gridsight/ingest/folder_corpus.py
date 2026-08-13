"""Ingest adapter for folder-structured, multi-category anomaly corpora.

`hf_scrape` derives its splits from bounding-box geometry over one flat corpus.
A second family of datasets -- MVTec AD and everything modelled on it -- already
ships the split as directories, one subtree per category, and adds the thing the
powerline corpus cannot supply: **per-defect ground-truth masks**. Those masks
are what turns `pixel_auroc` from `null` into a measured number.

The seam is `FolderLayout`: a declaration of where good frames, defect frames and
masks live inside a repo, with `{category}` interpolated per class. Anything that
lays its files out that way is ingestible without touching the powerline
extractors, and every frame still travels through the shared
`gridsight.ingest.corpus` resize/dedupe/write path.
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
from huggingface_hub import snapshot_download
from PIL import Image

from gridsight.config import DATA_ROOT, HF_CACHE, get_settings
from gridsight.ingest.corpus import (
    MAX_EDGE,
    TEST_DEFECT_CAP,
    TEST_GOOD_CAP,
    TRAIN_GOOD_CAP,
    class_counts,
    dedupe_indices,
    mask_count,
    resize_max_edge,
    write_masked_split,
    write_split,
)

log = logging.getLogger("gridsight.ingest.folder")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class FolderLayout:
    """Where one category's files sit inside the repo. `{category}` is interpolated."""

    train_good: str = "{category}/train/good"
    test_root: str = "{category}/test"
    test_good_name: str = "good"
    mask_root: str = "{category}/ground_truth"
    #: Mask filename derived from the defect frame's stem, inside `mask_root/<defect_type>/`.
    mask_name: str = "{stem}_mask.png"

    def resolve(self, root: Path, template: str, category: str) -> Path:
        return root / template.format(category=category)


#: MVTec AD's canonical on-disk layout, which `foersben/mvtec-ad` preserves.
MVTEC_LAYOUT = FolderLayout()


@dataclass(frozen=True)
class FolderCategorySpec:
    """One asset class to be cut out of one category of a folder-structured repo."""

    dataset_id: str
    category: str
    asset_class: str
    license: str
    rationale: str
    notes: str = ""
    layout: FolderLayout = MVTEC_LAYOUT
    #: Controlled-capture corpora photograph the same part in the same rig every
    #: time, so a perceptual gate tuned for scraped web imagery (distance 4) reads
    #: the entire class as one duplicate and deletes it. 0 keeps exact re-encodes
    #: out while preserving genuinely distinct physical samples.
    phash_distance: int = 0
    #: Non-commercial licences propagate to derived weights; recorded per class so
    #: the registry UI can state the restriction rather than the operator guessing.
    license_restriction: str = ""
    allow_patterns: tuple[str, ...] = field(default=())

    def patterns(self) -> list[str]:
        """Fetch only this category -- never the whole multi-gigabyte repo."""
        return list(self.allow_patterns) or [f"{self.category}/**"]


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _load(path: Path) -> Image.Image:
    """Load and immediately bound the longest edge, so a 400-frame pool fits in RAM."""
    with Image.open(path) as im:
        return resize_max_edge(im.convert("RGB").copy())


def _interleave(groups: list[list[Path]]) -> list[Path]:
    """Round-robin across defect types.

    Straight concatenation plus a cap would hand the whole defect budget to
    whichever type sorted first; interleaving keeps every defect type present in
    the 40 frames that survive.
    """
    out: list[Path] = []
    for i in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if i < len(g):
                out.append(g[i])
    return out


def _defect_pairs(root: Path, spec: FolderCategorySpec) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Defect frames paired with their ground-truth masks, grouped by defect type."""
    layout = spec.layout
    test_root = layout.resolve(root, layout.test_root, spec.category)
    mask_root = layout.resolve(root, layout.mask_root, spec.category)

    groups: list[list[Path]] = []
    types: list[str] = []
    for sub in sorted(p for p in test_root.iterdir() if p.is_dir()):
        if sub.name == layout.test_good_name:
            continue
        paths = _image_paths(sub)
        if paths:
            groups.append(paths)
            types.append(sub.name)

    ordered = _interleave(groups)
    pairs: list[tuple[Path, Path]] = []
    for img_path in ordered:
        mask = mask_root / img_path.parent.name / layout.mask_name.format(stem=img_path.stem)
        if not mask.exists():
            # A defect frame without a mask would poison the pixel AUROC by
            # scoring against an all-zero target, so it is dropped, loudly.
            log.warning("[%s] no mask for %s -- dropping frame", spec.asset_class, img_path.name)
            continue
        pairs.append((img_path, mask))
    return pairs, types


def ingest_category(spec: FolderCategorySpec, seed: int = 1234) -> dict[str, Any]:
    """Fetch one category, cut the Anomalib tree plus masks, return a provenance row."""
    log.info("[%s] snapshot_download %s %s", spec.asset_class, spec.dataset_id, spec.patterns())
    root = Path(
        snapshot_download(
            repo_id=spec.dataset_id,
            repo_type="dataset",
            allow_patterns=spec.patterns(),
            cache_dir=str(HF_CACHE),
            token=get_settings().hf_token or None,
        )
    )

    layout = spec.layout
    train_paths = _image_paths(layout.resolve(root, layout.train_good, spec.category))
    test_good_paths = _image_paths(
        layout.resolve(root, layout.test_root, spec.category) / layout.test_good_name
    )
    defect_paths, defect_types = _defect_pairs(root, spec)
    raw = {"train_good": len(train_paths), "test_good": len(test_good_paths), "defect": len(defect_paths)}

    rng = random.Random(seed)
    rng.shuffle(train_paths)
    rng.shuffle(test_good_paths)
    # Defect order is the interleave, not a shuffle: shuffling would undo the
    # per-type balance the interleave exists to create.

    # The corpus already ships a curated train/test partition of physically
    # distinct samples, so `seen` is shared across all three pools purely to catch
    # a frame duplicated *across* splits -- which would leak train into test.
    seen: list[imagehash.ImageHash] = []
    train_pool = [_load(p) for p in train_paths[: TRAIN_GOOD_CAP * 2]]
    train_keep = dedupe_indices(train_pool, seen, spec.phash_distance)[:TRAIN_GOOD_CAP]
    train_good = [train_pool[i] for i in train_keep]

    good_pool = [_load(p) for p in test_good_paths[: TEST_GOOD_CAP * 2]]
    good_keep = dedupe_indices(good_pool, seen, spec.phash_distance)[:TEST_GOOD_CAP]
    test_good = [good_pool[i] for i in good_keep]

    defect_pool = [_load(p) for p, _ in defect_paths[: TEST_DEFECT_CAP * 2]]
    defect_keep = dedupe_indices(defect_pool, seen, spec.phash_distance)[:TEST_DEFECT_CAP]
    pairs = [(defect_pool[i], _load_mask(defect_paths[i][1])) for i in defect_keep]

    class_dir = DATA_ROOT / spec.asset_class
    if class_dir.exists():
        shutil.rmtree(class_dir)
    write_split(train_good, class_dir / "train" / "good")
    write_split(test_good, class_dir / "test" / "good")
    write_masked_split(pairs, class_dir / "test" / "defect", class_dir / "ground_truth")

    counts = class_counts(class_dir)
    masks = mask_count(class_dir)
    if masks != counts["test_defect"]:
        raise RuntimeError(
            f"{spec.asset_class}: {counts['test_defect']} defect frames but {masks} masks -- "
            "anomalib pairs them positionally and would raise MisMatchError"
        )

    record: dict[str, Any] = {
        "_id": spec.asset_class,
        "asset_class": spec.asset_class,
        "dataset_id": spec.dataset_id,
        "source_category": spec.category,
        "license": spec.license,
        "license_restriction": spec.license_restriction,
        "rationale": spec.rationale,
        "extraction_notes": spec.notes,
        "defect_types": defect_types,
        "raw_candidates": {"good": raw["train_good"] + raw["test_good"], "defect": raw["defect"]},
        "after_dedupe": {"good": len(train_good) + len(test_good), "defect": len(pairs)},
        "sample_counts": counts,
        "ground_truth_masks": masks,
        "phash_distance": spec.phash_distance,
        "max_edge_px": MAX_EDGE,
    }
    log.info(
        "[%s] raw train/test-good/defect=%d/%d/%d -> train_good=%d test_good=%d test_defect=%d masks=%d",
        spec.asset_class,
        raw["train_good"],
        raw["test_good"],
        raw["defect"],
        counts["train_good"],
        counts["test_good"],
        counts["test_defect"],
        masks,
    )
    return record


def _load_mask(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("L").copy()

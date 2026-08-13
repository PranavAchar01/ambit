"""Ingest adapter for parquet-packed, per-category anomaly corpora.

`folder_corpus` handles sources that ship the split as directories. A third
family arrives as **Hugging Face parquet with named splits** -- one
`data/<category>.train-*.parquet` / `data/<category>.test-*.parquet` pair per
category, images and masks embedded as `Image` structs rather than files. VisA
is the important member: it is the only verified-loadable PCB corpus with
ground-truth masks, and its CC BY 4.0 licence is *permissive*, which the MVTec
mirrors are not.

The seam is `ParquetSchema`: a declaration of which columns hold the frame, the
mask, the binary label and (optionally) the defect type. Column names differ per
repo -- VisA calls them `image`/`mask`/`label`, `TheoM55/mvtec_all_objects_split`
calls them `image_path`/`mask_path`/`label`/`defect` -- but the geometry is the
same, so one adapter covers both and every frame still travels through the
shared `gridsight.ingest.corpus` resize/dedupe/write path.

Rows are selected from the label column *before* any pixels are decoded, then
only the chosen rows are materialised. A VisA category is a multi-hundred-MB
parquet of 1500x1000 frames; decoding all of it to keep 190 images would be the
difference between a 30-second ingest and an OOM.
"""

from __future__ import annotations

import io
import logging
import random
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
import pyarrow.parquet as pq
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

log = logging.getLogger("gridsight.ingest.parquet")

#: Rows per arrow batch while scanning for the selected indices. Small enough
#: that a batch of 1500x1000 JPEG blobs stays well inside RAM.
SCAN_BATCH = 16


@dataclass(frozen=True)
class ParquetSchema:
    """Which columns of a per-category parquet mean what."""

    image: str
    mask: str
    label: str
    #: Present only on sources that keep the defect taxonomy (MVTec mirrors do,
    #: VisA does not). Drives the round-robin that stops one defect type from
    #: eating the whole 40-frame budget.
    defect: str | None = None
    #: Guard column for repos that pack every category into one file; harmless
    #: to also apply on per-category files, where it is a no-op assertion.
    category: str | None = None
    normal_label: int = 0
    anomaly_label: int = 1


#: `BrachioLab/visa`: image(Image), mask(Image), label(int64). No defect taxonomy.
VISA_SCHEMA = ParquetSchema(image="image", mask="mask", label="label")

#: `TheoM55/mvtec_all_objects_split`: image_path(Image), mask_path(Image),
#: label(int64), plus real `object` and `defect` columns.
MVTEC_PARQUET_SCHEMA = ParquetSchema(
    image="image_path", mask="mask_path", label="label", defect="defect", category="object"
)


@dataclass(frozen=True)
class ParquetCategorySpec:
    """One asset class cut out of one category of a parquet-packed repo."""

    dataset_id: str
    category: str
    asset_class: str
    license: str
    rationale: str
    schema: ParquetSchema
    notes: str = ""
    #: Non-commercial licences propagate to derived weights; recorded per class so
    #: the registry UI can state the restriction rather than the operator guessing.
    license_restriction: str = ""
    #: Controlled-capture corpora photograph the same part in the same rig every
    #: time, so a perceptual gate tuned for scraped web imagery (distance 4) reads
    #: the entire class as one duplicate and deletes it. 0 keeps exact re-encodes
    #: out while preserving genuinely distinct physical samples.
    phash_distance: int = 0
    file_prefix: str = "data/{category}"
    allow_patterns: tuple[str, ...] = field(default=())

    def patterns(self) -> list[str]:
        """Fetch only this category -- never the whole multi-gigabyte repo."""
        return list(self.allow_patterns) or [self.file_prefix.format(category=self.category) + ".*"]

    def split_files(self, root: Path, split: str) -> list[Path]:
        prefix = self.file_prefix.format(category=self.category)
        return sorted((root / Path(prefix).parent).glob(f"{Path(prefix).name}.{split}-*.parquet"))


# ---------------------------------------------------------------------------
# Row selection: labels first, pixels later
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """A located row: which file, which position in it, and its label metadata."""

    file: Path
    index: int
    label: int
    defect: str


def _scan_labels(files: Sequence[Path], schema: ParquetSchema, category: str) -> list[_Row]:
    """Read only the label (and defect/category) columns -- no image bytes decoded."""
    columns = [schema.label]
    for optional in (schema.defect, schema.category):
        if optional:
            columns.append(optional)

    rows: list[_Row] = []
    for path in files:
        table = pq.read_table(path, columns=columns)
        labels = table.column(schema.label).to_pylist()
        defects = table.column(schema.defect).to_pylist() if schema.defect else [""] * len(labels)
        categories = table.column(schema.category).to_pylist() if schema.category else None
        for i, label in enumerate(labels):
            if categories is not None and categories[i] != category:
                continue
            rows.append(_Row(path, i, int(label), str(defects[i] or "")))
    return rows


def _split_pools(rows: Sequence[_Row], schema: ParquetSchema) -> tuple[list[_Row], list[_Row]]:
    normal = [r for r in rows if r.label == schema.normal_label]
    anomalous = [r for r in rows if r.label == schema.anomaly_label]
    return normal, anomalous


def _interleave_by_defect(rows: Sequence[_Row]) -> list[_Row]:
    """Round-robin across defect types.

    Straight concatenation plus a cap would hand the whole defect budget to
    whichever type sorted first; interleaving keeps every defect type present in
    the 40 frames that survive. Sources without a defect column collapse to a
    single group, which is the identity.
    """
    groups: dict[str, list[_Row]] = {}
    for row in rows:
        groups.setdefault(row.defect, []).append(row)
    ordered: list[_Row] = []
    columns = [groups[k] for k in sorted(groups)]
    for i in range(max((len(g) for g in columns), default=0)):
        for g in columns:
            if i < len(g):
                ordered.append(g[i])
    return ordered


def _decode(cell: Any) -> Image.Image | None:
    """Turn a Hugging Face `Image` struct into a PIL image."""
    if cell is None:
        return None
    blob = cell["bytes"] if isinstance(cell, dict) else cell
    if not blob:
        return None
    with Image.open(io.BytesIO(blob)) as im:
        return im.copy()


def _materialise(
    rows: Sequence[_Row], schema: ParquetSchema, want_mask: bool
) -> list[tuple[Image.Image, Image.Image | None]]:
    """Decode exactly the selected rows, streaming each file once.

    Returns frames in the order `rows` requested, already bounded to MAX_EDGE so
    a 240-frame pool of 1500x1000 originals fits in RAM.
    """
    columns = [schema.image] + ([schema.mask] if want_mask else [])
    wanted: dict[Path, set[int]] = {}
    for row in rows:
        wanted.setdefault(row.file, set()).add(row.index)

    decoded: dict[tuple[Path, int], tuple[Image.Image, Image.Image | None]] = {}
    for path, indices in wanted.items():
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=SCAN_BATCH, columns=columns):
            hits = [i for i in range(batch.num_rows) if offset + i in indices]
            if hits:
                images = batch.column(schema.image).to_pylist()
                masks = batch.column(schema.mask).to_pylist() if want_mask else None
                for i in hits:
                    frame = _decode(images[i])
                    if frame is None:
                        continue
                    mask = _decode(masks[i]) if masks is not None else None
                    decoded[path, offset + i] = (
                        resize_max_edge(frame.convert("RGB")),
                        mask.convert("L") if mask is not None else None,
                    )
            offset += batch.num_rows
    return [decoded[r.file, r.index] for r in rows if (r.file, r.index) in decoded]


def _has_foreground(mask: Image.Image) -> bool:
    return bool(mask.getbbox())


def _keep_by_dedupe(
    pool: Sequence[tuple[Image.Image, Image.Image | None]],
    seen: list[imagehash.ImageHash],
    distance: int,
    cap: int,
) -> list[int]:
    return dedupe_indices([p[0] for p in pool], seen, distance)[:cap]


def _take(items: Iterable[Any], keep: Sequence[int]) -> list[Any]:
    pool = list(items)
    return [pool[i] for i in keep]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest_parquet_category(spec: ParquetCategorySpec, seed: int = 1234) -> dict[str, Any]:
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

    train_files = spec.split_files(root, "train")
    test_files = spec.split_files(root, "test")
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"{spec.asset_class}: no parquet for category {spec.category!r} under {root} "
            f"(train={train_files}, test={test_files})"
        )

    train_rows = _scan_labels(train_files, spec.schema, spec.category)
    test_rows = _scan_labels(test_files, spec.schema, spec.category)
    train_normal, train_anomalous = _split_pools(train_rows, spec.schema)
    test_normal, test_anomalous = _split_pools(test_rows, spec.schema)
    if train_anomalous:
        # Every source verified here ships a normal-only train split; an anomalous
        # row in it would silently teach PatchCore that the defect is nominal.
        log.warning(
            "[%s] %d anomalous rows in the train split -- excluded from train/good",
            spec.asset_class,
            len(train_anomalous),
        )
    raw = {"train_good": len(train_normal), "test_good": len(test_normal), "defect": len(test_anomalous)}

    rng = random.Random(seed)
    rng.shuffle(train_normal)
    rng.shuffle(test_normal)
    # Defect order is the interleave, not a shuffle: shuffling would undo the
    # per-type balance the interleave exists to create.
    defect_ordered = _interleave_by_defect(test_anomalous)

    # The corpus already ships a curated train/test partition of physically
    # distinct samples, so `seen` is shared across all three pools purely to catch
    # a frame duplicated *across* splits -- which would leak train into test.
    seen: list[imagehash.ImageHash] = []
    train_pool = _materialise(train_normal[: TRAIN_GOOD_CAP * 2], spec.schema, want_mask=False)
    train_good = _take(train_pool, _keep_by_dedupe(train_pool, seen, spec.phash_distance, TRAIN_GOOD_CAP))

    good_pool = _materialise(test_normal[: TEST_GOOD_CAP * 2], spec.schema, want_mask=False)
    test_good = _take(good_pool, _keep_by_dedupe(good_pool, seen, spec.phash_distance, TEST_GOOD_CAP))

    defect_pool = _materialise(defect_ordered[: TEST_DEFECT_CAP * 2], spec.schema, want_mask=True)
    # A defect frame whose mask is missing or all-zero would poison the pixel
    # AUROC by scoring against an empty target, so it is dropped, loudly. An
    # entire class failing this check means the label polarity is inverted.
    masked = []
    for frame, mask in defect_pool:
        if mask is None or not _has_foreground(mask):
            log.warning("[%s] anomalous row with empty mask -- dropping frame", spec.asset_class)
            continue
        masked.append((frame, mask))
    if not masked:
        raise RuntimeError(
            f"{spec.asset_class}: every anomalous row had an empty mask -- "
            f"label polarity for {spec.schema.label!r} is probably inverted"
        )
    keep = dedupe_indices([f for f, _ in masked], seen, spec.phash_distance)[:TEST_DEFECT_CAP]
    pairs = [masked[i] for i in keep]

    class_dir = DATA_ROOT / spec.asset_class
    if class_dir.exists():
        shutil.rmtree(class_dir)
    write_split([f for f, _ in train_good], class_dir / "train" / "good")
    write_split([f for f, _ in test_good], class_dir / "test" / "good")
    write_masked_split(pairs, class_dir / "test" / "defect", class_dir / "ground_truth")

    counts = class_counts(class_dir)
    masks = mask_count(class_dir)
    if masks != counts["test_defect"]:
        raise RuntimeError(
            f"{spec.asset_class}: {counts['test_defect']} defect frames but {masks} masks -- "
            "anomalib pairs them positionally and would raise MisMatchError"
        )

    defect_types = sorted({r.defect for r in test_anomalous if r.defect}) or ["anomaly"]
    record: dict[str, Any] = {
        "_id": spec.asset_class,
        "asset_class": spec.asset_class,
        "dataset_id": spec.dataset_id,
        "source_category": spec.category,
        "source_files": sorted(p.name for p in train_files + test_files),
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

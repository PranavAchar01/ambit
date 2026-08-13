"""Phase 1: the scraped corpus must actually satisfy what training needs."""

from __future__ import annotations

from pathlib import Path

from conftest import requires_atlas
from gridsight.ingest.hf_scrape import (
    COMPONENT_MAX_SIDE,
    TOWER_MIN_SIDE,
    box_side,
    class_counts,
    resize_max_edge,
)


def is_normal_only(data_root: Path, cls: str) -> bool:
    """A class captured locally, with no labelled defect set.

    Split out rather than relaxing the corpus minimums: a downloaded corpus that
    arrives short of 20 train/good or 5 test/defect is a broken ingest and must
    still fail. A bench capture has no defect frames by construction, which is
    the state of the beachhead ICP and not a fault -- so it carries its own,
    stricter-where-it-can-be invariant below.
    """
    return not any((data_root / cls / "test" / "defect").glob("*.png"))


def test_every_class_meets_training_minimums(data_root: Path, scraped_classes: list[str]) -> None:
    assert len(scraped_classes) >= 5, f"need >=5 asset classes, found {scraped_classes}"
    for cls in scraped_classes:
        counts = class_counts(data_root / cls)
        if is_normal_only(data_root, cls):
            # Enough frames to fit a bank and still hold folds out for the
            # threshold, plus withheld frames that were never trained on.
            assert counts["train_good"] >= 8, f"{cls}: {counts['train_good']} train/good"
            assert counts["test_good"] == 0, f"{cls}: normal-only classes keep no test/good split"
            held = list((data_root / cls / "heldout").glob("*.png"))
            assert held, f"{cls}: normal-only class with no held-out frames to validate against"
            continue
        assert counts["train_good"] >= 20, f"{cls}: {counts['train_good']} train/good"
        assert counts["test_defect"] >= 5, f"{cls}: {counts['test_defect']} test/defect"


def test_images_are_within_the_resize_budget(data_root: Path, scraped_classes: list[str]) -> None:
    from PIL import Image

    for cls in scraped_classes:
        for path in sorted((data_root / cls / "train" / "good").glob("*.png"))[:5]:
            with Image.open(path) as im:
                assert max(im.size) <= 512, f"{path} is {im.size}, above the 512px longest edge"


def test_resize_preserves_aspect_and_caps_longest_edge() -> None:
    from PIL import Image

    out = resize_max_edge(Image.new("RGB", (2000, 1000)), 512)
    assert out.size == (512, 256)
    small = Image.new("RGB", (100, 50))
    assert resize_max_edge(small, 512).size == (100, 50)


def test_scale_split_bounds_do_not_overlap() -> None:
    # the whole taxonomy fix depends on these two ranges being disjoint
    assert TOWER_MIN_SIDE > COMPONENT_MAX_SIDE
    assert box_side([0.0, 0.0, 200.0, 40.0]) == 200.0


@requires_atlas
def test_provenance_recorded_for_every_class(data_root: Path, scraped_classes: list[str]) -> None:
    from gridsight.db.mongo import datasets_col

    recorded = {d["_id"]: d for d in datasets_col().find()}
    for cls in scraped_classes:
        assert cls in recorded, f"no provenance row for {cls}"
        row = recorded[cls]
        assert row["dataset_id"], f"{cls} has no source dataset id"
        assert row["rationale"], f"{cls} has no selection rationale"
        assert row["license"], f"{cls} has no licence recorded"
        minimum = 8 if is_normal_only(data_root, cls) else 20
        assert row["sample_counts"]["train_good"] >= minimum


@requires_atlas
def test_locally_captured_classes_record_their_split_by_filename(
    data_root: Path, scraped_classes: list[str]
) -> None:
    """A held-out frame is only evidence if it can be shown it was never trained on."""
    from gridsight.db.mongo import datasets_col

    recorded = {d["_id"]: d for d in datasets_col().find()}
    local = [c for c in scraped_classes if is_normal_only(data_root, c)]
    for cls in local:
        split = recorded[cls].get("split_by_filename")
        assert split, f"{cls}: no split recorded by filename"
        assert split["held_out"], f"{cls}: no held-out frames recorded"
        overlap = set(split["train_good"]) & set(split["held_out"])
        assert not overlap, f"{cls}: {sorted(overlap)} are both trained on and held out"
        assert len(split["held_out"]) == len(list((data_root / cls / "heldout").glob("*.png")))

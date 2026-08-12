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


def test_every_class_meets_training_minimums(data_root: Path, scraped_classes: list[str]) -> None:
    assert len(scraped_classes) >= 5, f"need >=5 asset classes, found {scraped_classes}"
    for cls in scraped_classes:
        counts = class_counts(data_root / cls)
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
def test_provenance_recorded_for_every_class(scraped_classes: list[str]) -> None:
    from gridsight.db.mongo import datasets_col

    recorded = {d["_id"]: d for d in datasets_col().find()}
    for cls in scraped_classes:
        assert cls in recorded, f"no provenance row for {cls}"
        row = recorded[cls]
        assert row["dataset_id"], f"{cls} has no source dataset id"
        assert row["rationale"], f"{cls} has no selection rationale"
        assert row["sample_counts"]["train_good"] >= 20

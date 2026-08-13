"""The parquet ingest adapter, and what the electronics classes it produced must satisfy.

The pure helpers are tested without network or parquet, because the properties
that matter -- defect-type balance survives the 40-frame cap, label polarity is
read from the schema rather than assumed -- are the ones that silently produce a
plausible-looking but wrong corpus when they break.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import requires_atlas
from gridsight.ingest.corpus import class_counts, mask_count
from gridsight.ingest.parquet_corpus import (
    MVTEC_PARQUET_SCHEMA,
    VISA_SCHEMA,
    ParquetSchema,
    _interleave_by_defect,
    _Row,
    _split_pools,
)

#: Classes this adapter produced. visa_pcb4 is ingested but deliberately NOT
#: trained -- it is the out-of-registry target for the cold-start/refusal demo --
#: so it must satisfy the corpus contract exactly like the trained ones.
ELECTRONICS_CLASSES = ("visa_pcb1", "visa_pcb2", "visa_pcb3", "visa_pcb4", "mvtec_cable")


def _row(index: int, label: int, defect: str = "") -> _Row:
    return _Row(Path("f.parquet"), index, label, defect)


def test_pools_split_on_the_schema_declared_labels() -> None:
    rows = [_row(0, 0), _row(1, 1), _row(2, 0), _row(3, 1)]
    normal, anomalous = _split_pools(rows, VISA_SCHEMA)
    assert [r.index for r in normal] == [0, 2]
    assert [r.index for r in anomalous] == [1, 3]

    # A source that inverts the convention is a schema change, not a code change.
    inverted = ParquetSchema(image="i", mask="m", label="l", normal_label=1, anomaly_label=0)
    normal, anomalous = _split_pools(rows, inverted)
    assert [r.index for r in normal] == [1, 3]
    assert [r.index for r in anomalous] == [0, 2]


def test_interleave_keeps_every_defect_type_inside_the_cap() -> None:
    # 30 frames of one type then 2 of another: a straight cap of 4 would keep only
    # the first type, which is exactly the failure the round-robin exists to stop.
    rows = [_row(i, 1, "bent_wire") for i in range(30)] + [_row(100 + i, 1, "cable_swap") for i in range(2)]
    ordered = _interleave_by_defect(rows)
    assert {r.defect for r in ordered[:4]} == {"bent_wire", "cable_swap"}
    assert len(ordered) == len(rows)


def test_interleave_is_identity_for_sources_without_a_defect_taxonomy() -> None:
    rows = [_row(i, 1) for i in range(5)]
    assert [r.index for r in _interleave_by_defect(rows)] == [0, 1, 2, 3, 4]


def test_the_two_shipped_schemas_name_different_columns() -> None:
    # The whole reason ParquetSchema exists: VisA and the MVTec mirror carry the
    # same geometry under different column names.
    assert (VISA_SCHEMA.image, VISA_SCHEMA.mask) == ("image", "mask")
    assert (MVTEC_PARQUET_SCHEMA.image, MVTEC_PARQUET_SCHEMA.mask) == ("image_path", "mask_path")
    assert MVTEC_PARQUET_SCHEMA.defect == "defect"
    assert VISA_SCHEMA.defect is None


@pytest.mark.parametrize("asset_class", ELECTRONICS_CLASSES)
def test_electronics_class_is_trainable_and_fully_masked(data_root: Path, asset_class: str) -> None:
    class_dir = data_root / asset_class
    if not class_dir.is_dir():
        pytest.skip(f"{asset_class} not ingested; run scripts/ingest_visa.py")
    counts = class_counts(class_dir)
    assert counts["train_good"] >= 20, f"{asset_class}: {counts['train_good']} train/good"
    assert counts["test_defect"] >= 5, f"{asset_class}: {counts['test_defect']} test/defect"
    # One mask per defect frame or anomalib pairs them positionally and misaligns
    # every pixel score after the first gap.
    assert mask_count(class_dir) == counts["test_defect"]


@requires_atlas
@pytest.mark.parametrize("asset_class", ELECTRONICS_CLASSES)
def test_provenance_records_the_licence(asset_class: str) -> None:
    from gridsight.db.mongo import datasets_col

    row = datasets_col().find_one({"_id": asset_class})
    if row is None:
        pytest.skip(f"{asset_class} not ingested; run scripts/ingest_visa.py")
    assert row["dataset_id"]
    assert row["rationale"]
    # The licence split is the product story -- VisA ships, MVTec does not -- so an
    # unlabelled row is a correctness bug, not a documentation gap.
    assert row["license"], f"{asset_class} has no licence recorded"
    assert row["license_restriction"], f"{asset_class} has no licence restriction recorded"
    expected = "CC BY 4.0" if asset_class.startswith("visa_") else "CC BY-NC-SA 4.0"
    assert row["license"] == expected

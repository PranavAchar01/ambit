"""Episodic memory: the second vector index, and the recall it enables.

The routing index answers "which specialist was trained on imagery like this?".
This one answers "have we seen this before?" -- the same maths over a different
memory. These tests exist because the two indexes share one code path, and a
regression that silently pointed recall at the models collection would still
return plausible-looking results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from conftest import first_probe, requires_atlas


def _embed(path: Path) -> list[float]:
    from gridsight.embed import get_embedder

    with Image.open(path) as im:
        return [float(x) for x in get_embedder().embed_one(im.convert("RGB"))]


def test_the_two_indexes_are_distinct_and_correctly_targeted() -> None:
    """A single mistyped collection name would make recall silently search models."""
    from gridsight.db.mongo import FINDINGS, MODELS, finding_recall_index, model_router_index

    router, recall = model_router_index(), finding_recall_index()
    assert router.name != recall.name
    assert router.collection == MODELS
    assert recall.collection == FINDINGS
    assert router.path == recall.path == "embedding"


def test_index_definitions_declare_the_right_dimensions_and_filters() -> None:
    from gridsight.db.mongo import finding_recall_index, model_router_index

    for idx, expected_filters in (
        (model_router_index(), {"asset_class"}),
        (finding_recall_index(), {"asset_class", "verdict"}),
    ):
        definition = idx.definition(512)
        vector_fields = [f for f in definition["fields"] if f["type"] == "vector"]
        assert len(vector_fields) == 1
        assert vector_fields[0]["numDimensions"] == 512
        assert vector_fields[0]["similarity"] == "cosine"
        assert vector_fields[0]["path"] == idx.path
        filters = {f["path"] for f in definition["fields"] if f["type"] == "filter"}
        assert filters == expected_filters


@requires_atlas
def test_recall_index_is_ready() -> None:
    from gridsight.db.mongo import finding_recall_index, search_index_status

    idx = finding_recall_index()
    status = search_index_status(idx)
    if status is None:
        pytest.skip("recall index not built; run scripts/build_recall_index.py")
    assert status == "READY", f"{idx.name} is {status}, so $vectorSearch returns nothing"


@requires_atlas
def test_every_finding_that_has_a_frame_carries_an_embedding() -> None:
    """A finding without a vector is invisible to recall -- silent memory loss."""
    from gridsight.db.mongo import findings_col

    total = findings_col().count_documents({"uploaded_image_id": {"$ne": None}})
    if total == 0:
        pytest.skip("no findings recorded yet")
    missing = findings_col().count_documents(
        {"uploaded_image_id": {"$ne": None}, "embedding": {"$in": [None, []]}}
    )
    assert missing == 0, f"{missing}/{total} findings have no embedding and cannot be recalled"


@requires_atlas
def test_embeddings_are_the_right_shape_and_normalised() -> None:
    from gridsight.db.mongo import findings_col

    doc = findings_col().find_one({"embedding": {"$exists": True, "$ne": []}}, {"embedding": 1})
    if doc is None:
        pytest.skip("no embedded findings")
    vec = np.asarray(doc["embedding"], dtype=np.float32)
    assert vec.shape == (512,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-3), "CLIP vectors must be L2-normalised"


@requires_atlas
def test_recall_finds_the_same_frame_first() -> None:
    """Querying with a frame that was itself inspected must return it at the top."""
    from gridsight.db.mongo import finding_recall_index, findings_col, vector_search

    seeded = findings_col().find_one(
        {"embedding": {"$exists": True, "$ne": []}, "verdict": "defect"},
        {"embedding": 1, "asset_class": 1},
    )
    if seeded is None:
        pytest.skip("no embedded defect findings to query with")

    hits = vector_search(list(seeded["embedding"]), k=10, index=finding_recall_index())
    assert hits, "recall returned nothing"

    # The same frame is legitimately inspected more than once across runs, so
    # several findings can carry an identical vector and tie at 1.0. Requiring a
    # specific one to win that tie would be testing arbitrary ordering; the real
    # invariant is that the query recalls itself among the perfect matches.
    perfect = {str(h["_id"]) for h in hits if h["score"] == pytest.approx(1.0, abs=1e-3)}
    assert perfect, "nothing scored ~1.0 against its own vector"
    assert str(seeded["_id"]) in perfect, "a frame did not recall itself"
    assert all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1))


@requires_atlas
def test_recall_returns_findings_not_models(data_root: Path, scraped_classes: list[str]) -> None:
    """The projections differ; a mis-targeted index would return model documents."""
    from gridsight.db.mongo import finding_recall_index, vector_search

    probes = [p for p in [first_probe(data_root, scraped_classes, "test/defect")] if p]
    if not probes:
        pytest.skip("no probe image")

    hits = vector_search(_embed(probes[0]), k=3, index=finding_recall_index())
    if not hits:
        pytest.skip("no findings indexed yet")
    for h in hits:
        assert "verdict" in h, "recall returned something that is not a finding"
        assert "training_samples" not in h, "recall returned a MODEL document"


@requires_atlas
def test_recall_honours_the_verdict_filter(data_root: Path, scraped_classes: list[str]) -> None:
    from gridsight.db.mongo import finding_recall_index, findings_col, vector_search

    if findings_col().count_documents({"verdict": "defect", "embedding": {"$ne": []}}) < 2:
        pytest.skip("not enough embedded defect findings")

    probe = first_probe(data_root, scraped_classes, "test/defect")
    if probe is None:
        pytest.skip("no class has a test/defect split on disk")
    hits = vector_search(_embed(probe), k=5, index=finding_recall_index(), filters={"verdict": "defect"})
    assert hits, "filtered recall returned nothing"
    assert {h["verdict"] for h in hits} == {"defect"}


@requires_atlas
def test_brute_force_fallback_agrees_with_the_index(data_root: Path, scraped_classes: list[str]) -> None:
    """The degraded path must rank the same way, or an outage changes the answers."""
    from gridsight.db.mongo import _brute_force_cosine, finding_recall_index, vector_search

    probes = [p for p in [first_probe(data_root, scraped_classes, "test/defect")] if p]
    if not probes:
        pytest.skip("no probe image")
    vector = _embed(probes[0])
    idx = finding_recall_index()

    indexed = vector_search(vector, k=3, index=idx)
    fallback = _brute_force_cosine(vector, 3, None, idx, None)
    if not indexed or not fallback:
        pytest.skip("nothing indexed")

    assert str(indexed[0]["_id"]) == str(fallback[0]["_id"]), "index and fallback disagree on the top hit"
    assert indexed[0]["score"] == pytest.approx(fallback[0]["score"], abs=5e-3)


@requires_atlas
def test_registry_layout_is_stable_and_well_formed() -> None:
    """The animation reads these angles; duplicates would overlap models on screen."""
    from gridsight.api.main import registry_layout

    layout: dict[str, Any] = registry_layout()
    models = layout["models"]
    if not models:
        pytest.skip("registry empty")

    assert layout["angle_is_illustrative"] is True
    angles = [m["angle"] for m in models]
    assert all(0 <= a < 2 * np.pi + 1e-6 for a in angles)
    assert len(set(round(a, 4) for a in angles)) == len(angles), "two models share an angle"

    for m in models:
        assert m["gate"] >= layout["route_threshold"], "a gate fell below the global floor"

    # deterministic: the same registry must lay out identically across calls
    again = registry_layout()
    assert [m["angle"] for m in again["models"]] == angles

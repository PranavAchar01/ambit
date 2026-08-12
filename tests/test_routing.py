"""Phase 4: does the vector router actually pick the right specialist?

Thresholds here are set below the measured operating point (79% top-1, 24/25
out-of-registry refusals) so the suite fails on a real regression rather than on
run-to-run noise -- not tuned upward to make a weak router look good.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from conftest import requires_atlas

MIN_TOP1_ACCURACY = 0.60
PROBES = 8


def _embed(path: Path) -> list[float]:
    from gridsight.embed import get_embedder

    with Image.open(path) as im:
        return [float(x) for x in get_embedder().embed_one(im.convert("RGB"))]


def _gate(doc: dict[str, Any]) -> float:
    from gridsight.config import get_settings

    return max(get_settings().route_threshold, float(doc.get("routing_threshold") or 0.0))


@requires_atlas
def test_vector_index_is_ready() -> None:
    from gridsight.config import get_settings
    from gridsight.db.mongo import search_index_status

    status = search_index_status(get_settings().vector_index_name)
    assert status == "READY", f"vector index is {status}, so $vectorSearch returns nothing"


@requires_atlas
def test_search_returns_scored_candidates(data_root: Path, registry_models: list[dict[str, Any]]) -> None:
    from gridsight.db.mongo import vector_search

    cls = registry_models[0]["asset_class"]
    probe = sorted((data_root / cls / "test" / "good").glob("*.png"))[0]
    hits = vector_search(_embed(probe), k=5)

    assert hits, "$vectorSearch returned nothing"
    assert len(hits) <= 5
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), "results are not ranked by score"
    for h in hits:
        assert 0.0 <= h["score"] <= 1.0
        assert h["asset_class"]


@requires_atlas
def test_held_out_frames_route_to_their_own_specialist(
    data_root: Path, registry_models: list[dict[str, Any]]
) -> None:
    from gridsight.db.mongo import vector_search

    correct = total = 0
    per_class: dict[str, tuple[int, int]] = {}
    for doc in registry_models:
        cls = doc["asset_class"]
        probes = sorted((data_root / cls / "test" / "good").glob("*.png"))[:PROBES]
        if not probes:
            continue
        hits = sum(1 for p in probes if vector_search(_embed(p), k=5)[0]["asset_class"] == cls)
        per_class[cls] = (hits, len(probes))
        correct += hits
        total += len(probes)

    assert total > 0
    accuracy = correct / total
    assert accuracy >= MIN_TOP1_ACCURACY, f"top-1 routing accuracy {accuracy:.2f}: {per_class}"


@requires_atlas
def test_an_out_of_registry_class_is_refused(
    data_root: Path, scraped_classes: list[str], registry_models: list[dict[str, Any]]
) -> None:
    from gridsight.db.mongo import vector_search

    registered = {d["asset_class"] for d in registry_models}
    withheld = [c for c in scraped_classes if c not in registered]
    if not withheld:
        pytest.skip("every scraped class is registered; nothing is out-of-registry")

    cls = withheld[0]
    probes = sorted((data_root / cls / "test" / "good").glob("*.png"))[:PROBES]
    refused = 0
    for p in probes:
        top = vector_search(_embed(p), k=5)[0]
        if top["score"] < _gate(top):
            refused += 1

    assert refused > len(probes) / 2, (
        f"{cls} is not in the registry but only {refused}/{len(probes)} frames were refused"
    )

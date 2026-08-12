"""MongoDB Atlas access layer: collections, GridFS, and the Atlas Vector Search index.

Atlas is a hard requirement, not a preference: `$vectorSearch` is an Atlas-only
aggregation stage. A self-hosted `mongod` simply does not implement it. When the
cluster cannot serve a vector index we fall back to brute-force cosine similarity
and announce it loudly -- never silently.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from gridsight.config import get_settings

try:  # pymongo >= 4.6 ships the synchronous GridFS bucket here
    from gridfs.synchronous.grid_file import GridFSBucket
except ImportError:  # pragma: no cover - older pymongo layout
    from gridfs import GridFSBucket  # type: ignore[no-redef]

log = logging.getLogger("gridsight.db")

MODELS = "models"
FINDINGS = "findings"
DATASETS = "datasets"
CHECKPOINTS = "checkpoints"
CHECKPOINT_WRITES = "checkpoint_writes"
WEIGHTS_BUCKET = "weights"
IMAGES_BUCKET = "images"

_client: MongoClient[dict[str, Any]] | None = None


def get_client() -> MongoClient[dict[str, Any]]:
    """Process-wide MongoClient (the driver pools connections internally)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = MongoClient(
            settings.mongodb_uri,
            appname="gridsight",
            serverSelectionTimeoutMS=20_000,
            connectTimeoutMS=20_000,
        )
    return _client


def get_db() -> Database[dict[str, Any]]:
    return get_client()[get_settings().mongodb_db]


def models_col() -> Collection[dict[str, Any]]:
    return get_db()[MODELS]


def findings_col() -> Collection[dict[str, Any]]:
    return get_db()[FINDINGS]


def datasets_col() -> Collection[dict[str, Any]]:
    return get_db()[DATASETS]


def weights_bucket() -> GridFSBucket:
    return GridFSBucket(get_db(), bucket_name=WEIGHTS_BUCKET)


def images_bucket() -> GridFSBucket:
    return GridFSBucket(get_db(), bucket_name=IMAGES_BUCKET)


def ensure_collections() -> None:
    """Create the collections and their supporting b-tree indexes."""
    db = get_db()
    existing = set(db.list_collection_names())
    for name in (MODELS, FINDINGS, DATASETS):
        if name not in existing:
            db.create_collection(name)
            log.info("created collection %s", name)

    models_col().create_index("asset_class")
    models_col().create_index("name", unique=True)
    models_col().create_index("created_at")
    findings_col().create_index("timestamp")
    findings_col().create_index([("asset_class", 1), ("timestamp", -1)])
    findings_col().create_index("verdict")
    datasets_col().create_index("asset_class", unique=True)


# --------------------------------------------------------------------------
# Atlas Vector Search index
# --------------------------------------------------------------------------


def vector_index_definition(dims: int) -> dict[str, Any]:
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dims,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "asset_class"},
        ]
    }


def search_index_status(name: str) -> str | None:
    """Return the Atlas search-index status, or None when it does not exist."""
    try:
        for idx in models_col().list_search_indexes():
            if idx.get("name") == name:
                return str(idx.get("status", "UNKNOWN"))
    except OperationFailure as exc:
        log.warning("list_search_indexes failed: %s", exc)
        return None
    return None


def ensure_vector_index(timeout_s: float = 420.0, poll_s: float = 5.0) -> str:
    """Create `model_router_idx` if absent and block until it reports READY.

    A `$vectorSearch` against a PENDING index returns an empty result set rather
    than an error, so polling to READY is mandatory before any routing happens.
    """
    settings = get_settings()
    name = settings.vector_index_name
    status = search_index_status(name)

    if status is None:
        model = SearchIndexModel(
            definition=vector_index_definition(settings.embed_dim),
            name=name,
            type="vectorSearch",
        )
        models_col().create_search_index(model)
        log.info("issued createSearchIndexes for %s (512-d, cosine)", name)

    deadline = time.monotonic() + timeout_s
    last = status
    while time.monotonic() < deadline:
        last = search_index_status(name)
        if last == "READY":
            log.info("vector index %s is READY", name)
            return "READY"
        if last == "FAILED":
            raise RuntimeError(f"Atlas vector index {name} entered FAILED state")
        time.sleep(poll_s)

    raise TimeoutError(
        f"Atlas vector index {name} did not reach READY within {timeout_s:.0f}s (last status: {last})"
    )


def vector_search(
    embedding: list[float],
    k: int = 5,
    num_candidates: int | None = None,
    exclude_ids: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Route a frame: $vectorSearch over `models.embedding`, cosine similarity.

    Falls back to in-process brute-force cosine only when the Atlas index is
    unavailable, and logs a loud warning when it does.
    """
    settings = get_settings()
    stage: dict[str, Any] = {
        "index": settings.vector_index_name,
        "path": "embedding",
        "queryVector": embedding,
        "numCandidates": num_candidates or max(64, k * 20),
        "limit": k,
    }
    projection = {
        "_id": 1,
        "name": 1,
        "asset_class": 1,
        "backbone": 1,
        "training_samples": 1,
        "image_threshold": 1,
        "routing_threshold": 1,
        "metrics": 1,
        "created_by": 1,
        "reference_image_id": 1,
        "score": {"$meta": "vectorSearchScore"},
    }
    try:
        results = list(models_col().aggregate([{"$vectorSearch": stage}, {"$project": projection}]))
    except OperationFailure as exc:
        log.error(
            "ATLAS VECTOR SEARCH UNAVAILABLE (%s) -- DEGRADING TO BRUTE-FORCE COSINE. "
            "Routing quality and latency are NOT representative of production.",
            exc,
        )
        return _brute_force_cosine(embedding, k, exclude_ids)

    if not results:
        status = search_index_status(settings.vector_index_name)
        if status != "READY":
            log.error(
                "VECTOR INDEX %s IS %s, NOT READY -- $vectorSearch returned nothing. "
                "DEGRADING TO BRUTE-FORCE COSINE.",
                settings.vector_index_name,
                status,
            )
            return _brute_force_cosine(embedding, k, exclude_ids)

    if exclude_ids:
        blocked = set(exclude_ids)
        results = [r for r in results if r["_id"] not in blocked]
    return results


def _brute_force_cosine(
    embedding: list[float], k: int, exclude_ids: list[Any] | None
) -> list[dict[str, Any]]:
    """Offline/degraded routing path. Announced loudly by every caller."""
    import numpy as np

    query = np.asarray(embedding, dtype=np.float32)
    query /= np.linalg.norm(query) + 1e-12

    blocked = set(exclude_ids or [])
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in models_col().find(
        {},
        {
            "embedding": 1,
            "name": 1,
            "asset_class": 1,
            "backbone": 1,
            "training_samples": 1,
            "image_threshold": 1,
            "routing_threshold": 1,
            "metrics": 1,
            "created_by": 1,
            "reference_image_id": 1,
        },
    ):
        if doc["_id"] in blocked:
            continue
        vec = np.asarray(doc.pop("embedding"), dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        # Atlas reports cosine score as (1 + cos) / 2; mirror that so thresholds match.
        doc["score"] = float((1.0 + float(query @ vec)) / 2.0)
        scored.append((doc["score"], doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]

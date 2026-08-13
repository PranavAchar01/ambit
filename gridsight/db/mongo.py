"""MongoDB Atlas access layer: collections, GridFS, and the Atlas Vector Search index.

Atlas is a hard requirement, not a preference: `$vectorSearch` is an Atlas-only
aggregation stage. A self-hosted `mongod` simply does not implement it. When the
cluster cannot serve a vector index we fall back to brute-force cosine similarity
and announce it loudly -- never silently.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
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
# Atlas Vector Search
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorIndex:
    """A named Atlas vector index over one collection.

    Ambit runs two of these and they must not diverge: `models.embedding`
    answers "which specialist is competent here?" and `findings.embedding`
    answers "have we seen this before?". Same maths, different memory.
    """

    name: str
    collection: str
    path: str = "embedding"
    filters: tuple[str, ...] = ()
    projection: dict[str, Any] = field(default_factory=dict)

    def col(self) -> Collection[dict[str, Any]]:
        return get_db()[self.collection]

    def definition(self, dims: int) -> dict[str, Any]:
        fields: list[dict[str, Any]] = [
            {
                "type": "vector",
                "path": self.path,
                "numDimensions": dims,
                "similarity": "cosine",
            }
        ]
        fields.extend({"type": "filter", "path": f} for f in self.filters)
        return {"fields": fields}


MODEL_PROJECTION: dict[str, Any] = {
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
}

FINDING_PROJECTION: dict[str, Any] = {
    "_id": 1,
    "timestamp": 1,
    "asset_class": 1,
    "verdict": 1,
    "severity": 1,
    "anomaly_score": 1,
    "raw_anomaly_score": 1,
    "routed_model_name": 1,
    "routing_score": 1,
    "agent_narrative": 1,
    "uploaded_image_id": 1,
    "heatmap_id": 1,
}


def model_router_index() -> VectorIndex:
    """Semantic memory: which specialist was trained on imagery like this?"""
    return VectorIndex(
        name=get_settings().vector_index_name,
        collection=MODELS,
        filters=("asset_class",),
        projection=MODEL_PROJECTION,
    )


def finding_recall_index() -> VectorIndex:
    """Episodic memory: which past inspections looked like this one?"""
    return VectorIndex(
        name=get_settings().recall_index_name,
        collection=FINDINGS,
        filters=("asset_class", "verdict"),
        projection=FINDING_PROJECTION,
    )


def search_index_status(index: VectorIndex | str) -> str | None:
    """Return the Atlas search-index status, or None when it does not exist."""
    idx = model_router_index() if isinstance(index, str) else index
    name = index if isinstance(index, str) else index.name
    try:
        for existing in idx.col().list_search_indexes():
            if existing.get("name") == name:
                return str(existing.get("status", "UNKNOWN"))
    except OperationFailure as exc:
        log.warning("list_search_indexes failed for %s: %s", name, exc)
        return None
    return None


def ensure_vector_index(
    index: VectorIndex | None = None, timeout_s: float = 420.0, poll_s: float = 5.0
) -> str:
    """Create the index if absent and block until it reports READY.

    A `$vectorSearch` against a PENDING index returns an empty result set rather
    than an error, so polling to READY is mandatory before anything queries it.
    """
    idx = index or model_router_index()
    dims = get_settings().embed_dim
    status = search_index_status(idx)

    if status is None:
        idx.col().create_search_index(
            SearchIndexModel(definition=idx.definition(dims), name=idx.name, type="vectorSearch")
        )
        log.info("issued createSearchIndexes for %s on %s (%d-d, cosine)", idx.name, idx.collection, dims)

    deadline = time.monotonic() + timeout_s
    last = status
    while time.monotonic() < deadline:
        last = search_index_status(idx)
        if last == "READY":
            log.info("vector index %s is READY", idx.name)
            return "READY"
        if last == "FAILED":
            raise RuntimeError(f"Atlas vector index {idx.name} entered FAILED state")
        time.sleep(poll_s)

    raise TimeoutError(
        f"Atlas vector index {idx.name} did not reach READY within {timeout_s:.0f}s (last status: {last})"
    )


def vector_search(
    embedding: list[float],
    k: int = 5,
    num_candidates: int | None = None,
    exclude_ids: list[Any] | None = None,
    index: VectorIndex | None = None,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Cosine k-NN over an Atlas vector index.

    Falls back to in-process brute-force cosine only when the index is
    unavailable or not READY, and says so loudly every time it does.
    """
    idx = index or model_router_index()
    stage: dict[str, Any] = {
        "index": idx.name,
        "path": idx.path,
        "queryVector": embedding,
        "numCandidates": num_candidates or max(64, k * 20),
        "limit": k,
    }
    if filters:
        stage["filter"] = filters

    projection = {**idx.projection, "score": {"$meta": "vectorSearchScore"}}
    try:
        results = list(idx.col().aggregate([{"$vectorSearch": stage}, {"$project": projection}]))
    except OperationFailure as exc:
        log.error(
            "ATLAS VECTOR SEARCH UNAVAILABLE on %s (%s) -- DEGRADING TO BRUTE-FORCE COSINE. "
            "Quality and latency are NOT representative of production.",
            idx.name,
            exc,
        )
        return _brute_force_cosine(embedding, k, exclude_ids, idx, filters)

    if not results:
        status = search_index_status(idx)
        if status != "READY":
            log.error(
                "VECTOR INDEX %s IS %s, NOT READY -- $vectorSearch returned nothing. "
                "DEGRADING TO BRUTE-FORCE COSINE.",
                idx.name,
                status,
            )
            return _brute_force_cosine(embedding, k, exclude_ids, idx, filters)

    if exclude_ids:
        blocked = set(exclude_ids)
        results = [r for r in results if r["_id"] not in blocked]
    return results


def _brute_force_cosine(
    embedding: list[float],
    k: int,
    exclude_ids: list[Any] | None,
    index: VectorIndex,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Offline/degraded path. Announced loudly by every caller."""
    import numpy as np

    query = np.asarray(embedding, dtype=np.float32)
    query /= np.linalg.norm(query) + 1e-12

    blocked = set(exclude_ids or [])
    projection = {**index.projection, index.path: 1}
    scored: list[tuple[float, dict[str, Any]]] = []

    for doc in index.col().find(filters or {}, projection):
        if doc["_id"] in blocked:
            continue
        raw = doc.pop(index.path, None)
        if not raw:
            continue
        vec = np.asarray(raw, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        # Atlas reports cosine score as (1 + cos) / 2; mirror it so thresholds match.
        doc["score"] = float((1.0 + float(query @ vec)) / 2.0)
        scored.append((doc["score"], doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]

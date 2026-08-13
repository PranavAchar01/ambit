"""Turn the `findings` log into searchable episodic memory.

Findings written before this existed carry no embedding, so their frames are
re-embedded from the copies already in GridFS -- no image is re-uploaded and no
inference is re-run. Then the second Atlas vector index is created and polled to
READY, exactly as the model router is.
"""

from __future__ import annotations

import argparse
import logging

from bson import ObjectId

from gridsight.agent.graph import load_image
from gridsight.db.mongo import ensure_vector_index, finding_recall_index, findings_col
from gridsight.embed import get_embedder

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
for _n in ("httpx", "urllib3", "pymongo", "root", "timm"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("gridsight.recall")


def backfill(limit: int | None = None) -> tuple[int, int]:
    """Embed every finding that has a frame in GridFS but no vector yet."""
    query = {
        "embedding": {"$in": [None, []]},
        "uploaded_image_id": {"$ne": None},
    }
    cursor = findings_col().find(query, {"uploaded_image_id": 1})
    todo = list(cursor if limit is None else cursor.limit(limit))
    if not todo:
        log.info("no findings need backfilling")
        return 0, 0

    embedder = get_embedder()
    done = failed = 0
    for doc in todo:
        try:
            image = load_image(str(doc["uploaded_image_id"]))
            vector = embedder.embed_one(image)
        except Exception as exc:  # noqa: BLE001 - a missing frame must not abort the run
            log.warning("finding %s: cannot re-embed (%s)", doc["_id"], exc)
            failed += 1
            continue
        findings_col().update_one(
            {"_id": ObjectId(str(doc["_id"]))},
            {"$set": {"embedding": [float(x) for x in vector]}},
        )
        done += 1
        if done % 10 == 0:
            log.info("backfilled %d/%d", done, len(todo))
    return done, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill finding embeddings and build the recall index")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    done, failed = backfill(args.limit)
    log.info("backfill complete: %d embedded, %d skipped", done, failed)

    total = findings_col().count_documents({})
    with_vec = findings_col().count_documents({"embedding": {"$exists": True, "$ne": []}})
    log.info("findings with an embedding: %d/%d", with_vec, total)

    if not args.skip_index:
        idx = finding_recall_index()
        status = ensure_vector_index(idx)
        log.info("index %s on %s -> %s", idx.name, idx.collection, status)

    print(f"\nepisodic memory: {with_vec}/{total} findings searchable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

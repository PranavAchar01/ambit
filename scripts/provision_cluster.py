"""Provision an empty Atlas cluster so Ambit can run against it.

Creates the collections, their b-tree indexes, and both Atlas vector indexes,
then blocks until each reports READY. Safe to re-run: every step is idempotent.

This exists as a script rather than a one-off because a `$vectorSearch` against
a PENDING index returns an empty result set instead of an error -- a freshly
migrated cluster would refuse every frame and look like a routing bug rather
than an index that had not finished building. Provisioning has to be a step you
can run and verify on its own.

    uv run python scripts/provision_cluster.py --uri-env ATLAS_BUILDFEST_URI

Data migration is deliberately NOT here. This only prepares the destination.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--uri-env",
        default="ATLAS_BUILDFEST_URI",
        help="env var holding the target cluster URI (default: ATLAS_BUILDFEST_URI)",
    )
    ap.add_argument("--db", default=None, help="database name (defaults to MONGODB_DB)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait per index")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    target = os.environ.get(args.uri_env, "").strip()
    if not target:
        print(f"{args.uri_env} is not set in .env", file=sys.stderr)
        return 1

    # Point the ordinary settings at the target BEFORE anything reads them --
    # get_settings() is lru_cached, so importing first would pin the old cluster.
    os.environ["MONGODB_URI"] = target
    if args.db:
        os.environ["MONGODB_DB"] = args.db

    from gridsight.config import get_settings
    from gridsight.db.mongo import (
        ensure_collections,
        ensure_vector_index,
        finding_recall_index,
        get_db,
        model_router_index,
        search_index_status,
    )

    get_settings.cache_clear()
    settings = get_settings()
    host = target.split("@")[-1].split("/")[0]
    print(f"target   {host}")
    print(f"database {settings.mongodb_db}")
    print(f"dims     {settings.embed_dim}\n")

    db = get_db()
    before = set(db.list_collection_names())

    ensure_collections()
    after = set(db.list_collection_names())
    created = sorted(after - before)
    print(f"collections present: {sorted(after)}")
    if created:
        print(f"  created: {created}")

    for index in (model_router_index(), finding_recall_index()):
        status = search_index_status(index)
        if status == "READY":
            print(f"\n{index.name:<22} already READY")
            continue
        print(f"\n{index.name:<22} {status or 'absent'} -> building (this can take minutes)")
        final = ensure_vector_index(index, timeout_s=args.timeout)
        print(f"{index.name:<22} {final}")
        if final != "READY":
            print(f"  {index.name} did not reach READY; queries would return empty", file=sys.stderr)
            return 1

    print("\nready. counts on the target:")
    for name in sorted(after):
        print(f"  {name:<20} {db[name].count_documents({})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

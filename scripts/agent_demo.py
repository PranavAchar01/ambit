"""Phase 4 evidence: drive the LangGraph agent through all three scenarios.

(a) an in-registry insulator frame  -> routes to the insulator specialist
(b) a corroded-metal frame          -> a different specialist wins
(c) a rail frame, withheld from training -> the agent refuses, cold-starts a
    specialist from reference images, and routes to that new model on retry.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image

from gridsight.agent.graph import inspect_frame, store_pil
from gridsight.config import DATA_ROOT
from gridsight.db.mongo import models_col, vector_search
from gridsight.embed import get_embedder

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
for noisy in ("httpx", "urllib3", "pymongo", "open_clip", "root", "timm"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("gridsight.demo")

WITHHELD = "rail_surface"


def upload(path: Path, kind: str = "upload") -> str:
    with Image.open(path) as im:
        return store_pil(im.convert("RGB"), path.name, kind=kind)


def show(title: str, state: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for step in state.get("trace", []):
        print(f"  [{step['node']:<12}] {step['message']}")
    print("-" * 100)
    print(f"  verdict        : {state.get('verdict')}")
    print(f"  asset_class    : {state.get('asset_class')}")
    print(f"  routed model   : {state.get('routed_model_name')} ({state.get('routed_model_id')})")
    print(f"  routing score  : {state.get('routing_score')}")
    print(f"  anomaly score  : {state.get('raw_anomaly_score')} (threshold {state.get('image_threshold')})")
    print(f"  severity       : {state.get('severity')}   regions: {len(state.get('bbox_regions', []))}")
    print(f"  decided by     : {state.get('decision_source')}")
    print(f"  reason         : {state.get('decision_reason')}")
    cands = state.get("candidates", [])
    if cands:
        print("  candidates     :")
        for c in cands:
            print(f"      {c['score']:.4f} (gate {c['gate']:.4f})  {c['name']:<36} [{c['asset_class']}]")
    if state.get("cold_start_info"):
        ci = state["cold_start_info"]
        print(
            f"  COLD START     : {ci['name']} | {ci['backbone']} | {ci['reference_images']} refs "
            f"| {ci['seconds']}s | {ci['weights_mb']} MB | named by {ci['naming_source']}"
        )
        print(f"  threshold rule : {ci['threshold_policy']}")
    print(f"  narrative ({state.get('narrative_source')}):")
    for line in (state.get("agent_narrative") or "").splitlines():
        print(f"      {line}")
    print(f"  finding id     : {state.get('finding_id')}   latency: {state.get('latency_ms')} ms")


def wait_until_routable(asset_class: str, probe: Path, timeout_s: float = 90.0) -> bool:
    """Atlas indexes a newly inserted vector asynchronously; wait for it to appear."""
    with Image.open(probe) as im:
        vector = [float(x) for x in get_embedder().embed_one(im.convert("RGB"))]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hits = vector_search(vector, k=5)
        if any(h["asset_class"] == asset_class for h in hits):
            return True
        time.sleep(3)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=int, default=8)
    args = parser.parse_args()

    stamp = int(time.time())

    # (a) in-registry insulator ------------------------------------------------
    insulator = sorted((DATA_ROOT / "insulator" / "test" / "defect").glob("*.png"))[0]
    show(
        f"(a) IN-REGISTRY INSULATOR FRAME  --  {insulator}",
        inspect_frame(upload(insulator), thread_id=f"demo-a-{stamp}"),
    )

    # (b) a different class must win ------------------------------------------
    corrosion = sorted((DATA_ROOT / "corrosion" / "test" / "defect").glob("*.png"))[0]
    show(
        f"(b) CORRODED METAL FRAME  --  {corrosion}",
        inspect_frame(upload(corrosion), thread_id=f"demo-b-{stamp}"),
    )

    # (c) withheld class: refuse -> cold start -> route on retry ---------------
    rail = sorted((DATA_ROOT / WITHHELD / "test" / "defect").glob("*.png"))[0]
    before = sorted(models_col().distinct("asset_class"))
    print(f"\n\nregistry before cold start: {before}")

    frame_id = upload(rail)
    show(
        f"(c1) WITHHELD RAIL FRAME, NO REFERENCES  --  {rail}",
        inspect_frame(frame_id, thread_id=f"demo-c1-{stamp}"),
    )

    refs = sorted((DATA_ROOT / WITHHELD / "train" / "good").glob("*.png"))[: args.references]
    ref_ids = [upload(p, kind="coldstart-reference") for p in refs]
    show(
        f"(c2) SAME FRAME + {len(refs)} REFERENCE IMAGES  --  cold start",
        inspect_frame(
            frame_id,
            thread_id=f"demo-c2-{stamp}",
            asset_class_hint="railroad track surface",
            reference_image_ids=ref_ids,
        ),
    )

    after = sorted(models_col().distinct("asset_class"))
    print(f"\nregistry after cold start : {after}")
    print(f"newly covered classes     : {sorted(set(after) - set(before))}")

    new_class = next(iter(sorted(set(after) - set(before))), None)
    if new_class:
        visible = wait_until_routable(new_class, rail)
        print(f"new model visible to $vectorSearch: {visible}")

    show(
        "(c3) SAME FRAME AGAIN, NO REFERENCES  --  must now route, not cold-start",
        inspect_frame(frame_id, thread_id=f"demo-c3-{stamp}"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

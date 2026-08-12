"""Cold start and checkpoint resume -- the two claims that are easiest to fake."""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from typing import Any

import pytest
from bson import ObjectId
from langgraph.graph import END, START, StateGraph
from PIL import Image
from typing_extensions import TypedDict

from conftest import requires_atlas
from gridsight.train.fewshot import choose_backbone, fit_fewshot


class _ResumeState(TypedDict, total=False):
    done: list[str]
    boom: bool


@requires_atlas
def test_checkpointer_resumes_a_thread_after_a_node_raises() -> None:
    """A crash mid-graph must leave earlier nodes committed, then continue."""
    from gridsight.agent.graph import make_saver

    calls: list[str] = []

    def first(state: _ResumeState) -> dict[str, Any]:
        calls.append("first")
        return {"done": [*state.get("done", []), "first"]}

    def second(state: _ResumeState) -> dict[str, Any]:
        calls.append("second")
        if state.get("boom"):
            raise RuntimeError("simulated crash")
        return {"done": [*state.get("done", []), "second"]}

    graph = StateGraph(_ResumeState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    saver = make_saver()
    app = graph.compile(checkpointer=saver)
    thread = f"test-resume-{uuid.uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread}}

    with pytest.raises(RuntimeError, match="simulated crash"):
        app.invoke({"done": [], "boom": True}, config=config)

    # `first` is committed to MongoDB and the graph is parked on `second`
    snapshot = app.get_state(config)
    assert snapshot.values["done"] == ["first"]
    assert "second" in snapshot.next

    # resuming must not re-run `first`
    calls.clear()
    app.update_state(config, {"boom": False})
    final = app.invoke(None, config=config)
    assert final["done"] == ["first", "second"]
    assert "first" not in calls, "resume re-ran an already-committed node"

    from gridsight.db.mongo import get_db

    db = get_db()
    assert db["checkpoints"].count_documents({"thread_id": thread}) > 0
    db["checkpoints"].delete_many({"thread_id": thread})
    db["checkpoint_writes"].delete_many({"thread_id": thread})


@requires_atlas
def test_fleet_training_graph_state_is_readable_by_thread_id() -> None:
    """The Phase 3 training loop is the thing that has to survive kill -9."""
    from gridsight.train.train_class import build_training_graph, make_saver

    app = build_training_graph(make_saver())
    snapshot = app.get_state({"configurable": {"thread_id": "fleet-v3"}})
    assert snapshot.values.get("completed"), "no completed classes recorded on the training thread"
    assert snapshot.values["pending"] == [], "the fleet-v3 run did not finish"


def test_backbone_choice_falls_back_to_padim_for_tiny_reference_sets() -> None:
    assert choose_backbone(2) == "padim"
    assert choose_backbone(3) == "padim"
    assert choose_backbone(4) == "patchcore"
    assert choose_backbone(12) == "patchcore"


def _reference_images(data_root: Path, cls: str, n: int) -> list[Image.Image]:
    out: list[Image.Image] = []
    for p in sorted((data_root / cls / "train" / "good").glob("*.png"))[:n]:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


def test_fewshot_fit_produces_a_usable_model(data_root: Path, scraped_classes: list[str]) -> None:
    """Cold start must yield a model that scores, with a threshold it derived itself."""
    from gridsight.inference import run_inference

    refs = _reference_images(data_root, scraped_classes[0], 6)
    module, cfg, info = fit_fewshot(refs)

    assert info["backbone"] == "patchcore"
    assert info["reference_images"] == 6
    assert info["image_threshold"] > 0
    assert "normal envelope" in info["threshold_policy"]

    # the references it was fitted on must sit at or below its own threshold
    scored = run_inference(module, cfg, refs[0])
    assert scored.raw_score <= info["image_threshold"] + 1e-4
    assert scored.anomaly_map.shape == (cfg.image_size, cfg.image_size)


def test_fewshot_refuses_an_empty_reference_set() -> None:
    with pytest.raises(ValueError, match="at least one reference image"):
        fit_fewshot([])


@requires_atlas
def test_cold_start_registers_a_genuinely_new_model(
    data_root: Path, scraped_classes: list[str], registry_models: list[dict[str, Any]]
) -> None:
    """The unroutable branch must mint a model that is then routable."""
    from gridsight.agent.graph import cold_start, store_pil
    from gridsight.db.mongo import images_bucket, models_col, weights_bucket

    registered = {d["asset_class"] for d in registry_models}
    withheld = [c for c in scraped_classes if c not in registered]
    if not withheld:
        pytest.skip("no withheld class available to cold-start")
    cls = withheld[0]

    refs = _reference_images(data_root, cls, 6)
    ref_ids = [store_pil(img, f"test_ref_{i}.png", kind="test") for i, img in enumerate(refs)]
    before = models_col().count_documents({})

    result = cold_start(
        {
            "reference_image_ids": ref_ids,
            "asset_class_hint": f"{cls} test fixture",
            "uploaded_image_id": ref_ids[0],
        }
    )
    new_id = result["routed_model_id"]
    try:
        assert models_col().count_documents({}) == before + 1
        doc = models_col().find_one({"_id": ObjectId(new_id)})
        assert doc is not None
        assert doc["created_by"] == "agent-coldstart"
        assert len(doc["embedding"]) == 512
        assert doc["weights_file_id"] is not None
        assert doc["routing_threshold"] > 0, "a cold-started model needs its own coverage gate"
        assert doc["training_samples"] == 6

        # its weights really are in GridFS and really do reload
        from gridsight.train.store import deserialize_patchcore, get_weights

        reloaded, _ = deserialize_patchcore(get_weights(doc["weights_file_id"]))
        assert reloaded.model.memory_bank.numel() > 0
    finally:
        doc = models_col().find_one({"_id": ObjectId(new_id)})
        if doc:
            buckets = (
                (weights_bucket(), "weights_file_id"),
                (images_bucket(), "reference_image_id"),
            )
            for bucket, key in buckets:
                if doc.get(key):
                    with contextlib.suppress(Exception):
                        bucket.delete(doc[key])
            models_col().delete_one({"_id": doc["_id"]})
        for rid in ref_ids:
            with contextlib.suppress(Exception):
                images_bucket().delete(ObjectId(rid))

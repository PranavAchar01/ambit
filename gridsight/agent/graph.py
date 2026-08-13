"""Phase 4 -- the routing agent.

    embed_frame -> route -> decide -+-> infer ------> narrate -> persist
                                    |                   ^
                                    +-> cold_start -----+
                                    |
                                    +-> narrate (refusal, no references supplied)

Every node checkpoints to MongoDB, so a run is resumable by thread_id. Images are
put in GridFS *before* the graph is invoked and referenced by id, which keeps the
checkpointed state small and makes each frame independently replayable.
"""

from __future__ import annotations

import io
import logging
import operator
import time
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

import numpy as np
from bson import ObjectId
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from PIL import Image

from gridsight.agent.llm import adjudicate_route, name_new_model, narrate_finding, slugify
from gridsight.config import get_settings
from gridsight.db.mongo import (
    findings_col,
    get_client,
    images_bucket,
    models_col,
    vector_search,
)
from gridsight.embed import centroid, get_embedder
from gridsight.inference import MODEL_CACHE, heatmap_png, run_inference
from gridsight.train.fewshot import fit_fewshot
from gridsight.train.store import put_weights, serialize_patchcore
from gridsight.train.train_class import ROUTING_GATE_PERCENTILE

log = logging.getLogger("gridsight.agent")

MAX_CANDIDATES = 5

#: How far below a model's own gate a frame may score and still be worth asking
#: the LLM about. Inside this band the vector score alone is not decisive.
AMBIGUOUS_MARGIN = 0.04


class AgentState(TypedDict, total=False):
    # --- inputs -----------------------------------------------------------
    uploaded_image_id: str
    asset_class_hint: str | None
    reference_image_ids: list[str]

    # --- embed_frame ------------------------------------------------------
    embedding: list[float]

    # --- route ------------------------------------------------------------
    candidates: list[dict[str, Any]]
    routing_score: float

    # --- decide -----------------------------------------------------------
    decision: str
    decision_source: str
    decision_reason: str
    routed_model_id: str | None
    routed_model_name: str | None
    asset_class: str | None

    # --- infer / cold_start ----------------------------------------------
    raw_anomaly_score: float
    anomaly_score: float
    verdict: str
    severity: float
    image_threshold: float
    bbox_regions: list[dict[str, Any]]
    heatmap_id: str | None
    reference_image_id: str | None
    cold_start_info: dict[str, Any]

    # --- narrate / persist ------------------------------------------------
    agent_narrative: str
    narrative_source: str
    finding_id: str
    latency_ms: int

    # --- observability ----------------------------------------------------
    trace: Annotated[list[dict[str, Any]], operator.add]
    started_ms: float
    error: str | None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _step(node: str, message: str, **extra: Any) -> dict[str, Any]:
    entry = {"node": node, "message": message, "at": time.time()}
    entry.update(extra)
    return entry


def load_image(image_id: str) -> Image.Image:
    buf = io.BytesIO()
    images_bucket().download_to_stream(ObjectId(image_id), buf)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB").copy()


def store_image(blob: bytes, filename: str, **metadata: Any) -> str:
    return str(images_bucket().upload_from_stream(filename, blob, metadata=metadata))


def store_pil(image: Image.Image, filename: str, **metadata: Any) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=True)
    return store_image(buf.getvalue(), filename, **metadata)


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


def embed_frame(state: AgentState) -> dict[str, Any]:
    """OpenCLIP-embed the uploaded frame into the registry's routing space."""
    image = load_image(state["uploaded_image_id"])
    vector = get_embedder().embed_one(image)
    return {
        "embedding": [float(x) for x in vector],
        "started_ms": state.get("started_ms") or time.time() * 1000.0,
        "trace": [
            _step(
                "embed_frame",
                f"Embedded frame with OpenCLIP ViT-B-32 ({len(vector)}-d)",
                dim=len(vector),
            )
        ],
    }


def route(state: AgentState) -> dict[str, Any]:
    """$vectorSearch the registry for specialists trained on similar imagery."""
    candidates = vector_search(state["embedding"], k=MAX_CANDIDATES)
    settings = get_settings()
    slim = [
        {
            "model_id": str(c["_id"]),
            "name": c["name"],
            "asset_class": c["asset_class"],
            "score": round(float(c["score"]), 4),
            "training_samples": c.get("training_samples"),
            "metrics": c.get("metrics", {}),
            # A model only accepts frames inside the coverage envelope it learned;
            # the global threshold is an additional floor, never a relaxation.
            "gate": round(max(settings.route_threshold, float(c.get("routing_threshold") or 0.0)), 4),
        }
        for c in candidates
    ]
    top = slim[0]["score"] if slim else 0.0
    summary = (
        f"Vector search returned {len(slim)} candidates; best {slim[0]['name']} @ {top:.4f}"
        if slim
        else "Vector search returned no candidates -- the registry is empty"
    )
    return {
        "candidates": slim,
        "routing_score": top,
        "trace": [_step("route", summary, candidates=slim)],
    }


def decide(state: AgentState) -> dict[str, Any]:
    """Threshold, then escalate to the LLM only inside the ambiguous band."""
    settings = get_settings()
    candidates = state.get("candidates", [])
    top = state.get("routing_score", 0.0)

    if not candidates:
        return {
            "decision": "unroutable",
            "decision_source": "threshold",
            "decision_reason": "The registry contains no models at all.",
            "routed_model_id": None,
            "trace": [_step("decide", "No candidates -- unroutable", decision="unroutable")],
        }

    best = candidates[0]
    gate = float(best.get("gate") or settings.route_threshold)

    if top >= gate:
        reason = (
            f"Top candidate {best['name']} scored {top:.4f}, at or above its own coverage "
            f"gate of {gate:.4f} (the 5th percentile of its training set's similarity to its "
            f"centroid)."
        )
        return {
            "decision": "route",
            "decision_source": "threshold",
            "decision_reason": reason,
            "routed_model_id": best["model_id"],
            "routed_model_name": best["name"],
            "asset_class": best["asset_class"],
            "trace": [
                _step(
                    "decide",
                    f"Routed to {best['name']} ({best['asset_class']}) at {top:.4f}",
                    decision="route",
                    source="threshold",
                )
            ],
        }

    if top >= gate - AMBIGUOUS_MARGIN:
        verdict, source = adjudicate_route(candidates, top, gate)
        if not verdict.should_cold_start and verdict.chosen_model_name:
            chosen = next((c for c in candidates if c["name"] == verdict.chosen_model_name), None)
            if chosen is not None:
                return {
                    "decision": "route",
                    "decision_source": source,
                    "decision_reason": verdict.reasoning,
                    "routed_model_id": chosen["model_id"],
                    "routed_model_name": chosen["name"],
                    "asset_class": chosen["asset_class"],
                    "trace": [
                        _step(
                            "decide",
                            f"Ambiguous at {top:.4f}; {source} adjudicated -> {chosen['name']}",
                            decision="route",
                            source=source,
                            reasoning=verdict.reasoning,
                        )
                    ],
                }
        return {
            "decision": "unroutable",
            "decision_source": source,
            "decision_reason": verdict.reasoning,
            "routed_model_id": None,
            "trace": [
                _step(
                    "decide",
                    f"Ambiguous at {top:.4f}; {source} refused to route",
                    decision="unroutable",
                    source=source,
                    reasoning=verdict.reasoning,
                )
            ],
        }

    reason = (
        f"Best candidate {best['name']} scored only {top:.4f}, more than {AMBIGUOUS_MARGIN} below "
        f"its coverage gate of {gate:.4f}. No registry model covers this asset class."
    )
    return {
        "decision": "unroutable",
        "decision_source": "threshold",
        "decision_reason": reason,
        "routed_model_id": None,
        "trace": [
            _step(
                "decide",
                f"{top:.4f} is below {best['name']}'s gate of {gate:.4f} -- refusing to guess",
                decision="unroutable",
                gate=gate,
            )
        ],
    }


def after_decide(state: AgentState) -> str:
    if state.get("decision") == "route":
        return "infer"
    if state.get("reference_image_ids"):
        return "cold_start"
    return "narrate"


def cold_start(state: AgentState) -> dict[str, Any]:
    """Fit a fresh specialist from the supplied references and register it."""
    ref_ids = state.get("reference_image_ids", [])
    images = [load_image(i) for i in ref_ids]
    module, cfg, info = fit_fewshot(images)

    naming, name_source = name_new_model(state.get("asset_class_hint"), len(images), info["backbone"])
    asset_class = slugify(naming.asset_class).replace("-", "_")

    # Route future frames of this class to the new model: the registry vector is
    # the centroid of the very references it was fitted on.
    vectors = get_embedder().embed_images(images)
    embedding = centroid(vectors)
    # Give the new model the same kind of coverage gate the seeded ones carry, so
    # it participates in routing on equal terms from its very first request.
    atlas_scores = (1.0 + (vectors @ embedding)) / 2.0
    routing_gate = float(np.percentile(atlas_scores, ROUTING_GATE_PERCENTILE))

    blob = serialize_patchcore(module, cfg)
    weights_id = put_weights(
        blob,
        f"{asset_class}_coldstart.pt",
        metadata={"asset_class": asset_class, "created_by": "agent-coldstart"},
    )

    golden_idx = int((vectors @ embedding).argmax())
    reference_id = store_pil(
        images[golden_idx], f"reference_{asset_class}.png", kind="reference", asset_class=asset_class
    )

    name = naming.name
    if models_col().find_one({"name": name}):
        name = f"{name}-{int(time.time())}"

    doc = {
        "name": name,
        "asset_class": asset_class,
        "backbone": f"{info['backbone']}/{cfg.backbone if info['backbone'] == 'patchcore' else 'resnet18'}",
        "embedding": [float(x) for x in embedding],
        "embedding_count": len(images),
        "routing_threshold": routing_gate,
        "image_threshold": float(info["image_threshold"]),
        "pixel_threshold": float(info["pixel_threshold"]),
        "weights_file_id": weights_id,
        "weights_bytes": len(blob),
        "reference_image_id": ObjectId(reference_id),
        "training_samples": len(images),
        "metrics": {"image_auroc": None, "pixel_auroc": None},
        "created_at": datetime.now(UTC),
        "created_by": "agent-coldstart",
        "provenance": {
            "source": "agent-coldstart",
            "reference_image_ids": ref_ids,
            "asset_class_hint": state.get("asset_class_hint"),
            "naming_source": name_source,
            "config": cfg.to_dict(),
            **info,
        },
    }
    result = models_col().insert_one(doc)
    model_id = str(result.inserted_id)

    log.info(
        "cold-started %s (%s) from %d references -> %s",
        name,
        asset_class,
        len(images),
        model_id,
    )
    return {
        "decision": "route",
        "routed_model_id": model_id,
        "routed_model_name": name,
        "asset_class": asset_class,
        "cold_start_info": {
            **info,
            "model_id": model_id,
            "name": name,
            "naming_source": name_source,
            "weights_mb": round(len(blob) / 1e6, 3),
        },
        "trace": [
            _step(
                "cold_start",
                f"No registry model matched. Fitted {name} from {len(images)} reference "
                f"images in {info['seconds']:.1f}s and registered it.",
                model_id=model_id,
                name=name,
                asset_class=asset_class,
                backbone=info["backbone"],
                seconds=info["seconds"],
            )
        ],
    }


def infer(state: AgentState) -> dict[str, Any]:
    """Load the winning specialist from GridFS and score the frame."""
    model_id = state["routed_model_id"]
    doc = models_col().find_one({"_id": ObjectId(str(model_id))})
    if doc is None:
        return {
            "error": f"routed model {model_id} vanished from the registry",
            "trace": [_step("infer", f"Model {model_id} not found", error=True)],
        }

    module, cfg = MODEL_CACHE.get(str(doc["weights_file_id"]))
    image = load_image(state["uploaded_image_id"])
    result = run_inference(module, cfg, image)

    heat_id = store_image(
        heatmap_png(result.anomaly_map, image.size),
        f"heatmap_{state['uploaded_image_id']}.png",
        kind="heatmap",
        asset_class=doc["asset_class"],
    )
    verdict = "defect" if result.is_defect else "nominal"
    regions = [r.to_dict() for r in result.regions]

    return {
        "raw_anomaly_score": round(result.raw_score, 5),
        "anomaly_score": round(result.normalized_score, 5),
        "image_threshold": round(result.image_threshold, 5),
        "verdict": verdict,
        "severity": round(result.severity(), 4),
        "bbox_regions": regions,
        "heatmap_id": heat_id,
        "reference_image_id": str(doc.get("reference_image_id")) if doc.get("reference_image_id") else None,
        "asset_class": doc["asset_class"],
        "trace": [
            _step(
                "infer",
                f"{doc['name']} scored {result.raw_score:.3f} vs threshold "
                f"{result.image_threshold:.3f} -> {verdict} ({len(regions)} region(s))",
                verdict=verdict,
                raw_score=round(result.raw_score, 4),
                threshold=round(result.image_threshold, 4),
                regions=len(regions),
                cache_hits=MODEL_CACHE.hits,
                cache_misses=MODEL_CACHE.misses,
            )
        ],
    }


def narrate(state: AgentState) -> dict[str, Any]:
    """OpenAI writes the operator-facing finding."""
    settings = get_settings()
    candidates = state.get("candidates", [])
    verdict = state.get("verdict") or ("unroutable" if state.get("decision") != "route" else "nominal")

    image = load_image(state["uploaded_image_id"])
    context = {
        "verdict": verdict,
        "asset_class": state.get("asset_class") or state.get("asset_class_hint") or "unknown",
        "model_name": state.get("routed_model_name") or "none",
        "routing_score": state.get("routing_score", 0.0),
        "raw_score": state.get("raw_anomaly_score", 0.0),
        # For a refusal the meaningful number is the gate the frame failed to
        # clear, not the model's anomaly threshold (which was never computed).
        "threshold": (
            state.get("image_threshold")
            if verdict != "unroutable"
            else float(candidates[0]["gate"]) if candidates else settings.route_threshold
        ),
        "severity": state.get("severity", 0.0),
        "regions": state.get("bbox_regions", []),
        "top_candidate": candidates[0]["name"] if candidates else None,
        "width": image.width,
        "height": image.height,
    }
    text, source = narrate_finding(context)
    return {
        "verdict": verdict,
        "agent_narrative": text,
        "narrative_source": source,
        "trace": [_step("narrate", f"Narrative written by {source}", source=source)],
    }


def persist(state: AgentState) -> dict[str, Any]:
    """Write the findings document."""
    started = state.get("started_ms") or time.time() * 1000.0
    latency = int(time.time() * 1000.0 - started)

    # The frame's CLIP vector is already computed by embed_frame; persisting it is
    # what turns findings from a log into searchable episodic memory.
    embedding = state.get("embedding") or []

    doc = {
        "timestamp": datetime.now(UTC),
        "uploaded_image_id": ObjectId(state["uploaded_image_id"]),
        "embedding": [float(x) for x in embedding],
        "asset_class": state.get("asset_class") or state.get("asset_class_hint"),
        "routed_model_id": ObjectId(state["routed_model_id"]) if state.get("routed_model_id") else None,
        "routed_model_name": state.get("routed_model_name"),
        "routing_score": round(float(state.get("routing_score", 0.0)), 5),
        "anomaly_score": state.get("anomaly_score"),
        "raw_anomaly_score": state.get("raw_anomaly_score"),
        "verdict": state.get("verdict", "unroutable"),
        "severity": state.get("severity", 0.0),
        "bbox_regions": state.get("bbox_regions", []),
        "heatmap_id": ObjectId(state["heatmap_id"]) if state.get("heatmap_id") else None,
        "reference_image_id": (
            ObjectId(state["reference_image_id"]) if state.get("reference_image_id") else None
        ),
        "agent_narrative": state.get("agent_narrative", ""),
        "narrative_source": state.get("narrative_source"),
        "decision_source": state.get("decision_source"),
        "decision_reason": state.get("decision_reason"),
        "candidates": state.get("candidates", []),
        "cold_start_info": state.get("cold_start_info"),
        "latency_ms": latency,
    }
    result = findings_col().insert_one(doc)
    return {
        "finding_id": str(result.inserted_id),
        "latency_ms": latency,
        "trace": [
            _step("persist", f"Finding {result.inserted_id} written in {latency} ms", latency_ms=latency)
        ],
    }


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


def make_saver() -> MongoDBSaver:
    return MongoDBSaver(
        client=get_client(),
        db_name=get_settings().mongodb_db,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )


def build_graph(saver: MongoDBSaver | None = None) -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("embed_frame", embed_frame)
    graph.add_node("route", route)
    graph.add_node("decide", decide)
    graph.add_node("cold_start", cold_start)
    graph.add_node("infer", infer)
    graph.add_node("narrate", narrate)
    graph.add_node("persist", persist)

    graph.add_edge(START, "embed_frame")
    graph.add_edge("embed_frame", "route")
    graph.add_edge("route", "decide")
    graph.add_conditional_edges(
        "decide",
        after_decide,
        {"infer": "infer", "cold_start": "cold_start", "narrate": "narrate"},
    )
    graph.add_edge("cold_start", "infer")
    graph.add_edge("infer", "narrate")
    graph.add_edge("narrate", "persist")
    graph.add_edge("persist", END)
    return graph.compile(checkpointer=saver or make_saver())


_APP: Any = None


def get_app() -> Any:
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def inspect_frame(
    uploaded_image_id: str,
    thread_id: str,
    asset_class_hint: str | None = None,
    reference_image_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run the agent end-to-end on one frame."""
    initial: AgentState = {
        "uploaded_image_id": uploaded_image_id,
        "asset_class_hint": asset_class_hint,
        "reference_image_ids": reference_image_ids or [],
        "trace": [],
        "started_ms": time.time() * 1000.0,
    }
    return dict(get_app().invoke(initial, config={"configurable": {"thread_id": thread_id}}))

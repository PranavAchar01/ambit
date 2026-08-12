"""Phase 5 -- the GridSight inference/agent service."""

from __future__ import annotations

import io
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from gridfs.errors import NoFile
from PIL import Image, UnidentifiedImageError

from gridsight.agent.graph import build_graph, get_app, make_saver, store_pil
from gridsight.analytics import compute_trends
from gridsight.config import get_settings
from gridsight.db.mongo import (
    datasets_col,
    ensure_collections,
    findings_col,
    images_bucket,
    models_col,
    search_index_status,
)
from gridsight.train.store import storage_report

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
for _noisy in ("httpx", "urllib3", "pymongo"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("gridsight.api")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(
    title="GridSight",
    version="1.0.0",
    description="Vector-routed machine-vision model registry for infrastructure inspection",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    ensure_collections()
    status = search_index_status(get_settings().vector_index_name)
    if status != "READY":
        log.error(
            "Atlas vector index is %s, not READY -- routing will fall back to brute-force cosine",
            status,
        )
    log.info(
        "GridSight API up | vector index %s | %d models registered", status, models_col().count_documents({})
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _read_image(upload: UploadFile) -> Image.Image:
    blob = await upload.read()
    if not blob:
        raise HTTPException(400, "empty upload")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
    try:
        with Image.open(io.BytesIO(blob)) as im:
            return im.convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise HTTPException(400, f"{upload.filename!r} is not a readable image") from exc


def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise HTTPException(400, f"{value!r} is not a valid id") from exc


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    """Make a Mongo document JSON-safe, dropping the bulky routing vector."""
    out: dict[str, Any] = {}
    for key, value in doc.items():
        if key == "embedding":
            out["embedding_dim"] = len(value) if value else 0
            continue
        if isinstance(value, ObjectId):
            out[key] = str(value)
        elif isinstance(value, datetime):
            out[key] = value.astimezone(UTC).isoformat()
        elif isinstance(value, dict):
            out[key] = _serialize(value)
        elif isinstance(value, list):
            out[key] = [
                _serialize(v) if isinstance(v, dict) else (str(v) if isinstance(v, ObjectId) else v)
                for v in value
            ]
        else:
            out[key] = value
    if "_id" in out:
        out["id"] = out.pop("_id")
    return out


def _result_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": state.get("verdict", "unroutable"),
        "asset_class": state.get("asset_class") or state.get("asset_class_hint"),
        "routed_model": {
            "id": state.get("routed_model_id"),
            "name": state.get("routed_model_name"),
        },
        "routing_score": state.get("routing_score", 0.0),
        "anomaly_score": state.get("anomaly_score"),
        "raw_anomaly_score": state.get("raw_anomaly_score"),
        "image_threshold": state.get("image_threshold"),
        "severity": state.get("severity", 0.0),
        "bbox_regions": state.get("bbox_regions", []),
        "candidates": state.get("candidates", []),
        "decision_source": state.get("decision_source"),
        "decision_reason": state.get("decision_reason"),
        "cold_start": state.get("cold_start_info"),
        "narrative": state.get("agent_narrative", ""),
        "narrative_source": state.get("narrative_source"),
        "uploaded_image_url": f"/image/{state.get('uploaded_image_id')}",
        "heatmap_url": f"/image/{state['heatmap_id']}" if state.get("heatmap_id") else None,
        "reference_image_url": (
            f"/image/{state['reference_image_id']}" if state.get("reference_image_id") else None
        ),
        "finding_id": state.get("finding_id"),
        "latency_ms": state.get("latency_ms"),
        "thread_id": state.get("thread_id"),
        "trace": state.get("trace", []),
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    status = search_index_status(get_settings().vector_index_name)
    return {
        "status": "ok",
        "vector_index": {"name": get_settings().vector_index_name, "status": status},
        "models": models_col().count_documents({}),
        "findings": findings_col().count_documents({}),
        "storage": storage_report(),
    }


@app.post("/inspect")
async def inspect(
    image: UploadFile = File(...),
    asset_class: str | None = Form(None),
) -> JSONResponse:
    """Run the routing agent on a drone frame."""
    frame = await _read_image(image)
    image_id = store_pil(frame, image.filename or "frame.png", kind="upload")
    thread_id = f"inspect-{uuid.uuid4().hex[:12]}"

    state = get_app().invoke(
        {
            "uploaded_image_id": image_id,
            "asset_class_hint": asset_class,
            "reference_image_ids": [],
            "trace": [],
            "started_ms": time.time() * 1000.0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    payload = _result_payload({**state, "thread_id": thread_id, "uploaded_image_id": image_id})
    log.info(
        "inspect verdict=%s asset_class=%s model=%s routing=%.4f latency=%sms",
        payload["verdict"],
        payload["asset_class"],
        payload["routed_model"]["name"],
        payload["routing_score"],
        payload["latency_ms"],
    )
    return JSONResponse(payload)


@app.post("/inspect/stream")
async def inspect_stream(
    image: UploadFile = File(...),
    asset_class: str | None = Form(None),
) -> StreamingResponse:
    """Same as /inspect but streams each agent step as it completes (SSE)."""
    frame = await _read_image(image)
    image_id = store_pil(frame, image.filename or "frame.png", kind="upload")
    thread_id = f"inspect-{uuid.uuid4().hex[:12]}"

    def events() -> Iterator[str]:
        merged: dict[str, Any] = {
            "uploaded_image_id": image_id,
            "thread_id": thread_id,
            "trace": [],
        }
        yield _sse("start", {"thread_id": thread_id, "uploaded_image_url": f"/image/{image_id}"})
        try:
            app_ = build_graph(make_saver())
            for chunk in app_.stream(
                {
                    "uploaded_image_id": image_id,
                    "asset_class_hint": asset_class,
                    "reference_image_ids": [],
                    "trace": [],
                    "started_ms": time.time() * 1000.0,
                },
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    merged.update({k: v for k, v in update.items() if k != "trace"})
                    for entry in update.get("trace", []):
                        merged["trace"].append({"node": node, **entry})
                        yield _sse("step", {"node": node, **entry})
            yield _sse("result", _result_payload(merged))
        except Exception as exc:  # noqa: BLE001 - surface failures to the client
            log.exception("inspect stream failed")
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/coldstart")
async def coldstart(
    references: list[UploadFile] = File(...),
    asset_class: str = Form(...),
    frame: UploadFile | None = File(None),
) -> JSONResponse:
    """Explicitly drive the cold-start branch, then re-score the original frame."""
    if not references:
        raise HTTPException(400, "at least one reference image is required")

    ref_ids: list[str] = []
    for upload in references:
        img = await _read_image(upload)
        ref_ids.append(store_pil(img, upload.filename or "reference.png", kind="coldstart-reference"))

    if frame is not None:
        probe = await _read_image(frame)
        probe_id = store_pil(probe, frame.filename or "frame.png", kind="upload")
    else:
        # Without a query frame, score the first reference so the caller still
        # gets a full result for the model that was just minted.
        probe_id = ref_ids[0]

    thread_id = f"coldstart-{uuid.uuid4().hex[:12]}"
    state = get_app().invoke(
        {
            "uploaded_image_id": probe_id,
            "asset_class_hint": asset_class,
            "reference_image_ids": ref_ids,
            "trace": [],
            "started_ms": time.time() * 1000.0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    payload = _result_payload({**state, "thread_id": thread_id, "uploaded_image_id": probe_id})
    log.info(
        "coldstart asset_class=%s references=%d model=%s",
        asset_class,
        len(ref_ids),
        payload["routed_model"]["name"],
    )
    return JSONResponse(payload)


@app.get("/models")
def list_models() -> dict[str, Any]:
    docs = [_serialize(d) for d in models_col().find().sort("created_at", 1)]
    for d in docs:
        d["reference_image_url"] = (
            f"/image/{d['reference_image_id']}" if d.get("reference_image_id") else None
        )
    return {
        "count": len(docs),
        "models": docs,
        "vector_index": search_index_status(get_settings().vector_index_name),
    }


@app.get("/datasets")
def list_datasets() -> dict[str, Any]:
    return {"datasets": [_serialize(d) for d in datasets_col().find()]}


@app.get("/findings")
def list_findings(
    asset_class: str | None = Query(None),
    verdict: str | None = Query(None),
    since: str | None = Query(None, description="ISO date lower bound"),
    until: str | None = Query(None, description="ISO date upper bound"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    query: dict[str, Any] = {}
    if asset_class:
        query["asset_class"] = asset_class
    if verdict:
        query["verdict"] = verdict
    bounds: dict[str, Any] = {}
    for key, raw in (("$gte", since), ("$lte", until)):
        if raw:
            try:
                bounds[key] = datetime.fromisoformat(raw).replace(tzinfo=UTC)
            except ValueError as exc:
                raise HTTPException(400, f"{raw!r} is not an ISO datetime") from exc
    if bounds:
        query["timestamp"] = bounds

    total = findings_col().count_documents(query)
    cursor = (
        findings_col()
        .find(query, {"candidates": 0})
        .sort("timestamp", -1)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    items = []
    for doc in cursor:
        item = _serialize(doc)
        item["heatmap_url"] = f"/image/{item['heatmap_id']}" if item.get("heatmap_id") else None
        item["uploaded_image_url"] = (
            f"/image/{item['uploaded_image_id']}" if item.get("uploaded_image_id") else None
        )
        items.append(item)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
        "findings": items,
    }


@app.get("/trends")
def trends(days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
    """Fleet health: defect rate over time, severity spread, and what is degrading fastest."""
    return compute_trends(findings_col(), models_col(), days=days)


@app.get("/image/{gridfs_id}")
def get_image(gridfs_id: str) -> StreamingResponse:
    oid = _oid(gridfs_id)
    buf = io.BytesIO()
    try:
        images_bucket().download_to_stream(oid, buf)
    except NoFile as exc:
        raise HTTPException(404, f"no image {gridfs_id}") from exc
    buf.seek(0)

    def chunks() -> Iterator[bytes]:
        while data := buf.read(64 * 1024):
            yield data

    return StreamingResponse(
        chunks(), media_type="image/png", headers={"Cache-Control": "public, max-age=31536000"}
    )


__all__ = ["app", "AsyncIterator"]

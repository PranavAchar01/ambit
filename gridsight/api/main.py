"""Phase 5 -- the GridSight inference/agent service."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import segno
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from gridfs.errors import NoFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from gridsight.agent.graph import build_graph, get_app, make_saver, store_pil
from gridsight.analytics import compute_trends
from gridsight.api.relay import relay
from gridsight.config import get_settings
from gridsight.db.mongo import (
    datasets_col,
    ensure_collections,
    finding_recall_index,
    findings_col,
    images_bucket,
    model_router_index,
    models_col,
    search_index_status,
    vector_search,
)
from gridsight.embed import get_embedder
from gridsight.train.store import storage_report
from gridsight.voice import ToolError, execute_tool, mint_client_secret, realtime_model, tool_schemas

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
    router, recall_idx = model_router_index(), finding_recall_index()
    status = search_index_status(router)
    return {
        "status": "ok",
        "vector_index": {"name": router.name, "status": status},
        "recall_index": {"name": recall_idx.name, "status": search_index_status(recall_idx)},
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


@app.get("/registry/layout")
def registry_layout() -> dict[str, Any]:
    """Stable polar coordinates for visualising the registry.

    The UI renders routing radially: distance from the centre is `1 - score`, i.e.
    the *real* measured similarity, and each model's coverage gate is a ring at
    `1 - routing_threshold`. A model's dot falling inside its own ring is exactly
    what "this frame is within the competence I learned" means.

    Only the ANGLE comes from here. It is derived by ranking models around a PCA
    of their centroids, then spacing them evenly so labels never collide -- it
    encodes neighbourhood ordering, not distance, and the UI says so.
    """
    import numpy as np

    docs = list(
        models_col()
        .find(
            {},
            {
                "name": 1,
                "asset_class": 1,
                "embedding": 1,
                "routing_threshold": 1,
                "created_by": 1,
                "reference_image_id": 1,
                "metrics": 1,
            },
        )
        .sort("created_at", 1)
    )
    if not docs:
        return {"models": [], "route_threshold": get_settings().route_threshold}

    vectors = np.array([d["embedding"] for d in docs], dtype=np.float32)
    if len(docs) >= 3:
        centred = vectors - vectors.mean(axis=0, keepdims=True)
        # first two right-singular vectors == the 2-d PCA plane
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        coords = centred @ vt[:2].T
        raw = np.arctan2(coords[:, 1], coords[:, 0])
    else:
        raw = np.linspace(0, 2 * np.pi, len(docs), endpoint=False)

    # even spacing by PCA-angle rank: keeps neighbours adjacent, guarantees no overlap
    order = np.argsort(raw)
    angles = np.empty(len(docs), dtype=np.float64)
    for slot, idx in enumerate(order):
        angles[idx] = 2 * np.pi * slot / len(docs)

    settings = get_settings()
    return {
        "route_threshold": settings.route_threshold,
        "angle_is_illustrative": True,
        "models": [
            {
                "model_id": str(d["_id"]),
                "name": d["name"],
                "asset_class": d["asset_class"],
                "angle": round(float(angles[i]), 5),
                "gate": round(max(settings.route_threshold, float(d.get("routing_threshold") or 0.0)), 4),
                "created_by": d.get("created_by"),
                "image_auroc": (d.get("metrics") or {}).get("image_auroc"),
                "reference_image_url": (
                    f"/image/{d['reference_image_id']}" if d.get("reference_image_id") else None
                ),
            }
            for i, d in enumerate(docs)
        ],
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


@app.post("/recall")
async def recall(
    image: UploadFile = File(...),
    k: int = Form(6),
    verdict: str | None = Form(None),
    asset_class: str | None = Form(None),
) -> JSONResponse:
    """Episodic memory: "have we seen this before?"

    Vector-searches past *findings* rather than models, so the answer comes from
    what the fleet has actually observed, not from what it was trained on.
    """
    frame = await _read_image(image)
    vector = [float(x) for x in get_embedder().embed_one(frame)]

    filters: dict[str, Any] = {}
    if verdict:
        filters["verdict"] = verdict
    if asset_class:
        filters["asset_class"] = asset_class

    idx = finding_recall_index()
    hits = vector_search(vector, k=max(1, min(k, 25)), index=idx, filters=filters or None)

    matches = []
    for h in hits:
        item = _serialize(h)
        item["similarity"] = round(float(h["score"]), 4)
        item["uploaded_image_url"] = (
            f"/image/{item['uploaded_image_id']}" if item.get("uploaded_image_id") else None
        )
        item["heatmap_url"] = f"/image/{item['heatmap_id']}" if item.get("heatmap_id") else None
        matches.append(item)

    defects = [m for m in matches if m.get("verdict") == "defect"]
    return JSONResponse(
        {
            "index": {"name": idx.name, "status": search_index_status(idx)},
            "count": len(matches),
            "recurrence": {
                "similar_defects": len(defects),
                "first_seen": min((m["timestamp"] for m in defects), default=None),
                "last_seen": max((m["timestamp"] for m in defects), default=None),
            },
            "matches": matches,
        }
    )


class VoiceToolCall(BaseModel):
    """One function call emitted by the realtime model, relayed by the browser."""

    name: str = Field(description="Registered voice tool name.")
    arguments: dict[str, Any] = Field(default_factory=dict)


@app.post("/voice/session")
def voice_session() -> JSONResponse:
    """Mint an ephemeral realtime credential for the browser.

    The returned `ek_` token is realtime-scoped -- verified to 401 against
    `chat.completions` and `models.list` -- and the session's model, instructions
    and tool list are bound server-side. `OPENAI_API_KEY` never leaves this process.
    """
    started = time.perf_counter()
    try:
        payload = mint_client_secret()
    except ToolError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a failed mint must not 500 silently
        log.exception("voice session mint failed")
        raise HTTPException(502, f"could not mint a realtime credential: {type(exc).__name__}") from exc

    log.info(
        "voice session minted model=%s tools=%d in %.2fs",
        payload["model"],
        len(payload["tools"]),
        time.perf_counter() - started,
    )
    return JSONResponse(payload)


@app.get("/voice/tools")
def voice_tools() -> dict[str, Any]:
    """The tool schemas, without spending a credential -- useful for inspection."""
    return {"model": realtime_model(), "tools": tool_schemas()}


@app.post("/voice/tool")
def voice_tool(call: VoiceToolCall) -> JSONResponse:
    """Execute one voice tool against MongoDB and hand the result back to the model."""
    started = time.perf_counter()
    try:
        result = execute_tool(call.name, call.arguments)
    except ToolError as exc:
        log.warning("voice tool %s rejected: %s", call.name, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - report the failure to the model, stay up
        log.exception("voice tool %s failed", call.name)
        raise HTTPException(500, f"{call.name} failed: {type(exc).__name__}") from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    log.info("voice tool %s args=%s in %sms", call.name, sorted(call.arguments), elapsed_ms)
    return JSONResponse(
        {"name": call.name, "arguments": call.arguments, "latency_ms": elapsed_ms, "result": result}
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_HTML = Path(__file__).resolve().parent / "capture.html"
FONTS_DIR = REPO_ROOT / "web" / "public" / "fonts"
# Written by scripts/tunnel.sh. Read per request so restarting the tunnel never
# means restarting the API -- quick tunnels get a new hostname every time.
TUNNEL_FILE = REPO_ROOT / "runtime" / "tunnel.json"

if FONTS_DIR.is_dir():
    app.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")


def _public_base() -> str | None:
    try:
        return str(json.loads(TUNNEL_FILE.read_text())["public_url"]).rstrip("/")
    except (OSError, ValueError, KeyError):
        return None


@app.get("/capture")
def capture_page() -> FileResponse:
    """The phone client. Served from the API so one tunnel covers page, socket and upload."""
    return FileResponse(CAPTURE_HTML, media_type="text/html")


@app.get("/live/config")
def live_config(session: str = Query("demo")) -> dict[str, Any]:
    base = _public_base()
    return {
        "session_id": session,
        "public_url": base,
        "capture_url": f"{base}/capture?s={session}" if base else None,
        "sessions": relay.stats(),
    }


@app.get("/live/qr.svg")
def live_qr(session: str = Query("demo")) -> Response:
    """QR for the capture URL -- typing a random trycloudflare hostname on a phone is miserable."""
    base = _public_base()
    if not base:
        raise HTTPException(503, "no public tunnel yet; run scripts/tunnel.sh")
    buf = io.BytesIO()
    segno.make(f"{base}/capture?s={session}", error="m").save(
        buf, kind="svg", scale=6, border=2, dark="#ff2d2d", light="#000000"
    )
    return Response(buf.getvalue(), media_type="image/svg+xml")


@app.websocket("/live/publish/{session_id}")
async def live_publish(ws: WebSocket, session_id: str) -> None:
    """Phone -> server. Binary frames are the viewfinder; text frames are agent events."""
    await ws.accept()
    relay.set_publisher(session_id, True)
    log.info("relay publisher connected session=%s", session_id)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (payload := msg.get("bytes")) is not None:
                relay.publish_frame(session_id, payload)
            elif (text := msg.get("text")) is not None:
                try:
                    relay.publish_message(session_id, json.loads(text))
                except json.JSONDecodeError:
                    log.warning("relay dropped non-JSON text frame session=%s", session_id)
    except WebSocketDisconnect:
        pass
    finally:
        relay.set_publisher(session_id, False)
        log.info("relay publisher gone session=%s", session_id)


@app.websocket("/live/watch/{session_id}")
async def live_watch(ws: WebSocket, session_id: str) -> None:
    """Server -> projected laptop view."""
    await ws.accept()
    viewer = relay.add_viewer(session_id, ws)

    async def drain() -> None:
        while True:
            await ws.receive()

    pump = asyncio.create_task(viewer.pump())
    reader = asyncio.create_task(drain())
    try:
        _, pending = await asyncio.wait({pump, reader}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        pump.cancel()
        reader.cancel()
        relay.remove_viewer(session_id, viewer)


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

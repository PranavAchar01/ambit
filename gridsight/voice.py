"""The voice agent's tool layer -- MongoDB is the answer, the model is only the mouth.

Every tool here runs *server-side*. That is not an implementation detail, it is the
whole point: the browser holds a realtime-scoped ephemeral credential and nothing
else, so an operator's question is answered out of the `findings` and `models`
collections rather than out of the model's priors. When the console shows
`search_findings -> 7 rows`, the audience is watching persistent context being read.

Transport is the GA Realtime API over WebRTC (`client.realtime`, not the legacy
`client.beta.realtime`). The server mints a short-lived `ek_` client secret whose
blast radius was measured, not assumed: it authenticates against
`POST /v1/realtime/calls` and returns 401 against `chat.completions` and
`models.list`. `OPENAI_API_KEY` never leaves this process.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from bson import ObjectId
from bson.errors import InvalidId
from openai import OpenAI
from openai.types.realtime.realtime_function_tool_param import RealtimeFunctionToolParam
from openai.types.realtime.realtime_session_create_request_param import (
    RealtimeSessionCreateRequestParam,
)

from gridsight.analytics import compute_trends
from gridsight.config import get_settings
from gridsight.db.mongo import (
    finding_recall_index,
    findings_col,
    model_router_index,
    models_col,
    search_index_status,
    vector_search,
)

log = logging.getLogger("gridsight.voice")

#: Verified live against this account's model list and a real client-secret mint.
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_VOICE = "cedar"
#: Long enough to survive a slow page load, short enough that a leaked token is worthless.
CLIENT_SECRET_TTL_S = 600

MAX_ROWS = 25

VOICE_INSTRUCTIONS = """You are GridSight's fleet analyst. You speak to inspection operators about a \
fleet of anomaly-detection specialists and the frames they have inspected.

Hard rule: every factual claim you make about the fleet must come from a tool call. You have no prior \
knowledge of this fleet. If a tool returns nothing, say the database has no record of it -- never \
estimate, never fill a gap from memory. If the operator asks something no tool covers, say so.

How the system works, so you can explain it: each registered model carries a 512-dimension CLIP \
centroid of its own training imagery, and a routing_threshold coverage gate set at the 5th percentile \
of its training set's self-similarity. An incoming frame is embedded and vector-searched against those \
centroids. If the best score is below the winner's gate, GridSight records verdict "unroutable" and \
refuses to score rather than applying a specialist trained on different hardware. Refusal is a feature.

Tool routing: fleet-wide rates and what is degrading -> query_trends. Specific past inspections -> \
search_findings. "Have we seen this before?" -> recall_similar, using a finding id from \
search_findings. "What can it inspect?" or index health -> registry_status. "Why did it refuse?" -> \
explain_refusal.

Style: you are being heard, not read. Two or three sentences. Say numbers as an engineer would -- \
"about eighty-two percent", "zero point nine one" -- and never read an ObjectId aloud. Lead with the \
answer, then one line of why."""


def realtime_model() -> str:
    """The realtime model id, overridable without a redeploy."""
    return os.environ.get("OPENAI_REALTIME_MODEL", DEFAULT_REALTIME_MODEL).strip() or (DEFAULT_REALTIME_MODEL)


def realtime_voice() -> str:
    return os.environ.get("OPENAI_REALTIME_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


class ToolError(ValueError):
    """A tool was called with arguments MongoDB cannot answer."""


def _jsonable(value: Any) -> Any:
    """Coerce BSON into something that survives `json.dumps` and a WebRTC data channel."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def _oid(value: str, field: str = "finding_id") -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise ToolError(f"{field} {value!r} is not a valid id") from exc


def _window_start(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=max(1, days))


def _parse_since(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ToolError(f"since={raw!r} is not an ISO date (expected e.g. 2026-08-01)") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _finding_row(doc: dict[str, Any]) -> dict[str, Any]:
    """One finding, trimmed to what is worth speaking aloud."""
    return {
        "finding_id": str(doc["_id"]),
        "timestamp": _jsonable(doc.get("timestamp")),
        "asset_class": doc.get("asset_class"),
        "verdict": doc.get("verdict"),
        "severity": round(float(doc.get("severity") or 0.0), 3),
        "routed_model": doc.get("routed_model_name"),
        "routing_score": round(float(doc.get("routing_score") or 0.0), 4),
        "narrative": (doc.get("agent_narrative") or "")[:400],
    }


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def query_trends(days: int = 30, asset_class: str | None = None) -> dict[str, Any]:
    """Fleet-health aggregation over the findings history."""
    days = max(1, min(int(days), 365))
    trends = compute_trends(findings_col(), models_col(), days=days)

    classes = trends["by_asset_class"]
    fastest = trends["failing_fastest"]
    if asset_class:
        classes = [row for row in classes if row["asset_class"] == asset_class]
        fastest = [row for row in fastest if row["asset_class"] == asset_class]
        if not classes:
            known = sorted({row["asset_class"] for row in trends["by_asset_class"] if row["asset_class"]})
            return {
                "window_days": days,
                "asset_class": asset_class,
                "inspections": 0,
                "note": f"no inspections of {asset_class!r} in the last {days} days",
                "known_asset_classes": known,
            }

    verdicts = {row["verdict"]: row["count"] for row in trends["verdict_counts"]}
    return {
        "window_days": days,
        "asset_class_filter": asset_class,
        "verdict_counts": verdicts,
        "total_inspections": sum(verdicts.values()),
        "by_asset_class": [
            {
                "asset_class": row["asset_class"],
                "inspections": row["inspections"],
                "defects": row["defects"],
                "defect_rate": row["defect_rate"],
                "mean_severity": row["mean_severity"],
            }
            for row in classes[:MAX_ROWS]
        ],
        "failing_fastest": [
            {
                "asset_class": row["asset_class"],
                "defect_rate": row["defect_rate"],
                "recent_defect_rate": row["recent_defect_rate"],
                "acceleration": row.get("acceleration"),
            }
            for row in fastest[:5]
        ],
        "registry_coverage": trends["registry_coverage"],
    }


def search_findings(
    asset_class: str | None = None,
    verdict: str | None = None,
    since: str | None = None,
    days: int | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Episodic memory by metadata: what did the fleet actually inspect?"""
    query: dict[str, Any] = {}
    if asset_class:
        query["asset_class"] = asset_class
    if verdict:
        if verdict not in {"nominal", "defect", "unroutable"}:
            raise ToolError(f"verdict={verdict!r} must be nominal, defect, or unroutable")
        query["verdict"] = verdict

    lower = _parse_since(since) or (_window_start(days) if days else None)
    if lower is not None:
        query["timestamp"] = {"$gte": lower}

    limit = max(1, min(int(limit), MAX_ROWS))
    total = findings_col().count_documents(query)
    cursor = findings_col().find(query, {"embedding": 0, "candidates": 0}).sort("timestamp", -1).limit(limit)
    rows = [_finding_row(doc) for doc in cursor]
    return {
        "filters": {
            "asset_class": asset_class,
            "verdict": verdict,
            "since": _jsonable(lower),
        },
        "matched": total,
        "returned": len(rows),
        "findings": rows,
    }


def recall_similar(finding_id: str, k: int = 5) -> dict[str, Any]:
    """Episodic memory by appearance: "have we seen this before?"

    Re-uses the frame embedding already stored on the finding and runs it back
    through `finding_recall_idx`, so recall costs one vector search and no CLIP
    forward pass. The source finding is excluded -- it would always score 1.0.
    """
    oid = _oid(finding_id)
    source = findings_col().find_one({"_id": oid})
    if source is None:
        raise ToolError(f"no finding {finding_id}")
    vector = source.get("embedding")
    if not vector:
        raise ToolError(f"finding {finding_id} predates episodic recall and has no stored embedding")

    idx = finding_recall_index()
    k = max(1, min(int(k), 15))
    hits = vector_search([float(x) for x in vector], k=k + 1, index=idx, exclude_ids=[oid])

    matches = []
    for hit in hits[:k]:
        row = _finding_row(hit)
        row["similarity"] = round(float(hit["score"]), 4)
        matches.append(row)

    defects = [m for m in matches if m["verdict"] == "defect"]
    return {
        "index": {"name": idx.name, "status": search_index_status(idx)},
        "source": _finding_row(source),
        "count": len(matches),
        "recurrence": {
            "similar_defects": len(defects),
            "first_seen": min((m["timestamp"] for m in defects), default=None),
            "last_seen": max((m["timestamp"] for m in defects), default=None),
        },
        "matches": matches,
    }


def registry_status() -> dict[str, Any]:
    """Semantic memory: which specialists exist, and is the routing index healthy?"""
    router, recall = model_router_index(), finding_recall_index()
    docs = list(models_col().find({}, {"embedding": 0}).sort("created_at", 1))
    return {
        "models": len(docs),
        "asset_classes": sorted({d["asset_class"] for d in docs}),
        "total_findings": findings_col().count_documents({}),
        "indexes": {
            router.name: search_index_status(router),
            recall.name: search_index_status(recall),
        },
        "route_threshold": get_settings().route_threshold,
        "registry": [
            {
                "name": d["name"],
                "asset_class": d["asset_class"],
                "backbone": d.get("backbone"),
                "training_samples": d.get("training_samples"),
                "routing_threshold": _jsonable(d.get("routing_threshold")),
                "image_auroc": _jsonable((d.get("metrics") or {}).get("image_auroc")),
                "pixel_auroc": _jsonable((d.get("metrics") or {}).get("pixel_auroc")),
                "created_by": d.get("created_by"),
                "created_at": _jsonable(d.get("created_at")),
            }
            for d in docs
        ],
    }


def explain_refusal(finding_id: str | None = None) -> dict[str, Any]:
    """Metacognitive memory: why did GridSight decline to score a frame?

    Reads back the candidate list and per-model coverage gate that were persisted
    at decision time, so the explanation is the recorded decision rather than a
    reconstruction. With no id, explains the most recent refusal.
    """
    if finding_id:
        doc = findings_col().find_one({"_id": _oid(finding_id)})
        if doc is None:
            raise ToolError(f"no finding {finding_id}")
    else:
        doc = findings_col().find_one({"verdict": "unroutable"}, sort=[("timestamp", -1)])
        if doc is None:
            return {"refused": False, "note": "no refusal has ever been recorded"}

    candidates = [
        {
            "name": c.get("name"),
            "asset_class": c.get("asset_class"),
            "score": _jsonable(c.get("score")),
            "gate": _jsonable(c.get("gate")),
            "shortfall": (
                round(float(c["gate"]) - float(c["score"]), 4)
                if c.get("gate") is not None and c.get("score") is not None
                else None
            ),
            "cleared_gate": (
                bool(float(c["score"]) >= float(c["gate"]))
                if c.get("gate") is not None and c.get("score") is not None
                else None
            ),
        }
        for c in (doc.get("candidates") or [])
    ]
    cold_start = doc.get("cold_start_info") or None
    return {
        "finding": _finding_row(doc),
        "refused": doc.get("verdict") == "unroutable",
        "decision_source": doc.get("decision_source"),
        "decision_reason": doc.get("decision_reason"),
        "global_route_threshold": get_settings().route_threshold,
        "candidates": candidates,
        "cold_start": (
            {
                "name": cold_start.get("name"),
                "reference_images": cold_start.get("reference_images"),
                "seconds": _jsonable(cold_start.get("seconds")),
                "weights_mb": _jsonable(cold_start.get("weights_mb")),
            }
            if cold_start
            else None
        ),
    }


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceTool:
    """One server-side tool, plus the JSON schema the realtime model picks from."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: tuple[VoiceTool, ...] = (
    VoiceTool(
        name="query_trends",
        description=(
            "Aggregate fleet health over the findings history: inspection and defect counts per asset "
            "class, mean severity, verdict breakdown, and which asset class is degrading fastest "
            "(defect rate in the recent half of the window minus the rate across the whole window). "
            "Use for 'what is failing fastest', 'how is the fleet doing', 'defect rate this month'."
        ),
        parameters=_obj(
            {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Look-back window in days. 30 unless the operator implies otherwise.",
                },
                "asset_class": {
                    "type": "string",
                    "description": (
                        "Optional snake_case class to narrow to, e.g. insulator, corrosion, "
                        "vegetation, transmission_tower, conductor. Omit for the whole fleet."
                    ),
                },
            },
            required=["days"],
        ),
        handler=query_trends,
    ),
    VoiceTool(
        name="search_findings",
        description=(
            "List individual past inspections newest-first, with their verdict, severity, routed model "
            "and narrative. Use for 'show me recent insulator defects', 'what did we refuse last week', "
            "or to obtain a finding_id to pass to recall_similar or explain_refusal."
        ),
        parameters=_obj(
            {
                "asset_class": {"type": "string", "description": "Optional snake_case asset class."},
                "verdict": {
                    "type": "string",
                    "enum": ["nominal", "defect", "unroutable"],
                    "description": "Optional verdict filter. 'unroutable' means GridSight refused.",
                },
                "since": {
                    "type": "string",
                    "description": "Optional ISO lower bound, e.g. 2026-08-01. Prefer `days` if relative.",
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Optional relative look-back in days. Ignored when `since` is given.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 25,
                    "description": "How many findings to return. Keep it small for speech.",
                },
            }
        ),
        handler=search_findings,
    ),
    VoiceTool(
        name="recall_similar",
        description=(
            "Answer 'have we seen this before?'. Vector-searches the finding_recall_idx index over past "
            "findings using the stored 512-d CLIP embedding of the given finding's frame, and reports "
            "how many visually similar past inspections were defects and when they were first and last "
            "seen. Requires a finding_id, which you get from search_findings."
        ),
        parameters=_obj(
            {
                "finding_id": {
                    "type": "string",
                    "description": "The 24-character finding id returned by search_findings.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 15,
                    "description": "How many similar past findings to retrieve.",
                },
            },
            required=["finding_id"],
        ),
        handler=recall_similar,
    ),
    VoiceTool(
        name="registry_status",
        description=(
            "Describe the model registry itself: how many specialists are registered, which asset "
            "classes they cover, each one's training-set size, coverage gate (routing_threshold) and "
            "measured image/pixel AUROC, whether it was seeded or cold-started by the agent, and the "
            "READY status of both Atlas vector indexes. Use for 'what can GridSight inspect', 'which "
            "models did we cold-start', 'is the index healthy'."
        ),
        parameters=_obj({}),
        handler=registry_status,
    ),
    VoiceTool(
        name="explain_refusal",
        description=(
            "Explain why GridSight refused to score a frame, from the candidate list and per-model "
            "coverage gates recorded at decision time: every candidate's routing score, its gate, and "
            "how far short it fell. Omit finding_id to explain the most recent refusal."
        ),
        parameters=_obj(
            {
                "finding_id": {
                    "type": "string",
                    "description": "Optional finding id. Omit for the latest unroutable finding.",
                }
            }
        ),
        handler=explain_refusal,
    ),
)

TOOLS_BY_NAME: dict[str, VoiceTool] = {t.name: t for t in TOOLS}


def tool_schemas() -> list[dict[str, Any]]:
    """The tool list registered on the realtime session."""
    return [t.schema() for t in TOOLS]


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one named tool against MongoDB and return a JSON-safe result."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(sorted(TOOLS_BY_NAME))}")
    allowed = set(tool.parameters.get("properties", {}))
    kwargs = {k: v for k, v in (arguments or {}).items() if k in allowed and v is not None}
    return _jsonable(tool.handler(**kwargs))


# ---------------------------------------------------------------------------
# ephemeral credentials
# ---------------------------------------------------------------------------


def mint_client_secret(ttl_seconds: int = CLIENT_SECRET_TTL_S) -> dict[str, Any]:
    """Mint a realtime-scoped `ek_` token for the browser.

    The session config -- model, voice, instructions, tools, turn detection -- is
    bound to the token here rather than sent from the client, so a stolen token
    cannot be repointed at a different model or stripped of its instructions.
    """
    settings = get_settings()
    if not settings.has_openai:
        raise ToolError("OPENAI_API_KEY is not configured; the voice agent cannot mint a credential")

    client = OpenAI(api_key=settings.openai_api_key)
    model = realtime_model()
    session: RealtimeSessionCreateRequestParam = {
        "type": "realtime",
        "model": model,
        "instructions": VOICE_INSTRUCTIONS,
        "audio": {
            "input": {
                "transcription": {"model": "whisper-1"},
                # Server-side semantic VAD gives natural turn-taking and barge-in for free.
                "turn_detection": {"type": "semantic_vad"},
            },
            "output": {"voice": realtime_voice()},
        },
        "tools": [cast(RealtimeFunctionToolParam, s) for s in tool_schemas()],
        "tool_choice": "auto",
    }
    secret = client.realtime.client_secrets.create(
        expires_after={"anchor": "created_at", "seconds": max(60, min(ttl_seconds, 7200))},
        session=session,
    )
    return {
        "client_secret": secret.value,
        "expires_at": secret.expires_at,
        "model": model,
        "voice": realtime_voice(),
        "tools": tool_schemas(),
    }


__all__ = [
    "TOOLS",
    "TOOLS_BY_NAME",
    "VOICE_INSTRUCTIONS",
    "ToolError",
    "VoiceTool",
    "execute_tool",
    "explain_refusal",
    "mint_client_secret",
    "query_trends",
    "realtime_model",
    "recall_similar",
    "registry_status",
    "search_findings",
    "tool_schemas",
]

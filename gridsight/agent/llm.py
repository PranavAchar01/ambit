"""OpenAI reasoning for the agent: adjudicate ambiguous routes, name new models,
and write the human-readable finding.

Every helper degrades to a deterministic fallback if the API is unreachable, and
labels which path produced the text so a fallback is never mistaken for model
output.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from gridsight.config import get_settings

log = logging.getLogger("gridsight.llm")

_client: OpenAI | None = None


def get_client() -> OpenAI | None:
    global _client
    settings = get_settings()
    if not settings.has_openai:
        return None
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


class RouteAdjudication(BaseModel):
    """Structured verdict for a routing score inside the ambiguous band."""

    chosen_model_name: str | None = Field(
        description="Registry model name to route to, or null to refuse and cold-start."
    )
    should_cold_start: bool = Field(description="True when no candidate genuinely covers this asset class.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One or two sentences justifying the choice.")


class ModelName(BaseModel):
    """Name for a freshly minted cold-start specialist."""

    name: str = Field(description="Lowercase kebab-case registry name, e.g. 'rail-surface-patchcore-v1'.")
    asset_class: str = Field(description="Lowercase snake_case asset class, e.g. 'rail_surface'.")


ADJUDICATE_SYSTEM = """You route drone inspection frames to specialist anomaly-detection models.
You are given the top candidates from a vector search over a model registry; each candidate was
trained on imagery of one asset class, and the score is the cosine similarity between the incoming
frame and that model's training-set centroid.

The top score fell into an ambiguous band: high enough that the frame might be covered, low enough
that it might be an asset class nobody has trained a model for yet.

Routing a frame to a model trained on a different asset class produces a confident, wrong answer,
which is worse than refusing. Prefer cold-starting a new model unless a candidate plausibly covers
the same physical component. Set should_cold_start=true when you refuse."""

NARRATE_SYSTEM = """You write terse inspection findings for utility and rail asset managers.
Given the routed model, the anomaly score relative to its decision threshold, and the located
regions, write 2-4 sentences: what asset this is, whether a defect is present and where, how severe,
and the recommended action. Be concrete and factual. Do not invent details that are not in the data.
If the verdict is nominal, say so plainly and recommend no action."""


def adjudicate_route(
    candidates: list[dict[str, Any]], top_score: float, threshold: float
) -> tuple[RouteAdjudication, str]:
    """Let the LLM break a tie in the ambiguous band. Returns (verdict, source)."""
    client = get_client()
    if client is None:
        return (
            RouteAdjudication(
                chosen_model_name=None,
                should_cold_start=True,
                confidence=0.0,
                reasoning="No OpenAI credentials available; refusing rather than guessing.",
            ),
            "fallback",
        )

    listing = "\n".join(
        f"- {c['name']} | asset_class={c['asset_class']} | score={c['score']:.4f} "
        f"| trained_on={c.get('training_samples', '?')} images"
        for c in candidates
    )
    prompt = (
        f"Routing threshold is {threshold:.2f}; the best candidate scored {top_score:.4f}.\n\n"
        f"Candidates:\n{listing}\n\nDecide whether to route or refuse."
    )
    try:
        completion = client.beta.chat.completions.parse(
            model=get_settings().openai_model,
            messages=[
                {"role": "system", "content": ADJUDICATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format=RouteAdjudication,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("structured parse returned no content")
        return parsed, "openai"
    except Exception as exc:  # noqa: BLE001 - the agent must stay up if OpenAI is down
        log.warning("route adjudication failed (%s); refusing rather than guessing", exc)
        return (
            RouteAdjudication(
                chosen_model_name=None,
                should_cold_start=True,
                confidence=0.0,
                reasoning=f"LLM adjudication unavailable ({type(exc).__name__}); refusing to guess.",
            ),
            "fallback",
        )


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def name_new_model(asset_class_hint: str | None, n_references: int, backbone: str) -> tuple[ModelName, str]:
    """Ask the LLM to name a newly minted specialist."""
    fallback_class = (asset_class_hint or "unknown-asset").lower().replace(" ", "_").replace("-", "_")
    fallback = ModelName(
        name=f"{slugify(fallback_class)}-{backbone}-coldstart-v1", asset_class=fallback_class
    )

    client = get_client()
    if client is None:
        return fallback, "fallback"

    try:
        completion = client.beta.chat.completions.parse(
            model=get_settings().openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You name entries in a machine-vision model registry for "
                    "infrastructure inspection. Names are lowercase kebab-case and end in a version.",
                },
                {
                    "role": "user",
                    "content": (
                        f"A new anomaly-detection specialist was just cold-started from "
                        f"{n_references} reference images using a {backbone} backbone. The operator "
                        f"described the asset as: {asset_class_hint or 'unspecified'}. "
                        "Name it and give its asset class."
                    ),
                },
            ],
            response_format=ModelName,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("structured parse returned no content")
        return parsed, "openai"
    except Exception as exc:  # noqa: BLE001
        log.warning("model naming failed (%s); using deterministic name", exc)
        return fallback, "fallback"


def _fallback_narrative(context: dict[str, Any]) -> str:
    if context["verdict"] == "unroutable":
        return (
            f"No model in the registry covers this asset class. The closest match was "
            f"{context.get('top_candidate') or 'none'} at a routing score of "
            f"{context.get('routing_score', 0.0):.3f}, below the {context.get('threshold', 0.0):.2f} "
            "floor. GridSight is refusing to score this frame rather than applying a specialist "
            "trained on different hardware. Supply a handful of known-good reference images to "
            "cold-start a specialist for it."
        )
    n = len(context.get("regions", []))
    if context["verdict"] == "defect":
        return (
            f"{context['asset_class']}: anomaly score {context['raw_score']:.2f} exceeds the "
            f"{context['threshold']:.2f} decision threshold learned by {context['model_name']}. "
            f"{n} region(s) flagged. Recommend a close-up re-inspection of the flagged area."
        )
    return (
        f"{context['asset_class']}: nominal. Anomaly score {context['raw_score']:.2f} sits below "
        f"{context['model_name']}'s {context['threshold']:.2f} threshold. No action required."
    )


def narrate_finding(context: dict[str, Any]) -> tuple[str, str]:
    """Write the operator-facing finding. Returns (text, source)."""
    client = get_client()
    if client is None:
        return _fallback_narrative(context), "fallback"

    regions = context.get("regions", [])
    region_text = (
        "; ".join(
            f"region at x={r['x']},y={r['y']} size {r['w']}x{r['h']} peak={r['score']:.2f}"
            for r in regions[:5]
        )
        or "no localised regions above the pixel threshold"
    )
    if context["verdict"] == "unroutable":
        prompt = (
            "The vector router found no registry model covering this frame's asset class. "
            f"Best candidate: {context.get('top_candidate') or 'none'} at routing score "
            f"{context.get('routing_score', 0.0):.3f}, against a floor of "
            f"{context.get('threshold', 0.0):.2f}. Explain to the operator that GridSight is "
            "refusing to score rather than guessing, and what they should do next."
        )
    else:
        prompt = (
            f"Asset class: {context['asset_class']}\n"
            f"Routed model: {context['model_name']} (routing score {context['routing_score']:.3f})\n"
            f"Anomaly score: {context['raw_score']:.3f} against threshold {context['threshold']:.3f}\n"
            f"Normalised severity: {context['severity']:.2f}\n"
            f"Verdict: {context['verdict']}\n"
            f"Located regions: {region_text}\n"
            f"Frame size: {context.get('width')}x{context.get('height')} px"
        )

    try:
        completion = client.chat.completions.create(
            model=get_settings().openai_model,
            messages=[
                {"role": "system", "content": NARRATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=260,
            temperature=0.2,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty narrative")
        return text, "openai"
    except Exception as exc:  # noqa: BLE001
        log.warning("narration failed (%s); using deterministic summary", exc)
        return _fallback_narrative(context), "fallback"

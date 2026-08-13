"""What is actually answering: the resolved VLM and text-LLM providers.

Reported at `GET /health` and logged once at startup, for the reason the
brute-force vector-search fallback is logged at ERROR on every call: a silently
degraded provider makes a broken deployment look healthy. Knowing that
narration fell back to the numbers, or that adjudication is running on the
deterministic refuse rule, is the difference between a demo you can trust and
one that is quietly not doing what the slide says.

Configuration only. Nothing here probes a provider or makes a network call --
`/health` is polled (`scripts/tunnel.sh` gates the tunnel on it), and a liveness
check that costs an inference request is a liveness check nobody can afford to
run often.
"""

from __future__ import annotations

from typing import Any

from gridsight.agent.llm import resolve_llm
from gridsight.audio.tts import tts_report
from gridsight.vision.vlm import get_provider


def provider_report() -> dict[str, Any]:
    """`{"vlm": {...}, "llm": {...}}` -- name, why it was chosen, and whether it can answer."""
    vlm, vlm_reason = get_provider()
    llm = resolve_llm()
    return {
        "vlm": {
            "name": vlm.name,
            "reason": vlm_reason,
            # False means narration ships the structured sentence written from
            # the numbers. That is a real narrative, not a stub -- but it did
            # not look at the pixels, and the finding records which.
            "describes_pixels": vlm.name != "none",
        },
        "llm": {
            "name": llm.name,
            "reason": llm.reason,
            "model": llm.model or None,
            # False means the ambiguous band resolves by the deterministic rule:
            # refuse. The system stays up and stays honest on zero credentials.
            "available": llm.available,
        },
        "tts": tts_report(),
    }

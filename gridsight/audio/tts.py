"""Speak a finding aloud, server-side.

Additive to the existing OpenAI Realtime voice agent, which is untouched: that
one is a conversation, this one is an announcement. When a frame resolves to
`defect` the operator should hear what is wrong without asking, because the
moment a defect is found is exactly the moment nobody is looking at the screen.

Two properties matter more than the audio:

**The key never reaches the browser.** Synthesis happens here and the bytes are
streamed to the client, the same reason the Realtime path mints an ephemeral
credential instead of shipping the account key.

**Failure is silent.** A synthesis error must never block or delay a verdict --
the verdict is the product, the audio is a courtesy. Every failure path returns
None and logs; nothing raises into a request handler.

Lives at `gridsight/audio/` rather than `gridsight/voice/` because
`gridsight/voice.py` is an existing 600-line module exporting fourteen names;
turning it into a package to make room for a sibling would be a refactor of the
working Realtime path for no benefit.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger("gridsight.tts")

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

#: Flash v2.5 is the low-latency model -- this is an interjection over a live
#: demo, not an audiobook, and a two-second wait to be told the board is bad is
#: worse than not being told.
ELEVENLABS_MODEL = "eleven_flash_v2_5"

#: A default public voice, overridable. Not a claim about which voice is best;
#: it is the one that exists without an account-specific id.
ELEVENLABS_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"

#: Findings are two or three sentences. A cap stops a pathological narrative
#: from turning into a minute of audio and a large bill.
MAX_CHARS = 600

TTS_TIMEOUT_S = 12.0


@dataclass(frozen=True)
class TTSChoice:
    name: str
    reason: str
    voice_id: str
    model: str

    @property
    def available(self) -> bool:
        return self.name != "none"


def resolve_tts() -> TTSChoice:
    """Which synthesiser, and why. Absent a key this is `none`, loudly."""
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or ELEVENLABS_DEFAULT_VOICE
    model = os.environ.get("ELEVENLABS_MODEL", "").strip() or ELEVENLABS_MODEL
    if not key:
        return TTSChoice("none", "ELEVENLABS_API_KEY is not set", voice, model)
    return TTSChoice("elevenlabs", "ELEVENLABS_API_KEY is present", voice, model)


def synthesize(text: str) -> bytes | None:
    """MP3 bytes for `text`, or None if speech is unavailable or failed.

    Returns None rather than raising, and never returns silence dressed as
    success: a caller that gets None renders text and disables the control,
    which is honest. Stub audio would not be.
    """
    choice = resolve_tts()
    if not choice.available:
        return None

    clean = " ".join((text or "").split())[:MAX_CHARS]
    if not clean:
        return None

    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    try:
        import httpx

        response = httpx.post(
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{choice.voice_id}",
            headers={"xi-api-key": key, "accept": "audio/mpeg"},
            json={
                "text": clean,
                "model_id": choice.model,
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.7},
            },
            timeout=TTS_TIMEOUT_S,
        )
        response.raise_for_status()
        audio = response.content
    except Exception as exc:  # noqa: BLE001 -- audio must never surface as an error
        log.warning("speech synthesis failed (%s: %s); the finding still renders", type(exc).__name__, exc)
        return None

    if not audio:
        log.warning("speech synthesis returned no audio; the finding still renders")
        return None
    return audio


def tts_report() -> dict[str, object]:
    choice = resolve_tts()
    return {
        "name": choice.name,
        "reason": choice.reason,
        # False means the UI renders the narrative as text and disables the
        # speaker control with a tooltip. No stub audio, no silent failure.
        "available": choice.available,
        "model": choice.model if choice.available else None,
    }

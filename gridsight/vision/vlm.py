"""Vision-language providers, selected by what is actually available.

Resolution order, and the reasoning for each step:

1. `VLM_PROVIDER` set explicitly -- an operator overriding the default is always
   obeyed, including `none`.
2. `FIREWORKS_API_KEY` present -- the primary. A hosted 32B vision model
   describes a bent header pin far better than anything that fits on a laptop.
3. A local model that loads -- the fallback. No network, no credentials.
4. Null -- narration falls back to the structured sentence, which is a real,
   honest description of the numbers and not a stub.

The selection is **logged at INFO on startup and reported at `GET /health`**,
for the same reason the brute-force vector-search fallback is: a silently
degraded provider makes a broken deployment look healthy.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from PIL import Image

from gridsight.vision.crops import crop_pair, hottest_region
from gridsight.vision.vlm_base import (
    NO_VISIBLE_DIFFERENCE,
    VLM_SYSTEM_PROMPT,
    VLMProvider,
    VLMResult,
    user_prompt,
)

log = logging.getLogger("gridsight.vlm")

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

#: Chosen by probing this account, not by reading a model card.
#:
#: `qwen3-vl-32b-instruct` and `qwen3-vl-8b-instruct` -- the intended primary and
#: fallback -- both return 404 NOT_FOUND here: neither is deployed on this
#: account, which serves 24 models and no Qwen-VL at all. So every plausibly
#: multimodal id was sent a 64x64 red PNG and asked to name the colour under the
#: same json_schema this module uses:
#:
#:   qwen3p8-max, qwen3p8-2p4t-a95b, glm-5p2, deepseek-v4-pro,
#:   nemotron-3-ultra-nvfp4, gpt-oss-120b   no image support
#:   qwen3p7-plus, kimi-k3, minimax-m3, inkling, muse-glimmer-30b   image accepted
#:
#: The survivors were then run on a real PCB crop pair, which separated them
#: much more sharply than the toy image did:
#:
#:   minimax-m3        2.08 s, clean JSON, a real physical description of the
#:                     component that moved                          <- primary
#:   inkling           2.05 s, empty content
#:   muse-glimmer-30b  3.97 s, empty content on the real task
#:   kimi-k3           8.12 s, leaks its reasoning trace into `content`
#:                     and blows the latency budget                  <- fallback
#:   qwen3p7-plus      same reasoning leak
#:
#: kimi-k3 is the fallback despite exceeding FIREWORKS_TIMEOUT_S: it is the only
#: other model that returns substance, `_parse` degrades its prose rather than
#: dropping it, and a fallback that times out lands on the structured narrative,
#: which is the correct end state anyway. `VLM_MODEL` overrides the primary.
FIREWORKS_PRIMARY = "accounts/fireworks/models/minimax-m3"
FIREWORKS_FALLBACK = "accounts/fireworks/models/kimi-k3"

#: A verdict must not wait indefinitely on a description; past this the structured
#: narrative ships.
#:
#: The spec called for 4 s. That number was written before the provider was
#: known, and it does not survive contact with this one: measured over four real
#: PCB crop pairs the primary took 2.07 / 2.66 / 3.48 / 4.96 s (mean 3.29), so a
#: 4 s ceiling discards roughly half of all successful descriptions -- and it
#: did exactly that on the first live inspection, which fell back to the numbers
#: with both models timing out.
#:
#: 8 s is the measured maximum plus headroom, and matches the local cap so there
#: is one number to reason about. `VLM_TIMEOUT_S` overrides it. The verdict
#: itself is decided in `infer`, upstream of narration, so this budget delays
#: how fast the *description* appears, never what the system decided.
FIREWORKS_TIMEOUT_S = float(os.environ.get("VLM_TIMEOUT_S", "8").strip() or 8)

#: The local model gets longer because it is not competing with a network round
#: trip, but it is still a hard cap: a laptop VLM that has started swapping will
#: not finish, and the operator is waiting.
LOCAL_TIMEOUT_S = 8.0

#: Tried in order; the first that loads wins. Same family as the hosted primary
#: first, so prompt behaviour transfers; smallest and weakest last.
LOCAL_CANDIDATES: tuple[str, ...] = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "HuggingFaceTB/SmolVLM-Instruct",
    "vikhyatk/moondream2",
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "component": {"type": ["string", "null"]},
        "difference_visible": {"type": "boolean"},
    },
    "required": ["description", "component", "difference_visible"],
    "additionalProperties": False,
}


def _data_uri(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"


def _parse(payload: str) -> tuple[str, str | None, bool]:
    """Read the structured answer, tolerating a model that wrapped it in prose."""
    text = payload.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    if not text:
        # An empty completion is a failed call wearing a success's clothes. Two
        # of the probed models return one on a hard crop; letting it through
        # would persist a finding whose description is the empty string.
        raise ValueError("provider returned an empty completion")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # A model that ignored the schema but obeyed the sentinel is still
        # answering the question; treat its prose as the description.
        visible = NO_VISIBLE_DIFFERENCE not in payload
        return payload.strip(), None, visible
    description = str(data.get("description") or "").strip()
    component = data.get("component")
    visible = bool(data.get("difference_visible"))
    if NO_VISIBLE_DIFFERENCE in description:
        visible = False
    if not description:
        raise ValueError("provider returned an empty description")
    return description, (str(component) if component else None), visible


class NullVLM:
    """Describes nothing, on purpose. Narration falls back to the numbers."""

    name = "none"

    def describe_difference(
        self, defect_crop: bytes, reference_crop: bytes, model_name: str, anomaly_score: float
    ) -> VLMResult:
        raise NotImplementedError("NullVLM never describes; callers must use the structured narrative")


class FireworksVLM:
    """Qwen3-VL on Fireworks, over their OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self._api_key = api_key
        self._model = model or os.environ.get("VLM_MODEL", "").strip() or FIREWORKS_PRIMARY
        self._fallback = FIREWORKS_FALLBACK
        self._client: Any = None
        self.name = f"vlm:{self._model.rsplit('/', 1)[-1]}"

    def _get_client(self) -> Any:
        # Constructed on first use, never at import and never in the API's
        # startup hook, which is synchronous and blocks the event loop.
        if self._client is None:
            from openai import OpenAI

            # max_retries=0 so the timeout is the real budget. The SDK retries
            # twice by default, which silently turned a 4 s deadline into a
            # measured 19 s wait across two models -- a verdict must not sit
            # behind a description that long.
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=FIREWORKS_BASE_URL,
                timeout=FIREWORKS_TIMEOUT_S,
                max_retries=0,
            )
        return self._client

    def _call(self, model: str, defect: bytes, reference: bytes, name: str, score: float) -> str:
        completion = self._get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt(name, score)},
                        {"type": "image_url", "image_url": {"url": _data_uri(reference)}},
                        {"type": "image_url", "image_url": {"url": _data_uri(defect)}},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "difference", "schema": RESPONSE_SCHEMA},
            },
            max_tokens=220,
            temperature=0.1,
        )
        return completion.choices[0].message.content or ""

    def describe_difference(
        self, defect_crop: bytes, reference_crop: bytes, model_name: str, anomaly_score: float
    ) -> VLMResult:
        started = time.time()
        try:
            payload = self._call(self._model, defect_crop, reference_crop, model_name, anomaly_score)
            used = self._model
        except Exception as exc:  # noqa: BLE001 -- any provider failure falls to the smaller model
            log.warning("%s failed (%s); retrying on %s", self._model, exc, self._fallback)
            payload = self._call(self._fallback, defect_crop, reference_crop, model_name, anomaly_score)
            used = self._fallback
        description, component, visible = _parse(payload)
        return VLMResult(
            description=description,
            component=component,
            difference_visible=visible,
            source=f"vlm:{used.rsplit('/', 1)[-1]}",
            latency_ms=int((time.time() - started) * 1000),
        )


class LocalVLM:
    """A vision model running on this machine. No network, no credentials."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        self.name = f"vlm:local-{repo_id.rsplit('/', 1)[-1].lower()}"
        self._model: Any = None
        self._processor: Any = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-local")

    def _load(self) -> tuple[Any, Any]:
        # Lazy and cached in-process: nothing about narration may slow API
        # startup, and the weights must not be re-read per inspection.
        if self._model is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            log.info("loading local VLM %s (first use)", self.repo_id)
            self._processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.repo_id, trust_remote_code=True, dtype="auto"
            )
            self._model.eval()
        return self._model, self._processor

    def _generate(self, defect: bytes, reference: bytes, model_name: str, score: float) -> str:
        import io as _io

        from PIL import Image as _Image

        model, processor = self._load()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": user_prompt(model_name, score)},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        images = [
            _Image.open(_io.BytesIO(reference)).convert("RGB"),
            _Image.open(_io.BytesIO(defect)).convert("RGB"),
        ]
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
        # Chat templates echo the prompt; keep only what came after it.
        return decoded.split(user_prompt(model_name, score))[-1].strip()

    def describe_difference(
        self, defect_crop: bytes, reference_crop: bytes, model_name: str, anomaly_score: float
    ) -> VLMResult:
        started = time.time()
        future = self._pool.submit(
            self._generate, defect_crop, reference_crop, model_name, anomaly_score
        )
        try:
            payload = future.result(timeout=LOCAL_TIMEOUT_S)
        except FutureTimeout as exc:
            # The worker is abandoned rather than killed -- Python cannot
            # interrupt a running generate -- but the caller is released on time
            # and the stale result is discarded when it eventually lands.
            future.cancel()
            raise TimeoutError(f"local VLM exceeded {LOCAL_TIMEOUT_S}s") from exc
        description, component, visible = _parse(payload)
        return VLMResult(
            description=description,
            component=component,
            difference_visible=visible,
            source=self.name,
            latency_ms=int((time.time() - started) * 1000),
        )


def _local_provider() -> tuple[LocalVLM | None, str]:
    """First local candidate whose weights are already on this machine.

    Deliberately does **not** download: a first inspection that silently pulls
    several gigabytes is not a fallback, it is an outage. `huggingface-cli
    download <repo>` opts in.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None, "huggingface_hub is not installed"

    try:
        import transformers  # noqa: F401
    except ImportError:
        return None, "transformers is not installed"

    missing: list[str] = []
    for repo in LOCAL_CANDIDATES:
        cached = try_to_load_from_cache(repo, "config.json")
        if isinstance(cached, str):
            return LocalVLM(repo), f"{repo} found in the local Hugging Face cache"
        missing.append(repo.rsplit("/", 1)[-1])
    return None, f"none of {', '.join(missing)} are downloaded (no weights in the local cache)"


def resolve_provider() -> tuple[VLMProvider, str]:
    """(provider, why). Never raises -- the worst case is Null, which is honest."""
    override = os.environ.get("VLM_PROVIDER", "").strip().lower()
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()

    if override == "none":
        return NullVLM(), "VLM_PROVIDER=none"
    if override == "fireworks":
        if not key:
            return NullVLM(), "VLM_PROVIDER=fireworks but FIREWORKS_API_KEY is not set"
        return FireworksVLM(key), "VLM_PROVIDER=fireworks"
    if override == "local":
        provider, why = _local_provider()
        if provider is None:
            return NullVLM(), f"VLM_PROVIDER=local but {why}"
        return provider, f"VLM_PROVIDER=local, {why}"

    if key:
        return FireworksVLM(key), "FIREWORKS_API_KEY is present"
    provider, why = _local_provider()
    if provider:
        return provider, f"no Fireworks key; {why}"
    return NullVLM(), f"no Fireworks key and no local model ({why})"


_RESOLVED: tuple[VLMProvider, str] | None = None


def get_provider() -> tuple[VLMProvider, str]:
    """Resolve once per process, and say so out loud the first time."""
    global _RESOLVED
    if _RESOLVED is None:
        _RESOLVED = resolve_provider()
        log.info("VLM provider: %s (%s)", _RESOLVED[0].name, _RESOLVED[1])
    return _RESOLVED


def reset_provider() -> None:
    """Drop the cached resolution. For tests and for `scripts/verify_vlm.py`."""
    global _RESOLVED
    _RESOLVED = None


def describe_defect(
    frame: Image.Image,
    reference: Image.Image | None,
    regions: list[dict[str, Any]],
    model_name: str,
    anomaly_score: float,
) -> VLMResult | None:
    """Describe what physically differs, or return None and let the numbers speak.

    None is returned -- never an invented sentence -- whenever the comparison
    cannot honestly be made:

    * the provider is Null, or fails, or exceeds its timeout;
    * no region was localised, so there is nothing to crop to;
    * there is no golden reference, so there is nothing to compare against.

    A returned result with `difference_visible=False` is a *successful* call:
    the model looked and saw nothing. That is a finding, and the caller states
    it plainly rather than substituting a guess.
    """
    provider, _ = get_provider()
    if isinstance(provider, NullVLM):
        return None

    region = hottest_region(regions)
    if region is None:
        log.info("no localised region -- skipping the VLM, nothing to crop to")
        return None
    if reference is None:
        log.info("no golden reference -- skipping the VLM, nothing to compare against")
        return None

    defect_crop, reference_crop = crop_pair(frame, reference, region)
    if reference_crop is None:
        return None

    try:
        return provider.describe_difference(defect_crop, reference_crop, model_name, anomaly_score)
    except Exception as exc:  # noqa: BLE001 -- every failure degrades to the structured narrative
        log.warning("VLM description failed (%s: %s); falling back to the structured narrative",
                    type(exc).__name__, exc)
        return None

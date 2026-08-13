"""The contract every vision-language provider implements, and the crops it is fed.

PatchCore says *where* and *how much*. It cannot say *what*, because it has no
language: a coreset of patch features supports "this region is 43.9 standard
units from anything I was fitted on" and nothing more. An operator asking "what
is wrong with it?" needs a sentence about the object, and the only honest source
for that sentence is a model that looked at the pixels.

The provider is therefore a seam, not a feature. Three things are fixed here and
must not vary between implementations, because `narrative_source` is persisted
on every finding and has to mean the same thing forever:

1. **The prompt.** One system prompt, one user template, identical everywhere.
2. **The shape.** `VLMResult`, with `difference_visible` as a machine-checkable
   boolean rather than prose to be parsed.
3. **The sentinel.** `NO_VISIBLE_DIFFERENCE` returned verbatim when the model
   cannot see a difference, so "it found nothing" is distinguishable from "it
   failed" -- and neither is ever silently replaced with a guess.

Only `name` differs, and it lands in `narrative_source`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Returned verbatim by the model when it cannot see a physical difference.
#: A sentinel rather than a probability: the honest answer to "what is wrong
#: with it?" is sometimes "I cannot see anything wrong", and a system that
#: cannot express that will invent something instead.
NO_VISIBLE_DIFFERENCE = "NO_VISIBLE_DIFFERENCE"

VLM_SYSTEM_PROMPT = (
    "You compare two images of the same manufactured part and describe physical differences.\n"
    "You do not diagnose causes, assess severity, or speculate about function.\n"
    "If you cannot see a clear physical difference, say so plainly."
)

VLM_USER_TEMPLATE = (
    "Image A: a known-good reference for {model_name}, cropped to one region.\n"
    "Image B: the same region of a unit under inspection. An anomaly detector scored\n"
    "this region {anomaly_score:.2f}.\n"
    "\n"
    "Describe only what physically differs in image B relative to image A. Be specific\n"
    "about which component and in what way. One or two sentences. If no clear difference\n"
    f"is visible, respond exactly: {NO_VISIBLE_DIFFERENCE}"
)


def user_prompt(model_name: str, anomaly_score: float) -> str:
    return VLM_USER_TEMPLATE.format(model_name=model_name, anomaly_score=anomaly_score)


@dataclass(frozen=True)
class VLMResult:
    """What a provider saw. `description` is never empty."""

    description: str
    component: str | None
    difference_visible: bool
    #: Provider identity as it will be persisted, e.g. `vlm:qwen3-vl-32b-instruct`.
    source: str
    latency_ms: int

    @property
    def is_no_difference(self) -> bool:
        return not self.difference_visible


@runtime_checkable
class VLMProvider(Protocol):
    """A thing that can compare two crops and say what differs."""

    name: str

    def describe_difference(
        self,
        defect_crop: bytes,
        reference_crop: bytes,
        model_name: str,
        anomaly_score: float,
    ) -> VLMResult: ...

"""The VLM contract: resolution, parsing, cropping, and when NOT to call.

Every test here runs offline. The provider seam exists so that the interesting
properties -- a failure degrades, a refusal stays silent, an empty completion is
treated as a failure rather than a description -- are checkable without a key,
a network, or Atlas.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from gridsight.vision import vlm as vlm_module
from gridsight.vision.crops import crop_pair, hottest_region, region_box
from gridsight.vision.vlm import (
    FireworksVLM,
    NullVLM,
    _parse,
    describe_defect,
    resolve_provider,
)
from gridsight.vision.vlm_base import NO_VISIBLE_DIFFERENCE, VLMResult

REGION = {"x": 40, "y": 40, "w": 120, "h": 90, "score": 0.9}


class _Spy:
    """Counts calls so "no call was made" is measured, not assumed."""

    def __init__(self, raises: bool = False) -> None:
        self.name = "vlm:spy"
        self.calls = 0
        self._raises = raises

    def describe_difference(
        self,
        defect_crop: bytes,  # noqa: ARG002 -- the Protocol's shape is the point
        reference_crop: bytes,  # noqa: ARG002
        model_name: str,  # noqa: ARG002
        anomaly_score: float,  # noqa: ARG002
    ) -> VLMResult:
        self.calls += 1
        if self._raises:
            raise RuntimeError("forced provider failure")
        return VLMResult("a pin is bent", "pin 3", True, self.name, 1)


@pytest.fixture(autouse=True)
def _clean_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("VLM_PROVIDER", "FIREWORKS_API_KEY", "VLM_MODEL", "VLM_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    vlm_module.reset_provider()


def _frames() -> tuple[Image.Image, Image.Image]:
    return Image.new("RGB", (320, 240), (10, 40, 10)), Image.new("RGB", (320, 240), (10, 40, 10))


# --- resolution -------------------------------------------------------------


def test_explicit_none_wins_over_a_present_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_whatever")
    monkeypatch.setenv("VLM_PROVIDER", "none")
    provider, reason = resolve_provider()
    assert isinstance(provider, NullVLM)
    assert "VLM_PROVIDER=none" in reason


def test_a_present_key_selects_fireworks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_whatever")
    provider, reason = resolve_provider()
    assert isinstance(provider, FireworksVLM)
    assert provider.name.startswith("vlm:")
    assert "FIREWORKS_API_KEY is present" in reason


def test_no_key_and_no_local_weights_resolves_to_null() -> None:
    provider, reason = resolve_provider()
    assert isinstance(provider, NullVLM)
    # The reason has to say *why*, or a silently degraded deployment looks fine.
    assert "no Fireworks key" in reason


def test_fireworks_asks_for_the_declared_primary_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_whatever")
    monkeypatch.setenv("VLM_MODEL", "accounts/fireworks/models/some-other-vlm")
    provider, _ = resolve_provider()
    assert isinstance(provider, FireworksVLM)
    assert provider.name == "vlm:some-other-vlm"


# --- parsing ----------------------------------------------------------------


def test_parse_reads_the_structured_shape() -> None:
    payload = json.dumps({"description": "pin 3 is bent", "component": "pin 3", "difference_visible": True})
    assert _parse(payload) == ("pin 3 is bent", "pin 3", True)


def test_parse_honours_the_sentinel_even_when_the_flag_disagrees() -> None:
    payload = json.dumps(
        {"description": NO_VISIBLE_DIFFERENCE, "component": None, "difference_visible": True}
    )
    _, _, visible = _parse(payload)
    assert visible is False


def test_parse_keeps_prose_from_a_model_that_ignored_the_schema() -> None:
    description, component, visible = _parse("The capacitor is missing.")
    assert description == "The capacitor is missing."
    assert component is None
    assert visible is True


def test_parse_rejects_an_empty_completion() -> None:
    # An empty completion is a failed call wearing a success's clothes; letting
    # it through would persist a finding whose description is "".
    with pytest.raises(ValueError, match="empty completion"):
        _parse("   ")


def test_parse_rejects_structured_output_with_no_description() -> None:
    with pytest.raises(ValueError, match="empty description"):
        _parse(json.dumps({"description": "", "component": None, "difference_visible": True}))


# --- cropping ---------------------------------------------------------------


def test_region_box_pads_and_clamps_to_the_frame() -> None:
    left, top, right, bottom = region_box(REGION, 320, 240)
    assert 0 <= left < right <= 320
    assert 0 <= top < bottom <= 240
    assert (right - left) > REGION["w"], "the box should be padded outwards"


def test_region_box_never_leaves_the_frame_for_an_edge_region() -> None:
    left, top, right, bottom = region_box({"x": 0, "y": 0, "w": 20, "h": 20, "score": 1.0}, 100, 100)
    assert (left, top) == (0, 0)
    assert right <= 100 and bottom <= 100


def test_hottest_region_picks_the_highest_score() -> None:
    regions = [{"x": 0, "y": 0, "w": 1, "h": 1, "score": 0.2}, {"x": 5, "y": 5, "w": 1, "h": 1, "score": 0.9}]
    assert hottest_region(regions) == regions[1]
    assert hottest_region([]) is None


def test_crop_pair_maps_the_reference_through_normalised_coordinates() -> None:
    frame = Image.new("RGB", (320, 240))
    reference = Image.new("RGB", (160, 120))  # half size, as a resized golden can be
    defect, ref = crop_pair(frame, reference, REGION)
    assert defect and ref
    # Both are PNG; the reference crop must be about half the linear size.
    with Image.open(__import__("io").BytesIO(ref)) as r, Image.open(__import__("io").BytesIO(defect)) as d:
        assert r.width < d.width


# --- when NOT to call -------------------------------------------------------


def test_null_provider_issues_no_call_and_returns_nothing() -> None:
    frame, reference = _frames()
    vlm_module._RESOLVED = (NullVLM(), "test")
    assert describe_defect(frame, reference, [REGION], "m", 1.0) is None


def test_no_localised_region_means_no_call() -> None:
    frame, reference = _frames()
    spy = _Spy()
    vlm_module._RESOLVED = (spy, "test")
    assert describe_defect(frame, reference, [], "m", 1.0) is None
    assert spy.calls == 0, "nothing to crop to, so nothing should have been asked"


def test_no_golden_reference_means_no_call() -> None:
    frame, _ = _frames()
    spy = _Spy()
    vlm_module._RESOLVED = (spy, "test")
    assert describe_defect(frame, None, [REGION], "m", 1.0) is None
    assert spy.calls == 0, "no reference to compare against, so nothing should have been asked"


def test_a_failing_provider_degrades_instead_of_raising() -> None:
    frame, reference = _frames()
    spy = _Spy(raises=True)
    vlm_module._RESOLVED = (spy, "test")
    assert describe_defect(frame, reference, [REGION], "m", 1.0) is None
    assert spy.calls == 1

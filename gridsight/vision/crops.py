"""Cut the two crops a vision model compares.

Cropping is load-bearing, not an optimisation. A VLM handed a whole board
narrates the whole board: it lists every component it can see and volunteers a
defect somewhere, because "describe what differs" over a hundred components is
an invitation to confabulate. Handed a tight crop that an anomaly detector has
already localised, it describes what is actually in front of it.

Both crops are taken from the *same* region in normalised coordinates, so the
reference crop shows the same part of the same board -- otherwise the model is
comparing a header to a capacitor and any difference it reports is an artefact
of the framing.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

#: Fraction of the region's size added on each edge. A box tight to the anomaly
#: often clips the component that carries it; ~20% restores enough surrounding
#: geometry for "the third header pin" to be a statement the model can make.
CROP_PADDING = 0.2

#: Minimum crop edge in pixels. A 12px box upscaled to a model's input is noise;
#: below this the crop is widened around its own centre.
MIN_CROP_PX = 96


def _clamp_box(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> tuple[int, int, int, int]:
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(width, int(round(x + w)))
    bottom = min(height, int(round(y + h)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def region_box(
    region: dict[str, Any], width: int, height: int, padding: float = CROP_PADDING
) -> tuple[int, int, int, int]:
    """Padded pixel box for one `bbox_regions` entry, clamped to the frame."""
    x, y = float(region.get("x", 0)), float(region.get("y", 0))
    w, h = float(region.get("w", 0)), float(region.get("h", 0))

    pad_x, pad_y = w * padding, h * padding
    x, y, w, h = x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y

    if w < MIN_CROP_PX:
        x -= (MIN_CROP_PX - w) / 2
        w = MIN_CROP_PX
    if h < MIN_CROP_PX:
        y -= (MIN_CROP_PX - h) / 2
        h = MIN_CROP_PX

    return _clamp_box(x, y, w, h, width, height)


def hottest_region(regions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The highest-scoring region, or None when nothing was localised."""
    scored = [r for r in regions if isinstance(r, dict)]
    if not scored:
        return None
    return max(scored, key=lambda r: float(r.get("score", 0.0)))


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def crop_pair(
    frame: Image.Image,
    reference: Image.Image | None,
    region: dict[str, Any],
) -> tuple[bytes, bytes | None]:
    """(defect crop, reference crop) as PNG bytes, cropped to the same region.

    The reference is mapped through *normalised* coordinates rather than pixel
    ones: the golden exemplar and the query frame are both resized on the way
    into GridFS but not necessarily to the same size, and cropping the reference
    at the query's pixel offsets would silently sample the wrong part of it.
    """
    box = region_box(region, frame.width, frame.height)
    defect = to_png_bytes(frame.crop(box))
    if reference is None:
        return defect, None

    left, top, right, bottom = box
    scale_x = reference.width / frame.width
    scale_y = reference.height / frame.height
    ref_box = _clamp_box(
        left * scale_x,
        top * scale_y,
        (right - left) * scale_x,
        (bottom - top) * scale_y,
        reference.width,
        reference.height,
    )
    return defect, to_png_bytes(reference.crop(ref_box))

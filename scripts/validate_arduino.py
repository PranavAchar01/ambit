"""Run the withheld Arduino frames through the live agent and report what happened.

`scripts/ingest_arduino.py` writes four frames to `data/arduino_uno/heldout/` and
keeps them out of every split, so nothing in the training path has ever seen
them. This script feeds those four through `POST /inspect` -- the same endpoint
the UI and the phone use, over the real registry -- and prints what the agent
decided about each.

**This script measures. It does not tune.** If a held-out frame is refused or
flagged, it prints the diagnosis and exits non-zero. There are exactly two ways
that happens and each has one honest reading:

*Refused* -- the routing score fell below the model's own coverage gate. The gate
was calibrated on frames from the same session as the training set, so a refusal
means the envelope is tighter than the capture variance the rig actually
produces. The fix is more reference frames or a steadier rig, not a wider gate.

*Flagged as defect* -- the raw score exceeded the image threshold, which is the
worst score any training frame produced. That threshold is an **in-sample**
maximum: PatchCore scores a frame that is in its own memory bank at near-zero
distance, so the worst training score says nothing about a frame the bank has
never seen. `--loo` measures the gap by refitting with one frame held out at a
time; on this capture set every unseen normal exceeded it.

Loosening either number until the table goes green would produce a specialist
that passes this script and fails on stage, so neither is touched here.

Note this leaves state behind, deliberately: `POST /inspect` writes a `findings`
document and stores the uploaded frame in GridFS on every call, exactly as a real
inspection does. Four validation runs mean four findings.

Run::

    .venv/bin/python scripts/validate_arduino.py
    .venv/bin/python scripts/validate_arduino.py --api http://127.0.0.1:8000 --class arduino_uno
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gridsight.config import DATA_ROOT  # noqa: E402
from gridsight.ingest.local_corpus import HELDOUT_DIRNAME  # noqa: E402
from gridsight.train.train_class import (  # noqa: E402
    GATE_HOLDOUT_FRACTION,
    GATE_HOLDOUT_MIN,
    list_images,
)

log = logging.getLogger("gridsight.validate.arduino")

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_CLASS = "arduino_uno"


def _multipart(path: Path, asset_class: str | None) -> tuple[bytes, str]:
    """Build a multipart body by hand -- the repo has no requests dependency."""
    boundary = f"----ambit{uuid.uuid4().hex}"
    parts: list[bytes] = []
    if asset_class:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="asset_class"\r\n\r\n'
            f"{asset_class}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def inspect(api: str, path: Path, asset_class: str | None) -> dict[str, Any]:
    body, content_type = _multipart(path, asset_class)
    req = urllib.request.Request(  # noqa: S310 -- fixed localhost API, not user input
        f"{api.rstrip('/')}/inspect", data=body, headers={"Content-Type": content_type}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
        payload: dict[str, Any] = json.loads(resp.read())
    return payload


def gate_split(n_train: int) -> tuple[int, int]:
    """Re-derive the fit/holdout split the gate was calibrated on, read-only.

    Reported because it is the first thing to look at when a gate misbehaves on a
    small class, and it is not otherwise visible anywhere.
    """
    n_hold = min(n_train - 1, max(GATE_HOLDOUT_MIN, int(n_train * GATE_HOLDOUT_FRACTION)))
    n_fit = n_train - n_hold
    if n_fit < 2:
        return n_train, n_train
    return n_fit, n_hold


def leave_one_out(asset_class: str, coreset: float) -> int:
    """Measure what a genuinely unseen *normal* frame scores, and print it.

    The image threshold is calibrated as the worst score over the reference
    images, but PatchCore scores a frame that is in its own memory bank at
    near-zero distance -- so that maximum says nothing about a frame the bank has
    never seen. Refitting N times, each time holding one training frame out,
    turns every training frame into an unseen normal and measures the gap
    directly. This is a diagnostic: it changes nothing and writes nothing.
    """
    from PIL import Image  # noqa: PLC0415 -- heavy imports stay off the default path

    from gridsight.inference import run_inference  # noqa: PLC0415
    from gridsight.train.fewshot import fit_normal_only  # noqa: PLC0415
    from gridsight.train.store import PatchcoreConfig  # noqa: PLC0415

    paths = list_images(DATA_ROOT / asset_class / "train" / "good")
    if len(paths) < 3:
        log.error("leave-one-out needs at least 3 training frames, found %d", len(paths))
        return 1
    images = [Image.open(p).convert("RGB") for p in paths]
    cfg = PatchcoreConfig(coreset_sampling_ratio=coreset)

    print("\n" + "=" * 118)
    print(f"{'HELD-OUT FRAME':<18}{'RAW (unseen)':>14}{'IN-SAMPLE MAX':>16}{'RATIO':>9}  WOULD FLAG")
    print("=" * 118)
    unseen: list[float] = []
    in_sample: list[float] = []
    for i, path in enumerate(paths):
        module, stats = fit_normal_only([im for j, im in enumerate(images) if j != i], cfg)
        raw = run_inference(module, cfg, images[i]).raw_score
        thr = stats["image_threshold"]
        unseen.append(raw)
        in_sample.append(thr)
        print(f"{path.name:<18}{raw:>14.4f}{thr:>16.4f}{raw / thr:>9.4f}  {'YES' if raw >= thr else 'no'}")
    for img in images:
        img.close()

    flagged = sum(1 for r, t in zip(unseen, in_sample, strict=True) if r >= t)
    ratios = [r / t for r, t in zip(unseen, in_sample, strict=True)]
    print("=" * 118)
    print(
        f"unseen normals flagged as defect: {flagged}/{len(paths)}   "
        f"ratio unseen/in-sample-max: min {min(ratios):.4f} mean {sum(ratios) / len(ratios):.4f} "
        f"max {max(ratios):.4f}"
    )
    print(
        "An in-sample maximum is not a decision boundary for out-of-sample frames. A leave-one-out "
        "calibration measures the quantity the threshold is actually asked to bound."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the held-out Arduino frames against /inspect")
    parser.add_argument("--api", default=DEFAULT_API, help="base URL of a running Ambit API")
    parser.add_argument("--class", dest="asset_class", default=DEFAULT_CLASS)
    parser.add_argument(
        "--expect-model",
        default=None,
        help="model name every frame must route to (default: <class>-patchcore-v1)",
    )
    parser.add_argument(
        "--loo",
        action="store_true",
        help="diagnostic: refit N times holding one training frame out, to measure "
        "what an unseen normal frame actually scores",
    )
    parser.add_argument("--coreset", type=float, default=0.25, help="coreset ratio for --loo refits")
    parser.add_argument(
        "--hint",
        action="store_true",
        help="send asset_class as a form hint (off by default -- routing should not need it)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    if args.loo:
        return leave_one_out(args.asset_class, args.coreset)
    expect_model = args.expect_model or f"{args.asset_class}-patchcore-v1"

    heldout_dir = DATA_ROOT / args.asset_class / HELDOUT_DIRNAME
    frames = list_images(heldout_dir)
    if not frames:
        log.error("no held-out frames in %s -- run scripts/ingest_arduino.py first", heldout_dir)
        return 1

    n_train = len(list_images(DATA_ROOT / args.asset_class / "train" / "good"))
    n_fit, n_hold = gate_split(n_train)

    log.info("posting %d held-out frames to %s/inspect", len(frames), args.api.rstrip("/"))
    results: list[tuple[Path, dict[str, Any]]] = []
    for path in frames:
        try:
            results.append((path, inspect(args.api, path, args.asset_class if args.hint else None)))
        except urllib.error.URLError as exc:
            log.error("cannot reach %s (%s) -- start it with:", args.api, exc)
            log.error("  .venv/bin/uvicorn gridsight.api.main:app --host 127.0.0.1 --port 8000")
            return 1

    print("\n" + "=" * 118)
    # The verdict compares the RAW score to the image threshold
    # (`inference.run_inference`: `is_defect = raw >= img_thr`), so the raw score
    # sits next to the threshold. `anomaly_score` is the normalised 0-1 severity
    # and is NOT comparable to the threshold -- printing the two side by side
    # would invite exactly that misreading.
    print(
        f"{'FRAME':<16}{'ROUTED MODEL':<28}{'ROUTE':>9}{'GATE':>9}{'RAW':>9}"
        f"{'IMG THRESH':>12}{'NORM':>7}{'SEVERITY':>10}  VERDICT"
    )
    print("=" * 118)
    failures: list[str] = []
    for path, res in results:
        routed = res["routed_model"]["name"] or "-"
        candidates = res.get("candidates") or []
        top = candidates[0] if candidates else {}
        gate = top.get("gate")
        raw = res.get("raw_anomaly_score")
        anomaly = res.get("anomaly_score")
        thresh = res.get("image_threshold")
        print(
            f"{path.name:<16}{routed[:27]:<28}{res['routing_score']:>9.4f}"
            f"{(f'{gate:.4f}' if gate is not None else '-'):>9}"
            f"{(f'{raw:.3f}' if raw is not None else '-'):>9}"
            f"{(f'{thresh:.3f}' if thresh is not None else '-'):>12}"
            f"{(f'{anomaly:.3f}' if anomaly is not None else '-'):>7}"
            f"{res.get('severity', 0.0):>10.3f}  {res['verdict'].upper()}"
        )
        if routed != expect_model:
            failures.append(f"{path.name}: routed to {routed!r}, expected {expect_model!r}")
        if res["verdict"] != "nominal":
            failures.append(f"{path.name}: verdict {res['verdict']!r}, expected 'nominal'")
    print("=" * 118)
    print(
        f"gate calibration: centroid fitted on {n_fit} of {n_train} training frames, "
        f"gate measured on {n_hold}"
    )
    print(f"expected model  : {expect_model}")

    if not failures:
        print(
            f"\nOK: {len(results)}/{len(results)} held-out frames routed to {expect_model} "
            "and returned nominal."
        )
        return 0

    print("\nFAILED ASSERTIONS:")
    for f in failures:
        print(f"  - {f}")

    refused = [r for _, r in results if r["verdict"] == "unroutable"]
    flagged = [r for _, r in results if r["verdict"] == "defect"]
    print("\nDIAGNOSIS:")
    if refused:
        scores = [r["routing_score"] for r in refused]
        gates = [((r.get("candidates") or [{}])[0]).get("gate") for r in refused]
        known = [g for g in gates if g is not None]
        print(
            f"  {len(refused)} frame(s) REFUSED: routing score {min(scores):.4f}-{max(scores):.4f} "
            f"against a gate of {(f'{min(known):.4f}' if known else 'unknown')}."
        )
        print(
            "  The coverage gate is tighter than the capture variance this rig produces. The gate is "
            f"the 5th percentile of {n_hold} held-out training frames' similarity to a centroid fitted "
            f"on {n_fit}."
        )
        print(
            "  RECOMMENDED: capture more reference frames spanning the real variation, and re-run "
            "the ingest and training. Do NOT lower the gate to make this pass -- the gate is the "
            "abstention property the whole system is built on."
        )
    if flagged:
        scores = [r.get("raw_anomaly_score") or 0.0 for r in flagged]
        thresholds = [r.get("image_threshold") or 0.0 for r in flagged]
        print(
            f"  {len(flagged)} frame(s) FLAGGED as defect: raw score {min(scores):.3f}-{max(scores):.3f} "
            f"vs image threshold {min(thresholds):.3f}."
        )
        print(
            "  The image threshold is the worst score any *training* frame produced -- and PatchCore "
            "scores a frame that is in its own memory bank at near-zero distance. The threshold is "
            "therefore an in-sample maximum with structurally no headroom for a frame that is not in "
            "the bank, which is every frame the model will ever be shown in production."
        )
        print("  Re-run with --loo to measure what an unseen normal frame actually scores.")
        print(
            "  RECOMMENDED: calibrate the image threshold leave-one-out rather than in-sample -- the "
            "same correction the routing gate already carries, where the percentile was moved off the "
            "training images for exactly this reason (train_class.ROUTING_GATE_PERCENTILE, and the "
            "comment above it). Do NOT simply raise the threshold until this table goes green."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

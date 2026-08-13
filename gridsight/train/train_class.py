"""Phase 3 -- fit a PatchCore specialist per asset class and register it in MongoDB.

The per-class loop is a LangGraph graph checkpointed to MongoDB. That is not
decoration: each class that finishes is committed to the checkpoint, so a
`kill -9` halfway through a fleet training run resumes at the class it died on
instead of relearning the ones already in the registry.
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from gridfs.errors import NoFile
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from PIL import Image

from gridsight.config import ARTIFACT_ROOT, DATA_ROOT, get_settings
from gridsight.db.mongo import (
    ensure_collections,
    ensure_vector_index,
    get_client,
    images_bucket,
    models_col,
    weights_bucket,
)
from gridsight.embed import centroid, get_embedder
from gridsight.inference import inference_device, run_inference
from gridsight.train.fewshot import (
    PIXEL_THRESHOLD_PERCENTILE,
    apply_pixel_stats,
    calibrate_pixel_stats,
)
from gridsight.train.store import (
    AnomalyModule,
    PatchcoreConfig,
    post_processor,
    put_weights,
    serialize_patchcore,
    storage_report,
)

log = logging.getLogger("gridsight.train")

STORAGE_WARN_PCT = 75.0


def class_dir(asset_class: str) -> Path:
    return DATA_ROOT / asset_class


def available_classes() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in DATA_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / "train" / "good").is_dir()
    )


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.png"))


# ---------------------------------------------------------------------------
# Training a single class
# ---------------------------------------------------------------------------


def mask_dir(asset_class: str) -> Path | None:
    """Ground-truth mask directory, if the source dataset shipped one.

    Only mask-bearing sources (MVTec AD) have one. Handing anomalib a mask_dir
    that does not exist -- or inventing empty masks for the powerline classes --
    would produce a pixel AUROC measured against a fabricated target, so the
    absence of masks is propagated as `pixel_auroc: null` instead.
    """
    d = class_dir(asset_class) / "ground_truth"
    return d if d.is_dir() and any(d.glob("*.png")) else None


def fit_class(asset_class: str, cfg: PatchcoreConfig) -> tuple[AnomalyModule, dict[str, float | None]]:
    """Fit PatchCore on train/good and evaluate on the held-out test split."""
    root = class_dir(asset_class)
    masks = mask_dir(asset_class)
    datamodule = Folder(
        name=asset_class,
        root=root,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defect",
        mask_dir="ground_truth" if masks else None,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        val_split_mode="from_test",
        val_split_ratio=0.5,
        seed=1234,
    )
    from gridsight.train.store import build_patchcore

    module = build_patchcore(cfg)
    engine = Engine(
        default_root_dir=str(ARTIFACT_ROOT / asset_class),
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    engine.fit(model=module, datamodule=datamodule)
    results = engine.test(model=module, datamodule=datamodule, verbose=False)

    metrics: dict[str, float | None] = {"image_auroc": None, "pixel_auroc": None}
    if results:
        flat = dict(results[0])
        for key, value in flat.items():
            low = key.lower()
            if "image_auroc" in low:
                metrics["image_auroc"] = round(float(value), 4)
            elif "pixel_auroc" in low:
                metrics["pixel_auroc"] = round(float(value), 4)
    if masks is None and metrics["pixel_auroc"] is not None:
        raise RuntimeError(f"{asset_class}: pixel AUROC reported without masks -- refusing to record it")
    log.info(
        "[%s] pixel metrics %s (mask_dir=%s)",
        asset_class,
        "measured against ground truth" if masks else "unavailable, recorded as null",
        masks,
    )
    # anomalib leaves `_pixel_threshold` as NaN for the mask-less classes, so it
    # is calibrated from the training set's own anomaly maps -- otherwise
    # heatmaps and region extraction silently produce NaN. The calibration is
    # applied to the mask-bearing classes too, overwriting the F1-optimal
    # threshold anomalib can now compute for them: inference normalises every
    # model's map against this same p99.5-of-good-training statistic, and a
    # registry where two models mean different things by "pixel_threshold" would
    # make heatmaps incomparable across a route. The *metric* is untouched --
    # `pixel_auroc` above is threshold-free and measured against real masks.
    train_images = [Image.open(p).convert("RGB") for p in list_images(root / "train" / "good")]
    module.to(inference_device())
    pixel_stats = calibrate_pixel_stats(module, train_images, cfg.image_size)
    apply_pixel_stats(module, pixel_stats)
    for img in train_images:
        img.close()
    log.info(
        "[%s] calibrated pixel threshold to %.4f (p%.1f over %d good training maps)",
        asset_class,
        pixel_stats["pixel_threshold"],
        PIXEL_THRESHOLD_PERCENTILE,
        len(train_images),
    )
    return module, metrics


#: A model's routing gate is the Nth percentile of *held-out* normal frames'
#: similarity to its centroid -- the coverage envelope it actually generalises to.
#:
#: The gate is min(this percentile, median - MIN_GATE_MARGIN). Both parts were needed
#: and the history is worth keeping, because the obvious fix did not work:
#:   1. Originally the percentile was taken over the TRAINING images. That is
#:      optimistic -- the centroid is fitted on those exact frames.
#:   2. Moving to a held-out slice removed that bias but did NOT fix `mvtec_leather`,
#:      which still refused 5/30 of its own good frames. Measurement showed why: the
#:      shift is train->test, not fit->holdout. 23% of leather's test/good frames fall
#:      below its entire training minimum, so no slice of train can predict them.
#:   3. The MIN_GATE_MARGIN floor fixed it: leather 5/30 -> 30/30, bottle 17/20 -> 20/20,
#:      tile 27/30 -> 30/30, every other class unchanged (their percentile was already
#:      wider than the floor).
ROUTING_GATE_PERCENTILE = 5.0

#: Fraction of train/good withheld from the centroid purely to calibrate the gate.
#: Small enough not to hurt the centroid, large enough for a stable percentile.
GATE_HOLDOUT_FRACTION = 0.2
GATE_HOLDOUT_MIN = 8

#: No model may claim a coverage envelope tighter than this below its own median.
#:
#: Measured justification: in-class held-out normals span 0.149 (insulator) and
#: 0.022 (mvtec_tile), but only 0.0035 for mvtec_leather -- a texture so homogeneous
#: that CLIP similarity saturates and the residual variation is noise, not signal.
#: In that band a percentile is meaningless: 23% of leather's own test/good frames
#: fall BELOW its entire training minimum, so any train-derived percentile refuses
#: them. Holding out from train does not help, because the shift is train->test, not
#: fit->holdout. The floor states the honest limit instead: differences smaller than
#: this are below the resolution at which this embedding space distinguishes coverage.
MIN_GATE_MARGIN = 0.012


def routing_embedding(asset_class: str) -> tuple[np.ndarray, int, Path, float]:
    """CLIP centroid over train/good, the image nearest it, and the routing gate.

    The centroid is fitted on one slice and the gate measured on another, so the
    gate answers "how far from the centroid does an *unseen* normal frame fall?"
    rather than "how tightly did I memorise my training set?".
    """
    paths = list_images(class_dir(asset_class) / "train" / "good")
    if not paths:
        raise FileNotFoundError(f"no training images for {asset_class}")

    vectors = get_embedder().embed_paths(paths)

    # Deterministic fit/holdout split, seeded so a re-registration reproduces the
    # same gate rather than drifting on every run.
    order = np.random.default_rng(1234).permutation(len(paths))
    n_hold = min(len(paths) - 1, max(GATE_HOLDOUT_MIN, int(len(paths) * GATE_HOLDOUT_FRACTION)))
    hold_idx, fit_idx = order[:n_hold], order[n_hold:]
    if len(fit_idx) < 2:  # tiny cold-start sets cannot afford a holdout
        hold_idx, fit_idx = order, order

    c = centroid(vectors[fit_idx])
    # Mirror Atlas's cosine scoring, (1 + cos) / 2, so the gate is directly
    # comparable to a $vectorSearch score.
    holdout = (1.0 + vectors[hold_idx] @ c) / 2.0
    percentile_gate = float(np.percentile(holdout, ROUTING_GATE_PERCENTILE))
    floored_gate = float(np.median(holdout)) - MIN_GATE_MARGIN
    gate = min(percentile_gate, floored_gate)

    # The golden exemplar is the frame nearest the centroid across the whole set.
    golden_idx = int(np.argmax(vectors @ c))
    return c, len(paths), paths[golden_idx], gate


def store_reference_image(path: Path, asset_class: str) -> Any:
    """Put the golden known-good exemplar into GridFS for the UI comparison."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        buf = __import__("io").BytesIO()
        rgb.save(buf, format="PNG", optimize=True)
        blob = buf.getvalue()
    return images_bucket().upload_from_stream(
        f"reference_{asset_class}.png", blob, metadata={"kind": "reference", "asset_class": asset_class}
    )


def register_model(
    asset_class: str,
    module: AnomalyModule,
    cfg: PatchcoreConfig,
    metrics: dict[str, float | None],
    created_by: str = "seed",
    name: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize weights into GridFS and write the registry document."""
    embedding, n_train, golden, routing_gate = routing_embedding(asset_class)

    blob = serialize_patchcore(module, cfg)
    report = storage_report()
    if report["pct_used"] > STORAGE_WARN_PCT:
        log.warning(
            "M0 storage at %.1f%% of 512MB (%.1f MB used) -- prune old models before adding more",
            report["pct_used"],
            report["data_mb"],
        )

    weights_id = put_weights(
        blob,
        f"{asset_class}_patchcore.pt",
        metadata={"asset_class": asset_class, "backbone": cfg.backbone, "bytes": len(blob)},
    )
    reference_id = store_reference_image(golden, asset_class)

    doc: dict[str, Any] = {
        "name": name or f"{asset_class}-patchcore-v1",
        "asset_class": asset_class,
        "backbone": f"patchcore/{cfg.backbone}",
        "embedding": [float(x) for x in embedding],
        "embedding_count": n_train,
        "routing_threshold": routing_gate,
        "image_threshold": float(post_processor(module).state_dict()["_image_threshold"].item()),
        "pixel_threshold": float(post_processor(module).state_dict()["_pixel_threshold"].item()),
        "weights_file_id": weights_id,
        "weights_bytes": len(blob),
        "reference_image_id": reference_id,
        "training_samples": n_train,
        "metrics": metrics,
        "created_at": datetime.now(UTC),
        "created_by": created_by,
        "provenance": provenance
        or {
            "source": "phase3-seed",
            "train_dir": str(class_dir(asset_class) / "train" / "good"),
            "config": cfg.to_dict(),
            "golden_reference": golden.name,
        },
    }
    # Replacing a registry entry orphans its old GridFS blob; on a 512MB M0 that
    # leak is the difference between six models fitting and not.
    previous = models_col().find_one({"name": doc["name"]}, {"weights_file_id": 1, "reference_image_id": 1})
    models_col().replace_one({"name": doc["name"]}, doc, upsert=True)
    if previous:
        for bucket, key in ((weights_bucket(), "weights_file_id"), (images_bucket(), "reference_image_id")):
            old = previous.get(key)
            if old is not None:
                with suppress(NoFile):
                    bucket.delete(old)
                    log.info("pruned superseded %s %s", key, old)

    stored = models_col().find_one({"name": doc["name"]})
    assert stored is not None
    log.info(
        "[%s] registered %s (weights %.2f MB, %d-d embedding over %d samples, "
        "routing_gate=%.4f, image_auroc=%s)",
        asset_class,
        doc["name"],
        len(blob) / 1e6,
        len(doc["embedding"]),
        n_train,
        routing_gate,
        metrics["image_auroc"],
    )
    return stored


def verify_roundtrip(
    asset_class: str, module: AnomalyModule, cfg: PatchcoreConfig, weights_id: Any
) -> dict[str, Any]:
    """Score a probe frame in-process, then again from a GridFS-reloaded copy."""
    from gridsight.train.store import deserialize_patchcore, get_weights

    probes = list_images(class_dir(asset_class) / "test" / "defect")[:1]
    if not probes:
        probes = list_images(class_dir(asset_class) / "train" / "good")[:1]
    with Image.open(probes[0]) as im:
        probe = im.convert("RGB").copy()

    module.to(inference_device())
    in_process = run_inference(module, cfg, probe).raw_score

    reloaded, rcfg = deserialize_patchcore(get_weights(weights_id))
    from_gridfs = run_inference(reloaded, rcfg, probe).raw_score

    delta = abs(in_process - from_gridfs)
    return {
        "probe": probes[0].name,
        "in_process_score": round(in_process, 6),
        "gridfs_score": round(from_gridfs, 6),
        "delta": delta,
        "match": delta < 1e-5,
    }


def train_and_register(asset_class: str, cfg: PatchcoreConfig) -> dict[str, Any]:
    started = time.time()
    log.info(
        "[%s] fitting PatchCore(%s, coreset=%.3f)", asset_class, cfg.backbone, cfg.coreset_sampling_ratio
    )
    module, metrics = fit_class(asset_class, cfg)
    doc = register_model(asset_class, module, cfg, metrics)
    rt = verify_roundtrip(asset_class, module, cfg, doc["weights_file_id"])
    log.info(
        "[%s] GridFS round-trip: in-process=%.6f reloaded=%.6f delta=%.2e match=%s",
        asset_class,
        rt["in_process_score"],
        rt["gridfs_score"],
        rt["delta"],
        rt["match"],
    )
    if not rt["match"]:
        raise RuntimeError(f"{asset_class}: GridFS round-trip score mismatch ({rt})")

    return {
        "asset_class": asset_class,
        "model_name": doc["name"],
        "weights_file_id": str(doc["weights_file_id"]),
        "weights_mb": round(doc["weights_bytes"] / 1e6, 3),
        "embedding_dim": len(doc["embedding"]),
        "training_samples": doc["training_samples"],
        "metrics": metrics,
        "roundtrip": rt,
        "seconds": round(time.time() - started, 1),
    }


# ---------------------------------------------------------------------------
# Checkpointed fleet-training graph
# ---------------------------------------------------------------------------


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class TrainState(TypedDict):
    pending: list[str]
    completed: list[str]
    results: Annotated[dict[str, Any], _merge]
    coreset_ratio: float


def _train_next(state: TrainState) -> dict[str, Any]:
    pending = list(state["pending"])
    if not pending:
        return {}
    asset_class = pending[0]
    cfg = PatchcoreConfig(coreset_sampling_ratio=state["coreset_ratio"])
    summary = train_and_register(asset_class, cfg)
    return {
        "pending": pending[1:],
        "completed": [*state["completed"], asset_class],
        "results": {asset_class: summary},
    }


def _should_continue(state: TrainState) -> str:
    return "train_next" if state["pending"] else END


def build_training_graph(saver: MongoDBSaver) -> Any:
    graph = StateGraph(TrainState)
    graph.add_node("train_next", _train_next)
    graph.add_edge(START, "train_next")
    graph.add_conditional_edges("train_next", _should_continue, {"train_next": "train_next", END: END})
    return graph.compile(checkpointer=saver)


def make_saver() -> MongoDBSaver:
    return MongoDBSaver(
        client=get_client(),
        db_name=get_settings().mongodb_db,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and register PatchCore specialists")
    parser.add_argument("--classes", nargs="*", help="asset classes to train (default: all scraped)")
    parser.add_argument("--exclude", nargs="*", default=[], help="asset classes to hold out")
    parser.add_argument("--thread-id", default="fleet-training-v1", help="checkpoint thread id")
    parser.add_argument("--resume", action="store_true", help="resume the thread instead of starting fresh")
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for noisy in ("httpx", "urllib3", "lightning.pytorch", "anomalib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.set_grad_enabled(False)

    ensure_collections()
    if not args.skip_index:
        ensure_vector_index()

    targets = [c for c in (args.classes or available_classes()) if c not in set(args.exclude)]
    if not targets:
        log.error("no asset classes to train; run the ingest phase first")
        return 1

    saver = make_saver()
    app = build_training_graph(saver)
    config = {"configurable": {"thread_id": args.thread_id}}

    if args.resume:
        snapshot = app.get_state(config)
        if not snapshot.values:
            log.error("no checkpoint found for thread_id=%s -- nothing to resume", args.thread_id)
            return 1
        done = snapshot.values.get("completed", [])
        left = snapshot.values.get("pending", [])
        log.info("RESUMING thread_id=%s | already trained=%s | remaining=%s", args.thread_id, done, left)
        final = app.invoke(None, config=config)
    else:
        log.info("training %d classes: %s (thread_id=%s)", len(targets), targets, args.thread_id)
        final = app.invoke(
            {"pending": targets, "completed": [], "results": {}, "coreset_ratio": args.coreset},
            config=config,
        )

    print("\n" + "=" * 123)
    print(
        f"{'ASSET CLASS':<20}{'WEIGHTS MB':>11}{'EMB':>5}{'SAMPLES':>9}{'IMG AUROC':>11}{'PIX AUROC':>11}"
        f"{'IN-PROC':>11}{'GRIDFS':>11}{'MATCH':>7}"
    )
    print("=" * 123)
    for name in sorted(final.get("results", {})):
        r = final["results"][name]
        rt = r["roundtrip"]
        img_auroc = r["metrics"]["image_auroc"]
        pix_auroc = r["metrics"]["pixel_auroc"]
        print(
            f"{name:<20}{r['weights_mb']:>11.3f}{r['embedding_dim']:>5}{r['training_samples']:>9}"
            f"{(f'{img_auroc:.4f}' if img_auroc is not None else 'n/a'):>11}"
            f"{(f'{pix_auroc:.4f}' if pix_auroc is not None else 'null'):>11}"
            f"{rt['in_process_score']:>11.5f}{rt['gridfs_score']:>11.5f}{str(rt['match']):>7}"
        )
    print("=" * 123)
    report = storage_report()
    print(
        f"registered models: {models_col().count_documents({})} | "
        f"Atlas M0 usage: {report['data_mb']:.1f} MB / 512 MB ({report['pct_used']}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

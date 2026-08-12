"""Persist PatchCore weights *in MongoDB* via GridFS, and load them back.

What actually gets stored is the coreset memory bank plus the post-processor's
threshold/normalisation statistics. The feature-extraction backbone is frozen
ImageNet weights identified by name, so it is reconstructed deterministically at
load time rather than duplicated into every registry entry -- that keeps each
model a few megabytes instead of a few hundred, which is what makes six models
fit comfortably inside an M0's 512MB ceiling.
"""

from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from anomalib.models import Padim, Patchcore
from bson import ObjectId

from gridsight.db.mongo import get_db, weights_bucket

log = logging.getLogger("gridsight.store")

FORMAT_VERSION = 2
DEFAULT_BACKBONE = "wide_resnet50_2"
DEFAULT_LAYERS = ("layer2", "layer3")

# Coreset ratio is deliberately small: it is the single lever that decides how
# many megabytes land in GridFS per model. 1% is the ratio used in the PatchCore
# paper's MVTec results, so this is the published operating point, not a hack.
DEFAULT_CORESET_RATIO = 0.01


@dataclass(frozen=True)
class PatchcoreConfig:
    backbone: str = DEFAULT_BACKBONE
    layers: tuple[str, ...] = DEFAULT_LAYERS
    coreset_sampling_ratio: float = DEFAULT_CORESET_RATIO
    num_neighbors: int = 9
    image_size: int = 256

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PatchcoreConfig:
        return PatchcoreConfig(
            backbone=d.get("backbone", DEFAULT_BACKBONE),
            layers=tuple(d.get("layers", DEFAULT_LAYERS)),
            coreset_sampling_ratio=float(d.get("coreset_sampling_ratio", DEFAULT_CORESET_RATIO)),
            num_neighbors=int(d.get("num_neighbors", 9)),
            image_size=int(d.get("image_size", 256)),
        )


def build_patchcore(cfg: PatchcoreConfig) -> Patchcore:
    """Construct an untrained PatchCore with a frozen pretrained backbone."""
    return Patchcore(
        backbone=cfg.backbone,
        layers=list(cfg.layers),
        pre_trained=True,
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
    )


def build_padim() -> Padim:
    """PaDiM fallback for reference sets too small for a meaningful coreset."""
    return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


#: PaDiM keeps its learned state in these buffers rather than a memory bank.
PADIM_BUFFERS = ("idx", "gaussian.mean", "gaussian.inv_covariance")

AnomalyModule = Patchcore | Padim


def post_processor(module: AnomalyModule) -> torch.nn.Module:
    """anomalib types `post_processor` as Optional; a fitted model always has one."""
    pp = module.post_processor
    if pp is None:
        raise ValueError(
            f"{type(module).__name__} has no post-processor; it was built with post_processor=False"
        )
    return pp


def serialize_patchcore(module: AnomalyModule, cfg: PatchcoreConfig) -> bytes:
    """torch.save the learned state + thresholds into an in-memory buffer."""
    payload: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "config": cfg.to_dict(),
        "post_processor": {k: v.detach().cpu() for k, v in post_processor(module).state_dict().items()},
    }

    if isinstance(module, Padim):
        buffers = dict(module.model.named_buffers())
        state = {name: buffers[name].detach().cpu().clone() for name in PADIM_BUFFERS}
        if state["gaussian.mean"].numel() == 0:
            raise ValueError("refusing to serialize a PaDiM with an unfitted Gaussian")
        payload["kind"] = "padim"
        payload["padim_state"] = state
    else:
        memory_bank = module.model.memory_bank.detach().cpu().contiguous().float()
        if memory_bank.numel() == 0:
            raise ValueError("refusing to serialize a PatchCore with an empty memory bank")
        payload["kind"] = "patchcore"
        payload["memory_bank"] = memory_bank

    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def deserialize_patchcore(blob: bytes) -> tuple[AnomalyModule, PatchcoreConfig]:
    """Rebuild a ready-to-infer model from a serialized blob."""
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported weight format {payload.get('format')!r}")

    cfg = PatchcoreConfig.from_dict(payload["config"])
    module: AnomalyModule
    if payload.get("kind", "patchcore") == "padim":
        module = build_padim()
        state = payload["padim_state"]
        for name in PADIM_BUFFERS:
            parent: Any = module.model
            *path, leaf = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, leaf, state[name].clone())
    else:
        module = build_patchcore(cfg)
        # the buffer is registered empty; replace it with the stored coreset
        module.model.memory_bank = payload["memory_bank"].clone()

    post_processor(module).load_state_dict(payload["post_processor"])
    module.eval()
    return module, cfg


# ---------------------------------------------------------------------------
# GridFS
# ---------------------------------------------------------------------------


def put_weights(blob: bytes, filename: str, metadata: dict[str, Any] | None = None) -> ObjectId:
    file_id = weights_bucket().upload_from_stream(filename, blob, metadata=metadata or {})
    log.info("stored %s (%.2f MB) in GridFS as %s", filename, len(blob) / 1e6, file_id)
    return ObjectId(str(file_id))


def get_weights(file_id: ObjectId | str) -> bytes:
    buf = io.BytesIO()
    weights_bucket().download_to_stream(ObjectId(str(file_id)), buf)
    return buf.getvalue()


def delete_weights(file_id: ObjectId | str) -> None:
    weights_bucket().delete(ObjectId(str(file_id)))


def storage_report() -> dict[str, float]:
    """M0 has a 512MB ceiling; callers use this to prune before it bites."""
    stats = get_db().command("dbstats")
    used_mb = float(stats["dataSize"]) / 1e6
    return {
        "data_mb": round(used_mb, 2),
        "storage_mb": round(float(stats["storageSize"]) / 1e6, 2),
        "index_mb": round(float(stats.get("indexSize", 0)) / 1e6, 2),
        "limit_mb": 512.0,
        "pct_used": round(100.0 * used_mb / 512.0, 1),
    }

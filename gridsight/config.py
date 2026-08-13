"""Central configuration for Ambit, loaded from the environment.

Secrets never live in source. `.env` is git-ignored and read at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_ROOT = REPO_ROOT / "data"
HF_CACHE = DATA_ROOT / ".hf_cache"
ARTIFACT_ROOT = REPO_ROOT / "artifacts"

#: Asset classes Ambit tries to populate from the Hugging Face Hub.
ASSET_CLASSES: tuple[str, ...] = (
    "insulator",
    "corrosion",
    "vegetation",
    "transmission_tower",
    "rail_surface",
    "catenary",
)


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set. Populate it in {REPO_ROOT / '.env'}.")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the process environment."""

    mongodb_uri: str
    mongodb_db: str
    openai_api_key: str
    openai_model: str
    hf_token: str
    vector_index_name: str
    recall_index_name: str
    route_threshold: float
    route_ambiguous_floor: float
    clip_model: str
    clip_pretrained: str
    embed_dim: int
    image_size: int
    allowed_origins: list[str] = field(default_factory=list)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve settings once per process."""
    origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return Settings(
        mongodb_uri=_require("MONGODB_URI"),
        mongodb_db=os.environ.get("MONGODB_DB", "gridsight"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        hf_token=os.environ.get("HF_TOKEN", "").strip(),
        vector_index_name=os.environ.get("VECTOR_INDEX_NAME", "model_router_idx"),
        recall_index_name=os.environ.get("RECALL_INDEX_NAME", "finding_recall_idx"),
        route_threshold=float(os.environ.get("ROUTE_THRESHOLD", "0.82")),
        route_ambiguous_floor=float(os.environ.get("ROUTE_AMBIGUOUS_FLOOR", "0.70")),
        clip_model=os.environ.get("CLIP_MODEL", "ViT-B-32"),
        clip_pretrained=os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k"),
        embed_dim=int(os.environ.get("EMBED_DIM", "512")),
        image_size=int(os.environ.get("IMAGE_SIZE", "256")),
        allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
    )

"""Shared fixtures.

Tests that touch Atlas are marked `live` and skip cleanly when the cluster is
unreachable, so `pytest` still runs offline. They are not mocked -- a passing
`live` test means the real cluster did the work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def _atlas_available() -> bool:
    if not os.environ.get("MONGODB_URI"):
        return False
    try:
        from gridsight.db.mongo import get_client

        get_client().admin.command("ping")
    except Exception:  # noqa: BLE001 - any failure means "cannot run live tests"
        return False
    return True


ATLAS = _atlas_available()
requires_atlas = pytest.mark.skipif(not ATLAS, reason="Atlas cluster not reachable")


@pytest.fixture(scope="session")
def data_root() -> Path:
    from gridsight.config import DATA_ROOT

    if not DATA_ROOT.exists():
        pytest.skip("data/ not populated; run gridsight.ingest.hf_scrape first")
    return DATA_ROOT


@pytest.fixture(scope="session")
def scraped_classes(data_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / "train" / "good").is_dir()
    )


def first_probe(data_root: Path, classes: list[str], split: str) -> Path | None:
    """The first frame in `split` across `classes`, or None if no class has one.

    Tests used to index `classes[0]` and glob directly, which assumes every
    registered class ships a labelled test split. A normal-only class -- captured
    on a bench, no defect set -- has neither `test/good` nor `test/defect`, and
    sorts first alphabetically, so that assumption turned into an IndexError
    rather than into the skip the test intended.
    """
    for cls in classes:
        found = sorted((data_root / cls / split).glob("*.png"))
        if found:
            return found[0]
    return None


@pytest.fixture(scope="session")
def registry_models() -> list[dict[str, Any]]:
    if not ATLAS:
        pytest.skip("Atlas cluster not reachable")
    from gridsight.db.mongo import models_col

    docs = list(models_col().find())
    if not docs:
        pytest.skip("registry is empty; run gridsight.train.train_class first")
    return docs


@pytest.fixture
def sample_image(data_root: Path, scraped_classes: list[str]) -> Image.Image:
    for cls in scraped_classes:
        candidates = sorted((data_root / cls / "test" / "defect").glob("*.png"))
        if candidates:
            with Image.open(candidates[0]) as im:
                return im.convert("RGB").copy()
    pytest.skip("no defect images available")

"""The "MongoDB holds the weights" claim, tested literally."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from conftest import requires_atlas
from gridsight.inference import run_inference
from gridsight.train.store import (
    PatchcoreConfig,
    deserialize_patchcore,
    get_weights,
    put_weights,
    serialize_patchcore,
)


def test_config_survives_a_dict_round_trip() -> None:
    cfg = PatchcoreConfig(backbone="wide_resnet50_2", coreset_sampling_ratio=0.02, image_size=256)
    assert PatchcoreConfig.from_dict(cfg.to_dict()) == cfg


@requires_atlas
def test_every_registered_model_reloads_from_gridfs_with_the_same_score(
    registry_models: list[dict[str, Any]], sample_image: Image.Image
) -> None:
    for doc in registry_models:
        assert doc["weights_file_id"] is not None, f"{doc['name']} has no weights in GridFS"
        assert len(doc["embedding"]) == 512, f"{doc['name']} embedding is not 512-d"

        blob = get_weights(doc["weights_file_id"])
        first, cfg = deserialize_patchcore(blob)
        second, cfg2 = deserialize_patchcore(blob)

        a = run_inference(first, cfg, sample_image).raw_score
        b = run_inference(second, cfg2, sample_image).raw_score
        assert a == pytest.approx(b, abs=1e-5), f"{doc['name']}: {a} vs {b} after reload"


@requires_atlas
def test_serialize_then_store_then_load_preserves_the_memory_bank(
    registry_models: list[dict[str, Any]],
) -> None:
    doc = registry_models[0]
    module, cfg = deserialize_patchcore(get_weights(doc["weights_file_id"]))

    blob = serialize_patchcore(module, cfg)
    file_id = put_weights(blob, "roundtrip_test.pt", metadata={"kind": "test"})
    try:
        reloaded, _ = deserialize_patchcore(get_weights(file_id))
        original = module.model.memory_bank.detach().cpu().numpy()
        copy = reloaded.model.memory_bank.detach().cpu().numpy()
        assert original.shape == copy.shape
        assert np.array_equal(original, copy)
    finally:
        from gridsight.train.store import delete_weights

        delete_weights(file_id)


def test_serializing_an_unfitted_model_is_refused() -> None:
    from gridsight.train.store import build_patchcore

    cfg = PatchcoreConfig()
    with pytest.raises(ValueError, match="empty memory bank"):
        serialize_patchcore(build_patchcore(cfg), cfg)

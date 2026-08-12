"""Verify Atlas $vectorSearch end-to-end and measure real routing behaviour.

Embeds held-out test frames from every scraped class -- including the one
deliberately withheld from training -- and reports what the live vector index
returns, how often the correct specialist wins, and how the per-model coverage
gate trades in-registry acceptance against out-of-registry refusal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from gridsight.config import DATA_ROOT, get_settings
from gridsight.db.mongo import models_col, search_index_status, vector_search
from gridsight.embed import get_embedder

logging.basicConfig(level=logging.WARNING)

PROBES_PER_CLASS = 25


def probe_paths(asset_class: str, n: int = PROBES_PER_CLASS) -> list[Path]:
    return sorted((DATA_ROOT / asset_class / "test" / "good").glob("*.png"))[:n]


def main() -> int:
    settings = get_settings()
    embedder = get_embedder()

    status = search_index_status(settings.vector_index_name)
    registered = {
        d["asset_class"]: float(d.get("routing_threshold") or 0.0)
        for d in models_col().find({}, {"asset_class": 1, "routing_threshold": 1})
    }
    all_classes = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))
    withheld = [c for c in all_classes if c not in registered]

    print(f"vector index '{settings.vector_index_name}' status : {status}")
    print(
        f"registered classes + coverage gates          : "
        f"{ {k: round(v, 4) for k, v in sorted(registered.items())} }"
    )
    print(f"withheld from training                       : {withheld}")
    print(f"global ROUTE_THRESHOLD floor                 : {settings.route_threshold}\n")

    def embed(path: Path) -> list[float]:
        with Image.open(path) as im:
            return [float(x) for x in embedder.embed_one(im.convert("RGB"))]

    # ---- raw aggregation output, exactly as Atlas returns it ----------------
    sample = probe_paths("insulator", 1)[0]
    print("=" * 104)
    print(f"$vectorSearch documents for a held-out insulator frame ({sample.name})")
    print("=" * 104)
    for doc in vector_search(embed(sample), k=5):
        print(
            f"  _id={doc['_id']}  {doc['name']:<34} {doc['asset_class']:<19} "
            f"score={doc['score']:.6f}  gate={float(doc.get('routing_threshold') or 0):.4f}"
        )

    # ---- collect every probe's top hit once, then analyse -------------------
    tops: dict[str, list[tuple[str, float, float]]] = {}
    for asset_class in all_classes:
        rows: list[tuple[str, float, float]] = []
        for path in probe_paths(asset_class):
            hits = vector_search(embed(path), k=5)
            if not hits:
                continue
            top = hits[0]
            rows.append(
                (
                    str(top["asset_class"]),
                    float(top["score"]),
                    max(settings.route_threshold, float(top.get("routing_threshold") or 0.0)),
                )
            )
        tops[asset_class] = rows

    print("\n" + "=" * 104)
    print(f"{'PROBE CLASS':<20}{'TOP-1 CORRECT':>15}{'ROUTED':>10}{'REFUSED':>10}{'MEAN TOP':>11}{'MAX':>9}")
    print("=" * 104)
    correct = total = 0
    for asset_class in all_classes:
        rows = tops[asset_class]
        if not rows:
            continue
        scores = np.array([r[1] for r in rows])
        routed = sum(1 for cls, score, gate in rows if score >= gate)
        if asset_class in registered:
            hits = sum(1 for cls, _, _ in rows if cls == asset_class)
            correct += hits
            total += len(rows)
            label = f"{hits}/{len(rows)}"
        else:
            label = "WITHHELD"
        print(
            f"{asset_class:<20}{label:>15}{f'{routed}/{len(rows)}':>10}"
            f"{f'{len(rows) - routed}/{len(rows)}':>10}{scores.mean():>11.4f}{scores.max():>9.4f}"
        )
    print("=" * 104)
    print(f"in-registry top-1 asset-class accuracy: {correct}/{total} ({100 * correct / total:.0f}%)")

    # ---- what the gate percentile buys, measured not assumed ---------------
    in_reg = [r for c in registered for r in tops.get(c, [])]
    out_reg = [r for c in withheld for r in tops.get(c, [])]
    if out_reg:
        print("\n" + "=" * 104)
        print("Decision-rule comparison (accept = frame is scored, refuse = agent declines)")
        print("=" * 104)
        print(f"{'RULE':<34}{'IN-REGISTRY ACCEPTED':>24}{'OUT-OF-REGISTRY REFUSED':>27}")
        for label, fn in [
            ("global 0.82", lambda s, _g: s >= 0.82),
            ("global 0.86", lambda s, _g: s >= 0.86),
            ("global 0.88", lambda s, _g: s >= 0.88),
            ("per-model coverage gate", lambda s, g: s >= g),
        ]:
            acc = sum(1 for _c, s, g in in_reg if fn(s, g))
            ref = sum(1 for _c, s, g in out_reg if not fn(s, g))
            print(f"{label:<34}{f'{acc}/{len(in_reg)}':>24}{f'{ref}/{len(out_reg)}':>27}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

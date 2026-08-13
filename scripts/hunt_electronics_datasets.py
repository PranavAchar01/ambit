"""Programmatic Hugging Face Hub sweep for electronics inspection imagery.

Search-only. Writes nothing to the registry or to data/. Used to source
candidate classes (dev boards, semiconductors, PCB/solder) for Ambit.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

from huggingface_hub import HfApi

QUERIES: tuple[str, ...] = (
    "arduino",
    "raspberry pi",
    "raspberrypi",
    "esp32",
    "microcontroller",
    "single board computer",
    "dev board",
    "development board",
    "breadboard",
    "electronics",
    "electronic components",
    "component detection",
    "semiconductor",
    "wafer",
    "wafer defect",
    "die",
    "silicon",
    "chip defect",
    "ic package",
    "bond wire",
    "solder ball",
    "bga",
    "lead frame",
    "pcb",
    "pcb defect",
    "printed circuit",
    "circuit board",
    "deeppcb",
    "hripcb",
    "pcb dslr",
    "solder",
    "solder joint",
    "smt",
    "smd",
    "electronic defect",
    "industrial anomaly",
    "anomaly detection",
    "visual inspection",
    "surface defect",
    "mvtec",
    "visa",
    "btad",
    "mpdd",
    "aitex",
    "kolektor",
    "magnetic tile",
    "defect detection",
)


def sweep(api: HfApi, queries: Iterable[str], limit: int = 60) -> dict[str, dict[str, object]]:
    hits: dict[str, dict[str, object]] = {}
    for q in queries:
        try:
            for d in api.list_datasets(search=q, limit=limit, full=False):
                row = hits.setdefault(
                    d.id,
                    {
                        "downloads": getattr(d, "downloads", 0) or 0,
                        "likes": getattr(d, "likes", 0) or 0,
                        "tags": list(getattr(d, "tags", []) or []),
                        "queries": [],
                    },
                )
                qs = row["queries"]
                assert isinstance(qs, list)
                qs.append(q)
        except Exception as exc:  # noqa: BLE001 - hub errors are per-query, keep sweeping
            print(f"!! query {q!r} failed: {exc}", file=sys.stderr)
    return hits


def main() -> int:
    api = HfApi()
    hits = sweep(api, QUERIES)
    print(f"unique datasets: {len(hits)}")
    ranked = sorted(hits.items(), key=lambda kv: -int(kv[1]["downloads"]))  # type: ignore[arg-type]
    for did, row in ranked:
        queries = sorted(set(row["queries"]))[:4]
        print(f"{row['downloads']:>8} dl {row['likes']:>4} likes  {did}  <- {queries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

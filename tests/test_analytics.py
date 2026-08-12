"""Trends aggregations, checked against a fixture collection with known counts."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from conftest import requires_atlas
from gridsight.analytics import compute_trends

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def _finding(asset_class: str, verdict: str, severity: float, days_ago: float) -> dict[str, Any]:
    return {
        "timestamp": NOW - timedelta(days=days_ago),
        "asset_class": asset_class,
        "verdict": verdict,
        "severity": severity,
    }


#: A 30-day window splits at day 15. The fixture is built so the two classes move
#: in opposite directions across that split:
#:   insulator  -- 1 defect / 3 nominal in the old half, 3 defects in the new half
#:                 => overall 4/7, recent 3/3, acceleration +0.43
#:   corrosion  -- 3 defects / 1 nominal in the old half, 3 nominal in the new half
#:                 => overall 3/7, recent 0/3, acceleration -0.43
#: One unroutable frame must be excluded from every rate but still counted in the
#: verdict mix.
FIXTURES = [
    *[_finding("insulator", "defect", 0.9, d) for d in (1, 2, 3)],
    _finding("insulator", "defect", 0.9, 20),
    *[_finding("insulator", "nominal", 0.1, d) for d in (21, 22, 23)],
    *[_finding("corrosion", "defect", 0.9, d) for d in (20, 21, 22)],
    _finding("corrosion", "nominal", 0.1, 23),
    *[_finding("corrosion", "nominal", 0.1, d) for d in (1, 2, 3)],
    _finding("rail_surface", "unroutable", 0.0, 5),
]


@pytest.fixture
def fixture_collections() -> Any:
    from gridsight.db.mongo import get_db

    db = get_db()
    suffix = uuid.uuid4().hex[:10]
    findings = db[f"_test_findings_{suffix}"]
    models = db[f"_test_models_{suffix}"]
    findings.insert_many([dict(f) for f in FIXTURES])
    models.insert_many(
        [
            {"name": "insulator-v1", "asset_class": "insulator", "created_by": "seed"},
            {"name": "corrosion-v1", "asset_class": "corrosion", "created_by": "seed"},
            {"name": "rail-v1", "asset_class": "rail_surface", "created_by": "agent-coldstart"},
        ]
    )
    try:
        yield findings, models
    finally:
        findings.drop()
        models.drop()


@requires_atlas
def test_defect_rates_match_the_fixtures(fixture_collections: Any) -> None:
    findings, models = fixture_collections
    result = compute_trends(findings, models, days=30, now=NOW)

    by_class = {r["asset_class"]: r for r in result["by_asset_class"]}
    assert by_class["insulator"]["inspections"] == 7
    assert by_class["insulator"]["defects"] == 4
    assert by_class["insulator"]["defect_rate"] == pytest.approx(0.5714, abs=1e-4)
    assert by_class["insulator"]["recent_defect_rate"] == pytest.approx(1.0)
    assert by_class["corrosion"]["inspections"] == 7
    assert by_class["corrosion"]["defects"] == 3
    assert by_class["corrosion"]["defect_rate"] == pytest.approx(0.4286, abs=1e-4)
    assert by_class["corrosion"]["recent_defect_rate"] == pytest.approx(0.0)

    # unroutable frames are not inspections of a known class
    assert "rail_surface" not in by_class


@requires_atlas
def test_unroutable_still_appears_in_the_verdict_mix(fixture_collections: Any) -> None:
    findings, models = fixture_collections
    counts = {v["verdict"]: v["count"] for v in compute_trends(findings, models, 30, NOW)["verdict_counts"]}
    assert counts == {"defect": 7, "nominal": 7, "unroutable": 1}


@requires_atlas
def test_failing_fastest_ranks_the_accelerating_class_first(fixture_collections: Any) -> None:
    findings, models = fixture_collections
    ranked = compute_trends(findings, models, days=30, now=NOW)["failing_fastest"]
    # insulator's defects are all in the newer half; corrosion's are all in the older half
    assert ranked[0]["asset_class"] == "insulator"
    assert ranked[0]["acceleration"] == pytest.approx(0.4286, abs=1e-4)
    assert ranked[-1]["asset_class"] == "corrosion"
    assert ranked[-1]["acceleration"] == pytest.approx(-0.4286, abs=1e-4)


@requires_atlas
def test_severity_buckets_and_daily_series(fixture_collections: Any) -> None:
    findings, models = fixture_collections
    result = compute_trends(findings, models, days=30, now=NOW)

    buckets = {b["bucket"]: b["count"] for b in result["severity_distribution"]}
    assert buckets.get("0.8-1.0") == 7        # every defect carries severity 0.9
    assert buckets.get("0.0-0.2") == 7        # every nominal carries 0.1
    assert "other" not in buckets             # the 0.0 unroutable is filtered out entirely

    series = result["defect_rate_over_time"]
    assert all(0.0 <= p["defect_rate"] <= 1.0 for p in series)
    assert {p["asset_class"] for p in series} == {"insulator", "corrosion"}


@requires_atlas
def test_registry_coverage_counts_cold_started_models(fixture_collections: Any) -> None:
    findings, models = fixture_collections
    coverage = compute_trends(findings, models, days=30, now=NOW)["registry_coverage"]
    assert coverage["total_models"] == 3
    assert coverage["total_inspections"] == len(FIXTURES)
    origins = {o["created_by"]: o["count"] for o in coverage["by_origin"]}
    assert origins == {"seed": 2, "agent-coldstart": 1}


@requires_atlas
def test_window_excludes_older_findings(fixture_collections: Any) -> None:
    findings, models = fixture_collections
    narrow = compute_trends(findings, models, days=7, now=NOW)
    by_class = {r["asset_class"]: r for r in narrow["by_asset_class"]}
    assert by_class["insulator"]["inspections"] == 3, "only the 3 recent insulator defects"
    assert by_class["insulator"]["defects"] == 3
    assert by_class["corrosion"]["defects"] == 0, "corrosion defects are all 20+ days old"

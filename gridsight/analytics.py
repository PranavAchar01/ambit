"""Fleet-health aggregations over the `findings` collection.

Kept separate from the HTTP layer so the pipelines can be exercised against a
fixture collection rather than whatever happens to be in the live database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pymongo.collection import Collection

SEVERITY_BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
SEVERITY_LABELS = {0.0: "0.0-0.2", 0.2: "0.2-0.4", 0.4: "0.4-0.6", 0.6: "0.6-0.8", 0.8: "0.8-1.0"}


def defect_rate_over_time(
    findings: Collection[dict[str, Any]], window_start: datetime
) -> list[dict[str, Any]]:
    return list(
        findings.aggregate(
            [
                {"$match": {"timestamp": {"$gte": window_start}, "verdict": {"$ne": "unroutable"}}},
                {
                    "$group": {
                        "_id": {
                            "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                            "asset_class": "$asset_class",
                        },
                        "total": {"$sum": 1},
                        "defects": {"$sum": {"$cond": [{"$eq": ["$verdict", "defect"]}, 1, 0]}},
                        "mean_severity": {"$avg": "$severity"},
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "day": "$_id.day",
                        "asset_class": "$_id.asset_class",
                        "total": 1,
                        "defects": 1,
                        "mean_severity": {"$round": [{"$ifNull": ["$mean_severity", 0]}, 4]},
                        "defect_rate": {"$round": [{"$divide": ["$defects", {"$max": ["$total", 1]}]}, 4]},
                    }
                },
                {"$sort": {"day": 1, "asset_class": 1}},
            ]
        )
    )


def by_asset_class(
    findings: Collection[dict[str, Any]], window_start: datetime, midpoint: datetime
) -> list[dict[str, Any]]:
    return list(
        findings.aggregate(
            [
                {"$match": {"timestamp": {"$gte": window_start}, "verdict": {"$ne": "unroutable"}}},
                {
                    "$group": {
                        "_id": "$asset_class",
                        "inspections": {"$sum": 1},
                        "defects": {"$sum": {"$cond": [{"$eq": ["$verdict", "defect"]}, 1, 0]}},
                        "mean_severity": {"$avg": "$severity"},
                        "max_severity": {"$max": "$severity"},
                        "recent_defects": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$and": [
                                            {"$eq": ["$verdict", "defect"]},
                                            {"$gte": ["$timestamp", midpoint]},
                                        ]
                                    },
                                    1,
                                    0,
                                ]
                            }
                        },
                        "recent_total": {"$sum": {"$cond": [{"$gte": ["$timestamp", midpoint]}, 1, 0]}},
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "asset_class": "$_id",
                        "inspections": 1,
                        "defects": 1,
                        "mean_severity": {"$round": [{"$ifNull": ["$mean_severity", 0]}, 4]},
                        "max_severity": {"$round": [{"$ifNull": ["$max_severity", 0]}, 4]},
                        "defect_rate": {
                            "$round": [{"$divide": ["$defects", {"$max": ["$inspections", 1]}]}, 4]
                        },
                        "recent_defect_rate": {
                            "$round": [
                                {"$divide": ["$recent_defects", {"$max": ["$recent_total", 1]}]},
                                4,
                            ]
                        },
                    }
                },
                {"$sort": {"defects": -1}},
            ]
        )
    )


def severity_distribution(
    findings: Collection[dict[str, Any]], window_start: datetime
) -> list[dict[str, Any]]:
    raw = list(
        findings.aggregate(
            [
                {"$match": {"timestamp": {"$gte": window_start}, "severity": {"$gt": 0}}},
                {
                    "$bucket": {
                        "groupBy": "$severity",
                        "boundaries": SEVERITY_BUCKETS,
                        "default": "other",
                        "output": {"count": {"$sum": 1}},
                    }
                },
            ]
        )
    )
    return [{"bucket": SEVERITY_LABELS.get(b["_id"], str(b["_id"])), "count": b["count"]} for b in raw]


def verdict_counts(findings: Collection[dict[str, Any]], window_start: datetime) -> list[dict[str, Any]]:
    return list(
        findings.aggregate(
            [
                {"$match": {"timestamp": {"$gte": window_start}}},
                {"$group": {"_id": "$verdict", "count": {"$sum": 1}}},
                {"$project": {"_id": 0, "verdict": "$_id", "count": 1}},
                {"$sort": {"verdict": 1}},
            ]
        )
    )


def registry_coverage(
    models: Collection[dict[str, Any]], findings: Collection[dict[str, Any]]
) -> dict[str, Any]:
    origins = list(
        models.aggregate(
            [
                {
                    "$group": {
                        "_id": "$created_by",
                        "count": {"$sum": 1},
                        "classes": {"$addToSet": "$asset_class"},
                    }
                },
                {"$project": {"_id": 0, "created_by": "$_id", "count": 1, "classes": 1}},
            ]
        )
    )
    return {
        "total_models": models.count_documents({}),
        "asset_classes": sorted(models.distinct("asset_class")),
        "by_origin": origins,
        "total_inspections": findings.count_documents({}),
    }


def compute_trends(
    findings: Collection[dict[str, Any]],
    models: Collection[dict[str, Any]],
    days: int = 30,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Everything the fleet-health view needs, in one round of aggregations."""
    now = now or datetime.now(UTC)
    window_start = now - timedelta(days=days)
    midpoint = now - timedelta(days=days / 2)

    classes = by_asset_class(findings, window_start, midpoint)
    # "Failing fastest" = how much the defect rate accelerated in the newer half
    # of the window relative to the window as a whole.
    fastest = sorted(
        (
            {**row, "acceleration": round(row["recent_defect_rate"] - row["defect_rate"], 4)}
            for row in classes
        ),
        key=lambda r: (r["acceleration"], r["defect_rate"]),
        reverse=True,
    )

    return {
        "window_days": days,
        "generated_at": now.isoformat(),
        "defect_rate_over_time": defect_rate_over_time(findings, window_start),
        "by_asset_class": classes,
        "failing_fastest": fastest,
        "severity_distribution": severity_distribution(findings, window_start),
        "verdict_counts": verdict_counts(findings, window_start),
        "registry_coverage": registry_coverage(models, findings),
    }

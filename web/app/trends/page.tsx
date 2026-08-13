"use client";

import { useEffect, useState } from "react";
import { DefectRateChart, FailingTable, SeverityHistogram } from "@/components/Charts";
import { fetchTrends } from "@/lib/api";
import type { TrendsResponse } from "@/lib/types";

/** Red means attention. A nominal verdict is deliberately colourless. */
const ATTENTION_VERDICTS = new Set(["defect", "unroutable"]);

export default function TrendsPage() {
  const [data, setData] = useState<TrendsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(30);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    fetchTrends(days)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "failed to load trends");
      });
    return () => {
      cancelled = true;
    };
  }, [days]);

  const coverage = data?.registry_coverage;
  const stats = coverage
    ? [
        { label: "Registered models", value: String(coverage.total_models) },
        { label: "Asset classes covered", value: String(coverage.asset_classes.length) },
        { label: "Total inspections", value: String(coverage.total_inspections) },
        {
          label: "Cold-started models",
          value: String(
            coverage.by_origin.find((o) => o.created_by === "agent-coldstart")?.count ?? 0,
          ),
        },
      ]
    : [];

  return (
    <div className="stack-lg">
      <header className="row-between">
        <div className="stack">
          <h1>Fleet health</h1>
          <p className="lede small">
            Aggregated by MongoDB over every finding Ambit has written.
          </p>
        </div>
        <label className="row">
          <span className="caps">Window</span>
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="select"
          >
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
          </select>
        </label>
      </header>

      {error ? (
        <p className="panel-alert small mono red" role="alert">
          {error}
        </p>
      ) : null}

      {data && data.backfill.count > 0 ? (
        // Volunteered, not buried. Any chart on this page plotted against time
        // is partly plotted against a shifted clock, and a reader who works
        // that out unaided has caught us rather than been told.
        <p className="panel small muted" role="note">
          <span className="caps">Includes backfilled timestamps</span>{" "}
          <span className="mono">
            {data.backfill.count} of {data.backfill.total} findings in this window (
            {(data.backfill.fraction * 100).toFixed(0)}%)
          </span>{" "}
          — {data.backfill.note}
        </p>
      ) : null}

      {stats.length > 0 ? (
        <div className="grid-4">
          {stats.map((s) => (
            <div key={s.label} className="stat">
              <div className="caps">{s.label}</div>
              <div className="stat-value">{s.value}</div>
            </div>
          ))}
        </div>
      ) : null}

      <section className="panel">
        <div className="panel-hd">
          <h2>Defect rate over time, by asset class</h2>
        </div>
        <DefectRateChart points={data?.defect_rate_over_time ?? []} />
      </section>

      <div className="grid-2">
        <section className="panel">
          <div className="panel-hd">
            <h2>Severity distribution</h2>
          </div>
          <SeverityHistogram data={data?.severity_distribution ?? []} />
        </section>
        <section className="panel">
          <div className="panel-hd">
            <h2>Verdict mix</h2>
            <p className="muted tiny" style={{ marginTop: "4px" }}>
              Includes frames the agent refused to score.
            </p>
          </div>
          <ul className="stack" style={{ gap: "8px" }}>
            {(data?.verdict_counts ?? []).map((v) => {
              const attention = ATTENTION_VERDICTS.has(v.verdict);
              return (
                <li
                  key={v.verdict}
                  className="row-between"
                  style={{ alignItems: "center", color: attention ? "var(--red)" : undefined }}
                >
                  <span className="mono small">{v.verdict}</span>
                  <span className="mono small" style={{ fontWeight: 700 }}>
                    {v.count}
                  </span>
                </li>
              );
            })}
            {(data?.verdict_counts ?? []).length === 0 ? (
              <li className="muted small">No findings yet.</li>
            ) : null}
          </ul>
        </section>
      </div>

      <section className="panel">
        <div className="panel-hd">
          <h2>Which part is failing fastest</h2>
          <p className="muted tiny" style={{ marginTop: "4px" }}>
            Ranked by how much each class&apos;s defect rate rose in the newer half of the window.
          </p>
        </div>
        <FailingTable rows={data?.failing_fastest ?? []} />
      </section>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { DefectRateChart, FailingTable, SeverityHistogram } from "@/components/Charts";
import { fetchTrends } from "@/lib/api";
import type { TrendsResponse } from "@/lib/types";

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
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Fleet health</h1>
          <p className="muted mt-1 text-sm">
            Aggregated by MongoDB over every finding GridSight has written.
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <span className="muted">Window</span>
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="rounded-md px-2 py-1.5 text-sm"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text)",
            }}
          >
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
          </select>
        </label>
      </header>

      {error ? (
        <p className="surface rounded-lg p-4 text-sm" style={{ color: "var(--danger)" }} role="alert">
          {error}
        </p>
      ) : null}

      {stats.length > 0 ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((s) => (
            <div key={s.label} className="surface rounded-lg p-3">
              <div className="muted text-xs uppercase tracking-wide">{s.label}</div>
              <div className="mono mt-1 text-2xl font-semibold">{s.value}</div>
            </div>
          ))}
        </div>
      ) : null}

      <section className="surface rounded-lg p-4">
        <h2 className="mb-3 text-sm font-semibold">Defect rate over time, by asset class</h2>
        <DefectRateChart points={data?.defect_rate_over_time ?? []} />
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <section className="surface rounded-lg p-4">
          <h2 className="mb-3 text-sm font-semibold">Severity distribution</h2>
          <SeverityHistogram data={data?.severity_distribution ?? []} />
        </section>
        <section className="surface rounded-lg p-4">
          <h2 className="mb-1 text-sm font-semibold">Verdict mix</h2>
          <p className="muted mb-3 text-xs">Includes frames the agent refused to score.</p>
          <ul className="m-0 flex list-none flex-col gap-2 p-0 text-sm">
            {(data?.verdict_counts ?? []).map((v) => (
              <li key={v.verdict} className="flex items-center justify-between">
                <span className="mono">{v.verdict}</span>
                <span className="mono font-semibold">{v.count}</span>
              </li>
            ))}
            {(data?.verdict_counts ?? []).length === 0 ? (
              <li className="muted">No findings yet.</li>
            ) : null}
          </ul>
        </section>
      </div>

      <section className="surface rounded-lg p-4">
        <h2 className="mb-1 text-sm font-semibold">Which part is failing fastest</h2>
        <p className="muted mb-3 text-xs">
          Ranked by how much each class&apos;s defect rate rose in the newer half of the window.
        </p>
        <FailingTable rows={data?.failing_fastest ?? []} />
      </section>
    </div>
  );
}

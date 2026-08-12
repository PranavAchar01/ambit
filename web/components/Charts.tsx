"use client";

import type { ClassTrend, TrendPoint } from "@/lib/types";

/**
 * Charts are hand-rolled SVG driven entirely by CSS custom properties, so they
 * re-colour with the theme instead of baking in a palette that only reads in one.
 */

const SERIES_COLOURS = ["#4f8cff", "#f59e0b", "#22c55e", "#e879f9", "#f87171", "#2dd4bf"];

export function seriesColour(index: number): string {
  return SERIES_COLOURS[index % SERIES_COLOURS.length] ?? "#4f8cff";
}

function EmptyState({ label }: { label: string }) {
  return (
    <div
      className="flex min-h-40 items-center justify-center rounded-md p-6 text-center text-sm"
      style={{ background: "var(--surface-2)", color: "var(--text-muted)" }}
    >
      {label}
    </div>
  );
}

export function DefectRateChart({ points }: { points: TrendPoint[] }) {
  if (points.length === 0) {
    return <EmptyState label="No inspections in this window yet." />;
  }

  const days = Array.from(new Set(points.map((p) => p.day))).sort();
  const classes = Array.from(new Set(points.map((p) => p.asset_class))).sort();

  const W = 720;
  const H = 260;
  const pad = { top: 16, right: 16, bottom: 34, left: 42 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const x = (day: string) =>
    days.length <= 1
      ? pad.left + plotW / 2
      : pad.left + (days.indexOf(day) / (days.length - 1)) * plotW;
  const y = (rate: number) => pad.top + (1 - rate) * plotH;

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full min-w-[34rem]"
        role="img"
        aria-label="Defect rate over time by asset class"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={pad.left}
              x2={W - pad.right}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--grid)"
              strokeWidth={1}
            />
            <text x={pad.left - 8} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {days.map((d, i) =>
          i % Math.max(1, Math.ceil(days.length / 6)) === 0 ? (
            <text
              key={d}
              x={x(d)}
              y={H - 12}
              textAnchor="middle"
              fontSize={11}
              fill="var(--text-muted)"
            >
              {d.slice(5)}
            </text>
          ) : null,
        )}
        {classes.map((cls, ci) => {
          const series = days
            .map((d) => points.find((p) => p.day === d && p.asset_class === cls))
            .filter((p): p is TrendPoint => Boolean(p));
          if (series.length === 0) return null;
          const path = series
            .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.day).toFixed(1)} ${y(p.defect_rate).toFixed(1)}`)
            .join(" ");
          return (
            <g key={cls}>
              <path d={path} fill="none" stroke={seriesColour(ci)} strokeWidth={2.5} />
              {series.map((p) => (
                <circle
                  key={`${cls}-${p.day}`}
                  cx={x(p.day)}
                  cy={y(p.defect_rate)}
                  r={3.5}
                  fill={seriesColour(ci)}
                >
                  <title>{`${cls} · ${p.day} · ${(p.defect_rate * 100).toFixed(0)}% of ${p.total}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <ul className="m-0 mt-2 flex list-none flex-wrap gap-x-4 gap-y-1 p-0 text-xs">
        {classes.map((cls, ci) => (
          <li key={cls} className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ background: seriesColour(ci) }}
            />
            {cls}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SeverityHistogram({ data }: { data: { bucket: string; count: number }[] }) {
  if (data.length === 0) return <EmptyState label="No severity samples yet." />;
  const max = Math.max(...data.map((d) => d.count), 1);
  const W = 460;
  const H = 220;
  const pad = { top: 14, right: 12, bottom: 34, left: 36 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const barW = plotW / data.length - 10;

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full min-w-[22rem]"
        role="img"
        aria-label="Distribution of defect severity"
      >
        <line
          x1={pad.left}
          x2={W - pad.right}
          y1={pad.top + plotH}
          y2={pad.top + plotH}
          stroke="var(--grid)"
        />
        {data.map((d, i) => {
          const h = (d.count / max) * plotH;
          const bx = pad.left + i * (plotW / data.length) + 5;
          return (
            <g key={d.bucket}>
              <rect
                x={bx}
                y={pad.top + plotH - h}
                width={barW}
                height={Math.max(h, 1)}
                rx={3}
                fill={seriesColour(i)}
              >
                <title>{`${d.bucket}: ${d.count}`}</title>
              </rect>
              <text
                x={bx + barW / 2}
                y={pad.top + plotH - h - 5}
                textAnchor="middle"
                fontSize={11}
                fill="var(--text)"
              >
                {d.count}
              </text>
              <text
                x={bx + barW / 2}
                y={H - 12}
                textAnchor="middle"
                fontSize={10}
                fill="var(--text-muted)"
              >
                {d.bucket}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function FailingTable({ rows }: { rows: ClassTrend[] }) {
  if (rows.length === 0) return <EmptyState label="No findings recorded yet." />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[34rem] border-collapse text-sm">
        <caption className="sr-only">Components ranked by how fast their defect rate is rising</caption>
        <thead>
          <tr className="muted text-left text-xs uppercase tracking-wide">
            <th scope="col" className="py-1.5 pr-3 font-medium">Asset class</th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">Inspections</th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">Defects</th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">Defect rate</th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">Recent</th>
            <th scope="col" className="py-1.5 text-right font-medium">Trend</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const accel = r.acceleration ?? 0;
            const colour =
              accel > 0.01 ? "var(--danger)" : accel < -0.01 ? "var(--ok)" : "var(--text-muted)";
            return (
              <tr key={r.asset_class} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="mono py-1.5 pr-3">{r.asset_class}</td>
                <td className="mono py-1.5 pr-3 text-right">{r.inspections}</td>
                <td className="mono py-1.5 pr-3 text-right">{r.defects}</td>
                <td className="mono py-1.5 pr-3 text-right">{(r.defect_rate * 100).toFixed(0)}%</td>
                <td className="mono py-1.5 pr-3 text-right">
                  {(r.recent_defect_rate * 100).toFixed(0)}%
                </td>
                <td className="mono py-1.5 text-right" style={{ color: colour }}>
                  {accel > 0 ? "▲" : accel < 0 ? "▼" : "—"} {(Math.abs(accel) * 100).toFixed(0)}pp
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

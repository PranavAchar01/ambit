"use client";

import type { ClassTrend, TrendPoint } from "@/lib/types";

/**
 * Charts are hand-rolled SVG in the two colours the system allows: black and
 * red, with white and grey standing in as the absence of red. Series are
 * separated by luminance rather than by hue, so nothing here reads as a
 * decorative palette — red only ever marks something that wants attention.
 */

const SERIES_COLOURS = ["#ff2d2d", "#ffffff", "#c11414", "#9a9a9a", "#ff8a8a", "#6e6e6e"];

export function seriesColour(index: number): string {
  return SERIES_COLOURS[index % SERIES_COLOURS.length] ?? "#ff2d2d";
}

/** Severity reads as intensity: grey at the low end, red at the high end. */
const SEVERITY_RAMP = ["#5f5f5f", "#9a9a9a", "#c11414", "#ff2d2d"];

function severityColour(index: number, total: number): string {
  const t = total <= 1 ? 1 : index / (total - 1);
  const step = Math.round(t * (SEVERITY_RAMP.length - 1));
  return SEVERITY_RAMP[step] ?? "#ff2d2d";
}

function EmptyState({ label }: { label: string }) {
  return <div className="empty">{label}</div>;
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
    <div className="table-wrap">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", minWidth: "34rem", height: "auto" }}
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
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text
              x={pad.left - 8}
              y={y(t) + 4}
              textAnchor="end"
              fontSize={11}
              fill="var(--fg-faint)"
              fontFamily="var(--mono)"
            >
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
              fill="var(--fg-faint)"
              fontFamily="var(--mono)"
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
      <ul className="row tiny" style={{ marginTop: "var(--step)" }}>
        {classes.map((cls, ci) => (
          <li key={cls} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span
              aria-hidden
              style={{
                display: "inline-block",
                width: "10px",
                height: "10px",
                background: seriesColour(ci),
              }}
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
    <div className="table-wrap">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", minWidth: "22rem", height: "auto" }}
        role="img"
        aria-label="Distribution of defect severity"
      >
        <line
          x1={pad.left}
          x2={W - pad.right}
          y1={pad.top + plotH}
          y2={pad.top + plotH}
          stroke="var(--line-bright)"
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
                fill={severityColour(i, data.length)}
              >
                <title>{`${d.bucket}: ${d.count}`}</title>
              </rect>
              <text
                x={bx + barW / 2}
                y={pad.top + plotH - h - 5}
                textAnchor="middle"
                fontSize={11}
                fill="var(--fg)"
                fontFamily="var(--mono)"
              >
                {d.count}
              </text>
              <text
                x={bx + barW / 2}
                y={H - 12}
                textAnchor="middle"
                fontSize={10}
                fill="var(--fg-faint)"
                fontFamily="var(--mono)"
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
    <div className="table-wrap">
      <table style={{ minWidth: "34rem" }}>
        <caption className="sr-only">Components ranked by how fast their defect rate is rising</caption>
        <thead>
          <tr>
            <th scope="col">Asset class</th>
            <th scope="col" className="num">Inspections</th>
            <th scope="col" className="num">Defects</th>
            <th scope="col" className="num">Defect rate</th>
            <th scope="col" className="num">Recent</th>
            <th scope="col" className="num">Trend</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const accel = r.acceleration ?? 0;
            // A rising defect rate is the only thing here worth alarming about.
            const colour =
              accel > 0.01 ? "var(--red)" : accel < -0.01 ? "var(--fg-dim)" : "var(--fg-faint)";
            return (
              <tr key={r.asset_class}>
                <td className="mono">{r.asset_class}</td>
                <td className="mono num">{r.inspections}</td>
                <td className="mono num">{r.defects}</td>
                <td className="mono num">{(r.defect_rate * 100).toFixed(0)}%</td>
                <td className="mono num">{(r.recent_defect_rate * 100).toFixed(0)}%</td>
                <td className="mono num" style={{ color: colour }}>
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

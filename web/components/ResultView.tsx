"use client";

import { imageUrl } from "@/lib/api";
import type { InspectResult } from "@/lib/types";
import { OverlayCanvas } from "./OverlayCanvas";

function VerdictBadge({ verdict }: { verdict: InspectResult["verdict"] }) {
  const style: Record<InspectResult["verdict"], { bg: string; label: string }> = {
    defect: { bg: "var(--danger)", label: "Defect" },
    nominal: { bg: "var(--ok)", label: "Nominal" },
    unroutable: { bg: "var(--warn)", label: "Unroutable" },
  };
  const s = style[verdict];
  return (
    <span
      className="rounded-full px-2.5 py-1 text-xs font-semibold"
      style={{ background: s.bg, color: "#0b0f14" }}
    >
      {s.label}
    </span>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="surface rounded-lg p-3">
      <div className="muted text-xs uppercase tracking-wide">{label}</div>
      <div className="mono mt-1 text-lg font-semibold">{value}</div>
      {hint ? <div className="muted mt-0.5 text-xs">{hint}</div> : null}
    </div>
  );
}

export function ResultView({ result }: { result: InspectResult }) {
  const frame = imageUrl(result.uploaded_image_url);
  const heat = imageUrl(result.heatmap_url);
  const golden = imageUrl(result.reference_image_url);
  const [winner, ...runnersUp] = result.candidates;

  return (
    <section className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        <VerdictBadge verdict={result.verdict} />
        {result.asset_class ? (
          <span className="mono text-sm">{result.asset_class}</span>
        ) : null}
        {result.latency_ms !== null ? (
          <span className="muted text-xs">{result.latency_ms} ms end-to-end</span>
        ) : null}
        {result.cold_start ? (
          <span
            className="rounded-full px-2.5 py-1 text-xs font-semibold"
            style={{ background: "var(--accent)", color: "#0b0f14" }}
          >
            New model minted: {result.cold_start.name}
          </span>
        ) : null}
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Inspected frame</h3>
          {frame ? (
            <OverlayCanvas imageUrl={frame} heatmapUrl={heat} regions={result.bbox_regions} />
          ) : null}
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold">
            Registry reference{result.asset_class ? ` — known-good ${result.asset_class}` : ""}
          </h3>
          {golden ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={golden}
              alt={`Known-good reference exemplar for ${result.asset_class ?? "this asset class"}`}
              className="w-full rounded-lg"
              style={{ border: "1px solid var(--border)" }}
            />
          ) : (
            <div
              className="surface flex min-h-40 items-center justify-center rounded-lg p-6 text-center text-sm"
              style={{ color: "var(--text-muted)" }}
            >
              No golden reference — the registry has no model for this asset class yet.
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Routing score"
          value={result.routing_score.toFixed(4)}
          hint={winner ? `gate ${winner.gate.toFixed(4)}` : undefined}
        />
        <Stat
          label="Anomaly score"
          value={result.raw_anomaly_score !== null ? result.raw_anomaly_score.toFixed(3) : "—"}
          hint={
            result.image_threshold !== null
              ? `threshold ${result.image_threshold.toFixed(3)}`
              : undefined
          }
        />
        <Stat label="Severity" value={result.severity.toFixed(2)} hint="0–1 normalised" />
        <Stat label="Regions" value={String(result.bbox_regions.length)} hint="connected components" />
      </div>

      <div className="surface rounded-lg p-4">
        <h3 className="text-sm font-semibold">Agent narrative</h3>
        <p className="mt-2 text-sm leading-relaxed">{result.narrative}</p>
        <p className="muted mt-2 text-xs">
          Written by {result.narrative_source === "openai" ? "OpenAI" : "deterministic fallback"}
          {result.finding_id ? ` · finding ${result.finding_id}` : ""}
        </p>
      </div>

      <div className="surface rounded-lg p-4">
        <h3 className="text-sm font-semibold">Routing decision</h3>
        <p className="muted mt-1 text-sm">{result.decision_reason}</p>
        <p className="muted mt-1 text-xs">
          Decided by {result.decision_source === "openai" ? "OpenAI adjudication" : result.decision_source}
        </p>
        {result.candidates.length > 0 ? (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[30rem] border-collapse text-sm">
              <caption className="sr-only">Vector search candidates and their scores</caption>
              <thead>
                <tr className="muted text-left text-xs uppercase tracking-wide">
                  <th scope="col" className="py-1.5 pr-3 font-medium">Model</th>
                  <th scope="col" className="py-1.5 pr-3 font-medium">Asset class</th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">Score</th>
                  <th scope="col" className="py-1.5 text-right font-medium">Gate</th>
                </tr>
              </thead>
              <tbody>
                {result.candidates.map((c, i) => (
                  <tr key={c.model_id} style={{ borderTop: "1px solid var(--border)" }}>
                    <td className="mono py-1.5 pr-3">
                      {c.name}
                      {i === 0 ? (
                        <span className="ml-2 text-xs font-semibold" style={{ color: "var(--accent)" }}>
                          winner
                        </span>
                      ) : null}
                    </td>
                    <td className="py-1.5 pr-3">{c.asset_class}</td>
                    <td className="mono py-1.5 pr-3 text-right">{c.score.toFixed(4)}</td>
                    <td className="mono py-1.5 text-right">{c.gate.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {runnersUp.length > 0 ? (
              <p className="muted mt-2 text-xs">
                Runners-up: {runnersUp.map((c) => `${c.name} (${c.score.toFixed(3)})`).join(", ")}
              </p>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

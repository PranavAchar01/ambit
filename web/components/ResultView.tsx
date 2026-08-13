"use client";

import { imageUrl } from "@/lib/api";
import type { InspectResult } from "@/lib/types";
import { OverlayCanvas } from "./OverlayCanvas";

/** Red is attention. A nominal verdict is deliberately colourless. */
const VERDICT: Record<InspectResult["verdict"], { className: string; label: string }> = {
  defect: { className: "badge badge-defect", label: "Defect" },
  nominal: { className: "badge badge-nominal", label: "Nominal" },
  unroutable: { className: "badge badge-unroutable", label: "Unroutable" },
};

function VerdictBadge({ verdict }: { verdict: InspectResult["verdict"] }) {
  const v = VERDICT[verdict];
  return <span className={v.className}>{v.label}</span>;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat">
      <div className="caps">{label}</div>
      <div className="stat-value">{value}</div>
      {hint ? <div className="stat-hint">{hint}</div> : null}
    </div>
  );
}

export function ResultView({ result }: { result: InspectResult }) {
  const frame = imageUrl(result.uploaded_image_url);
  const heat = imageUrl(result.heatmap_url);
  const golden = imageUrl(result.reference_image_url);
  const [winner, ...runnersUp] = result.candidates;

  return (
    <section className="stack-lg">
      <div className="row">
        <VerdictBadge verdict={result.verdict} />
        {result.asset_class ? <span className="mono small">{result.asset_class}</span> : null}
        {result.latency_ms !== null ? (
          <span className="mono tiny muted">{result.latency_ms} ms end-to-end</span>
        ) : null}
        {result.cold_start ? (
          <span className="badge badge-new">New model minted: {result.cold_start.name}</span>
        ) : null}
      </div>

      <div className="grid-2">
        <div className="stack">
          <h3 className="caps">Inspected frame</h3>
          {frame ? (
            <OverlayCanvas imageUrl={frame} heatmapUrl={heat} regions={result.bbox_regions} />
          ) : null}
        </div>
        <div className="stack">
          <h3 className="caps">
            Registry reference{result.asset_class ? ` — known-good ${result.asset_class}` : ""}
          </h3>
          {golden ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={golden}
              alt={`Known-good reference exemplar for ${result.asset_class ?? "this asset class"}`}
              style={{ width: "100%", border: "1px solid var(--line)" }}
            />
          ) : (
            <div className="empty" style={{ display: "grid", placeItems: "center", minHeight: "10rem" }}>
              No golden reference — the registry has no model for this asset class yet.
            </div>
          )}
        </div>
      </div>

      <div className="grid-4">
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

      <div className="panel">
        <h3 className="panel-hd">Agent narrative</h3>
        <div className="stack">
          <p>{result.narrative}</p>
          <p className="mono tiny muted">
            Written by {result.narrative_source === "openai" ? "OpenAI" : "deterministic fallback"}
            {result.finding_id ? ` · finding ${result.finding_id}` : ""}
          </p>
        </div>
      </div>

      <div className="panel">
        <h3 className="panel-hd">Routing decision</h3>
        <div className="stack">
          <p className="muted small">{result.decision_reason}</p>
          <p className="mono tiny muted">
            Decided by{" "}
            {result.decision_source === "openai" ? "OpenAI adjudication" : result.decision_source}
          </p>
          {result.candidates.length > 0 ? (
            <div className="stack">
              <div className="table-wrap">
                <table>
                  <caption className="sr-only">Vector search candidates and their scores</caption>
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col">Asset class</th>
                      <th scope="col" className="num">
                        Score
                      </th>
                      <th scope="col" className="num">
                        Gate
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.candidates.map((c, i) => (
                      <tr key={c.model_id} data-winner={i === 0 ? "true" : undefined}>
                        <td className="mono">{c.name}</td>
                        <td>{c.asset_class}</td>
                        <td className="num">{c.score.toFixed(4)}</td>
                        <td className="num">{c.gate.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {runnersUp.length > 0 ? (
                <p className="mono tiny muted">
                  Runners-up: {runnersUp.map((c) => `${c.name} (${c.score.toFixed(3)})`).join(", ")}
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

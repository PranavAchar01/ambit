"use client";

import { imageUrl } from "@/lib/api";
import type { InspectResult } from "@/lib/types";
import { OverlayCanvas } from "./OverlayCanvas";

/**
 * Say which provider actually wrote the sentence.
 *
 * This was a hardcoded `=== "openai" ? "OpenAI" : "deterministic fallback"`,
 * which renders every other provider -- including a real vision model -- as a
 * canned template. The value is persisted on the finding forever, so the label
 * has to track it rather than assume a two-provider world.
 */
function narrativeAttribution(source: string | null): string {
  if (!source || source === "none") return "No narrative was generated";
  if (source === "structured" || source === "fallback") return "Written from the numbers (deterministic)";
  if (source === "openai") return "Written by OpenAI";
  if (source.startsWith("vlm:")) return `Described by ${source.slice(4)} (vision model, from the pixels)`;
  return `Written by ${source}`;
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

      <div className="panel panel-narrative">
        <h3 className="panel-hd">What the frame shows</h3>
        <div className="stack">
          <p className="lede">{result.narrative}</p>
          <p className="mono tiny muted">
            {narrativeAttribution(result.narrative_source)}
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
            <details className="stack">
              <summary>
                Every candidate and the gate it had to clear ({result.candidates.length})
              </summary>
              <div className="table-wrap" style={{ marginTop: "calc(var(--step) * 2)" }}>
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
            </details>
          ) : null}
        </div>
      </div>
    </section>
  );
}

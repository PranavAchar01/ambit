"use client";

import type { InspectResult } from "@/lib/types";

/** Red is attention. A nominal verdict is deliberately colourless. */
const VERDICT: Record<InspectResult["verdict"], { tone: string; label: string; gloss: string }> = {
  defect: { tone: "defect", label: "Defect", gloss: "scored above this specialist's threshold" },
  nominal: { tone: "nominal", label: "Nominal", gloss: "inside the normal envelope this model learned" },
  unroutable: {
    tone: "unroutable",
    label: "Unroutable",
    gloss: "no specialist has competence over this frame",
  },
};

/**
 * The verdict, and the comparison that produced it.
 *
 * Scores are rendered as a comparison rather than as a bare number on purpose:
 * `0.9854 >= 0.9806` is the claim the system is actually making, and a lone
 * `0.9854` invites the reader to supply their own threshold.
 */
export function VerdictCard({ result }: { result: InspectResult }) {
  const v = VERDICT[result.verdict];
  const top = result.candidates[0];
  const routed = result.routed_model.name;
  const gate = top?.gate ?? null;
  const score = result.routing_score;
  const raw = result.raw_anomaly_score;
  const threshold = result.image_threshold;

  return (
    <div className="verdict-card" data-verdict={v.tone}>
      <div className="row-between">
        <div className="row">
          <span className={`badge badge-${v.tone}`}>{v.label}</span>
          {result.asset_class ? <span className="mono small">{result.asset_class}</span> : null}
        </div>
        {result.latency_ms !== null ? (
          <span className="mono tiny faint">{result.latency_ms} ms end-to-end</span>
        ) : null}
      </div>

      <p className="small muted" style={{ margin: 0 }}>
        {v.gloss}
      </p>

      <dl className="verdict-facts">
        <div>
          <dt className="caps">Routed to</dt>
          <dd className="mono small">{routed ?? "— nothing cleared a gate —"}</dd>
        </div>
        <div>
          <dt className="caps">Routing vs gate</dt>
          <dd className="mono small">
            {gate === null ? (
              score.toFixed(4)
            ) : (
              <>
                {score.toFixed(4)}{" "}
                <span className={score >= gate ? undefined : "red"}>{score >= gate ? "≥" : "<"}</span>{" "}
                {gate.toFixed(4)}
              </>
            )}
          </dd>
        </div>
        {raw !== null && threshold !== null ? (
          <div>
            <dt className="caps">Anomaly vs threshold</dt>
            <dd className="mono small">
              {raw.toFixed(3)}{" "}
              <span className={raw >= threshold ? "red" : undefined}>{raw >= threshold ? "≥" : "<"}</span>{" "}
              {threshold.toFixed(3)}
            </dd>
          </div>
        ) : null}
      </dl>

      {result.cold_start ? (
        <p className="small" style={{ margin: 0 }}>
          <span className="badge badge-new">New specialist minted</span>{" "}
          <span className="mono tiny muted">
            {result.cold_start.name} from {result.cold_start.reference_images} references in{" "}
            {result.cold_start.seconds}s
          </span>
        </p>
      ) : null}
    </div>
  );
}

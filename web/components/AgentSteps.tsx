"use client";

import type { AgentStep } from "@/lib/types";

const NODE_LABELS: Record<string, string> = {
  embed_frame: "Embedding frame",
  route: "Searching registry",
  decide: "Deciding route",
  cold_start: "Cold-starting a specialist",
  infer: "Running inference",
  narrate: "Writing the finding",
  persist: "Saving finding",
};

const ORDER = ["embed_frame", "route", "decide", "cold_start", "infer", "narrate", "persist"];

export function AgentSteps({ steps, running }: { steps: AgentStep[]; running: boolean }) {
  const seen = new Set(steps.map((s) => s.node));
  const pending = running
    ? ORDER.filter((n) => !seen.has(n) && n !== "cold_start").slice(0, 1)
    : [];

  return (
    <ol className="steps" aria-live="polite" aria-busy={running}>
      {steps.map((step, i) => (
        <li key={`${step.node}-${i}`} className="step">
          <span aria-hidden className="step-mark" />
          <span style={{ minWidth: 0 }}>
            <span>{NODE_LABELS[step.node] ?? step.node}</span>
            <span className="muted"> — {step.message}</span>
          </span>
        </li>
      ))}
      {pending.map((node) => (
        <li key={node} className="step">
          <span aria-hidden className="step-mark" data-live="true" />
          <span className="muted">{NODE_LABELS[node] ?? node}…</span>
        </li>
      ))}
    </ol>
  );
}

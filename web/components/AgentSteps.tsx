"use client";

import type { AgentStep, Candidate } from "@/lib/types";

const NODE_LABELS: Record<string, string> = {
  embed_frame: "Embedded frame",
  route: "Searched registry",
  decide: "Routed",
  cold_start: "Cold-started a specialist",
  infer: "Ran inference",
  narrate: "Described the frame",
  persist: "Persisted",
};

const PENDING_LABELS: Record<string, string> = {
  embed_frame: "Embedding frame",
  route: "Searching registry",
  decide: "Deciding route",
  cold_start: "Cold-starting a specialist",
  infer: "Running inference",
  narrate: "Describing the frame",
  persist: "Persisting",
};

const ORDER = ["embed_frame", "route", "decide", "cold_start", "infer", "narrate", "persist"];

/** ok = it did the thing; warn = it hesitated; stop = it declined. */
type Mark = "ok" | "warn" | "stop";

const MARK_GLYPH: Record<Mark, string> = { ok: "✓", warn: "⚠", stop: "✕" };

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function candidatesOf(step: AgentStep): Candidate[] {
  return Array.isArray(step.candidates) ? (step.candidates as Candidate[]) : [];
}

function markFor(step: AgentStep): Mark {
  if (step.node === "decide") {
    if (step.decision === "unroutable") return str(step.source) ? "warn" : "stop";
    return "ok";
  }
  if (step.node === "infer" && step.error === true) return "stop";
  if (step.node === "infer" && step.verdict === "defect") return "warn";
  return "ok";
}

/**
 * The specific thing this node did, preferring the structured values the server
 * already puts on the wire over its own prose. Every branch falls back to
 * `step.message`, so an unrecognised node or a missing extra degrades to the
 * server's sentence rather than to a blank cell.
 */
function detailFor(step: AgentStep, all: AgentStep[]): string {
  const message = step.message;
  switch (step.node) {
    case "embed_frame": {
      const dim = num(step.dim);
      return dim ? `OpenCLIP ViT-B-32 → ${dim}-d` : message;
    }
    case "route": {
      const k = candidatesOf(step).length;
      return k ? `$vectorSearch, k=${k}, model_router_idx` : message;
    }
    case "decide": {
      // The gate lives on the route step's candidates, not on this one.
      const routeStep = all.find((s) => s.node === "route");
      const top = routeStep ? candidatesOf(routeStep)[0] : undefined;
      const score = num(routeStep?.routing_score) ?? top?.score ?? null;
      const gate = num(step.gate) ?? top?.gate ?? null;
      const source = str(step.source);
      if (step.decision === "route" && top && score !== null && gate !== null) {
        const via = source && source !== "threshold" ? ` · ${source} adjudicated` : "";
        return `${top.name} · ${score.toFixed(4)} ≥ ${gate.toFixed(4)}${via}`;
      }
      if (step.decision === "unroutable" && score !== null && gate !== null) {
        const how = source && source !== "threshold" ? `${source} declined` : "below gate";
        return `${score.toFixed(4)} vs gate ${gate.toFixed(4)} — ${how}`;
      }
      return message;
    }
    case "cold_start": {
      const name = str(step.name);
      const backbone = str(step.backbone);
      const seconds = num(step.seconds);
      if (name && seconds !== null) {
        return `${name} · ${backbone ?? "patchcore"} · fitted in ${seconds.toFixed(1)}s`;
      }
      return message;
    }
    case "infer": {
      const raw = num(step.raw_score);
      const threshold = num(step.threshold);
      const regions = num(step.regions);
      if (raw !== null && threshold !== null) {
        const cmp = raw >= threshold ? "≥" : "<";
        const rgn = regions !== null && regions > 0 ? ` · ${regions} region(s)` : "";
        return `PatchCore · ${raw.toFixed(3)} ${cmp} threshold ${threshold.toFixed(3)}${rgn}`;
      }
      return message;
    }
    case "narrate":
      return str(step.source) ?? message;
    case "persist": {
      const latency = num(step.latency_ms);
      return latency !== null ? `findings · ${latency} ms end-to-end` : message;
    }
    default:
      return message;
  }
}

/**
 * `at` is a float epoch in SECONDS (`time.time()` server-side), so the gap to the
 * previous step is that step's wall-clock cost. The first row has nothing to
 * subtract from and shows a dash rather than a fabricated zero.
 */
function elapsedMs(step: AgentStep, previous: AgentStep | undefined): number | null {
  const at = num(step.at);
  const before = previous ? num(previous.at) : null;
  if (at === null || before === null) return null;
  const ms = Math.round((at - before) * 1000);
  return ms >= 0 ? ms : null;
}

export function AgentSteps({ steps, running }: { steps: AgentStep[]; running: boolean }) {
  const seen = new Set(steps.map((s) => s.node));
  const pending = running ? ORDER.filter((n) => !seen.has(n) && n !== "cold_start").slice(0, 1) : [];

  return (
    <ol className="steps" aria-live="polite" aria-busy={running}>
      {steps.map((step, i) => {
        const mark = markFor(step);
        const ms = elapsedMs(step, steps[i - 1]);
        return (
          <li key={`${step.node}-${i}`} className="step" data-mark={mark}>
            <span aria-hidden className="step-mark" data-mark={mark}>
              {MARK_GLYPH[mark]}
            </span>
            <span className="step-name">{NODE_LABELS[step.node] ?? step.node}</span>
            <span className="step-detail muted" title={step.message}>
              {detailFor(step, steps)}
            </span>
            <span className="step-time mono tiny faint">{ms === null ? "—" : `${ms} ms`}</span>
          </li>
        );
      })}
      {pending.map((node) => (
        <li key={node} className="step" data-mark="pending">
          <span aria-hidden className="step-mark" data-live="true" />
          <span className="step-name muted">{PENDING_LABELS[node] ?? node}…</span>
          <span className="step-detail" />
          <span className="step-time" />
        </li>
      ))}
    </ol>
  );
}

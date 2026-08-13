"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { AgentStep, Candidate, LayoutModel, RegistryLayout } from "@/lib/types";

/**
 * A live map of the routing decision.
 *
 * Geometry is honest: radial distance from the centre is `1 - score`, the real
 * measured similarity between the frame and each model's training centroid, and
 * each spoke carries a gate tick at `1 - routing_threshold`. A model's dot
 * landing INSIDE its own gate tick is precisely what "this frame is within the
 * competence that model demonstrably learned" means — so a refusal is visible as
 * every dot sitting beyond every tick, not merely asserted in text.
 *
 * Only the ANGLE is illustrative: it comes from ranking models around a PCA of
 * their centroids and spacing them evenly, which encodes neighbourhood ordering
 * rather than distance. The caption says so.
 */

const WIDTH = 660;
const HEIGHT = 470;
const CX = WIDTH / 2;
const CY = HEIGHT / 2;
const R_MAX = 152;

/** Scores below this clamp to the rim; nothing real scores lower. */
const SCORE_FLOOR = 0.7;

function radius(score: number): number {
  const t = (1 - score) / (1 - SCORE_FLOOR);
  return Math.max(10, Math.min(R_MAX, t * R_MAX));
}

function pointAt(angle: number, r: number): { x: number; y: number } {
  return { x: CX + Math.cos(angle) * r, y: CY + Math.sin(angle) * r };
}

type Phase = "idle" | "embedding" | "routing" | "routed" | "refused" | "coldstart";

function phaseFrom(steps: AgentStep[]): Phase {
  let phase: Phase = "idle";
  for (const s of steps) {
    if (s.node === "embed_frame") phase = "embedding";
    else if (s.node === "route") phase = "routing";
    else if (s.node === "decide") phase = s.decision === "route" ? "routed" : "refused";
    else if (s.node === "cold_start") phase = "coldstart";
  }
  return phase;
}

function candidatesFrom(steps: AgentStep[]): Candidate[] {
  for (let i = steps.length - 1; i >= 0; i--) {
    const s = steps[i];
    if (s && s.node === "route" && Array.isArray(s.candidates)) {
      return s.candidates as Candidate[];
    }
  }
  return [];
}

export function RegistryMap({ steps, running }: { steps: AgentStep[]; running: boolean }) {
  const [layout, setLayout] = useState<RegistryLayout | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/registry/layout`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d: RegistryLayout) => {
        if (!cancelled) setLayout(d);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [steps.length === 0]);

  const phase = phaseFrom(steps);
  const candidates = candidatesFrom(steps);
  const scores = useMemo(() => {
    const m = new Map<string, Candidate>();
    for (const c of candidates) m.set(c.model_id, c);
    return m;
  }, [candidates]);

  const winner = phase === "routed" || phase === "coldstart" ? candidates[0] ?? null : null;
  const models: LayoutModel[] = layout?.models ?? [];
  const active = phase !== "idle" && phase !== "embedding";

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full"
        role="img"
        aria-label="Live map of the routing decision across the model registry"
      >
        {/* similarity guide rings */}
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <circle
            key={f}
            cx={CX}
            cy={CY}
            r={R_MAX * f}
            fill="none"
            stroke="var(--grid)"
            strokeWidth={1}
            strokeDasharray={f === 1 ? undefined : "3 5"}
          />
        ))}

        {models.map((m) => {
          const cand = scores.get(m.model_id);
          const gateR = radius(m.gate);
          const gateP = pointAt(m.angle, gateR);
          const rimP = pointAt(m.angle, R_MAX);
          const isWinner = winner?.model_id === m.model_id;
          const dotR = cand ? radius(cand.score) : R_MAX;
          const dot = pointAt(m.angle, dotR);
          const inside = cand ? cand.score >= m.gate : false;

          const colour = !active
            ? "var(--text-muted)"
            : isWinner
              ? "var(--ok)"
              : inside
                ? "var(--accent)"
                : "var(--warn)";
          const label = pointAt(m.angle, R_MAX + 26);

          return (
            <g key={m.model_id}>
              {/* spoke */}
              <line
                x1={CX}
                y1={CY}
                x2={rimP.x}
                y2={rimP.y}
                stroke="var(--grid)"
                strokeWidth={1}
              />
              {/* gate tick: the model's declared competence boundary */}
              <line
                x1={gateP.x - Math.sin(m.angle) * 11}
                y1={gateP.y + Math.cos(m.angle) * 11}
                x2={gateP.x + Math.sin(m.angle) * 11}
                y2={gateP.y - Math.cos(m.angle) * 11}
                stroke={active && !inside ? "var(--warn)" : "var(--text-muted)"}
                strokeWidth={2.5}
                strokeLinecap="round"
                opacity={0.85}
              />
              {/* the frame's position on this model's spoke */}
              <circle
                cx={dot.x}
                cy={dot.y}
                r={isWinner ? 10 : 6}
                fill={colour}
                opacity={active ? 1 : 0.25}
                style={{ transition: "cx 700ms cubic-bezier(.2,.7,.3,1), cy 700ms cubic-bezier(.2,.7,.3,1), r 300ms" }}
              >
                <title>
                  {`${m.name}: score ${cand ? cand.score.toFixed(4) : "—"} vs gate ${m.gate.toFixed(4)}`}
                </title>
              </circle>
              {isWinner ? (
                <circle cx={dot.x} cy={dot.y} r={17} fill="none" stroke="var(--ok)" strokeWidth={2}>
                  <animate attributeName="r" values="12;22;12" dur="1.6s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.9;0;0.9" dur="1.6s" repeatCount="indefinite" />
                </circle>
              ) : null}
              <text
                x={label.x}
                y={label.y}
                textAnchor={Math.cos(m.angle) > 0.3 ? "start" : Math.cos(m.angle) < -0.3 ? "end" : "middle"}
                dominantBaseline="middle"
                fontSize={11}
                fill={isWinner ? "var(--ok)" : "var(--text-muted)"}
                fontWeight={isWinner ? 700 : 400}
              >
                {m.asset_class}
              </text>
              {cand ? (
                <text
                  x={label.x}
                  y={label.y + 13}
                  textAnchor={Math.cos(m.angle) > 0.3 ? "start" : Math.cos(m.angle) < -0.3 ? "end" : "middle"}
                  dominantBaseline="middle"
                  fontSize={10}
                  fill={colour}
                  className="mono"
                >
                  {cand.score.toFixed(3)}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* the incoming frame */}
        <circle
          cx={CX}
          cy={CY}
          r={13}
          fill={phase === "refused" ? "var(--warn)" : "var(--accent)"}
          opacity={phase === "idle" ? 0.3 : 1}
        />
        {running ? (
          <circle cx={CX} cy={CY} r={13} fill="none" stroke="var(--accent)" strokeWidth={2}>
            <animate attributeName="r" values="13;30;13" dur="1.8s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.8;0;0.8" dur="1.8s" repeatCount="indefinite" />
          </circle>
        ) : null}
        <text x={CX} y={CY + 34} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
          frame
        </text>

        {phase === "refused" ? (
          <text
            x={CX}
            y={22}
            textAnchor="middle"
            fontSize={13}
            fontWeight={700}
            fill="var(--warn)"
          >
            every model out of range — refusing
          </text>
        ) : null}
      </svg>

      <figcaption className="muted mt-2 text-xs">
        Distance from the centre is <span className="mono">1 − similarity</span>, measured. The tick on
        each spoke is that model&apos;s coverage gate; a dot <strong>inside</strong> its tick means the
        frame is within the competence that model learned. Angle is illustrative ordering only.
      </figcaption>
    </figure>
  );
}

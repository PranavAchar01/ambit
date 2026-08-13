"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchProviders, speak } from "@/lib/api";
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

const MUTE_KEY = "ambit.speech-muted";

/**
 * Speaks a defect narrative aloud, and is muted until asked.
 *
 * Default-muted on purpose: the demo is set up in a room full of people and a
 * laptop that announces every test frame during setup is a liability. One click
 * before going on stage turns it on.
 */
function SpeakerControl({ result }: { result: InspectResult }) {
  const [muted, setMuted] = useState(true);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [reason, setReason] = useState("checking for a speech provider");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const spokenRef = useRef<string | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem(MUTE_KEY);
    if (stored === "false") setMuted(false);
    fetchProviders()
      .then((h) => {
        setAvailable(h.providers.tts.available);
        setReason(h.providers.tts.available ? "speech ready" : "voice unavailable");
      })
      .catch(() => {
        setAvailable(false);
        setReason("voice unavailable");
      });
  }, []);

  const play = useCallback(async (text: string) => {
    const blob = await speak(text);
    // Synthesis failure is silent by contract: the verdict is already on
    // screen and must never wait on, or be interrupted by, audio.
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    if (audioRef.current) {
      audioRef.current.src = url;
      void audioRef.current.play().catch(() => undefined);
    }
  }, []);

  // The system announces the defect the moment it finds one -- but only once
  // per finding, or a re-render would talk over itself.
  useEffect(() => {
    if (muted || available !== true) return;
    if (result.verdict !== "defect" || !result.narrative) return;
    const key = result.finding_id ?? result.narrative;
    if (spokenRef.current === key) return;
    spokenRef.current = key;
    void play(result.narrative);
  }, [muted, available, result.verdict, result.narrative, result.finding_id, play]);

  const toggle = () => {
    setMuted((m) => {
      window.localStorage.setItem(MUTE_KEY, String(!m));
      return !m;
    });
  };

  return (
    <span className="row" style={{ gap: "var(--step)" }}>
      <audio ref={audioRef} style={{ display: "none" }} />
      <button
        type="button"
        className="btn-link"
        onClick={toggle}
        disabled={available === false}
        title={available === false ? "voice unavailable" : muted ? "unmute spoken findings" : "mute spoken findings"}
        aria-pressed={!muted}
      >
        {available === false ? "\u{1F507} voice unavailable" : muted ? "\u{1F507} muted" : "\u{1F50A} speaking"}
      </button>
      <span className="sr-only">{reason}</span>
    </span>
  );
}

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
        <span className="row" style={{ gap: "calc(var(--step) * 2)" }}>
          <SpeakerControl result={result} />
          {result.latency_ms !== null ? (
            <span className="mono tiny faint">{result.latency_ms} ms end-to-end</span>
          ) : null}
        </span>
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

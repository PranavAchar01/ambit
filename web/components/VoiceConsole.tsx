"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  connectVoice,
  type ToolCallRequest,
  type VoiceConnection,
  type VoiceStatus,
  type VoiceToolResponse,
} from "@/lib/voice";

/**
 * The voice console.
 *
 * The transcript alone would be unfalsifiable -- a language model can narrate a
 * fleet it has never seen. So the right-hand column shows the tool traffic:
 * which MongoDB-backed tool ran, with what arguments, how long Atlas took, and
 * the rows that came back. The spoken answer is only as good as what is sitting
 * next to it.
 */

interface Utterance {
  id: string;
  role: "operator" | "analyst";
  text: string;
  pending: boolean;
}

interface ToolEvent {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  latencyMs?: number;
  result?: unknown;
  error?: string;
}

const SUGGESTIONS = [
  "What can GridSight inspect right now?",
  "Which asset class is failing fastest this month?",
  "Why did GridSight refuse the last frame it could not route?",
  "Take the most recent insulator defect — have we seen it before?",
];

const STATUS_COPY: Record<VoiceStatus, string> = {
  idle: "Not connected",
  minting: "Minting ephemeral credential…",
  negotiating: "Negotiating WebRTC…",
  live: "Live",
  closed: "Disconnected",
  error: "Connection error",
};

function statusColor(status: VoiceStatus): string {
  if (status === "live") return "var(--ok)";
  if (status === "error") return "var(--danger)";
  if (status === "minting" || status === "negotiating") return "var(--warn)";
  return "var(--text-muted)";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function count(value: unknown): number | null {
  return Array.isArray(value) ? value.length : null;
}

/** One line an audience can read at a glance: what did Mongo actually return? */
function summarise(name: string, result: unknown): string {
  if (!isRecord(result)) return "no rows";
  switch (name) {
    case "registry_status": {
      const classes = count(result.asset_classes);
      return `${String(result.models)} models · ${classes ?? 0} asset classes · ${String(
        result.total_findings,
      )} findings`;
    }
    case "query_trends": {
      const verdicts = isRecord(result.verdict_counts) ? result.verdict_counts : {};
      const parts = Object.entries(verdicts).map(([k, v]) => `${String(v)} ${k}`);
      return `${String(result.total_inspections)} inspections over ${String(
        result.window_days,
      )} days — ${parts.join(", ")}`;
    }
    case "search_findings":
      return `${String(result.matched)} matched, ${String(result.returned)} returned`;
    case "recall_similar": {
      const recurrence = isRecord(result.recurrence) ? result.recurrence : {};
      return `${String(result.count)} similar past findings · ${String(
        recurrence.similar_defects ?? 0,
      )} were defects`;
    }
    case "explain_refusal": {
      const candidates = count(result.candidates) ?? 0;
      return result.refused === true
        ? `refused — ${candidates} candidates, none cleared its gate`
        : `not a refusal (${candidates} candidates recorded)`;
    }
    default:
      return `${Object.keys(result).length} fields`;
  }
}

export function VoiceConsole() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const connectionRef = useRef<VoiceConnection | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [notice, setNotice] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [toolCount, setToolCount] = useState(0);
  const [micAvailable, setMicAvailable] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [tools, setTools] = useState<ToolEvent[]>([]);
  const [typed, setTyped] = useState("");

  useEffect(() => {
    const node = transcriptRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [utterances]);

  useEffect(() => () => connectionRef.current?.close(), []);

  const upsertAssistant = useCallback((itemId: string, text: string, pending: boolean) => {
    setUtterances((prev) => {
      const existing = prev.findIndex((u) => u.id === itemId);
      if (existing === -1) return [...prev, { id: itemId, role: "analyst", text, pending }];
      const next = [...prev];
      next[existing] = { id: itemId, role: "analyst", text, pending };
      return next;
    });
  }, []);

  const connect = useCallback(async () => {
    if (connectionRef.current || !audioRef.current) return;
    setNotice(null);
    setUtterances([]);
    setTools([]);
    const deltas = new Map<string, string>();

    try {
      const conn = await connectVoice(audioRef.current, {
        onStatus: (next, detail) => {
          setStatus(next);
          if (detail) setNotice(detail);
        },
        onError: (message) => setNotice(message),
        onUserTranscript: (text) =>
          setUtterances((prev) => [
            ...prev,
            { id: `op-${prev.length}-${Date.now()}`, role: "operator", text, pending: false },
          ]),
        onAssistantDelta: (itemId, delta) => {
          const text = (deltas.get(itemId) ?? "") + delta;
          deltas.set(itemId, text);
          upsertAssistant(itemId, text, true);
        },
        onAssistantDone: (itemId, text) => {
          const final = text || deltas.get(itemId) || "";
          deltas.delete(itemId);
          upsertAssistant(itemId, final, false);
        },
        onToolStart: (call: ToolCallRequest) =>
          setTools((prev) => [
            ...prev,
            { callId: call.callId, name: call.name, args: call.args, status: "running" },
          ]),
        onToolResult: (callId, response: VoiceToolResponse) =>
          setTools((prev) =>
            prev.map((t) =>
              t.callId === callId
                ? {
                    ...t,
                    status: "done",
                    latencyMs: response.latency_ms,
                    result: response.result,
                  }
                : t,
            ),
          ),
        onToolError: (callId, message) =>
          setTools((prev) =>
            prev.map((t) => (t.callId === callId ? { ...t, status: "error", error: message } : t)),
          ),
      });
      connectionRef.current = conn;
      setModel(conn.model);
      setToolCount(conn.tools.length);
      setMicAvailable(conn.micAvailable);
      setMicOn(conn.micAvailable);
    } catch (err) {
      setStatus("error");
      setNotice(err instanceof Error ? err.message : "could not start the voice session");
    }
  }, [upsertAssistant]);

  const disconnect = useCallback(() => {
    connectionRef.current?.close();
    connectionRef.current = null;
    setStatus("closed");
    setMicAvailable(false);
  }, []);

  const ask = useCallback((question: string) => {
    const conn = connectionRef.current;
    if (!conn || !question.trim()) return;
    conn.sendText(question.trim());
    setUtterances((prev) => [
      ...prev,
      { id: `op-${prev.length}-${Date.now()}`, role: "operator", text: question.trim(), pending: false },
    ]);
  }, []);

  const live = status === "live";
  const busy = status === "minting" || status === "negotiating";

  return (
    <div className="flex flex-col gap-4">
      {/* Realtime speech: the answer is captioned live in the transcript below. */}
      <audio ref={audioRef} autoPlay className="hidden" />

      <div className="surface flex flex-wrap items-center gap-3 rounded-lg p-3">
        <span className="flex items-center gap-2 text-sm font-medium">
          <span
            aria-hidden
            className={`inline-block h-2.5 w-2.5 rounded-full ${live ? "step-active" : ""}`}
            style={{ background: statusColor(status) }}
          />
          {STATUS_COPY[status]}
        </span>

        <button
          type="button"
          onClick={live ? disconnect : () => void connect()}
          disabled={busy}
          className="rounded px-3 py-1.5 text-sm font-semibold disabled:opacity-60"
          style={{ background: live ? "var(--surface-2)" : "var(--accent)", color: live ? "var(--text)" : "#fff" }}
        >
          {live ? "Disconnect" : busy ? "Connecting…" : "Start voice session"}
        </button>

        {live && micAvailable ? (
          <button
            type="button"
            onClick={() => {
              const next = !micOn;
              connectionRef.current?.setMicEnabled(next);
              setMicOn(next);
            }}
            className="rounded px-3 py-1.5 text-sm"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
            aria-pressed={!micOn}
          >
            {micOn ? "Mute mic" : "Unmute mic"}
          </button>
        ) : null}

        <span className="muted mono ml-auto text-xs">
          {model ? `${model} · ${toolCount} MongoDB tools` : "no session"}
        </span>
      </div>

      {notice ? (
        <p className="surface rounded-lg p-3 text-sm" role="alert" style={{ color: "var(--warn)" }}>
          {notice}
        </p>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="surface flex min-w-0 flex-col rounded-lg p-4">
          <h2 className="text-sm font-semibold tracking-tight">Transcript</h2>
          <div
            ref={transcriptRef}
            className="mt-3 flex max-h-[26rem] min-h-[12rem] flex-col gap-3 overflow-y-auto"
            aria-live="polite"
          >
            {utterances.length === 0 ? (
              <p className="muted text-sm">
                {live
                  ? "Ask about the fleet — speak, or use a question below."
                  : "Start a session to talk to the fleet analyst."}
              </p>
            ) : (
              utterances.map((u) => (
                <div key={u.id} className="flex min-w-0 flex-col gap-1">
                  <span
                    className="text-xs font-semibold uppercase tracking-wide"
                    style={{ color: u.role === "operator" ? "var(--text-muted)" : "var(--accent)" }}
                  >
                    {u.role === "operator" ? "Operator" : "GridSight"}
                  </span>
                  <p className="m-0 break-words text-sm">
                    {u.text}
                    {u.pending ? <span className="step-active"> ▌</span> : null}
                  </p>
                </div>
              ))
            )}
          </div>

          <form
            className="mt-3 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              ask(typed);
              setTyped("");
            }}
          >
            <label className="sr-only" htmlFor="voice-typed">
              Ask a question by text
            </label>
            <input
              id="voice-typed"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              disabled={!live}
              placeholder={live ? "…or type a question" : "Connect first"}
              className="min-w-0 flex-1 rounded px-3 py-2 text-sm"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text)" }}
            />
            <button
              type="submit"
              disabled={!live || !typed.trim()}
              className="rounded px-3 py-2 text-sm font-semibold disabled:opacity-50"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
            >
              Ask
            </button>
          </form>

          <div className="mt-3 flex flex-wrap gap-2">
            {SUGGESTIONS.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => ask(q)}
                disabled={!live}
                className="rounded-full px-3 py-1 text-xs disabled:opacity-50"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
              >
                {q}
              </button>
            ))}
          </div>
        </section>

        <section className="surface flex min-w-0 flex-col rounded-lg p-4">
          <h2 className="text-sm font-semibold tracking-tight">MongoDB tool calls</h2>
          <p className="muted mt-1 text-xs">
            Every claim the analyst makes has to appear here first. Tools execute on the API, against
            Atlas — the browser only relays the call.
          </p>
          <ol className="m-0 mt-3 flex max-h-[30rem] list-none flex-col gap-3 overflow-y-auto p-0">
            {tools.length === 0 ? (
              <li className="muted text-sm">No tool has run yet.</li>
            ) : (
              tools.map((t) => (
                <li key={t.callId} className="min-w-0 rounded p-3" style={{ background: "var(--surface-2)" }}>
                  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                    <span className="mono text-sm font-semibold">{t.name}</span>
                    <span className="mono muted min-w-0 break-all text-xs">
                      {JSON.stringify(t.args)}
                    </span>
                    <span
                      className="mono ml-auto text-xs"
                      style={{
                        color:
                          t.status === "error"
                            ? "var(--danger)"
                            : t.status === "done"
                              ? "var(--ok)"
                              : "var(--warn)",
                      }}
                    >
                      {t.status === "running"
                        ? "querying Atlas…"
                        : t.status === "error"
                          ? "failed"
                          : `${t.latencyMs ?? 0} ms`}
                    </span>
                  </div>
                  <p className="m-0 mt-1.5 break-words text-sm">
                    {t.status === "error" ? t.error : t.status === "done" ? summarise(t.name, t.result) : "…"}
                  </p>
                  {t.status === "done" ? (
                    <details className="mt-2">
                      <summary className="muted cursor-pointer text-xs">Raw result</summary>
                      <pre
                        className="mono mt-2 max-h-56 overflow-auto rounded p-2 text-xs"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
                      >
                        {JSON.stringify(t.result, null, 2)}
                      </pre>
                    </details>
                  ) : null}
                </li>
              ))
            )}
          </ol>
        </section>
      </div>
    </div>
  );
}

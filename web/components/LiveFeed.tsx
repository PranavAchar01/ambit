"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { AgentStep, InspectResult } from "@/lib/types";

interface Props {
  session: string;
  onStart: () => void;
  onStep: (step: AgentStep) => void;
  onResult: (result: InspectResult) => void;
  onError: (message: string) => void;
}

type Phase = "idle" | "holding" | "inspecting" | "cooldown";

/**
 * The projected half of the phone link. Subscribes to the relay and paints
 * whatever the phone's camera is pointed at, then hands the agent events it
 * relays back up to the page so the phone drives exactly the same result view
 * as a drag-and-drop upload.
 */
export function LiveFeed({ session, onStart, onStep, onResult, onError }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const framesRef = useRef(0);
  const seqRef = useRef(0);
  const paintedRef = useRef(0);
  const decodingRef = useRef(0);

  const [online, setOnline] = useState(false);
  const [linked, setLinked] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [fps, setFps] = useState(0);
  const [captureUrl, setCaptureUrl] = useState<string | null>(null);

  // Kept in a ref so a parent passing inline arrows cannot retrigger the socket
  // effect on every render and put the relay into a reconnect loop.
  const handlers = useRef({ onStart, onStep, onResult, onError });
  handlers.current = { onStart, onStep, onResult, onError };

  useEffect(() => {
    fetch(`${API_BASE}/live/config?session=${encodeURIComponent(session)}`, { cache: "no-store" })
      .then((r) => r.json())
      .then((c: { capture_url: string | null }) => setCaptureUrl(c.capture_url))
      .catch(() => setCaptureUrl(null));
  }, [session]);

  /**
   * createImageBitmap decodes off the main thread and hands back something
   * drawImage can blit directly. The previous approach -- a fresh object URL
   * and an <img> src swap per frame -- put a decode plus a GC'd URL on the main
   * thread 20+ times a second and could not hold frame rate.
   */
  const paint = useCallback(async (blob: Blob) => {
    // Decoding is fast, but never let a backlog build: newest frame wins.
    if (decodingRef.current > 2) return;
    const seq = ++seqRef.current;
    decodingRef.current += 1;
    try {
      const bitmap = await createImageBitmap(blob);
      const canvas = canvasRef.current;
      // A frame that lost the race to a newer one is worthless -- drop it.
      if (!canvas || seq < paintedRef.current) {
        bitmap.close();
        return;
      }
      paintedRef.current = seq;
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
      bitmap.close();
      framesRef.current += 1;
    } finally {
      decodingRef.current -= 1;
    }
  }, []);

  // Divide by measured elapsed time, not by the nominal interval: a backgrounded
  // tab throttles setInterval, and assuming 1000ms turns a stall into a wildly
  // inflated frame rate rather than an obviously stale one.
  useEffect(() => {
    let since = performance.now();
    const id = setInterval(() => {
      const now = performance.now();
      const seconds = (now - since) / 1000;
      if (seconds > 0) setFps(Math.round(framesRef.current / seconds));
      framesRef.current = 0;
      since = now;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      const url = `${API_BASE.replace(/^http/, "ws")}/live/watch/${encodeURIComponent(session)}`;
      socket = new WebSocket(url);
      socket.binaryType = "blob";

      socket.onopen = () => setLinked(true);
      socket.onclose = () => {
        setLinked(false);
        setOnline(false);
        if (!closed) retry = setTimeout(connect, 1000);
      };
      socket.onerror = () => socket?.close();

      socket.onmessage = (ev: MessageEvent<Blob | string>) => {
        if (ev.data instanceof Blob) {
          void paint(ev.data);
          return;
        }
        const msg = JSON.parse(ev.data) as { type: string; [k: string]: unknown };
        switch (msg.type) {
          case "presence":
            setOnline(Boolean(msg.publisher_online));
            break;
          case "phase": {
            const next = String(msg.phase) as Phase;
            setPhase(next);
            setProgress(typeof msg.progress === "number" ? msg.progress : 0);
            if (next === "inspecting") handlers.current.onStart();
            break;
          }
          case "step":
            handlers.current.onStep(msg.data as AgentStep);
            break;
          case "result":
            handlers.current.onResult(msg.data as InspectResult);
            break;
          case "error":
            handlers.current.onError(String(msg.message ?? "phone reported an error"));
            break;
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      socket?.close();
    };
  }, [session, paint]);

  const label =
    phase === "inspecting"
      ? "Inspecting"
      : phase === "holding"
        ? `Holding ${(progress * 2).toFixed(1)}s / 2.0s`
        : phase === "cooldown"
          ? "Move, then hold still again"
          : "Waiting for a steady frame";

  return (
    <section className="panel">
      <div className="panel-hd row-between">
        <h2>Phone camera</h2>
        <span className="row tiny">
          <span className="mono muted">{session}</span>
          <span className="mono" style={{ color: linked ? "var(--fg-dim)" : "var(--red)" }}>
            {linked ? "relay up" : "relay down"}
          </span>
          {online ? <span className="mono muted">{fps} fps</span> : null}
        </span>
      </div>

      {online ? (
        <div className="stack">
          <canvas
            ref={canvasRef}
            role="img"
            aria-label="Live view from the paired phone camera"
            style={{
              width: "100%",
              background: "#000",
              border: "1px solid var(--line)",
              aspectRatio: "4 / 3",
              objectFit: "contain",
            }}
          />
          <div className="row-between">
            <span className="caps">{label}</span>
            <span className="mono tiny muted">480×360 preview · full-res on capture</span>
          </div>
          <div style={{ height: "6px", background: "var(--surface-2)", border: "1px solid var(--line)" }}>
            <div
              style={{
                height: "100%",
                width: `${Math.round((phase === "inspecting" ? 1 : progress) * 100)}%`,
                background: "var(--red)",
              }}
            />
          </div>
        </div>
      ) : (
        <div className="stack">
          <p className="small muted">
            No phone connected. Run <span className="mono">scripts/tunnel.sh</span>, then scan this
            on the phone — it must be the HTTPS tunnel URL, because the camera API refuses to run on
            a plain LAN address.
          </p>
          {captureUrl ? (
            <div className="row" style={{ alignItems: "flex-start" }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${API_BASE}/live/qr.svg?session=${encodeURIComponent(session)}`}
                alt={`QR code linking to ${captureUrl}`}
                style={{ width: "180px", border: "1px solid var(--line)" }}
              />
              <p className="mono tiny" style={{ wordBreak: "break-all" }}>
                {captureUrl}
              </p>
            </div>
          ) : (
            <p className="mono tiny faint">
              No tunnel yet — start scripts/tunnel.sh and reload.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

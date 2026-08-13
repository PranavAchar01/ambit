"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AgentSteps } from "@/components/AgentSteps";
import { ColdStartPanel } from "@/components/ColdStartPanel";
import { LiveFeed } from "@/components/LiveFeed";
import { RegistryMap } from "@/components/RegistryMap";
import { ResultView } from "@/components/ResultView";
import { VerdictCard } from "@/components/VerdictCard";
import { imageUrl, inspectStream } from "@/lib/api";
import type { AgentStep, InspectResult } from "@/lib/types";

type Source = "upload" | "live";

const SOURCE_KEY = "ambit.capture-source";

export default function InspectPage() {
  const [source, setSource] = useState<Source>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [origin, setOrigin] = useState<Source | null>(null);
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [result, setResult] = useState<InspectResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Read after mount rather than during render: the server has no localStorage
  // and a mismatched first paint would hydrate-error.
  useEffect(() => {
    const stored = window.localStorage.getItem(SOURCE_KEY);
    if (stored === "upload" || stored === "live") setSource(stored);
  }, []);

  const choose = useCallback((next: Source) => {
    setSource(next);
    window.localStorage.setItem(SOURCE_KEY, next);
  }, []);

  const run = useCallback(async (chosen: File) => {
    setFile(chosen);
    setOrigin("upload");
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(chosen);
    });
    setSteps([]);
    setResult(null);
    setError(null);
    setRunning(true);

    try {
      await inspectStream(chosen, null, {
        onStep: (step) => setSteps((prev) => [...prev, step]),
        onResult: (res) => setResult(res),
        onError: (message) => setError(message),
      });
    } catch (exc) {
      // inspectStream throws out of the function when the fetch itself is
      // rejected. Without this the UI stays "running" forever on a dropped API.
      setError(exc instanceof Error ? exc.message : "The API could not be reached.");
    } finally {
      setRunning(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped) void run(dropped);
    },
    [run],
  );

  const started = running || steps.length > 0 || result !== null;
  const liveFrame = origin === "live" ? imageUrl(result?.uploaded_image_url ?? null) : null;

  return (
    <div className="stack-lg">
      <header className="stack">
        <h1>Inspect a board</h1>
        <p className="lede small">
          The agent embeds the frame with OpenCLIP, vector-searches the MongoDB model registry for
          the specialist trained on the most similar imagery, and runs it. If nothing clears that
          specialist&apos;s coverage gate, it refuses rather than guessing.
        </p>
      </header>

      <div className="split">
        {/* ---------------------------------------------------------- capture */}
        <div className="stack">
          <div className="seg" role="group" aria-label="Frame source">
            <button
              type="button"
              className="seg-btn"
              aria-pressed={source === "upload"}
              onClick={() => choose("upload")}
            >
              Upload
            </button>
            <button
              type="button"
              className="seg-btn"
              aria-pressed={source === "live"}
              onClick={() => choose("live")}
            >
              Live
            </button>
          </div>

          {source === "upload" ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className="dropzone"
              data-drag={dragging ? "true" : undefined}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={(e) => {
                  const chosen = e.target.files?.[0];
                  if (chosen) void run(chosen);
                }}
              />
              <p className="small">
                Drag a frame here, or{" "}
                <button type="button" onClick={() => inputRef.current?.click()} className="btn-link">
                  choose a file
                </button>
                .
              </p>
            </div>
          ) : (
            <LiveFeed
              session="demo"
              onStart={() => {
                setFile(null);
                setOrigin("live");
                setPreviewUrl((prev) => {
                  if (prev) URL.revokeObjectURL(prev);
                  return null;
                });
                setSteps([]);
                setResult(null);
                setError(null);
                setRunning(true);
              }}
              onStep={(step) => setSteps((prev) => [...prev, step])}
              onResult={(res) => {
                setResult(res);
                setRunning(false);
              }}
              onError={(message) => {
                setError(message);
                setRunning(false);
              }}
            />
          )}

          <section className="panel stack">
            <div className="panel-hd row-between">
              <h2>Under inspection</h2>
              {origin ? <span className="badge badge-nominal">{origin.toUpperCase()}</span> : null}
            </div>
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="Frame being inspected" className="thumb" />
            ) : liveFrame ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={liveFrame} alt="Frame captured by the phone" className="thumb" />
            ) : (
              <p className="empty small">
                {source === "upload"
                  ? "No frame yet. Drop one above."
                  : "No frame yet. Hold the phone steady, or use its capture button."}
              </p>
            )}
            {file ? <p className="mono tiny muted">{file.name}</p> : null}
          </section>
        </div>

        {/* --------------------------------------------------------- analysis */}
        <div className="stack-lg">
          {!started ? (
            <div className="empty">Drop a frame, or connect a phone.</div>
          ) : (
            <>
              {result ? <VerdictCard result={result} /> : null}

              <section className="panel stack">
                <h2 className="panel-hd">Agent</h2>
                <AgentSteps
                  steps={result?.trace?.length ? result.trace : steps}
                  running={running}
                />
                <div style={{ marginTop: "calc(var(--step) * 2)" }}>
                  <RegistryMap
                    steps={result?.trace?.length ? result.trace : steps}
                    running={running}
                  />
                </div>
              </section>

              {error ? (
                <p className="panel-alert small red" role="alert">
                  {error}
                </p>
              ) : null}

              {result?.verdict === "unroutable" ? (
                <ColdStartPanel
                  frame={file}
                  suggestedClass=""
                  onComplete={(res) => {
                    setResult(res);
                    setSteps(res.trace ?? []);
                  }}
                />
              ) : null}

              {result ? <ResultView result={result} /> : null}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

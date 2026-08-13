"use client";

import { useCallback, useRef, useState } from "react";
import { AgentSteps } from "@/components/AgentSteps";
import { ColdStartPanel } from "@/components/ColdStartPanel";
import { LiveFeed } from "@/components/LiveFeed";
import { RegistryMap } from "@/components/RegistryMap";
import { ResultView } from "@/components/ResultView";
import { inspectStream } from "@/lib/api";
import type { AgentStep, InspectResult } from "@/lib/types";

export default function InspectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [result, setResult] = useState<InspectResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (chosen: File) => {
    setFile(chosen);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(chosen);
    });
    setSteps([]);
    setResult(null);
    setError(null);
    setRunning(true);

    await inspectStream(chosen, null, {
      onStep: (step) => setSteps((prev) => [...prev, step]),
      onResult: (res) => setResult(res),
      onError: (message) => setError(message),
    });
    setRunning(false);
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

      <LiveFeed
        session="demo"
        onStart={() => {
          setFile(null);
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
        {file ? <p className="mono muted tiny" style={{ marginTop: "8px" }}>{file.name}</p> : null}
      </div>

      {(running || steps.length > 0) && !result ? (
        <section className="panel">
          <h2 className="panel-hd">Agent</h2>
          <div className="grid-map">
            <RegistryMap steps={steps} running={running} />
            <div className="stack">
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl}
                  alt="Frame being inspected"
                  style={{ width: "100%", maxWidth: "13rem", border: "1px solid var(--line)" }}
                />
              ) : null}
              <AgentSteps steps={steps} running={running} />
            </div>
          </div>
        </section>
      ) : null}

      {error ? (
        <p className="panel-alert small red" role="alert">
          {error}
        </p>
      ) : null}

      {result ? (
        <>
          <section className="panel">
            <h2 className="panel-hd">Routing decision across the registry</h2>
            <div className="grid-map">
              <RegistryMap steps={result.trace?.length ? result.trace : steps} running={false} />
              <AgentSteps steps={result.trace?.length ? result.trace : steps} running={false} />
            </div>
          </section>
          {result.verdict === "unroutable" ? (
            <ColdStartPanel
              frame={file}
              suggestedClass=""
              onComplete={(res) => {
                setResult(res);
                setSteps(res.trace ?? []);
              }}
            />
          ) : null}
          <ResultView result={result} />
          <details className="panel">
            <summary>Agent trace ({result.trace?.length ?? steps.length} steps)</summary>
            <div style={{ marginTop: "calc(var(--step) * 2)" }}>
              <AgentSteps steps={result.trace?.length ? result.trace : steps} running={false} />
            </div>
          </details>
        </>
      ) : null}
    </div>
  );
}

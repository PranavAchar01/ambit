"use client";

import { useCallback, useRef, useState } from "react";
import { AgentSteps } from "@/components/AgentSteps";
import { ColdStartPanel } from "@/components/ColdStartPanel";
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
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Inspect a drone frame</h1>
        <p className="muted mt-1 max-w-3xl text-sm">
          The agent embeds the frame with OpenCLIP, vector-searches the MongoDB model registry for
          the specialist trained on the most similar imagery, and runs it. If nothing clears that
          specialist&apos;s coverage gate, it refuses rather than guessing.
        </p>
      </header>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className="rounded-xl p-6 text-center transition-colors"
        style={{
          background: dragging ? "var(--surface-2)" : "var(--surface)",
          border: `2px dashed ${dragging ? "var(--accent)" : "var(--border)"}`,
        }}
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
        <p className="text-sm">
          Drag a frame here, or{" "}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="font-semibold underline"
            style={{ color: "var(--accent)" }}
          >
            choose a file
          </button>
          .
        </p>
        {file ? <p className="muted mono mt-2 text-xs">{file.name}</p> : null}
      </div>

      {(running || steps.length > 0) && !result ? (
        <section className="surface rounded-lg p-4">
          <h2 className="mb-3 text-sm font-semibold">Agent</h2>
          <div className="grid gap-4 sm:grid-cols-[minmax(0,14rem)_1fr]">
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={previewUrl}
                alt="Frame being inspected"
                className="w-full rounded-md"
                style={{ border: "1px solid var(--border)" }}
              />
            ) : (
              <div />
            )}
            <AgentSteps steps={steps} running={running} />
          </div>
        </section>
      ) : null}

      {error ? (
        <p className="surface rounded-lg p-4 text-sm" style={{ color: "var(--danger)" }} role="alert">
          {error}
        </p>
      ) : null}

      {result ? (
        <>
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
          <details className="surface rounded-lg p-4">
            <summary className="cursor-pointer text-sm font-semibold">
              Agent trace ({result.trace?.length ?? steps.length} steps)
            </summary>
            <div className="mt-3">
              <AgentSteps steps={result.trace?.length ? result.trace : steps} running={false} />
            </div>
          </details>
        </>
      ) : null}
    </div>
  );
}

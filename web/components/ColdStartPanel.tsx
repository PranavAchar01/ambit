"use client";

import { useState } from "react";
import { coldStart } from "@/lib/api";
import type { InspectResult } from "@/lib/types";

interface Props {
  frame: File | null;
  suggestedClass: string;
  onComplete: (result: InspectResult) => void;
}

/**
 * Shown only when the agent refused to route. Collects a handful of known-good
 * reference images, trains a specialist, and re-runs the original frame against it.
 */
export function ColdStartPanel({ frame, suggestedClass, onComplete }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [assetClass, setAssetClass] = useState(suggestedClass);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const enough = files.length >= 4;

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const result = await coldStart(files, assetClass.trim() || "new_asset", frame);
      onComplete(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "cold start failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section
      className="rounded-lg p-4"
      style={{ background: "var(--surface)", border: "1px solid var(--warn)" }}
    >
      <h2 className="text-base font-semibold">No model in the registry covers this asset</h2>
      <p className="muted mt-1.5 text-sm">
        GridSight will not score this frame with a specialist trained on different hardware — that
        produces a confident, wrong answer. Upload a handful of <strong>known-good</strong> images of
        this asset and it will fit a new specialist, register it, and re-run this frame against it.
      </p>

      <div className="mt-4 flex flex-col gap-3">
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">Asset class name</span>
          <input
            type="text"
            value={assetClass}
            onChange={(e) => setAssetClass(e.target.value)}
            placeholder="e.g. rail_surface"
            className="mono rounded-md px-2.5 py-2 text-sm"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text)" }}
          />
        </label>

        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">Known-good reference images</span>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
            className="text-sm"
          />
          <span className="muted text-xs">
            {files.length === 0
              ? "PatchCore needs at least 4; 8–12 gives a tighter threshold."
              : `${files.length} selected${enough ? "" : " — at least 4 needed"}`}
          </span>
        </label>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void submit()}
            disabled={!enough || busy}
            className="rounded-md px-3.5 py-2 text-sm font-semibold disabled:opacity-45"
            style={{ background: "var(--accent)", color: "#0b0f14" }}
          >
            {busy ? "Training specialist…" : "Cold-start a specialist"}
          </button>
          {busy ? (
            <span className="muted text-xs">
              Fitting a memory bank and writing the weights to MongoDB…
            </span>
          ) : null}
        </div>

        {error ? (
          <p className="text-sm" style={{ color: "var(--danger)" }} role="alert">
            {error}
          </p>
        ) : null}
      </div>
    </section>
  );
}

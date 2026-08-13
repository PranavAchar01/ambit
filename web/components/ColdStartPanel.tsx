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
    <section className="panel-alert stack">
      <div>
        <h2>No model in the registry covers this asset</h2>
        <p className="small muted" style={{ marginTop: "6px" }}>
          Ambit will not score this frame with a specialist trained on different hardware — that
          produces a confident, wrong answer. Upload a handful of <strong>known-good</strong> images of
          this asset and it will fit a new specialist, register it, and re-run this frame against it.
        </p>
      </div>

      <label className="field">
        <span className="field-label">Asset class name</span>
        <input
          type="text"
          value={assetClass}
          onChange={(e) => setAssetClass(e.target.value)}
          placeholder="e.g. esp32_devkit"
          className="input"
        />
      </label>

      <label className="field">
        <span className="field-label">Known-good reference images</span>
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
        <span className="muted tiny">
          {files.length === 0
            ? "PatchCore needs at least 4; 8–12 gives a tighter threshold."
            : `${files.length} selected${enough ? "" : " — at least 4 needed"}`}
        </span>
      </label>

      <div className="row">
        <button
          type="button"
          onClick={() => void submit()}
          disabled={!enough || busy}
          className="btn btn-primary"
        >
          {busy ? "Training specialist…" : "Cold-start a specialist"}
        </button>
        {busy ? (
          <span className="muted tiny">
            Fitting a memory bank and writing the weights to MongoDB…
          </span>
        ) : null}
      </div>

      {error ? (
        <p className="small red" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}

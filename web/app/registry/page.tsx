"use client";

import { useEffect, useState } from "react";
import { fetchModels, imageUrl } from "@/lib/api";
import type { ModelsResponse } from "@/lib/types";

export default function RegistryPage() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchModels()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "failed to load registry");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Model registry</h1>
          <p className="muted mt-1 max-w-3xl text-sm">
            Every specialist GridSight can route to. Weights live in GridFS; the 512-d embedding is
            the centroid of what each model was trained on, and the gate is the coverage envelope it
            will accept a frame inside.
          </p>
        </div>
        {data ? (
          <span
            className="rounded-full px-2.5 py-1 text-xs font-semibold"
            style={{
              background: data.vector_index === "READY" ? "var(--ok)" : "var(--warn)",
              color: "#0b0f14",
            }}
          >
            Vector index {data.vector_index ?? "missing"}
          </span>
        ) : null}
      </header>

      {error ? (
        <p className="surface rounded-lg p-4 text-sm" style={{ color: "var(--danger)" }} role="alert">
          {error}
        </p>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {(data?.models ?? []).map((m) => {
          const ref = imageUrl(m.reference_image_url);
          return (
            <article key={m.id} className="surface flex flex-col gap-3 rounded-lg p-4">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <h2 className="mono truncate text-sm font-semibold">{m.name}</h2>
                  <p className="muted text-xs">{m.asset_class}</p>
                </div>
                <span
                  className="shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold"
                  style={{
                    background: m.created_by === "agent-coldstart" ? "var(--accent)" : "var(--surface-2)",
                    color: m.created_by === "agent-coldstart" ? "#0b0f14" : "var(--text-muted)",
                    border: "1px solid var(--border)",
                  }}
                >
                  {m.created_by === "agent-coldstart" ? "cold-started" : "seed"}
                </span>
              </div>

              {ref ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={ref}
                  alt={`Golden known-good reference for ${m.asset_class}`}
                  className="aspect-square w-full rounded-md object-cover"
                  style={{ border: "1px solid var(--border)" }}
                />
              ) : null}

              <dl className="m-0 grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
                {[
                  ["Backbone", m.backbone],
                  ["Training samples", String(m.training_samples)],
                  ["Routing gate", m.routing_threshold?.toFixed(4) ?? "—"],
                  ["Image threshold", m.image_threshold.toFixed(3)],
                  [
                    "Image AUROC",
                    m.metrics.image_auroc !== null ? m.metrics.image_auroc.toFixed(4) : "n/a",
                  ],
                  [
                    "Pixel AUROC",
                    m.metrics.pixel_auroc !== null ? m.metrics.pixel_auroc.toFixed(4) : "no masks",
                  ],
                  ["Weights", `${(m.weights_bytes / 1e6).toFixed(2)} MB`],
                  ["Embedding", `${m.embedding_dim}-d`],
                ].map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="muted">{k}</dt>
                    <dd className="mono m-0 text-right">{v}</dd>
                  </div>
                ))}
              </dl>
            </article>
          );
        })}
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { fetchModels, imageUrl } from "@/lib/api";
import type { ModelsResponse } from "@/lib/types";

/**
 * Dataset licence follows the corpus a model was trained on. MVTec AD is
 * CC BY-NC-SA 4.0 -- non-commercial -- and that constraint has to be visible
 * on the registry itself, not buried in a README.
 */
function licenceFor(assetClass: string): string {
  if (assetClass.startsWith("mvtec_")) return "MVTec AD — CC BY-NC-SA 4.0 — NON-COMMERCIAL USE ONLY";
  if (assetClass.startsWith("visa_")) return "VisA — CC BY 4.0 — commercial use permitted";
  return "Licence — unspecified";
}

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
    <div className="stack-lg">
      <header className="row-between">
        <div className="stack">
          <h1>Model registry</h1>
          <p className="lede small">
            Every specialist Ambit can route to. Weights live in GridFS; the 512-d embedding is
            the centroid of what each model was trained on, and the gate is the coverage envelope it
            will accept a frame inside.
          </p>
        </div>
        {data ? (
          <span
            className={`badge ${data.vector_index === "READY" ? "badge-nominal" : "badge-unroutable"}`}
          >
            Vector index {data.vector_index ?? "missing"}
          </span>
        ) : null}
      </header>

      {error ? (
        <p className="panel-alert small mono red" role="alert">
          {error}
        </p>
      ) : null}

      <div className="grid-cards">
        {(data?.models ?? []).map((m) => {
          const ref = imageUrl(m.reference_image_url);
          const coldStarted = m.created_by === "agent-coldstart";
          return (
            <article key={m.id} className="panel stack">
              <div className="row-between" style={{ alignItems: "flex-start" }}>
                <div style={{ minWidth: 0 }}>
                  <h2 className="mono" style={{ fontSize: "15px" }}>
                    {m.name}
                  </h2>
                  <p className="caps" style={{ marginTop: "4px" }}>
                    {m.asset_class}
                  </p>
                </div>
                <span className={coldStarted ? "tag tag-strong" : "tag"} style={{ flex: "0 0 auto" }}>
                  {coldStarted ? "cold-started" : "seed"}
                </span>
              </div>

              {ref ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={ref}
                  alt={`Golden known-good reference for ${m.asset_class}`}
                  style={{
                    width: "100%",
                    aspectRatio: "1 / 1",
                    objectFit: "cover",
                    border: "1px solid var(--line)",
                  }}
                />
              ) : null}

              <dl
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto",
                  gap: "5px 12px",
                  alignItems: "baseline",
                  margin: 0,
                }}
              >
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
                  <div key={k} style={{ display: "contents" }}>
                    <dt className="caps">{k}</dt>
                    <dd className="mono tiny" style={{ margin: 0, textAlign: "right" }}>
                      {v}
                    </dd>
                  </div>
                ))}
              </dl>

              <p className="tiny faint">{licenceFor(m.asset_class)}</p>
            </article>
          );
        })}
      </div>
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import type { BBoxRegion } from "@/lib/types";

interface Props {
  imageUrl: string;
  heatmapUrl: string | null;
  regions: BBoxRegion[];
  /** 0..1 multiplier on the heatmap's own alpha ramp */
  heatOpacity?: number;
}

/**
 * Deliberately no `crossOrigin`. Nothing here ever reads the canvas back --
 * no getImageData, no toDataURL -- so tainting costs nothing, while requesting
 * the image in CORS mode costs a whole failure path: /image responses are
 * `public, max-age=31536000`, and the plain response carries no `Vary`, so one
 * ordinary <img> load caches an entry with no Access-Control-Allow-Origin. The
 * next CORS-mode request reuses that entry, finds no ACAO header, and fails --
 * which showed up as an inspected frame that would not render while the golden
 * reference beside it, loaded plainly, did.
 */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${src}`));
    img.src = src;
  });
}

/**
 * Black-and-red ramp: red is attention and nothing else. A confident detection
 * is full red, a middling one a dimmer red, a weak one plain grey — never a hue.
 */
function severityColor(score: number): string {
  if (score >= 0.75) return "#ff2d2d";
  if (score >= 0.5) return "#c11414";
  return "#9a9a9a";
}

/** The canvas cannot resolve var(--mono), so read the stack off the document. */
function monoStack(): string {
  const declared = getComputedStyle(document.documentElement)
    .getPropertyValue("--mono")
    .trim();
  return declared || "monospace";
}

/**
 * Draws the frame, a translucent heatmap, and a crisp outlined box per detected
 * region with its severity. Rendered on a canvas so the overlay is exported with
 * the frame rather than floating over it in the DOM.
 */
export function OverlayCanvas({ imageUrl, heatmapUrl, regions, heatOpacity = 1 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showHeat, setShowHeat] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const base = await loadImage(imageUrl);
        const heat = showHeat && heatmapUrl ? await loadImage(heatmapUrl) : null;
        if (cancelled) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = base.naturalWidth;
        canvas.height = base.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(base, 0, 0);

        if (heat) {
          ctx.globalAlpha = heatOpacity;
          ctx.drawImage(heat, 0, 0, canvas.width, canvas.height);
          ctx.globalAlpha = 1;
        }

        if (!showBoxes) return;
        const scale = Math.max(canvas.width, canvas.height) / 640;
        const lineWidth = Math.max(2, 3 * scale);
        const fontSize = Math.max(12, 15 * scale);
        const mono = monoStack();

        regions.forEach((r, i) => {
          const colour = severityColor(r.score);
          ctx.lineWidth = lineWidth;
          ctx.strokeStyle = colour;
          ctx.setLineDash([]);
          ctx.strokeRect(r.x, r.y, r.w, r.h);

          // a dark halo so the outline reads against both bright and dark pixels
          ctx.lineWidth = lineWidth / 3;
          ctx.strokeStyle = "rgba(0,0,0,0.65)";
          ctx.strokeRect(r.x - lineWidth / 2, r.y - lineWidth / 2, r.w + lineWidth, r.h + lineWidth);

          const label = `#${i + 1} ${(r.score * 100).toFixed(0)}%`;
          ctx.font = `700 ${fontSize}px ${mono}`;
          const metrics = ctx.measureText(label);
          const padX = 6 * scale;
          const boxH = fontSize + 8 * scale;
          const boxW = metrics.width + padX * 2;
          // keep the chip inside the frame rather than letting it run off an edge
          const labelX = Math.min(Math.max(r.x - lineWidth / 2, 0), canvas.width - boxW);
          const labelY = r.y - boxH < 0 ? Math.min(r.y + r.h, canvas.height - boxH) : r.y - boxH;

          // the chip is always red on black: it marks a thing that wants looking at
          ctx.fillStyle = "#ff2d2d";
          ctx.fillRect(labelX, labelY, boxW, boxH);
          ctx.fillStyle = "#000000";
          ctx.fillText(label, labelX + padX, labelY + fontSize);
        });
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "render failed");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [imageUrl, heatmapUrl, regions, showHeat, showBoxes, heatOpacity]);

  return (
    <figure>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          border: "1px solid var(--line)",
          background: "var(--surface-2)",
        }}
      />
      <figcaption className="row tiny mono" style={{ marginTop: "var(--step)" }}>
        {heatmapUrl ? (
          <label className="row" style={{ gap: "6px" }}>
            <input
              type="checkbox"
              checked={showHeat}
              onChange={(e) => setShowHeat(e.target.checked)}
            />
            Heatmap
          </label>
        ) : null}
        <label className="row" style={{ gap: "6px" }}>
          <input
            type="checkbox"
            checked={showBoxes}
            onChange={(e) => setShowBoxes(e.target.checked)}
          />
          Regions ({regions.length})
        </label>
        {error ? <span className="red">{error}</span> : null}
      </figcaption>
    </figure>
  );
}

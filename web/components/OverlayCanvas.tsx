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

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${src}`));
    img.src = src;
  });
}

function severityColor(score: number): string {
  if (score >= 0.75) return "#ef4444";
  if (score >= 0.5) return "#f59e0b";
  return "#38bdf8";
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

        regions.forEach((r, i) => {
          const colour = severityColor(r.score);
          ctx.lineWidth = lineWidth;
          ctx.strokeStyle = colour;
          ctx.setLineDash([]);
          ctx.strokeRect(r.x, r.y, r.w, r.h);

          // a soft halo so the outline reads against both bright and dark pixels
          ctx.lineWidth = lineWidth / 3;
          ctx.strokeStyle = "rgba(0,0,0,0.65)";
          ctx.strokeRect(r.x - lineWidth / 2, r.y - lineWidth / 2, r.w + lineWidth, r.h + lineWidth);

          const label = `#${i + 1} ${(r.score * 100).toFixed(0)}%`;
          ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
          const metrics = ctx.measureText(label);
          const padX = 6 * scale;
          const boxH = fontSize + 8 * scale;
          const boxW = metrics.width + padX * 2;
          // keep the chip inside the frame rather than letting it run off an edge
          const labelX = Math.min(Math.max(r.x - lineWidth / 2, 0), canvas.width - boxW);
          const labelY = r.y - boxH < 0 ? Math.min(r.y + r.h, canvas.height - boxH) : r.y - boxH;

          ctx.fillStyle = colour;
          ctx.fillRect(labelX, labelY, boxW, boxH);
          ctx.fillStyle = "#0b0f14";
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
    <figure className="m-0">
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg"
        style={{ border: "1px solid var(--border)", background: "var(--surface-2)" }}
      />
      <figcaption className="mt-2 flex flex-wrap items-center gap-3 text-xs">
        {heatmapUrl ? (
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={showHeat}
              onChange={(e) => setShowHeat(e.target.checked)}
            />
            Heatmap
          </label>
        ) : null}
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={showBoxes}
            onChange={(e) => setShowBoxes(e.target.checked)}
          />
          Regions ({regions.length})
        </label>
        {error ? <span style={{ color: "var(--danger)" }}>{error}</span> : null}
      </figcaption>
    </figure>
  );
}

import type { AgentStep, InspectResult, ModelsResponse, TrendsResponse } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Turn a `/image/<id>` path returned by the API into an absolute URL. */
export function imageUrl(path: string | null): string | null {
  if (!path) return null;
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    throw new Error(`${path} responded ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as T;
}

export function fetchModels(): Promise<ModelsResponse> {
  return getJson<ModelsResponse>("/models");
}

export function fetchTrends(days = 30): Promise<TrendsResponse> {
  return getJson<TrendsResponse>(`/trends?days=${days}`);
}

export interface StreamHandlers {
  onStep: (step: AgentStep) => void;
  onResult: (result: InspectResult) => void;
  onError: (message: string) => void;
}

/**
 * POST a frame to /inspect/stream and surface each agent step as it lands.
 * Parses the SSE frames directly so the caller sees real progress, not a spinner.
 */
export async function inspectStream(
  file: File,
  assetClass: string | null,
  handlers: StreamHandlers,
): Promise<void> {
  const form = new FormData();
  form.append("image", file);
  if (assetClass) form.append("asset_class", assetClass);

  const res = await fetch(`${API_BASE}/inspect/stream`, { method: "POST", body: form });
  if (!res.ok || !res.body) {
    handlers.onError(`Inspection failed (${res.status})`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const eventLine = frame.split("\n").find((l) => l.startsWith("event: "));
      const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!eventLine || !dataLine) continue;
      const event = eventLine.slice(7).trim();
      const payload: unknown = JSON.parse(dataLine.slice(6));

      if (event === "step") handlers.onStep(payload as AgentStep);
      else if (event === "result") handlers.onResult(payload as InspectResult);
      else if (event === "error") handlers.onError((payload as { message: string }).message);
    }
  }
}

export async function coldStart(
  references: File[],
  assetClass: string,
  frame: File | null,
): Promise<InspectResult> {
  const form = new FormData();
  for (const ref of references) form.append("references", ref);
  form.append("asset_class", assetClass);
  if (frame) form.append("frame", frame);

  const res = await fetch(`${API_BASE}/coldstart`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Cold start failed (${res.status}): ${await res.text()}`);
  return (await res.json()) as InspectResult;
}

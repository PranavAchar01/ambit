import { API_BASE } from "./api";

/**
 * Browser transport for the voice agent.
 *
 * Two rules shape everything here. First, the browser never holds
 * `OPENAI_API_KEY` -- it asks our FastAPI for an ephemeral `ek_` client secret
 * whose scope is realtime-only. Second, tools do not run here: when the model
 * emits a function call we POST it to `/voice/tool`, which queries MongoDB, and
 * hand the rows back over the data channel. The model is the mouth; Atlas is the
 * memory.
 */

const REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls";

export interface VoiceToolSchema {
  type: "function";
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface VoiceSessionPayload {
  client_secret: string;
  expires_at: number;
  model: string;
  voice: string;
  tools: VoiceToolSchema[];
}

export interface VoiceToolResponse {
  name: string;
  arguments: Record<string, unknown>;
  latency_ms: number;
  result: unknown;
}

export type VoiceStatus =
  | "idle"
  | "minting"
  | "negotiating"
  | "live"
  | "closed"
  | "error";

export interface ToolCallRequest {
  callId: string;
  name: string;
  args: Record<string, unknown>;
}

export interface VoiceHandlers {
  onStatus: (status: VoiceStatus, detail?: string) => void;
  onUserTranscript: (text: string) => void;
  onAssistantDelta: (itemId: string, delta: string) => void;
  onAssistantDone: (itemId: string, text: string) => void;
  onToolStart: (call: ToolCallRequest) => void;
  onToolResult: (callId: string, response: VoiceToolResponse) => void;
  onToolError: (callId: string, message: string) => void;
  onError: (message: string) => void;
}

export interface VoiceConnection {
  readonly model: string;
  readonly voice: string;
  readonly tools: VoiceToolSchema[];
  /** False when the operator refused microphone access; text input still works. */
  readonly micAvailable: boolean;
  setMicEnabled: (enabled: boolean) => void;
  sendText: (text: string) => void;
  close: () => void;
}

/** Mint an ephemeral, realtime-scoped credential. Never returns the account key. */
export async function fetchVoiceSession(): Promise<VoiceSessionPayload> {
  const res = await fetch(`${API_BASE}/voice/session`, { method: "POST", cache: "no-store" });
  if (!res.ok) {
    throw new Error(`could not mint a voice credential (${res.status}): ${await res.text()}`);
  }
  return (await res.json()) as VoiceSessionPayload;
}

/** Execute one tool server-side, against MongoDB. */
export async function runVoiceTool(
  name: string,
  args: Record<string, unknown>,
): Promise<VoiceToolResponse> {
  const res = await fetch(`${API_BASE}/voice/tool`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, arguments: args }),
  });
  const payload: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = isRecord(payload) && typeof payload.detail === "string" ? payload.detail : res.statusText;
    throw new Error(detail);
  }
  return payload as VoiceToolResponse;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(source: Record<string, unknown>, key: string): string | null {
  const value = source[key];
  return typeof value === "string" ? value : null;
}

function parseArguments(raw: string | null): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** Transcript deltas arrive under two event names across API revisions. */
const ASSISTANT_DELTA_EVENTS = new Set([
  "response.output_audio_transcript.delta",
  "response.audio_transcript.delta",
  "response.output_text.delta",
  "response.text.delta",
]);

const ASSISTANT_DONE_EVENTS = new Set([
  "response.output_audio_transcript.done",
  "response.audio_transcript.done",
  "response.output_text.done",
  "response.text.done",
]);

/**
 * Open a WebRTC session against the GA Realtime API and wire the tool loop.
 *
 * The mic is optional on purpose: a denied permission degrades to a receive-only
 * audio transceiver, so the operator can still type a question and hear the
 * answer instead of hitting a dead end.
 */
export async function connectVoice(
  audioEl: HTMLAudioElement,
  handlers: VoiceHandlers,
): Promise<VoiceConnection> {
  handlers.onStatus("minting");
  const session = await fetchVoiceSession();

  const pc = new RTCPeerConnection();
  let micTrack: MediaStreamTrack | null = null;
  let micStream: MediaStream | null = null;

  pc.ontrack = (event) => {
    const [stream] = event.streams;
    if (stream) audioEl.srcObject = stream;
  };

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micTrack = micStream.getAudioTracks()[0] ?? null;
    if (micTrack) pc.addTrack(micTrack, micStream);
  } catch (err) {
    handlers.onError(
      err instanceof Error && err.name === "NotAllowedError"
        ? "Microphone blocked — you can still ask by typing, and the answer comes back as audio."
        : "No microphone available — falling back to typed questions.",
    );
  }
  if (!micTrack) pc.addTransceiver("audio", { direction: "recvonly" });

  const channel = pc.createDataChannel("oai-events");
  const transcripts = new Map<string, string>();

  const send = (payload: Record<string, unknown>): void => {
    if (channel.readyState === "open") channel.send(JSON.stringify(payload));
  };

  const handleToolCall = async (call: ToolCallRequest): Promise<void> => {
    handlers.onToolStart(call);
    let output: string;
    try {
      const response = await runVoiceTool(call.name, call.args);
      handlers.onToolResult(call.callId, response);
      output = JSON.stringify(response.result);
    } catch (err) {
      const message = err instanceof Error ? err.message : "tool failed";
      handlers.onToolError(call.callId, message);
      output = JSON.stringify({ error: message });
    }
    send({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.callId, output },
    });
    send({ type: "response.create" });
  };

  channel.onmessage = (event: MessageEvent<string>) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(event.data);
    } catch {
      return;
    }
    if (!isRecord(parsed)) return;
    const type = readString(parsed, "type");
    if (!type) return;

    if (type === "session.created") {
      handlers.onStatus("live");
      return;
    }
    if (type === "error") {
      const error = parsed.error;
      handlers.onError(
        isRecord(error) ? (readString(error, "message") ?? "realtime error") : "realtime error",
      );
      return;
    }
    if (type === "conversation.item.input_audio_transcription.completed") {
      const text = readString(parsed, "transcript");
      if (text && text.trim()) handlers.onUserTranscript(text.trim());
      return;
    }
    if (ASSISTANT_DELTA_EVENTS.has(type)) {
      const itemId = readString(parsed, "item_id") ?? readString(parsed, "response_id") ?? "current";
      const delta = readString(parsed, "delta") ?? "";
      transcripts.set(itemId, (transcripts.get(itemId) ?? "") + delta);
      handlers.onAssistantDelta(itemId, delta);
      return;
    }
    if (ASSISTANT_DONE_EVENTS.has(type)) {
      const itemId = readString(parsed, "item_id") ?? readString(parsed, "response_id") ?? "current";
      const text = readString(parsed, "transcript") ?? readString(parsed, "text") ?? transcripts.get(itemId) ?? "";
      transcripts.delete(itemId);
      handlers.onAssistantDone(itemId, text);
      return;
    }
    if (type === "response.function_call_arguments.done") {
      const callId = readString(parsed, "call_id");
      const name = readString(parsed, "name");
      if (!callId || !name) return;
      void handleToolCall({ callId, name, args: parseArguments(readString(parsed, "arguments")) });
    }
  };

  pc.onconnectionstatechange = () => {
    if (pc.connectionState === "failed") handlers.onStatus("error", "the peer connection dropped");
    else if (pc.connectionState === "closed") handlers.onStatus("closed");
  };

  handlers.onStatus("negotiating");
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const answer = await fetch(`${REALTIME_CALLS_URL}?model=${encodeURIComponent(session.model)}`, {
    method: "POST",
    body: offer.sdp ?? "",
    headers: {
      Authorization: `Bearer ${session.client_secret}`,
      "Content-Type": "application/sdp",
    },
  });
  if (!answer.ok) {
    pc.close();
    micStream?.getTracks().forEach((t) => t.stop());
    throw new Error(`Realtime call rejected (${answer.status}): ${await answer.text()}`);
  }
  await pc.setRemoteDescription({ type: "answer", sdp: await answer.text() });

  return {
    model: session.model,
    voice: session.voice,
    tools: session.tools,
    micAvailable: micTrack !== null,
    setMicEnabled: (enabled: boolean) => {
      if (micTrack) micTrack.enabled = enabled;
    },
    sendText: (text: string) => {
      send({
        type: "conversation.item.create",
        item: { type: "message", role: "user", content: [{ type: "input_text", text }] },
      });
      send({ type: "response.create" });
    },
    close: () => {
      channel.close();
      pc.close();
      micStream?.getTracks().forEach((t) => t.stop());
      audioEl.srcObject = null;
      handlers.onStatus("closed");
    },
  };
}

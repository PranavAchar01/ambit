import { VoiceConsole } from "@/components/VoiceConsole";

export const metadata = {
  title: "Voice — GridSight",
  description:
    "Ask the fleet a question out loud. Every answer is read out of MongoDB by a server-side tool, never from the model's priors.",
};

export default function VoicePage() {
  return (
    <div className="stack-lg">
      <header className="stack">
        <h1>Voice</h1>
        <p className="lede small">
          A realtime analyst wired to the same collections the agent writes to. It has no knowledge of
          this fleet: fleet health comes from an aggregation over <span className="mono">findings</span>,
          &ldquo;have we seen this before?&rdquo; is a vector search over{" "}
          <span className="mono">finding_recall_idx</span>, and a refusal is explained from the
          candidate scores and coverage gates recorded at decision time. Watch the tool column —
          that is the persistent context being read.
        </p>
        <p className="lede tiny">
          The browser holds a short-lived, realtime-scoped credential minted by the API; the OpenAI
          account key never leaves the server, and tools run there too.
        </p>
      </header>

      <VoiceConsole />
    </div>
  );
}

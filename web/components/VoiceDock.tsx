"use client";

import { useCallback, useEffect, useState } from "react";
import { VoiceConsole } from "./VoiceConsole";

/**
 * The voice analyst, docked instead of routed.
 *
 * It used to be a page, which meant leaving the inspection screen to ask a
 * question about the frame still on it -- and, because App Router unmounts a
 * route on navigation, tearing down the WebRTC session every time. Mounted in
 * the layout it survives navigation, and the operator can ask about a verdict
 * while looking at it.
 *
 * `VoiceConsole` is never unmounted, only hidden. Collapsing is a `display`
 * change, so the `<audio>` element the connection writes into stays in the DOM
 * and a live session keeps talking while the panel is shut -- `connect()` bails
 * early if that element is missing, so conditionally rendering it would break
 * the session rather than merely hide it.
 */
export function VoiceDock() {
  const [open, setOpen] = useState(false);

  // The panel is fixed to the bottom-right, where a verdict card would
  // otherwise sit under it. Padding the page while open keeps the two apart
  // instead of relying on the operator to scroll.
  useEffect(() => {
    document.body.dataset.dock = open ? "open" : "closed";
    return () => {
      delete document.body.dataset.dock;
    };
  }, [open]);

  const toggle = useCallback(() => setOpen((v) => !v), []);

  return (
    <div className="dock">
      <div className="dock-panel" data-open={open ? "true" : "false"} aria-hidden={!open}>
        <div className="dock-hd row-between">
          <h2 className="caps">Voice analyst</h2>
          <button type="button" className="btn-link" onClick={toggle} aria-label="Collapse voice panel">
            Close
          </button>
        </div>
        <p className="tiny muted dock-note">
          Every answer is read out of MongoDB by a server-side tool, never from the model&apos;s
          priors — fleet health is an aggregation over <span className="mono">findings</span>,
          &ldquo;have we seen this before?&rdquo; is a vector search over{" "}
          <span className="mono">finding_recall_idx</span>, and a refusal is explained from the
          candidate scores recorded at decision time. The browser holds a short-lived,
          realtime-scoped credential minted by the API; the account key never leaves the server.
        </p>
        <VoiceConsole />
      </div>

      <button
        type="button"
        className="dock-toggle"
        onClick={toggle}
        aria-expanded={open}
        aria-label={open ? "Collapse the voice analyst" : "Ask the voice analyst"}
      >
        <span aria-hidden>{open ? "×" : "◉"}</span>
      </button>
    </div>
  );
}

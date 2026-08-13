"""A dumb fan-out relay for the phone viewfinder.

The phone publishes cheap 480x360 preview frames; the projected laptop view
subscribes and watches. Nothing here inspects, decodes, or stores anything --
inference stays on the ordinary POST /inspect path at full resolution.

Two delivery rules, and they differ on purpose:

* Frames are *latest-wins*. A viewfinder that queues is a viewfinder that
  drifts: if a viewer's socket stalls for a second at 8fps, replaying those
  eight stale frames is strictly worse than showing the newest one. Old frames
  are dropped, and the drop count is reported so the staleness is visible
  rather than silent.
* Messages (agent steps, verdicts, phase changes) are *ordered and buffered*.
  Dropping the step that says "refused" would be a lie, so these queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any

from starlette.websockets import WebSocket

log = logging.getLogger("gridsight.relay")

# One viewer that cannot keep up should not be able to grow memory without
# bound. Steps arrive a few per inspection, so this is never reached in
# practice -- it is a backstop, not a tuning knob.
MAX_QUEUED_MESSAGES = 64


class Viewer:
    """A single subscribed socket, with its own independent send loop."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._frame: bytes | None = None
        self._messages: deque[str] = deque(maxlen=MAX_QUEUED_MESSAGES)
        self._wake = asyncio.Event()
        self.dropped_frames = 0
        self.sent_frames = 0

    @property
    def has_pending_frame(self) -> bool:
        return self._frame is not None

    def offer_frame(self, data: bytes) -> None:
        if self._frame is not None:
            self.dropped_frames += 1
        self._frame = data
        self._wake.set()

    def offer_message(self, text: str) -> None:
        self._messages.append(text)
        self._wake.set()

    async def pump(self) -> None:
        """Drain to the socket forever. Messages first, then the newest frame."""
        while True:
            await self._wake.wait()
            self._wake.clear()

            # Ordered before frames: a verdict should never sit behind a JPEG.
            while self._messages:
                await self.ws.send_text(self._messages.popleft())

            frame, self._frame = self._frame, None
            if frame is not None:
                await self.ws.send_bytes(frame)
                self.sent_frames += 1


class Session:
    """One phone paired with zero or more projected viewers."""

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.viewers: set[Viewer] = set()
        self.publisher_online = False
        self.frames_published = 0
        # Accumulated at session level: per-viewer counters vanish on disconnect,
        # which would report a clean 0 for a session that was visibly stuttering.
        self.frames_dropped = 0
        # Replayed to a viewer that joins mid-stream, so the projected panel
        # paints immediately instead of staying black until the next frame.
        self.last_frame: bytes | None = None
        self.last_status: str | None = None


class Relay:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def session(self, session_id: str) -> Session:
        return self._sessions.setdefault(session_id, Session(session_id))

    def publish_frame(self, session_id: str, data: bytes) -> None:
        sess = self.session(session_id)
        sess.frames_published += 1
        sess.last_frame = data
        for viewer in sess.viewers:
            if viewer.has_pending_frame:
                sess.frames_dropped += 1
            viewer.offer_frame(data)

    def publish_message(self, session_id: str, payload: dict[str, Any]) -> None:
        sess = self.session(session_id)
        text = json.dumps(payload, default=str)
        if payload.get("type") == "phase":
            sess.last_status = text
        for viewer in sess.viewers:
            viewer.offer_message(text)

    def add_viewer(self, session_id: str, ws: WebSocket) -> Viewer:
        sess = self.session(session_id)
        viewer = Viewer(ws)
        sess.viewers.add(viewer)
        # Prime the newcomer with whatever the phone last sent.
        if sess.last_status:
            viewer.offer_message(sess.last_status)
        if sess.last_frame:
            viewer.offer_frame(sess.last_frame)
        viewer.offer_message(json.dumps({"type": "presence", "publisher_online": sess.publisher_online}))
        return viewer

    def remove_viewer(self, session_id: str, viewer: Viewer) -> None:
        self.session(session_id).viewers.discard(viewer)

    def set_publisher(self, session_id: str, online: bool) -> None:
        sess = self.session(session_id)
        sess.publisher_online = online
        if not online:
            sess.last_frame = None
        self.publish_message(session_id, {"type": "presence", "publisher_online": online})

    def stats(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.id,
                "publisher_online": s.publisher_online,
                "viewers": len(s.viewers),
                "frames_published": s.frames_published,
                "frames_dropped": s.frames_dropped,
            }
            for s in self._sessions.values()
        ]


relay = Relay()

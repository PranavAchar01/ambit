#!/usr/bin/env bash
# Expose the GridSight API over HTTPS so a phone can reach /capture.
#
# One tunnel is enough: the capture page, the relay WebSocket and the /inspect
# upload are all served by the API on the same origin, so there is no CORS to
# configure and no second hostname to plumb through the web app.
#
# getUserMedia refuses to run outside a secure context, and on iOS Safari it
# fails *silently* -- no prompt, no error. http://<lan-ip>:8000 will look like a
# broken camera. That is the entire reason this script exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8000}"
SESSION="${2:-demo}"
RUNTIME="$ROOT/runtime"
LOG="$RUNTIME/cloudflared.log"
STATE="$RUNTIME/tunnel.json"

command -v cloudflared >/dev/null || { echo "cloudflared not found: brew install cloudflared"; exit 1; }

# Bind to loopback IPv4 explicitly. "localhost" resolves to ::1 first on macOS,
# and anything else holding [::]:PORT -- a stray Docker container, say -- would
# silently be the thing published to a public URL instead of GridSight.
HOST="127.0.0.1"
TARGET="http://$HOST:$PORT"

# A bare 200 on /health proves nothing: plenty of services answer {"status":"ok"}.
# Check for a field only GridSight serves before exposing this to the internet.
if ! curl -sf -m 5 "$TARGET/health" | grep -q '"vector_index"'; then
  echo "Refusing to tunnel: $TARGET is not the GridSight API."
  echo "Start it first:  .venv/bin/python -m uvicorn gridsight.api.main:app --host $HOST --port $PORT"
  echo "If something else owns port $PORT, pass a free one:  scripts/tunnel.sh 8010"
  exit 1
fi

mkdir -p "$RUNTIME"
: > "$LOG"

cleanup() {
  if [[ -n "${PID:-}" ]]; then kill "$PID" 2>/dev/null || true; fi
  rm -f "$STATE"
}
trap cleanup EXIT INT TERM

cloudflared tunnel --url "$TARGET" --no-autoupdate >"$LOG" 2>&1 &
PID=$!

URL=""
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [[ -n "$URL" ]] && break
  if ! kill -0 "$PID" 2>/dev/null; then echo "cloudflared exited early; see $LOG"; exit 1; fi
  sleep 1
done

if [[ -z "$URL" ]]; then
  echo "tunnel did not report a hostname within 60s; see $LOG"
  exit 1
fi

# The API reads this per request, so restarting the tunnel never means
# restarting the API -- quick tunnels hand out a new hostname every time.
printf '{"public_url":"%s"}\n' "$URL" > "$STATE"

cat <<BANNER

  tunnel up      $URL  ->  $TARGET
  phone          $URL/capture?s=$SESSION
  projected      http://localhost:3000  (scan the QR on that page)

  Leave this running. Ctrl-C tears the tunnel down.

BANNER

wait "$PID"

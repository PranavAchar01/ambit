"""Prove the training loop survives a hard kill.

Starts a fleet training run in its own process session, waits until exactly two
asset classes have been committed to the MongoDB checkpoint, sends SIGKILL while
the third is mid-fit, and reports the surviving registry + checkpoint state.
The resume itself is a separate invocation (`train_class --resume`) so the two
halves are independently verifiable.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "artifacts" / "kill_resume" / "run1.log"
CLASSES = ["conductor", "corrosion", "insulator", "transmission_tower", "vegetation"]
THREAD_ID = "fleet-v1"


def reset() -> None:
    from gridsight.db.mongo import ensure_collections, get_db

    db = get_db()
    for name in (
        "models",
        "checkpoints",
        "checkpoint_writes",
        "weights.files",
        "weights.chunks",
        "images.files",
        "images.chunks",
    ):
        if name in db.list_collection_names():
            db[name].drop()
    ensure_collections()
    print("[reset] registry, checkpoints and GridFS cleared", flush=True)


def committed_classes() -> list[str]:
    from gridsight.db.mongo import models_col

    return sorted(d["asset_class"] for d in models_col().find({}, {"asset_class": 1}))


def checkpoint_state() -> dict[str, object]:
    from gridsight.train.train_class import build_training_graph, make_saver

    app = build_training_graph(make_saver())
    snap = app.get_state({"configurable": {"thread_id": THREAD_ID}})
    return {
        "completed": snap.values.get("completed", []),
        "pending": snap.values.get("pending", []),
        "next": list(snap.next),
    }


def main() -> int:
    reset()
    LOG.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "gridsight.train.train_class",
        "--classes",
        *CLASSES,
        "--thread-id",
        THREAD_ID,
        "--skip-index",
    ]
    with LOG.open("wb") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[start] pid={proc.pid} training {CLASSES}", flush=True)

    deadline = time.time() + 900
    killed_at: str | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[error] process exited on its own with code {proc.returncode}", flush=True)
            return 1
        done = committed_classes()
        if len(done) >= 2:
            time.sleep(15)  # let the third class get well into its fit
            print(f"[kill] {len(done)} classes committed ({done}); sending SIGKILL to {proc.pid}", flush=True)
            os.kill(proc.pid, signal.SIGKILL)
            killed_at = time.strftime("%H:%M:%S")
            break
        time.sleep(3)

    if killed_at is None:
        print("[error] never reached two committed classes", flush=True)
        proc.kill()
        return 1

    proc.wait(timeout=30)
    print(
        f"[kill] delivered at {killed_at}; wait() reported {proc.returncode} "
        f"({'SIGKILL' if proc.returncode == -signal.SIGKILL else 'other'})",
        flush=True,
    )

    alive = True
    try:
        os.kill(proc.pid, 0)
    except ProcessLookupError:
        alive = False
    print(f"[verify] pid {proc.pid} alive after kill: {alive}", flush=True)

    survived = committed_classes()
    state = checkpoint_state()
    print(f"[state] models surviving in registry : {survived}", flush=True)
    print(f"[state] checkpoint completed         : {state['completed']}", flush=True)
    print(f"[state] checkpoint pending           : {state['pending']}", flush=True)
    print(f"[state] graph resumes at node        : {state['next']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

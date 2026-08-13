# Ambit

**A registry of inspection competence — and a router that knows when to refuse.**

The most dangerous thing a quality-control model can do is confidently pass a board it has never
seen. Vision models that find defects already work; what does not exist is anything that decides
**which model to run, and whether any of them should run at all.**

Ambit is that decider. MongoDB Atlas holds a registry of anomaly-detection specialists, each model
document carrying a 512-d embedding of the imagery it was trained on and a **coverage gate** — the
envelope it demonstrably learned. An agent embeds an incoming frame, `$vectorSearch`es the registry,
loads the winning specialist straight out of GridFS, and runs it. If nothing clears the winner's
gate, the agent **refuses to guess**, says so, and can cold-start a new specialist from a handful of
known-good references — which then lives in the registry for every future board.

The name is the thesis: an *ambit* is the bounds of what something covers.

---

## What was built during the event

Ambit is not a from-scratch project. The registry, the router, the coverage gates, the agent, the
API and the UI pre-date this event, and the table below says so line by line. Everything marked
**Built during event** is new work; everything marked **Pre-existing** was already running when the
event started, at commit `6903de1`.

| Component | Status |
|---|---|
| Vector-routed model registry on Atlas (`$vectorSearch` over per-model CLIP centroids) | Pre-existing |
| Per-model coverage gates and the abstention rule | Pre-existing |
| LangGraph agent, checkpointed to MongoDB, resumable by `thread_id` | Pre-existing |
| Few-shot cold-start from ~8 references | Pre-existing |
| PatchCore weights in GridFS; exact round-trip verification | Pre-existing |
| Episodic recall (`finding_recall_idx`, `POST /recall`) | Pre-existing |
| Registry animation / routing map driven by live SSE | Pre-existing |
| Voice agent over OpenAI Realtime, MongoDB-backed tools | Pre-existing |
| Electronics rescope: VisA PCB classes + MVTec transistor/cable, licences recorded | Pre-existing |
| Phone-camera capture: WebSocket relay, viewfinder, hold-still detection | Pre-existing |
| FastAPI endpoints, Next.js UI, two-colour design system | Pre-existing |
| **`arduino_uno` class from original photography** — local-directory ingest adapter, provenance with the split recorded by filename, held-out validation harness | **Built during event** |
| **Normal-only training path** — a class with no labelled defect set trains, and records `null` for both AUROCs instead of a fabricated `0.0` | **Built during event** |
| **Out-of-sample image-threshold calibration** — the threshold was an in-sample maximum; measured, every unseen normal frame exceeded it | **Built during event** |
| **Fireworks vision-language defect narration** behind a provider seam (Fireworks / local / null), with the crop pair, the sentinel and the degradation policy | **Built during event** |
| **`LLM_PROVIDER` switch** routing text-LLM calls through OpenAI or OpenRouter, with a deterministic refuse when neither key is present | **Built during event** |
| **Provider reporting** at `GET /health` and on startup | **Built during event** |
| **Main screen restructured** into capture / analysis columns; agent steps as discrete rows with per-step latency; verdict card rendering score-vs-gate as a comparison | **Built during event** |
| **Manual shutter and stability-detection toggle** on the phone capture page | **Built during event** |
| **Backfilled-timestamp disclosure** surfaced in `/trends` | **Built during event** |
| ElevenLabs spoken findings | **Not built** — see *Not done* |
| Projects / tenant grouping, registry and trends redesign | **Not built** — see *Not done* |

Nothing in the "Pre-existing" rows was re-presented as new, and nothing in the "Built during event"
rows existed before it. The git history is the audit trail: every event commit is after `6903de1`.

---

## The result that matters

Five specialists are registered. A sixth board, `visa_pcb4`, is deliberately withheld — the registry
has never seen it. Measured over 125 in-registry and 25 out-of-registry held-out frames:

| decision rule | in-registry accepted | out-of-registry refused |
|---|---|---|
| global threshold 0.82 | 125/125 | **0/25** |
| global threshold 0.86 | 125/125 | **0/25** |
| global threshold 0.88 | 125/125 | **0/25** |
| **per-model coverage gate** | 119/125 | **25/25** |

The withheld board scores **0.9399 mean / 0.9525 max** against the registry — *above every global
threshold tested*. **No fixed cut point can refuse it at all**, because it genuinely *is* a PCB and
the registry is full of PCBs. Only the per-model gate declines it, 25 times out of 25.

That is the whole argument in one table. The system knows it is looking at a PCB-like object and
still declines, because no specialist has competence over *that particular board* — which is exactly
what happens when a hardware team spins Rev C on Tuesday.

The refusal is a **near miss**, not an obvious mismatch: `visa_pcb4` scores 0.9428 against
`visa_pcb3`, landing inside the 0.04 ambiguous band below that model's gate, so the agent escalates
to OpenAI with the top-5 candidate metadata and **OpenAI adjudicates the refusal**. The trace reads
*"Ambiguous at 0.9428; openai refused to route"*.

Cost of that refusal, stated plainly: 6 of 125 in-registry frames are refused too. At prototype
volume, re-shooting six frames is cheaper than passing one uninspected board.

**Top-1 routing is 125/125 (100%)** — including `pcb1` and `pcb2`, which turn out to be the two faces
of the same HC-SR04 board (0.8797 centroid similarity, the closest pair in the registry) and still
separate 25/25.

---

## Why Atlas, not a local mongod

`$vectorSearch` is an Atlas-only aggregation stage. A self-hosted `mongod` does not implement it, so
the vector index that *is* the premise of this project cannot be built or queried anywhere else —
routing a frame is a k-NN query over 512-d CLIP centroids. Three more things live in the same
cluster for the same reason: PatchCore weights in GridFS (a coreset is ~7 MB, so the registry entry
and the artifact it points at never drift apart), LangGraph checkpoints in `checkpoints` /
`checkpoint_writes` so a `kill -9` mid-training resumes instead of relearning, and findings
alongside both so fleet-health aggregations run where the data already is.

There *is* an offline fallback — if the index is missing, `PENDING`, or the aggregation errors,
`vector_search()` degrades to brute-force cosine over every registry document — but it logs
`ATLAS VECTOR SEARCH UNAVAILABLE … DEGRADING TO BRUTE-FORCE COSINE` at ERROR on every call, and
`/health` reports index status, because a silent fallback makes a broken deployment look healthy.

Atlas usage: **62 MB / 512 MB** (12.1%) on an M0.

---

## Architecture

```mermaid
flowchart TB
    PH["Phone /capture<br/>viewfinder · hold-still"] -->|"JPEG @30fps (wss)"| RLY["relay<br/>latest-wins fan-out"]
    RLY --> UI
    PH -->|"full-res PNG"| API
    UI["Next.js UI<br/>projected view · overlay · trends"] -->|"multipart + SSE"| API["FastAPI<br/>/inspect · /coldstart · /trends"]
    API --> G

    subgraph G["LangGraph agent (checkpointed per node)"]
        direction TB
        E["embed_frame<br/>OpenCLIP ViT-B-32 → 512-d"] --> R["route<br/>$vectorSearch k=5"]
        R --> D{"decide<br/>score ≥ model's gate?"}
        D -->|yes| I["infer<br/>PatchCore from GridFS"]
        D -->|"ambiguous band"| L["OpenAI adjudicates<br/>(structured output)"]
        L --> I
        D -->|"below gate"| C["cold_start<br/>fit few-shot PatchCore"]
        C --> I
        I --> N["narrate<br/>OpenAI writes the finding"]
        N --> P["persist"]
    end

    G <--> M
    subgraph M["MongoDB Atlas"]
        direction LR
        MOD[("models<br/>+ model_router_idx<br/>512-d cosine")]
        WGT[("GridFS: weights")]
        IMG[("GridFS: images")]
        FND[("findings<br/>+ finding_recall_idx")]
        CKP[("checkpoints")]
        DS[("datasets")]
    end
```

The routing embedding is deliberately **not** an OpenAI embedding: OpenAI's embedding endpoints are
text-only and cannot embed an inspection frame. Routing uses local OpenCLIP; OpenAI is used only for
reasoning — adjudicating ambiguous routes, naming new models, and writing findings.

### How routing actually decides

A single global threshold cannot work, and the measurements say so. Atlas scores cosine similarity as
`(1 + cos) / 2`, compressing every natural image into roughly `[0.75, 0.96]`, so one cut point cannot
both accept in-registry frames and reject out-of-registry ones. Instead **each model stores its own
coverage gate** — a low percentile of its own held-out training frames' similarity to its own
centroid, floored by a minimum margin. A frame routes only if it clears the winning model's gate.

Inter-model centroid similarity (minimum gate in the registry is 0.9629):

| | cable | transistor | pcb1 | pcb2 | pcb3 |
|---|---|---|---|---|---|
| **mvtec_cable** | 1.0000 | 0.7281 | 0.6711 | 0.6723 | 0.6494 |
| **mvtec_transistor** | 0.7281 | 1.0000 | 0.7289 | 0.7716 | 0.7547 |
| **visa_pcb1** | 0.6711 | 0.7289 | 1.0000 | 0.8797 | 0.8626 |
| **visa_pcb2** | 0.6723 | 0.7716 | 0.8797 | 1.0000 | 0.9077 |
| **visa_pcb3** | 0.6494 | 0.7547 | 0.8626 | 0.9077 | 1.0000 |

Every off-diagonal sits below every gate, so cross-class traffic is not merely unlikely — it is
inadmissible.

---

## The registry, and what licence each model carries

| class | source | licence | image AUROC | pixel AUROC | gate |
|---|---|---|---|---|---|
| `visa_pcb1` | VisA | **CC BY 4.0** | 0.8967 | 0.9873 | 0.9794 |
| `visa_pcb2` | VisA | **CC BY 4.0** | 0.8600 | 0.9717 | 0.9797 |
| `visa_pcb3` | VisA | **CC BY 4.0** | 0.9967 | 0.9755 | 0.9801 |
| `mvtec_transistor` | MVTec AD | CC BY-NC-SA 4.0 | 0.9967 | 0.9456 | 0.9692 |
| `mvtec_cable` | MVTec AD | CC BY-NC-SA 4.0 | 0.9933 | 0.9892 | 0.9629 |
| `visa_pcb4` | VisA | CC BY 4.0 | — **withheld** — | — | — |
| `arduino_uno` | **original photography** | **unrestricted** | **—** | **—** | 0.9806 |

Licence is a positioning constraint, not an appendix, and the registry UI states it per model:

- **VisA (`pcb1`–`pcb4`) is CC BY 4.0** — attribution only. The PCB classes carry the commercial story.
- **MVTec AD (`transistor`, `cable`) is CC BY-NC-SA 4.0** — non-commercial, and ShareAlike
  propagates. Demo, research and internal evaluation only; never in a shipped product.
- **`arduino_uno` is ours.** The frames were shot by the authors on a bench rig, so the specialist
  derived from them carries no upstream condition at all — neither VisA's attribution term nor
  MVTec's NonCommercial one.

Pixel AUROC is real for every corpus class, because both corpora ship per-defect ground-truth masks.

### Why `arduino_uno` has no AUROC, and why that is the correct value

Both metrics are `null`, the UI renders a dash, and the tooltip reads *"no labelled defect set —
normal-only training"*. That is not a gap in the ingest. **A prototype shop has no defect set**:
there is no library of known-bad Rev A boards to measure against, and there never will be, because
the whole point of the revision is that it is new. Ten known-good frames is the honest input.

The alternative was worse than useless. With a single label present, torchmetrics computes a binary
AUROC of `0.0`, warns, and returns — so the unguarded path writes a **fabricated** `0.0` into the
registry, where the UI renders it as a real measurement of a real model. `null` is the true value.

This is also the ICP argument in one row of a table: every other class in this registry needed
somebody else's labelled corpus, and this one needed a phone.

---

## Inspecting from a phone

`scripts/tunnel.sh` exposes the API over HTTPS and prints a QR. The phone opens `/capture`, streams a
480×360 viewfinder to the projected laptop view, and when the operator **holds still for two
seconds** it uploads one full-resolution frame to `/inspect` and relays the agent's steps back.

```bash
./scripts/tunnel.sh          # then scan the QR on http://localhost:3000
```

Three things worth knowing:

- **The tunnel is load-bearing, not convenience.** `getUserMedia` only exists in a secure context,
  and on iOS Safari it fails *silently* on a plain LAN address — no prompt, no error. The capture
  page is served by FastAPI rather than Next, so page, socket and upload share one origin: no CORS,
  no second hostname.
- **The inspected frame is PNG, and that is measured.** JPEG recompression costs 0.01–0.024 of
  routing score. Against `visa_pcb1`'s 0.9794 gate, one defect frame scores 0.9854 as PNG and routes,
  but 0.9662 at q92 and 0.9586 at q60 — refused both times. The compression artefacts, not the board,
  would have decided the verdict. Previews stay JPEG because they are throwaway.
- **Frames are latest-wins; messages are not.** A viewfinder that queues drifts, so a stalled viewer
  drops stale frames. Agent steps and verdicts queue in order, because dropping the step that says
  *refused* would be a lie. Sustains 60 fps end to end with zero drops.

Every registered model is trained on VisA/MVTec studio capture, so a phone photo of a real board will
usually come back `unroutable`. **That is the coverage gate working**, and cold-start is the answer.

---

## Setup

```bash
uv venv --python 3.12 && uv sync --all-groups
```

`.env` is git-ignored. See `.env.example` for the full list. `MONGODB_URI` is the only one that is
strictly required — Atlas is not optional, because `$vectorSearch` is Atlas-only. Everything else
selects a capability and degrades visibly when absent:

| variable | absent means |
|---|---|
| `MONGODB_URI` | **fatal.** `get_settings()` raises at import |
| `MONGODB_DB` | defaults to `gridsight` |
| `HF_TOKEN` | corpus ingest from the Hub is unauthenticated |
| `OPENAI_API_KEY` | ambiguous-band adjudication refuses deterministically; the voice agent is unavailable |
| `FIREWORKS_API_KEY` | no vision description; narration falls back to the structured sentence and records `narrative_source: "structured"` |
| `OPENROUTER_API_KEY` | only consulted when `LLM_PROVIDER=openrouter` or no OpenAI key is set |
| `LLM_PROVIDER` | `openai` when its key is present, else `openrouter`, else the deterministic refuse |
| `VLM_PROVIDER` | Fireworks if its key is present, else a locally cached vision model, else none |
| `VLM_MODEL` / `VLM_TIMEOUT_S` | the measured primary, and an 8 s budget |

`GET /health` reports which of these actually resolved, so a deployment that is quietly running
without a provider is visible rather than plausible.

```bash
uv run python scripts/ingest_visa.py      # VisA pcb1-pcb4 (CC BY 4.0)
uv run python scripts/ingest_mvtec.py     # MVTec transistor + cable (CC BY-NC-SA 4.0)
```

```bash
# create the Atlas vector index (polls to READY), then train and register.
# visa_pcb4 is deliberately withheld for the refusal demo.
uv run python -m gridsight.train.train_class \
  --classes visa_pcb1 visa_pcb2 visa_pcb3 mvtec_transistor mvtec_cable \
  --thread-id electronics-v1

# resume a run that died mid-training, instead of relearning what is registered
uv run python -m gridsight.train.train_class --resume --thread-id electronics-v1
```

```bash
uv run uvicorn gridsight.api.main:app --host 127.0.0.1 --port 8000   # API
pnpm --dir web install && pnpm --dir web dev                          # UI
```

Verification and evidence:

```bash
uv run python scripts/verify_routing.py      # live $vectorSearch + decision-rule comparison
uv run python scripts/agent_demo.py          # the agent scenarios, end to end
uv run python scripts/kill_resume_test.py    # SIGKILL mid-training, then --resume
uv run ruff check gridsight tests scripts && uv run mypy && uv run pytest && pnpm --dir web build
```

> `seed_findings.py --spread-days` runs **real** inference on real held-out frames; only the
> `timestamp` is shifted so the trends view has a timeline, and every such document is flagged
> `backfilled: true` with its true `observed_at`. Nothing else about a finding is synthetic.

---

## Collections

| collection | contents |
|---|---|
| `models` | registry: name, asset class, backbone, **512-d embedding**, `routing_threshold` (coverage gate), image/pixel thresholds, `weights_file_id`, `reference_image_id`, training samples, metrics, `created_by` (`seed` \| `agent-coldstart`), provenance |
| `findings` | one row per inspected frame: verdict, routed model, routing score, anomaly score, severity, bbox regions, heatmap id, narrative, latency — plus a 512-d embedding indexed by `finding_recall_idx` for episodic recall |
| `datasets` | ingest provenance: dataset id, licence, selection rationale, extraction notes, counts |
| `checkpoints` / `checkpoint_writes` | owned by the LangGraph MongoDB checkpointer |
| `weights.*` (GridFS) | serialized PatchCore coresets + post-processor thresholds |
| `images.*` (GridFS) | uploaded frames, heatmaps, golden reference exemplars |

A registry entry stores the **coreset memory bank and thresholds**, not the frozen ImageNet backbone,
which is reconstructed by name at load time. That keeps a model ~7 MB instead of ~250 MB.

---

## Demo

1. **The registry.** `/registry` — five specialists, each with its golden known-good exemplar, its
   coverage gate, its weights size, and its dataset licence. The badge reads *Vector index READY*.
2. **Routing works.** On `/`, drop `web/public/demo/pcb1_defect.png`. Watch the agent steps:
   *embedding → searching registry → routed to `visa_pcb1-patchcore-v1` at 0.9854 → running
   inference*. The heatmap and outlined regions land beside the registry's golden reference, with the
   full candidate table — every runner-up and the gate it failed.
3. **It refuses rather than guessing.** Drop `web/public/demo/pcb4_unroutable.png` — a board never
   trained on. It scores 0.9428, *above every global threshold*, and is still declined, adjudicated
   by OpenAI inside the ambiguous band.
4. **Cold start.** Give the inline uploader the 8 known-good frames in `web/public/demo/`
   (`pcb4_refs_*.png`), name the class, submit. It fits a few-shot PatchCore, has OpenAI name it,
   writes weights to GridFS, registers it, and re-runs the original frame against a model that did
   not exist a moment ago.
5. **Fleet health.** `/trends` — defect rate over time per class, severity distribution, verdict mix
   including refusals, and which part is failing fastest. All MongoDB aggregation pipelines.
6. **Ask it.** `/voice` — an OpenAI Realtime agent over five MongoDB-backed tools, including
   `explain_refusal` and `recall_similar` (episodic recall over past findings).

---

## The model providers, and what each one is for

Three seams, three different jobs. Each is chosen by key presence and reported at `GET /health`, so
a degraded provider is visible rather than silent — the same reasoning that makes the brute-force
vector-search fallback log at ERROR on every call.

**Fireworks — describing the defect.** PatchCore localises and scores; it has no language, so it
cannot say *what* is wrong. A defect frame is cropped to its highest-scoring region, the golden
reference is cropped to the *same* region, and both go to a vision model on Fireworks with one
question: what physically differs? The answer is specific — *"the silver rectangular component near
the top of the board appears slightly shifted or rotated … the solder joints on the pins at the
bottom edge look less uniform"* — where the numbers alone could only ever say `53.41 > 39.08`.

Cropping is load-bearing. A VLM given a whole board narrates the whole board and volunteers a defect
somewhere; given a region an anomaly detector has already localised, it describes what is there.

Verdict-critical work never touches the network: OpenCLIP embedding, `$vectorSearch` and PatchCore
inference are local or Atlas. The description is an outer layer with a hard timeout and a structured
fallback, and it is never called at all on a refusal — there is no reference to compare against, and
describing an unrecognised part is exactly the confabulation abstention exists to prevent.

**OpenRouter — adjudicating the ambiguous band.** A frame scoring within 0.04 below a model's gate is
not decided by the vector score alone; a structured-output call breaks the tie. `LLM_PROVIDER`
switches that call, and the model-naming call, between OpenAI and OpenRouter — a base URL and a model
string, with OpenRouter's `models` failover in the request body so one provider outage does not end a
demo. With **neither** key present the band resolves by the deterministic rule: **refuse**. That is
the correct default for a system whose whole claim is that it declines when it does not know, and it
means the demo runs on zero credentials.

**ElevenLabs — speaking the finding.** Not built; see *Not done*.

### Narration is written once and stored, not re-inferred

The description is produced at inspection time and persisted onto the finding — `agent_narrative`,
`narrative_source`, `vlm_component`, `vlm_difference_visible`, `vlm_latency_ms`. **The voice agent
reads it out of MongoDB and never calls the vision model itself.**

That is the persistent-context property, concretely: the expensive act of looking happens once, and
every later question — asked minutes later, from another process, by voice — is answered from a
stored fact rather than from a fresh inference that might say something slightly different. A system
that re-infers on every question does not have a memory; it has a habit.

---

## Honest limits

- **6 of 125 in-registry frames are refused.** The gate buys 25/25 abstention at that cost. Stated
  above rather than buried, because it is the real trade.
- **The registry is studio imagery.** VisA and MVTec are captured in fixed rigs. A phone photo of a
  real board under bench lighting will usually be refused — correct behaviour, but it means the
  live-capture demo needs a cold-start first.
- **A cold-started model's threshold comes from ~8 references**, so its normal envelope is tight and
  it flags aggressively until retrained on a fuller set. It is at least now calibrated on frames the
  memory bank was *not* fitted on: measured leave-one-out on a 10-frame capture, **all ten** genuinely
  normal frames scored above the old in-sample threshold once withheld (24.5–35.5 against 23.3). The
  correction costs two extra fits, which moved cold-start from ~3.3 s to ~9 s.
- **`arduino_uno`'s specialist is fitted on ten frames from one session** under one lighting
  condition, and its coverage gate is calibrated from a centroid fitted on two of them (the
  fit/holdout split floors at 8 held out). Routing survives it — 4/4 held-out frames route at
  0.9888–0.9961 against a gate of 0.9806 — but this is a thin class and it is stated as one.
- **No defect has ever been run through `arduino_uno`.** Its threshold bounds the normal side only;
  the other side of that boundary is unmeasured until a real bad board exists.
- **The vision description is not verified against ground truth.** It is a model's account of what
  differs between two crops. It is recorded with its provider and latency so it can be audited, but
  no claim is made that it is correct — only that it looked.
- **Decisions inside the 0.04 ambiguous band are not deterministic**, because an LLM adjudicates
  them. The same frame at slightly different compression can land on either side.
- **`gridsight/ingest/hf_scrape.py` is dead code** for a purged powerline corpus, kept only for its
  provenance-logging pattern. It is not on any live path.

---

## Not done

Stated plainly rather than left for a reader to notice:

- **ElevenLabs spoken findings.** Not built. No `ELEVENLABS_API_KEY` was available and the work was
  cut for time. The existing OpenAI Realtime voice agent is untouched and still works.
- **Projects / tenant grouping.** The registry is still a flat list of classes rather than being
  grouped under an owning project.
- **`/registry` and `/trends` redesign.** Both still render their original layout. The backfill
  disclosure was added to `/trends`; the chart restructure was not.
- **`/voice` is still a route**, not the floating widget on the main screen.
- **Cold-start latency regressed** from ~3.3 s to ~9 s, as the price of calibrating the image
  threshold out of sample. The fix is to cache backbone features across calibration folds, measured
  at roughly a 30% saving and not attempted.
- **The vision path is exercised only against Fireworks.** The local provider is implemented and
  selected when weights are cached, but no local vision model has been downloaded on this machine,
  so that branch resolves to null here and is unexercised.

---

## Licence

Code in this repository is MIT (see `LICENSE`). Dataset imagery is **not** ours to relicense: VisA is
CC BY 4.0 and MVTec AD is CC BY-NC-SA 4.0 (non-commercial). Trained model artifacts inherit the
constraints of the corpus they were fitted on, which is why the registry surfaces licence per model.

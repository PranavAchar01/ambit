# GridSight v2 — specification

**A competence-routing layer for fleets of computer-vision models, on MongoDB Atlas.**

Status legend: **[BUILT]** shipped and measured · **[V2]** specified here, not yet built · **[PENDING]** blocked on a verification currently running.

---

## 1. Thesis

Vision models that find defects already work. What does not exist is anything that decides **which
model to run**. Every asset class needs its own specialist, and in the field there is no barcode to
scan and no MES to ask — a drone over a right-of-way has nothing to identify what it is looking at.

GridSight is that decider. MongoDB holds a registry of anomaly-detection specialists; each carries a
512-d embedding of the imagery it was trained on. An agent embeds an incoming frame, vector-searches
the registry, loads the winning specialist out of GridFS, and runs it. If nothing clears the winner's
coverage gate, the agent **refuses to guess** and cold-starts a new specialist from a handful of
reference images — which then lives in the registry for every future flight.

### The qualifying test for any deployment

> **Is the identity of the thing already known from data at the moment of inspection?**

- **Yes** (PCB with a barcode, wafer with a lot ID) → routing is a nice-to-have; the MES already answered it.
- **No** (drone over a line, produce on a belt, mixed recycling stream, robot in an unmapped scene) → routing is load-bearing and nothing else does it.

GridSight targets the second case. Infrastructure inspection is the flagship because it is the purest
instance of it.

---

## 2. What the system is, abstractly

Five primitives, none of them domain-specific:

| # | primitive | implementation |
|---|---|---|
| 1 | **Routing by training distribution** | `$vectorSearch` over per-model CLIP centroids |
| 2 | **Explicit competence boundaries** | `routing_threshold` — p5 of a model's own train-set self-similarity |
| 3 | **Abstention** | score below the winner's gate → `verdict: "unroutable"` |
| 4 | **Few-shot cold-start** | PatchCore memory bank fitted from ~8 normals in ~3s |
| 5 | **Normal-only learning** | no defect labels ever required |

Economic consequence that makes the long tail addressable: **~7 MB per model**. Ten thousand
specialists is ~70 GB.

---

## 3. Persistent context — the memory architecture

PatchCore has no gradient-trained parameters. Training stores a **coreset of patch feature vectors**;
inference is nearest-neighbour against that store. So a model *is* remembered examples of normal, and
scoring is retrieval. GridSight is a memory system at two levels: vector search picks **which** memory
is relevant, then nearest-neighbour searches **inside** it.

| memory type | contents | store | lifetime | status |
|---|---|---|---|---|
| **Semantic / skill** | what it knows how to inspect | `models` + GridFS `weights` | permanent, grows | **[BUILT]** |
| **Metacognitive** | where each model stops being competent | `models.routing_threshold` | permanent | **[BUILT]** |
| **Episodic** | every frame inspected, and why it routed that way | `findings` | append-only | **[BUILT]** |
| **Episodic *recall*** | "have we seen this before?" | `findings.embedding` + `finding_recall_idx` | permanent | **[V2]** |
| **Working** | in-flight agent state, resumable | `checkpoints` / `checkpoint_writes` | per-thread | **[BUILT]** |
| **Provenance** | why each capability exists | `datasets` | permanent | **[BUILT]** |
| **Sensory** | the pixels themselves | GridFS `images` | permanent | **[BUILT]** |

Runtime hierarchy: in-process LRU (hot) → GridFS in Atlas (warm) → vector index decides what to promote.

### The accumulation loop

Refusal → cold-start → register is a **permanent expansion of competence**, shared across processes.
Measured: a rail specialist minted in 3.3 s from 8 images became visible to a different process
immediately; the same frame re-scored at routing 0.9467 instead of being refused.

### Missing loop, named honestly **[V2]**

Episodic memory currently flows one way — findings are written and power dashboards, but nothing
feeds back into the registry. Three consolidations that should exist:

1. High-confidence nominal frames extend a model's memory bank.
2. A *cluster* of refusals in embedding space auto-proposes a cold-start.
3. Score drift over time marks a model stale (same maths as the existing `failing_fastest` metric, pointed at model health).

---

## 4. Data model

Database `gridsight` on Atlas M0 (MongoDB 8.0.29).

### `models` **[BUILT]**
```
_id, name, asset_class, backbone,
embedding: [512 floats]        # L2-normalised centroid of the training set's CLIP embeddings
embedding_count,
routing_threshold: float       # coverage gate: p5 of self-similarity, in Atlas (1+cos)/2 space
image_threshold, pixel_threshold,
weights_file_id  -> GridFS weights.*
reference_image_id -> GridFS images.*   # golden known-good exemplar
training_samples,
metrics { image_auroc, pixel_auroc },
created_at, created_by: "seed" | "agent-coldstart",
provenance {...}
```

### `findings` **[BUILT]** + **[V2]** additions
```
_id, timestamp, uploaded_image_id, asset_class,
routed_model_id, routed_model_name, routing_score,
anomaly_score, raw_anomaly_score, verdict: "nominal"|"defect"|"unroutable",
severity, bbox_regions [{x,y,w,h,score}], heatmap_id,
agent_narrative, narrative_source, decision_source, decision_reason,
candidates [...], cold_start_info, latency_ms
embedding: [512 floats]        # [V2] the frame's CLIP vector — already computed, currently discarded
```

### Indexes

| index | collection | path | dims | status |
|---|---|---|---|---|
| `model_router_idx` | `models` | `embedding` | 512, cosine | **[BUILT]** READY |
| `finding_recall_idx` | `findings` | `embedding` | 512, cosine | **[V2]** |

Both created programmatically via `create_search_index()` and polled to `READY` — a `$vectorSearch`
against a `PENDING` index returns an empty result set rather than erroring, so polling is mandatory.

### Hosting constraint

Atlas is required, not preferred: `$vectorSearch` is Atlas-only. The offline fallback degrades to
brute-force cosine and logs `ATLAS VECTOR SEARCH UNAVAILABLE … DEGRADING TO BRUTE-FORCE COSINE` at
ERROR on every call. Never silent.

---

## 5. Agent graph **[BUILT]**

```
embed_frame → route → decide ─┬→ infer ──────→ narrate → persist
                              │                  ↑
                              ├→ cold_start ─────┘
                              └→ narrate (refusal, no references supplied)
```

Every node checkpoints to MongoDB; runs are resumable by `thread_id`. Images are written to GridFS
*before* invocation and referenced by id, keeping checkpointed state small and each frame replayable.

Decision rule: route if `score ≥ max(ROUTE_THRESHOLD, model.routing_threshold)`; escalate to OpenAI
structured-output adjudication within a 0.04 band below that; refuse beneath it.

**[V2]** adds a `recall` node between `route` and `decide`: vector-search `findings` for visually
similar past inspections and attach them to state, so `narrate` can say *"this is the third time this
month"* and the voice agent can answer *"have we seen this before?"*

---

## 6. v2 deliverables

### 6.1 Second registry source — MVTec AD **[VERIFIED]**

**Why:** current top-1 routing is **79%** because the powerline classes are visually *nested* — a
tower crop contains insulators and cables, so their centroids overlap. Object categories that are
genuinely distinct should push routing far higher and make the registry demo self-evident.

**Selected: `foersben/mvtec-ad`** — verified by download, not by reputation. 15 categories
(bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush,
transistor, wood, zipper), 6,612 PNGs / 5.27 GB total, **ground-truth masks present**, and it
preserves the canonical layout `<category>/train/good/`, `<category>/test/<defect>/`,
`<category>/ground_truth/<defect>/<stem>_mask.png`. Crucially it is **per-category fetchable**, so
ingest never pays 5.3 GB. Evidence: bottle+toothbrush pulled in 60.7 s (266 MB); a 15-category slice
in 42 s; `grid` fetched fresh, unauthenticated, ungated.

```python
from huggingface_hub import snapshot_download
root = snapshot_download(
    repo_id="foersben/mvtec-ad", repo_type="dataset",
    allow_patterns=[f"{CATEGORY}/**"],
)
```

Rejected, with reasons: `Voxel51/mvtec-ad` and its three byte-identical re-uploads are FiftyOne
exports that flatten every category into `data/data_0/*.png`, destroying per-category access;
`BrachioLab/mvtec-ad` packages cleanly as parquet but ships **no masks**; `katiehahm/mvtec_ad` has no
train/good split at all; several mirrors are single 5.3 GB archives with no per-category fetch.
`TheoM55/mvtec_anomaly_detection` is a viable runner-up but declares no licence.

> **LICENCE — a real constraint, not a formality.** MVTec AD is **CC BY-NC-SA 4.0: non-commercial
> only**, and ShareAlike propagates to derivatives. Fine for a hackathon demo, research, or internal
> evaluation. **Not** fine for shipping GridSight commercially with MVTec-derived weights baked in.
> The powerline corpus carries no such restriction, so the commercial story must rest on that;
> MVTec is for demonstrating the routing thesis. This must be stated on the registry UI.

**Work:** `gridsight/ingest/hf_scrape.py` is currently hard-wired to the powerline corpus and its
bbox geometry. It must generalise to folder-structured multi-category datasets via a source-adapter
seam, leaving the existing powerline extractors intact.

### 6.2 Real pixel AUROC **[UNBLOCKED]**

`pixel_auroc` is `null` for every current model because no source dataset ships masks, and anomalib
leaves `_pixel_threshold` as NaN for the same reason (GridSight already calibrates around this from
the p99.5 of good-image anomaly maps). MVTec ships `ground_truth/` masks, so this is now unblocked: pass `mask_dir` to the
`Folder` datamodule and **report a measured pixel AUROC for the first time**. The powerline classes
keep `pixel_auroc: null` — honestly, because no mask exists for them.

### 6.3 Episodic recall **[V2]**

- Persist the frame embedding already computed in `embed_frame` onto the finding.
- Create `finding_recall_idx`; generalise `vector_search()` in `gridsight/db/mongo.py`, which
  currently hard-codes the models collection and the `embedding` path in *both* the aggregation and
  the brute-force fallback.
- `POST /recall` → similar past findings with scores, dates, verdicts.
- Feeds the narrative, the voice agent, and a "recurrence" signal in trends.

### 6.4 Registry animation **[V2]** — the money shot

A live visualisation of the routing decision, driven by **real** SSE agent steps, not a canned loop.

**Layout — radial, driven by measured scores rather than a projection:**

- The **query frame sits at the centre**.
- Each registry model is a node at radius `r = 1 − score` — its *actual* measured similarity. Angular
  position comes from a deterministic PCA of the stored centroids, purely for stable, non-overlapping
  placement.
- Each model draws a **gate ring** at radius `1 − routing_threshold`. This is the honest geometric
  statement of the coverage gate: **a model's dot falling inside its own ring means the frame is
  within the competence it demonstrably learned.**

**Timeline, bound to the agent's real steps:**

| agent step | animation |
|---|---|
| `embed_frame` | query point materialises at centre; 512-d vector shimmer |
| `route` | rings appear; candidate nodes fly to their true radii; scores count up |
| `decide` → route | winner's ring pulses green, node clearly inside it; others dim |
| `decide` → unroutable | **every** node sits outside its ring; all rings flash amber — the refusal is *visible*, not just stated |
| `cold_start` | a new node materialises at the centre and draws its own ring around the query |
| `infer` | weights stream from a GridFS glyph into the winner |

Honesty constraint: radii are real similarities; **angles are illustrative only** and the UI must say
so. No fabricated geometry.

Accessible fallback: the same information as an ordered table for screen readers and
`prefers-reduced-motion`.

### 6.5 Voice agent over persistent context **[VERIFIED]**

An operator asks GridSight about the fleet and it answers from accumulated history.

**Tools (server-side, because they query MongoDB):**

| tool | backed by |
|---|---|
| `query_trends(days, asset_class?)` | `gridsight/analytics.py::compute_trends` |
| `search_findings(asset_class?, verdict?, since?)` | `findings` query |
| `recall_similar(image_id \| finding_id, k)` | `finding_recall_idx` vector search |
| `registry_status()` | `models` + index health |
| `explain_refusal(finding_id)` | stored candidates + gates |

Representative questions: *"what's failing fastest?"*, *"have we seen this defect before?"*, *"how
many insulator defects this month?"*, *"which models did we cold-start, and why?"*

**Architecture: OpenAI Realtime API over WebRTC, with server-minted ephemeral client secrets.**
Verified live against the installed SDK (`openai 2.54.0`) and this API key:

- `client.realtime` is **GA, not beta** — `.calls` (WebRTC SDP exchange), `.client_secrets`
  (ephemeral `ek_` tokens), `.connect()` (server-side WebSocket with built-in auto-reconnect).
  `client.beta.realtime` is legacy and must not be used.
- **Measured latency: 1.02 s to first audio**, versus **≥4.4 s** for a serial
  transcribe→chat→TTS pipeline (0.93 s STT + ~1.2 s LLM + 2.29 s TTS). Realtime also gives native
  barge-in and server-side `semantic_vad` turn detection, both of which would otherwise be hand-built.
- The transcription/TTS endpoints (`client.audio.transcriptions`, `client.audio.speech`) are retained
  only for the non-interactive path — reading a stored `agent_narrative` aloud.

**Security, verified rather than assumed.** FastAPI mints an ephemeral credential via
`client.realtime.client_secrets.create(...)`; minting takes 0.16–0.88 s and yields an `ek_`-prefixed
token that honours `expires_at`. Its blast radius was tested directly: the token authenticates
against `POST /v1/realtime/calls` but returns **401 against `chat.completions` and `models.list`** —
scope is realtime-only. The browser therefore never sees `OPENAI_API_KEY`.

---

## 7. Interfaces

### API **[BUILT]** unless marked
```
GET  /health                     index status, model + finding counts, storage headroom
POST /inspect                    multipart frame → full verdict
POST /inspect/stream             same, SSE per agent step   ← drives the animation
POST /coldstart                  reference images + class name
GET  /models                     registry listing
GET  /findings                   paginated, filterable
GET  /trends                     fleet-health aggregations
GET  /image/{gridfs_id}          streams frames, heatmaps, references
POST /recall                     [V2] visually similar past findings
POST /voice/session              [V2] mint ephemeral voice credential
POST /voice/tool                 [V2] execute a voice tool call against MongoDB
```

### UI **[BUILT]** unless marked
Next.js App Router, TS strict, Tailwind, `pnpm`.
`/` inspect (drag-drop, live agent steps, canvas overlay beside golden reference, cold-start flow) ·
`/registry` · `/trends` · **[V2]** routing animation on `/` · **[V2]** voice console.

---

## 8. Measured results

### v2 — the combined 11-class registry (current)

| metric | value |
|---|---|
| top-1 routing accuracy | **245/270 (91%)** |
| &nbsp;&nbsp;— MVTec categories | **145/145 (100%)** |
| &nbsp;&nbsp;— powerline categories | **100/125 (80%)** |
| in-registry accepted | **253/270** |
| out-of-registry refused (`rail_surface`) | **25/25** |
| Atlas M0 usage | ~108 MB / 512 MB (21%) |

**Adding MVTec did not fix powerline routing, and the headline number is partly dilution.**
MVTec routes perfectly, which proves the router separates genuinely distinct domains. The nested
powerline classes went 99/125 → 100/125 — statistically unchanged. The 79% → 91% jump is mostly 145
easy probes entering the average. Stated plainly because the opposite reading is the tempting one.

Decision-rule comparison, re-measured on all 11 classes:

| rule | in-registry accepted | out-of-registry refused |
|---|---|---|
| global 0.82 | 268/270 | 10/25 |
| global 0.86 | 253/270 | 23/25 |
| global 0.88 | 236/270 | 25/25 |
| **per-model coverage gate** | **253/270** | **25/25** |

The gate now **strictly dominates** both tuned global thresholds — it matches 0.86's acceptance and
0.88's refusal simultaneously. That is stronger than the v1 claim, and it only holds because of the
gate defect described below.

### Pixel AUROC — real for the first time

| class | image AUROC | pixel AUROC |
|---|---|---|
| mvtec_leather | 1.0000 | **0.9943** |
| mvtec_hazelnut | 1.0000 | **0.9915** |
| mvtec_bottle | 1.0000 | **0.9865** |
| mvtec_screw | 0.9233 | **0.9861** |
| mvtec_tile | 1.0000 | **0.9637** |
| mvtec_transistor | 0.9967 | **0.9456** |
| all 5 powerline classes | 0.49 – 0.91 | **null** (no masks exist) |

Mask integrity was verified before any of these were trusted: 40 masks per class, zero unpaired,
zero size mismatches, all binary, none empty, defect area 0.14–27%. Anomalib additionally asserts
`image_stem in mask_stem` and raises `MisMatchError`, so a silent misalignment is impossible.

### A real defect found by adversarial verification

`mvtec_leather` initially refused **17/25 of its own held-out good frames**. Two hypotheses were
tested and both were wrong before the cause was found:

1. *Fit/holdout bias* — calibrating the gate on held-out normals instead of training ones. Made it
   **worse** (5/30 admitted).
2. Measurement then showed the truth: leather's entire class spans **0.0035** of similarity (vs 0.149
   for insulator), CLIP has saturated, and **23% of its test/good frames fall below its whole training
   minimum**. The shift is train→test, so no slice of train can predict it.
3. Fixed with `MIN_GATE_MARGIN` — no model may claim an envelope tighter than 0.012 below its own
   median, because differences smaller than that are below the resolution at which this embedding
   space distinguishes coverage. Result: leather 5/30 → **30/30**, bottle 17/20 → **20/20**,
   tile 27/30 → **30/30**, every other class unchanged.

Gates are recomputed without retraining (`scripts/recalibrate_gates.py`) because they depend only on
embeddings, never on the PatchCore memory bank.

---

## 8b. Measured baseline (v1, superseded by the above)

| metric | value |
|---|---|
| top-1 routing accuracy | **99/125 (79%)** |
| in-registry accepted | **115/125** |
| out-of-registry refused | **24/25** |
| GridFS weight round-trip | **exact**, delta 0.0 on all 11 |
| cold-start | **3.3 s**, 8 images, 12.6 MB |
| checkpoint resume | SIGKILL, `wait()` = −9, resumed without relearning |
| Atlas M0 usage | ~60 MB / 512 MB |

Per-class image AUROC: insulator 0.910 · conductor 0.834 · transmission_tower 0.717 · corrosion
0.628 · **vegetation 0.490**.

Decision-rule comparison, measured:

| rule | in-registry accepted | out-of-registry refused |
|---|---|---|
| global 0.82 | 123/125 | 10/25 |
| global 0.88 | 91/125 | 25/25 |
| **per-model coverage gate** | **115/125** | **24/25** |

---

## 9. Known defects carried into v2

**`vegetation` is unfixed and its diagnosis is the general lesson.** A supervised CLIP probe reaches
only **0.591** on those crops versus 0.972/0.975 for insulator/corrosion — the label is barely in the
pixels. Cause: 35% of "encroachments" are grazing box overlaps (an artifact of axis-aligned boxes
around diagonal cables) and 17% of crops retain <20% of the conductor, so the evidence is cropped out.

The general rule this establishes, and which constrains every future deployment:

> PatchCore detects **appearance** anomalies. It does not detect **relational** ones ("too close to"),
> **count-based** ones, or **logical** ones. Domains whose defects are relational need a different
> formulation, not a bigger model.

Resolution options: a corridor-anchored redefinition, or documenting the class as unlearnable from
bbox geometry. **Shipping a leaky proxy is not an option.** A measurement harness
(`scripts/eval_veg_variant.py`) exists and works.

---

## 10. Acceptance criteria for v2

1. ≥5 additional visually distinct categories ingested from a **verified-loadable** Hub dataset, provenance recorded.
2. Top-1 routing on the combined registry **measured and reported honestly**, whatever it is.
3. `pixel_auroc` non-null for at least the mask-bearing classes — or an explicit statement of why not.
4. `finding_recall_idx` READY; `/recall` returns visually similar past findings for a real frame.
5. Routing animation driven by **live SSE steps**, showing route, refusal, and cold-start; legible in light and dark; reduced-motion fallback.
6. Voice agent answers ≥4 distinct fleet questions using MongoDB-backed tools, with no API key in the browser.
7. `ruff` clean · `mypy` clean · `pytest` green · `pnpm build` green.
8. Nothing fabricated. Every metric traceable to an executed command.

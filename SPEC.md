# GridSight v2 — specification

**A registry of inspection competence, on MongoDB Atlas. It knows what it can inspect, and refuses what it cannot.**

Status legend: **[BUILT]** shipped and measured · **[V2]** specified here, not yet built · **[PENDING]** blocked on a verification currently running.

**Scope, as of this revision: electronics only** — semiconductors, PCBs, and dev boards. Every powerline
and general-object specialist has been purged from the registry (§6.0). The architecture is unchanged;
what it inspects is not.

---

## 0. Positioning — read this before writing any pitch, README, or submission text

### 0.1 The load-bearing claim

> **The most dangerous thing a quality-control model can do is confidently pass a board it has never seen.**

Routing is *how* the system knows which inspector applies. **Abstention is what it does when none of them do.** Cold-start is how it repairs that gap in three seconds. The registry is the product; the UI is a window onto it.

### 0.2 Beachhead ICP — low-volume, high-mix, fast-revision hardware

A hardware team builds forty units of a prototype board.

- **No inspection machine.** Automated optical inspection starts in the tens of thousands and must be programmed per board from CAD. That cost amortises over ten thousand units, not forty.
- **No defect data.** Supervised defect detection needs defect examples that do not exist, because the board did not exist last month.
- **No stability.** Every design revision invalidates whatever was built for the previous one.

Result: QC at prototype scale is a person squinting under a desk lamp, and nothing accumulates.

Defense and physical-AI startups are the sharpest instance — low volume, high consequence, weekly revisions. **Say that as a sentence, not as the headline.** Leading with defense invites ITAR and data-handling questions that consume the Q&A and do not win points.

### 0.3 The qualifying test

> **Is there an inspector in the fleet with demonstrated competence over this exact thing, right now?**

| | |
|---|---|
| **Identity unknown** — a mixed tray of boards arrives unlabelled; a bench camera sees whatever is placed under it; an operator holds a part up and does not tell the system which it is | Routing is load-bearing. Nothing else does it |
| **Identity known, competence uncertain** — Rev C arrives and Rev B's inspector is stale; 40 part types × 5 revisions = 200 specialists; a contract assembler runs a board it did not design | **Abstention and cold-start are load-bearing.** This is the beachhead |
| **Identity known, competence stable** — one high-volume SKU, AOI already programmed and amortised | Not our customer |

Both live columns are the same five primitives. Only the emphasis moves.

### 0.4 Anticipated objection, with the answer

**"If they know what board it is, why do they need routing?"**

> Because they do not know what *revision* it is, and the person holding the board is not going to tell the system. Forty part types across five revisions is two hundred specialists. The question is never "what is this" — it is "does anything in the fleet actually have competence over this, or am I about to silently pass a board nobody has ever inspected."

**"Doesn't AOI already solve this?"**

> AOI is CAD-driven: it imports Gerber and centroid files and checks a defined list — component present, aligned, solder within spec. Three things follow. It needs design files you may not have, for boards you did not design. It catches only what is on the list, so contamination, substrate damage and counterfeit-but-similar parts are structural blind spots. And programming it takes hours per board, which never pays back over forty units. We need eight photographs and no CAD at all.

### 0.5 The thirty-second pitch

> "A hardware startup builds forty units of a prototype board. There is no inspection machine — those start around fifty thousand dollars and have to be programmed per board from CAD files, which never pays back over forty units. And there is no defect data to train on, because the board did not exist last month.
>
> So we do not train on defects. We show it eight good units and mint an inspector in three seconds. It flags anything that deviates. And when Rev C shows up next week, it does not quietly pass it — it tells you it does not recognise the board, and mints a new inspector on the spot.
>
> Every board, every revision, permanently, in MongoDB. The registry only ever grows."

### 0.6 The line that carries the technical half

Say this explicitly. It is the deepest on-theme claim in the architecture and it is literally true:

> **"Our models have no trained weights. PatchCore stores a coreset of patch features — remembered examples of normal — and inference is nearest-neighbour against that store. The database is not sitting next to the model. The database *is* the model. Vector search picks which memory applies; nearest-neighbour searches inside it. It is memory all the way down."**

### 0.7 Naming — now actively wrong

`GridSight` named the powerline origin, and that corpus has been **purged from the product**. The name now points at something the system no longer does, which is worse than merely unhelpful: a judge who looks it up finds an electrical-grid story that contradicts the demo. **Rename before the repo is public.** Something competence-, registry-, or boundary-shaped. Cheap now, impossible later. If the name stays, the tagline must carry the entire load and the powerline origin must not appear anywhere in the pitch.

### 0.8 Contribution disclosure — a disqualification risk, not a formality

Where event rules require new work only and demand that judges can identify what was built during the event, an unlabelled `[BUILT]` corpus is a hazard. **The first section of the README must separate pre-existing work from work completed during the event window**, mapping directly onto the `[BUILT]` / `[V2]` legend already used throughout this document. The tracking discipline exists; make it the first thing a judge reads.

### 0.9 Licence is a positioning constraint, not an appendix

The registry now spans two licences and the difference decides what can be claimed commercially:

| source | licence | what it permits |
|---|---|---|
| **VisA** (`pcb1`–`pcb4`) | **CC BY 4.0** — permissive, attribution only | Commercial use is fine. **The PCB classes are the commercial story.** |
| **MVTec AD** (`transistor`, `cable`) | CC BY-NC-SA 4.0 — **non-commercial**, ShareAlike propagates | Demo, research, internal evaluation only. Never in a shipped product. |
| **Live-captured boards** | ours | Unrestricted |

This is the reverse of the old position, where the permissively-licensed corpus was the powerline data. Now the **PCB classes carry the commercial claim and MVTec is demo-only**. State the non-commercial constraint on the registry UI.

---

## 1. Thesis

Vision models that find defects already work. What does not exist is anything that decides **which model to run, and whether any of them should run at all.**

Every board — and every revision of every board — needs its own specialist. At prototype volume there is no AOI recipe to load, no CAD to import, no defect set to train on, and frequently no barcode: an operator holding a board up to a bench camera does not tell the system which revision it is.

GridSight is that decider. MongoDB holds a registry of anomaly-detection specialists; each carries a 512-d embedding of the imagery it was trained on and an explicit gate marking where its competence ends. An agent embeds an incoming frame, vector-searches the registry, loads the winning specialist out of GridFS, and runs it. If nothing clears the winner's coverage gate, the agent **refuses to guess** and cold-starts a new specialist from a handful of reference images — which then lives in the registry for every future inspection, in every process.

---

## 2. What the system is, abstractly

Five primitives, none of them domain-specific:

| # | primitive | implementation | why the beachhead needs it |
|---|---|---|---|
| 1 | **Routing by training distribution** | `$vectorSearch` over per-model CLIP centroids | 200 specialists across boards × revisions; nobody scans anything |
| 2 | **Explicit competence boundaries** | `routing_threshold` — a per-model coverage gate | Rev B's inspector must not claim Rev C |
| 3 | **Abstention** | score below the winner's gate → `verdict: "unroutable"` | **The safety property.** Silently passing an uninspected board is the failure that costs money |
| 4 | **Few-shot cold-start** | PatchCore memory bank fitted from ~8 normals in ~3 s | A new revision must be coverable during the build, not next sprint |
| 5 | **Normal-only learning** | no defect labels ever required | There is no defect data at prototype volume, and there never will be |

Economic consequence that makes the long tail addressable: **~7 MB per model**. Ten thousand specialists is ~70 GB. Two hundred specialists — one shop's entire board catalogue across every revision — is 1.4 GB.

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
| **Episodic *recall*** | "have we seen this before?" | `findings.embedding` + `finding_recall_idx` | permanent | **[BUILT]** |
| **Working** | in-flight agent state, resumable | `checkpoints` / `checkpoint_writes` | per-thread | **[BUILT]** |
| **Provenance** | why each capability exists | `datasets` | permanent | **[BUILT]** |
| **Sensory** | the pixels themselves | GridFS `images` | permanent | **[BUILT]** |

Runtime hierarchy: in-process LRU (hot) → GridFS in Atlas (warm) → vector index decides what to promote.

### The accumulation loop

Refusal → cold-start → register is a **permanent expansion of competence**, shared across processes.
Measured: a specialist minted in 3.3 s from 8 images became visible to a different process
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
routing_threshold: float       # coverage gate, in Atlas (1+cos)/2 space
image_threshold, pixel_threshold,
weights_file_id  -> GridFS weights.*
reference_image_id -> GridFS images.*   # golden known-good exemplar
training_samples,
metrics { image_auroc, pixel_auroc },
created_at, created_by: "seed" | "agent-coldstart",
provenance {...}               # includes source dataset id and LICENCE
```

### `findings` **[BUILT]**

```
_id, timestamp, uploaded_image_id, asset_class,
routed_model_id, routed_model_name, routing_score,
anomaly_score, raw_anomaly_score, verdict: "nominal"|"defect"|"unroutable",
severity, bbox_regions [{x,y,w,h,score}], heatmap_id,
agent_narrative, narrative_source, decision_source, decision_reason,
candidates [...], cold_start_info, latency_ms,
embedding: [512 floats]        # the frame's CLIP vector -- makes findings searchable
```

### Indexes

| index | collection | path | dims | status |
|---|---|---|---|---|
| `model_router_idx` | `models` | `embedding` | 512, cosine | **[BUILT]** READY |
| `finding_recall_idx` | `findings` | `embedding` | 512, cosine | **[BUILT]** READY |

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

### 6.0 Rescope to electronics **[BUILT]**

The registry previously held five powerline classes and six general MVTec objects. All eleven were
purged except `transistor`; `scripts/purge_registry.py` removes the registry document, the GridFS
weights, the reference image, the provenance row, the on-disk corpus, and **the findings that pointed
at them** — a finding whose routed model no longer exists would leave `/trends` and
`finding_recall_idx` describing a fleet that is not there.

Measured: 11 → 1 models, 63 findings removed, Atlas 21.1% → 6.0%.

The purge is scope-driven, not quality-driven. Some dropped classes were the best performers in the
registry (`mvtec_leather` at pixel AUROC 0.9943). They were removed because a hazelnut is not a
semiconductor, and a registry that inspects hazelnuts undermines the pitch it is supposed to prove.

### 6.1 The electronics registry **[PENDING]**

**Selected sources, both verified by download rather than by reputation:**

| source | classes | licence | why |
|---|---|---|---|
| **`BrachioLab/visa`** | `pcb1`, `pcb2`, `pcb3`, `pcb4` | **CC BY 4.0 — permissive** | Four real populated PCBs with per-category parquet, normal-only train splits, 100 anomalies per test split, and **100% mask coverage**. This is the commercial-safe half of the registry |
| **`TheoM55/mvtec_all_objects_split`** | `transistor`, `cable` | CC BY-NC-SA 4.0 — non-commercial | Real semiconductor package and electrical wiring, per-category parquet with a genuine `object` column |

Both are per-category addressable, so ingest never pays a multi-gigabyte blind download.

Rejected, with reasons recorded: `imaadd05/visa-anomaly-detection` and `foersben/mvtec-ad` flatten
labels through the datasets loader so per-category normals are unrecoverable; `csgaobb/AnoGen-MVTec-VisA`
is diffusion-generated and GridSight's ingest promises nothing synthetic; DAGM's ten classes are
near-identical grey textures that would actively *worsen* routing; `Real-IAD` is 622 GB and gated.

**Open risk, to be reported honestly: `pcb1`–`pcb4` may collide in routing space.** They are all
green boards photographed in the same rig. This is the same nesting failure that capped the powerline
classes at 79%, and if it recurs it changes the demo — the fleet would need to be broader (PCB +
transistor + cable + dev board) rather than four near-identical boards. The verification pass measures
this rather than assuming it.

**Arduino / dev boards: [PENDING].** A Hub search is running. If usable photographs do not exist, the
class comes from **live capture at the venue**, which is the stronger demo anyway (§6.6).

### 6.2 Real pixel AUROC **[BUILT]**

`pixel_auroc` was `null` for every model because no source shipped masks, and anomalib leaves
`_pixel_threshold` as NaN for the same reason. Both VisA and MVTec ship ground-truth masks, so
`mask_dir` is passed to the `Folder` datamodule and pixel AUROC is **measured against real ground
truth**. Mask integrity is verified before any pixel metric is trusted: paired, binary, non-empty,
plausible defect area. Anomalib additionally asserts `image_stem in mask_stem` and raises
`MisMatchError`, so silent misalignment is impossible.

### 6.3 Episodic recall **[BUILT]**

- The frame embedding computed in `embed_frame` is persisted onto the finding.
- `finding_recall_idx` is READY; `vector_search()` was generalised into a `VectorIndex` abstraction
  driving both indexes, replacing code hard-wired to `models.embedding` in the aggregation *and* the
  brute-force fallback.
- `POST /recall` → visually similar past findings with scores, dates, verdicts, plus a recurrence
  summary (first seen / last seen / count).
- Feeds the narrative, the voice agent, and a recurrence signal in trends.

### 6.4 Registry animation **[BUILT]** — the money shot

A live visualisation of the routing decision, driven by **real** SSE agent steps, not a canned loop.

- The **query frame sits at the centre**; radial distance is `1 − score`, the *actual* measured similarity.
- Each model's spoke carries a **gate tick** at `1 − routing_threshold`. A dot landing **inside** its
  tick is the honest geometric statement of coverage: the frame is within the competence that model
  demonstrably learned.
- On refusal, **every dot sits outside every tick** — the abstention is *visible*, not asserted.
- Cold-start materialises a new node and draws its gate around the query.

Honesty constraint: radii are real similarities; **angles are illustrative ordering only** (PCA rank,
evenly spaced to avoid collisions) and the caption says so. Accessible fallback: the same information
as an ordered table for screen readers and `prefers-reduced-motion`.

### 6.5 Voice agent over persistent context **[BUILT]**

An operator asks about the fleet and it answers from accumulated history, not from a prompt.

| tool | backed by |
|---|---|
| `query_trends(days, asset_class?)` | `gridsight/analytics.py::compute_trends` |
| `search_findings(asset_class?, verdict?, since?)` | `findings` query |
| `recall_similar(finding_id, k)` | `finding_recall_idx` vector search |
| `registry_status()` | `models` + index health |
| `explain_refusal(finding_id)` | stored candidates + gates |

**Architecture: OpenAI Realtime API over WebRTC, with server-minted ephemeral client secrets.**
Verified live against the installed SDK (`openai 2.54.0`):

- `client.realtime` is **GA, not beta**. `client.beta.realtime` is legacy and must not be used.
- **1.02 s to first audio**, versus **≥4.4 s** for a serial transcribe→chat→TTS pipeline. Realtime
  also gives native barge-in and server-side `semantic_vad` turn detection.
- Ephemeral `ek_` token scope tested directly: authenticates against `/v1/realtime/calls`, returns
  **401 against `chat.completions` and `models.list`**. The browser never sees `OPENAI_API_KEY`.
- End-to-end verified: session connects, `query_trends` executes against Atlas in **607 ms**, and the
  spoken answer matched the measured data.

The UI shows **which tool ran and what it returned**, so an audience sees the answer coming from
MongoDB rather than from the model's imagination.

---

## 6.6 Demo runbook — the beachhead demo **[V2]**

This is the deliverable everything else serves. Build order in §6.7 is derived from it.

### 6.6.1 The arc — refusal before detection

| # | beat | what it proves |
|---|---|---|
| 1 | Registry shows the known electronics fleet — four PCBs, a transistor package, a cable harness | The fleet exists and is legible |
| 2 | Hold up **the target board** → `unroutable`; every node sits outside its ring | **Abstention.** The honest "I do not know" |
| 3 | Shoot 8 good units **live, on the table** → cold-start, ~3 s | Learning, in the room, in front of them |
| 4 | A **ninth, previously unseen good unit** → `nominal` | It learned *normal*, not those eight photographs |
| 5 | Introduce the defect → `defect` + heatmap | It works |
| 6 | Query from a **second browser session / process** → the new specialist is already there | **Persistence.** The MongoDB moment |
| 7 | Voice: *"what did we just learn, and why did you refuse the first one?"* | Memory answering for itself |

**Beat 2 must precede beat 5.** Every competing demo detects something. This one *declines* to — visibly, with every node outside its gate ring — and then repairs itself. That sequence is the entire thesis and it is legible in under thirty seconds.

**Beat 4 is not optional.** Without an unseen good unit, the demo is indistinguishable from memorisation.

**Beat 6 is the cheapest high-value beat in the build.** A cold-started specialist becoming visible to a different process was already measured; showing it costs a second browser window.

### 6.6.2 Photometric discipline — the primary failure mode

PatchCore fitted on eight images under one lighting condition scores **everything** as anomalous under another. This is the single most likely cause of a failed live demo, and bare PCBs make it worse: solder mask is glossy and specular highlights move with the light.

**Mitigation: fit the specialist at the venue, minutes before.** This converts the largest risk into the strongest beat — *"we are minting this inspector right now, on this table, in this room."*

Required rig:

- fixed camera on a copy stand or small tripod; fixed working distance;
- **portable LED panel**, ideally diffused, so venue lighting is not in the loop and solder-mask glare is controlled;
- matte background, same for reference and test frames;
- reference frames and test frames photometrically identical by construction, not by luck.

### 6.6.3 Defect selection

§9 establishes that PatchCore detects **appearance** anomalies, not relational, count-based, or logical ones. The staged defect must therefore produce a large appearance delta **from the fixed camera angle in use**.

- A pin bent along the camera axis may barely change pixels. Bent laterally, under a fixed top-down view, it does.
- Good candidates on a board: a lifted or tombstoned component, a solder bridge, a missing header pin, a scratched trace, a visibly misaligned IC.
- Poor candidates: a wrong-value resistor (identical silhouette), a missing decoupling cap in a dense field (count-based, §9), reversed polarity on an unmarked part.
- **Test the exact defect, on the exact rig, before the event.** Do not discover this at the venue.

### 6.6.4 Single view for the demo; multi-view as roadmap

A 360 scan requires per-view models or pose normalisation and will consume the build window. **Demo a fixed single view** — top-down on the board. State the extension in one sentence, because it follows naturally from the thesis rather than adding a system:

> A two-sided board becomes two canonical views, each its own registry entry; the router selects which side it is looking at.

### 6.6.5 Second-half framing — generality without the powerline story

The powerline corpus is **purged and must not appear in the pitch** (§0.7). Generality is now argued from the architecture and from the licence split, in one sentence:

> The registry is domain-agnostic — swap the encoder and the same routing, abstention and cold-start run over acoustic or telemetry signatures. What makes it work here is that identity must be *inferred*: there is no barcode on a bare board under a bench camera.

If a judge asks what else it has run on, the honest answer is available: it was previously seeded with drone-captured powerline assets and scored 79% top-1 routing across nested asset classes. **Offer that only under questioning**, and frame it as prior evidence of generality, not as current product scope.

### 6.6.6 Q&A ammunition — hold, do not volunteer

§9's diagnosis — a supervised probe at 0.591, 35% grazing box overlaps, "shipping a leaky proxy is not an option" — is the strongest answer available to a probing judge. **Keep it in the README and in reserve.** Volunteering it mid-pitch spends the demo clock on a negative result; deploying it under questioning demonstrates measured self-knowledge.

---

## 6.7 Build priority — ordered by contribution to winning

| rank | item | rationale |
|---|---|---|
| 1 | **Live cold-start on the target board** (§6.6) | This is the demo. Nothing else matters if it fails |
| 2 | **Electronics registry ingest** (§6.1) | Makes the registry legible *and* is the ICP made visible — four PCBs, a transistor, a cable harness |
| 3 | **Registry animation** (§6.4) | Beat 2 dies without it. The refusal must be *seen*, not merely stated |
| 4 | **Cross-process persistence proof** (beat 6) | Nearly free; highest theme payoff per minute of work |
| 5 | **Voice agent** (§6.5) | Strong, but only after 1–4 |
| 6 | Episodic recall (§6.3), pixel AUROC (§6.2) | Repo credibility and Q&A ammunition rather than stage time |

### Sponsor-alignment check

Where an event offers named sponsor prizes, verify before committing:

- **Voice stack.** OpenAI Realtime measured 1.02 s to first audio versus ≥4.4 s serial. Do not trade that away blindly — but if a voice sponsor offers a prize, evaluate whether their realtime agent platform closes the gap before deciding.
- **LLM adjudication.** The structured-output adjudicator in the 0.04 band is a single call behind an interface; route it through a sponsor gateway if credits are in play. One config line.

---

## 7. Interfaces

### API **[BUILT]** unless marked

```
GET  /health                     index status, model + finding counts, storage headroom
POST /inspect                    multipart frame → full verdict
POST /inspect/stream             same, SSE per agent step   ← drives the animation
POST /coldstart                  reference images + class name
GET  /models                     registry listing
GET  /registry/layout            polar layout for the routing animation
GET  /findings                   paginated, filterable
GET  /trends                     fleet-health aggregations
GET  /recall                     visually similar past findings
GET  /image/{gridfs_id}          streams frames, heatmaps, references
POST /voice/session              mint ephemeral voice credential
POST /voice/tool                 execute a voice tool call against MongoDB
```

### UI **[BUILT]**
Next.js App Router, TS strict, Tailwind, `pnpm`.
`/` inspect (drag-drop, live agent steps, routing animation, canvas overlay beside golden reference,
cold-start flow) · `/registry` · `/trends` · `/voice`.

---

## 8. Measured results

### 8.1 Electronics registry **[MEASURED]**

| class | source | licence | image AUROC | pixel AUROC | gate |
|---|---|---|---|---|---|
| `visa_pcb1` | VisA | **CC BY 4.0** | 0.8967 | **0.9873** | 0.9794 |
| `visa_pcb2` | VisA | **CC BY 4.0** | 0.8600 | **0.9717** | 0.9797 |
| `visa_pcb3` | VisA | **CC BY 4.0** | 0.9967 | **0.9755** | 0.9801 |
| `mvtec_transistor` | MVTec | CC BY-NC-SA 4.0 | 0.9967 | **0.9456** | 0.9692 |
| `mvtec_cable` | MVTec | CC BY-NC-SA 4.0 | 0.9933 | **0.9892** | 0.9629 |
| `visa_pcb4` | VisA | CC BY 4.0 | — **withheld** — | — | — |

`pixel_auroc` is real for every class; Atlas usage 62 MB / 512 MB (12.1%).

**Top-1 routing: 125/125 (100%).** The `pcb1`–`pcb4` collision risk flagged in §6.1 **did not
materialise** — four green boards photographed in the same rig still separate perfectly.

**The abstention result is the headline, and it is far stronger than the powerline registry's:**

| rule | in-registry accepted | out-of-registry refused |
|---|---|---|
| global 0.82 | 125/125 | **0/25** |
| global 0.86 | 125/125 | **0/25** |
| global 0.88 | 125/125 | **0/25** |
| **per-model coverage gate** | 119/125 | **25/25** |

Withheld `visa_pcb4` scores **0.9399 mean / 0.9525 max** against the registry — *above every global
threshold tested*. **No fixed threshold can refuse it at all.** Only the per-model gate does, 25 times
out of 25.

This is §0.1 demonstrated rather than asserted: a board nobody has inspected scores high enough to
sail through any global cut, because it genuinely *is* a PCB and the registry is full of PCBs. The
system knows it is looking at a PCB-like object and still declines, because no specialist has
competence over *that particular board*. That is precisely the Rev C scenario from §0.4.

Cost of that refusal, stated honestly: 6 of 125 in-registry frames are also refused (119/125
accepted). At prototype volume, re-shooting six frames is cheaper than passing one uninspected board.

### 8.2 Prior evidence — the powerline + MVTec registry (superseded, historical)

These were measured on the corpus that has since been purged. Retained because they are the evidence
behind the design decisions in §2 and §9, and because they are the honest answer to "what else has
this run on?" (§6.6.5).

| metric | value |
|---|---|
| top-1 routing, 11-class combined | 245/270 (91%) |
| — MVTec categories | 145/145 (100%) |
| — powerline categories | 100/125 (80%) |
| in-registry accepted | 253/270 |
| out-of-registry refused | 25/25 |
| GridFS weight round-trip | **exact**, delta 0.0 on all 11 |
| cold-start | **3.3 s**, 8 images |
| checkpoint resume | SIGKILL, `wait()` = −9, resumed without relearning |

Decision-rule comparison, measured on that registry:

| rule | in-registry accepted | out-of-registry refused |
|---|---|---|
| global 0.82 | 268/270 | 10/25 |
| global 0.86 | 253/270 | 23/25 |
| global 0.88 | 236/270 | 25/25 |
| **per-model coverage gate** | **253/270** | **25/25** |

The gate **strictly dominates** both tuned global thresholds — matching 0.86's acceptance and 0.88's
refusal simultaneously. Two findings from that run carry forward:

1. **Adding visually distinct categories did not fix nested ones.** MVTec routed 145/145 while the
   nested powerline classes stayed at ~80%. The combined 91% was largely dilution. This is exactly
   why §6.1 flags the `pcb1`–`pcb4` collision risk rather than assuming distinctness.
2. **A percentile gate needs spread.** `mvtec_leather` refused 17/25 of its own good frames because
   its entire class spanned 0.0035 of similarity — CLIP had saturated. Fixed with `MIN_GATE_MARGIN`:
   no model may claim an envelope tighter than 0.012 below its own median. Gates recompute without
   retraining (`scripts/recalibrate_gates.py`) because they depend only on embeddings.

---

## 9. Design constraints carried forward

The `vegetation` class is gone with the powerline corpus, but **its diagnosis is the general lesson**
and it constrains defect selection for the live demo (§6.6.3).

A supervised CLIP probe reached only **0.591** on those crops versus 0.972/0.975 for other classes —
the label was barely in the pixels. Cause: 35% of "encroachments" were grazing bounding-box overlaps
and 17% of crops retained <20% of the object that defined the anomaly.

> PatchCore detects **appearance** anomalies. It does not detect **relational** ones ("too close to"),
> **count-based** ones ("six screws instead of eight"), or **logical** ones. Domains whose defects are
> relational need a different formulation, not a bigger model.

On a board this rules out, as reliable demo defects: wrong-value components with identical
silhouettes, missing parts in a dense repeated field, and anything defined by spacing rather than
appearance. **Shipping a leaky proxy is not an option.** The measurement harness
(`scripts/eval_veg_variant.py`) remains in the repo as the method for testing any future label
definition before trusting it.

---

## 10. Acceptance criteria for v2

**Demo-critical — failure here means no demo:**

1. Live cold-start on the target board succeeds at the venue, under the demo rig, with the venue's lighting, in under 10 s end to end.
2. An unseen good unit scores `nominal` and the staged defect scores `defect`, on that same freshly minted specialist.
3. The target board is `unroutable` against the pre-existing registry **before** cold-start, and the animation shows every node outside its gate ring.
4. The cold-started specialist is visible from a second process/session without restart.

**Substance — what makes it credible under inspection:**

5. ≥5 electronics classes ingested from **verified-loadable** Hub datasets, provenance and licence recorded per class.
6. Top-1 routing on the electronics registry **measured and reported honestly**, including a negative result if `pcb1`–`pcb4` collide.
7. `pixel_auroc` non-null wherever masks exist — or an explicit statement of why not.
8. `finding_recall_idx` READY; `/recall` returns visually similar past findings for a real frame.
9. Routing animation driven by **live SSE steps**, showing route, refusal, and cold-start; legible in light and dark; reduced-motion fallback.
10. Voice agent answers ≥4 distinct fleet questions using MongoDB-backed tools, with no API key in the browser.

**Hygiene:**

11. README opens with an explicit separation of pre-existing work from work completed during the event window (§0.8).
12. The MVTec non-commercial constraint is stated on the registry UI, and the VisA/MVTec licence split (§0.9) is stated in the README.
13. No powerline framing anywhere in the pitch, README, or UI (§0.7).
14. `ruff` clean · `mypy` clean · `pytest` green · `pnpm build` green.
15. Nothing fabricated. Every metric traceable to an executed command.

# GridSight progress

**v1 is complete and measured. v2 is in flight.** See [SPEC.md](SPEC.md) for what v2 is and why.

---

## v1 — shipped, all criteria met

| phase | evidence |
|---|---|
| 1 — HF scrape | 6 asset classes, 120/30/40 each; provenance in `datasets` |
| 2 — schema + vector index | `model_router_idx` READY, created over the driver and polled |
| 3 — train + register | 5 specialists, GridFS round-trip **exact** (delta 0.0) |
| checkpoint resume | SIGKILL, `wait()` = −9, resumed without relearning |
| 4 — agent | routes, refuses, cold-starts in 3.3 s, re-routes at 0.9467 |
| 5 — API | all endpoints exercised against the live cluster |
| 6 — UI | screenshots, dark + light, no mobile overflow |
| 7 — verify | ruff clean · mypy clean · 25 tests · `pnpm build` green |

Baseline: top-1 routing **99/125 (79%)** · in-registry accepted **115/125** · out-of-registry
refused **24/25** · Atlas ~60 MB / 512 MB.

Per-class image AUROC: insulator 0.910 · conductor 0.834 · transmission_tower 0.717 ·
corrosion 0.628 · **vegetation 0.490**. Pixel AUROC `null` everywhere — no source dataset had masks.

---

## v2 — done so far

### Episodic recall — **complete and tested**

The registry answers *"which specialist knows this?"*. Findings now answer *"have we seen this
before?"* — the same maths over a different memory.

- Generalised the vector layer, which was hard-wired to `models.embedding` in **both** the `$vectorSearch`
  aggregation and the brute-force fallback. Now a `VectorIndex` dataclass drives both indexes.
- Persisted the frame embedding that `embed_frame` was already computing and **discarding**.
- Backfilled all 66 pre-existing findings by re-embedding their frames from GridFS — no image
  re-uploaded, no inference re-run (`scripts/build_recall_index.py`).
- `finding_recall_idx` **READY**, 66/66 findings searchable. `POST /recall` returns matches plus a
  recurrence summary (first seen / last seen / count).
- **10 new tests** in `tests/test_recall.py`, all passing.

Measured, live:
```
"have we seen this before?" for an insulator defect frame
  1.0000  insulator  defect   sev=0.9152  2026-07-19
  0.8407  insulator  nominal  sev=0.0     2026-07-17
  0.8265  insulator  defect   sev=0.9327  2026-07-21
```

One test failure worth recording, because it was the *data* being right rather than the code being
wrong: asserting a frame recalls *itself first* fails when the same frame was inspected several times
across demo runs — several findings then share an identical vector and tie at 1.0. The honest
invariant is that the query appears **among** the perfect matches, not that it wins an arbitrary tie.

### Registry animation — **complete and verified**

`web/components/RegistryMap.tsx` + `GET /registry/layout`.

Geometry is honest rather than decorative: radial distance is **`1 − similarity`, measured**, and each
spoke carries a tick at that model's coverage gate. A dot inside its tick is literally *"this frame is
within the competence this model learned"* — so **refusal is visible**, not merely asserted: every dot
sits outside every tick. Only the angle is illustrative (PCA rank, evenly spaced to avoid collisions),
and the caption says so.

Screenshots: `artifacts/screenshots/10-registry-map-routed.png`, `11-registry-map-refused.png`.

### MVTec AD — dataset verified, ingestion in flight

`foersben/mvtec-ad` chosen **by downloading it**, not by reputation: 15 categories, canonical layout,
per-category fetchable (so ingest never pays 5.3 GB), and **ground-truth masks present**. Rejected:
`Voxel51/mvtec-ad` + 3 byte-identical clones (flatten categories into `data/data_0/`, destroying
per-category access), `BrachioLab` (no masks), `katiehahm` (no train/good split).

**Licence is a real constraint:** CC BY-NC-SA 4.0, non-commercial, ShareAlike propagates. Fine for the
demo and research; **not** fine for shipping commercially with MVTec-derived weights. The commercial
story rests on the powerline corpus.

Mask integrity independently verified before trusting any pixel metric:

- bottle / hazelnut / screw: **40 defects / 40 masks each, 0 unpaired, 0 size mismatches, 0 non-binary,
  0 empty**; defect area fractions 0.14 %–27 % — real localisation signal.
- Anomalib **cannot silently misalign** them: it pairs by sorted order then asserts
  `image_stem in mask_stem` and raises `MisMatchError` otherwise. A misalignment crashes training
  rather than producing a fake pixel AUROC.

### Voice agent — architecture verified, build in flight

**OpenAI Realtime API over WebRTC with server-minted ephemeral secrets**, verified live against the
installed SDK (`openai 2.54.0`):

- `client.realtime` is **GA, not beta**; `client.beta.realtime` is legacy and unused.
- **1.02 s to first audio** vs **≥4.4 s** for a serial STT→chat→TTS pipeline (0.93 + ~1.2 + 2.29).
- Ephemeral `ek_` token scope tested directly: authenticates against `/v1/realtime/calls`, returns
  **401 against `chat.completions` and `models.list`**. The browser never sees `OPENAI_API_KEY`.

Tools execute **server-side** because they query MongoDB: `query_trends`, `search_findings`,
`recall_similar`, `registry_status`, `explain_refusal`.

---

## In flight right now

A build workflow is running two agents in parallel over disjoint file sets, then an independent
verifier that re-runs lint, mypy, pytest, `pnpm build`, and **re-measures combined routing**:

- MVTec ingest + training — 3 of 6 categories ingested (bottle, hazelnut, screw); training not started.
- Voice agent — `gridsight/voice.py` written (591 lines); UI not yet.

Open question the verifier will settle honestly: **does adding visually distinct categories actually
raise the 79 % routing number?** The hypothesis is yes, because the powerline classes are nested and
MVTec's are not — but it will be reported as measured, not as hoped.

---

## Carried forward, unfixed

**`vegetation` is still 0.490.** Diagnosed, not repaired: a supervised CLIP probe reaches only
**0.591** on those crops versus 0.972/0.975 for insulator/corrosion, so the label is barely in the
pixels. 35 % of "encroachments" are grazing box overlaps (an artifact of axis-aligned boxes around
diagonal cables) and 17 % of crops keep <20 % of the conductor, cropping out the evidence.

The general rule this establishes, which constrains every future deployment:

> PatchCore detects **appearance** anomalies. Not **relational** ones ("too close to"), not
> count-based ones, not logical ones. Relational defects need a different formulation, not a bigger model.

Resolution is either a corridor-anchored redefinition or documenting the class as unlearnable from
bbox geometry. **Shipping a leaky proxy is not an option.** Harness: `scripts/eval_veg_variant.py`.

## Not done

- **No Vercel deploy** — `VERCEL_TOKEN` absent from `.env`. `pnpm build` is green, so it is deploy-ready.
- **No GitHub push** — `GITHUB_TOKEN` absent, no remote configured. Committed locally on `main`.
- **Browser microphone untested** — the voice agent's server side can be proven from here; the mic
  path cannot, and will be reported as unverified rather than claimed.

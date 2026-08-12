# GridSight build progress

**Status: complete.** All seven phases shipped; all exit criteria met except deployment,
which is blocked on credentials that were never supplied (see the bottom of this file).

## Phase status

| phase | state | evidence |
|---|---|---|
| 1 — HF scrape | done | `data/ingest_report.json`, `datasets` collection, 6 classes @ 120/30/40 |
| 2 — schema + vector index | done | `model_router_idx` READY, `artifacts/verify_routing.log` |
| 3 — train + register | done | 5 models, exact GridFS round-trips, `artifacts/fleet_v3.log` |
| checkpoint resume | done | `artifacts/kill_resume/{run1,resume}.log` — SIGKILL, `wait()` = -9 |
| 4 — LangGraph agent | done | `artifacts/agent_demo.log` — all three scenarios |
| 5 — FastAPI | done | all endpoints exercised; `/health` reports index status |
| 6 — Next.js UI | done | `artifacts/screenshots/*.png`, dark + light, mobile no-overflow |
| 7 — verify | done | ruff clean, mypy clean (18 files), 25 tests pass, `pnpm build` green |

## Final measured numbers

Registry (5 seeded specialists; `rail_surface` deliberately withheld for the cold-start demo):

| class | image AUROC | routing gate | weights | train samples |
|---|---|---|---|---|
| insulator | 0.910 | 0.8399 | 7.55 MB | 120 |
| conductor | 0.834 | 0.8275 | 6.67 MB | 106 |
| transmission_tower | 0.717 | 0.8343 | 7.55 MB | 120 |
| corrosion | 0.628 | 0.8719 | 7.55 MB | 120 |
| vegetation | 0.490 | 0.8346 | 7.55 MB | 120 |

Pixel AUROC is `null` for every class: no Hub dataset here ships ground-truth masks.

Routing, live against Atlas over 150 held-out frames:
top-1 accuracy **99/125 (79%)**, in-registry accepted **115/125**, out-of-registry refused **24/25**.

Atlas M0 usage: ~60 MB of 512 MB.

## The two problems that were found and fixed

1. **Routing was 18/30 at first.** Cause, established by experiment: four classes were crops of the
   *same* aerial frames and are nested (a tower box contains insulators and cables). Centering the
   embedding space did not help (61% → 63%); tighter crops helped a little (69%); making **scale**
   the discriminator (towers ≥180px, components ≤160px, cropped tight and upscaled rather than
   padded) reached 88% on balanced pools. Shipped corpus: 79%.
2. **A single global threshold could not both accept and refuse.** Replaced with a **per-model
   coverage gate** = the 5th percentile of each model's own training-set similarity to its own
   centroid. Measured: 115/125 accepted + 24/25 refused, versus 136/125→91/125 for the best global
   cut at comparable refusal. `ROUTE_THRESHOLD` remains an additional floor.

Also fixed along the way: anomalib leaves `_pixel_threshold` as NaN without masks, which silently
produced all-NaN heatmaps and zero detected regions. Pixel stats are now calibrated from the p99.5 of
anomaly maps over known-good training images, and inference refuses to emit a NaN heatmap.

## Not done, and why

- **No Vercel deploy.** `VERCEL_TOKEN` was listed as `<paste or omit>` in the brief and is absent
  from `.env`, so there is nothing to authenticate with. `pnpm --dir web build` is green, so the app
  is deploy-ready.
- **No GitHub push.** `GITHUB_TOKEN` is likewise absent and the repo has no remote configured. Work
  is committed locally on `main`.

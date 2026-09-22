# Completed controlled-compute anticipation pilot

The separately identified second attempt completed all 12 recordings and all six SNN training/evaluation runs. It contains 18,007 real sensor readings and 16,802 eligible forecasting examples. The data comes from seeded PyTorch matrix/transfer/rest bursts, not an Ollama model or a Hermes bot.

**Result: none of the six SNN runs met the predeclared promising-pilot criteria.** Persistence was the strongest baseline selected on validation. The mixed-memory arm did not demonstrate a consistent improvement across seeds. Preserve this negative result; do not use held-out scores to choose replacement seeds or retune this campaign.

## Fixed experiment

The original [protocol](campaign-spec.md) was retained: 12 × 150-second sessions, 100 ms requested polling, sessions 1–6 training / 7–9 validation / 10–12 test, five causal hardware features, one- and five-second temperature/power deltas, identical eligible examples for all models, and training-only normalization. All six SNN runs completed 20 epochs. Checkpoints were selected on validation. Hidden input weights remained fixed; the forecasting readouts learned with time-resolved SynapticDistill OTTT.

Training and evaluation finished in 22.61 seconds within the shared 1,200-second budget. All collectors finalized with zero write failures. From 18,007 source rows, the ETL constructed 18,006 frames, all valid. It rejected 600 early frames without complete history, 600 late frames without future targets, and four examples whose first future observation exceeded the lateness allowance, leaving 16,802 eligible examples.

## Equal-session held-out errors

Primary error averages five-second temperature/power MAE scaled by their training target standard deviations. Lower is better. A promising result required at least 5% improvement over the validation-selected baseline and no greater than 5% degradation on either five-second target.

| Model | Seed | Validation primary | Test primary | +5s temperature MAE (°C) | +5s power MAE (W) | Promising |
|---|---:|---:|---:|---:|---:|---|
| persistence | — | 0.64196 | 0.54588 | 4.5429 | 68.7071 | baseline |
| current_ridge | — | 0.64942 | 0.56805 | 4.7411 | 71.2870 | baseline |
| history_ridge | — | 0.64261 | 0.58711 | 4.9012 | 73.6617 | baseline |
| uniform LIF | 123 | 0.63487 | 0.54936 | 4.5839 | 68.9590 | No |
| mixed LIF | 123 | 0.62235 | 0.54354 | 4.5550 | 67.9246 | No |
| uniform LIF | 456 | 0.61968 | 0.54303 | 4.5368 | 68.0763 | No |
| mixed LIF | 456 | 0.62376 | 0.56602 | 4.6420 | 72.2968 | No |
| uniform LIF | 789 | 0.63605 | 0.54642 | 4.5649 | 68.5047 | No |
| mixed LIF | 789 | 0.62418 | 0.54697 | 4.5442 | 68.9635 | No |

## Capture and provenance

| Session | Split | Source rows | Eligible examples |
|---|---|---:|---:|
| session-01 | train | 1500 | 1400 |
| session-02 | train | 1501 | 1401 |
| session-03 | train | 1501 | 1399 |
| session-04 | train | 1500 | 1400 |
| session-05 | train | 1500 | 1400 |
| session-06 | train | 1501 | 1401 |
| session-07 | validation | 1500 | 1400 |
| session-08 | validation | 1501 | 1399 |
| session-09 | validation | 1501 | 1401 |
| session-10 | test | 1500 | 1400 |
| session-11 | test | 1501 | 1400 |
| session-12 | test | 1501 | 1401 |

This run is local under `artifacts/anticipation-20260920-attempt2/` in the primary checkout. Raw Parquet, source/quality/split manifests, all six checkpoints, predictions, learning curves, per-session metrics at both horizons, spike diagnostics, PNG/SVG plots, runtime provenance and the artifact hash index are retained there. Large artifacts remain local.

- Capture/trainer/report source: `8761d0c5c8bb589bd2cb9a7d9199ce3ec2a711b6`.
- ETL source: `c649af4bda43e0faa113850a63fa59bdb5445355`.
- Collector source: `825645283612c59c3fed5a24ad713f813e8336cf` ([shutdown repair PR](https://github.com/rmems/gaming-telemetry/pull/51)).
- SynapticDistill source: `a1fa491c70397b96967ba6cf8f08c2ce2fbb2fb7`.
- `prepared/prepared.json` SHA-256: `ac646ec1103c16b8cd4395ab45a6e2856537a71f72f9b97e8bdc98381ba6b976`.
- `results/comparison.json` SHA-256: `95e48f368e85894f81ae1e74f4c40044f407e38cfa0e0338b0110b53b26607cf`.
- `artifact-index.json` SHA-256: `47c41fcfad0bc819f3aed9382fdcb9f0444c52e25964a9e2d74fe438aa801dc5`.

The [first failed attempt](pilot-2026-09-20.md) remains unchanged. Its session 3 did not finalize, and its partial recordings were not pooled into this campaign. The collector repair preserves one SIGINT receiver across collection waits; a real-signal regression failed before the repair and passed afterward. All 12 full recordings in this new attempt finalized successfully.

## Interpretation and next workload

This is a completed test of a small forecasting capability on controlled GPU bursts. It does not establish general workload anticipation or autonomous hardware control. Three held-out sessions support only a pilot conclusion. The existing exp-025 model remains lineage with a different task and feature contract.

The user subsequently selected actual Hermes bot workloads on installed Ollama models. That is a separate campaign using advertised maximum model contexts and measured residency, excluding muse-glimmer and nemotron-3.5-lightning. It must retain its own capture provenance and results; these controlled-compute scores must not be represented as Hermes/Ollama performance.

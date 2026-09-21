# Completed Hermes/Ollama anticipation pilot

The actual-model campaign completed all 12 recordings and all six SNN training/evaluation runs. It contains **18,001 real hardware sensor readings and 16,781 eligible forecasting examples** collected while Hermes ran tasks on installed Ollama models. The six SNN runs each completed 20 epochs; their forecasting readouts learned with time-resolved SynapticDistill OTTT. Ollama supplied the workload, not training labels or model inputs.

**Result: none of the six SNN runs met the predeclared promising-pilot criteria.** Persistence was the strongest baseline selected on validation and also beat every SNN on the held-out test sessions. Uniform-memory test error averaged 1.01% above persistence; mixed-memory error averaged 1.91% above it. Preserve this negative result without selecting replacement seeds or retuning on test results.

## Actual model configuration

Each model used its advertised architecture maximum context, verified through Ollama metadata and runtime residency. No 8k or 32k context override was used. These are configured context capacities, not a claim that every request filled its entire window. Actual reported tokens, cache usage, tool events and interruptions are retained in each task audit.

| Ollama model | Context tokens | Sessions | Total residency (GiB) | VRAM residency (GiB) | Derived CPU residency (GiB) |
|---|---:|---|---:|---:|---:|
| gemma4:12b | 262,144 | 1, 4, 7, 10 | 10.65 | 8.72–8.91 | 1.75–1.93 |
| granite4.2:8b | 131,072 | 2, 5, 8, 11 | 29.50 | 12.22 | 17.28 |
| Ornith-1.5-9B:latest | 262,144 | 3, 6, 9, 12 | 19.29 | 10.54 | 8.75 |

Muse Glimmer and Nemotron 3.5 Lightning were explicitly excluded. CPU offload was measured and retained as part of the workload condition. Existing user Hermes profiles were not rewritten; each run used a fresh campaign-owned Hermes home and scratch directory with a local Ollama endpoint.

## Capture and task outcomes

Protocol `hermes-ollama-inference-v2` preassigned sessions 1–6 to training, 7–9 to validation, and 10–12 to test. Each model appears twice in training and once in each held-out split. Every collector ran for 150 seconds at a requested 100 ms interval under `WORKLOAD_CLASS=ai-compute`. The model was preloaded before recording; a task began at second 20, with a 100-second parent timebox, five-second SIGTERM grace, verified model unload by second 130, and recording through second 150.

All 12 collectors finalized with exit code zero, zero write failures, and zero restarts. There were 18,001 valid causal frames and no invalid-input rejections. Excluding 600 frames without full history, 600 without future targets, and 20 late target matches left 16,781 examples. Normalization was fit only on training sessions; every comparison used the same eligible examples.

The bots made **51 observed tool calls**. Two tasks met the strict verified-complete criteria; ten were incomplete. Five task outputs passed the independent fixture verifier, including three whose nonzero bot exit kept their task outcome incomplete. All four Granite tasks reached the predeclared timebox and exited without SIGKILL. Their interruption token counters are marked partial/unknown rather than interpreted as zero inference. These are valid hardware workloads, not twelve successful agent tasks.

Tool access was constrained by the task prompt and isolated configuration, with an audit of explicit structured path arguments. Three malformed out-of-scratch read attempts were recorded in sessions 4, 10, and 12. This is not a filesystem sandbox or complete parsing of paths embedded in terminal commands. Raw tool inputs and outcomes remain available for inspection. Prompts, tool results, token counters, schedules, session IDs, and run position never enter Spikenaut inputs.

| Session | Split | Model | Source rows | Eligible examples | Tool calls | Bot outcome |
|---|---|---|---:|---:|---:|---|
| session-01 | train | gemma4:12b | 1500 | 1400 | 5 | incomplete |
| session-02 | train | granite4.2:8b | 1501 | 1401 | 2 | incomplete |
| session-03 | train | Ornith-1.5-9B:latest | 1500 | 1394 | 5 | verified_complete |
| session-04 | train | gemma4:12b | 1500 | 1394 | 6 | incomplete |
| session-05 | train | granite4.2:8b | 1500 | 1400 | 4 | incomplete |
| session-06 | train | Ornith-1.5-9B:latest | 1500 | 1400 | 5 | incomplete |
| session-07 | validation | gemma4:12b | 1500 | 1392 | 5 | verified_complete |
| session-08 | validation | granite4.2:8b | 1500 | 1400 | 2 | incomplete |
| session-09 | validation | Ornith-1.5-9B:latest | 1500 | 1400 | 4 | incomplete |
| session-10 | test | gemma4:12b | 1500 | 1400 | 5 | incomplete |
| session-11 | test | granite4.2:8b | 1500 | 1400 | 2 | incomplete |
| session-12 | test | Ornith-1.5-9B:latest | 1500 | 1400 | 6 | incomplete |

## Fixed forecasting comparison

Spikenaut observed VRAM occupancy, GPU power and temperature, graphics clock, and memory clock, predicting temperature/power deltas at one and five seconds. Hidden weights, neuron counts, membrane constants, seeds and learning rate followed the fixed protocol. Checkpoints and ridge regularization were selected on validation only. Training/evaluation took **19.73 seconds** within the shared 1,200-second budget, including the new startup/finalization reserve.

Primary error is equal-session five-second temperature/power MAE standardized by training target scales; lower is better. A promising result required at least 5% improvement over the validation-selected baseline, with neither five-second target more than 5% worse.

| Model | Seed | Validation primary | Test primary | +5s temperature MAE (°C) | +5s power MAE (W) | Promising |
|---|---:|---:|---:|---:|---:|---|
| persistence | — | 0.44878 | 0.49785 | 1.5686 | 21.1382 | baseline |
| current_ridge | — | 0.50877 | 0.55964 | 1.7052 | 24.6565 | baseline |
| history_ridge | — | 0.47576 | 0.52430 | 1.5896 | 23.2216 | baseline |
| uniform LIF | 123 | 0.45192 | 0.50301 | 1.5782 | 21.4590 | No |
| mixed LIF | 123 | 0.45517 | 0.50442 | 1.5850 | 21.4829 | No |
| uniform LIF | 456 | 0.45426 | 0.50272 | 1.5975 | 21.1368 | No |
| mixed LIF | 456 | 0.45680 | 0.50878 | 1.5684 | 22.1353 | No |
| uniform LIF | 789 | 0.45414 | 0.50291 | 1.5919 | 21.2391 | No |
| mixed LIF | 789 | 0.45834 | 0.50883 | 1.5766 | 22.0128 | No |

## Artifacts and provenance

The full local artifact root is `artifacts/anticipation-20260921-hermes/` in the primary checkout. It retains raw Parquet, manifests and quality reports, six checkpoints, all predictions and learning curves, per-session physical metrics at both horizons, spike diagnostics, PNG/SVG plots, Hermes streams and task verification, maximum-context preflights, preserved failed probes, runtime provenance, and an artifact hash index. Large data and isolated Hermes homes remain local.

An independent audit verified the hashes and sizes of all 66,557 indexed files, totaling 1,103,607,544 bytes (about 1.03 GiB). The 19 isolated Hermes homes from this campaign and its preflights account for most files and bytes through bundled tool binaries, language-server payloads, and skills. Every model covers the same 8,392 held-out examples: 4,192 validation and 4,200 test. Independently recomputed primary errors and physical MAE/RMSE agree with the saved comparison within `1e-12`. Ollama reported no resident models after completion.

- spikenaut source: `fc932a72a141e321097bb516731fb32834a417ae`.
- etl source: `c649af4bda43e0faa113850a63fa59bdb5445355`.
- collector source: `825645283612c59c3fed5a24ad713f813e8336cf`.
- synapticdistill source: `a1fa491c70397b96967ba6cf8f08c2ce2fbb2fb7`.
- `prepared/prepared.json` SHA-256: `0a9c86a05eddcc63e8430e23b7b50bf9f24d136def33bc725873b306673509f4`.
- `results/comparison.json` SHA-256: `3dbe96d49b9e2e070106d50b70f7bb571e17e5fb07b1821902826acc4ec17dc7`.
- `artifact-index.json` SHA-256: `73883c20a9710d02089768b01cd675308d563ce8bf101a026b1bb32f96e48c0d`.

The earlier [controlled-compute result](pilot-2026-09-21-compute.md) and [failed first attempt](pilot-2026-09-20.md) remain separate and unchanged. Their data was not pooled into this campaign. Primary scores from different campaigns use different training scales and should not be ranked as though they came from the same test set.

## Interpretation

This completes an actual-model test of a small hardware-forecasting capability. The learned readouts did not improve over persistence on these three held-out sessions, and mixed memory did not help consistently. That is evidence about this fixed 16-LIF pilot on one workstation and these bounded tasks; it does not establish broad workload generalization or an autonomous hardware controller. The next scientific change requires a separately declared experiment, not post-hoc tuning of this test set.

CodeRabbit completed its review of the implementation and raised seven findings. A single fix batch added Hermes CI coverage, process-group race handling, trainer timeout headroom, prepared target identifiers, a specific timeout assertion, and two documentation corrections. The fix passed 108 Python tests and an independent scoped review. The subsequent CodeRabbit review was rate-limited at the time of capture; external review and SonarCloud/Codacy status remain distinct from the completed experimental result.

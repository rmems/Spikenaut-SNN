# Train Spikenaut to anticipate its machine’s state

## Goal and first campaign

Develop one measurable capability of Spikenaut’s artificial nervous system: **use recent internal state to anticipate how the machine will change over the next one and five seconds**.

Your choices set the first campaign: fresh real telemetry, automated AI/compute workloads, approximately one hour of collection and experimentation, plus historical v3 cleanup through `spikenaut-telemetry-etl`. Preparing and testing the experiment tools happens before that one-hour runtime.

The repositories have these roles:

| Component | Role in this campaign |
|---|---|
| `gaming-telemetry` | Record the workstation’s actual sensors during controlled compute workloads; its collector already supports a configurable workload class. |
| `spikenaut-telemetry-etl` | Validate recordings, construct forecasting examples, preserve source provenance, and produce a cleaned v3 research view. |
| `SynapticDistill.jl` | Supply the Julia learning machinery for small SNN forecasting experiments. |
| `Spikenaut-SNN` | Own the experiment configuration, runner, candidate checkpoints, comparisons, and results report. |
| Historical v3 | Provide a separately audited historical reference. Its absent timestamps prevent evaluation in seconds. |
| `synthetic-factory`, `LiquidCortex.jl`, `Theseus-Quarry` | Follow-up options for controlled memory challenges, a larger temporal teacher, and mining-domain transfer, respectively. |

The local Julia trainer loads successfully, and PyTorch can access the RTX 5080 for workload generation. The published [v3 corpus](https://huggingface.co/datasets/rmems/Spikenaut-SNN-Telemetry) and [gaming corpus](https://huggingface.co/datasets/rmems/gaming-telemetry) remain separate sources.

## Data preparation and collection

- Add two ETL operations: `audit-v3` for historical data and `prepare-anticipation` for completed collector sessions. Both produce versioned manifests, quality reports, and explicit exclusion reasons.
- Audit v3’s state/outcome joins, duplicate identifiers, split membership, finite sensor values, suspicious zeros, and existing 64-sample targets. Preserve the published splits and original source rows.
- Produce an additive `v3-forecast-eligible-v1` view. Flag zero temperature and zero power/clock readings as suspect; exclude affected forecasting windows. Preserve original indices when checking the 64-sample horizon—filtering must never bring formerly distant rows together. Missing timestamps, rewards, and action labels remain missing.
- Record the observed distribution differences between v3 splits. Do not rebalance the historical test set based on its results. The audit already found **11,997 zero-temperature test readings**, compared with **13 in training**.
- Collect **12 fresh sessions of 150 seconds each**, requesting a **100 ms polling interval** and setting `WORKLOAD_CLASS=ai-compute`. Each session contains 20 seconds of idle observation, 110 seconds of seeded compute/memory bursts with varying durations, and 20 seconds of recovery.
- Generate workloads with bounded PyTorch matrix operations and host/device transfers. These are controlled stimuli; all training sensor values come from the collector. Limit workload allocations to 2 GiB and retain the workstation’s existing thermal and power controls.
- Assign sessions before capture: **1–6 training, 7–9 validation, 10–12 test**. Give each session a distinct stimulus seed. Record workload schedules for auditing; exclude schedules, session identifiers, and elapsed-run position from model inputs.
- Finalize each recording with a graceful collector shutdown. Require complete manifests, zero reported write failures, and sufficient valid windows in every assigned session. A deficient capture produces an incomplete campaign report rather than a silently changed split.

The controlled PyTorch stimulus above remains the binding original protocol.
The separately identified `hermes-ollama-inference-v1` variant in
[README.md](README.md) uses the same sensor, split, timing, ETL, and forecasting
contracts but different active-workload provenance. Captures from the two
protocols remain separate.

## Forecasting contract and training experiments

Create a separate experimental feature map using five observed quantities: **VRAM used, GPU power, GPU temperature, graphics clock, and memory clock**. Preserve their actual meanings; VRAM occupancy is not memory utilization, and graphics clock is not SM clock.

- Form causal 100 ms frames while retaining source timestamps and sample ages. Reject stale inputs older than 200 ms, clock reversals, and windows crossing session boundaries or invalid-data gaps.
- Define four targets: temperature change and power change at **+1 second** and **+5 seconds**. Match each future target to the first actual observation at or after its deadline, allowing at most 100 ms lateness. Missing future observations invalidate that example.
- Fit normalization only on training sessions. Freeze it for validation/test; record constant features and out-of-range values.
- Use the same eligible examples for every model, including a common five-second history warm-up.

Run this fixed comparison:

| Model | Purpose |
|---|---|
| Persistence | Predict zero change from the current temperature and power. |
| Current-state ridge regression | Test what the current sensor row alone can predict. |
| History ridge regression | Add causal sensor values from 0.5, 1, 2, and 5 seconds earlier. |
| 16-LIF SNN, uniform memory | Use a 0.5-second membrane time constant for every neuron. |
| 16-LIF SNN, mixed memory | Use four neurons each at 0.1, 0.5, 2, and 5 seconds. |

For the two SNN arms, use identical seeded input-weight initialization and fixed hidden weights; **train the four-output forecasting readout** with SynapticDistill’s time-resolved OTTT. This first experiment asks whether the neural state provides useful predictive information and whether multiple memory timescales help.

Use seeds **123, 456, and 789**, giving six SNN runs. Reset membranes and learning traces between sessions. Use a 0.5-second readout trace, supervised squared error on training-standardized target changes, learning rate `0.01`, and up to 20 epochs. Select each checkpoint using validation performance only. Select ridge regularization from `{0.001, 0.01, 0.1, 1, 10}` on validation.

Reserve approximately 20 minutes for training and evaluation. If the budget expires, retain completed checkpoints and mark unfinished comparisons explicitly. No run receives an unreported extension or a replacement seed.

The forecast checkpoint has its own feature-map identifier, four-output contract, time units, normalization, and source hashes. Existing exp-025 has a different task and input contract, so it will be documented as lineage rather than placed in a misleading direct performance comparison.

## Evidence, verification, and delivery

Use **five-second prediction error** as the primary result: average temperature/power MAE after scaling each target by its training-set standard deviation, with sessions weighted equally. Also report physical-unit MAE/RMSE at both horizons, per-session errors, variation across seeds, spike activity, silent neurons, and input rejection rates.

Predeclare a promising pilot result as:

- At least **5% lower primary error** than the strongest baseline selected on validation.
- No greater than **5% degradation on either five-second target**.
- Results reported for every seed and every test session.

Three test sessions support a pilot conclusion; they do not establish broad workload generalization. If the SNN loses, preserve that result and use the diagnostics to choose the next experiment.

Verify the machinery before capture:

- Invalid sensors, stale values, clock reversals, and incomplete manifests are detected.
- Future observations cannot enter inputs or normalization.
- History, targets, and neural state cannot cross session boundaries.
- Historical cleanup preserves original 64-sample distances and split assignments.
- Training changes readout weights; evaluation leaves weights unchanged.
- OTTT uses time-resolved outputs, and checkpoint reload reproduces predictions.
- A small deterministic fixture exercises the complete collection-format → ETL → training → report path.

Store raw recordings and run artifacts under a unique directory beneath Spikenaut’s existing `artifacts/` area. Deliver checkpoints, data-quality reports, split and provenance manifests, predictions, learning curves, and a comparison table with a plain-language conclusion about whether anticipation improved.

Publish the reusable ETL changes and experiment runner/report in their respective repository PRs. Large captures stay local for this pilot. Dataset publication and replacement of the shipped model bank are separate delivery steps.

# Machine-state anticipation pilot

This experiment forecasts changes in observed GPU temperature and power at one and five seconds. Its five inputs are VRAM occupancy (MiB), GPU power (W), GPU temperature (C), graphics clock (MHz), and memory clock (MHz). Graphics clock is not SM clock; occupancy is not utilization. The existing exp-025 model is lineage only: its inputs and task do not support a direct performance comparison.

Despite its name, `gaming-telemetry` records workstation sensors during these automated PyTorch workloads. No game needs to be running; the collector is explicitly labeled `WORKLOAD_CLASS=ai-compute`.

The binding protocol is [campaign-spec.md](campaign-spec.md). The campaign fixes 12 sessions before acquisition, each with 20 seconds idle, 110 seconds of seeded compute/transfer/rest bursts, and 20 seconds recovery. Sessions 1–6 train, 7–9 validate, 10–12 test. All five model families share the same eligible examples and five-second history requirement. At least 500 eligible examples are required in each actual session; failed captures produce an incomplete campaign, never a reassigned split.

## Environments and preflight

Use Python with NumPy and PyArrow for ETL/reporting, Python with CUDA-enabled PyTorch for acquisition, and Julia with SynapticDistill and JSON3. These can be separate environments. No shipped model-bank or GPU thermal/power settings are changed.

Build the existing `gaming-telemetry` collector with `cargo build --release --bin gaming-telemetry`. Install the sibling ETL PR with its `v3` extras. Record repository commits and source hashes with each run. For Julia, use the checked-out SynapticDistill project and its scripts environment (both are read-only dependencies):

```bash
export JULIA_LOAD_PATH="/path/to/SynapticDistill.jl:/path/to/SynapticDistill.jl/scripts:@stdlib"
julia --startup-file=no tools/anticipation/test_train.jl
python -m pytest -q
```

The deterministic synthetic collector-format fixture lives in `tests/anticipation_fixture.py`. Generate it in a separate preflight directory, prepare it with the ETL command below (its explicit minimum is one), run Julia training, and render the report. Fixture predictions are never pooled with real campaign predictions. A separate short hardware smoke recording verifies collector startup, actual sensor schema, and graceful finalization.

## Real campaign

Choose a new output directory below the primary checkout's `artifacts/`. The runner refuses an existing campaign manifest. It writes the split assignment and seeded schedules before acquisition. The schedule audit is excluded from model inputs. Tensors occupy approximately 176 MiB explicitly; allocation is bounded below 2 GiB, with a CUDA allocator cap and peak allocator reporting.

```bash
python -m tools.anticipation.campaign /path/to/Spikenaut-SNN/artifacts/unique-run \
  --collector /path/to/gaming-telemetry/target/release/gaming-telemetry
spikenaut-etl prepare-anticipation --input /path/to/unique-run/campaign.json \
  --output /path/to/unique-run/prepared
```

### Hermes agent workload variant

The separate `hermes-ollama-inference-v1` acquisition protocol measures the same
five collector sensors and retains the same 12-session split, timing, ETL,
forecast models, and training budget. Its active stimulus is one real, bounded
Hermes file-processing task per session instead of the PyTorch microbenchmark.
This is a distinct dataset source and must not be pooled with the controlled
PyTorch campaign.

The runner cycles `gemma4:12b`, `granite4.2:8b`, and
`Ornith-1.5-9B:latest` across the 12 sessions. Before each collector starts, it
reads the model architecture and maximum context from Ollama `/api/show`, loads
that exact maximum, and verifies the effective `/api/ps` context. It records
total, GPU-resident, and derived CPU-resident model bytes without requiring full GPU residency. It
never reduces the context, downloads a model, selects another model after a
failure, or falls back to direct generation. `muse-glimmer:30b` and
`nemotron-3.5-lightning:30b` are explicitly excluded.

Hermes runs with a fresh per-session `HERMES_HOME`, a dedicated synthetic
scratch directory, local custom-provider configuration, no provider fallback,
and only the `terminal` and `file` toolsets. Ambient rules, profiles, memories,
skills, plugins, MCP servers, and provider credentials are excluded. Each task
uses `--max-turns 4`, an 80-second Hermes run budget, and an independent hard
deadline at 120 seconds from sensor capture start. A session is not accepted as
an agent workload unless the stream records at least one tool call. Budget
limited runs retain their truthful status; process errors and hard timeouts make
the campaign incomplete. The runner unloads only the model it loaded and
verifies removal during cleanup.

Use a new output directory. Do not point this runner at an existing controlled
campaign or at a user Hermes profile:

```bash
python -m tools.anticipation.hermes_campaign \
  /path/to/Spikenaut-SNN/artifacts/unique-hermes-run \
  --collector /path/to/gaming-telemetry/target/release/gaming-telemetry
spikenaut-etl prepare-anticipation \
  --input /path/to/unique-hermes-run/campaign.json \
  --output /path/to/unique-hermes-run/prepared
```

The three named models must already be installed and the local endpoint must be
`127.0.0.1:11434`. The runner refuses to start if any model is already resident,
because unloading a user-owned model would disrupt another session. Maximum
context allocation can use both GPU and system memory and may fail on the local
machine; such a failure is recorded and stops the campaign rather than silently
changing the protocol. The original PyTorch invocation and its 2 GiB allocation
limit remain unchanged and apply only to that original protocol.

ETL retains source timestamps, sample ages, invalid frames, segment boundaries, source hashes, and immutable split assignments. Inputs are causal 100 ms frames with maximum source age 200 ms. Targets are the first actual observation at or after the frame deadline with at most 100 ms lateness. Invalid gaps interrupt history and neural state. Normalization uses training sessions only and records constant features and held-out values outside training ranges.

Training and evaluation share a 1,200-second maximum budget. The evaluation wrapper fits baselines first, then gives Julia the remaining budget, including process startup/loading. It refuses an existing evaluation directory, enforces the process deadline, and writes budget/unfinished-run evidence. Julia runs the six arm/seed combinations round-robin for up to 20 epochs and reserves 15% of its remaining budget for evaluation. It retains completed validation-selected checkpoints and labels unfinished work. The readout uses normalized exponential spike traces, four-output squared error, and time-resolved SynapticDistill OTTT. Hidden weights remain fixed. Each checkpoint includes input/target normalization, feature and output contracts, time constants in seconds, and source hashes.

```bash
python -m tools.anticipation.evaluate /path/to/unique-run/prepared/prepared.json \
  /path/to/unique-run/results --julia-version 1.12.7 --budget-seconds 1200
# Optional exported PNG/SVG charts require Matplotlib:
python -m tools.anticipation.plot /path/to/unique-run/results
```

The report selects the strongest baseline using validation only. Primary error is the equally weighted mean over sessions of temperature/power five-second MAE divided by training target standard deviation. Physical MAE/RMSE are also averaged over session metrics, with every session retained separately in JSON. A promising run needs at least 5% lower primary error and no more than 5% degradation on either five-second target. Every seed is reported; three test sessions support only a pilot conclusion.

## Historical reference

```bash
spikenaut-etl audit-v3 --input /path/to/Spikenaut-SNN-Telemetry \
  --output /path/to/unique-run/historical
```

The additive `v3-forecast-eligible-v1` view preserves published splits and original source indices. It audits state/outcome joins, identifiers, sensor values, and existing 64-sample targets without compressing gaps. Missing timestamps, rewards, and actions remain missing. Historical sample horizons cannot be scored as seconds. Do not rebalance the published test split based on its observed distribution.

Raw captures, checkpoints, predictions, manifests, and reports stay local for this pilot. Dataset publication and shipped model-bank replacement are separate delivery steps.

The separate `Anticipation Python` CI job installs the pinned dependencies in
`tools/anticipation/requirements-test.txt` and exercises failure handling and
scientific metrics without a GPU. The collector-format integration additionally
needs PyArrow and the sibling ETL package. Julia checks use the explicitly loaded
SynapticDistill dependency; a CUDA GPU is needed only for real acquisition.

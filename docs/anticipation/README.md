# Machine-state anticipation pilot

This experiment forecasts changes in observed GPU temperature and power at one and five seconds. Its five inputs are VRAM occupancy (MiB), GPU power (W), GPU temperature (C), graphics clock (MHz), and memory clock (MHz). Graphics clock is not SM clock; occupancy is not utilization. The existing exp-025 model is lineage only: its inputs and task do not support a direct performance comparison.

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

ETL retains source timestamps, sample ages, invalid frames, segment boundaries, source hashes, and immutable split assignments. Inputs are causal 100 ms frames with maximum source age 200 ms. Targets are the first actual observation at or after the frame deadline with at most 100 ms lateness. Invalid gaps interrupt history and neural state. Normalization uses training sessions only and records constant features and held-out values outside training ranges.

Training and evaluation share a 1,200-second maximum budget. Run baseline fitting first, then give Julia the remaining budget, including process startup/loading. Julia runs the six arm/seed combinations round-robin for up to 20 epochs and reserves 15% of its remaining budget for evaluation. It retains completed validation-selected checkpoints and labels unfinished work. The readout uses normalized exponential spike traces, four-output squared error, and time-resolved SynapticDistill OTTT. Hidden weights remain fixed. Each checkpoint includes input/target normalization, feature and output contracts, time constants in seconds, and source hashes.

```bash
julia --startup-file=no tools/anticipation/train.jl \
  /path/to/unique-run/prepared/prepared.json /path/to/unique-run/results/snn REMAINING_SECONDS
python -m tools.anticipation.report /path/to/unique-run/prepared/prepared.json \
  /path/to/unique-run/results
```

The report selects the strongest baseline using validation only. Primary error is the equally weighted mean over sessions of temperature/power five-second MAE divided by training target standard deviation. Physical MAE/RMSE are also averaged over session metrics, with every session retained separately in JSON. A promising run needs at least 5% lower primary error and no more than 5% degradation on either five-second target. Every seed is reported; three test sessions support only a pilot conclusion.

## Historical reference

```bash
spikenaut-etl audit-v3 --input /path/to/Spikenaut-SNN-Telemetry \
  --output /path/to/unique-run/historical
```

The additive `v3-forecast-eligible-v1` view preserves published splits and original source indices. It audits state/outcome joins, identifiers, sensor values, and existing 64-sample targets without compressing gaps. Missing timestamps, rewards, and actions remain missing. Historical sample horizons cannot be scored as seconds. Do not rebalance the published test split based on its observed distribution.

Raw captures, checkpoints, predictions, manifests, and reports stay local for this pilot. Dataset publication and shipped model-bank replacement are separate delivery steps.

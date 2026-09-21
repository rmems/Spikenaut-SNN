# Machine-state anticipation pilot

This experiment forecasts changes in observed GPU temperature and power at one and five seconds. Its five inputs are VRAM occupancy (MiB), GPU power (W), GPU temperature (C), graphics clock (MHz), and memory clock (MHz). Graphics clock is not SM clock; occupancy is not utilization. The existing exp-025 model is lineage only: its inputs and task do not support a direct performance comparison.

Despite its name, `gaming-telemetry` records workstation sensors during automated PyTorch or Hermes/Ollama workloads. No game needs to be running; the collector is explicitly labeled `WORKLOAD_CLASS=ai-compute`.

Completed results are available for the separate [controlled-compute campaign](pilot-2026-09-21-compute.md) and [maximum-context Hermes/Ollama campaign](pilot-2026-09-21-hermes.md). Both completed all 12 recordings and six SNN training runs; neither met the predeclared improvement criterion. Their captures, checkpoints, and predictions remain separate local artifacts.

The binding protocol is [campaign-spec.md](campaign-spec.md). The campaign fixes 12 sessions before acquisition, each with 20 seconds idle, 110 seconds of seeded compute/transfer/rest bursts, and 20 seconds recovery. Sessions 1–6 train, 7–9 validate, 10–12 test. All five model families share the same eligible examples and five-second history requirement. At least 500 eligible examples are required in each actual session; failed captures produce an incomplete campaign, never a reassigned split.

## Environments and preflight

Use Python with NumPy and PyArrow for ETL/reporting, Python with CUDA-enabled PyTorch for acquisition, and Julia with SynapticDistill and JSON3. These can be separate environments. The Hermes variant also requires Bubblewrap (`bwrap`) to verify agent-written Python without host filesystem or network access. No shipped model-bank or GPU thermal/power settings are changed.

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

The completed, sealed Hermes campaign used the `hermes-ollama-inference-v2`
protocol. The terminal-enabled source used `hermes-ollama-inference-v3`, which
added the finite preload lease and owned-request cleanup described below.
Current source emits `hermes-ollama-inference-v4`; it restricts Hermes to file
tools and keeps expected fixture answers outside the agent-visible scratch
directory. These variants
measure the same five collector sensors and retain the same 12-session split,
timing, ETL, forecast models, training budget, and bounded Hermes file-processing
tasks. They are distinct dataset sources and must not be pooled with the
controlled PyTorch campaign or represented as the same protocol version.

The runner cycles `gemma4:12b`, `granite4.2:8b`, and
`Ornith-1.5-9B:latest` across the 12 sessions. Before each collector starts, it
reads the model architecture and maximum context from Ollama `/api/show`, loads
that exact maximum, and verifies the effective `/api/ps` context. It records
total, GPU-resident, and derived CPU-resident model bytes without requiring full GPU residency. It
never reduces the context, downloads a model, selects another model after a
failure, or falls back to direct generation. `muse-glimmer:30b` and
`nemotron-3.5-lightning:30b` are explicitly excluded.

Every Ollama control-plane sequence has a 30-second absolute deadline. This
bounds `/api/version`, preflight and post-load `/api/ps`, and `/api/show` even
when a server drip-feeds bytes often enough to avoid the socket inactivity
timeout.

Preloading retains the 120-second logical campaign limit. The HTTP request is
owned by a worker with a separate 180-second completion deadline, and uses an
explicit 180-second Ollama residency lease instead of an indefinite keep-alive.
If the logical limit expires, capture fails, but finalization joins that same
request before unloading and verifying the exact model. A transport timeout or
error without confirmed request completion remains a hard cleanup failure; an
empty residency query is not reported as proof of cleanup. The finite lease is
defense in depth if the client exits. This follows Ollama v0.33.3's
[unload path](https://github.com/ollama/ollama/blob/v0.33.3/server/routes.go#L377-L387)
and [canceled-load cleanup](https://github.com/ollama/ollama/blob/v0.33.3/server/sched.go#L686-L706)
without assuming pending loads serialize behind later unload requests. These
lease and completion limits are recorded in the campaign resource envelope and
successful runtime metadata.

Protocol `hermes-ollama-inference-v4` runs Hermes with a fresh per-session
`HERMES_HOME`, a dedicated synthetic
scratch directory, local custom-provider configuration, no provider fallback,
and only the `file` toolset. Ambient rules, profiles, memories,
skills, plugins, MCP servers, and provider credentials are excluded. Each task
uses `--max-turns 4`, an 80-second Hermes run budget, and an independent hard
deadline at 120 seconds from sensor capture start. Prompts name each input and
output by absolute path while expected answers remain only in the parent harness,
outside the agent-visible scratch directory. Python candidates run only during
verification inside a Bubblewrap namespace with no network and read-only mounts
for the interpreter, candidate, and parent-owned verifier. A trusted parent runner
supplies fresh randomized normalization cases to a
subordinate candidate process and compares its untrusted results against answers
computed only in the parent. The worker receives no attestation secret or expected
answers. The `pure-normalization-v1` contract accepts a single `normalize` function
using a bounded AST subset: string/list operations, assignments, conditionals,
loops and comprehensions. Imports, I/O, reflection, private names, arbitrary calls
and nested functions are rejected before compilation. Only the listed pure
builtins and normalization methods are reachable from candidate code, so it
cannot inspect or impersonate its caller or write to the result channel. Ordinary
list-comprehension and loop solutions are covered by positive tests; frame and
builtins escapes are covered by negative tests. This is functional testing of
sampled behavior, not proof of general correctness. The sandbox omits `/proc`, and the worker
installs a libseccomp filter before importing candidate code that denies process
creation, cross-process memory access, and signaling. Address-space, CPU, and
file-size limits apply to the sandbox; outer stdout/stderr go to size-limited
regular files and host reads are capped at 64 KiB per stream. The launch-time
`RLIMIT_NPROC` ceiling permits Bubblewrap startup on busy shared-UID systems;
seccomp enforces the candidate no-fork boundary. Verification fails closed when
Bubblewrap, `prlimit`, or `libseccomp.so.2` is unavailable. This verifier sandbox
does not turn the
Hermes process itself into a general filesystem sandbox. A session is not accepted
as an agent workload unless the stream confirms the configured local model, at
least one permitted tool call, and a terminal result. Plain stdout diagnostics are
retained alongside parsed JSON events; malformed object-like lines are rejected.

Session cleanup receives the absolute 130-second acquisition deadline. If an
Ollama response exhausts that remaining budget, the session fails immediately
while a bounded, non-daemon worker retains ownership and reconciles the model.
This lets collector shutdown proceed without waiting for another full HTTP
cleanup budget. Deferred cleanup never counts as confirmed model absence.

An ordinary nonzero bot result remains a valid hardware workload when the
stream has positive usage and no explicit infrastructure error, while its task
outcome is recorded as incomplete. At the 100-second outer limit, the parent
sends SIGTERM. An interrupted run is accepted as `timeboxed` only when Hermes
emits its terminal result and exits during the five-second grace period without
SIGKILL. Its zero token counters are labeled partial rather than interpreted as
zero work. Task completion comes only from the known fixture verifier, never
from generated text. The runner unloads and verifies its exact owned model after
every bot run, before the recovery interval, and requires cleanup by 130 seconds.

Use a new output directory. Do not point this runner at an existing controlled
campaign or at a user Hermes profile:

```bash
python -m tools.anticipation.hermes_campaign \
  /path/to/Spikenaut-SNN/artifacts/unique-hermes-run \
  --collector /path/to/gaming-telemetry/target/release/gaming-telemetry \
  --hermes /path/to/hermes/venv/bin/hermes
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

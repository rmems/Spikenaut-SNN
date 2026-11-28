# TemporalFocus on the Spikenaut host and FPGA

The first usable integration is an offline host analysis of ordered stimulus
events. The current UART interface can carry a selected input event. The attention
kernels would require new state and arithmetic on the FPGA. Keep the shipped
bank as the analog reference when comparing event encodings.
[Feeder and live lanes](../../README.md#vahtisiru-feeder-and-live-input-lanes)
· [TemporalFocus exports, pinned revision](https://github.com/rmems/TemporalFocus.jl/blob/5eda961c0ed29d6bafb4a1086ee98921cef1408f/src/TemporalFocus.jl)

| Primitive | Placement | Contract and consequence |
|---|---|---|
| `SpikeEvent` | Host adapter; lane addressing already present in HDL | An event has a one-based neuron ID, Float32 time and Float32 value. Convert zero-based UART axon `a` to Julia ID `a+1`. Keep lane and time separate. |
| `SpikeTrain` | Host adapter | Preserves the vector's order and duplicates. Validate ordering at ingress and retain silent ticks separately to reconstruct the frames. |
| `TemporalBuffer`, `prune!` | Host candidate | A mutable history in the caller's time units. Pruning by age can retain future events. Feed only the prefix available at the tick being scored and reset at session boundaries. |
| `temporal_weight` | Host reference; possible FPGA approximation | Computes `exp(-abs(dt)/τ)` in Float32 for finite differences and positive finite τ. An HDL lookup table would need a time range, scale, rounding rule and error bound. |
| `spike_attention_discrete` | Host candidate | Accumulates products of values with the same ID, ignoring time, then applies a readout. Multiple context events contribute multiple times. AER routing only selects a destination. |
| `spike_attention_temporal` | Host candidate | Weights each matching-ID pair by temporal distance. An offline comparison at tick t must exclude future context to remain causal. |
| `spike_attention_continuous` | Host candidate | Bounds the gap between paired events by the smaller buffer window. Old pairs can still contribute until the caller prunes each buffer at the scoring tick. Hardware would need a bounded history, overflow policy and accumulator-width analysis. |
| `normalize_l1!`, `normalize_max!` | Host diagnostics only | Divide by a positive sum or positive maximum respectively. Empty vectors or nonpositive denominators pass through. Both change activation scale and are inapplicable to the shipped sensor min–max transform. |

The kernels apply a readout with one row per source neuron and one column per
output. With a 16×3 readout the result has three entries. Sharing that shape with
comfort, temperature and power establishes no agreement between attention and the
bank's keep-LIF dynamics. The exp-026 recurrence uses a one-tick delayed spike
vector with a signed matrix; none of the attention kernels implements that update.
[Kernels and normalization](https://github.com/rmems/TemporalFocus.jl/tree/5eda961c0ed29d6bafb4a1086ee98921cef1408f/src)
· [Upstream test vectors](https://github.com/rmems/TemporalFocus.jl/blob/5eda961c0ed29d6bafb4a1086ee98921cef1408f/test/runtests.jl)
· [Exp-026 recurrence](../recurrence/README.md)

## Replay boundary

Use `(axon, tick)` for an input event. Keep tick as an integer frame index
and record the session's configured interval separately (500 ms for the live
baseline). For Julia, choose one time unit for event timestamps, τ and the
retention window. With one source event on axon 2 at tick 0 and one matching
context event at tick 1, a unit readout yields 1 for discrete attention and
`exp(-1)` for temporal attention at τ = 1 tick.

Preserve 16 lanes in the logical stimulus. The current silicon consumes the
lowest nonzero lane, so at most one of lanes 0–4 may carry an event in each
frame of a board-compatible candidate. Lanes 5–15 stay zero. Encode the selected
amplitude on the signed Q8.8 grid. Every nonzero word on the selected lane
produces the same binary event regardless of sign or magnitude. Julia's event
value can retain both. Board comparisons therefore
need the emitted lane/tick sequence as well as the numeric frame values.
Compare against the FPGA's consumption tick. A new UART frame can replace a
pending event before the FPGA consumes it, so arrival order alone cannot
establish event alignment.
SpikeStream supplies event-stream features; converting raw telemetry into an
event sequence is a separate experimental choice under #97. Board spike,
membrane and action agreement remains unmeasured for this integration.
[UART decoder issue](https://github.com/rmems/silicon-hdl/issues/112)
· [Silicon reference and simulation harness](https://github.com/rmems/silicon-hdl/pull/116)
· [SpikeStream package](https://github.com/rmems/SpikeStream.jl/tree/a44980763558df083070dac7aeb2471d03a0df6c)

## Prepare a paired replay for SpikeStream

```sh
python -m tools.temporal_replay capture.jsonl --encoding logged > logged.jsonl
python -m tools.temporal_replay capture.jsonl --encoding delta > delta.jsonl
python -m tools.temporal_replay capture.jsonl --encoding delta --format events > delta.tsv
```

Use one contiguous live-loop capture with five measured `raw` values, the five
recorded wire `stim` values, gain and a tick index on each row. The command
checks the whole input before emitting frames. A missing sample, tick reset,
or legacy capture without raw values requires a different input. Both JSONLs
retain all ticks, including silence. The bridge's `replay_log` reads their
five compact stimulus entries and zero-fills the other eleven lanes. Match
the recording's tick interval explicitly when replaying it. The bridge numbers
its response ticks afresh; use frame position within each pass to align them
to the original capture ticks.
[Replay reader](https://github.com/rmems/silicon-bridge/blob/b0cdcf8776dbe98fcb1519185edddbc4ce74fb5e/examples/replay_log.rs)

`logged` preserves each recorded wire stimulus. The experimental `delta`
encoder applies frozen binary32 min–max scaling to the raw measurements. Its
first tick initializes all five anchors and emits silence. On later ticks it
sends a unit pulse for the lane with the largest absolute change from its
anchor if that change reaches 1/32. Ties select the lowest lane. Only the
winning lane updates its anchor, so a persistent change on another lane
remains eligible at the next tick. This threshold is an experimental setting.

The TSV contains integer `tick<TAB>axon` pairs for nonzero inputs. SpikeStream
accepts a zero-based ID as an opaque label and its constructor takes time first.
Use the full capture's start and exclusive end to count silent ticks in rates.
The example bounds below are specific to the Cyberpunk recording.

```julia
using SpikeStream
events = map(readlines("delta.tsv")) do line
    tick, axon = parse.(Int, split(line))
    SpikeEvent(tick, axon)
end
population_spike_features(events; t_start=1, t_end=4908, n_total_units=16)
```

On the 4,907-tick `live-fpga-log-cyberpunk2077-run1.jsonl` capture (SHA-256
`20f99b1bafae84e76c00f40311df9c056881696976ff6bc17c88fe2364489efb`),
the logged sequence drove 4,907 input events. The delta sequence drove 199,
distributed across lanes 0–4 as 7, 103, 61, 17 and 11. SpikeStream reported
five active units and 0.04055 input events per frame on the delta stream.
These measurements characterize the prepared inputs. Adopting a candidate
under #97 still requires paired board spike and membrane responses.
[SpikeStream revision used for the measurement](https://github.com/rmems/SpikeStream.jl/tree/c7b8cf6bcd40da9ae116c3d69d4c0cda2641d240)

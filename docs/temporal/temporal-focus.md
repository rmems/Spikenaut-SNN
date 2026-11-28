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
| `spike_attention_continuous` | Host candidate | Applies the temporal kernel within the smaller of two buffer windows. A hardware implementation would need a bounded history, overflow policy and accumulator-width analysis. |
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
amplitude on the signed Q8.8 grid. Two positive words addressing the same lane
can produce the same binary event on this decoder. Board comparisons therefore
need the emitted lane/tick sequence as well as the numeric frame values.
SpikeStream supplies event-stream features; converting raw telemetry into an
event sequence is a separate experimental choice under #97. Board spike,
membrane and action agreement remains unmeasured for this integration.
[UART decoder issue](https://github.com/rmems/silicon-hdl/issues/112)
· [Silicon reference and simulation harness](https://github.com/rmems/silicon-hdl/pull/116)
· [SpikeStream package](https://github.com/rmems/SpikeStream.jl/tree/a44980763558df083070dac7aeb2471d03a0df6c)

# SynapticDistill readout from a frozen exp-025 replay

Run from the repository root. The default replay uses the ten-tick method fixture.
For a recorded session, pass the five-sensor projection and its episode IDs to
`replay_frozen` with `--jsonl`. This adapter consumes its existing trace and
manifest. It reuses the Julia environment pinned to SynapticDistill `a1fa491`.

```sh
python -m tools.replay_frozen --out-dir replay_out
julia --startup-file=no --project=tools/anticipation -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=tools/anticipation tools/distill_readout.jl replay_out
```

| Stage | Mapping |
|---|---|
| Hardware capture → software reference | Project each successful contiguous raw poll to `episode_id` and the five named sensors in live order. Preserve acquisition order and use a distinct episode per session. The frozen analog adapter supplies a 16-wide input and resets the hidden state at each episode boundary. |
| Frozen hidden state → `SpikeBatch` | For each step, a zero-based hidden neuron ID sets row `id + 1` in a 16×1 Float32 binary matrix. The readout consumes hidden spikes. The rotated wire stimulus and a capture's aggregate spike count cannot identify those neurons. |
| Shipped weights → Julia model | The provenance digest binds the 48 signed Q8.8 readout words. In each neuron-major group of three, the words are comfort, temperature, power. Julia's column-major reshape gives `weights[head, hidden]` with shape 3×16. |
| `model_step(model, batch)` → output | `logits = vec(weights * batch.spikes)` gives the three same-tick scores. Each is checked exactly against the corresponding frozen replay score. The 16-wide hidden output in NIR and the three regression heads are distinct interfaces. |
| Labels for a future trainer | Recover the exp-025 targets from the normalized stimulus using `comfort = clamp(1 - 0.5*temp - 0.5*power, 0, 1)`. A missing temperature/power reading or zero-temperature sentinel withholds the target. A future run must separate training and held-out sessions and reset learning traces between episodes. |

The checker verifies the checkpoint and readout bytes against the [promoted
artifact manifest](../dataset/merged_v2/provenance.json) and the replay trace
against the digest in its manifest. It reads the weights once and performs a
frozen forward pass. The regression test separately calls `train_step!` on
private weight copies for both learning rules using a synthetic labeled tick.

**Recorded mapping check.** The Cyberpunk 2077 run-1 raw capture had 4,907 ticks,
SHA-256 `20f99b1bafae84e76c00f40311df9c056881696976ff6bc17c88fe2364489efb`.
Its five-sensor projection as one episode (`gpu-000000`) produced frozen
software trace `bdf584f63e0e9cff98f002cf53006cea0416a27234ff789b2314b110773ce105`
with the default `k=4`. All **14,721** Julia readout values matched that
trace exactly. This qualifies the handoff for a readout experiment. Recorded
training runs, teacher quality and comparison with board responses remain
open under [#95](https://github.com/rmems/Spikenaut-SNN/issues/95).

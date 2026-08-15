# merged_v2 retrain (signed E/I + K-WTA)

Source trainer: rmems/SynapticDistill.jl `scripts/spikenaut_train.jl` (PR https://github.com/rmems/SynapticDistill.jl/pull/30).
Data: `qubic_ticks_snn.jsonl` (27,430 rows), 20 epochs.
Decay is a **keep** factor (0.85 → `00DA`); Rust leak = `1 - keep`.
Hidden: mixed-sign, Dale 80:20 (rows 13–16 inhibitory), K-WTA k=4.
Q8.8: signed two's complement. `parameters_output_weights.mem` is 48 values.

Cite: Grok Build: Grok 4.6 (high)

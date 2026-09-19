# Frozen-replay method fixture (RM-1692 / GH #60)

`telemetry.jsonl` is a **synthetic, hand-written** v3 `state_telemetry`
input for `tools/replay_frozen.py`: two sessions (`gpu-000001`,
`gpu-000002`), five rows each, covering the nominal encode path plus one
absent live column (`sm_clock_mhz`, line 4) and one explicit `null`
(`mem_clock_mhz`, line 5) so the per-sensor `missing` trace field is
exercised.

This is a method fixture: it proves the replay pipeline composes and is
deterministic. It is **not** measured hardware telemetry and **not** a
held-out research result — do not quote numbers from it as evaluation
evidence.

Invocation:

    python3 tools/replay_frozen.py            # fixture input, shipped bank
    python3 tools/replay_frozen.py --jsonl PATH/state_telemetry.jsonl \
        --split test --out-dir out/

# Recurrence-comparison method fixture (GH #3 / exp-026)

`telemetry.jsonl` is a **synthetic, hand-generated** v3 `state_telemetry`
input for `tools/recurrence_eval.py`: five 32-row sessions -- two in the
train range (`gpu-000001`, `gpu-000002`), one in the validation range
(`gpu-000140`) and two in the test range (`gpu-000170`, `gpu-000171`) --
built from smooth cosine ramps of the five live columns, plus one absent
live column (line 6) and one explicit `null` (line 101).

It exists so CI can prove the comparison pipeline composes end to end:
stepping every arm, fitting the Dale-constrained readouts on the train
sessions, selecting on validation, scoring the test sessions, and writing
a deterministic `comparison.json`. It is **not** measured hardware and
**not** a research result -- do not quote numbers from it as evidence.
The recorded exp-026 comparison ran on the Hub v3 `state_telemetry`
shards; see `docs/recurrence/`.

Invocation:

    python3 tools/recurrence_eval.py             # fixture input
    python3 tools/recurrence_eval.py --jsonl PATH/state_telemetry.jsonl \
        --out-dir out/

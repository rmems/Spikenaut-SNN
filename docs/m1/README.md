# M1 machine interoception benchmark: cohort preflight

[GitHub #21](https://github.com/rmems/Spikenaut-SNN/issues/21) /
[RM-2322](https://linear.app/rpd-34/issue/RM-2322/experiment-benchmark-nervous-system-feature-families-and-spike)
needs an identical, session-disjoint cohort for later feature/encoder comparisons.
The frozen bank and the reported exp-026 baseline comparison remain unchanged.

`python3 tools/m1_cohort.py` (also `python3 -m tools.m1_cohort`) prints a
deterministic JSON report for the existing 160-row synthetic recurrence fixture.
It is a method check, not held-out hardware evidence.

For a local cohort, project the data to `episode_id` and the five live sensor
columns described in the [exp-026 projection](../recurrence/README.md#held-out-data),
then pass:

```sh
python3 tools/m1_cohort.py --jsonl state_telemetry.jsonl \
  --split-manifest cohort-splits.json --source-revision "DATASET_REVISION"
```

Use schema `spikenaut.split-manifest.v1` and set `train`, `val`, and `test`
to nonempty lists of whole ASCII `gpu-NNNNNN` episode IDs. Include every input episode
exactly once. Unlike replay's slice selector, preflight refuses silently dropped
rows or manifest-only episodes, as well as cross-split overlap, interleaved
sessions, duplicate row keys, nonfinite sensor inputs, and columns outside the
five-sensor projection. File order within each episode is retained; no clock
or chronological order is inferred. For valid rows it reports session/tick
counts and sensor availability. Null/absent readings and the v3 temperature
zero-dropout sentinel are counted separately from observed readings. Zero on
other sensor columns is an observation. A tick with all five sensors absent is
kept with all five marked missing.

The SHA-256 values cover the exact bytes consumed. Replace `DATASET_REVISION`
with the dataset revision you used; it is optional and is not authenticated.
The fixture, including equivalent copies with re-ordered JSON keys or different
line endings, cannot be relabeled by setting a revision.

This only checks the five-sensor projection. It does not qualify the full #20
state contract, estimate stale data without trustworthy timestamps, fit
normalization, evaluate an encoder, or run a checkpoint. No hardware,
training, or safety-action result is produced.

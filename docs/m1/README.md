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
  --split-manifest cohort-splits.json --source-revision <dataset-revision>
```

Use schema `spikenaut.split-manifest.v1` and set `train`, `val`, and `test`
to nonempty lists of whole `gpu-NNNNNN` episode IDs. Include every input episode
exactly once. Unlike replay's slice selector, preflight refuses silently dropped
rows or manifest-only episodes, as well as cross-split overlap, interleaved
sessions, duplicate row keys, nonfinite sensor inputs, and columns outside the
five-sensor projection. For valid rows it reports session/tick counts and sensor
availability, counting null/absent readings separately from observed zero.
The SHA-256 values cover the exact bytes consumed. A supplied source revision is
a declaration; the tool does not access the dataset provider or authenticate
measurement provenance. The fixture cannot be relabeled by setting a revision.

This only checks the five-sensor projection. It does not qualify the full #20
state contract, estimate stale data without trustworthy timestamps, fit
normalization, evaluate an encoder, or run a checkpoint. No hardware,
training, or safety-action result is produced.

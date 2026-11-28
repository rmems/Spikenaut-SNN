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


## RAW / KINETIC / HYBRID coverage

[GitHub #14](https://github.com/rmems/Spikenaut-SNN/issues/14) /
[RM-2319](https://linear.app/rpd-34/issue/RM-2319/research-consume-kinetic-signals-before-axon-encoder-and-benchmark-raw)
uses the same preflight input. With Python 3.11+, run `python3 tools/m1_arm_plan.py` (or
`python3 -m tools.m1_arm_plan`) with the same input flags. The plan audits
how many identical ticks could be scored by all three arms before any
encoding or training is attempted.

- RAW: the five observed hardware sensors at the current row.
- KINETIC: eight-row trailing SMAs of temperature and power, matching the
  current Rust SMA window; eligibility requires full windows, not its
  partial warm-up estimates. No feature values are computed in this audit.
- HYBRID: raw and kinetic inputs side by side. A row is eligible only when
  all features in both families are available.

The availability pass uses only the current and earlier rows and resets
at episode boundaries. An absent/null reading or the v3 zero-temperature dropout sentinel resets the
affected SMA history. Zero on other sensors is observed data. It reports per-split common row counts
and a digest over the eligible episode/source-line sequence, so a
subsequent evaluation can check that all arms used the same rows. An
empty split is visible in `splits_without_common_rows`; it is not ready
for evaluation.

All three experimental inputs propose the same rate encoder so the arms
vary by feature family. The lockfile pins crates.io versions and checksums. Physical
cadence must be established separately; row counts do not imply a clock.
Normalization must be fitted on train only by a future run. Each spike arm
requires a separate compatible checkpoint. The attested exp-025 digest is
recorded solely as an analog reference. There is no route through this tool
to run that bank on spikes, and no checkpoint or measured result is produced.

Run the arm-planning regression tests with
`python3 -m unittest tests.test_m1_arm_plan -v`.

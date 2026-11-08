# exp-026: recurrence vs the frozen exp-025 baseline

Input `hub:rmems/Spikenaut-SNN-Telemetry v3/state_telemetry/{train,validation,test}-00000.parquet projected to episode_id + five live columns, file order (see docs/recurrence/README.md)` (sha256:b1e601077100aabcc24021bcf86b8eeabbd9f5c1140623cdbc4ea08d946372da); k = 4, I-drive = 0.05. Splits: train 569344 ticks / 139 sessions, val 118784 / 29, test 117653 / 29.

Primary = mean over the three heads of the session-weighted MAE (equal weight per session) in target units [0, 1]; lower is better.

| arm | gain | lambda_rel | val primary | test primary | test MAE comfort | test MAE temp | test MAE power | spk/tick | patterns | cofire | I spikes | silent | decision agreement vs ff-shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `ff-shipped` | - | - | 0.0226 | 0.1055 | 0.1741 | 0.0693 | 0.0731 | 3.591 | 268 | 0.264 | 207005 | 6,7,8,9 | 1.000 |
| `ff-refit` | - | 0.0001 | 0.0369 | 0.1068 | 0.1984 | 0.0556 | 0.0664 | 3.591 | 268 | 0.264 | 207005 | 6,7,8,9 | 1.000 |
| `rec-g0.0625` | 0.0625 | 0.0001 | 0.0359 | 0.1061 | 0.1971 | 0.0541 | 0.0672 | 3.591 | 273 | 0.264 | 207005 | 6,7,8,9 | 1.000 |
| `rec-g0.125` | 0.125 | 0.0001 | 0.0191 | 0.0967 | 0.1724 | 0.0511 | 0.0665 | 3.568 | 418 | 0.262 | 207005 | 6,7,8,9 | 1.000 |
| `rec-g0.25` | 0.25 | 0.0001 | 0.0388 | 0.1487 | 0.2640 | 0.1138 | 0.0683 | 3.265 | 345 | 0.247 | 207005 | 6,7,8,9 | 1.000 |
| `rec-g0.375` | 0.375 | 0.01 | 0.2519 | 0.3602 | 0.6596 | 0.3471 | 0.0738 | 2.180 | 246 | 0.169 | 207005 | 6,7,8,9 | 1.000 |
| `rec-g0.5` | 0.5 | 0.0001 | 0.3998 | 0.4114 | 0.7564 | 0.4060 | 0.0717 | 1.904 | 110 | 0.140 | 207005 | 6,7,8,9 | 1.000 |

## Baselines (float, not quantized)

| baseline | val primary | test primary | test MAE comfort | test MAE temp | test MAE power |
|---|---:|---:|---:|---:|---:|
| `constant-train-mean` | 0.0118 | 0.0816 | 0.0791 | 0.1014 | 0.0645 |
| `linear-inputs-ridge` | 0.0012 | 0.0090 | 0.0043 | 0.0099 | 0.0129 |
| `persistence` | 0.0005 | 0.0006 | 0.0006 | 0.0006 | 0.0007 |

## Verdict

Validation-selected recurrent arm: `rec-g0.125`.
Against `ff-refit`: primary improvement +9.48%, worst head degradation +0.23%; promising = **True**.
Against `ff-shipped`: primary improvement +8.40%.
Best arm on test is `rec-g0.125` at 0.0967; it does not beat the `constant-train-mean` baseline (0.0816).

Criterion (predeclared): >= 5% lower test primary than the refit feed-forward arm and no head > 5% worse; selection on validation only.

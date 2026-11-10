# exp-026: recurrence vs the frozen exp-025 baseline

Input `hub:rmems/Spikenaut-SNN-Telemetry v3/state_telemetry/{train,validation,test}-00000.parquet projected to episode_id + five live columns, file order (see docs/recurrence/README.md)` (sha256:b1e601077100aabcc24021bcf86b8eeabbd9f5c1140623cdbc4ea08d946372da); k = None, I-drive = 0.05. Splits: train 569344 ticks / 139 sessions, val 118784 / 29, test 117653 / 29.

Primary = mean over the three heads of the session-weighted MAE (equal weight per session) in target units [0, 1]; lower is better.

| arm | gain | lambda_rel | val primary | test primary | test MAE comfort | test MAE temp | test MAE power | spk/tick | patterns | cofire | I spikes | silent | decision agreement vs ff-shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `ff-shipped` | - | - | 0.9222 | 0.8538 | 1.4013 | 0.9004 | 0.2596 | 5.628 | 165 | 0.852 | 223637 | 6,7,8,9 | 1.000 |
| `ff-refit` | - | 0.1 | 0.2233 | 0.2617 | 0.4755 | 0.2373 | 0.0724 | 5.628 | 165 | 0.852 | 223637 | 6,7,8,9 | 1.000 |
| `rec-g0.0625` | 0.0625 | 0.1 | 0.2274 | 0.2822 | 0.5116 | 0.2626 | 0.0725 | 5.171 | 194 | 0.803 | 223637 | 6,7,8,9 | 1.000 |
| `rec-g0.125` | 0.125 | 0.1 | 0.2611 | 0.3132 | 0.5724 | 0.2947 | 0.0725 | 4.500 | 328 | 0.746 | 223637 | 6,7,8,9 | 1.000 |
| `rec-g0.25` | 0.25 | 0.1 | 0.3171 | 0.3476 | 0.6212 | 0.3471 | 0.0746 | 3.588 | 354 | 0.758 | 223637 | 6,7,8,9 | 1.000 |
| `rec-g0.375` | 0.375 | 0.1 | 0.3622 | 0.3893 | 0.6923 | 0.3999 | 0.0757 | 2.602 | 273 | 0.569 | 223637 | 6,7,8,9 | 1.000 |
| `rec-g0.5` | 0.5 | 0.01 | 0.4111 | 0.4186 | 0.7372 | 0.4408 | 0.0778 | 1.931 | 86 | 0.422 | 223637 | 6,7,8,9 | 1.000 |

## Baselines (float, not quantized)

| baseline | val primary | test primary | test MAE comfort | test MAE temp | test MAE power |
|---|---:|---:|---:|---:|---:|
| `constant-train-mean` | 0.0118 | 0.0816 | 0.0791 | 0.1014 | 0.0645 |
| `linear-inputs-ridge` | 0.0012 | 0.0090 | 0.0043 | 0.0099 | 0.0129 |
| `persistence` | 0.0005 | 0.0006 | 0.0006 | 0.0006 | 0.0007 |

## Verdict

Validation-selected recurrent arm: `rec-g0.0625`.
Against `ff-refit`: primary improvement -7.82%, worst head degradation +10.66%; promising = **False**.
Against `ff-shipped`: primary improvement +66.94%.
Best arm on test is `ff-refit` at 0.2617; it does not beat the `constant-train-mean` baseline (0.0816).

Criterion (predeclared): >= 5% lower test primary than the refit feed-forward arm and no head > 5% worse; selection on validation only.

# Anticipation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement and review these tasks.

**Goal:** Execute the supplied causal forecasting pilot and publish reproducible ETL and experiment code.
**Architecture:** ETL owns raw validation and causal examples; Spikenaut owns acquisition, Julia readout training, common comparisons, and reporting. Historical v3 is an independent additive audit.
**Tech Stack:** Python, PyArrow, NumPy, PyTorch, Julia, SynapticDistill, existing Rust collector.
**Spec:** docs/anticipation/campaign-spec.md (binding full values and acceptance criteria).

## Global Constraints

All numerical protocol requirements in the spec apply verbatim. No changes to shipped model bank or dataset publication. Raw artifacts live beneath the primary Spikenaut artifacts directory. Preflight precedes campaign timing. Train/validation/test remain sessions 1–6/7–9/10–12.

### Task 1: Fresh telemetry ETL
Files: ETL src/spikenaut_etl/anticipation.py, CLI registration, tests/test_anticipation.py.
Implement prepare-anticipation against completed collector Parquet sessions. Input campaign JSON has sessions [{session_id, split, path, seed}]. Output prepared.json has schema_version, feature_map, target_names, sessions [{session_id, split, frames, examples}], normalization, quality, provenance. Each frame: timestamp_ms, source_timestamp_ms, age_ms, segment_id, x (five values). Each example: frame_index, history_indices (0.5,1,2,5s), target_timestamps_ms, y (temperature1,power1,temperature5,power5). Feature units MB,W,C,MHz,MHz. Retain frames for neural warm-up and gaps. Training-only mean/std with zero variance scale 1. Explicit rejects and complete manifests required. Communicate any necessary schema extension before integration.
- [ ] Write and run failing tests for causality, future deadline, stale and invalid gaps, reversals, normalization leakage, split/session boundaries, manifest failures.
- [ ] Implement operation and run full ETL tests/lint.

### Task 2: Historical audit
Files: ETL src/spikenaut_etl/audit_v3.py, tests/test_audit_v3.py; root integrates CLI to avoid shared edit.
Input original local v3 corpus. Output versioned audit report, manifest with source hashes, additive v3-forecast-eligible-v1 Parquet rows with original indices and splits preserved. Audit joins/duplicates/split membership/finite readings/suspicious zeros/64-sample outcomes. Never compress index distances; absent fields remain missing. Report per-split distributions and exact exclusion reasons.
- [ ] Write and run adversarial failing tests.
- [ ] Implement, test, audit full corpus; preserve originals.

### Task 3: Julia readout trainer
Files: Spikenaut tools/anticipation/train.jl, test_train.jl, Project.toml.
Consume prepared.json from Task 1. Produce predictions.json per arm/seed, checkpoint.json and learning curves; agreed CLI train.jl PREPARED OUTPUT BUDGET_SECONDS. Use time-resolved SynapticDistill OTTT for fixed 16-LIF hidden weights; exact seeds, membrane and trace constants, four outputs, loss and epochs in spec. Freeze normalized values; reset state/traces each session and segment; train only eligible rows with warm-up. Return per-example prediction with session_id/frame_index; diagnostics. Shared total time budget across six runs; mark unfinished. Checkpoint reload and unchanged evaluation weights tests required.
- [ ] Write and run failing learning/time-resolution/reset/reload tests.
- [ ] Implement and validate deterministic fixture.

### Task 4: Capture, comparisons, end-to-end verification
Files: Spikenaut tools/anticipation/campaign.py, workload.py, report.py; tests/test_anticipation_campaign.py.
Preassign sessions, persist seeds and schedules; monotonic idle/burst/recovery timing, bounded tensors, graceful SIGINT. Verify all manifests. Fit and validation-select ridge baselines; session-weighted standardized primary, physical MAE/RMSE, per-session and all-seed reporting. Predeclare criteria. Run full collection-format fixture through ETL/Julia/report before capture.
- [ ] Write tests for schedules/allocation limits/metrics/frozen normalization.
- [ ] Implement integration, execute preflight, then capture 12 sessions and 20-minute training budget.

### Task 5: Review and publish
Review both diffs against full spec, address findings, run targeted and repository checks. Commit code and compact results report; push dedicated branches and create linked PRs using GitHub plugin. Record local artifact paths, exact commits/hashes, completed and incomplete comparisons, and evidence-based conclusion.

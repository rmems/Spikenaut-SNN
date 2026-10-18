@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Layout, toolchain, feature list, full CI command list, and commit conventions are in
`AGENTS.md` (imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Rust 1.98.1 and Python 3.13 (Linux, CPU only).
`rust-toolchain.toml` makes rustup install 1.98.1 on first use.

| Command | Status |
|---|---|
| `cargo fmt --check` | verified |
| `cargo clippy --locked --all-targets --features myelin,training -- -D warnings` | verified |
| `cargo test --locked` | verified |
| `cargo test --locked --features myelin,training` | verified |
| `cargo check --locked --no-default-features` | verified |
| `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --features myelin,training` | verified |
| `python3 tools/verify_q88.py` (+ `--self-test`) | verified |
| `python3 tools/measure_hamming.py --self-test` | verified |
| `python3 tools/live_stim_parity.py`, `python3 tools/decision_parity.py`, `python3 tools/check_model_card.py` | verified |
| `python3 tools/verify_model_bank.py --select merged_v2` | verified |
| `python3 tools/replay_frozen.py` | verified (output dir `/replay_out/` is gitignored) |
| `python3 -m unittest tests.test_model_bank tests.test_replay_frozen` | verified (65 tests, 2 skipped) |
| Julia anticipation tests (`tools/anticipation`) | not run (Julia not installed) |
| Hermes/anticipation `pytest` block | not run (needs `bwrap`, a host `sysctl` change, and hash-pinned NumPy/pytest) |
| Remaining `python3 -m tools.*` variants and `--self-test` flags in `AGENTS.md` | not run individually; CI runs both script and `-m` forms |

Run fmt and clippy before committing. Never pass `--all-features` (enables `myelin-cuda`, needs CUDA 13.2+).

## Architecture map

- `src/lib.rs`: crate root. `model.rs` + `mem.rs` + `json.rs` load the `merged_v2` bank; `graph.rs` builds the NIR graph.
- `src/stim.rs`, `src/encode.rs`: live exp-025 front end (`LiveStimAdapter`) and the deprecated `TelemetryEncoder` channel map.
- `src/decision.rs`, `src/decision_helpers.rs`: output-row decision contract (argmax, fail-closed).
- `src/silicon.rs`, `src/kinetic.rs`, `src/critic.rs`, `src/wiring.rs`, `src/ipc.rs`, `src/neuromod_host.rs`: thin glue over the published sibling crates.
- `src/training.rs` (feature `training`), `src/myelin.rs` (feature `myelin`).
- `dataset/merged_v2/` + `config.json`: the shipped bank; `tools/*.py` verify it. Python mirrors of Rust logic live in `tools/*_core.py`.
- `tests/*.rs` are Rust integration tests; `tests/test_*.py` are Python. `pytest.ini` sets `pythonpath = .`.

## Invariants and non-goals

- Research artifact, not a validated supervisor; FPGA parity is unproven (README "Status"). Do not write docs or comments that claim otherwise.
- `Cargo.toml` has `publish = false`, so there is no `exclude`/`include` list to update.
- Dependencies resolve from crates.io only; do not add git or path dependencies. `Cargo.lock` is tracked, so `--locked` is correct.
- Q8.8 encoding is a shared contract with the FPGA side. Re-run the `tools/` verifiers after any encoding change.
- Do not change the shipped `dataset/merged_v2/` weights as a side effect of other work.
- Rust sources start with `// SPDX-License-Identifier: MIT OR Apache-2.0`. Keep it on new `.rs` files (`examples/myelin_timing.rs` currently lacks it).

## Gotchas

- Never run `cargo publish`.
- First build downloads the pinned toolchain and compiles the dependency tree; allow a few minutes.
- `.claude/` and `.cursor/` are gitignored, so keep shared agent guidance in this file or `AGENTS.md`.
- Edit only what the task needs.

# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

Spikenaut-SNN-v2 is a 16-neuron Leaky-Integrate-and-Fire (LIF) spiking neural network intended to
learn a compact temporal representation of machine state from live hardware and node telemetry,
targeting a Xilinx Artix-7 FPGA (see `README.md`). It is the small-supervisor layer of the
Artificial Interoception / Neuromorphic Supervisor program. Per the README's "Status — read this
first", it is a **research artifact, not a validated supervisor**, and FPGA parity is unproven. Don't
write docs or code comments that claim otherwise.

## Layout

| Path | Contents |
|------|----------|
| `src/` | Rust crate: model, encoding/stim, decision contract, NIR graph, IPC, Q8.8 memory, kinetic/silicon/neuromod glue, `training` |
| `dataset/merged_v2/` | Shipped model bank: Q8.8 `.mem` parameter files, `snn_model.json`, `model_bank.json` |
| `config.json` | Machine-readable model header |
| `tools/*.py` | Python verification tools (stdlib only): Q8.8 verifier, model-bank attestation, parity and replay tools |
| `tools/anticipation/` | Anticipation campaign (Python + Julia project). Needs third-party packages (NumPy, Matplotlib, and PyTorch for acquisition); see `requirements-test.txt` |
| `tests/` | Rust integration tests (`*.rs`, run by `cargo test`) and Python tests (pytest/unittest, `test_*.py`) |
| `docs/anticipation/` | Anticipation docs |

## Toolchain

- Rust **1.98.1** with rustfmt and clippy (`rust-toolchain.toml`, `rust-version`). It's pinned to the
  highest MSRV among the crates.io dependencies, so don't loosen it.
- Feature `training` (off by default) pulls `plasticity-lab`.
- Python **3.11** in CI. The top-level `tools/*.py` scripts use the standard library only;
  `tools/anticipation/` does not, and its tests use the hash-pinned
  `tools/anticipation/requirements-test.txt`.
- Julia **1.12.7** for `tools/anticipation` (CI).
- The Hermes tests need `bubblewrap` and `libseccomp2`. CI also sets
  `kernel.apparmor_restrict_unprivileged_userns=0` (Ubuntu).
- No GPU needed for CI or the tests. Real acquisition in the anticipation campaign
  (`tools/anticipation/campaign.py`) needs a CUDA GPU and CUDA-enabled PyTorch
  (`docs/anticipation/README.md`). FPGA loading is out of scope for CI.

## Commands (from `.github/workflows/ci.yml`)

```bash
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo check --locked --no-default-features
cargo test --locked --all-features
RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --all-features

# Python tools. CI also runs each one except replay_frozen.py with --self-test, and runs
# most of them in the `python3 -m tools.<name>` form too.
python3 tools/verify_q88.py
python3 tools/measure_hamming.py
python3 tools/live_stim_parity.py
python3 tools/decision_parity.py
python3 tools/check_model_card.py
python3 tools/verify_model_bank.py --select merged_v2
python3 tools/replay_frozen.py
python3 -m unittest tests.test_model_bank tests.test_replay_frozen -v

# Anticipation (Julia + pytest)
julia --startup-file=no --project=tools/anticipation -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=tools/anticipation tools/anticipation/test_train.jl
python -m pip install --require-hashes --only-binary=:all: -r tools/anticipation/requirements-test.txt
python -m pytest -q tests/test_anticipation_campaign.py tests/test_collector_lifecycle.py \
  tests/test_evaluation_deadline.py tests/test_hermes_*.py tests/test_path_confinement.py
```

## Conventions visible in the repo

- Q8.8 fixed-point encoding is a contract shared with the FPGA side. Use `tools/verify_q88.py` and
  the parity tools when changing encoding.
- GitHub Actions are pinned to commit SHAs with exact version comments, and Dependabot bumps
  them. Keep new actions pinned the same way.
- Commit subjects follow Conventional Commits, usually with a scope (`feat(nir):`, `feat(deps):`),
  and end with the PR number. Some also carry a Linear or GitHub issue ID (`RM-1713`, `GH#60`).

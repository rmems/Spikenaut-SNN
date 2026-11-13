# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

Spikenaut-SNN-v2 is a 16-neuron LIF network that learns a compact temporal representation of
machine state from live hardware and node telemetry, targeting a Xilinx Artix-7 FPGA (see
`README.md`). It is the small-supervisor layer of the Artificial Interoception / Neuromorphic
Supervisor program. Per the README's "Status — read this first", it is a **research artifact,
not a validated supervisor**, and FPGA parity is unproven. Don't claim otherwise in docs or comments.

## Layout

| Path | Contents |
|------|----------|
| `src/` | Rust crate: model, encoding/stim, decision contract, NIR graph, IPC, Q8.8 memory, kinetic/silicon/neuromod glue, `training` |
| `dataset/merged_v2/` | Shipped model bank: Q8.8 `.mem` parameter files, `snn_model.json`, `model_bank.json` |
| `dataset/merged_v2_rec_v1/` | exp-026 research variant (GH #3): the shipped hidden parameters plus a `spikenaut.recurrence.v1` I->E lateral-inhibition block and a refit readout. Not the shipped bank; not loaded by the crate; regenerate only with `tools/recurrence_eval.py --emit-checkpoint` on the recorded input |
| `config.json` | Machine-readable model header |
| `tools/*.py` | Python verification tools (stdlib only): Q8.8 verifier, Hamming measurement, model-bank attestation, model-card check, parity and replay tools |
| `tools/anticipation/` | Anticipation campaign (Python + Julia). `requirements-test.txt` pins CI tests (NumPy, pytest; not PyTorch or Matplotlib). Acquisition needs CUDA PyTorch; plots need Matplotlib |
| `tests/` | Rust integration tests (`*.rs`, run by `cargo test`) and Python tests (pytest/unittest, `test_*.py` plus `model_bank_review_cases.py`, see `pytest.ini`) |
| `examples/` | `myelin_timing.rs`; needs the `myelin-cuda` feature (GPU only, not built in CI) |
| `docs/` | `anticipation/` campaign docs, `recurrence/` (exp-026 comparison report and `comparison.json`), `MYELIN_COMPATIBILITY.md`, `superpowers/plans/` |

## Toolchain

- Rust **1.99.0** with rustfmt and clippy (`rust-toolchain.toml`, `rust-version`). It's pinned to the
  highest MSRV among the locked dependencies (`plasticity-lab` 0.3.0 declares 1.99.0), so don't loosen it.
- Features are off by default. `training` pulls `plasticity-lab` 0.3, which shares this crate's
  `neuromod` 0.7 engine — there is no second neuromod pin. `myelin` pulls the CPU-safe
  `myelin-accelerator` stub. Do not pass
  `--all-features`: that also enables `myelin-cuda`, which needs a CUDA 13.2+ toolkit. CI uses
  `--features myelin,training`.
- Python **3.11** in CI. Top-level `tools/*.py` are stdlib only. `requirements-test.txt` pins
  NumPy and pytest for anticipation tests; it does not install PyTorch or Matplotlib.
- Julia **1.12.7** for `tools/anticipation` (CI).
- Hermes tests need the `bwrap` preflight in the command block (`bubblewrap`, `libseccomp2`, AppArmor). The `sysctl` changes `kernel.apparmor_restrict_unprivileged_userns` for the whole host and is not restored automatically.
- No GPU for CI or the tests. Real acquisition (`tools/anticipation/campaign.py`) needs a CUDA GPU
  and CUDA-enabled PyTorch. FPGA loading is out of scope for CI.

## Commands (from `.github/workflows/ci.yml`)

```bash
cargo fmt --check
cargo clippy --locked --all-targets --features myelin,training -- -D warnings
cargo test --locked
cargo check --locked --no-default-features
cargo test --locked --features myelin,training
RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --features myelin,training

# Python: script and `python3 -m` where CI runs both. check_model_card.py is script-only; replay_frozen.py and recurrence_eval.py have no --self-test.
python3 tools/verify_q88.py
python3 -m tools.verify_q88
python3 tools/verify_q88.py --self-test
python3 -m tools.verify_q88 --self-test
python3 tools/measure_hamming.py
python3 -m tools.measure_hamming
python3 tools/measure_hamming.py --self-test
python3 -m tools.measure_hamming --self-test
python3 tools/live_stim_parity.py
python3 -m tools.live_stim_parity
python3 tools/live_stim_parity.py --self-test
python3 -m tools.live_stim_parity --self-test
python3 tools/decision_parity.py
python3 -m tools.decision_parity
python3 tools/decision_parity.py --self-test
python3 -m tools.decision_parity --self-test
python3 tools/check_model_card.py
python3 tools/check_model_card.py --self-test
python3 tools/verify_model_bank.py --select merged_v2
python3 -m tools.verify_model_bank --select merged_v2
python3 tools/verify_model_bank.py --self-test
python3 -m tools.verify_model_bank --self-test
python3 -m unittest tests.test_model_bank -v
python3 -m unittest tests.model_bank_review_cases -v
python3 tools/replay_frozen.py
python3 -m tools.replay_frozen
python3 -m unittest tests.test_replay_frozen -v
python3 tools/recurrence_eval.py
python3 -m tools.recurrence_eval
python3 tools/verify_model_bank.py dataset/merged_v2_rec_v1/model_bank.json --select merged_v2_rec_v1
python3 tools/replay_frozen.py --bank-manifest dataset/merged_v2_rec_v1/model_bank.json
python3 -m unittest tests.test_recurrence tests.test_recurrence_variant -v

julia --startup-file=no --project=tools/anticipation -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=tools/anticipation tools/anticipation/test_train.jl
sudo apt-get update
sudo apt-get install --yes bubblewrap libseccomp2
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
bwrap --version
bwrap --ro-bind / / --unshare-net -- /bin/true
python -m pip install --require-hashes --only-binary=:all: -r tools/anticipation/requirements-test.txt
command -v bwrap
python -m pytest -q tests/test_anticipation_campaign.py tests/test_collector_lifecycle.py \
  tests/test_evaluation_deadline.py tests/test_hermes_*.py tests/test_path_confinement.py \
  tests/test_pytest_config.py
```

## Conventions visible in the repo

- Q8.8 fixed-point encoding is a contract shared with the FPGA side. Use `tools/verify_q88.py` and
  the parity tools when changing encoding.
- GitHub Actions are pinned to commit SHAs with exact version comments, and Dependabot bumps
  them. Keep new actions pinned the same way.
- Commit subjects follow Conventional Commits. Scopes are common (`feat(nir):`) and so are
  unscoped subjects (`feat:`, `docs:`). Squash-merge adds `(#NN)`; don't invent a PR number.
  Some subjects also name an issue (`RM-1713`, `GH#60`) when the change tracks one.

# Myelin-accelerator 0.2.0 — Spikenaut consumer compatibility

Issue [#73](https://github.com/rmems/Spikenaut-SNN/issues/73). Consumer
validation of the **published** `myelin-accelerator =0.2.0` crate
(`pkg:cargo/myelin-accelerator@0.2.0`, commit
`6cfb49c60fb7f95818e87d0ea0e1ce76c2360fb5`, SHA-256
`afa83f62e0f7682560d3bc780f9b7382e8002bd945bf4b53324f2c648661c0ee`).
This is a downstream-evaluation record, not a release gate for Myelin.

```bibtex
@software{myelin_accelerator,
  title  = {Myelin-Accelerator},
  author = {Cardenas Montoya, Raul},
  year   = {2026},
  url    = {https://github.com/Limen-Neural/myelin-accelerator}
}
```

## Boundary

Myelin is a Rust/CUDA compute layer: Poisson, STDP and ternary GEMV/GEMM
launches plus host packing/reference utilities. It does not parse
Spikenaut JSON/`.mem`, execute a NIR graph, or supply an SNN framework.
Spikenaut's loaders, exact-byte attestation, Q8.8 interpretation and the
analog `LiveStimAdapter` front end are unchanged and unchallenged here.

## Compatibility gaps (exact shipped-bank CUDA execution unsupported)

| Spikenaut side | Myelin v0.2 contract | Verdict |
|---|---|---|
| Per-neuron decay (live bank: uniform `0x00DA` ≈ 0.8515625, schema allows any `(0, 1)`) and per-neuron thresholds | Fixed LIF dynamics: decay 0.85, threshold 1.0, reset 0.0, 2 refractory ticks (`oracle::LIF_*`) | No exact parity; a 16×16 workload validates the kernel, not the shipped model |
| No refractory counter in Spikenaut stepping semantics | 2-tick refractory state advanced on every step | State layout differs; per-timestep comparison must carry `refract` explicitly |
| Analog current front end (`v = decay·v + W @ stim`) | Poisson spike encoding (`U < rate`) | Different input modality; Poisson must not replace the shipped analog path |
| Dense Q8.8 `16×16` hidden matrix on a NIR `Linear` node | Group-scaled packed-ternary GEMV/GEMM (`pack_ternary_matrix`, `uniform_group_scales`) | Requires requantization into the ternary layout; not a drop-in matmul |
| `lif_step` / `lif_step_weighted` as callable dynamics | Loaded `KernelModule` symbols only, no `GpuAccelerator` wrapper; `get_function` returns `cust::function::Function` | Raw launches need a direct `cust` dependency (Myelin's own GPU tests import `cust`). Not declared; deferred to an explicit follow-up |

Exact external-model interoperability is
[myelin-accelerator#43](https://github.com/Limen-Neural/myelin-accelerator/issues/43) /
LIM-1461 (v0.3.0). Spikenaut semantics were not changed to manufacture
parity.

## Numerical contracts used by the harness

- Poisson encode vs `poisson_encode_oracle`: exact `u32` (NaN rate follows
  the oracle's `fminf`/`fmaxf` rule, documented in the oracle source).
- Ternary GEMV/GEMM vs oracle: absolute `1e-4`, relative `1e-5`
  (`TERNARY_ABS_TOL` / `TERNARY_REL_TOL`); device uses FMA order and
  fast-math FTZ, the scalar oracle does not emulate either.
- LIF step vs oracle: exact membrane bits and `u32` state, valid only for
  subnormal-free traces (device `fma.rn.ftz.f32` flushes subnormals;
  `f32::mul_add` keeps them). The harness asserts no subnormals.
- CPU-backend launches are not implementations: every launch method on a
  CPU `GpuAccelerator` returns `GpuError::Unavailable`, and
  `require_gpu()` fails closed.

## Environment evidence

- GPU: NVIDIA RTX 5080 (PCI `GB203`), `sm_120` target; `nvcc` 13.3.73
  (satisfies the published CUDA 13.2+ requirement); Rust 1.98.1; Fedora 44.
- GPU validation run 2026-10-02: driver 615.71.09 (the earlier NVML
  kernel/userspace mismatch is gone after the maintainer's host-side
  fix), CUDA toolkit 13.3, `myelin-cuda` feature, package checksum as
  above, Spikenaut revision `0fb6c7c6` (workspace dirty: uncommitted
  #73 consumer work).

## GPU results (RTX 5080, `require_gpu`, kernels proven executed)

- `tests/myelin_gpu.rs`: **5 passed, 0 failed** — 16-wide Poisson exact
  vs oracle (incl. NaN rule, empty no-op, length rejection, repeatability),
  STDP 2×2 vs published expectations with exact replay, seeded 16×16
  ternary GEMV vs oracle within abs 1e-4/rel 1e-5, 6 lifecycle cycles.
- Compute Sanitizer memcheck over the GPU consumer: **0 errors**.
- Timing (release scratch harness in `/tmp`, medians of 10–20 reps after
  warm-up; CPU baseline is the scalar oracle):

| workload | transfer-inclusive | kernel-only | CPU oracle |
|---|---|---|---|
| poisson 16-wide | 31.6 µs | 4.8 µs | <0.1 µs (below resolution) |
| poisson 1M | 704.3 µs | 7.6 µs | 272.2 µs |
| ternary GEMV 16×16 | 40.0 µs | 6.7 µs | 0.3 µs |
| ternary GEMV 1024×1024 | 231.4 µs | 141.5 µs | 4547.1 µs |

 Honest crossover: at 16-wide the CPU wins outright (GPU is pure
 overhead); at 1M Poisson the kernel is ~36× faster than the CPU but
 transfer-inclusive is ~2.6× slower, so the GPU pays off only with
 resident data; at 1024×1024 GEMV the GPU wins ~20× even inclusive.
 No useful 16-neuron crossover — as the issue anticipated, a slowdown
 there is a valid result.

## Status

- CPU/default path: done and green (packing, oracle, fallback fail-closed,
  registry pin, contract tests, Python replay/bank gates).
- GPU path: 5/5 device tests pass, sanitizer clean, timing recorded.
- Rescope (maintainer decision 2026-10-02, option 1): no `cust`
  dependency. The per-timestep device-vs-oracle weighted-LIF diff rides
  with [myelin-accelerator#43](https://github.com/Limen-Neural/myelin-accelerator/issues/43) /
  LIM-1461; the CPU-side 16×16 oracle trace in `tests/myelin_cpu.rs`
  pins what that future diff must reproduce.
- Recommendation: Myelin stays an **optional experimental consumer**
  (`myelin` / `myelin-cuda`, both off by default). It accelerates
  Poisson/STDP/ternary primitives with a proven GPU crossover only at
  batch scale; it cannot execute the shipped bank and must never replace
  the analog front end.

## Commands (reproduce)

```sh
# Default build: CUDA-free, no toolkit/GPU needed
cargo test --locked
# CPU-only Myelin consumer
cargo test --locked --features myelin
# Lints and format
cargo clippy --locked --all-targets
cargo clippy --locked --all-targets --features myelin
cargo fmt --check
# Python gates (shipped-bank preservation)
python3 -m pytest tests/test_replay_frozen.py tests/test_model_bank.py -q
# Real CUDA (needs driver + sm_120 GPU; device tests are #[ignore]d)
CUDA_NVCC=/usr/local/cuda/bin/nvcc cargo test --locked --features myelin-cuda --test myelin_gpu -- --ignored --nocapture
# Sanitizer (needs driver; binary path varies by build hash)
compute-sanitizer --tool memcheck ./target/debug/deps/myelin_gpu-<hash> --ignored
```

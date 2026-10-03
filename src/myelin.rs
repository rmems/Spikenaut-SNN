// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional CPU-safe consumer surface for `myelin-accelerator` 0.2.0.
//!
//! This module is gated on the `myelin` Cargo feature. It enables the
//! published registry dependency without its `cuda` feature, so the default
//! build stays CUDA-free and this path needs no toolkit or GPU.
//!
//! ```text
//! myelin-accelerator 0.2.0  ← compute layer (Poisson, STDP, ternary GEMV/GEMM)
//!              ✕
//! exp-025 signed weights    ← never loaded here, never rewritten
//! analog front end          ← stays `stim::LiveStimAdapter`, not Poisson
//! ```
//!
//! CUDA-specific launches live behind `myelin-cuda` (`myelin` plus
//! `myelin-accelerator/cuda`). Weighted LIF has no high-level wrapper in
//! v0.2.0: its raw `lif_step` / `lif_step_weighted` symbols are reachable
//! only through `KernelModule::get_function`, which returns a `cust`
//! launch handle. This crate declares no `cust` dependency, so raw LIF
//! launches are explicitly out of scope until that dependency is reviewed
//! and added. See the Phase 0–1 finding in the module docs below.
//!
//! Compatibility gaps, numerical contracts and environment evidence live in
//! `docs/MYELIN_COMPATIBILITY.md`.

pub use myelin_accelerator::{
    Backend, CapabilityReport, ExecutionPolicy, FallbackReason, GpuAccelerator, GpuBuffer,
    GpuContext, GpuError, KernelModule, probe_capabilities,
};

/// Pinned consumer version for this qualification (issue #73).
pub const MYELIN_PIN: &str = "=0.2.0";

/// Published Myelin v0.2 fixed LIF decay (`oracle::LIF_DECAY`).
///
/// This is the kernel's constant, not a Spikenaut per-neuron parameter.
/// Spikenaut decays are per-neuron Q8.8 values; exact shipped-bank parity
/// through this kernel remains unsupported.
pub const MYELIN_LIF_DECAY: f32 = 0.85;

/// Published Myelin v0.2 fixed LIF threshold (`oracle::LIF_THRESHOLD`).
pub const MYELIN_LIF_THRESHOLD: f32 = 1.0;

/// Published Myelin v0.2 fixed LIF reset (`oracle::LIF_RESET`).
pub const MYELIN_LIF_RESET: f32 = 0.0;

/// Published Myelin v0.2 fixed refractory ticks (`oracle::LIF_REFRACT_TICKS`).
pub const MYELIN_LIF_REFRACT_TICKS: u32 = 2;

/// Whether this build enabled the CUDA launch path.
///
/// True only under `myelin-cuda`. The `myelin` feature alone is the CPU-safe
/// stub: capability probing and host utilities work, launches fail closed.
#[must_use]
pub const fn is_cuda_build() -> bool {
    cfg!(feature = "myelin-cuda")
}

/// One-line backend summary for evidence logs: backend, readiness, fallback.
#[must_use]
pub fn backend_summary(accelerator: &GpuAccelerator) -> String {
    let backend = match accelerator.selected_backend() {
        Backend::Cuda => "cuda",
        Backend::Cpu => "cpu",
    };
    let ready = accelerator.is_ready();
    match accelerator.fallback() {
        Some(record) => format!(
            "backend={backend} ready={ready} fallback={} detail={}",
            record.reason.code(),
            record.detail
        ),
        None => format!("backend={backend} ready={ready} fallback=none"),
    }
}

/// Construct under the caller-approved CPU-fallback policy.
///
/// Infallible by contract: may select CPU and always records the reason via
/// [`GpuAccelerator::fallback`].
#[must_use]
pub fn prefer_gpu() -> GpuAccelerator {
    GpuAccelerator::new()
}

/// Fail-closed GPU construction: never returns a CPU-backend accelerator.
///
/// Returns `GpuError::Unavailable` when CUDA is not built or no device is
/// usable. GPU-gated tests must use this, never [`prefer_gpu`].
pub fn require_gpu() -> Result<GpuAccelerator, GpuError> {
    GpuAccelerator::require_gpu()
}

/// CUDA-only launch surface, available under `myelin-cuda`.
///
/// The high-level Poisson / STDP / ternary wrappers need no direct `cust`
/// use: they take host slices through [`GpuAccelerator`] methods. Raw
/// `lif_step` / `lif_step_weighted` launches are intentionally absent here:
/// they require `cust::stream::Stream` and the `launch!` macro against the
/// `cust::function::Function` returned by `KernelModule::get_function`, and
/// this crate does not depend on `cust`.
#[cfg(feature = "myelin-cuda")]
pub mod cuda {
    use super::{GpuAccelerator, GpuError};

    /// Fail-closed constructor for GPU-gated tests.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] when no usable CUDA device is present. A CPU
    /// fallback success must never count as a CUDA pass.
    pub fn require_gpu_accelerator() -> Result<GpuAccelerator, GpuError> {
        GpuAccelerator::require_gpu()
    }
}

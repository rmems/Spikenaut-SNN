// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-safe consumer contract for `myelin-accelerator` 0.2.0 (issue #73).
//!
//! Gated on the `myelin` feature. Needs no CUDA toolkit or GPU: host
//! packing/reference utilities plus capability/fallback reporting through
//! public APIs. GPU launch methods must fail closed, never silently compute.

#![cfg(feature = "myelin")]

use myelin_accelerator::bitpacking::{pack_binary, pack_ternary, unpack_binary, unpack_ternary};
use myelin_accelerator::oracle::{
    CaseRng, LIF_DECAY, LIF_REFRACT_TICKS, LIF_RESET, LIF_THRESHOLD, lif_step_weighted_oracle,
    poisson_encode_oracle,
};
use spikenaut_snn::myelin::{
    Backend, GpuAccelerator, GpuBuffer, GpuError, MYELIN_LIF_DECAY, MYELIN_LIF_REFRACT_TICKS,
    MYELIN_LIF_RESET, MYELIN_LIF_THRESHOLD, backend_summary, is_cuda_build, probe_capabilities,
};

/// Full registry checksum from `GET /api/v1/crates/myelin-accelerator/0.2.0`,
/// verified against `sha256sum` of the downloaded `.crate` file.
const MYELIN_CRATE_SHA256: &str =
    "afa83f62e0f7682560d3bc780f9b7382e8002bd945bf4b53324f2c648661c0ee";

fn is_unavailable(error: &GpuError) -> bool {
    matches!(error, GpuError::Unavailable { .. } | GpuError::NoGpu)
}

#[test]
fn ternary_packing_round_trips_including_empty_and_boundaries() {
    assert_eq!(
        unpack_ternary(&pack_ternary(&[]), Some(0)),
        Vec::<i8>::new()
    );
    for values in [
        vec![-1, 0, 1, 1],
        vec![1; 16],
        vec![-1; 17],
        vec![0, 1, -1, 0, 1, -1],
    ] {
        assert_eq!(
            unpack_ternary(&pack_ternary(&values), Some(values.len())),
            values
        );
    }
}

#[test]
fn binary_packing_round_trips_including_empty() {
    assert_eq!(
        unpack_binary(&pack_binary(&[]), Some(0)),
        Vec::<bool>::new()
    );
    let values = vec![true, false, true, true, false];
    assert_eq!(
        unpack_binary(&pack_binary(&values), Some(values.len())),
        values
    );
}

#[test]
fn poisson_oracle_matches_independent_expectations_and_repeats() {
    // Independent scalar contract: rate 0.0 never spikes, rate 1.0 always
    // spikes (`r in [0,1)` vs clamped threshold). Empty stays empty.
    assert_eq!(poisson_encode_oracle(&[], 42), Vec::<u32>::new());
    assert_eq!(poisson_encode_oracle(&[0.0, 1.0], 7), vec![0, 1]);

    let stimuli = [0.25, 0.5, 0.75, 0.0, 1.0];
    let first = poisson_encode_oracle(&stimuli, 1234);
    let replay = poisson_encode_oracle(&stimuli, 1234);
    assert_eq!(first, replay);
    assert_ne!(
        poisson_encode_oracle(&stimuli, 1234),
        poisson_encode_oracle(&stimuli, 5678)
    );
}

#[test]
fn adapter_lif_constants_match_the_published_oracle() {
    assert_eq!(MYELIN_LIF_DECAY, LIF_DECAY);
    assert_eq!(MYELIN_LIF_THRESHOLD, LIF_THRESHOLD);
    assert_eq!(MYELIN_LIF_RESET, LIF_RESET);
    assert_eq!(MYELIN_LIF_REFRACT_TICKS, LIF_REFRACT_TICKS);
    assert_eq!(
        (MYELIN_LIF_DECAY, MYELIN_LIF_THRESHOLD, MYELIN_LIF_RESET),
        (0.85, 1.0, 0.0)
    );
    assert_eq!(MYELIN_LIF_REFRACT_TICKS, 2);
}

#[test]
fn cpu_accelerator_reports_its_fallback_and_fails_closed() {
    let accelerator = GpuAccelerator::new();
    let fallback = accelerator
        .fallback()
        .expect("a CPU-backend accelerator always records its fallback");
    assert_eq!(accelerator.selected_backend(), Backend::Cpu);
    assert!(!accelerator.is_ready());
    assert!(!fallback.detail.contains("/home/"));
    assert!(!backend_summary(&accelerator).is_empty());

    let error = match GpuAccelerator::require_gpu() {
        Err(error) => error,
        Ok(_) => panic!("no GPU in the CPU build"),
    };
    assert!(is_unavailable(&error));
    // Exact reason only without the cuda feature; with `myelin-cuda` on a
    // GPU-less host the reason names the runtime failure instead.
    #[cfg(not(feature = "myelin-cuda"))]
    assert_eq!(
        error.fallback_reason(),
        Some(spikenaut_snn::myelin::FallbackReason::CudaFeatureNotBuilt)
    );
}

#[test]
fn cpu_launches_are_not_cpu_implementations() {
    let accelerator = GpuAccelerator::new();
    let stimuli = GpuBuffer::<f32>::from_slice(&[0.5, 0.5]).expect("host buffer");
    let mut spikes = GpuBuffer::<u32>::alloc(2).expect("host buffer");
    assert!(is_unavailable(
        &accelerator
            .poisson_encode(&stimuli, &mut spikes, 1)
            .unwrap_err()
    ));

    let mut weights = GpuBuffer::<f32>::from_slice(&[0.1, 0.2]).expect("host buffer");
    let pre = GpuBuffer::<f32>::from_slice(&[0.0, 0.0]).expect("host buffer");
    let post = GpuBuffer::<f32>::from_slice(&[0.0, 0.0]).expect("host buffer");
    let mut pre_traces = GpuBuffer::<f32>::from_slice(&[0.0, 0.0]).expect("host buffer");
    let mut post_traces = GpuBuffer::<f32>::from_slice(&[0.0, 0.0]).expect("host buffer");
    assert!(is_unavailable(
        &accelerator
            .stdp_update(
                &mut weights,
                &pre,
                &post,
                &mut pre_traces,
                &mut post_traces,
                1,
                2,
                0.01
            )
            .unwrap_err()
    ));
    assert!(is_unavailable(&accelerator.synchronize().unwrap_err()));
    assert!(match accelerator.kernels() {
        Err(error) => is_unavailable(&error),
        Ok(_) => panic!("no kernels without a GPU"),
    });
}

#[test]
fn capability_probe_is_cpu_without_the_cuda_feature() {
    let report = probe_capabilities();
    #[cfg(not(feature = "myelin-cuda"))]
    {
        assert!(!report.cuda_built);
        assert_eq!(report.selected_backend, Backend::Cpu);
        assert!(!is_cuda_build());
    }
    #[cfg(feature = "myelin-cuda")]
    {
        assert!(report.cuda_built);
        assert!(is_cuda_build());
    }
}

#[test]
fn myelin_resolves_from_crates_io_at_pinned_zero_two_zero() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"))
        .expect("read Cargo.lock");
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains("name = \"myelin-accelerator\""))
        .expect("Cargo.lock has myelin-accelerator");

    assert!(
        entry.contains("source = \"registry+https://github.com/rust-lang/crates.io-index\""),
        "myelin-accelerator must resolve from crates.io, got:\n{entry}",
    );
    assert!(
        entry.contains("version = \"0.2.0\""),
        "myelin-accelerator must resolve to the pinned 0.2.0 release, got:\n{entry}",
    );
    assert!(
        entry.contains(MYELIN_CRATE_SHA256),
        "Cargo.lock checksum must match the full published checksum, got:\n{entry}",
    );
}

/// 16×16 weighted-LIF oracle trace under the documented v0.2 dynamics.
///
/// No GPU is involved: this pins the membrane/refractory/spike trace the
/// future device diff (raw `lif_step_weighted` symbol, needs `cust`) must
/// reproduce bit-for-bit at every timestep. Independent expectations come
/// straight from the documented rule — decay 0.85, threshold 1.0, reset
/// 0.0, 2 refractory ticks — not from re-running the oracle.
#[test]
fn weighted_lif_oracle_16x16_trace_matches_documented_dynamics() {
    const N: usize = 16;

    // Silence: zero current can never reach threshold; state never moves.
    let weights = vec![0.5f32; N * N];
    let mut membrane = vec![0.0f32; N];
    let mut refract = vec![0u32; N];
    for tick in 0..4 {
        let spikes = lif_step_weighted_oracle(&mut membrane, &weights, &[0.0; N], &mut refract, N);
        assert_eq!(
            spikes,
            vec![0u32; N],
            "silence must not spike at tick {tick}"
        );
        assert_eq!(
            membrane,
            vec![0.0f32; N],
            "silence must not charge at tick {tick}"
        );
        assert_eq!(
            refract,
            vec![0u32; N],
            "silence must not refract at tick {tick}"
        );
    }

    // Saturation: unit weights and inputs give current 16.0 per neuron, so
    // every neuron spikes, resets to 0.0, and sits out 2 refractory ticks:
    // spikes at ticks 0, 3, 6 — period 3 — with membranes pinned at zero.
    let weights = vec![1.0f32; N * N];
    let mut membrane = vec![0.0f32; N];
    let mut refract = vec![0u32; N];
    for tick in 0..7 {
        let spikes = lif_step_weighted_oracle(&mut membrane, &weights, &[1.0; N], &mut refract, N);
        let expected = u32::from(tick % 3 == 0);
        assert_eq!(
            spikes,
            vec![expected; N],
            "saturated 16-wide must spike exactly at ticks 0, 3, 6 (tick {tick})"
        );
        assert_eq!(
            membrane,
            vec![0.0f32; N],
            "reset pins membranes at tick {tick}"
        );
    }

    // Seeded workload determinism: the same fixture replays bit-for-bit, so
    // a future device mismatch can be blamed on the device, not the trace.
    let mut rng = CaseRng::new(516_016);
    let weights: Vec<f32> = (0..N * N).map(|_| rng.next_unit_f32()).collect();
    let inputs: Vec<f32> = (0..N).map(|_| rng.next_unit_f32()).collect();
    let run = || {
        let mut membrane = vec![0.0f32; N];
        let mut refract = vec![0u32; N];
        let mut trace = Vec::with_capacity(8 * N);
        for _ in 0..8 {
            let spikes =
                lif_step_weighted_oracle(&mut membrane, &weights, &inputs, &mut refract, N);
            trace.extend_from_slice(&spikes);
            trace.extend(membrane.iter().map(|v| v.to_bits()));
        }
        (trace, refract)
    };
    assert_eq!(run(), run(), "seeded 16×16 trace must replay exactly");
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Real-CUDA consumer tests for `myelin-accelerator` 0.2.0 (issue #73).
//!
//! Gated on `myelin-cuda` and `#[ignore]`d: every test requires a working
//! CUDA device via fail-closed [`require_gpu_accelerator`]. A CPU fallback,
//! PTX compilation alone, or a skip never counts as a pass.
//!
//! Status: executed 2026-10-02 on RTX 5080 (driver 615.71.09, CUDA 13.3) —
//! 5 passed, 0 failed; Compute Sanitizer memcheck 0 errors. The `#[ignore]`
//! markers remain because every test needs a real device; run with
//! `-- --ignored` on a provisioned sm_120 runner.

#![cfg(feature = "myelin-cuda")]

use myelin_accelerator::bitpacking::{pack_ternary_matrix, uniform_group_scales};
use myelin_accelerator::oracle::{
    CaseRng, TERNARY_ABS_TOL, TERNARY_REL_TOL, assert_exact, assert_f32, poisson_encode_oracle,
    ternary_gemm_oracle, ternary_gemv_oracle,
};
use myelin_accelerator::{GpuAccelerator, GpuBuffer};
use spikenaut_snn::myelin::cuda::require_gpu_accelerator;

fn gpu() -> GpuAccelerator {
    require_gpu_accelerator().expect("GPU required: no CPU fallback counts as a CUDA pass")
}

fn read_spikes(accelerator: &GpuAccelerator, stimuli: &[f32], seed: u32) -> Vec<u32> {
    let d_stimuli = GpuBuffer::from_slice(stimuli).expect("upload stimuli");
    let mut d_spikes = GpuBuffer::<u32>::alloc(stimuli.len()).expect("alloc spikes");
    accelerator
        .poisson_encode(&d_stimuli, &mut d_spikes, seed)
        .expect("poisson_encode");
    d_spikes.to_vec().expect("read spikes")
}

#[test]
#[ignore = "requires CUDA device (sm_120); run with --ignored"]
fn gpu_poisson_matches_oracle_exact_on_16_wide_fixture() {
    let accelerator = gpu();
    // 16-wide Spikenaut-shaped fixture: silence, saturation, mid-rates and
    // the documented NaN rule (NaN rate encodes as 1.0 per the oracle's
    // fminf/fmaxf semantics, matching myelin's own NaN differential test).
    let stimuli = [
        0.0,
        1.0,
        0.25,
        0.5,
        0.75,
        0.1,
        0.9,
        0.33,
        0.66,
        0.125,
        0.875,
        0.4,
        0.6,
        0.05,
        0.95,
        f32::NAN,
    ];
    let seed = 0x5eed_1234;
    let got = read_spikes(&accelerator, &stimuli, seed);
    let expected = poisson_encode_oracle(&stimuli, seed);
    assert_exact(&got, &expected, u64::from(seed), "poisson 16-wide");
    // Repeatability on device.
    let replay = read_spikes(&accelerator, &stimuli, seed);
    assert_exact(
        &replay,
        &expected,
        u64::from(seed),
        "poisson 16-wide replay",
    );
}

#[test]
#[ignore = "requires CUDA device (sm_120); run with --ignored"]
fn gpu_poisson_empty_is_a_no_op_and_mismatched_lengths_fail() {
    let accelerator = gpu();
    let d_stimuli = GpuBuffer::<f32>::from_slice(&[]).expect("upload empty");
    let mut d_spikes = GpuBuffer::<u32>::alloc(0).expect("alloc empty");
    accelerator
        .poisson_encode(&d_stimuli, &mut d_spikes, 7)
        .expect("empty poisson_encode is a defined no-op");
    assert!(d_spikes.to_vec().expect("read empty").is_empty());

    let d_stimuli = GpuBuffer::from_slice(&[0.5, 0.5]).expect("upload stimuli");
    let mut d_short = GpuBuffer::<u32>::alloc(1).expect("alloc short");
    assert!(
        accelerator
            .poisson_encode(&d_stimuli, &mut d_short, 7)
            .is_err(),
        "mismatched output length must be rejected"
    );
}

#[test]
#[ignore = "requires CUDA device (sm_120); run with --ignored"]
fn gpu_stdp_advances_weights_and_traces_per_published_expectations() {
    // Full-element expectations derived from the published kernel source
    // (`cu/spiking_network.cu` @ 0.2.0: `A_PLUS = 0.01`, `A_MINUS = 0.012`,
    // `TAU = 20.0`, clamp `[0.0, 2.0]`):
    //   trace[i] = decay * trace[i] + spike[i]
    //   w += A_PLUS * post * (decay * pre_trace + pre)
    //   w -= A_MINUS * pre * (decay * post_trace + post)
    // with `decay = exp(-dt_ms / TAU)`. Fixture: 2x2 unit weights and
    // traces, pre = post = [1, 0], dt = 20 ms, so decay = exp(-1).
    // Tolerance 1e-5 absorbs device `__expf` vs host `exp`.
    let accelerator = gpu();
    let run_once = |accelerator: &GpuAccelerator| {
        let mut weights = GpuBuffer::from_slice(&[1.0f32; 4]).expect("weights");
        let pre = GpuBuffer::from_slice(&[1.0f32, 0.0]).expect("pre spikes");
        let post = GpuBuffer::from_slice(&[1.0f32, 0.0]).expect("post spikes");
        let mut pre_traces = GpuBuffer::from_slice(&[1.0f32; 2]).expect("pre traces");
        let mut post_traces = GpuBuffer::from_slice(&[1.0f32; 2]).expect("post traces");
        accelerator
            .stdp_update(
                &mut weights,
                &pre,
                &post,
                &mut pre_traces,
                &mut post_traces,
                2,
                2,
                20.0,
            )
            .expect("stdp_update");
        (
            weights.to_vec().expect("read weights"),
            pre_traces.to_vec().expect("read pre traces"),
            post_traces.to_vec().expect("read post traces"),
        )
    };
    let (weights, pre_traces, post_traces) = run_once(&accelerator);
    let decay = (-1.0f32).exp();
    let expected_traces = [1.0 + decay, decay];
    for (i, (&got, &expected)) in pre_traces.iter().zip(&expected_traces).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "pre_traces[{i}]: got {got}, expected {expected}"
        );
    }
    for (i, (&got, &expected)) in post_traces.iter().zip(&expected_traces).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "post_traces[{i}]: got {got}, expected {expected}"
        );
    }
    // Row-major [post][pre]: w[0] sees LTP+LTD, w[1] LTP only, w[2] LTD
    // only, w[3] neither.
    let expected_weights = [
        1.0 + 0.01 * (1.0 + decay) - 0.012 * (1.0 + decay),
        1.0 + 0.01 * decay,
        1.0 - 0.012 * decay,
        1.0,
    ];
    for (i, (&got, &expected)) in weights.iter().zip(&expected_weights).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "weights[{i}]: got {got}, expected {expected}"
        );
    }
    // Separate identical fixture state repeats device output exactly.
    let (weights2, pre2, post2) = run_once(&accelerator);
    assert_eq!((weights, pre_traces, post_traces), (weights2, pre2, post2));
}

#[test]
#[ignore = "requires CUDA device (sm_120); run with --ignored"]
fn gpu_ternary_gemv_matches_oracle_on_seeded_16x16() {
    let accelerator = gpu();
    let (m, k, group_size) = (16usize, 16usize, 8usize);
    let seed = 2026_1002u64;
    let mut rng = CaseRng::new(seed);
    let trits: Vec<i8> = (0..m * k).map(|_| rng.next_trit()).collect();
    let packed = pack_ternary_matrix(&trits, m, k);
    let scales = uniform_group_scales(m, k, group_size, 0.5);
    let x: Vec<f32> = (0..k).map(|_| rng.next_unit_f32()).collect();
    let expected = ternary_gemv_oracle(&packed, &scales, &x, m, k, group_size, false);

    let d_packed = GpuBuffer::from_slice(&packed).expect("upload packed");
    let d_scales = GpuBuffer::from_slice(&scales).expect("upload scales");
    let d_x = GpuBuffer::from_slice(&x).expect("upload x");
    let mut d_y = GpuBuffer::<f32>::alloc(m).expect("alloc y");
    accelerator
        .ternary_gemv(
            &d_packed,
            &d_scales,
            &d_x,
            &mut d_y,
            m as i32,
            k as i32,
            group_size as i32,
            false,
        )
        .expect("ternary_gemv");
    let got = d_y.to_vec().expect("read y");
    assert_f32(
        &got,
        &expected,
        TERNARY_ABS_TOL,
        TERNARY_REL_TOL,
        seed,
        "ternary_gemv 16x16",
    );

    let mut d_y = GpuBuffer::<f32>::alloc(m).expect("alloc y");
    assert!(
        accelerator
            .ternary_gemv(
                &d_packed,
                &d_scales,
                &d_x,
                &mut d_y,
                -1,
                k as i32,
                group_size as i32,
                false
            )
            .is_err(),
        "negative m must be rejected"
    );
}

#[test]
#[ignore = "requires CUDA device (sm_120); run with --ignored"]
fn gpu_ternary_gemm_matches_oracle_on_seeded_fixture() {
    let accelerator = gpu();
    let (m, k, n, group_size) = (8usize, 16usize, 4usize, 8usize);
    let seed = 2026_1003u64;
    let mut rng = CaseRng::new(seed);
    let trits: Vec<i8> = (0..m * k).map(|_| rng.next_trit()).collect();
    let packed = pack_ternary_matrix(&trits, m, k);
    let scales = uniform_group_scales(m, k, group_size, 0.5);
    let b: Vec<f32> = (0..k * n).map(|_| rng.next_unit_f32()).collect();
    let expected = ternary_gemm_oracle(&packed, &scales, &b, m, k, n, group_size, false);

    let d_packed = GpuBuffer::from_slice(&packed).expect("upload packed");
    let d_scales = GpuBuffer::from_slice(&scales).expect("upload scales");
    let d_b = GpuBuffer::from_slice(&b).expect("upload b");
    let mut d_c = GpuBuffer::<f32>::alloc(m * n).expect("alloc c");
    accelerator
        .ternary_gemm(
            &d_packed,
            &d_scales,
            &d_b,
            &mut d_c,
            m as i32,
            k as i32,
            n as i32,
            group_size as i32,
            false,
        )
        .expect("ternary_gemm");
    let got = d_c.to_vec().expect("read c");
    assert_f32(
        &got,
        &expected,
        TERNARY_ABS_TOL,
        TERNARY_REL_TOL,
        seed,
        "ternary_gemm 8x16x4",
    );
}

#[test]
#[ignore = "requires CUDA device (sm_120); run with --ignored"]
fn gpu_lifecycle_survives_repeated_create_launch_drop_cycles() {
    // Teardown/cycle coverage for the sanitizer gate: each iteration builds
    // a fresh accelerator, allocates, launches, synchronizes, reads back
    // and drops everything. Sanitizer memcheck 2026-10-02: 0 errors.
    for cycle in 0..6 {
        let accelerator = gpu();
        let stimuli = [0.1 * (cycle as f32 + 1.0); 16];
        let got = read_spikes(&accelerator, &stimuli, cycle);
        let expected = poisson_encode_oracle(&stimuli, cycle);
        assert_exact(&got, &expected, u64::from(cycle), "lifecycle cycle");
        accelerator.synchronize().expect("synchronize");
    }
}

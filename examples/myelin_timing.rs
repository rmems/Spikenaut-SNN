//! Timing harness behind the published numbers in
//! `docs/MYELIN_COMPATIBILITY.md` (issue #73).
//!
//! Requires a real CUDA device: run with
//! `CUDA_NVCC=<nvcc> cargo run --release --example myelin_timing` on a
//! provisioned sm_120 runner. Reports medians after warm-up for
//! transfer-inclusive vs kernel-only vs CPU-oracle latency on small and
//! batched Poisson and ternary-GEMV workloads. Fixture seeds are fixed
//! (`10_731`, `20_261_002`) so a rerun reproduces the same inputs.

use myelin_accelerator::bitpacking::{pack_ternary_matrix, uniform_group_scales};
use myelin_accelerator::oracle::{CaseRng, poisson_encode_oracle, ternary_gemv_oracle};
use myelin_accelerator::{GpuAccelerator, GpuBuffer};
use std::time::Instant;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// One timed closure, repeated `reps` times; returns microseconds per rep.
fn time_reps(reps: usize, mut work: impl FnMut()) -> Vec<f64> {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        work();
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    samples
}

fn report(kind: &str, label: &str, incl: Vec<f64>, kern: Vec<f64>, cpu: Vec<f64>) {
    println!(
        "{kind}/{label}: transfer-inclusive={:.1}us kernel-only={:.1}us cpu-oracle={:.1}us",
        median(incl),
        median(kern),
        median(cpu)
    );
}

fn time_poisson(accelerator: &GpuAccelerator, label: &str, n: usize) {
    let mut rng = CaseRng::new(10_731);
    let stimuli: Vec<f32> = (0..n).map(|_| rng.next_unit_f32()).collect();
    for _ in 0..5 {
        launch_poisson_once(accelerator, &stimuli);
    }
    let incl = time_reps(20, || {
        let _ = launch_poisson_once(accelerator, &stimuli);
    });
    let resident = GpuBuffer::from_slice(&stimuli).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(n).unwrap();
    let kern = time_reps(20, || {
        accelerator
            .poisson_encode_async(&resident, &mut spikes, 1)
            .unwrap();
        accelerator.synchronize().unwrap();
    });
    let cpu = time_reps(20, || {
        let _ = poisson_encode_oracle(&stimuli, 1);
    });
    report("poisson", label, incl, kern, cpu);
}

/// Upload, launch, synchronize and read back one Poisson Encoding.
fn launch_poisson_once(accelerator: &GpuAccelerator, stimuli: &[f32]) -> Vec<u32> {
    let resident = GpuBuffer::from_slice(stimuli).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(stimuli.len()).unwrap();
    accelerator
        .poisson_encode(&resident, &mut spikes, 1)
        .unwrap();
    spikes.to_vec().unwrap()
}

const GROUP: usize = 8;

fn time_gemv(accelerator: &GpuAccelerator, label: &str, m: usize, k: usize) {
    let mut rng = CaseRng::new(20_261_002);
    let trits: Vec<i8> = (0..m * k).map(|_| rng.next_trit()).collect();
    let packed = pack_ternary_matrix(&trits, m, k);
    let scales = uniform_group_scales(m, k, GROUP, 0.5);
    let x: Vec<f32> = (0..k).map(|_| rng.next_unit_f32()).collect();
    for _ in 0..3 {
        launch_gemv_once(accelerator, &packed, &scales, &x, m, k);
    }
    let reps = if m > 16 { 10 } else { 20 };
    let incl = time_reps(reps, || {
        launch_gemv_once(accelerator, &packed, &scales, &x, m, k);
    });
    let resident_packed = GpuBuffer::from_slice(&packed).unwrap();
    let resident_scales = GpuBuffer::from_slice(&scales).unwrap();
    let resident_x = GpuBuffer::from_slice(&x).unwrap();
    let mut y = GpuBuffer::<f32>::alloc(m).unwrap();
    let kern = time_reps(reps, || {
        launch_gemv_async(
            accelerator,
            &resident_packed,
            &resident_scales,
            &resident_x,
            &mut y,
            m,
            k,
        );
    });
    let cpu = time_reps(reps, || {
        let _ = ternary_gemv_oracle(&packed, &scales, &x, m, k, GROUP, false);
    });
    report("gemv", label, incl, kern, cpu);
}

/// Upload, launch, synchronize and read back one ternary GEMV.
fn launch_gemv_once(
    accelerator: &GpuAccelerator,
    packed: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let resident_packed = GpuBuffer::from_slice(packed).unwrap();
    let resident_scales = GpuBuffer::from_slice(scales).unwrap();
    let resident_x = GpuBuffer::from_slice(x).unwrap();
    let mut y = GpuBuffer::<f32>::alloc(m).unwrap();
    launch_gemv_async(
        accelerator,
        &resident_packed,
        &resident_scales,
        &resident_x,
        &mut y,
        m,
        k,
    );
    y.to_vec().unwrap()
}

/// Async GEMV launch plus the synchronize the caller must not skip.
fn launch_gemv_async(
    accelerator: &GpuAccelerator,
    packed: &GpuBuffer<u32>,
    scales: &GpuBuffer<f32>,
    x: &GpuBuffer<f32>,
    y: &mut GpuBuffer<f32>,
    m: usize,
    k: usize,
) {
    accelerator
        .ternary_gemv_async(
            packed,
            scales,
            x,
            y,
            m as i32,
            k as i32,
            GROUP as i32,
            false,
        )
        .unwrap();
    accelerator.synchronize().unwrap();
}

fn main() {
    let accelerator = GpuAccelerator::require_gpu().expect("GPU required");
    println!(
        "backend={:?} ready={}",
        accelerator.selected_backend(),
        accelerator.is_ready()
    );

    time_poisson(&accelerator, "small", 16);
    time_poisson(&accelerator, "batched", 1_048_576);
    time_gemv(&accelerator, "small", 16, 16);
    time_gemv(&accelerator, "batched", 1024, 1024);
}

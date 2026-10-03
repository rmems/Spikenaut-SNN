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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let acc = GpuAccelerator::require_gpu().expect("GPU required");
    println!(
        "backend={:?} ready={}",
        acc.selected_backend(),
        acc.is_ready()
    );

    // ---- Poisson: small (16) vs batched (1M) ----
    for (label, n) in [("small", 16usize), ("batched", 1_048_576usize)] {
        let mut rng = CaseRng::new(10_731);
        let stim: Vec<f32> = (0..n).map(|_| rng.next_unit_f32()).collect();
        // Warm-up (includes JIT/context already done at first use).
        for _ in 0..5 {
            let d = GpuBuffer::from_slice(&stim).unwrap();
            let mut s = GpuBuffer::<u32>::alloc(n).unwrap();
            acc.poisson_encode(&d, &mut s, 1).unwrap();
        }
        // Transfer-inclusive: upload + launch + sync + download.
        let mut incl = Vec::new();
        for _ in 0..20 {
            let t = Instant::now();
            let d = GpuBuffer::from_slice(&stim).unwrap();
            let mut s = GpuBuffer::<u32>::alloc(n).unwrap();
            acc.poisson_encode(&d, &mut s, 1).unwrap();
            let _ = s.to_vec().unwrap();
            incl.push(t.elapsed().as_secs_f64() * 1e6);
        }
        // Kernel-only: pre-uploaded, async launch + explicit synchronize.
        let d = GpuBuffer::from_slice(&stim).unwrap();
        let mut s = GpuBuffer::<u32>::alloc(n).unwrap();
        let mut kern = Vec::new();
        for _ in 0..20 {
            let t = Instant::now();
            acc.poisson_encode_async(&d, &mut s, 1).unwrap();
            acc.synchronize().unwrap();
            kern.push(t.elapsed().as_secs_f64() * 1e6);
        }
        // CPU baseline: scalar oracle.
        let mut cpu = Vec::new();
        for _ in 0..20 {
            let t = Instant::now();
            let _ = poisson_encode_oracle(&stim, 1);
            cpu.push(t.elapsed().as_secs_f64() * 1e6);
        }
        println!(
            "poisson/{label}: transfer-inclusive={:.1}us kernel-only={:.1}us cpu-oracle={:.1}us",
            median(incl),
            median(kern),
            median(cpu)
        );
    }

    // ---- Ternary GEMV: 16x16 vs 1024x1024 ----
    for (label, m, k) in [
        ("small", 16usize, 16usize),
        ("batched", 1024usize, 1024usize),
    ] {
        let group = 8usize;
        let mut rng = CaseRng::new(20_261_002);
        let trits: Vec<i8> = (0..m * k).map(|_| rng.next_trit()).collect();
        let packed = pack_ternary_matrix(&trits, m, k);
        let scales = uniform_group_scales(m, k, group, 0.5);
        let x: Vec<f32> = (0..k).map(|_| rng.next_unit_f32()).collect();
        for _ in 0..3 {
            let dp = GpuBuffer::from_slice(&packed).unwrap();
            let ds = GpuBuffer::from_slice(&scales).unwrap();
            let dx = GpuBuffer::from_slice(&x).unwrap();
            let mut dy = GpuBuffer::<f32>::alloc(m).unwrap();
            acc.ternary_gemv(
                &dp,
                &ds,
                &dx,
                &mut dy,
                m as i32,
                k as i32,
                group as i32,
                false,
            )
            .unwrap();
        }
        let reps = if m > 16 { 10 } else { 20 };
        let mut incl = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let dp = GpuBuffer::from_slice(&packed).unwrap();
            let ds = GpuBuffer::from_slice(&scales).unwrap();
            let dx = GpuBuffer::from_slice(&x).unwrap();
            let mut dy = GpuBuffer::<f32>::alloc(m).unwrap();
            acc.ternary_gemv(
                &dp,
                &ds,
                &dx,
                &mut dy,
                m as i32,
                k as i32,
                group as i32,
                false,
            )
            .unwrap();
            let _ = dy.to_vec().unwrap();
            incl.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let dp = GpuBuffer::from_slice(&packed).unwrap();
        let ds = GpuBuffer::from_slice(&scales).unwrap();
        let dx = GpuBuffer::from_slice(&x).unwrap();
        let mut dy = GpuBuffer::<f32>::alloc(m).unwrap();
        let mut kern = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            acc.ternary_gemv_async(
                &dp,
                &ds,
                &dx,
                &mut dy,
                m as i32,
                k as i32,
                group as i32,
                false,
            )
            .unwrap();
            acc.synchronize().unwrap();
            kern.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let mut cpu = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let _ = ternary_gemv_oracle(&packed, &scales, &x, m, k, group, false);
            cpu.push(t.elapsed().as_secs_f64() * 1e6);
        }
        println!(
            "gemv/{label}: transfer-inclusive={:.1}us kernel-only={:.1}us cpu-oracle={:.1}us",
            median(incl),
            median(kern),
            median(cpu)
        );
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoding-level RAW vs KINETIC comparison (issue #14).
//!
//! Issue #14 asks for a benchmark of RAW vs KINETIC vs HYBRID inputs against
//! held-out Supervisor v3 metrics (macro F1, false-safe rate, warning lead
//! time, ...). None of that exists in this repository: there is no training
//! loop, no Supervisor v3 evaluation code, no HYBRID feature adapter, and no
//! real telemetry checked in (see the `src/kinetic.rs` module docs and the
//! `Spikenaut-SNN-Telemetry` dataset reference in `README.md`). That
//! benchmark needs infrastructure that lives outside `Spikenaut-SNN`.
//!
//! What *is* fully reproducible here is the encoding layer itself. This file
//! compares two arms over the same synthetic, strictly positive raw series,
//! through the same historical [`TelemetryEncoder`]:
//!
//! - **RAW** — the raw sample alone, rate-coded on channel 0; channels 1–15
//!   sit at the encoder's idle floor.
//! - **KINETIC** — [`KineticPipeline`]'s 11 named features via
//!   `KineticFeatures::to_encoder_frame`; channels 11–15 sit at the floor.
//!
//! It reports (`cargo test --test kinetic_raw_ablation -- --nocapture`) total
//! spike count, a per-channel spike histogram, and [`KineticPipeline::step`]
//! wall-clock cost per tick — the "feature computation overhead" issue #14
//! asks be measured rather than assumed negligible. There is no HYBRID arm
//! and no accuracy/F1/lead-time claim here; this is encoding statistics only.

#![allow(deprecated)]

use std::time::Instant;

use spikenaut_snn::encode::{CHANNEL_COUNT, INPUT_RANGE, TelemetryEncoder};
use spikenaut_snn::kinetic::{KineticPipeline, RAW_RANGE};

/// Deterministic, strictly positive synthetic series (oscillation + drift).
///
/// Strictly positive so kinetic-signals' surprise and log-return volatility
/// are defined on every tick (see `src/kinetic.rs`'s `observe_transition`).
fn synthetic_series(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            220.0 + 40.0 * (t / 17.0).sin() + 0.05 * t
        })
        .collect()
}

/// Map a raw sample onto channel 0 alone; channels 1.. stay at the idle floor.
///
/// Mirrors `KineticFeatures::to_encoder_frame`'s affine mapping (same
/// [`RAW_RANGE`] and [`INPUT_RANGE`]) so the two arms differ only in *how
/// many channels carry a real signal*, not in the range convention used to
/// get there.
fn raw_only_frame(raw: f64) -> [f32; CHANNEL_COUNT] {
    let (lo, hi) = INPUT_RANGE;
    let (min, max) = RAW_RANGE;
    let unit = ((raw - min) / (max - min)).clamp(0.0, 1.0);
    let mut frame = [lo; CHANNEL_COUNT];
    frame[0] = lo + (hi - lo) * unit as f32;
    frame
}

#[derive(Debug)]
struct ArmStats {
    total_spikes: usize,
    spikes_per_channel: [usize; CHANNEL_COUNT],
}

fn run_arm(series: &[f64], mut next_frame: impl FnMut(f64) -> [f32; CHANNEL_COUNT]) -> ArmStats {
    let mut encoder = TelemetryEncoder::new().expect("historical coin-map encoder");
    let mut spikes_per_channel = [0usize; CHANNEL_COUNT];
    for &raw in series {
        let frame = next_frame(raw);
        let output = encoder
            .encode_step(&frame)
            .expect("synthetic series stays finite");
        for spike in &output.spikes {
            spikes_per_channel[usize::from(spike.channel)] += 1;
        }
    }
    ArmStats {
        total_spikes: spikes_per_channel.iter().sum(),
        spikes_per_channel,
    }
}

/// KINETIC drives more channels away from the idle floor than RAW does, so it
/// should emit measurably more total spikes on the same underlying series.
/// This is a real (not tautological) claim: it depends on the computed
/// feature values actually landing away from zero, not just on channel count.
#[test]
fn kinetic_emits_more_total_spikes_than_raw_on_the_same_series() {
    let series = synthetic_series(512);

    let raw_stats = run_arm(&series, raw_only_frame);

    let mut pipeline = KineticPipeline::new();
    let kinetic_stats = run_arm(&series, |raw| {
        pipeline
            .step(raw)
            .unwrap_or_else(|err| panic!("{err}"))
            .to_encoder_frame()
    });

    eprintln!(
        "RAW:     {} total spikes, per-channel {:?}",
        raw_stats.total_spikes, raw_stats.spikes_per_channel
    );
    eprintln!(
        "KINETIC: {} total spikes, per-channel {:?}",
        kinetic_stats.total_spikes, kinetic_stats.spikes_per_channel
    );

    assert!(raw_stats.total_spikes > 0);
    assert!(kinetic_stats.total_spikes > 0);
    assert!(
        kinetic_stats.total_spikes > raw_stats.total_spikes,
        "RAW={} KINETIC={}: expected kinetic preprocessing to drive more \
         channels away from the idle floor, not fewer",
        raw_stats.total_spikes,
        kinetic_stats.total_spikes,
    );
}

/// Feature-computation overhead, measured rather than assumed negligible
/// (issue #14 acceptance criterion). The bound is generous on purpose: this
/// guards against a catastrophic regression, not a performance promise.
#[test]
fn kinetic_feature_computation_overhead_is_measured_and_bounded() {
    let series = synthetic_series(512);
    let reps = 20;

    let mut pipeline = KineticPipeline::new();
    let start = Instant::now();
    for _ in 0..reps {
        for &raw in &series {
            pipeline.step(raw).unwrap_or_else(|err| panic!("{err}"));
        }
    }
    let elapsed = start.elapsed();
    let ticks = reps * series.len();
    let ns_per_tick = elapsed.as_nanos() as f64 / ticks as f64;

    eprintln!("KineticPipeline::step: {ticks} ticks in {elapsed:?} ({ns_per_tick:.1} ns/tick)");

    assert!(ns_per_tick.is_finite());
    assert!(
        ns_per_tick < 10_000_000.0,
        "kinetic feature computation took {ns_per_tick:.1} ns/tick, \
         over the 10 ms/tick regression ceiling"
    );
}

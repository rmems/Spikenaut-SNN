// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `kinetic-signals` integration (issue #14): a fixed raw
//! fixture must produce a finite, deterministic kinetic feature sequence that
//! then rate-encodes through the existing `axon-encoder` path. The crate must
//! resolve from crates.io; it does not replace that encoder.

use std::path::Path;

use spikenaut_snn::encode::{CHANNEL_COUNT, INPUT_RANGE, TelemetryEncoder};
use spikenaut_snn::kinetic::{
    FEATURE_COUNT, FEATURE_NAMES, HISTORY_WINDOW, KINETIC_SIGNALS_CRATE_VERSION, KineticPipeline,
};

/// A 64-sample power-like fixture: slow climb, one jump, then a settle.
///
/// Long enough for [`spikenaut_snn::kinetic::HISTORY_WINDOW`] and for
/// `compute_hurst` (32 samples) to leave its `H = 0.5` default. Units are
/// watts-shaped so surprise / log-returns are defined (strictly positive).
fn power_fixture() -> [f64; HISTORY_WINDOW] {
    let mut samples = [180.0_f64; HISTORY_WINDOW];
    for (tick, sample) in samples.iter_mut().enumerate() {
        *sample = match tick {
            0..=31 => 180.0 + tick as f64 * 0.5,
            32 => 260.0,
            33..=47 => 248.0 - (tick - 33) as f64 * 0.25,
            _ => 240.0,
        };
    }
    samples
}

/// Acceptance: every named kinetic feature on the fixture is finite.
#[test]
fn the_fixture_yields_a_finite_feature_vector() {
    let mut pipeline = KineticPipeline::new();
    let mut last = None;
    for (tick, &raw) in power_fixture().iter().enumerate() {
        let features = pipeline
            .step(raw)
            .unwrap_or_else(|err| panic!("tick {tick}: {err}"));
        assert!(
            features.is_finite(),
            "tick {tick}: NaN/Inf in {:?}",
            features.as_array()
        );
        for (name, value) in FEATURE_NAMES.iter().zip(features.as_array()) {
            assert!(
                value.is_finite(),
                "tick {tick}: {name} = {value} is not finite"
            );
        }
        last = Some(features);
    }
    let last = last.expect("the fixture is non-empty");
    assert_eq!(pipeline.ticks(), HISTORY_WINDOW);
    assert!(last.hurst >= 0.0 && last.hurst <= 1.0);
    assert!(last.entropy_relative >= 0.0 && last.entropy_relative <= 1.0);
    assert!(last.volatility >= 0.0 && last.volatility <= 1.0);
}

/// Repeated replay of the same fixture/config produces the same sequence.
#[test]
fn replay_is_deterministic() {
    let fixture = power_fixture();
    let first = replay(&fixture);
    let second = replay(&fixture);
    assert_eq!(first, second, "the same prefix must replay bit-for-bit");
    assert_eq!(first.len(), fixture.len());
}

/// Feature values at time `t` depend only on samples `≤ t`.
///
/// A fresh pipeline fed the prefix `[0..=t]` must match the features recorded
/// at `t` on the full replay. If a future sample leaked into a windowed
/// statistic, the prefix run would disagree.
#[test]
fn features_at_t_depend_only_on_samples_up_to_t() {
    let fixture = power_fixture();
    let full = replay(&fixture);

    for t in 0..fixture.len() {
        let prefix = replay(&fixture[..=t]);
        assert_eq!(
            prefix.last(),
            Some(&full[t]),
            "tick {t}: prefix replay drifted from the full-run snapshot"
        );
    }
}

/// The kinetic vector feeds the #9 encoder path: finite, in range, 16 wide.
#[test]
fn kinetic_features_encode_through_axon_encoder() {
    let mut pipeline = KineticPipeline::new();
    let mut encoder = TelemetryEncoder::new().expect("shipped encoder");
    let fixture = power_fixture();

    for (tick, &raw) in fixture.iter().enumerate() {
        let (features, output) = pipeline
            .encode_step(raw, &mut encoder)
            .unwrap_or_else(|err| panic!("tick {tick}: {err}"));
        assert!(
            features.is_finite(),
            "tick {tick}: kinetic vector not finite"
        );
        let frame = features.to_encoder_frame();
        assert_eq!(frame.len(), CHANNEL_COUNT);
        let (lo, hi) = INPUT_RANGE;
        for (channel, &value) in frame.iter().enumerate() {
            assert!(
                value.is_finite() && (lo..=hi).contains(&value),
                "tick {tick} channel {channel} = {value}"
            );
        }
        for (channel, &value) in frame.iter().enumerate().skip(FEATURE_COUNT) {
            assert_eq!(
                value, lo,
                "unused width stays at the floor (channel {channel})"
            );
        }
        for spike in &output.spikes {
            assert!(
                usize::from(spike.channel) < CHANNEL_COUNT,
                "tick {tick}: spike on channel {}",
                spike.channel
            );
        }
    }
}

/// A rejected raw sample is a no-op for both the pipeline and the encoder.
#[test]
fn a_non_finite_raw_sample_does_not_touch_the_encoder() {
    let mut victim_pipeline = KineticPipeline::new();
    let mut control_pipeline = KineticPipeline::new();
    let mut victim_encoder = TelemetryEncoder::new().expect("shipped encoder");
    let mut control_encoder = TelemetryEncoder::new().expect("shipped encoder");

    for &raw in &power_fixture()[..8] {
        let (expected_features, expected_output) = control_pipeline
            .encode_step(raw, &mut control_encoder)
            .expect("finite");
        let (actual_features, actual_output) = victim_pipeline
            .encode_step(raw, &mut victim_encoder)
            .expect("finite");
        assert_eq!(actual_features, expected_features);
        assert_eq!(actual_output, expected_output);
    }

    let err = victim_pipeline
        .encode_step(f64::NAN, &mut victim_encoder)
        .expect_err("NaN must be rejected");
    assert!(err.to_string().contains("non-finite raw"));
    assert_eq!(victim_pipeline.ticks(), control_pipeline.ticks());

    let (expected_features, expected_output) = control_pipeline
        .encode_step(200.0, &mut control_encoder)
        .expect("finite");
    let (actual_features, actual_output) = victim_pipeline
        .encode_step(200.0, &mut victim_encoder)
        .expect("finite");
    assert_eq!(actual_features, expected_features);
    assert_eq!(actual_output, expected_output);
}

/// Acceptance from issue #14: `kinetic-signals` resolves to 0.4.x from
/// crates.io, not from a git or sibling-path pin, and the recorded version
/// matches the lockfile.
#[test]
fn kinetic_signals_resolves_from_crates_io() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");

    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "kinetic-signals""#))
        .expect("Cargo.lock has a kinetic-signals package entry");

    assert!(
        entry.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "kinetic-signals must come from the crates.io registry, got:\n{entry}",
    );
    assert!(
        entry.contains(&format!(r#"version = "{KINETIC_SIGNALS_CRATE_VERSION}""#)),
        "recorded version {KINETIC_SIGNALS_CRATE_VERSION} must match the lockfile, got:\n{entry}",
    );
    assert!(
        !lock.contains("source = \"git+") && !lock.contains("[[patch"),
        "every locked package must come from the registry",
    );
    for forbidden in ["neuromod", "silicon-bridge"] {
        assert!(
            !lock.contains(&format!("name = \"{forbidden}\"")),
            "{forbidden} must stay out of the dependency tree",
        );
    }
}

fn replay(samples: &[f64]) -> Vec<[f64; FEATURE_COUNT]> {
    let mut pipeline = KineticPipeline::new();
    samples
        .iter()
        .enumerate()
        .map(|(tick, &raw)| {
            pipeline
                .step(raw)
                .unwrap_or_else(|err| panic!("tick {tick}: {err}"))
                .as_array()
        })
        .collect()
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `kinetic-signals` integration (issue #14): a fixed raw
//! fixture must produce a finite, deterministic kinetic feature sequence that
//! then rate-encodes through the **live** exp-025 5-column path — five legal
//! sensors on axons 0–4, axons 5–15 never written. The crate must resolve from
//! crates.io, and it does not replace `axon-encoder`.
//!
//! The deprecated coin `TelemetryEncoder` is not on this path; its own
//! acceptance lives in `tests/telemetry_encoding.rs`.

use std::path::Path;

use spikenaut_snn::encode::{
    CHANNEL_COUNT, INPUT_RANGE, LIVE_COLUMNS, LIVE_LEGAL_COLUMNS, LiveTelemetryEncoder,
};
use spikenaut_snn::kinetic::{
    FEATURE_COUNT, FEATURE_NAMES, HISTORY_WINDOW, KINETIC_SIGNALS_CRATE_VERSION, KineticPipeline,
    LiveKineticFrontEnd,
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

/// A five-sensor live reading fixture, in `LIVE_COLUMNS` order.
///
/// Axon 1 (`power_w`) replays [`power_fixture`]; the other four are shaped like
/// the same episode (a climb, a jump at tick 32, then a settle) in their own
/// physical units, so every sensor exercises a different part of its frozen
/// span rather than tracking one shared curve.
fn live_fixture() -> [[f64; LIVE_LEGAL_COLUMNS]; HISTORY_WINDOW] {
    let power = power_fixture();
    std::array::from_fn(|tick| {
        let unit = (power[tick] - 180.0) / 100.0;
        [
            20.0 + unit * 50.0,       // mem_util_pct
            power[tick],              // power_w
            45.0 + unit * 20.0,       // gpu_temp_c
            1_400.0 + unit * 1_200.0, // sm_clock_mhz
            7_000.0 + unit * 6_000.0, // mem_clock_mhz
        ]
    })
}

/// The live front end replays deterministically and never leaves samples `<= t`.
///
/// Same two properties the single pipeline is held to, asserted on all five
/// sensors at once: a fresh front end fed the prefix `[0..=t]` must match the
/// features recorded at `t` on the full replay.
#[test]
fn live_features_at_t_depend_only_on_samples_up_to_t() {
    let fixture = live_fixture();
    let full = replay_live(&fixture);
    assert_eq!(full, replay_live(&fixture), "replay must be bit-for-bit");

    for t in 0..fixture.len() {
        let prefix = replay_live(&fixture[..=t]);
        assert_eq!(
            prefix.last(),
            Some(&full[t]),
            "tick {t}: prefix replay drifted from the full-run snapshot",
        );
    }
}

/// Drive the whole live fixture through a fresh front end and encoder.
///
/// Asserts the per-tick invariants — five-wide frame, every feature and every
/// axon value finite and inside [`INPUT_RANGE`] — and returns the per-axon
/// spike counts so callers can assert on the train as a whole. Taking a fresh
/// pair each call is what makes it a replay: no state survives between runs.
fn encode_live_fixture() -> [usize; CHANNEL_COUNT] {
    let mut front_end = LiveKineticFrontEnd::new();
    let mut encoder =
        LiveTelemetryEncoder::for_shipped_merged_v2().expect("the legitimate live pairing");
    let (lo, hi) = INPUT_RANGE;
    let mut fired = [0_usize; CHANNEL_COUNT];

    for (tick, &reading) in live_fixture().iter().enumerate() {
        let (features, output) = front_end
            .encode_step(reading, &mut encoder)
            .unwrap_or_else(|err| panic!("tick {tick}: {err}"));

        assert_eq!(features.len(), LIVE_LEGAL_COLUMNS);
        let frame = LiveKineticFrontEnd::to_live_frame(&features);
        assert_eq!(
            frame.len(),
            LIVE_LEGAL_COLUMNS,
            "the live frame is five wide, not sixteen",
        );

        for (axon, sensor) in features.iter().enumerate() {
            assert!(
                sensor.is_finite(),
                "tick {tick} axon {axon} ({}): NaN/Inf in {:?}",
                LIVE_COLUMNS[axon],
                sensor.as_array(),
            );
            let value = frame[axon];
            assert!(
                value.is_finite() && (lo..=hi).contains(&value),
                "tick {tick} axon {axon} ({}) = {value}",
                LIVE_COLUMNS[axon],
            );
        }

        for spike in &output.spikes {
            fired[usize::from(spike.channel)] += 1;
        }
    }
    fired
}

/// The live path: a 5-wide frame, finite throughout, and nothing on axons 5-15.
#[test]
fn kinetic_features_encode_through_the_live_five_column_encoder() {
    let fired = encode_live_fixture();
    assert!(
        fired[..LIVE_LEGAL_COLUMNS].iter().all(|&count| count > 0),
        "every live axon fired over the fixture: {fired:?}",
    );
    assert!(
        fired[LIVE_LEGAL_COLUMNS..].iter().all(|&count| count == 0),
        "axons 5-15 are unused width on this bank and must stay at zero: {fired:?}",
    );
}

/// The same fixture through a fresh front end and encoder yields the same train.
///
/// `encode_step` is the streaming path, so the spike train is a deterministic
/// function of the readings and the configuration — nothing here may depend on
/// a random draw.
#[test]
fn the_live_encode_path_replays_deterministically() {
    assert_eq!(
        encode_live_fixture(),
        encode_live_fixture(),
        "replay must reproduce the same per-axon spike counts",
    );
}

/// A rejected raw reading is a no-op for every pipeline and for the encoder.
#[test]
fn a_non_finite_raw_sample_does_not_touch_the_encoder() {
    let mut victim_front_end = LiveKineticFrontEnd::new();
    let mut control_front_end = LiveKineticFrontEnd::new();
    let mut victim_encoder = LiveTelemetryEncoder::new().expect("live constants");
    let mut control_encoder = LiveTelemetryEncoder::new().expect("live constants");

    for &reading in &live_fixture()[..8] {
        let (expected_features, expected_output) = control_front_end
            .encode_step(reading, &mut control_encoder)
            .expect("finite");
        let (actual_features, actual_output) = victim_front_end
            .encode_step(reading, &mut victim_encoder)
            .expect("finite");
        assert_eq!(actual_features, expected_features);
        assert_eq!(actual_output, expected_output);
    }

    // A single bad sensor rejects the whole reading; the other four stay put.
    let mut faulty = live_fixture()[8];
    faulty[2] = f64::NAN;
    let err = victim_front_end
        .encode_step(faulty, &mut victim_encoder)
        .expect_err("NaN must be rejected");
    assert!(err.to_string().contains("non-finite raw"));
    assert_eq!(victim_front_end.ticks(), control_front_end.ticks());
    assert_eq!(
        victim_encoder, control_encoder,
        "the encoder must not have advanced on a rejected reading",
    );

    let next = live_fixture()[8];
    let (expected_features, expected_output) = control_front_end
        .encode_step(next, &mut control_encoder)
        .expect("finite");
    let (actual_features, actual_output) = victim_front_end
        .encode_step(next, &mut victim_encoder)
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
    assert!(
        !entry.contains("neuromod") && !entry.contains("silicon-bridge"),
        "kinetic-signals itself must not depend on neuromod or silicon-bridge, got:\n{entry}",
    );
    assert!(
        !lock.contains("name = \"silicon-bridge\""),
        "silicon-bridge must stay out of the dependency tree",
    );
}

fn replay_live(
    readings: &[[f64; LIVE_LEGAL_COLUMNS]],
) -> Vec<[[f64; FEATURE_COUNT]; LIVE_LEGAL_COLUMNS]> {
    let mut front_end = LiveKineticFrontEnd::new();
    readings
        .iter()
        .enumerate()
        .map(|(tick, &reading)| {
            let features = front_end
                .step(reading)
                .unwrap_or_else(|err| panic!("tick {tick}: {err}"));
            std::array::from_fn(|axon| features[axon].as_array())
        })
        .collect()
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

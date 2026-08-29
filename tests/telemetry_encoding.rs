// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `axon-encoder` integration (issue #9): a 16-wide frame
//! shaped like the documented channel map must rate-encode into a defined spike
//! count at the model's 1 kHz clock, and `axon-encoder` must resolve from
//! crates.io.

use std::path::Path;

use axon_encoder::Encoder;
use axon_encoder::encoders::RateEncoder;
use axon_encoder::error::EncoderError;
use axon_encoder::types::EncodedOutput;
use spikenaut_snn::encode::{
    BASE_RATE_HZ, CHANNEL_COUNT, CHANNEL_MAP, DT_SECONDS, INPUT_RANGE, MAX_RATE_HZ,
    TelemetryEncoder, TelemetrySource,
};
use spikenaut_snn::model::NEURON_COUNT;

/// One second of ticks at the model's 1 kHz clock.
const TICKS_PER_SECOND: usize = 1000;

/// Ticks between spikes on a saturated channel: `MAX_RATE_HZ * DT_SECONDS` is
/// `0.2`, so the accumulator crosses 1.0 on every fifth tick.
const TICKS_PER_SPIKE: usize = 5;

/// A 16-wide frame shaped like the documented channel map: each telemetry
/// source drives its own channel pair at its own intensity, spanning the whole
/// of [`INPUT_RANGE`] from an idle DNX feed to a saturated thermal reading.
fn telemetry_frame() -> [f32; CHANNEL_COUNT] {
    let sources = TelemetrySource::ALL.len();
    let mut frame = [0.0_f32; CHANNEL_COUNT];
    for (index, source) in TelemetrySource::ALL.into_iter().enumerate() {
        let intensity = index as f32 / (sources - 1) as f32;
        for channel in source.channels() {
            frame[channel] = intensity;
        }
    }
    frame
}

/// The firing rate a frame value maps to, in hertz.
fn expected_rate_hz(value: f32) -> f32 {
    let (min, max) = INPUT_RANGE;
    let normalized = ((value - min) / (max - min)).clamp(0.0, 1.0);
    BASE_RATE_HZ + normalized * (MAX_RATE_HZ - BASE_RATE_HZ)
}

/// Spikes per channel in one [`EncodedOutput`].
fn counts_per_channel(output: &EncodedOutput) -> [usize; CHANNEL_COUNT] {
    let mut counts = [0_usize; CHANNEL_COUNT];
    for spike in &output.spikes {
        counts[usize::from(spike.channel)] += 1;
    }
    counts
}

/// Acceptance criterion from issue #9: `RateEncoder::try_new` with an explicit
/// `dt_seconds` accepts the shipped configuration and encodes a 16-wide frame
/// without panicking.
#[test]
fn a_sixteen_wide_frame_encodes_without_panicking() {
    let mut encoder = RateEncoder::try_new(BASE_RATE_HZ, MAX_RATE_HZ, INPUT_RANGE, DT_SECONDS)
        .expect("the shipped configuration is a valid RateEncoder");

    // The 1 kHz clock is the whole point of passing `dt_seconds` explicitly:
    // `RateEncoder::new` would have substituted a 100 ms compatibility step.
    assert_eq!(encoder.dt_seconds(), DT_SECONDS);
    assert_ne!(DT_SECONDS, RateEncoder::DEFAULT_DT_SECONDS);

    let frame = telemetry_frame();
    assert_eq!(frame.len(), NEURON_COUNT, "one channel per LIF unit");

    let output = encoder.encode(&frame);
    assert!(
        output.spikes.len() <= CHANNEL_COUNT,
        "batch mode emits at most one spike per channel, got {}",
        output.spikes.len(),
    );
    for spike in &output.spikes {
        assert!(
            usize::from(spike.channel) < CHANNEL_COUNT,
            "spike on channel {} is outside the map",
            spike.channel,
        );
        assert!(spike.polarity, "rate encoding emits positive spikes only");
    }
}

/// Streaming mode is deterministic: over one second of 1 ms ticks every channel
/// emits its mapped firing rate, so the spike count is defined rather than
/// merely non-panicking.
#[test]
fn one_second_of_ticks_reproduces_the_mapped_rates() {
    let mut encoder = TelemetryEncoder::new().expect("shipped encoder");
    let frame = telemetry_frame();

    let mut totals = [0_usize; CHANNEL_COUNT];
    for _ in 0..TICKS_PER_SECOND {
        let output = encoder.encode_step(&frame);
        for (channel, count) in counts_per_channel(&output).into_iter().enumerate() {
            totals[channel] += count;
        }
    }

    for (channel, &count) in totals.iter().enumerate() {
        // One second of ticks, so the count is the rate in hertz, up to the one
        // spike still sitting in the accumulator.
        let expected = expected_rate_hz(frame[channel]);
        assert!(
            (count as f32 - expected).abs() <= 1.0,
            "channel {channel} ({}) fired {count} times, expected ~{expected} Hz",
            CHANNEL_MAP[channel].label(),
        );
    }

    // The idle DNX pair still ticks (a silent channel would be indistinguishable
    // from a dead feed) and the saturated thermal pair is the loudest.
    assert!(totals[0] > 0, "the base rate keeps an idle channel alive");
    for source in TelemetrySource::ALL {
        let [low, high] = source.channels();
        assert_eq!(totals[low], totals[high], "{}", source.label());
        assert!(
            totals[low] <= totals[CHANNEL_COUNT - 1],
            "thermal is saturated, so no source outruns it",
        );
    }
}

/// Every fifth tick, and only every fifth tick, a saturated frame fires all 16
/// channels: 200 Hz at a 1 ms step is exactly one spike per five ticks.
#[test]
fn a_saturated_frame_fires_every_fifth_tick() {
    let mut encoder = TelemetryEncoder::new().expect("shipped encoder");
    let frame = [INPUT_RANGE.1; CHANNEL_COUNT];

    let per_tick = MAX_RATE_HZ * DT_SECONDS;
    assert!(
        (per_tick * TICKS_PER_SPIKE as f32 - 1.0).abs() < 1e-6,
        "{TICKS_PER_SPIKE} ticks of {per_tick} expected spikes must make one spike",
    );

    for tick in 1..=(TICKS_PER_SPIKE * 3) {
        let fired = encoder.encode_step(&frame).spikes.len();
        let expected = if tick % TICKS_PER_SPIKE == 0 {
            CHANNEL_COUNT
        } else {
            0
        };
        assert_eq!(fired, expected, "tick {tick}");
    }

    // A cold start is a cold start: the accumulators go back to zero.
    encoder.reset();
    for tick in 1..TICKS_PER_SPIKE {
        assert!(encoder.encode_step(&frame).spikes.is_empty(), "tick {tick}");
    }
    assert_eq!(encoder.encode_step(&frame).spikes.len(), CHANNEL_COUNT);
}

/// An invalid configuration comes back as an error, not a panic. `try_new` is
/// the reason to prefer it over `RateEncoder::new`, which unwraps.
#[test]
fn an_invalid_configuration_is_an_error() {
    for (dt_seconds, label) in [(0.0, "zero"), (-0.001, "negative"), (f32::NAN, "NaN")] {
        let error = TelemetryEncoder::try_new(BASE_RATE_HZ, MAX_RATE_HZ, INPUT_RANGE, dt_seconds)
            .expect_err("a non-positive or non-finite time step must be rejected");
        assert_eq!(
            error,
            EncoderError::NonPositiveOrNonFinite {
                parameter: "dt_seconds",
            },
            "{label} dt_seconds",
        );
    }

    assert_eq!(
        TelemetryEncoder::try_new(MAX_RATE_HZ, BASE_RATE_HZ, INPUT_RANGE, DT_SECONDS)
            .expect_err("base_rate above max_rate must be rejected"),
        EncoderError::RateOrder,
    );
    assert_eq!(
        TelemetryEncoder::try_new(BASE_RATE_HZ, MAX_RATE_HZ, (1.0, 1.0), DT_SECONDS)
            .expect_err("a degenerate range must be rejected"),
        EncoderError::InvalidRange { parameter: "range" },
    );
}

/// The wrapper is a wrapper: it must agree tick for tick with a bare
/// `axon-encoder` `RateEncoder`, proving we exercise the real crate rather than
/// a local stand-in.
#[test]
fn the_wrapper_matches_a_bare_rate_encoder() {
    let mut wrapped = TelemetryEncoder::new().expect("shipped encoder");
    let mut bare = RateEncoder::try_new(BASE_RATE_HZ, MAX_RATE_HZ, INPUT_RANGE, DT_SECONDS)
        .expect("shipped configuration");
    let frame = telemetry_frame();

    for tick in 0..64 {
        assert_eq!(
            wrapped.encode_step(&frame),
            bare.encode_step(&frame),
            "tick {tick}",
        );
    }
    assert_eq!(wrapped.as_rate_encoder(), &bare, "state must stay in step");
}

/// Acceptance criterion from issue #9: `axon-encoder` resolves to 0.4.x from
/// crates.io, not from a git or sibling-path pin, and it does not drag
/// `neuromod` or `silicon-bridge` into the tree.
#[test]
fn axon_encoder_resolves_from_crates_io() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");

    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "axon-encoder""#))
        .expect("Cargo.lock has an axon-encoder package entry");

    assert!(
        entry.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "axon-encoder must come from the crates.io registry, got:\n{entry}",
    );
    assert!(
        entry.contains(r#"version = "0.4."#),
        "axon-encoder must resolve to 0.4.x, got:\n{entry}",
    );
    // The `"0.4"` requirement is `>=0.4.0, <0.5.0`; nothing may pin a checkout.
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

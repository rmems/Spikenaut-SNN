// SPDX-License-Identifier: MIT OR Apache-2.0

//! Telemetry → spikes, the front end of the 16-LIF population.
//!
//! [`crate::graph`] describes what the network *is*; this module describes what
//! it *eats*. The shipped model takes 16 continuous telemetry channels
//! (`n_channels` in `config.json`) and runs on a 1 kHz clock, so the input has
//! to be a spike train, not a float vector. `axon-encoder`'s [`RateEncoder`]
//! does that conversion, and [`TelemetryEncoder`] pins it to this model's
//! channel map and time step.
//!
//! # Channel map
//!
//! [`CHANNEL_MAP`] is the documented 16-channel telemetry layout, two channels
//! per source:
//!
//! | Channels | Source                    | Function                                    |
//! | -------- | ------------------------- | ------------------------------------------- |
//! | 0–1      | [`TelemetrySource::Dnx`]  | PoUW solver health and neural baselines     |
//! | 2–3      | [`TelemetrySource::Quai`] | Live on-chain reflex and sync confidence    |
//! | 4–5      | [`TelemetrySource::Qubic`] | Epoch and tick cadence monitoring           |
//! | 6–7      | [`TelemetrySource::Kaspa`] | High-frequency DAG settlement tracking      |
//! | 8–9      | [`TelemetrySource::Monero`] | Node stability and CPU L3 cache contention |
//! | 10–11    | [`TelemetrySource::Ocean`] | Data liquidity and staking prep             |
//! | 12–13    | [`TelemetrySource::Verus`] | CPU-heavy validator tracking                |
//! | 14–15    | [`TelemetrySource::Thermal`] | Pain receptors: power and temperature     |
//!
//! A frame is a `[f32; 16]` in [`INPUT_RANGE`] order-matched to that table.
//! Normalising raw telemetry into that range is the caller's job; this module
//! only clamps, and it only clamps *finite* values.
//!
//! # Non-finite samples
//!
//! A sensor that emits `NaN` or an infinity gets its frame rejected whole, by
//! every entry point, before any encoder state moves. See [`NonFiniteFrame`].
//!
//! Nothing is substituted, because there is no honest substitute. Zero is the
//! worst of them: on channels 14–15 it reads as a cool, idle GPU, which is
//! exactly the state the pain receptors exist to contradict. Silence is no
//! better — [`BASE_RATE_HZ`] is non-zero precisely so that a live-but-idle feed
//! stays distinguishable from a dead one, and a sample substituted away is a
//! dead feed that does not look like one.
//!
//! So the frame comes back as an error naming the offending channels, the
//! accumulators are left exactly as they were, and the next finite frame
//! encodes normally on the same channel. Recovery needs no [`reset`], and a
//! caller that ignores the result gets an `unused_must_use` warning rather than
//! a quietly deaf thermal channel.
//!
//! [`reset`]: TelemetryEncoder::reset
//!
//! # Batch and streaming
//!
//! [`RateEncoder`] has two modes and they are not interchangeable:
//!
//! - [`TelemetryEncoder::encode`] draws one independent Poisson-like sample per
//!   channel, so a channel emits at most one spike per call and the result is
//!   random.
//! - [`TelemetryEncoder::encode_step`] accumulates `rate_hz * dt_seconds` per
//!   channel and fires when the accumulator crosses 1.0. It is deterministic,
//!   and it is the mode that matches a fixed 1 ms hardware tick.
//!
//! # Scope
//!
//! This is the encoder front end only. Wiring the spike train into the NIR
//! graph, Q8.8 / FPGA mapping, and neuromodulator gain curves all belong to
//! their own tickets. This module deliberately stays independent of `neuromod`.

use std::fmt;

use axon_encoder::Encoder;
use axon_encoder::encoders::RateEncoder;
use axon_encoder::error::EncoderError;
use axon_encoder::types::EncodedOutput;

use crate::model::{CLOCK_HZ, NEURON_COUNT};

/// Width of one telemetry frame (`n_channels` in `config.json`).
///
/// The map is one input channel per unit, so this equals [`NEURON_COUNT`].
pub const CHANNEL_COUNT: usize = NEURON_COUNT;

/// Number of input channels each telemetry source drives.
pub const CHANNELS_PER_SOURCE: usize = 2;

/// Firing rate, in hertz, of a channel sitting at the bottom of [`INPUT_RANGE`].
///
/// Non-zero on purpose: a silent channel is indistinguishable from a dead feed,
/// so the floor doubles as a liveness tick.
pub const BASE_RATE_HZ: f32 = 5.0;

/// Firing rate, in hertz, of a channel saturated at the top of [`INPUT_RANGE`].
///
/// One fifth of the 1 kHz clock, so a saturated channel spikes every fifth tick
/// and still leaves headroom below the one-spike-per-step ceiling.
pub const MAX_RATE_HZ: f32 = 200.0;

/// The span a frame value is normalised against before it is mapped to a rate.
///
/// Values outside the span are clamped, not rejected.
pub const INPUT_RANGE: (f32, f32) = (0.0, 1.0);

/// Duration of one encode step, in seconds: the model's 1 kHz clock.
///
/// This is the `dt_seconds` handed to [`RateEncoder::try_new`], and it is the
/// same 1 ms step [`crate::model::TIMESTEP_SECONDS`] uses to invert the stored
/// decay rates, so the encoder and the LIF population share a time base.
pub const DT_SECONDS: f32 = 1.0 / CLOCK_HZ as f32;

/// A telemetry feed backing one pair of input channels.
///
/// See the [module docs](self#channel-map) for the full table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TelemetrySource {
    /// Dynex: PoUW solver health and neural baselines (channels 0–1).
    Dnx,
    /// Quai: live on-chain reflex and sync confidence (channels 2–3).
    Quai,
    /// Qubic: epoch and tick cadence monitoring (channels 4–5).
    Qubic,
    /// Kaspa: high-frequency DAG settlement tracking (channels 6–7).
    Kaspa,
    /// Monero: node stability and CPU L3 cache contention (channels 8–9).
    Monero,
    /// Ocean: data liquidity and staking prep (channels 10–11).
    Ocean,
    /// Verus: CPU-heavy validator tracking (channels 12–13).
    Verus,
    /// Thermal pain receptors: power and temperature (channels 14–15).
    Thermal,
}

impl TelemetrySource {
    /// Every source, in channel order.
    pub const ALL: [Self; CHANNEL_COUNT / CHANNELS_PER_SOURCE] = [
        Self::Dnx,
        Self::Quai,
        Self::Qubic,
        Self::Kaspa,
        Self::Monero,
        Self::Ocean,
        Self::Verus,
        Self::Thermal,
    ];

    /// The source's name as it appears in the repository's channel map.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dnx => "DNX",
            Self::Quai => "Quai",
            Self::Qubic => "Qubic",
            Self::Kaspa => "Kaspa",
            Self::Monero => "XMR",
            Self::Ocean => "Ocean",
            Self::Verus => "Verus",
            Self::Thermal => "Thermal",
        }
    }

    /// The input channels this source drives.
    #[must_use]
    pub const fn channels(self) -> [usize; CHANNELS_PER_SOURCE] {
        let first = self as usize * CHANNELS_PER_SOURCE;
        [first, first + 1]
    }

    /// Whether this source is one of the thermal pain receptors.
    ///
    /// Channels 14–15 carry power and temperature; the network is trained to
    /// treat them as a damage signal rather than as ordinary telemetry.
    #[must_use]
    pub const fn is_pain_receptor(self) -> bool {
        matches!(self, Self::Thermal)
    }
}

/// The source driving each of the 16 input channels, in channel order.
pub const CHANNEL_MAP: [TelemetrySource; CHANNEL_COUNT] = [
    TelemetrySource::Dnx,
    TelemetrySource::Dnx,
    TelemetrySource::Quai,
    TelemetrySource::Quai,
    TelemetrySource::Qubic,
    TelemetrySource::Qubic,
    TelemetrySource::Kaspa,
    TelemetrySource::Kaspa,
    TelemetrySource::Monero,
    TelemetrySource::Monero,
    TelemetrySource::Ocean,
    TelemetrySource::Ocean,
    TelemetrySource::Verus,
    TelemetrySource::Verus,
    TelemetrySource::Thermal,
    TelemetrySource::Thermal,
];

/// A telemetry frame rejected for carrying a value that is not finite.
///
/// Returned by [`TelemetryEncoder::encode_step`] and
/// [`TelemetryEncoder::encode`] when any channel holds a `NaN`, a `+inf` or a
/// `-inf`. The rejection is all-or-nothing: no channel's accumulator advanced
/// and no spike was emitted, so the encoder is left in exactly the state it was
/// in before the call and the next finite frame encodes normally.
///
/// It names every offending channel rather than just the first, because one
/// upstream fault usually takes out a source's whole channel pair, and because
/// [`touches_pain_receptor`](Self::touches_pain_receptor) is the question a
/// caller actually has to answer: a missing thermal reading is a safety event,
/// not a dropped sample.
///
/// # Example
///
/// ```
/// use spikenaut_snn::encode::{CHANNEL_COUNT, TelemetryEncoder};
///
/// let mut encoder = TelemetryEncoder::new()?;
/// let mut frame = [0.5_f32; CHANNEL_COUNT];
/// frame[14] = f32::NAN;
///
/// let rejected = encoder.encode_step(&frame).unwrap_err();
/// assert_eq!(rejected.channels().collect::<Vec<_>>(), [14]);
/// assert!(rejected.touches_pain_receptor());
/// assert_eq!(
///     rejected.to_string(),
///     "non-finite telemetry on channel 14 (Thermal); \
///      the thermal pain receptors are affected",
/// );
/// # Ok::<(), axon_encoder::EncoderError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NonFiniteFrame {
    /// Which channels were not finite, indexed in channel order. At least one
    /// entry is always `true`.
    offenders: [bool; CHANNEL_COUNT],
}

impl NonFiniteFrame {
    /// The rejection for `frame`, or `None` if every channel is finite.
    ///
    /// Scans the whole frame up front, which is what makes rejection atomic:
    /// [`RateEncoder`] walks channels in order and mutates each accumulator as
    /// it goes, so bailing out part-way through a delegated call would leave
    /// the earlier channels advanced by a frame that was never accepted.
    fn from_frame(frame: &[f32; CHANNEL_COUNT]) -> Option<Self> {
        let mut offenders = [false; CHANNEL_COUNT];
        let mut rejected = false;
        for (channel, &value) in frame.iter().enumerate() {
            if !value.is_finite() {
                offenders[channel] = true;
                rejected = true;
            }
        }
        rejected.then_some(Self { offenders })
    }

    /// The offending channel indices, ascending. Never empty.
    pub fn channels(self) -> impl Iterator<Item = usize> {
        self.offenders
            .into_iter()
            .enumerate()
            .filter_map(|(channel, offending)| offending.then_some(channel))
    }

    /// The lowest offending channel index.
    #[must_use]
    pub fn first_channel(self) -> usize {
        self.channels()
            .next()
            .expect("a rejection always names at least one channel")
    }

    /// How many channels were not finite. Always at least one.
    #[must_use]
    pub fn count(self) -> usize {
        self.offenders
            .iter()
            .filter(|offending| **offending)
            .count()
    }

    /// Whether `channel` was one of the offenders.
    #[must_use]
    pub fn contains(self, channel: usize) -> bool {
        self.offenders.get(channel).copied().unwrap_or(false)
    }

    /// Whether the rejection touches a thermal pain receptor (channels 14–15).
    ///
    /// The escalation hook. Losing a market feed for a tick is a dropped
    /// sample; losing a temperature reading is the hardware-protection input
    /// going dark, and a caller should treat the two differently.
    #[must_use]
    pub fn touches_pain_receptor(self) -> bool {
        self.channels()
            .any(|channel| CHANNEL_MAP[channel].is_pain_receptor())
    }
}

impl fmt::Display for NonFiniteFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("non-finite telemetry on channel")?;
        if self.count() > 1 {
            f.write_str("s")?;
        }
        for (position, channel) in self.channels().enumerate() {
            let separator = if position == 0 { " " } else { ", " };
            write!(f, "{separator}{channel} ({})", CHANNEL_MAP[channel].label())?;
        }
        if self.touches_pain_receptor() {
            f.write_str("; the thermal pain receptors are affected")?;
        }
        Ok(())
    }
}

impl std::error::Error for NonFiniteFrame {}

/// A rate encoder fixed to this model's channel map and 1 kHz clock.
///
/// A thin wrapper over `axon-encoder`'s [`RateEncoder`]. The wrapper exists for
/// the frame type: [`Encoder::encode`] takes any-width slice, while every method
/// here takes `&[f32; CHANNEL_COUNT]`, so a frame that does not match the
/// documented map is a compile error rather than a quietly short spike train.
///
/// # Example
///
/// ```
/// use spikenaut_snn::encode::{CHANNEL_COUNT, DT_SECONDS, TelemetryEncoder};
///
/// let mut encoder = TelemetryEncoder::new()?;
/// assert_eq!(encoder.dt_seconds(), DT_SECONDS);
///
/// // Every channel saturated: 200 Hz at a 1 ms step is one spike every 5 ticks.
/// let frame = [1.0_f32; CHANNEL_COUNT];
/// for tick in 0..4 {
///     assert!(encoder.encode_step(&frame)?.spikes.is_empty(), "tick {tick}");
/// }
/// assert_eq!(encoder.encode_step(&frame)?.spikes.len(), CHANNEL_COUNT);
///
/// // A non-finite thermal sample is rejected by name, and costs nothing: the
/// // accumulators never moved, so the cycle picks up exactly where it left off.
/// let mut faulty = frame;
/// faulty[14] = f32::NAN;
/// let rejected = encoder.encode_step(&faulty).unwrap_err();
/// assert_eq!(rejected.first_channel(), 14);
/// assert!(rejected.touches_pain_receptor());
///
/// for tick in 0..4 {
///     assert!(encoder.encode_step(&frame)?.spikes.is_empty(), "tick {tick}");
/// }
/// assert_eq!(encoder.encode_step(&frame)?.spikes.len(), CHANNEL_COUNT);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryEncoder {
    inner: RateEncoder,
}

impl TelemetryEncoder {
    /// Build the encoder this model ships with.
    ///
    /// [`BASE_RATE_HZ`] to [`MAX_RATE_HZ`] over [`INPUT_RANGE`], stepped at
    /// [`DT_SECONDS`].
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError`] if the shipped constants ever stop being a valid
    /// [`RateEncoder`] configuration. They are checked by the test suite, so in
    /// practice this is infallible.
    pub fn new() -> Result<Self, EncoderError> {
        Self::try_new(BASE_RATE_HZ, MAX_RATE_HZ, INPUT_RANGE, DT_SECONDS)
    }

    /// Build the encoder with an explicit configuration.
    ///
    /// `dt_seconds` is always explicit here: [`RateEncoder::new`] would silently
    /// substitute its 100 ms compatibility step, which is a hundred ticks of
    /// this model's clock.
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError`] if the rates are not finite and non-negative
    /// with `base_rate_hz <= max_rate_hz`, if `range` is not a finite increasing
    /// span, or if `dt_seconds` is not finite and strictly positive.
    pub fn try_new(
        base_rate_hz: f32,
        max_rate_hz: f32,
        range: (f32, f32),
        dt_seconds: f32,
    ) -> Result<Self, EncoderError> {
        Ok(Self {
            inner: RateEncoder::try_new(base_rate_hz, max_rate_hz, range, dt_seconds)?,
        })
    }

    /// The configured step duration, in seconds.
    #[must_use]
    pub fn dt_seconds(&self) -> f32 {
        self.inner.dt_seconds()
    }

    /// Encode one frame as a single 1 ms tick (streaming, deterministic).
    ///
    /// Each channel accumulates `rate_hz * dt_seconds` and emits a spike when
    /// the accumulator crosses 1.0, so consecutive calls form a spike train at
    /// the channel's mapped rate. This is the mode that matches the hardware
    /// tick; see the [module docs](self#batch-and-streaming).
    ///
    /// # Errors
    ///
    /// Returns [`NonFiniteFrame`] if any channel holds a `NaN` or an infinity.
    /// The whole frame is checked before the first accumulator is touched, so a
    /// rejected frame is a no-op: no channel advanced, no spike was emitted,
    /// and the next finite frame encodes normally. See the
    /// [module docs](self#non-finite-samples) for why nothing is substituted.
    pub fn encode_step(
        &mut self,
        frame: &[f32; CHANNEL_COUNT],
    ) -> Result<EncodedOutput, NonFiniteFrame> {
        match NonFiniteFrame::from_frame(frame) {
            Some(rejected) => Err(rejected),
            None => Ok(self.inner.encode_step(frame)),
        }
    }

    /// Encode one frame as an independent Poisson draw (batch, stochastic).
    ///
    /// Emits at most one spike per channel, with probability
    /// `1 - exp(-rate_hz * dt_seconds)`. Carries no state between calls, so the
    /// spike count is random; use [`encode_step`](Self::encode_step) when the
    /// train has to follow the clock.
    ///
    /// # Errors
    ///
    /// Returns [`NonFiniteFrame`] on the same terms as
    /// [`encode_step`](Self::encode_step). This path keeps no accumulator to
    /// poison, but a channel silently dropped from the draw is just as
    /// invisible here, so the frame is validated before any spike is drawn.
    pub fn encode(
        &mut self,
        frame: &[f32; CHANNEL_COUNT],
    ) -> Result<EncodedOutput, NonFiniteFrame> {
        match NonFiniteFrame::from_frame(frame) {
            Some(rejected) => Err(rejected),
            None => Ok(self.inner.encode(frame)),
        }
    }

    /// Clear the per-channel accumulators, returning the encoder to a cold start.
    ///
    /// Only affects [`encode_step`](Self::encode_step); the batch path is
    /// stateless.
    ///
    /// This is *not* the recovery path for a bad sample. A frame rejected as
    /// [`NonFiniteFrame`] never reached the accumulators, so there is nothing
    /// to clear; calling `reset` there would throw away the phase of the
    /// fifteen healthy channels for no reason.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The wrapped `axon-encoder` encoder.
    ///
    /// An escape hatch for the parts of the [`RateEncoder`] surface this wrapper
    /// does not re-expose, such as the neuromodulator-gain entry points.
    #[must_use]
    pub fn as_rate_encoder(&self) -> &RateEncoder {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_map_covers_every_channel_once() {
        assert_eq!(CHANNEL_MAP.len(), CHANNEL_COUNT);
        assert_eq!(
            TelemetrySource::ALL.len() * CHANNELS_PER_SOURCE,
            CHANNEL_COUNT,
        );

        // `channels()` is derived from the variant order; `CHANNEL_MAP` is
        // written out by hand. They must agree.
        for source in TelemetrySource::ALL {
            for channel in source.channels() {
                assert_eq!(CHANNEL_MAP[channel], source, "channel {channel}");
            }
        }
    }

    #[test]
    fn thermal_owns_the_last_channel_pair() {
        assert_eq!(TelemetrySource::Thermal.channels(), [14, 15]);
        assert!(TelemetrySource::Thermal.is_pain_receptor());
        assert_eq!(
            TelemetrySource::ALL
                .iter()
                .filter(|source| source.is_pain_receptor())
                .count(),
            1,
            "thermal is the only pain receptor",
        );
        assert_eq!(TelemetrySource::Dnx.label(), "DNX");
    }

    #[test]
    fn dt_seconds_is_one_millisecond() {
        assert_eq!(DT_SECONDS, 0.001_f32);
        assert!(
            (f64::from(DT_SECONDS) - crate::model::TIMESTEP_SECONDS).abs() < 1e-9,
            "the encoder step must be the model's simulation step",
        );
    }

    #[test]
    fn shipped_configuration_is_valid() {
        let encoder = TelemetryEncoder::new().expect("shipped constants are a valid encoder");
        assert_eq!(encoder.dt_seconds(), DT_SECONDS);
    }

    #[test]
    fn a_rejection_names_every_offending_channel() {
        let mut frame = [0.5_f32; CHANNEL_COUNT];
        frame[0] = f32::NAN;
        frame[14] = f32::INFINITY;
        frame[15] = f32::NEG_INFINITY;

        let rejected = NonFiniteFrame::from_frame(&frame).expect("three channels are not finite");
        assert_eq!(rejected.channels().collect::<Vec<_>>(), [0, 14, 15]);
        assert_eq!(rejected.count(), 3);
        assert_eq!(rejected.first_channel(), 0);
        assert!(rejected.contains(14));
        assert!(!rejected.contains(1));
        assert!(
            !rejected.contains(CHANNEL_COUNT),
            "out of range is not an offender"
        );
        assert!(rejected.touches_pain_receptor());
        assert_eq!(
            rejected.to_string(),
            "non-finite telemetry on channels 0 (DNX), 14 (Thermal), 15 (Thermal); \
             the thermal pain receptors are affected",
        );
    }

    #[test]
    fn a_rejection_away_from_thermal_is_not_a_pain_event() {
        let mut frame = [0.5_f32; CHANNEL_COUNT];
        frame[3] = f32::NAN;

        let rejected = NonFiniteFrame::from_frame(&frame).expect("channel 3 is not finite");
        assert!(!rejected.touches_pain_receptor());
        assert_eq!(
            rejected.to_string(),
            "non-finite telemetry on channel 3 (Quai)",
        );
    }

    #[test]
    fn a_finite_frame_is_not_a_rejection() {
        // The edges of f32 are finite and must stay encodable; only NaN and the
        // infinities are rejected.
        let frame = [
            f32::MIN,
            f32::MAX,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1e30,
            -1e30,
            0.5,
        ]
        .into_iter()
        .cycle()
        .take(CHANNEL_COUNT)
        .collect::<Vec<_>>();
        let frame: [f32; CHANNEL_COUNT] = frame.try_into().expect("sixteen values");
        assert_eq!(NonFiniteFrame::from_frame(&frame), None);
    }

    #[test]
    fn a_non_positive_step_is_an_error_not_a_panic() {
        let error = TelemetryEncoder::try_new(BASE_RATE_HZ, MAX_RATE_HZ, INPUT_RANGE, 0.0)
            .expect_err("dt_seconds must be strictly positive");
        assert_eq!(
            error,
            EncoderError::NonPositiveOrNonFinite {
                parameter: "dt_seconds",
            },
        );
    }
}

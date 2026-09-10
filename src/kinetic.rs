// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side kinetic preprocessing *upstream* of [`crate::encode`].
//!
//! [`kinetic-signals`](https://crates.io/crates/kinetic-signals) 0.4.x is a
//! causal temporal feature layer. It does **not** replace `axon-encoder`:
//!
//! ```text
//! raw telemetry
//!         ↓
//! kinetic-signals          ← this module (Hurst / Hawkes / surprise /
//! (host-side Rust)            volatility / entropy / EMA-SMA / Z-score /
//!         ↓                   moments — the 0.4.0 public surface)
//! adapter → [f32; 16]
//!         ↓
//! axon-encoder             ← [`crate::encode::TelemetryEncoder`]
//! (continuous → spikes)
//! ```
//!
//! This is the KINETIC arm of issue #14. RAW vs KINETIC vs HYBRID training
//! and held-out Supervisor v3 metrics are out of scope here: this module
//! only proves the published crate can sit in front of the existing encoder
//! without inventing kinetic-signals APIs or blocking FPGA parity. Software
//! and FPGA models should receive the same encoded sequence; Hurst / Hawkes
//! RTL is a later ticket.
//!
//! # Causality
//!
//! [`KineticPipeline::step`] updates estimators from the sample at time `t`
//! and windowed statistics over samples `≤ t` only. Nothing is fit on a
//! future split. A leakage test in `tests/kinetic_encoding.rs` checks that
//! the feature vector at `t` is unchanged when later samples are withheld.
//!
//! # Projection
//!
//! kinetic-signals returns named scalars, not a 16-wide axon frame. The
//! adapter in [`KineticFeatures::to_encoder_frame`] maps a small first-pass
//! set onto the existing encoder width and leaves unused channels at zero —
//! unused width, not invented signals. The squash functions are this crate's;
//! they are not part of kinetic-signals.

use std::collections::VecDeque;
use std::fmt;

use axon_encoder::types::EncodedOutput;
use kinetic_signals::{
    EMA, HawkesParams, SMA, SurpriseParams, VolEstimator, ZScore, compute_hawkes, compute_hurst,
    compute_shannon_entropy, compute_signal_stats, compute_surprise, detect_anomaly,
};

use crate::encode::{CHANNEL_COUNT, INPUT_RANGE, NonFiniteFrame, TelemetryEncoder};

/// crates.io version this integration was wired against.
///
/// `tests/kinetic_encoding.rs` asserts the lockfile resolves this exact
/// `0.4.x` package from `registry+https://github.com/rust-lang/crates.io-index`.
pub const KINETIC_SIGNALS_CRATE_VERSION: &str = "0.4.0";

/// Declared physical range of the raw series this first-pass adapter accepts.
///
/// kinetic-signals itself is domain-agnostic and does not normalise. Mapping
/// into [`INPUT_RANGE`] is the adapter's job, done only at projection time so
/// Hurst / surprise / moments see the raw units.
pub const RAW_RANGE: (f64, f64) = (0.0, 400.0);

/// EMA period handed to [`EMA::new`].
pub const EMA_PERIOD: usize = 8;

/// SMA window handed to [`SMA::new`].
pub const SMA_WINDOW: usize = 8;

/// Ring-buffer capacity handed to [`VolEstimator::new`].
pub const VOL_WINDOW: usize = 16;

/// Causal history retained for Hurst, entropy, and moments.
///
/// [`compute_hurst`] needs at least 32 samples before it leaves the
/// uninformative `H = 0.5` default; 64 gives it a short but legal window.
pub const HISTORY_WINDOW: usize = 64;

/// Histogram bins handed to [`compute_shannon_entropy`].
pub const ENTROPY_BINS: usize = 10;

/// How many named kinetic features the first-pass set produces.
pub const FEATURE_COUNT: usize = 11;

/// Names of [`KineticFeatures::as_array`], in order, for diagnostics.
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "raw",
    "ema",
    "sma",
    "z_score",
    "volatility",
    "surprise",
    "hurst",
    "entropy_relative",
    "hawkes_intensity",
    "skewness",
    "kurtosis",
];

/// A raw sample rejected before any kinetic-signals estimator moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NonFiniteSample;

impl fmt::Display for NonFiniteSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("non-finite raw telemetry sample")
    }
}

impl std::error::Error for NonFiniteSample {}

/// Failure of the kinetic front end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KineticError {
    /// The raw sample was `NaN` or infinite; pipeline state is unchanged.
    NonFiniteSample(NonFiniteSample),
    /// The projected frame was rejected by [`TelemetryEncoder`].
    NonFiniteFrame(NonFiniteFrame),
}

impl fmt::Display for KineticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteSample(err) => write!(f, "{err}"),
            Self::NonFiniteFrame(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for KineticError {}

impl From<NonFiniteSample> for KineticError {
    fn from(err: NonFiniteSample) -> Self {
        Self::NonFiniteSample(err)
    }
}

impl From<NonFiniteFrame> for KineticError {
    fn from(err: NonFiniteFrame) -> Self {
        Self::NonFiniteFrame(err)
    }
}

/// One causal tick of kinetic-signals features, computed from samples `≤ t`.
///
/// Every field is a value the 0.4.0 public API actually produces — not a
/// renamed or invented statistic. Projection into the 16-wide encoder frame
/// is [`to_encoder_frame`](Self::to_encoder_frame).
#[derive(Debug, Clone, PartialEq)]
pub struct KineticFeatures {
    /// The raw sample at `t`.
    pub raw: f64,
    /// [`EMA`] value after incorporating `raw`.
    pub ema: f64,
    /// [`SMA`] value after incorporating `raw`.
    pub sma: f64,
    /// [`ZScore::compute`] of `raw` against moments of the causal window.
    pub z_score: f64,
    /// [`VolEstimator::rms`] over absolute log-returns seen so far.
    pub volatility: f64,
    /// [`compute_surprise`] against the previous sample (`0.0` on the first tick).
    pub surprise: f64,
    /// [`compute_hurst`] `h` on the causal window (`0.5` before 32 samples).
    pub hurst: f64,
    /// [`compute_shannon_entropy`] relative entropy on the causal window.
    pub entropy_relative: f64,
    /// [`compute_hawkes`] intensity over anomaly events at times `≤ t`.
    pub hawkes_intensity: f64,
    /// [`compute_signal_stats`] skewness on the causal window.
    pub skewness: f64,
    /// [`compute_signal_stats`] excess kurtosis on the causal window.
    pub kurtosis: f64,
}

impl KineticFeatures {
    /// The named feature vector, in [`FEATURE_NAMES`] order.
    #[must_use]
    pub fn as_array(&self) -> [f64; FEATURE_COUNT] {
        [
            self.raw,
            self.ema,
            self.sma,
            self.z_score,
            self.volatility,
            self.surprise,
            self.hurst,
            self.entropy_relative,
            self.hawkes_intensity,
            self.skewness,
            self.kurtosis,
        ]
    }

    /// Whether every named feature is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.as_array().iter().all(|value| value.is_finite())
    }

    /// Map the named features onto the 16-wide [`TelemetryEncoder`] frame.
    ///
    /// Channels 0–10 carry the first-pass set; 11–15 stay at the bottom of
    /// [`INPUT_RANGE`] as unused width. Squashing is this adapter's, so a
    /// kinetic-signals scalar that is already in `[0, 1]` (Hurst, relative
    /// entropy, RMS volatility) is copied, not reshaped.
    #[must_use]
    pub fn to_encoder_frame(&self) -> [f32; CHANNEL_COUNT] {
        let (lo, hi) = INPUT_RANGE;
        let mut frame = [lo; CHANNEL_COUNT];
        let projected = [
            normalize_raw(self.raw),
            normalize_raw(self.ema),
            normalize_raw(self.sma),
            squash_signed(self.z_score),
            clamp_unit(self.volatility),
            squash_nonneg(self.surprise),
            clamp_unit(self.hurst),
            clamp_unit(self.entropy_relative),
            squash_nonneg(self.hawkes_intensity),
            squash_signed(self.skewness),
            squash_signed(self.kurtosis),
        ];
        for (slot, value) in projected.into_iter().enumerate() {
            frame[slot] = value.clamp(lo, hi);
        }
        frame
    }
}

/// Causal streaming adapter: one raw series → kinetic-signals → encoder frame.
///
/// State is updated only from the sample passed to [`step`](Self::step).
/// [`VolEstimator`] is not `Clone` in 0.4.0, so this pipeline is not either:
/// replay means constructing a fresh instance and feeding the same prefix.
pub struct KineticPipeline {
    ema: EMA,
    sma: SMA,
    vol: VolEstimator,
    surprise_params: SurpriseParams,
    hawkes_params: HawkesParams,
    history: VecDeque<f64>,
    event_times: Vec<f64>,
    previous: Option<f64>,
    ticks: usize,
}

impl KineticPipeline {
    /// Build the first-pass pipeline with the constants above and
    /// kinetic-signals [`Default`] parameter structs (`dt = 0.001`, matching
    /// the model's 1 kHz clock).
    #[must_use]
    pub fn new() -> Self {
        Self {
            ema: EMA::new(EMA_PERIOD),
            sma: SMA::new(SMA_WINDOW),
            vol: VolEstimator::new(VOL_WINDOW),
            surprise_params: SurpriseParams::default(),
            hawkes_params: HawkesParams::default(),
            history: VecDeque::with_capacity(HISTORY_WINDOW),
            event_times: Vec::new(),
            previous: None,
            ticks: 0,
        }
    }

    /// Samples accepted so far (the causal clock).
    #[must_use]
    pub fn ticks(&self) -> usize {
        self.ticks
    }

    /// Incorporate `raw` and return the feature vector at this tick.
    ///
    /// # Errors
    ///
    /// [`NonFiniteSample`] if `raw` is not finite. Estimators are not
    /// updated, so the next finite sample continues the same trajectory.
    pub fn step(&mut self, raw: f64) -> Result<KineticFeatures, NonFiniteSample> {
        if !raw.is_finite() {
            return Err(NonFiniteSample);
        }

        let ema = self.ema.update(raw);
        let sma = self.sma.update(raw);
        push_capped(&mut self.history, raw, HISTORY_WINDOW);

        let (surprise, volatility) = self.observe_transition(raw);
        let window = self.history.make_contiguous();
        let stats = compute_signal_stats(window);
        let z_score = ZScore::compute(raw, stats.mean, stats.variance.sqrt());
        let hurst = compute_hurst(window);
        let entropy = compute_shannon_entropy(window, ENTROPY_BINS);
        let hawkes = compute_hawkes(&self.event_times, &self.hawkes_params);

        self.previous = Some(raw);
        self.ticks += 1;

        Ok(KineticFeatures {
            raw,
            ema,
            sma,
            z_score,
            volatility: f64::from(volatility),
            surprise,
            hurst: hurst.h,
            entropy_relative: entropy.relative,
            hawkes_intensity: hawkes.intensity,
            skewness: stats.skewness,
            kurtosis: stats.kurtosis,
        })
    }

    /// [`Self::step`] then [`TelemetryEncoder::encode_step`] on the projected frame.
    ///
    /// # Errors
    ///
    /// [`KineticError::NonFiniteSample`] if `raw` is not finite (encoder
    /// untouched). [`KineticError::NonFiniteFrame`] if the projection is
    /// rejected — that path is defensive: a finite kinetic vector should
    /// project into [`INPUT_RANGE`].
    pub fn encode_step(
        &mut self,
        raw: f64,
        encoder: &mut TelemetryEncoder,
    ) -> Result<(KineticFeatures, EncodedOutput), KineticError> {
        let features = self.step(raw)?;
        let output = encoder.encode_step(&features.to_encoder_frame())?;
        Ok((features, output))
    }

    /// Surprise and volatility from `raw` versus the previous sample.
    ///
    /// Both kinetic-signals entry points need a defined log-ratio, so a first
    /// tick or a non-positive pair contributes no transition and no Hawkes
    /// event. Anomalous transitions are recorded at `ticks * dt` — a time
    /// that is always `≤ t` for the sample just accepted.
    fn observe_transition(&mut self, raw: f64) -> (f64, f32) {
        let Some(previous) = self.previous else {
            return (0.0, self.vol.rms());
        };
        let surprise = compute_surprise(raw, previous, &self.surprise_params);
        if previous > 0.0 && raw > 0.0 {
            self.vol.push((raw / previous).ln().abs() as f32);
            if detect_anomaly(&surprise, &self.surprise_params) {
                let t = self.ticks as f64 * self.surprise_params.dt;
                self.event_times.push(t);
            }
        }
        (surprise.surprise, self.vol.rms())
    }
}

impl Default for KineticPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Drop the oldest sample when `history` reaches `cap`.
///
/// A [`VecDeque`] keeps the eviction O(1). `Vec::remove(0)` would copy the
/// whole window on every tick after fill; the kinetic-signals slice APIs then
/// see a contiguous view via [`VecDeque::make_contiguous`].
fn push_capped(history: &mut VecDeque<f64>, raw: f64, cap: usize) {
    if history.len() == cap {
        history.pop_front();
    }
    history.push_back(raw);
}

/// Affine map of a raw-unit value through [`RAW_RANGE`] into [`INPUT_RANGE`].
fn normalize_raw(value: f64) -> f32 {
    let (min, max) = RAW_RANGE;
    let (lo, hi) = INPUT_RANGE;
    let span = max - min;
    if span <= 0.0 {
        return lo;
    }
    let unit = ((value - min) / span).clamp(0.0, 1.0);
    lo + (hi - lo) * unit as f32
}

/// Keep an already-unit interval in `[0, 1]`.
fn clamp_unit(value: f64) -> f32 {
    value.clamp(0.0, 1.0) as f32
}

/// Map an unbounded signed score into `(0, 1)` with a tanh squash.
fn squash_signed(value: f64) -> f32 {
    (0.5 + 0.5 * (value / 3.0).tanh()) as f32
}

/// Map an unbounded nonnegative score into `[0, 1)`.
fn squash_nonneg(value: f64) -> f32 {
    (1.0 - (-value.abs()).exp()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_finite_sample_does_not_advance_the_clock() {
        let mut pipeline = KineticPipeline::new();
        assert_eq!(pipeline.step(100.0).expect("finite").raw, 100.0);
        assert_eq!(pipeline.ticks(), 1);
        assert_eq!(pipeline.step(f64::NAN), Err(NonFiniteSample));
        assert_eq!(pipeline.step(f64::INFINITY), Err(NonFiniteSample));
        assert_eq!(pipeline.ticks(), 1);
        assert_eq!(pipeline.step(110.0).expect("recovered").raw, 110.0);
        assert_eq!(pipeline.ticks(), 2);
    }

    #[test]
    fn the_first_tick_has_no_surprise() {
        let features = KineticPipeline::new()
            .step(180.0)
            .expect("a finite sample steps");
        assert_eq!(features.surprise, 0.0);
        assert_eq!(features.ema, 180.0);
        assert_eq!(features.sma, 180.0);
        assert!(features.is_finite());
    }

    #[test]
    fn projection_stays_inside_the_encoder_range() {
        let mut pipeline = KineticPipeline::new();
        let features = pipeline.step(250.0).expect("finite");
        let frame = features.to_encoder_frame();
        let (lo, hi) = INPUT_RANGE;
        for (channel, &value) in frame.iter().enumerate() {
            assert!(
                value.is_finite() && (lo..=hi).contains(&value),
                "channel {channel} = {value}"
            );
        }
        for (channel, &value) in frame.iter().enumerate().skip(FEATURE_COUNT) {
            assert_eq!(
                value, lo,
                "unused width stays at the floor (channel {channel})"
            );
        }
    }

    #[test]
    fn feature_names_match_the_array() {
        assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
        let features = KineticPipeline::new().step(1.0).expect("finite");
        assert_eq!(features.as_array().len(), FEATURE_NAMES.len());
    }
}

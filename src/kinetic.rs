// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side kinetic preprocessing *upstream* of the live exp-025 encoder.
//!
//! [`kinetic-signals`](https://crates.io/crates/kinetic-signals) 0.4.x is a
//! causal temporal feature layer. It does **not** replace `axon-encoder`:
//!
//! ```text
//! raw telemetry            ← five live GPU sensors, [`LIVE_COLUMNS`] order
//!         ↓
//! kinetic-signals          ← this module (Hurst / Hawkes / surprise /
//! (host-side Rust)            volatility / entropy / EMA-SMA / Z-score /
//!         ↓                   moments — the 0.4.0 public surface)
//! adapter → [f32; 5]       ← [`LiveKineticFrontEnd::to_live_frame`]
//!         ↓
//! axon-encoder             ← [`LiveTelemetryEncoder`], axons 0–4 only
//! (continuous → spikes)      (axons 5–15 unwritten, held at zero)
//! ```
//!
//! One [`KineticPipeline`] runs per live sensor, so the five series keep
//! independent estimator state. Axons 5–15 are never written: they are unused
//! width on the exp-025 bank, held at *zero* in train, and this module does not
//! put a liveness tick — or anything else — on them. The deprecated coin
//! [`crate::encode::CHANNEL_MAP`] encoder, which does emit a nonzero base rate
//! on its unused channels, is not on this path at all.
//!
//! This is the KINETIC arm of issue #14. RAW vs KINETIC vs HYBRID training and
//! held-out Supervisor v3 metrics are out of scope here: this module only
//! proves the published crate can sit in front of the live 5-column encoder
//! without inventing kinetic-signals APIs or blocking FPGA parity. Software
//! and FPGA models should receive the same encoded sequence; Hurst / Hawkes
//! RTL is a later ticket.
//!
//! # Causality
//!
//! [`KineticPipeline::step`] updates estimators from the sample at time `t`
//! and windowed statistics over samples `≤ t` only. Nothing is fit on a
//! future split. A leakage test in `tests/kinetic_encoding.rs` checks that
//! the feature vector at `t` is unchanged when later samples are withheld, for
//! the single pipeline and for the five-sensor front end.
//!
//! # Projection
//!
//! kinetic-signals returns named scalars, not an axon frame.
//! [`LiveKineticFrontEnd::to_live_frame`] uses the **identity projection**:
//! each sensor's own normalised raw value lands on its own axon, in
//! [`LIVE_COLUMNS`] order, and the eleven kinetic features come back alongside
//! the frame as audit data rather than as stimulus. That is deliberate. Which
//! kinetic feature — if any — deserves an axon is exactly the RAW / KINETIC /
//! HYBRID question issue #14 parks, and choosing one here would answer it by
//! accident, on a bank that was trained on raw sensors.
//!
//! Normalisation is per sensor, through [`LIVE_RAW_RANGES`], because the five
//! live sensors do not share a physical span. The normaliser is this crate's;
//! it is not part of kinetic-signals.
//!
//! # Missing values
//!
//! A non-finite raw sample is rejected, never substituted, on the same terms as
//! [`crate::encode`]: [`LiveKineticFrontEnd::step`] rejects the whole five-wide
//! reading before any of the five pipelines moves, so the estimators stay in
//! the state they were in and the next finite reading continues the same
//! trajectory.
//!
//! Dropout sentinels are a *different* problem and this module does not solve
//! it. `gpu_temp_c == 0` is a dropout marker on this dataset, not a cold GPU,
//! and 0.0 is a finite number: it passes straight through here and normalises
//! to the bottom of [`crate::encode::INPUT_RANGE`]. Masking sentinels is the
//! caller's job under the state contract being defined in issue #20.

use std::collections::VecDeque;
use std::fmt;

use axon_encoder::types::EncodedOutput;
use kinetic_signals::{
    EMA, HawkesParams, SMA, SurpriseParams, VolEstimator, ZScore, compute_hawkes, compute_hurst,
    compute_shannon_entropy, compute_signal_stats, compute_surprise, detect_anomaly,
};

use crate::encode::{
    INPUT_RANGE, LIVE_COLUMNS, LIVE_LEGAL_COLUMNS, LiveTelemetryEncoder, NonFiniteLiveFrame,
};

/// crates.io version this integration was wired against.
///
/// `tests/kinetic_encoding.rs` asserts the lockfile resolves this exact
/// `0.4.x` package from `registry+https://github.com/rust-lang/crates.io-index`.
pub const KINETIC_SIGNALS_CRATE_VERSION: &str = "0.4.0";

/// Declared per-sensor raw span, in [`LIVE_COLUMNS`] order.
///
/// These are the `frozen_minmax` spans the shipped
/// `dataset/merged_v2/snn_model.json` records for the five legal columns, and
/// `shipped_bank_frozen_minmax_matches_live_raw_ranges` pins them to that
/// artifact. Normalising against the span the bank was frozen against is the
/// only defensible choice available: a single shared span cannot serve
/// `gpu_temp_c` (0–69 °C) and `mem_clock_mhz` (405–14801 MHz) at once, and
/// picking one would peg the clock axons at the top of [`INPUT_RANGE`]
/// forever.
///
/// This is *not* a claim that the training pipeline's arithmetic is reproduced
/// bit-for-bit — the sidecar records the spans, not the code. It is a claim
/// that the adapter normalises against the recorded spans rather than an
/// invented one.
///
/// kinetic-signals itself is domain-agnostic and does not normalise. Mapping
/// into [`INPUT_RANGE`] is the adapter's job, done only at projection time so
/// Hurst / surprise / moments see the raw units.
pub const LIVE_RAW_RANGES: [(f64, f64); LIVE_LEGAL_COLUMNS] = [
    // mem_util_pct
    (0.0, 75.0),
    // power_w
    (8.527_000_427_246_094, 302.845_001_220_703_1),
    // gpu_temp_c
    (0.0, 69.0),
    // sm_clock_mhz
    (180.0, 2910.0),
    // mem_clock_mhz
    (405.0, 14801.0),
];

/// EMA period handed to [`EMA::new`].
pub const EMA_PERIOD: usize = 8;

/// SMA window handed to [`SMA::new`].
pub const SMA_WINDOW: usize = 8;

/// Ring-buffer capacity handed to [`VolEstimator::new`].
pub const VOL_WINDOW: usize = 16;

/// Causal history retained for Hurst, entropy, moments, and Hawkes events.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KineticError {
    /// A raw sample was `NaN` or infinite; pipeline state is unchanged.
    NonFiniteSample(NonFiniteSample),
    /// The projected live frame was rejected by [`LiveTelemetryEncoder`].
    NonFiniteLiveFrame(NonFiniteLiveFrame),
}

impl fmt::Display for KineticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteSample(err) => write!(f, "{err}"),
            Self::NonFiniteLiveFrame(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for KineticError {}

impl From<NonFiniteSample> for KineticError {
    fn from(err: NonFiniteSample) -> Self {
        Self::NonFiniteSample(err)
    }
}

impl From<NonFiniteLiveFrame> for KineticError {
    fn from(err: NonFiniteLiveFrame) -> Self {
        Self::NonFiniteLiveFrame(err)
    }
}

/// One causal tick of kinetic-signals features, computed from samples `≤ t`.
///
/// Every field is a value the 0.4.0 public API actually produces — not a
/// renamed or invented statistic. These are audit data: under the identity
/// projection none of them reaches an axon, and the frame
/// [`LiveKineticFrontEnd::to_live_frame`] builds carries [`raw`](Self::raw)
/// only. See the [module docs](self#projection).
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
    /// [`compute_hawkes`] post-event intensity at the last anomaly (`μ` if none).
    ///
    /// The 0.4.0 batch API evaluates at the last event time, so this value
    /// stays put between anomalies rather than decaying on every tick.
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
    /// Anomaly event times for [`compute_hawkes`], capped like [`history`].
    event_times: VecDeque<f64>,
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
            event_times: VecDeque::with_capacity(HISTORY_WINDOW),
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
        let hawkes = compute_hawkes(self.event_times.make_contiguous(), &self.hawkes_params);

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
        // kinetic-signals 0.4.0 already zeros surprise when either sample is
        // <= 0, so this call is defined for non-positive pairs. The positivity
        // gate below is only for log-return volatility and Hawkes events.
        let surprise = compute_surprise(raw, previous, &self.surprise_params);
        if previous > 0.0 && raw > 0.0 {
            self.vol.push((raw / previous).ln().abs() as f32);
            if detect_anomaly(&surprise, &self.surprise_params) {
                let t = self.ticks as f64 * self.surprise_params.dt;
                push_capped(&mut self.event_times, t, HISTORY_WINDOW);
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

/// The live exp-025 front end: five causal pipelines, one per [`LIVE_COLUMNS`]
/// sensor, feeding [`LiveTelemetryEncoder`].
///
/// Each sensor gets its own [`KineticPipeline`], so `power_w`'s volatility
/// window never sees a `mem_clock_mhz` sample. The five advance together or not
/// at all: [`step`](Self::step) validates the whole reading before touching any
/// of them, so a single `NaN` sensor cannot leave the other four a tick ahead.
///
/// Axons 5–15 are not part of the frame this produces and are never written.
/// [`KineticPipeline`] is not [`Clone`] (0.4.0's `VolEstimator` is not), so
/// neither is this: replay means a fresh instance fed the same prefix.
///
/// # Example
///
/// ```
/// use spikenaut_snn::encode::{LIVE_LEGAL_COLUMNS, LiveTelemetryEncoder};
/// use spikenaut_snn::kinetic::LiveKineticFrontEnd;
///
/// let mut front_end = LiveKineticFrontEnd::new();
/// let mut encoder = LiveTelemetryEncoder::for_shipped_merged_v2()?;
///
/// // mem_util_pct, power_w, gpu_temp_c, sm_clock_mhz, mem_clock_mhz.
/// let reading = [42.0, 180.0, 61.0, 1_900.0, 9_500.0];
/// let (features, output) = front_end.encode_step(reading, &mut encoder)?;
///
/// assert_eq!(features.len(), LIVE_LEGAL_COLUMNS);
/// assert!(features.iter().all(|sensor| sensor.is_finite()));
/// assert!(
///     output
///         .spikes
///         .iter()
///         .all(|spike| usize::from(spike.channel) < LIVE_LEGAL_COLUMNS),
///     "axons 5-15 are unused width and stay at zero",
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LiveKineticFrontEnd {
    /// One pipeline per live sensor, keyed in [`LIVE_COLUMNS`] order.
    pipelines: [KineticPipeline; LIVE_LEGAL_COLUMNS],
}

impl LiveKineticFrontEnd {
    /// Build one [`KineticPipeline`] per live sensor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pipelines: std::array::from_fn(|_| KineticPipeline::new()),
        }
    }

    /// Readings accepted so far (the causal clock).
    ///
    /// Shared by all five pipelines, because [`step`](Self::step) advances them
    /// together or rejects the reading whole.
    #[must_use]
    pub fn ticks(&self) -> usize {
        self.pipelines[0].ticks()
    }

    /// The live sensor driving each axon, in axon order. Always
    /// [`LIVE_COLUMNS`].
    #[must_use]
    pub fn sensors(&self) -> [&'static str; LIVE_LEGAL_COLUMNS] {
        LIVE_COLUMNS
    }

    /// Incorporate one five-wide raw reading and return the per-sensor
    /// features at this tick.
    ///
    /// `reading` is in [`LIVE_COLUMNS`] order and in raw physical units —
    /// percent, watts, degrees Celsius, megahertz — not normalised. The
    /// estimators are deliberately fed raw units so Hurst, surprise and the
    /// moments describe the sensor rather than the adapter's squash.
    ///
    /// # Errors
    ///
    /// [`NonFiniteSample`] if *any* sensor is not finite. The whole reading is
    /// checked before the first pipeline is stepped, so a rejection is a no-op
    /// across all five and the next finite reading continues the same
    /// trajectories. Nothing is substituted; see the
    /// [module docs](self#missing-values).
    pub fn step(
        &mut self,
        reading: [f64; LIVE_LEGAL_COLUMNS],
    ) -> Result<[KineticFeatures; LIVE_LEGAL_COLUMNS], NonFiniteSample> {
        // Checked up front for the same reason `NonFiniteFrame::from_frame`
        // scans the whole frame: rejecting part-way through would leave the
        // earlier sensors advanced by a reading that was never accepted, and
        // the five would silently drift out of tick alignment.
        if reading.iter().any(|sample| !sample.is_finite()) {
            return Err(NonFiniteSample);
        }

        Ok(std::array::from_fn(|axon| {
            self.pipelines[axon]
                .step(reading[axon])
                .expect("the whole reading was checked finite above")
        }))
    }

    /// Project per-sensor features onto the five live axons.
    ///
    /// The identity projection: axon `i` carries sensor `i`'s own raw value,
    /// normalised through [`LIVE_RAW_RANGES`]`[i]` into [`INPUT_RANGE`]. None
    /// of the eleven kinetic features reaches an axon — choosing which one
    /// would answer the RAW / KINETIC / HYBRID question issue #14 parks. See
    /// the [module docs](self#projection).
    ///
    /// The result is five wide. Axons 5–15 are not in it, are never written,
    /// and stay at zero.
    #[must_use]
    pub fn to_live_frame(
        features: &[KineticFeatures; LIVE_LEGAL_COLUMNS],
    ) -> [f32; LIVE_LEGAL_COLUMNS] {
        std::array::from_fn(|axon| normalize_live(features[axon].raw, LIVE_RAW_RANGES[axon]))
    }

    /// [`step`](Self::step), then [`LiveTelemetryEncoder::encode_step`] on the
    /// projected live frame.
    ///
    /// # Errors
    ///
    /// [`KineticError::NonFiniteSample`] if any sensor is not finite; neither
    /// the pipelines nor the encoder are touched.
    /// [`KineticError::NonFiniteLiveFrame`] if the encoder rejects the
    /// projection. That path is defensive: normalisation clamps into
    /// [`INPUT_RANGE`], so a finite reading projects to a finite frame.
    pub fn encode_step(
        &mut self,
        reading: [f64; LIVE_LEGAL_COLUMNS],
        encoder: &mut LiveTelemetryEncoder,
    ) -> Result<([KineticFeatures; LIVE_LEGAL_COLUMNS], EncodedOutput), KineticError> {
        let features = self.step(reading)?;
        let output = encoder.encode_step(&Self::to_live_frame(&features))?;
        Ok((features, output))
    }
}

impl Default for LiveKineticFrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

/// Drop the oldest entry when `buf` reaches `cap`.
///
/// A [`VecDeque`] keeps the eviction O(1). `Vec::remove(0)` would copy the
/// whole window on every tick after fill; the kinetic-signals slice APIs then
/// see a contiguous view via [`VecDeque::make_contiguous`].
fn push_capped(buf: &mut VecDeque<f64>, value: f64, cap: usize) {
    if buf.len() == cap {
        buf.pop_front();
    }
    buf.push_back(value);
}

/// Affine map of one sensor's raw value through its [`LIVE_RAW_RANGES`] span
/// into [`INPUT_RANGE`].
///
/// Values outside the frozen span clamp rather than reject: a GPU that runs
/// one degree hotter than anything in the training episodes is a saturated
/// axon, not a fault. Only non-finite samples are refused, and they are
/// refused before this is ever reached.
///
/// A degenerate span (`max <= min`) would be a corrupt sidecar; it maps to the
/// bottom of the range rather than producing an infinity or a `NaN`. The
/// shipped spans are all strictly increasing and the test suite pins them.
fn normalize_live(value: f64, span: (f64, f64)) -> f32 {
    let (min, max) = span;
    let (lo, hi) = INPUT_RANGE;
    let width = max - min;
    if width <= 0.0 {
        return lo;
    }
    let unit = ((value - min) / width).clamp(0.0, 1.0);
    lo + (hi - lo) * unit as f32
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

    /// A plausible mid-load reading, in [`LIVE_COLUMNS`] order and raw units.
    const READING: [f64; LIVE_LEGAL_COLUMNS] = [42.0, 180.0, 61.0, 1_900.0, 9_500.0];

    #[test]
    fn the_live_projection_stays_inside_the_encoder_range() {
        let mut front_end = LiveKineticFrontEnd::new();
        let features = front_end.step(READING).expect("finite");
        let frame = LiveKineticFrontEnd::to_live_frame(&features);

        assert_eq!(frame.len(), LIVE_LEGAL_COLUMNS);
        let (lo, hi) = INPUT_RANGE;
        for (axon, &value) in frame.iter().enumerate() {
            assert!(
                value.is_finite() && (lo..=hi).contains(&value),
                "axon {axon} ({}) = {value}",
                LIVE_COLUMNS[axon],
            );
        }

        // Out-of-span readings clamp to the edges rather than escaping the
        // range or rejecting.
        let extremes = front_end
            .step([-500.0, 1e9, -0.5, 0.0, 1e12])
            .expect("finite, if physically absurd");
        assert_eq!(
            LiveKineticFrontEnd::to_live_frame(&extremes),
            [lo, hi, lo, lo, hi],
        );
    }

    #[test]
    fn each_live_sensor_lands_on_its_own_axon() {
        let mut front_end = LiveKineticFrontEnd::new();
        assert_eq!(front_end.sensors(), LIVE_COLUMNS);

        // Each sensor at the top of its own frozen span, one at a time: only
        // that axon may saturate.
        for (axon, (_, max)) in LIVE_RAW_RANGES.into_iter().enumerate() {
            let mut reading = LIVE_RAW_RANGES.map(|(min, _)| min);
            reading[axon] = max;
            let features = front_end.step(reading).expect("finite");
            let frame = LiveKineticFrontEnd::to_live_frame(&features);
            for (slot, &value) in frame.iter().enumerate() {
                let expected = if slot == axon {
                    INPUT_RANGE.1
                } else {
                    INPUT_RANGE.0
                };
                assert_eq!(
                    value, expected,
                    "axon {slot} ({}) while {} is saturated",
                    LIVE_COLUMNS[slot], LIVE_COLUMNS[axon],
                );
            }
            assert_eq!(features[axon].raw, max);
        }
    }

    #[test]
    fn a_degenerate_span_maps_to_the_floor_rather_than_a_nan() {
        assert_eq!(normalize_live(7.0, (5.0, 5.0)), INPUT_RANGE.0);
        assert_eq!(normalize_live(7.0, (9.0, 1.0)), INPUT_RANGE.0);
        assert!(LIVE_RAW_RANGES.iter().all(|&(min, max)| min < max));
    }

    #[test]
    fn a_single_non_finite_sensor_rejects_the_whole_reading() {
        let mut front_end = LiveKineticFrontEnd::new();
        front_end.step(READING).expect("finite");
        assert_eq!(front_end.ticks(), 1);

        let mut faulty = READING;
        faulty[3] = f64::NAN;
        assert_eq!(front_end.step(faulty), Err(NonFiniteSample));
        faulty[3] = f64::NEG_INFINITY;
        assert_eq!(front_end.step(faulty), Err(NonFiniteSample));

        // Every pipeline, not just the offending one, stayed put.
        assert_eq!(front_end.ticks(), 1);
        for (axon, pipeline) in front_end.pipelines.iter().enumerate() {
            assert_eq!(pipeline.ticks(), 1, "axon {axon} ({})", LIVE_COLUMNS[axon]);
        }
        assert_eq!(front_end.step(READING).expect("recovered").len(), 5);
        assert_eq!(front_end.ticks(), 2);
    }

    #[test]
    fn the_live_front_end_encodes_only_the_five_live_axons() {
        let mut front_end = LiveKineticFrontEnd::new();
        let mut encoder =
            LiveTelemetryEncoder::for_shipped_merged_v2().expect("the legitimate live pairing");

        let mut fired = [0_usize; crate::encode::CHANNEL_COUNT];
        for tick in 0..32 {
            let (features, output) = front_end
                .encode_step(READING, &mut encoder)
                .unwrap_or_else(|err| panic!("tick {tick}: {err}"));
            assert_eq!(features.len(), LIVE_LEGAL_COLUMNS);
            assert!(features.iter().all(KineticFeatures::is_finite));
            for spike in &output.spikes {
                fired[usize::from(spike.channel)] += 1;
            }
        }

        assert!(
            fired[..LIVE_LEGAL_COLUMNS].iter().all(|&count| count > 0),
            "every live axon fires: {fired:?}",
        );
        assert!(
            fired[LIVE_LEGAL_COLUMNS..].iter().all(|&count| count == 0),
            "axons 5-15 are unused width and stay at zero: {fired:?}",
        );
    }

    #[test]
    fn a_rejected_reading_does_not_touch_the_encoder() {
        let mut front_end = LiveKineticFrontEnd::new();
        let mut encoder = LiveTelemetryEncoder::new().expect("live constants");
        let cold = encoder.clone();

        let mut faulty = READING;
        faulty[0] = f64::INFINITY;
        assert_eq!(
            front_end.encode_step(faulty, &mut encoder),
            Err(KineticError::NonFiniteSample(NonFiniteSample)),
        );
        assert_eq!(encoder, cold, "the encoder never saw the rejected reading");
        assert_eq!(front_end.ticks(), 0);
    }

    #[test]
    fn the_live_error_display_names_its_cause() {
        assert_eq!(
            KineticError::from(NonFiniteSample).to_string(),
            "non-finite raw telemetry sample",
        );

        let rejected = LiveTelemetryEncoder::new()
            .expect("live constants")
            .encode_step(&[0.0, f32::NAN, 0.0, 0.0, 0.0])
            .unwrap_err();
        assert_eq!(
            KineticError::from(rejected).to_string(),
            "non-finite live telemetry on axon 1 (power_w)",
        );
    }

    #[test]
    fn shipped_bank_frozen_minmax_matches_live_raw_ranges() {
        let parsed = crate::json::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/dataset/merged_v2/snn_model.json"
        )))
        .expect("shipped snn_model.json");
        let frozen = parsed.get("frozen_minmax").expect("frozen_minmax object");

        for (axon, sensor) in LIVE_COLUMNS.into_iter().enumerate() {
            let span = frozen
                .get(sensor)
                .and_then(crate::json::Json::as_array)
                .unwrap_or_else(|| panic!("frozen_minmax has a span for {sensor}"));
            let recorded = (
                span[0].as_f64().expect("min"),
                span[1].as_f64().expect("max"),
            );
            assert_eq!(
                recorded, LIVE_RAW_RANGES[axon],
                "axon {axon} ({sensor}) must normalise against the frozen span",
            );
        }
    }

    #[test]
    fn feature_names_match_the_array() {
        assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
        let features = KineticPipeline::new().step(1.0).expect("finite");
        assert_eq!(features.as_array().len(), FEATURE_NAMES.len());
    }

    #[test]
    fn non_positive_samples_yield_finite_zero_surprise() {
        let mut pipeline = KineticPipeline::new();
        assert_eq!(pipeline.step(100.0).expect("finite").surprise, 0.0);
        let zeroed = pipeline.step(0.0).expect("zero is a finite sample");
        assert!(zeroed.is_finite());
        assert_eq!(zeroed.surprise, 0.0);
        let negative = pipeline.step(-8.0).expect("negative is a finite sample");
        assert!(negative.is_finite());
        assert_eq!(negative.surprise, 0.0);
        assert!(pipeline.step(120.0).expect("recovered").is_finite());
    }

    #[test]
    fn hawkes_event_history_stays_capped() {
        let mut pipeline = KineticPipeline::new();
        // Alternate a calm level and a jump so nearly every transition is an
        // anomaly and would otherwise grow `event_times` without bound.
        for tick in 0..(HISTORY_WINDOW * 4) {
            let raw = if tick % 2 == 0 { 100.0 } else { 220.0 };
            let features = pipeline.step(raw).expect("finite");
            assert!(features.is_finite(), "tick {tick}");
        }
        assert!(pipeline.event_times.len() <= HISTORY_WINDOW);
        assert_eq!(pipeline.event_times.len(), HISTORY_WINDOW);
    }
}

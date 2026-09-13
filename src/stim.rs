// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shipped bank's front end: raw GPU sensors → analog `stim`.
//!
//! [`crate::graph`] describes what the network *is*. [`crate::encode`] names
//! what it eats — [`LIVE_COLUMNS`] on axons 0–4, axons 5–15 unwritten — and
//! ships a rate encoder that gets those columns right. This module is the part
//! neither of them was: the adapter that produces the vector
//! `dataset/merged_v2` was actually trained and evaluated on.
//!
//! That vector is **analog current**, not a spike train. The reference stepper
//! in `tools/hamming_lif.py` is
//!
//! ```text
//! v[t+1] = decay * v[t] + W @ stim        (decay is a KEEP factor)
//! ```
//!
//! with `stim` the 16-wide encoder output and Poisson pre-spikes explicitly
//! unused when `learn=false` — `tools/HAMMING_PROTOCOL.md` records the exp-024
//! condition as "analog current, Poisson unused". [`LiveStimAdapter`] builds
//! that `stim`.
//!
//! # Three constructors, one `Ok`
//!
//! | Constructor | Columns | Modality | Result |
//! | --- | --- | --- | --- |
//! | [`TelemetryEncoder::for_shipped_merged_v2`] | wrong (coin, 16-wide) | spikes | [`Err(LiveMapMismatch)`][LiveMapMismatch] |
//! | [`LiveTelemetryEncoder::for_shipped_merged_v2`] | right ([`LIVE_COLUMNS`]) | wrong (spikes) | [`Err(SpikeModalityMismatch)`][SpikeModalityMismatch] |
//! | [`LiveStimAdapter::for_shipped_merged_v2`] | right ([`LIVE_COLUMNS`]) | right (analog) | `Ok` — [`Infallible`] |
//!
//! The first two are refusals kept deliberately distinct, because they are
//! different mistakes. This one has nothing to refuse, and its error type says
//! so rather than its documentation.
//!
//! [`TelemetryEncoder::for_shipped_merged_v2`]: crate::encode::TelemetryEncoder::for_shipped_merged_v2
//! [`LiveTelemetryEncoder::for_shipped_merged_v2`]: LiveTelemetryEncoder::for_shipped_merged_v2
//!
//! # Normalisation
//!
//! Per sensor, through the `frozen_minmax` spans the shipped sidecar records:
//! `(raw - min) / (max - min)`, clamped to `[0, 1]`. Those spans are
//! [`LIVE_RAW_RANGES`], pinned to `dataset/merged_v2/snn_model.json` by
//! `shipped_bank_frozen_minmax_matches_live_raw_ranges`; this module reuses
//! them rather than restating them, and reuses `kinetic`'s own affine map
//! rather than restating that either.
//!
//! Two details are load-bearing for parity with the reference, and both are
//! easy to get silently wrong:
//!
//! - **Clamping, not rejecting.** A GPU hotter than anything in the training
//!   episodes is a saturated axon, not a fault. `tools/hamming_encode.py`
//!   clamps; so does this.
//! - **The binary32 snap.** The bank's arithmetic is `f32` throughout, and the
//!   reference snaps each raw sample onto the binary32 grid *before* the
//!   affine map (`hamming_const.f32`). Skipping that snap lands the result on
//!   a different `f32` for between 27% and 36% of uniform in-span readings,
//!   depending on the sensor — a one-ulp difference, about 3e-8 at mid-span,
//!   which any tolerance looser than that absorbs silently.
//!   [`LiveStimAdapter::stim`] snaps.
//!
//! [`crate::kinetic`]'s projection deliberately does not snap: it feeds a rate
//! encoder rather than the bank, and is not pinned to this reference.
//!
//! # Unused axons
//!
//! The sidecar records `unused_axons: "5:15"`. [`UNUSED_AXONS`] is that
//! contract, and this adapter never writes into it — the vector starts at
//! [`NO_STIMULUS`] and only axons 0–4 are ever assigned. Those eleven slots are
//! unused *width*, not fake channels: they are exactly `0.0`, so they
//! contribute exactly nothing to `W @ stim`. `tools/hamming_core.py` flags any
//! non-zero value there as an `UNUSED-AXON LEAK`; the Rust side makes the leak
//! unrepresentable instead of detectable.
//!
//! On this bank they are inert twice over, for two independent reasons. The
//! `snn_model.json` weights on columns 5–15 are training residue far below the
//! Q8.8 grid — none is exactly zero, and the largest is about `8e-15` against a
//! grid step of `1/256` — so the decode snaps every one of them to zero, and
//! the graph's `Linear` node could not be moved by those axons
//! even if something did write to them. That is pinned by
//! `the_graph_cannot_be_moved_by_an_unused_axon`, and it is also why the graph
//! round trip cannot double as the leak check: with a zero weight, any value
//! multiplies to zero.
//!
//! Zero is meaningful here in a way it is not on the rate path. A normalised
//! `0.0` on [`LiveTelemetryEncoder`] still fires at
//! [`BASE_RATE_HZ`][crate::encode::BASE_RATE_HZ]; an idle sensor never reads as
//! idle, so *no stimulus* has no encoding at all. On this path it is the
//! literal `0.0` the contract is written in terms of — the same value a missing
//! sensor and a reading at or below its frozen minimum both produce.
//!
//! # Parity
//!
//! `tools/live_stim_parity.py` runs `tools/hamming_encode.py` over
//! `tools/fixtures/live_stim/` and pins the result; `tests/live_stim.rs` reads
//! the same files and asserts this adapter reproduces the pin exactly — bit
//! pattern for bit pattern, not within a tolerance.
//!
//! What that covers: both ends of every frozen span, mid-load readings,
//! out-of-span clamping, missing and null sensors, the binary32 snap on every
//! sensor, negative zero, subnormals, and the refusal boundary — `NaN`, the
//! infinities, and the largest `f64` that still fits binary32 against the
//! smallest that does not. The readings JSON has no literal for travel as raw
//! bits in the pin's `bit_exact` section, because otherwise the entire refusal
//! contract would have no cross-language coverage at all.
//!
//! What it does not cover: a degenerate span. `normalize_live` short-circuits
//! on `max <= min` while the reference only short-circuits on `max == min` and
//! would otherwise divide by a negative width — span `(9.0, 1.0)` and value
//! `7.0` give `0.0` here and `0.25` there. Unreachable on this bank (the
//! shipped spans are strictly increasing, pinned on both sides against the
//! sidecar), and recorded rather than fixed because the fix would be inventing
//! an answer for a corrupt artifact.
//!
//! # Scope
//!
//! This is the input side only. It is **not** a claim that this crate runs the
//! shipped bank: `tools/hamming_lif.py` is still the only place in this
//! repository that steps *these weights* through a membrane, and
//! [`crate::model::Neuron::membrane_potential`] is decoded and never advanced.
//! ([`crate::neuromod_host`] does step a LIF, but a default published
//! `neuromod::LifNeuron` that never sees `merged_v2`'s weights — a different
//! claim.) What the adapter produces is the vector the graph's
//! `Input → Linear` edge consumes; see
//! [Feeding the graph](#feeding-the-graph).

use std::convert::Infallible;

use crate::encode::{
    CHANNEL_COUNT, INPUT_RANGE, LIVE_COLUMNS, LIVE_LEGAL_COLUMNS, NonFiniteLiveFrame,
};
#[cfg(doc)]
use crate::encode::{LiveMapMismatch, LiveTelemetryEncoder, SpikeModalityMismatch};
use crate::kinetic::{LIVE_RAW_RANGES, normalize_live};

/// What an axon carrying no stimulus holds: exactly `0.0`.
///
/// Three things encode to it, and the bank cannot tell them apart — which is
/// the point, because the contract is written in terms of the value, not its
/// cause:
///
/// - an axon in [`UNUSED_AXONS`], which this adapter never writes;
/// - a sensor that is missing or null in the reading;
/// - a reading at or below its [`LIVE_RAW_RANGES`] minimum.
///
/// It is `0.0` and not [`INPUT_RANGE`]`.0` by intent: the reference encoder
/// returns a literal zero for a missing column rather than the bottom of the
/// output range, and `the_reference_range_is_the_unit_interval` holds the two
/// equal.
pub const NO_STIMULUS: f32 = 0.0;

/// The axons this bank does not use, as a range: `5..16`.
///
/// The shipped `dataset/merged_v2/snn_model.json` records this as
/// `unused_axons: "5:15"`, and
/// `the_sidecar_unused_axon_contract_matches_this_range` pins the two together.
/// Every axon in it is [`NO_STIMULUS`] on every vector this module produces.
pub const UNUSED_AXONS: std::ops::Range<usize> = LIVE_LEGAL_COLUMNS..CHANNEL_COUNT;

// The reference normalises onto the unit interval, and `normalize_live` maps
// onto `INPUT_RANGE`. They are the same interval today; if that ever stops
// being true, this module needs its own affine map rather than a silent
// re-scale of the bank's inputs.
const _: () = assert!(
    INPUT_RANGE.0 == 0.0 && INPUT_RANGE.1 == 1.0,
    "the analog stim contract normalises onto the unit interval",
);

/// Raw live sensors → the 16-wide analog `stim` vector `merged_v2` consumes.
///
/// Stateless: there is nothing to accumulate, because analog current is not a
/// spike train. One reading in, one vector out, no clock. That is the whole
/// difference from [`LiveTelemetryEncoder`], and the reason this type has no
/// `reset` and no `dt_seconds`.
///
/// # Example
///
/// ```
/// use spikenaut_snn::encode::{CHANNEL_COUNT, LIVE_LEGAL_COLUMNS};
/// use spikenaut_snn::stim::{LiveStimAdapter, NO_STIMULUS, UNUSED_AXONS};
///
/// // The one constructor for this bank that returns `Ok`.
/// let adapter = LiveStimAdapter::for_shipped_merged_v2()
///     .expect("the analog adapter has nothing to refuse");
///
/// // mem_util_pct, power_w, gpu_temp_c, sm_clock_mhz, mem_clock_mhz, raw.
/// let stim = adapter.stim([42.0, 180.0, 61.0, 1_900.0, 9_500.0])?;
///
/// assert_eq!(stim.len(), CHANNEL_COUNT);
/// assert!(stim[..LIVE_LEGAL_COLUMNS].iter().all(|&v| (0.0..=1.0).contains(&v)));
///
/// // The `unused_axons: "5:15"` contract: exactly zero, not merely small.
/// assert!(stim[UNUSED_AXONS].iter().all(|&v| v == NO_STIMULUS));
///
/// // A sensor below its frozen minimum clamps; it is not a rejection.
/// let cold = adapter.stim([0.0, 0.0, 0.0, 0.0, 0.0])?;
/// assert_eq!(cold, [NO_STIMULUS; CHANNEL_COUNT]);
///
/// // A non-finite sensor rejects the whole reading, by name.
/// let rejected = adapter.stim([42.0, f64::NAN, 61.0, 1_900.0, 9_500.0]).unwrap_err();
/// assert_eq!(rejected.sensors().collect::<Vec<_>>(), ["power_w"]);
/// # Ok::<(), spikenaut_snn::NonFiniteLiveFrame>(())
/// ```
///
/// # Feeding the graph
///
/// `stim` is consumed by the `Input → Linear` edge of the graph
/// [`crate::graph::build_lif_graph`] builds: the `Linear` node carries the
/// learned 16×16 matrix and computes `I = W @ stim`, which is the input term
/// of the reference stepper. The eleven unused axons are along for the ride
/// and contribute exactly nothing.
///
/// ```
/// use nir_rs::NirNode;
/// use nir_rs::types::TensorData;
/// use spikenaut_snn::encode::{CHANNEL_COUNT, LIVE_LEGAL_COLUMNS};
/// use spikenaut_snn::graph::{self, LINEAR_NODE};
/// use spikenaut_snn::model::{NEURON_COUNT, SnnModel};
/// use spikenaut_snn::stim::LiveStimAdapter;
///
/// let adapter = LiveStimAdapter::for_shipped_merged_v2()
///     .expect("the analog adapter has nothing to refuse");
/// let stim = adapter.stim([42.0, 180.0, 61.0, 1_900.0, 9_500.0])?;
///
/// let model = SnnModel::load_default()?;
/// let graph = graph::build_lif_graph(&model)?;
/// let Some(NirNode::Linear(linear)) = graph.get(LINEAR_NODE) else {
///     panic!("the graph carries the learned weights on its Linear node");
/// };
/// let TensorData::F64(weight) = linear.weight.data() else {
///     panic!("the weight tensor is f64");
/// };
///
/// for unit in 0..NEURON_COUNT {
///     // `Linear`: I = W @ stim, over all 16 axons.
///     let current: f64 = (0..CHANNEL_COUNT)
///         .map(|axon| weight[unit * CHANNEL_COUNT + axon] * f64::from(stim[axon]))
///         .sum();
///
///     // The same sum over the five live axons alone, to the bit: axons 5-15
///     // are exactly zero, so they cannot shift the current they multiply into.
///     let live_only: f64 = (0..LIVE_LEGAL_COLUMNS)
///         .map(|axon| weight[unit * CHANNEL_COUNT + axon] * f64::from(stim[axon]))
///         .sum();
///     assert_eq!(current, live_only, "unit {unit}");
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Advancing the membrane from there is not this crate's job; see the
/// [module docs](self#scope).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LiveStimAdapter;

impl LiveStimAdapter {
    /// The front end for the shipped `merged_v2` bank.
    ///
    /// Right columns ([`LIVE_COLUMNS`], axons 0–4, axons 5–15 unwritten) and
    /// right modality (analog current). The two rate-encoder constructors of
    /// the same name refuse; see the
    /// [three-constructor table](self#three-constructors-one-ok).
    ///
    /// # Errors
    ///
    /// Never — the error type is [`Infallible`], so "this pairing is correct"
    /// is a fact the compiler carries rather than a sentence in prose. The
    /// `Result` is kept so the three constructors can be compared side by
    /// side, and so a caller that switched from one of the refusing encoders
    /// does not have to change shape to find out it now works.
    pub fn for_shipped_merged_v2() -> Result<Self, Infallible> {
        Ok(Self)
    }

    /// Build the adapter.
    ///
    /// Identical to [`for_shipped_merged_v2`](Self::for_shipped_merged_v2)
    /// without the `Result`: there is only one bank and one set of frozen
    /// spans, so there is no second configuration this could produce.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The sensor each live axon carries, in axon order.
    #[must_use]
    pub fn sensors(&self) -> [&'static str; LIVE_LEGAL_COLUMNS] {
        LIVE_COLUMNS
    }

    /// The analog `stim` vector for one raw live reading.
    ///
    /// `reading` is in [`LIVE_COLUMNS`] order and in each sensor's own raw
    /// units — percent, watts, degrees Celsius, megahertz — *not* normalised.
    /// Normalising is exactly what this does, through [`LIVE_RAW_RANGES`]; see
    /// the [module docs](self#normalisation).
    ///
    /// Axons 5–15 come back as [`NO_STIMULUS`], and so does any sensor at or
    /// below its frozen minimum.
    ///
    /// # Errors
    ///
    /// Returns [`NonFiniteLiveFrame`], naming every offending sensor, if any
    /// reading is not representable as a finite binary32 — a `NaN`, an
    /// infinity, or a finite `f64` too large for the `f32` grid the bank's
    /// arithmetic runs on. The whole reading is checked before any axon is
    /// written, so a rejected reading yields no partial vector.
    ///
    /// The reference refuses exactly the same values, in two places rather
    /// than one: `hamming_encode._live_number` rejects the non-finite ones and
    /// `hamming_const.f32` rejects what binary32 cannot hold. Every case is
    /// pinned in the parity fixture's `bit_exact` section, which carries them
    /// as raw bits because JSON has no literal for any of them.
    ///
    /// Nothing is substituted for a rejected sensor, for the reason the
    /// [`crate::encode`] module docs give: on this path the honest-looking
    /// substitute is `0.0`, and `0.0` already means something specific.
    pub fn stim(
        &self,
        reading: [f64; LIVE_LEGAL_COLUMNS],
    ) -> Result<[f32; CHANNEL_COUNT], NonFiniteLiveFrame> {
        self.stim_optional(reading.map(Some))
    }

    /// [`stim`](Self::stim) for a reading whose sensors may be absent.
    ///
    /// A `None` encodes as [`NO_STIMULUS`] — never an imputed neighbour, never
    /// a refit span, never the last good value. That is the reference rule
    /// (`tools/hamming_encode.py`: "`None` / missing is **0**"), and it is what
    /// a v3 `state_telemetry` row with a null or absent column already means.
    ///
    /// Masking a *dropout sentinel* is a different problem and this does not
    /// solve it: `gpu_temp_c == 0` is a finite number that passes straight
    /// through as a legitimate reading at the bottom of its span. That rule
    /// belongs to the state contract in issue #20.
    ///
    /// # Errors
    ///
    /// [`NonFiniteLiveFrame`] on the same terms as [`stim`](Self::stim). A
    /// `None` is never an offender — absent is not the same as invalid.
    pub fn stim_optional(
        &self,
        reading: [Option<f64>; LIVE_LEGAL_COLUMNS],
    ) -> Result<[f32; CHANNEL_COUNT], NonFiniteLiveFrame> {
        // Snap onto the binary32 grid first, then check: the bank's arithmetic
        // is f32, so a finite f64 the f32 grid cannot hold is not a reading it
        // can take, and it arrives here as an infinity. An absent sensor stands
        // in as `NO_STIMULUS` for the scan only -- it is finite, and it is not
        // an offender.
        let snapped: [f32; LIVE_LEGAL_COLUMNS] =
            std::array::from_fn(|axon| reading[axon].map_or(NO_STIMULUS, |raw| raw as f32));
        if let Some(rejected) = NonFiniteLiveFrame::from_frame(&snapped) {
            return Err(rejected);
        }

        // Only the live axons are ever assigned, so the `unused_axons: "5:15"`
        // contract holds by construction rather than by inspection.
        let mut stim = [NO_STIMULUS; CHANNEL_COUNT];
        for (axon, &span) in LIVE_RAW_RANGES.iter().enumerate() {
            if reading[axon].is_some() {
                stim[axon] = normalize_live(f64::from(snapped[axon]), span);
            }
        }
        Ok(stim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plausible mid-load reading, in raw sensor units.
    const READING: [f64; LIVE_LEGAL_COLUMNS] = [42.0, 180.0, 61.0, 1_900.0, 9_500.0];

    fn adapter() -> LiveStimAdapter {
        LiveStimAdapter::for_shipped_merged_v2().expect("the analog pairing is the correct one")
    }

    #[test]
    fn the_analog_constructor_is_the_one_that_succeeds() {
        assert!(LiveStimAdapter::for_shipped_merged_v2().is_ok());
        assert_eq!(adapter(), LiveStimAdapter::new());
        assert_eq!(LiveStimAdapter::new(), LiveStimAdapter);
    }

    #[test]
    fn unused_axons_are_exactly_zero_not_merely_small() {
        let stim = adapter().stim(READING).expect("finite");
        assert_eq!(UNUSED_AXONS, LIVE_LEGAL_COLUMNS..CHANNEL_COUNT);
        for axon in UNUSED_AXONS {
            // Bit equality, not `== 0.0`: `-0.0` compares equal to `0.0` and
            // would pass a value comparison while being a different f32.
            assert_eq!(
                stim[axon].to_bits(),
                0.0_f32.to_bits(),
                "axon {axon} is unused width and must be a positive zero",
            );
        }
        // The live axons are not all zero on this reading, so the assertion
        // above is not vacuously true of the whole vector.
        assert!(stim[..LIVE_LEGAL_COLUMNS].iter().any(|&v| v != NO_STIMULUS));
    }

    #[test]
    fn each_sensor_normalises_through_its_own_frozen_span() {
        let adapter = adapter();
        assert_eq!(adapter.sensors(), LIVE_COLUMNS);

        // One sensor at the top of its own span, the rest at the bottom of
        // theirs: only that axon may saturate.
        for (axon, &(_, max)) in LIVE_RAW_RANGES.iter().enumerate() {
            let mut reading = LIVE_RAW_RANGES.map(|(min, _)| min);
            reading[axon] = max;
            let stim = adapter.stim(reading).expect("finite");
            for (slot, &value) in stim.iter().enumerate() {
                let expected = if slot == axon { 1.0 } else { NO_STIMULUS };
                assert_eq!(
                    value, expected,
                    "axon {slot} while {} is saturated",
                    LIVE_COLUMNS[axon],
                );
            }
        }
    }

    #[test]
    fn out_of_span_readings_clamp_rather_than_reject() {
        let stim = adapter()
            .stim([-12.5, 1_000.0, -0.5, 9_000.0, 0.0])
            .expect("out of span is not a rejection");
        assert_eq!(stim[0], NO_STIMULUS);
        assert_eq!(stim[1], 1.0);
        assert_eq!(stim[2], NO_STIMULUS);
        assert_eq!(stim[3], 1.0);
        assert_eq!(stim[4], NO_STIMULUS);
    }

    #[test]
    fn a_missing_sensor_encodes_as_no_stimulus() {
        let mut reading = READING.map(Some);
        reading[1] = None;
        let stim = adapter()
            .stim_optional(reading)
            .expect("absent is not invalid");
        assert_eq!(stim[1], NO_STIMULUS);

        // Absent is not a rejection, and it does not disturb its neighbours.
        let present = adapter().stim(READING).expect("finite");
        for axon in (0..LIVE_LEGAL_COLUMNS).filter(|&axon| axon != 1) {
            assert_eq!(stim[axon], present[axon], "axon {axon}");
        }
    }

    #[test]
    fn a_non_finite_sensor_rejects_the_whole_reading_by_name() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut reading = READING;
            reading[3] = bad;
            let rejected = adapter().stim(reading).unwrap_err();
            assert_eq!(rejected.channels().collect::<Vec<_>>(), [3]);
            assert_eq!(rejected.sensors().collect::<Vec<_>>(), ["sm_clock_mhz"]);
        }
    }

    /// A finite `f64` the binary32 grid cannot hold is not a legal reading.
    ///
    /// The reference refuses it too -- `hamming_const.f32` raises rather than
    /// returning an infinity -- and the alternative here would be an axon
    /// silently pinned at the top of its span. The boundary itself is pinned
    /// across both languages by `the_adapter_agrees_on_readings_json_cannot_carry`,
    /// which carries the largest `f64` that still snaps to `f32::MAX` and the
    /// smallest one that does not.
    #[test]
    fn a_reading_outside_binary32_is_rejected_rather_than_saturated() {
        let mut reading = READING;
        reading[0] = 1e300;
        let rejected = adapter().stim(reading).unwrap_err();
        assert_eq!(rejected.sensors().collect::<Vec<_>>(), ["mem_util_pct"]);

        // Not merely "bigger than the span": that clamps, and is fine.
        assert_eq!(
            adapter()
                .stim([1e30, 180.0, 61.0, 1_900.0, 9_500.0])
                .expect("finite")[0],
            1.0
        );
    }

    #[test]
    fn every_offending_sensor_is_named_not_just_the_first() {
        let mut reading = READING;
        reading[0] = f64::NAN;
        reading[4] = f64::INFINITY;
        let rejected = adapter().stim(reading).unwrap_err();
        assert_eq!(rejected.count(), 2);
        assert_eq!(
            rejected.sensors().collect::<Vec<_>>(),
            ["mem_util_pct", "mem_clock_mhz"],
        );
    }

    /// The reference snaps every raw sample onto the binary32 grid before the
    /// affine map. Skipping that snap is a one-ulp error on a third of
    /// ordinary readings -- small enough to survive a tolerance, large enough
    /// to be a different encoder.
    #[test]
    fn raw_samples_are_snapped_to_binary32_before_normalising() {
        // mem_util_pct, span 0..75. The f64 and its binary32 snap normalise to
        // two different f32 values.
        let raw = 23.540_267_130_810_89_f64;
        let (min, max) = LIVE_RAW_RANGES[0];
        let unsnapped = (((raw - min) / (max - min)).clamp(0.0, 1.0)) as f32;
        let snapped = adapter()
            .stim([raw, 180.0, 61.0, 1_900.0, 9_500.0])
            .expect("finite")[0];

        assert_ne!(
            snapped, unsnapped,
            "this fixture value exists to make the snap observable",
        );
        assert_eq!(snapped, 0.313_870_25_f32);
    }

    #[test]
    fn the_reference_range_is_the_unit_interval() {
        // `normalize_live` maps onto INPUT_RANGE; the reference encoder maps
        // onto [0, 1]. The const assertion in this module holds them equal --
        // this is the runtime spelling of it, for a reader grepping tests.
        assert_eq!(INPUT_RANGE, (NO_STIMULUS, 1.0));
    }

    #[test]
    fn the_sidecar_unused_axon_contract_matches_this_range() {
        let parsed = crate::json::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/dataset/merged_v2/snn_model.json"
        )))
        .expect("shipped snn_model.json");
        let recorded = parsed
            .get("unused_axons")
            .and_then(crate::json::Json::as_str)
            .expect("the sidecar records unused_axons");

        // `"5:15"` is inclusive on both ends, Julia-style; UNUSED_AXONS is the
        // half-open Rust spelling of the same eleven axons.
        assert_eq!(
            recorded,
            format!("{}:{}", UNUSED_AXONS.start, UNUSED_AXONS.end - 1),
        );
        assert_eq!(UNUSED_AXONS.len(), CHANNEL_COUNT - LIVE_LEGAL_COLUMNS);
    }

    #[test]
    fn the_legal_columns_are_the_sidecar_columns() {
        let parsed = crate::json::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/dataset/merged_v2/snn_model.json"
        )))
        .expect("shipped snn_model.json");
        let recorded = parsed
            .get("legal_columns")
            .and_then(crate::json::Json::as_array)
            .expect("the sidecar records legal_columns");

        assert_eq!(recorded.len(), LIVE_LEGAL_COLUMNS);
        for (axon, column) in recorded.iter().enumerate() {
            assert_eq!(column.as_str(), Some(LIVE_COLUMNS[axon]), "axon {axon}");
        }
    }
}

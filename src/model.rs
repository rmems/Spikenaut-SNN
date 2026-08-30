// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shipped `merged_v2` model as it is stored on disk.
//!
//! [`SnnModel`] is a direct, validated decoding of
//! `dataset/merged_v2/snn_model.json`: 16 leaky integrate-and-fire neurons,
//! each with a decay rate, a firing threshold, a learned input weight row, and
//! the simulator state (membrane potential, last-spike flag) captured when the
//! model was saved.
//!
//! Every number is snapped onto the Q8.8 grid as it is decoded (see
//! [`quantize_q8_8`]) and every documented numeric invariant is enforced there,
//! so a decoded [`SnnModel`] is exactly the parameter set the `.mem` artifacts
//! hold and cannot carry a decay rate outside `(0, 1)`.
//!
//! [`SnnModel::load_default`] is that artifact, not a post-exp-009 legal-encoder
//! retrain and not the session-holdout 5-ch v3 encoder. See
//! [`MERGED_V2_PROVENANCE`].
//!
//! This module does no NIR mapping. See [`crate::graph`] for that.

use std::fmt;
use std::path::{Path, PathBuf};

use nir_rs::NirError;
use nir_rs::types::Tensor;

use crate::json::{self, Json, JsonError};

/// Neuron count of the shipped `merged_v2` model (`n_neurons` in `config.json`).
pub const NEURON_COUNT: usize = 16;

/// Simulation clock of the shipped model, in hertz (`clock_hz` in `config.json`).
pub const CLOCK_HZ: f64 = 1000.0;

/// Duration of one simulation step, in seconds.
pub const TIMESTEP_SECONDS: f64 = 1.0 / CLOCK_HZ;

/// Path of the shipped model, relative to the repository root.
pub const MODEL_RELATIVE_PATH: &str = "dataset/merged_v2/snn_model.json";

/// Scale of the Q8.8 fixed-point grid: one integer code is `1 / 256`.
///
/// `weight_format` in `config.json`, and the encoding of every `.mem` artifact
/// beside the model.
pub const Q8_8_SCALE: f64 = 256.0;

/// Most negative Q8.8 value, hex `8000`.
pub const Q8_8_MIN: f64 = -128.0;

/// Most positive Q8.8 value, hex `7FFF`.
pub const Q8_8_MAX: f64 = 127.996_093_75;

/// Snap `value` onto the nearest value the Q8.8 grid can represent.
///
/// `snn_model.json` prints Q8.8 codes as seven-significant-digit decimals, so
/// the text is truncated: `0.7539062` is the printed form of
/// `0x00C1 = 193/256 = 0.75390625`, and neuron 1's decay `0.808594` is
/// `0x00CF = 207/256 = 0.80859375`. 139 of the model's 288 stored numbers are
/// short of their exact value this way.
///
/// Reading the decimals verbatim would leave the graph up to half an LSB away
/// from `parameters_decay.mem`, `parameters.mem` and `parameters_weights.mem`,
/// which is what a hardware/software equivalence claim cannot afford. Decoding
/// snaps every numeric field back onto the grid, so the graph carries exactly
/// the values the FPGA holds.
///
/// Q8.8 values are dyadic, hence exact in binary floating point: the snapped
/// value round-trips through `f64` without further error.
///
/// # This rounds; it does not validate
///
/// The result is a Q8.8 code only when `value` is inside
/// `[Q8_8_MIN, Q8_8_MAX]`. Outside it, the return is the nearest multiple of
/// `1/256`, which is not a code the format can hold: `quantize_q8_8(128.0)` is
/// `128.0`, one past the largest code, and a large enough input overflows the
/// intermediate product to infinity. `NaN` stays `NaN`.
///
/// **It deliberately does not saturate.** Clamping 200.0 down to
/// [`Q8_8_MAX`] would turn a wrong number into a plausible weight and lose the
/// evidence that anything was wrong — the silent clamp-to-range failure that
/// `tools/verify_q88.py` exists to catch. Letting the out-of-range value
/// through unchanged is what lets the validators see it and refuse the model.
///
/// So this is the rounding half of the pair, and the caller owns the other
/// half. Use [`is_q8_8`] to ask whether a value is representable before or
/// after snapping; the decode path snaps first and range-checks the result,
/// which is how a boundary code printed at seven digits still loads.
#[must_use]
pub fn quantize_q8_8(value: f64) -> f64 {
    (value * Q8_8_SCALE).round() / Q8_8_SCALE
}

/// Whether `value` is exactly representable as a Q8.8 code.
///
/// True when it is finite, within `[Q8_8_MIN, Q8_8_MAX]`, and lands on the
/// `1/256` grid. This is the public counterpart to [`quantize_q8_8`], which
/// rounds without judging: `is_q8_8(quantize_q8_8(v))` is the question worth
/// asking of an arbitrary `v`, and answers false exactly when `v` is out of
/// range rather than merely off-grid.
#[must_use]
pub fn is_q8_8(value: f64) -> bool {
    value.is_finite()
        && (Q8_8_MIN..=Q8_8_MAX).contains(&value)
        && (value * Q8_8_SCALE).fract() == 0.0
}

/// Provenance stamp for the shipped `merged_v2` artifact loaded by this crate.
///
/// This is the repository file at [`MODEL_RELATIVE_PATH`], a 16-neuron LIF
/// population with known training-path defects. It is not a post-exp-009
/// legal-encoder retrain and not the session-holdout 5-ch v3 encoder.
pub const MERGED_V2_PROVENANCE: &str = "shipped merged_v2 artifact: 16-neuron LIF; known training-path defects (monotonic hidden weights, lockstep); not a post-exp-009 legal-encoder retrain; not session-holdout 5-ch v3";

/// Embedded copy of `dataset/merged_v2/snn_model.json`.
///
/// Compiled in so [`SnnModel::load_default`] does not resolve a
/// `CARGO_MANIFEST_DIR` path at runtime.
const SHIPPED_MODEL_JSON: &str = include_str!("../dataset/merged_v2/snn_model.json");

/// Checkout path of the shipped `merged_v2` JSON.
///
/// This is a source-tree helper (`CARGO_MANIFEST_DIR` + [`MODEL_RELATIVE_PATH`]).
/// Deployed binaries should use [`SnnModel::load_default`], which reads the
/// embedded copy and does not depend on this path existing at runtime.
#[must_use]
pub fn default_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_RELATIVE_PATH)
}

/// One leaky integrate-and-fire unit as stored in `snn_model.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Neuron {
    /// Per-step membrane decay multiplier, in `(0, 1)`.
    ///
    /// This is the discrete-time factor `v[t+1] = decay_rate * v[t] + I`, not a
    /// NIR time constant. Use [`Neuron::tau_seconds`] to convert.
    ///
    /// Decoding rejects anything outside `(0, 1)`, so a model that came from
    /// [`SnnModel::from_json_str`] always satisfies the invariant. The field is
    /// public, so a hand-built [`Neuron`] can still break it; every conversion
    /// re-checks rather than assuming.
    pub decay_rate: f64,
    /// Membrane potential captured when the model was saved (simulator state).
    pub membrane_potential: f64,
    /// Firing threshold, mapped to the NIR `v_threshold` parameter.
    pub threshold: f64,
    /// Whether the unit spiked on the last saved step (simulator state).
    pub last_spike: bool,
    /// Learned input weights, one per input channel (`NEURON_COUNT` entries).
    ///
    /// Row `i` of `parameters_weights.mem`. The hidden layer is purely
    /// feed-forward — the README records that the network has no recurrent
    /// feedback — so these weigh the graph's input, not the population's own
    /// spikes. [`crate::graph`] places them on a NIR `Linear` node.
    pub weights: Vec<f64>,
}

impl Neuron {
    /// The NIR membrane time constant, in seconds.
    ///
    /// NIR's LIF integrates `tau * dv/dt = (v_leak - v) + R*I`, so one step of
    /// length `dt` decays the membrane by `exp(-dt / tau)`. Inverting the
    /// stored per-step multiplier gives `tau = -dt / ln(decay_rate)`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Schema`] unless `decay_rate` lies in `(0, 1)`,
    /// where the inversion is defined.
    pub fn tau_seconds(&self, timestep_seconds: f64) -> Result<f64, ModelError> {
        tau_from_decay(self.decay_rate, timestep_seconds)
    }
}

/// Convert a per-step decay multiplier into a NIR membrane time constant.
///
/// See [`Neuron::tau_seconds`].
///
/// # Errors
///
/// Returns [`ModelError::Schema`] if `decay_rate` is outside `(0, 1)`, if
/// `timestep_seconds` is not finite and positive, or if the two are finite but
/// so extreme that `-dt / ln(decay_rate)` is not itself a finite positive
/// number.
///
/// The last case is not hypothetical: `tau_from_decay(0.5, f64::MAX)` overflows
/// to `inf`, and a subnormal timestep against the smallest on-grid decay rate
/// (`1 / 256`) underflows to `0.0`. Both would put a meaningless time constant
/// on a NIR `LIF` node, so the result is checked, not just the inputs.
pub fn tau_from_decay(decay_rate: f64, timestep_seconds: f64) -> Result<f64, ModelError> {
    if !timestep_seconds.is_finite() || timestep_seconds <= 0.0 {
        return Err(ModelError::Schema(format!(
            "timestep must be finite and positive, got {timestep_seconds}"
        )));
    }
    if !(decay_rate.is_finite() && decay_rate > 0.0 && decay_rate < 1.0) {
        return Err(ModelError::Schema(format!(
            "decay_rate must lie in (0, 1) to invert into a time constant, got {decay_rate}"
        )));
    }
    let tau = -timestep_seconds / decay_rate.ln();
    if !(tau.is_finite() && tau > 0.0) {
        return Err(ModelError::Schema(format!(
            "decay_rate {decay_rate} at timestep {timestep_seconds} gives a time constant of \
             {tau}, expected a finite positive number"
        )));
    }
    Ok(tau)
}

/// The decoded `merged_v2` model.
#[derive(Debug, Clone, PartialEq)]
pub struct SnnModel {
    /// The units, in file order. Row `i` of every weight vector indexes this list.
    pub neurons: Vec<Neuron>,
}

impl SnnModel {
    /// Decode the shipped `merged_v2` model from the embedded JSON.
    ///
    /// This is the repository artifact described by [`MERGED_V2_PROVENANCE`]:
    /// 16-neuron LIF, known training-path defects, not a post-exp-009
    /// legal-encoder retrain and not session-holdout 5-ch v3.
    ///
    /// The bytes are compiled in (`include_str!`), so this does not depend on
    /// a checkout path or `CARGO_MANIFEST_DIR` at runtime. [`default_model_path`]
    /// remains available for source-tree tools.
    ///
    /// # Errors
    ///
    /// See [`SnnModel::from_json_str`]. Also [`ModelError::Schema`] if the
    /// embedded document does not contain exactly [`NEURON_COUNT`] units.
    pub fn load_default() -> Result<Self, ModelError> {
        let model = Self::from_json_str(SHIPPED_MODEL_JSON)?;
        require_merged_v2_width(&model)?;
        Ok(model)
    }

    /// Read and decode a model file.
    ///
    /// # Errors
    ///
    /// - [`ModelError::Io`] if the file cannot be read
    /// - [`ModelError::Json`] if it is not well-formed JSON
    /// - [`ModelError::Schema`] if it is not a well-formed SNN model
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ModelError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_str(&text)
    }

    /// Decode a model from JSON text.
    ///
    /// # Errors
    ///
    /// - [`ModelError::Json`] if `text` is not well-formed JSON
    /// - [`ModelError::Schema`] if the document does not hold a non-empty
    ///   `neurons` list whose entries all carry the expected fields and a
    ///   square weight matrix, or if a numeric field breaks its documented
    ///   invariant: every number must be finite and inside the Q8.8 range, and
    ///   `decay_rate` must lie in `(0, 1)`
    /// - [`ModelError::Schema`] if the document or any neuron carries a member
    ///   this decoder does not read. An unrecognised member is a schema change,
    ///   and refusing it here is what stops a later revision of the artifact
    ///   from decoding partially and still being labelled `merged_v2`.
    ///
    /// Numbers are snapped onto the Q8.8 grid; see [`quantize_q8_8`].
    pub fn from_json_str(text: &str) -> Result<Self, ModelError> {
        let document = json::parse(text)?;
        reject_unknown_members("the document", &document, &["neurons"])?;
        let entries = neuron_entries(&document)?;

        let neurons = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| parse_neuron(index, entry, entries.len()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { neurons })
    }

    /// Number of units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Whether the model has no units. Never true for a decoded model.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Firing thresholds, in unit order.
    #[must_use]
    pub fn thresholds(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.threshold).collect()
    }

    /// Per-step decay multipliers, in unit order.
    #[must_use]
    pub fn decay_rates(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.decay_rate).collect()
    }

    /// NIR membrane time constants in seconds, in unit order.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Schema`] if any decay rate cannot be inverted; see
    /// [`Neuron::tau_seconds`].
    pub fn taus_seconds(&self, timestep_seconds: f64) -> Result<Vec<f64>, ModelError> {
        self.neurons
            .iter()
            .map(|n| n.tau_seconds(timestep_seconds))
            .collect()
    }

    /// The learned input weight matrix as a row-major `[units, units]` tensor.
    ///
    /// Row `i` holds the weights unit `i` applies to the input channels, laid
    /// out exactly like `parameters_weights.mem`. [`crate::graph`] puts this
    /// tensor on the `Linear` node between `Input` and the LIF population, so
    /// all 256 learned values reach the graph.
    ///
    /// # Errors
    ///
    /// - [`ModelError::Schema`] if any unit's weight row is not exactly `units`
    ///   long. Flattening discards the row boundaries, so a length check on the
    ///   concatenation alone is not enough: rows of 3 and 1 also total 4 and
    ///   would pass as a 2×2 tensor with one weight shifted into the wrong
    ///   neuron. Decoding enforces this per row already; the check covers the
    ///   hand-built and mutated [`SnnModel`] values the public fields allow.
    /// - [`ModelError::Nir`] if `nir-rs` rejects the tensor.
    pub fn weight_tensor(&self) -> Result<Tensor, ModelError> {
        let units = self.len();
        let mut data = Vec::with_capacity(units * units);
        for (index, neuron) in self.neurons.iter().enumerate() {
            if neuron.weights.len() != units {
                return Err(ModelError::Schema(format!(
                    "neuron {index} has {} weights, expected {units} (one per neuron)",
                    neuron.weights.len()
                )));
            }
            for (column, &weight) in neuron.weights.iter().enumerate() {
                check_q8_8(&format!("neuron {index} weight {column}"), weight)?;
            }
            data.extend_from_slice(&neuron.weights);
        }
        Ok(Tensor::from_f64(vec![units, units], data)?)
    }
}

/// Refuse a model that is not the shipped 16-unit population.
///
/// Only the `load_default*` entry points call this, and only they stamp
/// `Provenance::MERGED_V2`. The width is what the stamp asserts: the `.mem`
/// artifacts, the FPGA bitstream and `config.json`'s `n_channels` all agree on
/// 16, so a model of any other size is not the thing being claimed, whatever
/// else is right about it. Generic callers go through the unstamped builders
/// and are not held to this.
fn require_merged_v2_width(model: &SnnModel) -> Result<(), ModelError> {
    if model.len() != NEURON_COUNT {
        return Err(ModelError::Schema(format!(
            "shipped merged_v2 must have {NEURON_COUNT} neurons, got {}",
            model.len()
        )));
    }
    Ok(())
}

/// Decode one entry of the `neurons` array.
///
/// `index` names the unit in error messages; `expected_weights` is the length
/// the row must have, which the caller sets to the number of units so the
/// matrix comes out square.
///
/// The contract is stricter than it looks, and deliberately asymmetric:
///
/// - Unknown members are refused, not ignored — see [`reject_unknown_members`]
///   for why the provenance stamp depends on that.
/// - `weights`, `threshold` and `membrane_potential` go through
///   [`q8_8_field`], which *snaps* them onto the Q8.8 grid rather than
///   demanding they already sit on it. The artifact prints at seven
///   significant digits, so off-grid input is expected here.
/// - `decay_rate` is not a Q8.8 field. It is a per-step multiplier and must lie
///   strictly inside `(0, 1)`, since [`tau_from_decay`] takes its logarithm.
/// - `last_spike` is simulator state, carried but never placed on the graph.
///
/// # Errors
///
/// [`ModelError::Schema`], naming the unit and the field, if a member is
/// missing, has the wrong JSON type, breaks one of the invariants above, or is
/// not read by this decoder at all.
/// The neuron array, rejecting a document that does not carry a non-empty one.
fn neuron_entries(document: &Json) -> Result<&[Json], ModelError> {
    let entries = document
        .get("neurons")
        .ok_or_else(|| ModelError::Schema("missing top-level `neurons` key".into()))?
        .as_array()
        .ok_or_else(|| ModelError::Schema("`neurons` is not an array".into()))?;
    if entries.is_empty() {
        return Err(ModelError::Schema("`neurons` is empty".into()));
    }
    Ok(entries)
}

/// Decode the three scalars, in the order their errors must surface.
fn parse_neuron_scalars(fields: &NeuronFields<'_>) -> Result<(f64, f64, f64), ModelError> {
    let index = fields.index;
    let decay_rate = decay_field(
        &format!("neuron {index} field `decay_rate`"),
        fields.number("decay_rate")?,
    )?;
    let membrane_potential = q8_8_field(
        &format!("neuron {index} field `membrane_potential`"),
        fields.number("membrane_potential")?,
    )?;
    let threshold = q8_8_field(
        &format!("neuron {index} field `threshold`"),
        fields.number("threshold")?,
    )?;
    Ok((decay_rate, membrane_potential, threshold))
}

/// Field accessors for one neuron's JSON object.
///
/// Exists so every error message names the neuron and field it came from
/// without threading `index` through each lookup by hand.
struct NeuronFields<'a> {
    index: usize,
    entry: &'a Json,
}

impl<'a> NeuronFields<'a> {
    /// The raw value of `name`, or a schema error naming the missing field.
    fn get(&self, name: &str) -> Result<&'a Json, ModelError> {
        let index = self.index;
        self.entry
            .get(name)
            .ok_or_else(|| ModelError::Schema(format!("neuron {index} is missing `{name}`")))
    }

    /// The value of `name` as a number, or a schema error naming its type.
    fn number(&self, name: &str) -> Result<f64, ModelError> {
        let index = self.index;
        let value = self.get(name)?;
        value.as_f64().ok_or_else(|| {
            ModelError::Schema(format!(
                "neuron {index} field `{name}` is a {}, expected a number",
                value.type_name()
            ))
        })
    }

    /// The value of `name` as a boolean, or a schema error naming its type.
    fn boolean(&self, name: &str) -> Result<bool, ModelError> {
        let index = self.index;
        let value = self.get(name)?;
        value.as_bool().ok_or_else(|| {
            ModelError::Schema(format!(
                "neuron {index} field `{name}` is a {}, expected a boolean",
                value.type_name()
            ))
        })
    }
}

/// Decode one neuron's weight row, checking arity and the Q8.8 grid.
fn parse_neuron_weights(
    fields: &NeuronFields<'_>,
    expected_weights: usize,
) -> Result<Vec<f64>, ModelError> {
    let index = fields.index;
    let weights_value = fields.get("weights")?;
    let weights_items = weights_value.as_array().ok_or_else(|| {
        ModelError::Schema(format!(
            "neuron {index} field `weights` is a {}, expected an array",
            weights_value.type_name()
        ))
    })?;
    if weights_items.len() != expected_weights {
        return Err(ModelError::Schema(format!(
            "neuron {index} has {} weights, expected {expected_weights} (one per neuron)",
            weights_items.len()
        )));
    }
    weights_items
        .iter()
        .enumerate()
        .map(|(column, weight)| {
            let value = weight.as_f64().ok_or_else(|| {
                ModelError::Schema(format!(
                    "neuron {index} weight {column} is a {}, expected a number",
                    weight.type_name()
                ))
            })?;
            q8_8_field(&format!("neuron {index} weight {column}"), value)
        })
        .collect()
}

fn parse_neuron(index: usize, entry: &Json, expected_weights: usize) -> Result<Neuron, ModelError> {
    reject_unknown_members(
        &format!("neuron {index}"),
        entry,
        &[
            "decay_rate",
            "last_spike",
            "membrane_potential",
            "threshold",
            "weights",
        ],
    )?;
    let fields = NeuronFields { index, entry };

    // Decoded before the struct literal to keep the original error precedence:
    // a neuron with both a bad weight and a bad scalar reports the weight.
    let weights = parse_neuron_weights(&fields, expected_weights)?;
    let last_spike = fields.boolean("last_spike")?;

    let (decay_rate, membrane_potential, threshold) = parse_neuron_scalars(&fields)?;

    Ok(Neuron {
        decay_rate,
        membrane_potential,
        threshold,
        last_spike,
        weights,
    })
}

/// Reject an object that carries members this decoder does not read.
///
/// The graph builder stamps `Provenance::MERGED_V2` on whatever this decoder
/// produces. Ignoring an unrecognised member would let a later revision of the
/// artifact — a retrain that adds per-neuron output weights, say — decode
/// partially and still ship under that stamp, describing a model the graph does
/// not contain. Failing here makes the schema change visible at load time
/// instead of silent in the graph.
///
/// `context` names the object for the error message. A non-object is an error
/// too: every caller has already committed to reading members off it.
fn reject_unknown_members(context: &str, value: &Json, known: &[&str]) -> Result<(), ModelError> {
    let members = value.as_object().ok_or_else(|| {
        ModelError::Schema(format!(
            "{context} is not an object (found {})",
            value.type_name()
        ))
    })?;
    // `BTreeMap` iterates in key order, so the message is stable across runs.
    let unknown: Vec<&str> = members
        .keys()
        .map(String::as_str)
        .filter(|name| !known.contains(name))
        .collect();
    if !unknown.is_empty() {
        return Err(ModelError::Schema(format!(
            "{context} carries unknown member(s) `{}`; this decoder reads only `{}`",
            unknown.join("`, `"),
            known.join("`, `"),
        )));
    }
    Ok(())
}

/// Accept one stored number: it must be finite and Q8.8-representable, and it
/// is snapped onto the grid before the caller ever sees it.
///
/// The range check runs on the *snapped* value, not on the literal in the file.
/// `snn_model.json` prints at seven significant digits, which is why
/// [`check_finite_and_in_range`] tolerates off-grid input at all — but that
/// same formatting writes the largest Q8.8 code, `0x7FFF` = 127.996_093_75, as
/// `127.9961`, which overshoots [`Q8_8_MAX`] by 6.25e-6. Checking the literal
/// would reject a boundary value the encoding can hold perfectly well, purely
/// because of how it was printed. Checking what it encodes to does not: the
/// question that matters is whether the number lands on a real Q8.8 code, and
/// `127.9961` lands on the largest one exactly.
///
/// This does not widen the range. A value that snaps past the top still fails:
/// 127.998_046_875 scales to 32767.5, rounds away from zero to code 32768, and
/// 128.0 is not encodable. Half an LSB either side of the extreme codes is the
/// whole of what this admits, and it is admitted at the negative end too for
/// symmetry.
///
/// Only the decode path does this. [`check_q8_8`], the public-API boundary,
/// demands an exactly-representable value, so there is no rounding interval to
/// forgive there.
///
/// `context` names the field for the error message, which quotes the value as
/// written rather than as snapped, so it points at the file.
fn q8_8_field(context: &str, value: f64) -> Result<f64, ModelError> {
    if !value.is_finite() {
        return Err(ModelError::Schema(format!(
            "{context} is {value}, expected a finite number"
        )));
    }
    // `value * Q8_8_SCALE` overflows to infinity near `f64::MAX`, so the
    // snapped value gets its own range check rather than being trusted.
    let snapped = quantize_q8_8(value);
    if !(Q8_8_MIN..=Q8_8_MAX).contains(&snapped) {
        return Err(ModelError::Schema(format!(
            "{context} is {value}, outside the Q8.8 range [{Q8_8_MIN}, {Q8_8_MAX}]"
        )));
    }
    Ok(snapped)
}

/// Check one number is finite and inside the Q8.8 range, saying nothing about
/// whether it lands on the grid.
///
/// This is the decode pre-check: `snn_model.json` prints Q8.8 values at limited
/// precision (`0.7539062` for `193/256`), so off-grid input is *expected* there
/// and [`q8_8_field`] snaps it. Rejecting it would reject the shipped artifact.
fn check_finite_and_in_range(context: &str, value: f64) -> Result<(), ModelError> {
    if !value.is_finite() {
        return Err(ModelError::Schema(format!(
            "{context} is {value}, expected a finite number"
        )));
    }
    if !(Q8_8_MIN..=Q8_8_MAX).contains(&value) {
        return Err(ModelError::Schema(format!(
            "{context} is {value}, outside the Q8.8 range [{Q8_8_MIN}, {Q8_8_MAX}]"
        )));
    }
    Ok(())
}

/// Check one number is a value the Q8.8 grid can actually hold, **without**
/// changing it.
///
/// This is the re-check for numbers arriving through the public [`SnnModel`]
/// fields, which never cross the decode path. It is stricter than
/// [`check_finite_and_in_range`] in one way and looser in another: being inside
/// the range is not the same as being representable — the grid holds multiples
/// of `1/256`, so `0.1` is in range with no code — but unlike [`q8_8_field`] it
/// will not quantize, because silently rounding a caller's number would hand
/// back a graph whose weights differ from the ones they set, with no error.
pub(crate) fn check_q8_8(context: &str, value: f64) -> Result<(), ModelError> {
    check_finite_and_in_range(context, value)?;
    let scaled = value * Q8_8_SCALE;
    if scaled.fract() != 0.0 {
        return Err(ModelError::Schema(format!(
            "{context} is {value}, which is not Q8.8-representable \
             (nearest codes {}/256 and {}/256)",
            scaled.floor(),
            scaled.ceil()
        )));
    }
    Ok(())
}

/// Accept a stored decay multiplier, enforcing the `(0, 1)` invariant that
/// [`Neuron::decay_rate`] documents.
///
/// Without this, a model with a nonsensical decay decodes cleanly and only
/// fails much later, in [`crate::graph`] — and never at all for a caller that
/// reads [`SnnModel::weight_tensor`] or the fields directly.
fn decay_field(context: &str, value: f64) -> Result<f64, ModelError> {
    let decay_rate = q8_8_field(context, value)?;
    if !(decay_rate > 0.0 && decay_rate < 1.0) {
        return Err(ModelError::Schema(format!(
            "{context} is {decay_rate}, expected a per-step decay multiplier in (0, 1)"
        )));
    }
    Ok(decay_rate)
}

/// Everything that can go wrong loading a model or mapping it into NIR.
#[derive(Debug)]
#[non_exhaustive]
pub enum ModelError {
    /// The model file could not be read.
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying I/O failure.
        source: std::io::Error,
    },
    /// The model file is not well-formed JSON.
    Json(JsonError),
    /// The JSON is well-formed but is not a well-formed SNN model.
    Schema(String),
    /// `nir-rs` rejected a tensor or graph built from the model.
    Nir(NirError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            Self::Json(source) => write!(f, "cannot parse model JSON: {source}"),
            Self::Schema(message) => write!(f, "invalid SNN model: {message}"),
            Self::Nir(source) => write!(f, "nir-rs rejected the model: {source}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::Schema(_) => None,
            Self::Nir(source) => Some(source),
        }
    }
}

impl From<JsonError> for ModelError {
    fn from(source: JsonError) -> Self {
        Self::Json(source)
    }
}

impl From<NirError> for ModelError {
    fn from(source: NirError) -> Self {
        Self::Nir(source)
    }
}

#[cfg(test)]
mod tests;

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
#[must_use]
pub fn quantize_q8_8(value: f64) -> f64 {
    (value * Q8_8_SCALE).round() / Q8_8_SCALE
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
    ///
    /// Numbers are snapped onto the Q8.8 grid; see [`quantize_q8_8`].
    pub fn from_json_str(text: &str) -> Result<Self, ModelError> {
        let document = json::parse(text)?;
        let entries = document
            .get("neurons")
            .ok_or_else(|| ModelError::Schema("missing top-level `neurons` key".into()))?
            .as_array()
            .ok_or_else(|| ModelError::Schema("`neurons` is not an array".into()))?;

        if entries.is_empty() {
            return Err(ModelError::Schema("`neurons` is empty".into()));
        }

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

fn require_merged_v2_width(model: &SnnModel) -> Result<(), ModelError> {
    if model.len() != NEURON_COUNT {
        return Err(ModelError::Schema(format!(
            "shipped merged_v2 must have {NEURON_COUNT} neurons, got {}",
            model.len()
        )));
    }
    Ok(())
}

fn parse_neuron(index: usize, entry: &Json, expected_weights: usize) -> Result<Neuron, ModelError> {
    let field = |name: &str| -> Result<&Json, ModelError> {
        entry
            .get(name)
            .ok_or_else(|| ModelError::Schema(format!("neuron {index} is missing `{name}`")))
    };
    let number = |name: &str| -> Result<f64, ModelError> {
        let value = field(name)?;
        value.as_f64().ok_or_else(|| {
            ModelError::Schema(format!(
                "neuron {index} field `{name}` is a {}, expected a number",
                value.type_name()
            ))
        })
    };

    let weights_value = field("weights")?;
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
    let weights = weights_items
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
        .collect::<Result<Vec<_>, _>>()?;

    let last_spike_value = field("last_spike")?;
    let last_spike = last_spike_value.as_bool().ok_or_else(|| {
        ModelError::Schema(format!(
            "neuron {index} field `last_spike` is a {}, expected a boolean",
            last_spike_value.type_name()
        ))
    })?;

    Ok(Neuron {
        decay_rate: decay_field(
            &format!("neuron {index} field `decay_rate`"),
            number("decay_rate")?,
        )?,
        membrane_potential: q8_8_field(
            &format!("neuron {index} field `membrane_potential`"),
            number("membrane_potential")?,
        )?,
        threshold: q8_8_field(
            &format!("neuron {index} field `threshold`"),
            number("threshold")?,
        )?,
        last_spike,
        weights,
    })
}

/// Accept one stored number: it must be finite and Q8.8-representable, and it
/// is snapped onto the grid before the caller ever sees it.
///
/// `context` names the field for the error message.
fn q8_8_field(context: &str, value: f64) -> Result<f64, ModelError> {
    check_finite_and_in_range(context, value)?;
    Ok(quantize_q8_8(value))
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
mod tests {
    use super::*;
    use nir_rs::types::TensorData;

    fn stub_json(units: usize) -> String {
        let rows: Vec<String> = (0..units)
            .map(|i| {
                let weights: Vec<String> = (0..units).map(|j| format!("{}.0", i + j)).collect();
                format!(
                    r#"{{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [{}], "last_spike": false}}"#,
                    weights.join(", ")
                )
            })
            .collect();
        format!(r#"{{"neurons": [{}]}}"#, rows.join(", "))
    }

    #[test]
    fn loads_shipped_model() {
        let model = SnnModel::load_default().unwrap();
        assert_eq!(model.len(), NEURON_COUNT);
        assert!(!model.is_empty());
        assert_eq!(model.neurons[0].weights.len(), NEURON_COUNT);
        assert_eq!(model.neurons[0].threshold, 1.125);
        assert_eq!(model.neurons[0].decay_rate, 0.796875);
        assert_eq!(model.neurons[0].membrane_potential, 0.0);
        assert!(!model.neurons[0].last_spike);
        assert_eq!(model.thresholds().len(), NEURON_COUNT);
        assert_eq!(model.decay_rates().len(), NEURON_COUNT);
        assert!(MERGED_V2_PROVENANCE.contains("shipped merged_v2 artifact"));
        assert!(MERGED_V2_PROVENANCE.contains("not a post-exp-009 legal-encoder retrain"));
    }

    #[test]
    fn load_default_matches_the_checkout_file() {
        let embedded = SnnModel::load_default().unwrap();
        let from_file = SnnModel::from_path(default_model_path()).unwrap();
        assert_eq!(embedded, from_file);
    }

    #[test]
    fn shipped_loader_requires_sixteen_units() {
        let stub = SnnModel::from_json_str(&stub_json(3)).unwrap();
        let err = require_merged_v2_width(&stub).unwrap_err();
        assert!(err.to_string().contains("must have 16 neurons"));
        let shipped = SnnModel::from_path(default_model_path()).unwrap();
        require_merged_v2_width(&shipped).unwrap();
    }

    #[test]
    fn weight_tensor_is_square_and_row_major() {
        let model = SnnModel::from_json_str(&stub_json(3)).unwrap();
        let tensor = model.weight_tensor().unwrap();
        assert_eq!(tensor.shape(), [3, 3]);
        assert_eq!(tensor.numel(), 9);
        let TensorData::F64(values) = tensor.data() else {
            panic!("expected an f64 tensor");
        };
        // Row i, column j holds i + j for the stub.
        assert_eq!(values, &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
    }

    /// Flattening throws away the row boundaries, so the total length is not a
    /// sufficient check: rows of 3 and 1 also total 4 and would be accepted as
    /// a 2×2 tensor with `4.0` shifted from unit 1 column 0 into column 1.
    /// `SnnModel`'s fields are public, so decoding is not the only way in.
    #[test]
    fn weight_tensor_rejects_ragged_rows() {
        let row = |weights: Vec<f64>| Neuron {
            decay_rate: 0.5,
            membrane_potential: 0.0,
            threshold: 1.0,
            last_spike: false,
            weights,
        };

        let ragged = SnnModel {
            neurons: vec![row(vec![1.0, 2.0, 3.0]), row(vec![4.0])],
        };
        // The corrupting case: 3 + 1 == 2 * 2, so the length check alone passes.
        assert_eq!(
            ragged
                .neurons
                .iter()
                .map(|n| n.weights.len())
                .sum::<usize>(),
            ragged.len() * ragged.len(),
        );
        let err = ragged.weight_tensor().unwrap_err();
        assert!(matches!(err, ModelError::Schema(_)), "{err}");
        let message = err.to_string();
        assert!(message.contains("neuron 0"), "{message}");
        assert!(message.contains("3 weights"), "{message}");
        assert!(message.contains("expected 2"), "{message}");

        // A short row that does not sum to units² is caught too, and named.
        let short = SnnModel {
            neurons: vec![row(vec![1.0, 2.0]), row(vec![3.0])],
        };
        let message = short.weight_tensor().unwrap_err().to_string();
        assert!(message.contains("neuron 1"), "{message}");

        // Square rows still build.
        let square = SnnModel {
            neurons: vec![row(vec![1.0, 2.0]), row(vec![3.0, 4.0])],
        };
        let tensor = square.weight_tensor().unwrap();
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(tensor.data(), &TensorData::F64(vec![1.0, 2.0, 3.0, 4.0]));
    }

    /// A model mutated after decoding must not corrupt the matrix either.
    #[test]
    fn weight_tensor_rejects_a_mutated_row() {
        let mut model = SnnModel::load_default().unwrap();
        assert!(model.weight_tensor().is_ok());

        // Move one weight from unit 3 to unit 4: the total is still 256.
        let moved = model.neurons[3].weights.pop().unwrap();
        model.neurons[4].weights.push(moved);
        assert_eq!(
            model.neurons.iter().map(|n| n.weights.len()).sum::<usize>(),
            NEURON_COUNT * NEURON_COUNT,
        );

        let message = model.weight_tensor().unwrap_err().to_string();
        assert!(message.contains("neuron 3"), "{message}");
        assert!(message.contains("15 weights"), "{message}");
    }

    #[test]
    fn tau_inverts_the_decay_multiplier() {
        let tau = tau_from_decay(0.5, TIMESTEP_SECONDS).unwrap();
        // Re-applying one step of decay must return the original multiplier.
        assert!(((-TIMESTEP_SECONDS / tau).exp() - 0.5).abs() < 1e-12);
        // A slower decay integrates over a longer time constant.
        assert!(tau_from_decay(0.9, TIMESTEP_SECONDS).unwrap() > tau);
    }

    #[test]
    fn tau_rejects_out_of_range_inputs() {
        for decay in [0.0, 1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            assert!(tau_from_decay(decay, TIMESTEP_SECONDS).is_err());
        }
        assert!(tau_from_decay(0.5, 0.0).is_err());
        assert!(tau_from_decay(0.5, -1.0).is_err());
        assert!(tau_from_decay(0.5, f64::NAN).is_err());
        assert!(tau_from_decay(0.5, f64::INFINITY).is_err());
    }

    /// Both inputs can be finite and in range while the quotient is not. The
    /// documented contract is that an uninvertible input errors, so a
    /// non-finite `tau` must not escape as `Ok` — it would reach a NIR `LIF`
    /// node as a meaningless time constant.
    #[test]
    fn tau_rejects_non_finite_results() {
        // Overflow: -f64::MAX / ln(0.5) is about 1.44 * f64::MAX.
        let err = tau_from_decay(0.5, f64::MAX).unwrap_err();
        assert!(matches!(err, ModelError::Schema(_)));
        assert!(err.to_string().contains("time constant"), "{err}");
        assert!(tau_from_decay(0.5, 1e308).is_ok(), "1e308 still inverts");

        // Underflow: a subnormal timestep against the smallest decay rate the
        // Q8.8 grid can hold (1/256) divides away to exactly zero.
        let smallest_on_grid = 1.0 / Q8_8_SCALE;
        let subnormal: f64 = 5e-324;
        assert!(subnormal > 0.0 && subnormal.is_finite());
        let err = tau_from_decay(smallest_on_grid, subnormal).unwrap_err();
        assert!(matches!(err, ModelError::Schema(_)));
        assert!(err.to_string().contains("time constant"), "{err}");

        // Nothing in between is rejected: every value that does invert into a
        // finite positive tau still succeeds.
        for dt in [1e-300, 1e-10, TIMESTEP_SECONDS, 1.0, 1e100] {
            for decay in [1.0 / 256.0, 0.5, 0.796875, 0.94921875] {
                let tau = tau_from_decay(decay, dt).unwrap();
                assert!(tau.is_finite() && tau > 0.0, "tau({decay}, {dt}) = {tau}");
            }
        }
    }

    /// A non-finite `tau` must not reach the graph either.
    #[test]
    fn taus_seconds_rejects_extreme_timesteps() {
        let model = SnnModel::load_default().unwrap();
        assert!(model.taus_seconds(f64::MAX).is_err());
        assert!(model.taus_seconds(TIMESTEP_SECONDS).is_ok());

        // The shipped decays run 0.796875..=0.94921875, so `|ln(decay)|` is at
        // most 0.227 and a subnormal timestep still divides to a nonzero
        // subnormal — no underflow for this model.
        assert!(model.taus_seconds(5e-324).is_ok());

        // Underflow needs a decay far from 1. `1/256` is the smallest the Q8.8
        // grid holds, so this is a decodable model, not an impossible one.
        let fast = SnnModel {
            neurons: vec![Neuron {
                decay_rate: 1.0 / Q8_8_SCALE,
                membrane_potential: 0.0,
                threshold: 1.0,
                last_spike: false,
                weights: vec![0.0],
            }],
        };
        assert!(fast.taus_seconds(5e-324).is_err());
        assert!(fast.taus_seconds(TIMESTEP_SECONDS).is_ok());
    }

    #[test]
    fn taus_cover_every_unit() {
        let model = SnnModel::load_default().unwrap();
        let taus = model.taus_seconds(TIMESTEP_SECONDS).unwrap();
        assert_eq!(taus.len(), NEURON_COUNT);
        assert!(taus.iter().all(|tau| *tau > 0.0 && tau.is_finite()));
    }

    #[test]
    fn rejects_malformed_models() {
        let cases = [
            (r#"{}"#, "missing top-level"),
            (r#"{"neurons": 3}"#, "not an array"),
            (r#"{"neurons": []}"#, "is empty"),
            (
                r#"{"neurons": [{"membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
                "missing `decay_rate`",
            ),
            (
                r#"{"neurons": [{"decay_rate": "x", "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
                "expected a number",
            ),
            (
                r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": 1.0, "last_spike": false}]}"#,
                "expected an array",
            ),
            (
                r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0, 2.0], "last_spike": false}]}"#,
                "expected 1",
            ),
            (
                r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": 0}]}"#,
                "expected a boolean",
            ),
        ];
        for (text, expected) in cases {
            let err = SnnModel::from_json_str(text).unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains(expected),
                "expected {message:?} to contain {expected:?}"
            );
        }
    }

    /// The stored decimals are truncated Q8.8 codes; decoding must restore the
    /// exact values, or the graph is not the model the FPGA runs.
    #[test]
    fn decoding_restores_exact_q8_8_values() {
        let model = SnnModel::load_default().unwrap();

        // `0.808594` in the JSON; `parameters_decay.mem` line 2 is `00CF`.
        assert_eq!(model.neurons[1].decay_rate, 207.0 / 256.0);
        assert_eq!(model.neurons[1].decay_rate, 0.808_593_75);
        // `0.7539062` in the JSON; `parameters_weights.mem` line 2 is `00C1`.
        assert_eq!(model.neurons[0].weights[1], 193.0 / 256.0);
        assert_eq!(model.neurons[0].weights[1], 0.753_906_25);

        // No stored number is left off the grid.
        for neuron in &model.neurons {
            for value in [
                neuron.decay_rate,
                neuron.threshold,
                neuron.membrane_potential,
            ]
            .into_iter()
            .chain(neuron.weights.iter().copied())
            {
                assert_eq!(value * Q8_8_SCALE, (value * Q8_8_SCALE).round(), "{value}");
            }
        }
    }

    #[test]
    fn quantize_q8_8_snaps_to_the_nearest_code() {
        assert_eq!(quantize_q8_8(0.808_594), 207.0 / 256.0);
        assert_eq!(quantize_q8_8(0.753_906_2), 193.0 / 256.0);
        // Already on the grid: unchanged.
        assert_eq!(quantize_q8_8(1.125), 1.125);
        assert_eq!(quantize_q8_8(-0.027_343_75), -0.027_343_75);
        // Never moves by more than half an LSB.
        for code in -2000..2000 {
            let value = f64::from(code) / 512.0;
            assert!((quantize_q8_8(value) - value).abs() <= 0.5 / 256.0);
        }
    }

    /// `decay_rate` is documented as lying in `(0, 1)`; decoding is the boundary
    /// that enforces it, so a caller reading the fields or the weight tensor
    /// directly can never hold an invalid model.
    #[test]
    fn decoding_enforces_numeric_invariants() {
        let cases = [
            (
                r#"{"neurons": [{"decay_rate": 0.0, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
                "decay multiplier in (0, 1)",
            ),
            (
                r#"{"neurons": [{"decay_rate": 1.0, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
                "decay multiplier in (0, 1)",
            ),
            (
                r#"{"neurons": [{"decay_rate": -0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
                "decay multiplier in (0, 1)",
            ),
            (
                r#"{"neurons": [{"decay_rate": 1.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
                "decay multiplier in (0, 1)",
            ),
            (
                r#"{"neurons": [{"decay_rate": 1e300, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
                "outside the Q8.8 range",
            ),
            (
                r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 500.0, "weights": [1.0], "last_spike": false}]}"#,
                "outside the Q8.8 range",
            ),
            (
                r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [-1e9], "last_spike": false}]}"#,
                "outside the Q8.8 range",
            ),
        ];
        for (text, expected) in cases {
            let err = SnnModel::from_json_str(text).unwrap_err();
            let message = err.to_string();
            assert!(matches!(err, ModelError::Schema(_)), "{message}");
            assert!(
                message.contains(expected),
                "expected {message:?} to contain {expected:?}"
            );
        }

        // A decay rate that rounds *onto* a boundary is rejected too: 0.999 is
        // one LSB short of 1.0 and snaps to exactly 1.0.
        let err = SnnModel::from_json_str(
            r#"{"neurons": [{"decay_rate": 0.999, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("decay multiplier in (0, 1)"));
    }

    #[test]
    fn reports_json_and_io_failures() {
        let err = SnnModel::from_json_str("not json").unwrap_err();
        assert!(matches!(err, ModelError::Json(_)));
        assert!(err.to_string().contains("cannot parse model JSON"));

        let missing = default_model_path().with_file_name("does-not-exist.json");
        let err = SnnModel::from_path(&missing).unwrap_err();
        assert!(matches!(err, ModelError::Io { .. }));
        assert!(err.to_string().contains("does-not-exist.json"));
    }
}

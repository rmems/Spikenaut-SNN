// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shipped `merged_v2` model as it is stored on disk.
//!
//! [`SnnModel`] is a direct, validated decoding of
//! `dataset/merged_v2/snn_model.json`: 16 leaky integrate-and-fire neurons,
//! each with a decay rate, a firing threshold, a recurrent weight row, and the
//! simulator state (membrane potential, last-spike flag) captured when the model
//! was saved.
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

/// Provenance stamp for the shipped `merged_v2` artifact loaded by this crate.
///
/// This is the repository file at [`MODEL_RELATIVE_PATH`], a 16-neuron LIF
/// population with known training-path defects. It is not a post-exp-009
/// legal-encoder retrain and not the session-holdout 5-ch v3 encoder.
pub const MERGED_V2_PROVENANCE: &str = "shipped merged_v2 artifact: 16-neuron LIF; known training-path defects (monotonic hidden weights, lockstep); not a post-exp-009 legal-encoder retrain; not session-holdout 5-ch v3";

/// Absolute path of the shipped `merged_v2` model in this checkout.
///
/// Resolved from `CARGO_MANIFEST_DIR`, which is the repository root, so callers
/// and tests do not depend on the working directory.
#[must_use]
pub fn default_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_RELATIVE_PATH)
}

/// One leaky integrate-and-fire unit as stored in `snn_model.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Neuron {
    /// Per-step membrane decay multiplier, in `(0, 1)`.
    ///
    /// This is the discrete-time factor `v[t+1] = decay_rate * v[t] + ...`, not
    /// a NIR time constant. Use [`Neuron::tau_seconds`] to convert.
    pub decay_rate: f64,
    /// Membrane potential captured when the model was saved (simulator state).
    pub membrane_potential: f64,
    /// Firing threshold, mapped to the NIR `v_threshold` parameter.
    pub threshold: f64,
    /// Whether the unit spiked on the last saved step (simulator state).
    pub last_spike: bool,
    /// Recurrent input weights, one per source neuron (`NEURON_COUNT` entries).
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
/// Returns [`ModelError::Schema`] if `decay_rate` is outside `(0, 1)` or
/// `timestep_seconds` is not finite and positive.
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
    Ok(-timestep_seconds / decay_rate.ln())
}

/// The decoded `merged_v2` model.
#[derive(Debug, Clone, PartialEq)]
pub struct SnnModel {
    /// The units, in file order. Row `i` of every weight vector indexes this list.
    pub neurons: Vec<Neuron>,
}

impl SnnModel {
    /// Decode the shipped `merged_v2` model at [`default_model_path`].
    ///
    /// This is the repository artifact described by [`MERGED_V2_PROVENANCE`]:
    /// 16-neuron LIF, known training-path defects, not a post-exp-009
    /// legal-encoder retrain and not session-holdout 5-ch v3.
    ///
    /// # Errors
    ///
    /// See [`SnnModel::from_path`]. Also [`ModelError::Schema`] if the file
    /// does not contain exactly [`NEURON_COUNT`] units.
    pub fn load_default() -> Result<Self, ModelError> {
        let model = Self::from_path(default_model_path())?;
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
    ///   square weight matrix
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

    /// The recurrent weight matrix as a row-major `[units, units]` tensor.
    ///
    /// Row `i` holds the incoming weights of unit `i`. The Input → LIF → Output
    /// graph this crate builds does not place these weights on an edge: mapping
    /// recurrence onto NIR `Affine` / `Linear` nodes belongs to the exporter
    /// tickets, so the matrix is exposed here for those consumers.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Nir`] if the tensor shape and data length disagree,
    /// which decoding already rules out.
    pub fn weight_tensor(&self) -> Result<Tensor, ModelError> {
        let units = self.len();
        let mut data = Vec::with_capacity(units * units);
        for neuron in &self.neurons {
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
            weight.as_f64().ok_or_else(|| {
                ModelError::Schema(format!(
                    "neuron {index} weight {column} is a {}, expected a number",
                    weight.type_name()
                ))
            })
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
        decay_rate: number("decay_rate")?,
        membrane_potential: number("membrane_potential")?,
        threshold: number("threshold")?,
        last_spike,
        weights,
    })
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

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked Q8.8 export of the shipped model through `silicon-bridge`.
//!
//! The published exporter validates readout matrices as `K × N` (outputs by
//! hidden neurons). The checked-in FPGA vault and silicon-hdl `OutputLayer`
//! address the same values as `N × K` (neuron by output). This module performs
//! that translation explicitly: it transposes into the exporter, then restores
//! neuron-major order before presenting `parameters_output_weights.mem`.
//!
//! No UART feature is enabled here. Producing deterministic parameter images
//! and proving a live board protocol are intentionally separate contracts.

use crate::json::Json;
use crate::model::{ModelError, SHIPPED_MODEL_JSON, SnnModel, q8_8_field};
use silicon_bridge::{FpgaParameterExporter, ParameterShapeError};
use std::error::Error;
use std::fmt;

/// The four FPGA memory-image names, in hidden-layer execution order.
pub const FPGA_MEM_FILENAMES: [&str; 4] = [
    "parameters.mem",
    "parameters_weights.mem",
    "parameters_decay.mem",
    "parameters_output_weights.mem",
];

/// One deterministic `$readmemh` image generated from the shipped model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FpgaMemFile {
    /// Basename expected by the checked-in FPGA vault.
    pub name: &'static str,
    /// Uppercase 16-bit hexadecimal words, one per line, with a final newline.
    pub contents: String,
}

/// Checked signed Q8.8 words for the complete shipped FPGA bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShippedFpgaImage {
    /// One threshold per hidden neuron.
    pub thresholds: Vec<i16>,
    /// Hidden weights flattened `N × N`, neuron-major.
    pub weights: Vec<i16>,
    /// One per-step decay word per hidden neuron.
    pub decay_rates: Vec<i16>,
    /// Readout weights flattened `N × K`, neuron-major for silicon-hdl.
    pub output_weights: Vec<i16>,
    /// Outgoing Dale polarity flags retained from the model sidecar.
    pub inhibitory: Vec<bool>,
}

impl ShippedFpgaImage {
    /// Render the four memory images without writing or overwriting files.
    #[must_use]
    pub fn mem_files(&self) -> [FpgaMemFile; 4] {
        [
            FpgaMemFile {
                name: FPGA_MEM_FILENAMES[0],
                contents: words_to_mem(&self.thresholds),
            },
            FpgaMemFile {
                name: FPGA_MEM_FILENAMES[1],
                contents: words_to_mem(&self.weights),
            },
            FpgaMemFile {
                name: FPGA_MEM_FILENAMES[2],
                contents: words_to_mem(&self.decay_rates),
            },
            FpgaMemFile {
                name: FPGA_MEM_FILENAMES[3],
                contents: words_to_mem(&self.output_weights),
            },
        ]
    }
}

/// Failure to decode or validate the shipped model as an FPGA parameter image.
#[derive(Debug)]
#[non_exhaustive]
pub enum SiliconExportError {
    /// The checked-in model or its readout sidecar violates its schema.
    Model(ModelError),
    /// `silicon-bridge` rejected a shape, finite-value, or Q8.8 range contract.
    Parameters(ParameterShapeError),
}

impl fmt::Display for SiliconExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(source) => write!(formatter, "cannot decode FPGA model: {source}"),
            Self::Parameters(source) => write!(formatter, "invalid FPGA parameter image: {source}"),
        }
    }
}

impl Error for SiliconExportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(source) => Some(source),
            Self::Parameters(source) => Some(source),
        }
    }
}

impl From<ModelError> for SiliconExportError {
    fn from(source: ModelError) -> Self {
        Self::Model(source)
    }
}

impl From<ParameterShapeError> for SiliconExportError {
    fn from(source: ParameterShapeError) -> Self {
        Self::Parameters(source)
    }
}

/// Validate and encode the complete shipped model with `silicon-bridge` 0.3.
///
/// Hidden parameters pass through the published exporter's rejecting checked
/// path. Per-neuron readout rows are transposed to the exporter's `K × N`
/// contract for validation and signed Q8.8 encoding, then transposed back to
/// the FPGA vault's `N × K` address order.
///
/// # Errors
///
/// Returns [`SiliconExportError`] if the embedded model or readout sidecar is
/// malformed, non-finite, out of signed Q8.8 range, or shape-incompatible.
pub fn export_shipped_fpga_image() -> Result<ShippedFpgaImage, SiliconExportError> {
    let model = SnnModel::load_default()?;
    let (readout_n_by_k, inhibitory) = shipped_readout()?;
    let neurons = model.len();
    let outputs = readout_n_by_k
        .first()
        .map(Vec::len)
        .ok_or_else(|| schema("the shipped readout has no neuron rows"))?;

    let thresholds = model
        .neurons
        .iter()
        .map(|neuron| neuron.threshold as f32)
        .collect();
    let weights = model
        .neurons
        .iter()
        .map(|neuron| neuron.weights.iter().map(|&value| value as f32).collect())
        .collect();
    let decay_rates = model
        .neurons
        .iter()
        .map(|neuron| neuron.decay_rate as f32)
        .collect();

    let mut readout_k_by_n = vec![vec![0.0_f32; neurons]; outputs];
    for (neuron, row) in readout_n_by_k.iter().enumerate() {
        for (output, &value) in row.iter().enumerate() {
            readout_k_by_n[output][neuron] = value;
        }
    }

    let mut exporter = FpgaParameterExporter::from_params(thresholds, weights, decay_rates);
    exporter.set_output_weights(readout_k_by_n);
    let parameters = exporter.try_export()?;
    let encoded_k_by_n = parameters
        .output_weights
        .ok_or_else(|| schema("silicon-bridge omitted the required readout block"))?;

    let mut output_weights = Vec::with_capacity(neurons * outputs);
    for neuron in 0..neurons {
        for output in 0..outputs {
            output_weights.push(encoded_k_by_n[output * neurons + neuron]);
        }
    }

    Ok(ShippedFpgaImage {
        thresholds: parameters.thresholds,
        weights: parameters.weights,
        decay_rates: parameters.decay_rates,
        output_weights,
        inhibitory,
    })
}

fn shipped_readout() -> Result<(Vec<Vec<f32>>, Vec<bool>), SiliconExportError> {
    let document = crate::json::parse(SHIPPED_MODEL_JSON).map_err(ModelError::from)?;
    let outputs_value = document
        .get("n_outputs")
        .and_then(Json::as_f64)
        .ok_or_else(|| schema("top-level `n_outputs` is missing or is not a number"))?;
    if !outputs_value.is_finite()
        || outputs_value <= 0.0
        || outputs_value.fract() != 0.0
        || outputs_value > usize::MAX as f64
    {
        return Err(schema(format!(
            "top-level `n_outputs` must be a positive integer, got {outputs_value}"
        )));
    }
    let outputs = outputs_value as usize;
    let neurons = document
        .get("neurons")
        .and_then(Json::as_array)
        .ok_or_else(|| schema("top-level `neurons` is missing or is not an array"))?;

    let mut readout = Vec::with_capacity(neurons.len());
    let mut inhibitory = Vec::with_capacity(neurons.len());
    for (neuron, entry) in neurons.iter().enumerate() {
        let values = entry
            .get("output_weights")
            .and_then(Json::as_array)
            .ok_or_else(|| {
                schema(format!(
                    "neuron {neuron} field `output_weights` is missing or is not an array"
                ))
            })?;
        if values.len() != outputs {
            return Err(schema(format!(
                "neuron {neuron} has {} output weights, expected {outputs}",
                values.len()
            )));
        }
        let row = values
            .iter()
            .enumerate()
            .map(|(output, value)| {
                let value = value.as_f64().ok_or_else(|| {
                    schema(format!(
                        "neuron {neuron} output weight {output} is a {}, expected a number",
                        value.type_name()
                    ))
                })?;
                q8_8_field(&format!("neuron {neuron} output weight {output}"), value)
                    .map(|snapped| snapped as f32)
                    .map_err(SiliconExportError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        readout.push(row);

        let flag = entry
            .get("inhibitory")
            .and_then(Json::as_bool)
            .ok_or_else(|| {
                schema(format!(
                    "neuron {neuron} field `inhibitory` is missing or is not a boolean"
                ))
            })?;
        inhibitory.push(flag);
    }
    Ok((readout, inhibitory))
}

fn words_to_mem(words: &[i16]) -> String {
    let mut text = String::with_capacity(words.len() * 5);
    for &word in words {
        use fmt::Write as _;
        writeln!(&mut text, "{:04X}", word as u16).expect("writing to String cannot fail");
    }
    text
}

fn schema(message: impl Into<String>) -> SiliconExportError {
    SiliconExportError::Model(ModelError::Schema(message.into()))
}

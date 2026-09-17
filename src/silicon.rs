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
    let outputs = readout_width(&readout_n_by_k)?;
    let (thresholds, weights, decay_rates) = model_parameters(&model);
    let readout_k_by_n = transpose_for_exporter(&readout_n_by_k, neurons, outputs);

    let mut exporter = FpgaParameterExporter::from_params(thresholds, weights, decay_rates);
    exporter.set_output_weights(readout_k_by_n);
    let parameters = exporter.try_export()?;
    let encoded_k_by_n = parameters
        .output_weights
        .ok_or_else(|| schema("silicon-bridge omitted the required readout block"))?;

    Ok(ShippedFpgaImage {
        thresholds: parameters.thresholds,
        weights: parameters.weights,
        decay_rates: parameters.decay_rates,
        output_weights: restore_neuron_major(&encoded_k_by_n, neurons, outputs),
        inhibitory,
    })
}

type FloatParameters = (Vec<f32>, Vec<Vec<f32>>, Vec<f32>);

fn readout_width(readout: &[Vec<f32>]) -> Result<usize, SiliconExportError> {
    readout
        .first()
        .map(Vec::len)
        .ok_or_else(|| schema("the shipped readout has no neuron rows"))
}

fn model_parameters(model: &SnnModel) -> FloatParameters {
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
    (thresholds, weights, decay_rates)
}

fn transpose_for_exporter(
    neuron_major: &[Vec<f32>],
    neurons: usize,
    outputs: usize,
) -> Vec<Vec<f32>> {
    let mut exporter_order = vec![vec![0.0_f32; neurons]; outputs];
    for (neuron, row) in neuron_major.iter().enumerate() {
        for (output, &value) in row.iter().enumerate() {
            exporter_order[output][neuron] = value;
        }
    }
    exporter_order
}

fn restore_neuron_major(exporter_order: &[i16], neurons: usize, outputs: usize) -> Vec<i16> {
    let mut neuron_major = Vec::with_capacity(neurons * outputs);
    for neuron in 0..neurons {
        for output in 0..outputs {
            neuron_major.push(exporter_order[output * neurons + neuron]);
        }
    }
    neuron_major
}

fn shipped_readout() -> Result<(Vec<Vec<f32>>, Vec<bool>), SiliconExportError> {
    let document = crate::json::parse(SHIPPED_MODEL_JSON).map_err(ModelError::from)?;
    let outputs = shipped_output_count(&document)?;
    let neurons = document
        .get("neurons")
        .and_then(Json::as_array)
        .ok_or_else(|| schema("top-level `neurons` is missing or is not an array"))?;
    neurons
        .iter()
        .enumerate()
        .map(|(neuron, entry)| shipped_neuron_sidecar(entry, neuron, outputs))
        .collect::<Result<Vec<_>, _>>()
        .map(|sidecars| sidecars.into_iter().unzip())
}

fn shipped_output_count(document: &Json) -> Result<usize, SiliconExportError> {
    let value = document
        .get("n_outputs")
        .and_then(Json::as_f64)
        .ok_or_else(|| schema("top-level `n_outputs` is missing or is not a number"))?;
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(schema(format!(
            "top-level `n_outputs` must be a positive integer, got {value}"
        )));
    }
    Ok(value as usize)
}

fn shipped_neuron_sidecar(
    entry: &Json,
    neuron: usize,
    outputs: usize,
) -> Result<(Vec<f32>, bool), SiliconExportError> {
    Ok((
        shipped_output_weights(entry, neuron, outputs)?,
        shipped_inhibitory_flag(entry, neuron)?,
    ))
}

fn shipped_output_weights(
    entry: &Json,
    neuron: usize,
    outputs: usize,
) -> Result<Vec<f32>, SiliconExportError> {
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
    values
        .iter()
        .enumerate()
        .map(|(output, value)| shipped_output_weight(value, neuron, output))
        .collect()
}

fn shipped_output_weight(
    value: &Json,
    neuron: usize,
    output: usize,
) -> Result<f32, SiliconExportError> {
    let value = value.as_f64().ok_or_else(|| {
        schema(format!(
            "neuron {neuron} output weight {output} is a {}, expected a number",
            value.type_name()
        ))
    })?;
    q8_8_field(&format!("neuron {neuron} output weight {output}"), value)
        .map(|snapped| snapped as f32)
        .map_err(SiliconExportError::from)
}

fn shipped_inhibitory_flag(entry: &Json, neuron: usize) -> Result<bool, SiliconExportError> {
    entry
        .get("inhibitory")
        .and_then(Json::as_bool)
        .ok_or_else(|| {
            schema(format!(
                "neuron {neuron} field `inhibitory` is missing or is not a boolean"
            ))
        })
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

#[cfg(test)]
mod tests {
    use super::{restore_neuron_major, transpose_for_exporter};

    #[test]
    fn readout_layout_round_trips_through_exporter_order() {
        let neuron_major = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let exporter_order = transpose_for_exporter(&neuron_major, 3, 2);
        assert_eq!(
            exporter_order,
            vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]]
        );

        let encoded_exporter_order = vec![1_i16, 3, 5, 2, 4, 6];
        assert_eq!(
            restore_neuron_major(&encoded_exporter_order, 3, 2),
            vec![1_i16, 2, 3, 4, 5, 6]
        );
    }
}

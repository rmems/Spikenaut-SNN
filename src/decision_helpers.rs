// SPDX-License-Identifier: MIT OR Apache-2.0
//! Private helpers for decision error display and readout shape.

use std::fmt;

use super::{DecisionError, OUTPUT_WEIGHT_COUNT};
use crate::model::NEURON_COUNT;

pub(super) fn write_non_finite_indices(
    f: &mut fmt::Formatter<'_>,
    indices: &[usize],
) -> fmt::Result {
    let noun = if indices.len() == 1 {
        "score"
    } else {
        "scores"
    };
    let index_word = if indices.len() == 1 {
        "index"
    } else {
        "indexes"
    };
    let rendered = indices
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    write!(f, "non-finite output {noun} at {index_word} {rendered}")
}

pub(super) fn validate_readout_shape(
    neuron_major: &[f64],
    spikes: &[bool],
) -> Result<(), DecisionError> {
    if neuron_major.is_empty() {
        return Err(DecisionError::EmptyRow);
    }
    if neuron_major.len() != OUTPUT_WEIGHT_COUNT {
        return Err(DecisionError::WidthMismatch {
            got: neuron_major.len(),
            expected: OUTPUT_WEIGHT_COUNT,
        });
    }
    if spikes.len() != NEURON_COUNT {
        return Err(DecisionError::WidthMismatch {
            got: spikes.len(),
            expected: NEURON_COUNT,
        });
    }
    Ok(())
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Strict decoder for the four-file, signed Q8.8 FPGA memory bank.
//!
//! This is deliberately independent of `snn_model.json`.  File names, line
//! layout, token spelling, dimensions, and the LIF decay domain are part of the
//! boundary: accepting a vaguely similar text file would make the provenance
//! claim unauditable.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::decision::OUTPUT_WEIGHT_COUNT;
use crate::model::{NEURON_COUNT, Neuron, Q8_8_SCALE, SnnModel};

/// Canonical files, in threshold/decay/hidden-weight/readout order.
pub const MEM_BANK_FILENAMES: [&str; 4] = [
    "parameters.mem",
    "parameters_decay.mem",
    "parameters_weights.mem",
    "parameters_output_weights.mem",
];

/// A validated memory bank, including the readout image that is not part of
/// the repository's hidden-layer NIR topology.
#[derive(Debug, Clone, PartialEq)]
pub struct Q88MemBank {
    /// Hidden-layer model reconstructed from thresholds, decays, and weights.
    pub model: SnnModel,
    /// Neuron-major 16×3 decision-module weights.
    pub output_weights: Vec<f64>,
}

impl Q88MemBank {
    /// Read the four canonical files from `directory`.
    ///
    /// No JSON file is opened or embedded on this path.
    pub fn from_dir(directory: impl AsRef<Path>) -> Result<Self, MemBankError> {
        let directory = directory.as_ref();
        reject_unrecognized_images(directory)?;
        let thresholds = read_image(directory, 0, NEURON_COUNT)?;
        let decays = read_image(directory, 1, NEURON_COUNT)?;
        let weights = read_image(directory, 2, NEURON_COUNT * NEURON_COUNT)?;
        let output_weights = read_image(directory, 3, OUTPUT_WEIGHT_COUNT)?;

        for (line, &decay) in decays.iter().enumerate() {
            if !(decay > 0.0 && decay < 1.0) {
                return Err(MemBankError::Value {
                    path: directory.join(MEM_BANK_FILENAMES[1]),
                    line: line + 1,
                    message: format!("decay {decay} is outside the required open interval (0, 1)"),
                });
            }
        }

        let neurons = (0..NEURON_COUNT)
            .map(|row| Neuron {
                decay_rate: decays[row],
                membrane_potential: 0.0,
                threshold: thresholds[row],
                last_spike: false,
                weights: weights[row * NEURON_COUNT..(row + 1) * NEURON_COUNT].to_vec(),
            })
            .collect();
        Ok(Self {
            model: SnnModel { neurons },
            output_weights,
        })
    }
}

fn reject_unrecognized_images(directory: &Path) -> Result<(), MemBankError> {
    let entries = std::fs::read_dir(directory).map_err(|source| MemBankError::Io {
        path: directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| MemBankError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.extension().is_some_and(|extension| extension == "mem")
            && !MEM_BANK_FILENAMES
                .iter()
                .any(|expected| path.file_name().is_some_and(|name| name == *expected))
        {
            return Err(MemBankError::UnrecognizedImage { path });
        }
    }
    Ok(())
}

fn read_image(
    directory: &Path,
    file_index: usize,
    expected: usize,
) -> Result<Vec<f64>, MemBankError> {
    let path = directory.join(MEM_BANK_FILENAMES[file_index]);
    let bytes = std::fs::read(&path).map_err(|source| MemBankError::Io {
        path: path.clone(),
        source,
    })?;
    let text = String::from_utf8(bytes).map_err(|source| MemBankError::Utf8 {
        path: path.clone(),
        source,
    })?;

    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut values = Vec::new();
    for (line_index, token) in normalized.lines().enumerate() {
        if token.len() != 4
            || !token
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte))
        {
            return Err(MemBankError::Token {
                path,
                line: line_index + 1,
                token: token.to_owned(),
            });
        }
        let bits = u16::from_str_radix(token, 16).expect("four validated hexadecimal digits");
        values.push(f64::from(bits.cast_signed()) / Q8_8_SCALE);
    }
    if values.len() != expected {
        return Err(MemBankError::Length {
            path,
            expected,
            actual: values.len(),
        });
    }
    Ok(values)
}

/// Location-aware failure while decoding a Q8.8 memory bank.
#[derive(Debug)]
#[non_exhaustive]
pub enum MemBankError {
    /// A required canonical file could not be read.
    Io {
        /// Canonical path that failed.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// A required file was not UTF-8 text.
    Utf8 {
        /// Canonical path that failed.
        path: PathBuf,
        /// Underlying UTF-8 error.
        source: std::string::FromUtf8Error,
    },
    /// A `.mem` image outside the canonical four-file bank was present.
    UnrecognizedImage {
        /// Unexpected image path.
        path: PathBuf,
    },
    /// A line was not exactly four uppercase hexadecimal digits.
    Token {
        /// Canonical image path.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// Rejected line contents.
        token: String,
    },
    /// An image had the wrong number of words.
    Length {
        /// Canonical image path.
        path: PathBuf,
        /// Required number of words.
        expected: usize,
        /// Observed number of words.
        actual: usize,
    },
    /// A decoded word violated a semantic constraint.
    Value {
        /// Canonical image path.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// Semantic failure description.
        message: String,
    },
}

impl fmt::Display for MemBankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "read {}: {source}", path.display()),
            Self::Utf8 { path, source } => {
                write!(f, "decode {} as UTF-8: {source}", path.display())
            }
            Self::UnrecognizedImage { path } => {
                write!(f, "unrecognized memory image {}", path.display())
            }
            Self::Token { path, line, token } => write!(
                f,
                "{}:{line}: expected four uppercase hexadecimal digits, got {token:?}",
                path.display()
            ),
            Self::Length {
                path,
                expected,
                actual,
            } => write!(
                f,
                "{}: expected {expected} words, got {actual}",
                path.display()
            ),
            Self::Value {
                path,
                line,
                message,
            } => write!(f, "{}:{line}: {message}", path.display()),
        }
    }
}

impl std::error::Error for MemBankError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Utf8 { source, .. } => Some(source),
            _ => None,
        }
    }
}

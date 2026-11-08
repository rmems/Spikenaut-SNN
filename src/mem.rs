// SPDX-License-Identifier: MIT OR Apache-2.0

//! Strict decoder for the four-file, signed Q8.8 FPGA memory bank.
//!
//! This is deliberately independent of `snn_model.json`.  File names, line
//! layout, token spelling, dimensions, and the LIF decay domain are part of the
//! boundary: accepting a vaguely similar text file would make the provenance
//! claim unauditable.

use std::fmt;
use std::io::Read;
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

/// Required word count of each canonical image, in [`MEM_BANK_FILENAMES`]
/// order. The single source for both the read bound and the length check.
const MEM_BANK_WORDS: [usize; 4] = [
    NEURON_COUNT,
    NEURON_COUNT,
    NEURON_COUNT * NEURON_COUNT,
    OUTPUT_WEIGHT_COUNT,
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
    /// No JSON file is opened or embedded on this path. The `.mem` images
    /// carry no simulator state, so every neuron comes back at rest
    /// (`membrane_potential` 0, `last_spike` false).
    ///
    /// # Errors
    ///
    /// - [`MemBankError::Io`] if `directory` cannot be listed or a canonical
    ///   file cannot be read
    /// - [`MemBankError::UnrecognizedImage`] if `directory` holds any other
    ///   `.mem` file
    /// - [`MemBankError::TooLarge`] if an image is longer than its word count
    ///   allows, before it is decoded
    /// - [`MemBankError::Utf8`], [`MemBankError::Token`] or
    ///   [`MemBankError::Length`] if an image is not exactly its word count of
    ///   four-uppercase-hex-digit lines
    /// - [`MemBankError::Value`] if a decay word is outside `(0, 1)`
    pub fn from_dir(directory: impl AsRef<Path>) -> Result<Self, MemBankError> {
        let directory = directory.as_ref();
        reject_unrecognized_images(directory)?;
        let [thresholds, decays, weights, output_weights] = snapshot_canonical_images(directory)?;
        let thresholds = parse_image(directory, 0, thresholds)?;
        let decays = parse_image(directory, 1, decays)?;
        let weights = parse_image(directory, 2, weights)?;
        let output_weights = parse_image(directory, 3, output_weights)?;

        validate_decays(directory, &decays)?;

        // Lengths are checked by `parse_image`, so the zip is exact: one
        // threshold, one decay and one `NEURON_COUNT`-wide row per neuron.
        let neurons = thresholds
            .into_iter()
            .zip(decays)
            .zip(weights.as_chunks::<NEURON_COUNT>().0)
            .map(|((threshold, decay_rate), row)| Neuron {
                decay_rate,
                membrane_potential: 0.0,
                threshold,
                last_spike: false,
                weights: row.to_vec(),
            })
            .collect();
        Ok(Self {
            model: SnnModel { neurons },
            output_weights,
        })
    }
}

fn validate_decays(directory: &Path, decays: &[f64]) -> Result<(), MemBankError> {
    for (line, &decay) in decays.iter().enumerate() {
        if !(decay > 0.0 && decay < 1.0) {
            return Err(MemBankError::Value {
                path: directory.join(MEM_BANK_FILENAMES[1]),
                line: line + 1,
                message: format!("decay {decay} is outside the required open interval (0, 1)"),
            });
        }
    }
    Ok(())
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

const HEX_WORD_LEN: usize = 4;
const MAX_LINE_SUFFIX_BYTES: usize = 2;

fn max_canonical_image_bytes(expected_words: usize) -> usize {
    expected_words.saturating_mul(HEX_WORD_LEN + MAX_LINE_SUFFIX_BYTES)
}

/// Read all four canonical images, bounded, before any is decoded.
fn snapshot_canonical_images(directory: &Path) -> Result<[Vec<u8>; 4], MemBankError> {
    let mut snapshots: [Vec<u8>; 4] = Default::default();
    for (file_index, slot) in snapshots.iter_mut().enumerate() {
        *slot = read_bounded_image(directory, file_index)?;
    }
    Ok(snapshots)
}

fn read_bounded_image(directory: &Path, file_index: usize) -> Result<Vec<u8>, MemBankError> {
    let path = directory.join(MEM_BANK_FILENAMES[file_index]);
    let max_bytes = max_canonical_image_bytes(MEM_BANK_WORDS[file_index]);
    let file = std::fs::File::open(&path).map_err(|source| MemBankError::Io {
        path: path.clone(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| MemBankError::Io {
            path: path.clone(),
            source,
        })?;
    if bytes.len() > max_bytes {
        return Err(MemBankError::TooLarge {
            path,
            max_bytes,
            actual: bytes.len(),
        });
    }
    Ok(bytes)
}

/// Decode canonical image `file_index` of `directory` from its snapshot.
///
/// Takes the snapshot by value so the UTF-8 check reuses its buffer rather
/// than copying it.
fn parse_image(
    directory: &Path,
    file_index: usize,
    bytes: Vec<u8>,
) -> Result<Vec<f64>, MemBankError> {
    let path = directory.join(MEM_BANK_FILENAMES[file_index]);
    let expected = MEM_BANK_WORDS[file_index];
    let text = String::from_utf8(bytes).map_err(|source| MemBankError::Utf8 {
        path: path.clone(),
        source,
    })?;

    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut values = Vec::new();
    for (line_index, token) in normalized.lines().enumerate() {
        if token.len() != HEX_WORD_LEN
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
    /// An image exceeded the bounded size for its required word count.
    TooLarge {
        /// Canonical image path.
        path: PathBuf,
        /// Maximum permitted byte length.
        max_bytes: usize,
        /// Observed byte length.
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
            Self::TooLarge {
                path,
                max_bytes,
                actual,
            } => write!(
                f,
                "{}: expected at most {max_bytes} bytes, got {actual}",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Q8_8_MAX, Q8_8_MIN};

    fn shipped_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("dataset/merged_v2")
    }

    /// Decode `bytes` as canonical image `file_index` of a notional directory.
    fn parse(file_index: usize, bytes: impl Into<Vec<u8>>) -> Result<Vec<f64>, MemBankError> {
        parse_image(Path::new("bank"), file_index, bytes.into())
    }

    /// A threshold-sized image of `lines`, LF-separated, no final newline.
    fn sixteen_lines(lines: &[&str]) -> String {
        assert_eq!(lines.len(), NEURON_COUNT);
        lines.join("\n")
    }

    #[test]
    fn words_decode_as_signed_q8_8() {
        let mut lines = ["0000"; NEURON_COUNT];
        let words = [
            ("0001", 1.0 / 256.0),
            ("00DA", 218.0 / 256.0),
            ("0100", 1.0),
            ("7FFF", Q8_8_MAX),
            ("8000", Q8_8_MIN),
            ("FFFF", -1.0 / 256.0),
            ("FFF9", -7.0 / 256.0),
        ];
        for (slot, (word, _)) in words.iter().enumerate() {
            lines[slot] = word;
        }
        let values = parse(0, sixteen_lines(&lines)).unwrap();
        for (slot, (word, expected)) in words.iter().enumerate() {
            assert_eq!(values[slot], *expected, "{word}");
        }
        assert!(values[words.len()..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn every_line_ending_and_a_missing_final_newline_are_accepted() {
        let lf = format!("{}\n", sixteen_lines(&["0100"; NEURON_COUNT]));
        for text in [
            lf.clone(),
            lf.replace('\n', "\r\n"),
            lf.replace('\n', "\r"),
            lf.trim_end().to_owned(),
        ] {
            assert_eq!(parse(0, text).unwrap(), vec![1.0; NEURON_COUNT]);
        }
    }

    /// Token spelling is part of the boundary: only four uppercase ASCII hex
    /// digits, unpadded. A multi-byte character must be refused, not panic on
    /// a byte-length check that happens to come out at four.
    #[test]
    fn a_malformed_token_is_reported_with_its_line() {
        for bad in [
            "00da",
            " 000",
            "0000 ",
            "000",
            "00000",
            "+001",
            "-001",
            "00G0",
            "0x00",
            "",
            "é00",
            "\u{FF10}000",
        ] {
            let mut lines = ["0000"; NEURON_COUNT];
            lines[2] = bad;
            match parse(0, sixteen_lines(&lines)) {
                Err(MemBankError::Token { line, token, .. }) => {
                    assert_eq!(line, 3, "{bad:?}");
                    assert_eq!(token, bad);
                }
                other => panic!("{bad:?}: expected a token error, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_word_count_must_be_exact() {
        for count in [0, NEURON_COUNT - 1, NEURON_COUNT + 1] {
            match parse(0, "0000\n".repeat(count)) {
                Err(MemBankError::Length {
                    expected, actual, ..
                }) => {
                    assert_eq!(expected, NEURON_COUNT);
                    assert_eq!(actual, count);
                }
                other => panic!("{count} words: expected a length error, got {other:?}"),
            }
        }

        // A trailing blank line is a malformed line, not a free pass.
        let padded = format!("{}\n\n", sixteen_lines(&["0000"; NEURON_COUNT]));
        assert!(matches!(
            parse(0, padded),
            Err(MemBankError::Token { line: 17, .. })
        ));
    }

    #[test]
    fn a_non_utf8_image_is_refused() {
        let mut bytes = sixteen_lines(&["0000"; NEURON_COUNT]).into_bytes();
        bytes[1] = 0xFF;
        assert!(matches!(parse(0, bytes), Err(MemBankError::Utf8 { .. })));
    }

    /// The bound leaves room for CRLF on every line and nothing more, and does
    /// not overflow for an absurd word count.
    #[test]
    fn the_size_bound_admits_crlf_and_saturates() {
        let crlf = "0000\r\n".repeat(NEURON_COUNT);
        assert_eq!(crlf.len(), max_canonical_image_bytes(NEURON_COUNT));
        parse(0, crlf).expect("a fully CRLF-terminated image is in bounds");
        assert_eq!(max_canonical_image_bytes(usize::MAX), usize::MAX);
        assert_eq!(
            MEM_BANK_WORDS.map(max_canonical_image_bytes),
            [96, 96, 1536, OUTPUT_WEIGHT_COUNT * 6],
        );
    }

    #[test]
    fn decays_must_lie_strictly_inside_the_unit_interval() {
        let directory = Path::new("bank");
        validate_decays(directory, &[1.0 / 256.0, 0.5, 255.0 / 256.0]).unwrap();

        for bad in [0.0, -0.0, 1.0, -1.0 / 256.0, Q8_8_MAX, Q8_8_MIN] {
            let mut decays = [0.5; NEURON_COUNT];
            decays[4] = bad;
            match validate_decays(directory, &decays) {
                Err(MemBankError::Value {
                    path,
                    line,
                    message,
                }) => {
                    assert_eq!(line, 5, "{bad}");
                    assert!(path.ends_with(MEM_BANK_FILENAMES[1]), "{}", path.display());
                    assert!(message.contains("(0, 1)"), "{message}");
                }
                other => panic!("{bad}: expected a value error, got {other:?}"),
            }
        }
    }

    /// The `.mem` bank and the embedded JSON are the same parameters.
    #[test]
    fn the_shipped_bank_matches_the_shipped_json() {
        let bank = Q88MemBank::from_dir(shipped_dir()).unwrap();
        let json = SnnModel::load_default().unwrap();
        assert_eq!(bank.model.len(), NEURON_COUNT);
        assert_eq!(bank.model.thresholds(), json.thresholds());
        assert_eq!(bank.model.decay_rates(), json.decay_rates());
        for (from_mem, from_json) in bank.model.neurons.iter().zip(&json.neurons) {
            assert_eq!(from_mem.weights, from_json.weights);
            // The images carry no simulator state.
            assert_eq!(from_mem.membrane_potential, 0.0);
            assert!(!from_mem.last_spike);
        }
        assert_eq!(bank.output_weights.len(), OUTPUT_WEIGHT_COUNT);
    }

    #[test]
    fn a_missing_directory_or_file_names_its_path() {
        let root = tempfile::tempdir().unwrap();
        let absent = root.path().join("absent");
        match Q88MemBank::from_dir(&absent) {
            Err(MemBankError::Io { path, .. }) => assert_eq!(path, absent),
            other => panic!("expected an I/O error, got {other:?}"),
        }

        // An empty directory reports the first canonical file it needed.
        match Q88MemBank::from_dir(root.path()) {
            Err(MemBankError::Io { path, .. }) => {
                assert!(path.ends_with(MEM_BANK_FILENAMES[0]), "{}", path.display());
            }
            other => panic!("expected an I/O error, got {other:?}"),
        }
    }

    #[test]
    fn a_copied_bank_ignores_other_files_and_refuses_a_zero_decay() {
        let root = tempfile::tempdir().unwrap();
        for name in MEM_BANK_FILENAMES {
            std::fs::copy(shipped_dir().join(name), root.path().join(name)).unwrap();
        }
        // Not a `.mem` image, so not part of the bank's boundary.
        std::fs::write(root.path().join("notes.txt"), "not an image").unwrap();
        assert_eq!(
            Q88MemBank::from_dir(root.path()).unwrap(),
            Q88MemBank::from_dir(shipped_dir()).unwrap(),
        );

        let decay = root.path().join(MEM_BANK_FILENAMES[1]);
        let text = std::fs::read_to_string(&decay).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        lines[6] = "0000";
        std::fs::write(&decay, lines.join("\n")).unwrap();
        match Q88MemBank::from_dir(root.path()) {
            Err(error @ MemBankError::Value { line: 7, .. }) => {
                let message = error.to_string();
                assert!(message.contains("parameters_decay.mem:7"), "{message}");
            }
            other => panic!("expected a decay value error, got {other:?}"),
        }
    }

    #[test]
    fn errors_render_their_location_and_chain_their_source() {
        use std::error::Error as _;

        let path = Path::new("bank").join(MEM_BANK_FILENAMES[0]);
        let token = MemBankError::Token {
            path: path.clone(),
            line: 3,
            token: "00da".into(),
        };
        assert_eq!(
            token.to_string(),
            format!(
                "{}:3: expected four uppercase hexadecimal digits, got \"00da\"",
                path.display()
            ),
        );
        assert!(token.source().is_none());

        let length = MemBankError::Length {
            path: path.clone(),
            expected: 16,
            actual: 15,
        };
        assert_eq!(
            length.to_string(),
            format!("{}: expected 16 words, got 15", path.display()),
        );

        let utf8 = parse(0, vec![0xFF]).unwrap_err();
        assert!(utf8.to_string().contains("as UTF-8"), "{utf8}");
        assert!(utf8.source().is_some());

        let unrecognized = MemBankError::UnrecognizedImage { path };
        assert!(
            unrecognized
                .to_string()
                .starts_with("unrecognized memory image")
        );
        assert!(unrecognized.source().is_none());
    }
}

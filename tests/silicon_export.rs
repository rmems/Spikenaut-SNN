// SPDX-License-Identifier: MIT OR Apache-2.0

//! Contract tests for the published silicon-bridge export adapter.

use silicon_bridge::ParameterShapeError;
use spikenaut_snn::{
    FPGA_MEM_FILENAMES, ModelError, ShippedFpgaImage, SiliconExportError, export_shipped_fpga_image,
};
use std::error::Error;
use std::path::Path;

#[test]
fn mem_files_render_signed_words_as_twos_complement_hex() {
    let image = ShippedFpgaImage {
        thresholds: vec![i16::MIN, -1, 0, i16::MAX],
        weights: vec![0x0100, -0x0100],
        decay_rates: vec![0x00DA],
        output_weights: Vec::new(),
        inhibitory: Vec::new(),
    };
    let [thresholds, weights, decay, readout] = image.mem_files();

    assert_eq!(thresholds.name, "parameters.mem");
    assert_eq!(thresholds.contents, "8000\nFFFF\n0000\n7FFF\n");
    assert_eq!(weights.contents, "0100\nFF00\n");
    assert_eq!(decay.contents, "00DA\n");
    assert_eq!(readout.name, "parameters_output_weights.mem");
    assert_eq!(readout.contents, "", "an empty block renders no lines");
}

#[test]
fn every_exported_line_is_one_uppercase_four_digit_word() {
    let image = export_shipped_fpga_image().expect("the shipped model is exportable");
    let words = [
        image.thresholds.len(),
        image.weights.len(),
        image.decay_rates.len(),
        image.output_weights.len(),
    ];
    for (file, expected_lines) in image.mem_files().into_iter().zip(words) {
        assert!(
            file.contents.ends_with('\n'),
            "{} lacks a final newline",
            file.name
        );
        let lines: Vec<&str> = file.contents.lines().collect();
        assert_eq!(lines.len(), expected_lines, "{}", file.name);
        for line in lines {
            assert!(
                line.len() == 4
                    && line
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte)),
                "{}: {line:?} is not a 4-digit uppercase hex word",
                file.name,
            );
        }
    }
}

#[test]
fn export_is_deterministic() {
    assert_eq!(
        export_shipped_fpga_image().expect("first export"),
        export_shipped_fpga_image().expect("second export"),
    );
}

#[test]
fn export_errors_name_their_layer_and_chain_their_source() {
    let model = SiliconExportError::from(ModelError::Schema("bad readout".to_owned()));
    assert_eq!(
        model.to_string(),
        "cannot decode FPGA model: invalid SNN model: bad readout"
    );
    assert!(
        model
            .source()
            .is_some_and(|source| source.is::<ModelError>())
    );

    let parameters = SiliconExportError::from(ParameterShapeError::EmptyLayer);
    assert!(
        parameters
            .to_string()
            .starts_with("invalid FPGA parameter image: "),
        "{parameters}",
    );
    assert!(
        parameters
            .source()
            .is_some_and(|source| source.is::<ParameterShapeError>())
    );
}

#[test]
fn checked_export_reproduces_every_shipped_mem_file_byte_for_byte() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let image = export_shipped_fpga_image().expect("the shipped model is exportable");
    let generated = image.mem_files();

    assert_eq!(
        generated.each_ref().map(|file| file.name),
        FPGA_MEM_FILENAMES
    );
    for file in generated {
        let path = root.join("dataset/merged_v2").join(file.name);
        let committed = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(file.contents, committed, "{} drifted", file.name);
    }
}

#[test]
fn signed_words_and_neuron_major_readout_order_survive_the_adapter() {
    let image = export_shipped_fpga_image().expect("the shipped model is exportable");

    assert_eq!(image.thresholds.len(), 16);
    assert_eq!(image.weights.len(), 16 * 16);
    assert_eq!(image.decay_rates.len(), 16);
    assert_eq!(image.output_weights.len(), 16 * 3);
    assert_eq!(
        image.inhibitory,
        [false; 12].into_iter().chain([true; 4]).collect::<Vec<_>>()
    );

    // Hidden row 6 begins with approximately -1.0. Signed Q8.8 must preserve
    // that as two's-complement FF00 rather than silently unsigned-clamping it.
    assert_eq!(image.weights[6 * 16] as u16, 0xFF00);

    // The vault and silicon-hdl address the readout N×K (neuron-major). The
    // published exporter validates K×N, so this pins both adapter transposes:
    // neuron 12's classes remain adjacent at addresses 36, 37, and 38.
    assert_eq!(image.output_weights[12 * 3] as u16, 0x0000);
    assert_eq!(image.output_weights[12 * 3 + 1] as u16, 0xFFF7);
    assert_eq!(image.output_weights[12 * 3 + 2] as u16, 0xFFE9);
}

#[test]
fn silicon_bridge_resolves_from_crates_io_at_zero_three() {
    let lock = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))
        .expect("read Cargo.lock");
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains("name = \"silicon-bridge\""))
        .expect("Cargo.lock has a silicon-bridge package entry");

    assert!(
        entry.contains("version = \"0.3."),
        "silicon-bridge must resolve to 0.3.x, got:\n{entry}",
    );
    assert!(
        entry.contains("source = \"registry+https://github.com/rust-lang/crates.io-index\""),
        "silicon-bridge must resolve from crates.io, got:\n{entry}",
    );
}

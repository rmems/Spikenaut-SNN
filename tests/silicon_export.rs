// SPDX-License-Identifier: MIT OR Apache-2.0

//! Contract tests for the published silicon-bridge export adapter.

use spikenaut_snn::{FPGA_MEM_FILENAMES, export_shipped_fpga_image};
use std::path::Path;

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

// SPDX-License-Identifier: MIT OR Apache-2.0

//! `config.json` agrees with the crate and with the shipped artifacts.
//!
//! `config.json` is model-card metadata: it is what a Hugging Face consumer
//! reads to learn the model's shape without cloning the weights. Nothing in
//! this repository loaded it before this file existed -- not the crate, not
//! `tools/` -- so every value in it was free to drift away from the constants
//! and artifacts it describes, silently and indefinitely. One had:
//! `memory_bytes` read `1638` against an artifact set of 336 Q8.8 codes, which
//! is 672 bytes, and nothing anywhere compared the two.
//!
//! These tests make the file load-bearing. A value that disagrees with the
//! code or with the `.mem` files is now a failing test rather than a stale
//! line nobody reads.
//!
//! They also exercise [`spikenaut_snn::json`] against a second real document.
//! Until now its only production input was `snn_model.json`, which is one
//! shape: an object holding an array of uniform objects. `config.json` has
//! string values, an array of strings, an integer and a fractional number at
//! the top level, so it covers grammar the model loader never reaches.

use std::path::Path;

use spikenaut_snn::CHANNEL_COUNT;
use spikenaut_snn::json::{self, Json};
use spikenaut_snn::model::{CLOCK_HZ, NEURON_COUNT, Q8_8_SCALE};

/// Every `.mem` artifact in `dataset/merged_v2`, by name.
///
/// Listed rather than globbed: a glob would silently shrink if an artifact
/// were removed, and this test would then pass by measuring less.
const MEM_ARTIFACTS: [&str; 4] = [
    "parameters.mem",
    "parameters_decay.mem",
    "parameters_weights.mem",
    "parameters_output_weights.mem",
];

/// Bytes per Q8.8 code: eight integer bits and eight fractional bits.
const BYTES_PER_Q8_8_CODE: usize = 2;

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Parse `config.json` with the crate's own reader.
fn config() -> Json {
    let path = repo_root().join("config.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    json::parse(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// A numeric member, or a panic naming the member that was missing or of the
/// wrong type.
///
/// Deliberately not an `Option`-returning helper: a missing key must fail the
/// test, never skip the assertion that depended on it.
fn number(config: &Json, key: &str) -> f64 {
    let value = config
        .get(key)
        .unwrap_or_else(|| panic!("config.json has no `{key}`"));
    value.as_f64().unwrap_or_else(|| {
        panic!(
            "config.json `{key}` is a {}, expected a number",
            value.type_name()
        )
    })
}

/// A string member, with the same failure discipline as [`number`].
fn string<'a>(config: &'a Json, key: &str) -> &'a str {
    let value = config
        .get(key)
        .unwrap_or_else(|| panic!("config.json has no `{key}`"));
    value.as_str().unwrap_or_else(|| {
        panic!(
            "config.json `{key}` is a {}, expected a string",
            value.type_name()
        )
    })
}

/// Count the Q8.8 codes in one `.mem` artifact.
fn mem_code_count(name: &str) -> usize {
    let path = repo_root().join("dataset/merged_v2").join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.split_whitespace().count()
}

/// The population width is stated in three places; they must agree.
#[test]
fn n_neurons_matches_the_crate() {
    let config = config();
    let stated = number(&config, "n_neurons");
    assert_eq!(
        stated, NEURON_COUNT as f64,
        "config.json `n_neurons` ({stated}) disagrees with NEURON_COUNT ({NEURON_COUNT})",
    );
}

/// The input width drives the encoder, not just the documentation.
#[test]
fn n_channels_matches_the_encoder() {
    let config = config();
    let stated = number(&config, "n_channels");
    assert_eq!(
        stated, CHANNEL_COUNT as f64,
        "config.json `n_channels` ({stated}) disagrees with CHANNEL_COUNT ({CHANNEL_COUNT})",
    );
}

/// The clock sets the timestep every decay rate is inverted against, so a
/// drifted value here would describe a model with different time constants
/// from the one the crate builds.
#[test]
fn clock_hz_matches_the_crate() {
    let config = config();
    let stated = number(&config, "clock_hz");
    assert_eq!(
        stated, CLOCK_HZ,
        "config.json `clock_hz` ({stated}) disagrees with CLOCK_HZ ({CLOCK_HZ})",
    );
}

/// `memory_bytes` must be the size of the artifacts actually shipped.
///
/// This is the value that had drifted: the file claimed 1638 bytes against
/// 336 Q8.8 codes, which is 672. Deriving the expected number from the `.mem`
/// files rather than hard-coding 672 means the assertion keeps holding if the
/// artifact set legitimately changes -- a retrain that adds a bias file, say --
/// and fails only when `config.json` stops describing what is on disk.
#[test]
fn memory_bytes_matches_the_shipped_artifacts() {
    let codes: usize = MEM_ARTIFACTS.iter().copied().map(mem_code_count).sum();

    // Non-vacuity: an empty or truncated artifact set must fail here rather
    // than agree with a `memory_bytes` of zero.
    assert!(
        codes >= NEURON_COUNT * NEURON_COUNT,
        "only {codes} Q8.8 codes found across {} artifacts; the hidden weight \
         matrix alone is {}",
        MEM_ARTIFACTS.len(),
        NEURON_COUNT * NEURON_COUNT,
    );

    let expected = codes * BYTES_PER_Q8_8_CODE;
    let stated = number(&config(), "memory_bytes");
    assert_eq!(
        stated, expected as f64,
        "config.json `memory_bytes` ({stated}) disagrees with the shipped \
         artifacts ({codes} Q8.8 codes x {BYTES_PER_Q8_8_CODE} bytes = {expected})",
    );
}

/// The declared weight format must be the one the crate actually implements.
#[test]
fn weight_format_is_the_q8_8_grid_the_crate_uses() {
    assert_eq!(
        string(&config(), "weight_format"),
        "q8.8-fixed-point",
        "config.json `weight_format` must name the grid `Q8_8_SCALE` ({Q8_8_SCALE}) implements",
    );
}

/// `externally_reported` must name members that actually exist.
///
/// The list marks the values this repository cannot verify -- the training
/// figures and the FPGA power number, none of which any run here produces.
/// That disclosure is only worth anything if it stays attached to real keys,
/// so a rename that orphaned an entry would quietly un-qualify a claim.
#[test]
fn externally_reported_names_real_members() {
    let config = config();
    let listed = config
        .get("externally_reported")
        .expect("config.json has no `externally_reported`")
        .as_array()
        .expect("config.json `externally_reported` must be an array");

    assert!(
        !listed.is_empty(),
        "an empty `externally_reported` list would claim everything is verified here",
    );

    for entry in listed {
        let key = entry.as_str().unwrap_or_else(|| {
            panic!(
                "`externally_reported` holds a {}, expected a string",
                entry.type_name()
            )
        });
        assert!(
            config.get(key).is_some(),
            "`externally_reported` names `{key}`, which config.json does not define",
        );
    }
}

/// The helpers must fail loudly on a missing member.
///
/// Without this, a future edit that renamed a key would turn every assertion
/// above into a silent skip, and the drift these tests exist to catch would
/// go back to being invisible.
#[test]
fn a_missing_member_is_a_failure_not_a_skip() {
    let config = config();
    assert!(
        config.get("no_such_member_exists").is_none(),
        "the fixture for this test must name a member config.json does not have",
    );

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        number(&config, "no_such_member_exists")
    }))
    .is_err();
    assert!(
        panicked,
        "`number` must panic on a missing member, not return a default"
    );
}

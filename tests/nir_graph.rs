// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `nir-rs` integration: the shipped `merged_v2` model must
//! build a valid 16-LIF NIR graph, and `nir-rs` must resolve from crates.io.

use std::path::{Path, PathBuf};

use nir_rs::types::{MetadataValue, TensorData};
use nir_rs::{NirGraph, NirNode};
use spikenaut_snn::graph::{INPUT_NODE, LIF_NODE, OUTPUT_NODE, load_default_lif_graph};
use spikenaut_snn::model::{MERGED_V2_PROVENANCE, NEURON_COUNT, SnnModel, TIMESTEP_SECONDS};

/// The 16-LIF graph is `Input → LIF → Output`: three nodes, two edges.
#[test]
fn graph_has_three_nodes_and_two_edges() {
    let graph = load_default_lif_graph().expect("build the 16-LIF graph");

    assert_eq!(graph.len(), 3, "expected Input, LIF and Output nodes");
    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(graph.edges.len(), 2, "expected Input→LIF and LIF→Output");

    assert_eq!(
        graph.nodes.keys().map(String::as_str).collect::<Vec<_>>(),
        [INPUT_NODE, LIF_NODE, OUTPUT_NODE],
    );
    assert_eq!(
        graph.edges,
        [
            (INPUT_NODE.to_owned(), LIF_NODE.to_owned()),
            (LIF_NODE.to_owned(), OUTPUT_NODE.to_owned()),
        ],
    );

    // NIR wire type names, not informal aliases.
    assert_eq!(graph.get(INPUT_NODE).unwrap().type_name(), "Input");
    assert_eq!(graph.get(LIF_NODE).unwrap().type_name(), "LIF");
    assert_eq!(graph.get(OUTPUT_NODE).unwrap().type_name(), "Output");

    graph
        .validate_structure()
        .expect("structurally valid graph");

    assert_eq!(
        graph.metadata.get("provenance"),
        Some(&MetadataValue::String(MERGED_V2_PROVENANCE.to_string())),
        "graph must stamp the shipped merged_v2 provenance",
    );
}

/// Every node carries all 16 units, and every LIF parameter is a 16-element vector.
#[test]
fn graph_carries_sixteen_units() {
    let graph = load_default_lif_graph().expect("build the 16-LIF graph");

    let Some(NirNode::Input(input)) = graph.get(INPUT_NODE) else {
        panic!("expected an Input node");
    };
    let Some(NirNode::Output(output)) = graph.get(OUTPUT_NODE) else {
        panic!("expected an Output node");
    };
    assert_eq!(input.shape, [NEURON_COUNT]);
    assert_eq!(output.shape, [NEURON_COUNT]);

    let Some(NirNode::Lif(lif)) = graph.get(LIF_NODE) else {
        panic!("expected a LIF node");
    };
    for (name, tensor) in [
        ("tau", &lif.tau),
        ("r", &lif.r),
        ("v_leak", &lif.v_leak),
        ("v_threshold", &lif.v_threshold),
    ] {
        assert_eq!(tensor.shape(), [NEURON_COUNT], "{name} shape");
        assert_eq!(tensor.numel(), NEURON_COUNT, "{name} length");
    }
    assert!(lif.v_reset.is_none(), "v_reset defaults to zeros in NIR");
}

/// Thresholds are copied verbatim and decay rates invert into positive time
/// constants that reproduce the stored per-step decay.
#[test]
fn lif_parameters_round_trip_the_model() {
    let model = SnnModel::load_default().expect("load merged_v2");
    let graph = spikenaut_snn::build_lif_graph(&model).expect("build the 16-LIF graph");

    let Some(NirNode::Lif(lif)) = graph.get(LIF_NODE) else {
        panic!("expected a LIF node");
    };
    let TensorData::F64(thresholds) = lif.v_threshold.data() else {
        panic!("expected an f64 v_threshold");
    };
    assert_eq!(thresholds, &model.thresholds());

    let TensorData::F64(taus) = lif.tau.data() else {
        panic!("expected an f64 tau");
    };
    for (tau, decay) in taus.iter().zip(model.decay_rates()) {
        assert!(*tau > 0.0 && tau.is_finite(), "tau must be positive");
        assert!(
            ((-TIMESTEP_SECONDS / tau).exp() - decay).abs() < 1e-12,
            "tau must reproduce the stored decay rate",
        );
    }
}

/// The recurrent weight matrix survives loading, even though this graph does
/// not place it on an edge.
#[test]
fn recurrent_weights_load_as_a_square_tensor() {
    let model = SnnModel::load_default().expect("load merged_v2");
    let weights = model.weight_tensor().expect("build the weight tensor");
    assert_eq!(weights.shape(), [NEURON_COUNT, NEURON_COUNT]);
    assert_eq!(weights.numel(), NEURON_COUNT * NEURON_COUNT);
}

/// A hand-built reference graph of the same shape must match ours, proving we
/// are exercising the real `nir-rs` graph API rather than a local stand-in.
#[test]
fn matches_a_hand_built_nir_graph() {
    use nir_rs::nodes::{Input, Output};

    let ours = load_default_lif_graph().expect("build the 16-LIF graph");

    let mut reference = NirGraph::new();
    reference
        .insert_node(
            INPUT_NODE,
            NirNode::Input(Input {
                shape: vec![NEURON_COUNT],
                metadata: Default::default(),
            }),
        )
        .unwrap();
    reference
        .insert_node(LIF_NODE, ours.get(LIF_NODE).unwrap().clone())
        .unwrap();
    reference
        .insert_node(
            OUTPUT_NODE,
            NirNode::Output(Output {
                shape: vec![NEURON_COUNT],
                metadata: Default::default(),
            }),
        )
        .unwrap();
    reference.add_edge(INPUT_NODE, LIF_NODE);
    reference.add_edge(LIF_NODE, OUTPUT_NODE);
    reference.metadata = ours.metadata.clone();

    assert_eq!(ours, reference);
}

/// Acceptance criterion from issue #8: `nir-rs` resolves from crates.io, not
/// from a git or sibling-path pin.
#[test]
fn nir_rs_resolves_from_crates_io() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Every dependency table, comments stripped. Scoping to `*dependencies*`
    // sections keeps `[lib] path` and prose about the rule out of the check;
    // the manifest has no `#` inside a string, so cutting at the first one is
    // exact.
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    let mut in_dependencies = false;
    let mut dependency_lines: Vec<&str> = Vec::new();
    for line in manifest.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some(header) = line.strip_prefix('[') {
            in_dependencies = header.trim_end_matches(']').contains("dependencies");
        } else if in_dependencies && !line.is_empty() {
            dependency_lines.push(line);
        }
    }
    let dependencies = dependency_lines.join("\n");

    for forbidden in ["git =", "path =", "git=", "path="] {
        assert!(
            !dependencies.contains(forbidden),
            "no dependency may be pinned with `{forbidden}`, found in:\n{dependencies}",
        );
    }
    // The allowed set is exact, so a third dependency (issue #9 added
    // `axon-encoder`) or a rename still fails here.
    let mut names: Vec<&str> = dependency_lines
        .iter()
        .map(|line| line.split('=').next().unwrap_or("").trim())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["axon-encoder", "nir-rs"],
        "unexpected dependency set:\n{dependencies}",
    );

    let lock_path: PathBuf = root.join("Cargo.lock");
    let lock = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", lock_path.display()));
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "nir-rs""#))
        .expect("Cargo.lock has a nir-rs package entry");

    assert!(
        entry.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "nir-rs must come from the crates.io registry, got:\n{entry}",
    );
    assert!(
        entry.contains(r#"version = "0.4."#),
        "nir-rs must resolve to 0.4.x, got:\n{entry}",
    );
}

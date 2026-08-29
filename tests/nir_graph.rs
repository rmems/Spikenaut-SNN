// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `nir-rs` integration: the shipped `merged_v2` model must
//! build a valid 16-LIF NIR graph, and `nir-rs` must resolve from crates.io.

use std::path::{Path, PathBuf};

use nir_rs::types::{MetadataValue, TensorData};
use nir_rs::{NirGraph, NirNode};
use spikenaut_snn::graph::{
    INPUT_NODE, LIF_NODE, LINEAR_NODE, OUTPUT_NODE, load_default_lif_graph,
};
use spikenaut_snn::model::{MERGED_V2_PROVENANCE, NEURON_COUNT, SnnModel, TIMESTEP_SECONDS};

/// The 16-LIF graph is `Input → Linear → LIF → Output`: four nodes, three edges.
#[test]
fn graph_has_four_nodes_and_three_edges() {
    let graph = load_default_lif_graph().expect("build the 16-LIF graph");

    assert_eq!(
        graph.len(),
        4,
        "expected Input, Linear, LIF and Output nodes"
    );
    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(
        graph.edges.len(),
        3,
        "expected Input→Linear, Linear→LIF and LIF→Output",
    );

    assert_eq!(
        graph.nodes.keys().map(String::as_str).collect::<Vec<_>>(),
        [INPUT_NODE, LINEAR_NODE, LIF_NODE, OUTPUT_NODE],
    );
    assert_eq!(
        graph.edges,
        [
            (INPUT_NODE.to_owned(), LINEAR_NODE.to_owned()),
            (LINEAR_NODE.to_owned(), LIF_NODE.to_owned()),
            (LIF_NODE.to_owned(), OUTPUT_NODE.to_owned()),
        ],
    );

    // NIR wire type names, not informal aliases.
    assert_eq!(graph.get(INPUT_NODE).unwrap().type_name(), "Input");
    assert_eq!(graph.get(LINEAR_NODE).unwrap().type_name(), "Linear");
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

/// All 256 learned weights are on the `Linear` node, so the graph is a
/// function of what the model learned.
#[test]
fn learned_weights_sit_on_the_linear_node() {
    let model = SnnModel::load_default().expect("load merged_v2");
    let weights = model.weight_tensor().expect("build the weight tensor");
    assert_eq!(weights.shape(), [NEURON_COUNT, NEURON_COUNT]);
    assert_eq!(weights.numel(), NEURON_COUNT * NEURON_COUNT);

    let graph = spikenaut_snn::build_lif_graph(&model).expect("build the 16-LIF graph");
    let Some(NirNode::Linear(linear)) = graph.get(LINEAR_NODE) else {
        panic!("expected a Linear node");
    };
    assert_eq!(linear.weight, weights);
}

/// Changing a single weight must change the graph. Before the `Linear` node
/// existed, changing all 256 produced a bit-for-bit identical graph.
#[test]
fn changing_a_weight_changes_the_graph() {
    let model = SnnModel::load_default().expect("load merged_v2");
    let baseline = spikenaut_snn::build_lif_graph(&model).expect("build the 16-LIF graph");

    let mut perturbed = model.clone();
    // One Q8.8 code: the smallest change the hardware can represent.
    perturbed.neurons[0].weights[0] += 1.0 / 256.0;

    let changed = spikenaut_snn::build_lif_graph(&perturbed).expect("build the perturbed graph");
    assert_ne!(changed, baseline);
    assert_ne!(changed.get(LINEAR_NODE), baseline.get(LINEAR_NODE));
}

/// Read a Q8.8 `.mem` artifact: one four-digit two's-complement hex code per
/// line, divided by 256.
fn read_q8_8_mem(name: &str) -> Vec<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("dataset/merged_v2")
        .join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.split_whitespace()
        .map(|code| {
            let bits = u16::from_str_radix(code, 16)
                .unwrap_or_else(|e| panic!("{}: bad code {code:?}: {e}", path.display()));
            f64::from(bits.cast_signed()) / 256.0
        })
        .collect()
}

/// The graph must carry exactly the parameters the FPGA holds.
///
/// `snn_model.json` prints truncated decimals — `0.808594` for `00CF`,
/// `0.7539062` for `00C1` — so the loader snaps every number back onto the Q8.8
/// grid. Without that, 139 of the 288 values would be off by up to half an LSB
/// and this test would fail.
#[test]
fn graph_parameters_match_the_q8_8_mem_artifacts() {
    let graph = load_default_lif_graph().expect("build the 16-LIF graph");
    let Some(NirNode::Linear(linear)) = graph.get(LINEAR_NODE) else {
        panic!("expected a Linear node");
    };
    let Some(NirNode::Lif(lif)) = graph.get(LIF_NODE) else {
        panic!("expected a LIF node");
    };

    let TensorData::F64(weights) = linear.weight.data() else {
        panic!("expected an f64 weight");
    };
    assert_eq!(weights, &read_q8_8_mem("parameters_weights.mem"));

    let TensorData::F64(thresholds) = lif.v_threshold.data() else {
        panic!("expected an f64 v_threshold");
    };
    assert_eq!(thresholds, &read_q8_8_mem("parameters.mem"));

    let TensorData::F64(taus) = lif.tau.data() else {
        panic!("expected an f64 tau");
    };
    let TensorData::F64(rs) = lif.r.data() else {
        panic!("expected an f64 r");
    };
    for ((tau, r), decay) in taus
        .iter()
        .zip(rs)
        .zip(read_q8_8_mem("parameters_decay.mem"))
    {
        // tau encodes the stored decay exactly...
        assert!(
            ((-TIMESTEP_SECONDS / tau).exp() - decay).abs() < 1e-12,
            "tau must reproduce the Q8.8 decay {decay}",
        );
        // ...and r is derived from that same exact value.
        assert!((r - 1.0 / (1.0 - decay)).abs() < 1e-12);
    }
}

/// NIR integrates `tau * dv/dt = (v_leak - v) + R*I`, so one step is
/// `v[t+1] = decay*v[t] + R*(1 - decay)*I`. The shipped fixed-point model steps
/// as `v[t+1] = decay*v[t] + I`, so `R` must cancel the `(1 - decay)` factor.
/// With `R = 1` the input would arrive at 5-20% of its trained magnitude.
///
/// The decay round-trip test above cannot catch this: it never drives an input.
#[test]
fn one_nir_step_reproduces_the_models_step() {
    let model = SnnModel::load_default().expect("load merged_v2");
    let graph = spikenaut_snn::build_lif_graph(&model).expect("build the 16-LIF graph");
    let Some(NirNode::Lif(lif)) = graph.get(LIF_NODE) else {
        panic!("expected a LIF node");
    };
    let TensorData::F64(taus) = lif.tau.data() else {
        panic!("expected an f64 tau");
    };
    let TensorData::F64(rs) = lif.r.data() else {
        panic!("expected an f64 r");
    };

    for (unit, ((tau, r), decay)) in taus.iter().zip(rs).zip(model.decay_rates()).enumerate() {
        let current = 0.75; // any input current
        let v = 0.5; // any starting membrane potential

        let step_decay = (-TIMESTEP_SECONDS / tau).exp();
        let nir = step_decay * v + r * (1.0 - step_decay) * current;
        let expected = decay * v + current;

        assert!(
            (nir - expected).abs() < 1e-12,
            "unit {unit}: NIR steps to {nir}, the model steps to {expected}",
        );
        // Not vacuous: R = 1 would drop 80-95% of the input.
        let unscaled = step_decay * v + (1.0 - step_decay) * current;
        assert!((unscaled - expected).abs() > 0.1 * current);
    }
}

/// A hand-built reference graph of the same shape must match ours, proving we
/// are exercising the real `nir-rs` graph API rather than a local stand-in.
#[test]
fn matches_a_hand_built_nir_graph() {
    use nir_rs::nodes::{Input, Linear, Output};

    let model = SnnModel::load_default().expect("load merged_v2");
    let ours = spikenaut_snn::build_lif_graph(&model).expect("build the 16-LIF graph");

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
        .insert_node(
            LINEAR_NODE,
            NirNode::Linear(Linear {
                weight: model.weight_tensor().expect("build the weight tensor"),
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
    reference.add_edge(INPUT_NODE, LINEAR_NODE);
    reference.add_edge(LINEAR_NODE, LIF_NODE);
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

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `nir-rs` integration: the shipped `merged_v2` model must
//! build a valid 16-LIF NIR graph, and `nir-rs` must resolve from crates.io.

use std::path::{Path, PathBuf};

use nir_rs::types::{MetadataValue, TensorData};
use nir_rs::{NirGraph, NirNode};
use spikenaut_snn::graph::{
    INPUT_NODE, LIF_NODE, LINEAR_NODE, OUTPUT_NODE, Provenance, load_default_lif_graph,
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

/// Provenance is a claim about which artifact the numbers came from, so only
/// `load_default_lif_graph` — which loads the artifact itself — may stamp it.
/// A perturbed or caller-built model used to come out labelled as the shipped
/// `merged_v2` artifact, which is exactly the mislabelled-experiment failure
/// the evidence discipline exists to prevent.
#[test]
fn only_the_shipped_artifact_is_labelled_as_the_shipped_artifact() {
    let model = SnnModel::load_default().expect("load merged_v2");

    // The one graph that may claim it.
    let shipped = load_default_lif_graph().expect("build the 16-LIF graph");
    assert_eq!(
        shipped.metadata.get("provenance"),
        Some(&MetadataValue::String(MERGED_V2_PROVENANCE.to_string())),
    );

    // A model the crate cannot vouch for gets no label — not a wrong one.
    let mut perturbed = model.clone();
    perturbed.neurons[0].weights[0] += 1.0 / 256.0;
    for unlabelled in [
        spikenaut_snn::build_lif_graph(&perturbed).expect("build the perturbed graph"),
        spikenaut_snn::build_lif_graph(&model).expect("build from a caller-held model"),
    ] {
        assert_eq!(
            unlabelled.metadata.get("provenance"),
            None,
            "only load_default_lif_graph may stamp the shipped provenance",
        );
        assert_eq!(unlabelled.metadata.get("source"), None);
        // The build parameters are still recorded.
        assert_eq!(
            unlabelled.metadata.get("timestep_seconds"),
            Some(&MetadataValue::F64(TIMESTEP_SECONDS)),
        );
    }

    // A caller who knows the origin supplies their own label.
    let labelled = spikenaut_snn::build_lif_graph_with_provenance(
        &perturbed,
        TIMESTEP_SECONDS,
        Some(Provenance {
            source: "exp-042/perturbed.json",
            description: "one weight bumped by a single Q8.8 code; not the shipped artifact",
        }),
    )
    .expect("build the labelled graph");
    let Some(MetadataValue::String(stamp)) = labelled.metadata.get("provenance") else {
        panic!("expected a provenance string");
    };
    assert_ne!(stamp, MERGED_V2_PROVENANCE);
    assert!(stamp.contains("not the shipped artifact"));
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
    // zip() stops at the shortest iterator, so without this the loop would run
    // fewer times and still pass if any of the three were short — a test that
    // silently checks less than it claims.
    let decays = read_q8_8_mem("parameters_decay.mem");
    assert_eq!(taus.len(), NEURON_COUNT, "one tau per neuron");
    assert_eq!(rs.len(), NEURON_COUNT, "one resistance per neuron");
    assert_eq!(decays.len(), NEURON_COUNT, "one decay per neuron");

    let mut checked = 0usize;
    for ((tau, r), decay) in taus.iter().zip(rs).zip(decays) {
        // tau encodes the stored decay exactly...
        assert!(
            ((-TIMESTEP_SECONDS / tau).exp() - decay).abs() < 1e-12,
            "tau must reproduce the Q8.8 decay {decay}",
        );
        // ...and r is derived from that same exact value.
        assert!((r - 1.0 / (1.0 - decay)).abs() < 1e-12);
        checked += 1;
    }
    assert_eq!(
        checked, NEURON_COUNT,
        "every neuron's decay must be checked"
    );
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

/// Split a `Cargo.toml` into the two scopes the pin check needs: every line of
/// every dependency table, and the names declared by the runtime
/// `[dependencies]` table alone.
///
/// The two claims are different sizes, so one scope cannot serve both.
///
/// The wide scope is every dependency table there is: `[dependencies]`,
/// `[dev-dependencies]`, `[build-dependencies]`, `[target.'cfg(..)'.dependencies]`
/// and the `[dependencies.<name>]` table form. A `git =` or `path =` pin is
/// forbidden in all of them, so that scan must not narrow.
///
/// The narrow scope is `[dependencies]` alone, in either spelling. Only that
/// table has to read exactly `nir-rs`: issue #8 asks for registry resolution,
/// not a ban on dev-dependencies, and a future test helper must not trip the
/// check with a message about nir-rs.
///
/// Comments are stripped first, which also keeps `[lib] path` and the prose
/// about this rule out of the scan; the manifest has no `#` inside a string, so
/// cutting at the first one is exact.
/// The dependency-table kind, with any `[target.<cfg or triple>.…]` prefix
/// removed.
///
/// Cargo nests dependency tables under `target.'cfg(...)'` and under bare
/// target triples, and a cfg expression can itself contain dots, so this
/// searches for the table name rather than splitting on the first separator.
/// `dev-` and `build-` tables keep their prefix, so they stay out of the
/// runtime set whether or not they are target-specific.
fn dependency_table_kind(section: &str) -> &str {
    let Some(rest) = section.strip_prefix("target.") else {
        return section;
    };
    for kind in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(at) = rest.rfind(&format!(".{kind}")) {
            return &rest[at + 1..];
        }
    }
    rest
}

fn dependency_tables(manifest: &str) -> (Vec<&str>, Vec<String>) {
    let mut section = String::new();
    let mut pinned: Vec<&str> = Vec::new();
    let mut runtime: Vec<String> = Vec::new();
    for line in manifest.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some(header) = line.strip_prefix('[') {
            section = header.trim_end_matches(']').to_string();
            // `[dependencies.nir-rs]` names its dependency in the header, so
            // the body below carries no `name =` line to pick it up from.
            if let Some(name) = dependency_table_kind(&section).strip_prefix("dependencies.") {
                runtime.push(name.to_string());
            }
            continue;
        }
        if line.is_empty() || !section.contains("dependencies") {
            continue;
        }
        pinned.push(line);
        if dependency_table_kind(&section) == "dependencies" {
            runtime.push(line.split('=').next().unwrap_or("").trim().to_string());
        }
    }
    (pinned, runtime)
}

/// The scope split above is only worth having if it actually holds on the
/// manifests it was written for, and the shipped `Cargo.toml` exercises none of
/// them: it has one table and one dependency.
///
/// Each case below is a manifest this repository does not have yet but could,
/// and each one is a way the check could go wrong — silently missing a pin, or
/// failing on something that is not a violation at all.
#[test]
fn the_dependency_scan_scopes_each_claim_correctly() {
    // A git pin outside `[dependencies]` is still a git pin. Narrowing the
    // scan to the runtime table would let all three of these through.
    for table in [
        "[dev-dependencies]",
        "[build-dependencies]",
        "[target.'cfg(unix)'.dependencies]",
    ] {
        let manifest = format!(
            "[package]\nname = \"x\"\n\n{table}\nhelper = {{ git = \"https://example.invalid/h\" }}\n"
        );
        let (pinned, _) = dependency_tables(&manifest);
        assert!(
            pinned.join("\n").contains("git ="),
            "a git pin in {table} must still be scanned, got: {pinned:?}"
        );
    }

    // A registry dev-dependency is not a violation of issue #8. It must not
    // reach the runtime-table assertion, which would report it as a stray
    // dependency alongside nir-rs.
    let manifest = "[dependencies]\nnir-rs = \"0.4.2\"\n\n[dev-dependencies]\nproptest = \"1\"\n";
    let (pinned, runtime) = dependency_tables(manifest);
    assert_eq!(runtime, ["nir-rs"], "dev-dependencies are not runtime ones");
    assert!(
        pinned.iter().any(|line| line.starts_with("proptest")),
        "the wide scan still covers it: {pinned:?}"
    );

    // Rewriting the pin as a table is the same dependency spelled differently.
    let manifest = "[dependencies.nir-rs]\nversion = \"0.4.2\"\n";
    let (pinned, runtime) = dependency_tables(manifest);
    assert_eq!(runtime, ["nir-rs"], "the table form declares nir-rs too");
    assert!(
        pinned.iter().any(|line| line.starts_with("version")),
        "and its body is still scanned for pins: {pinned:?}"
    );

    // A target-specific dependency is linked into the build like any other,
    // so it has to reach the runtime assertion -- otherwise a manifest could
    // add exactly the crates the non-goals forbid and still pass the
    // exact-set check. Both spellings, plus the dev- form that must stay out.
    let manifest = "[dependencies]\nnir-rs = \"0.4.2\"\n\n\
         [target.'cfg(unix)'.dependencies]\nneuromod = \"0.5\"\n\n\
         [target.'cfg(windows)'.dependencies.silicon-bridge]\nversion = \"0.1\"\n\n\
         [target.'cfg(unix)'.dev-dependencies]\nproptest = \"1\"\n";
    let (_, runtime) = dependency_tables(manifest);
    assert_eq!(
        runtime,
        ["nir-rs", "neuromod", "silicon-bridge"],
        "target-specific dependencies are runtime dependencies; \
         target-specific dev-dependencies are not"
    );

    // `[lib] path` is not a path pin. Only comment stripping and the section
    // filter keep it out.
    let manifest =
        "[lib]\npath = \"src/lib.rs\"\n\n[dependencies]\nnir-rs = \"0.4.2\" # not a path = pin\n";
    let (pinned, runtime) = dependency_tables(manifest);
    assert_eq!(runtime, ["nir-rs"]);
    assert!(
        !pinned.join("\n").contains("path ="),
        "neither `[lib] path` nor a comment is a pin: {pinned:?}"
    );
}

/// Acceptance criterion from issue #8: `nir-rs` resolves from crates.io, not
/// from a git or sibling-path pin.
#[test]
fn nir_rs_resolves_from_crates_io() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    let (pinned, runtime) = dependency_tables(&manifest);
    let dependencies = pinned.join("\n");

    for forbidden in ["git =", "path =", "git=", "path="] {
        assert!(
            !dependencies.contains(forbidden),
            "no dependency in any table may be pinned with `{forbidden}`, found in:\n{dependencies}",
        );
    }
    assert_eq!(
        runtime,
        ["nir-rs"],
        "`[dependencies]` must declare nir-rs and nothing else, found: {runtime:?}",
    );

    let lock_path: PathBuf = root.join("Cargo.lock");
    let lock = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", lock_path.display()));
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains("name = \"nir-rs\""))
        .expect("Cargo.lock has a nir-rs package entry");

    assert!(
        entry.contains("source = \"registry+https://github.com/rust-lang/crates.io-index\""),
        "nir-rs must come from the crates.io registry, got:\n{entry}",
    );
    assert!(
        entry.contains("version = \"0.4."),
        "nir-rs must resolve to 0.4.x, got:\n{entry}",
    );
}

/// A caller-built model with a non-finite threshold or weight must be rejected
/// before it reaches the graph.
///
/// `SnnModel::neurons` and `Neuron::weights` are public, so a hand-built or
/// post-load-mutated model never crosses the decode-time checks. The builder
/// already re-validates decay rates and weight-row lengths for exactly this
/// case; without the value checks a `NaN` threshold or weight would ride into
/// the `Lif` / `Linear` tensors, and `validate_structure()` would not catch it
/// because it only inspects graph structure.
#[test]
fn non_finite_public_parameters_are_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut model = spikenaut_snn::SnnModel::load_default().expect("load the shipped model");
        model.neurons[2].threshold = bad;
        let err = spikenaut_snn::build_lif_graph(&model)
            .expect_err("a non-finite threshold must not reach the graph");
        assert!(
            format!("{err}").contains("neuron 2 threshold"),
            "error should name the offending threshold, got: {err}"
        );

        let mut model = spikenaut_snn::SnnModel::load_default().expect("load the shipped model");
        model.neurons[5].weights[7] = bad;
        let err = spikenaut_snn::build_lif_graph(&model)
            .expect_err("a non-finite weight must not reach the graph");
        assert!(
            format!("{err}").contains("neuron 5 weight 7"),
            "error should name the offending weight, got: {err}"
        );
    }

    // Out-of-range but finite is rejected too, and the shipped model still builds.
    let mut model = spikenaut_snn::SnnModel::load_default().expect("load the shipped model");
    model.neurons[0].weights[0] = 1e9;
    assert!(spikenaut_snn::build_lif_graph(&model).is_err());
    let model = spikenaut_snn::SnnModel::load_default().expect("load the shipped model");
    assert!(spikenaut_snn::build_lif_graph(&model).is_ok());
}

/// A caller-built parameter that is in range but off the Q8.8 grid must be
/// rejected, not copied into the graph.
///
/// Being inside `[-128, 127.99609375]` is not the same as being representable:
/// Q8.8 holds multiples of `1/256`, so `0.1` has no code. Decoding snaps values
/// onto the grid, but the public fields bypass that — and a graph built from an
/// off-grid weight silently stops matching what the `.mem` artifacts can hold,
/// which is the equivalence this crate exists to preserve.
#[test]
fn off_grid_public_parameters_are_rejected() {
    for off_grid in [0.1, 1.0 / 3.0, 0.751] {
        let mut model = spikenaut_snn::SnnModel::load_default().expect("load the shipped model");
        model.neurons[4].weights[9] = off_grid;
        let err = spikenaut_snn::build_lif_graph(&model)
            .expect_err("an off-grid weight must not reach the graph");
        let text = format!("{err}");
        assert!(
            text.contains("neuron 4 weight 9") && text.contains("not Q8.8-representable"),
            "error should name the field and the reason, got: {text}"
        );
    }

    // On-grid neighbours of the same magnitude are still accepted, so the check
    // rejects off-grid values rather than simply anything unusual.
    let mut model = spikenaut_snn::SnnModel::load_default().expect("load the shipped model");
    model.neurons[4].weights[9] = 26.0 / 256.0;
    assert!(spikenaut_snn::build_lif_graph(&model).is_ok());

    // And the shipped artifact, whose values are all on the grid, still builds.
    let model = spikenaut_snn::SnnModel::load_default().expect("load the shipped model");
    assert!(spikenaut_snn::build_lif_graph(&model).is_ok());
}

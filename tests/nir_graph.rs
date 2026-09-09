// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `nir-rs` integration: the shipped `merged_v2` model must
//! build a valid 16-LIF NIR graph, and `nir-rs` must resolve from crates.io.

use std::path::{Component, Path, PathBuf};

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
    // The allowed set is exact, so a third dependency (issue #9 added
    // `axon-encoder`) or a rename still fails here. Sorted, so the manifest's
    // declaration order is not part of the contract.
    let mut names = runtime.clone();
    names.sort_unstable();
    assert_eq!(
        names,
        ["axon-encoder", "nir-rs"],
        "`[dependencies]` must declare exactly axon-encoder and nir-rs, found: {names:?}",
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

/// The README's Ecosystem table is the dependency contract written out in
/// prose, and its **Declared** marker is a claim about `Cargo.toml` today.
/// Nothing tied that claim to the manifest, so the table could name a
/// dependency this crate does not have -- or miss one it does -- with every
/// test still passing.
///
/// When #9 added `axon-encoder`, the table was updated by hand; nothing would
/// have failed if it had not been.
///
/// Only the **Declared** rows are checked. The rest of the table is intent --
/// crates to adopt once they exist -- and explicit non-dependencies, neither of
/// which has anything in the manifest to agree with.
///
/// This does not police the prose, and should not be read as if it did. A row
/// that implies a dependency without using the marker reads as a
/// non-dependency here and passes -- which is what the `neuromod` row did for
/// as long as its relationship said "crates.io dependency" against a manifest
/// saying it "stays out of the tree on purpose". That one was caught by
/// reading, not by a test, and a rewording like it still would be. The marker
/// is what this test makes load-bearing; the wording around it is review's
/// job.
#[test]
fn the_readme_dependency_table_agrees_with_the_manifest() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");

    let section = readme_section(&readme, "## Ecosystem");
    assert!(!section.is_empty(), "README.md has an Ecosystem section");

    let mut declared = readme_declared_components(&section);
    let (_, mut runtime) = dependency_tables(&manifest);
    declared.sort_unstable();
    runtime.sort_unstable();
    assert_eq!(
        declared, runtime,
        "every crate the README marks **Declared** must be in `[dependencies]`, \
         and every dependency must be marked there",
    );
}

/// The crates the Ecosystem table marks **Declared**, given the section body.
///
/// Every row has to be Component / Role / Relationship, so a table that grew a
/// column fails here rather than being silently misparsed into agreement.
fn readme_declared_components(section: &[String]) -> Vec<String> {
    let mut declared: Vec<String> = Vec::new();
    let mut rows = 0usize;
    for line in component_table(section) {
        let cells = table_cells(&line);
        assert_eq!(
            cells.len(),
            3,
            "every Ecosystem row is Component / Role / Relationship: {line}",
        );
        if cells[0] == "Component" || cells[0].starts_with("---") {
            continue;
        }
        rows += 1;
        let relationship = outside_code_spans(&rendered_cell(&cells[2]));
        if !relationship.contains("**Declared**") {
            continue;
        }
        // Every component is named in backticks, linked or not. Read from the
        // *visible* cell for the same reason the marker above is: a commented
        // -out `<!-- `nir-rs` -->` ahead of the real name would otherwise win
        // the `nth(1)` and the guard would compare the manifest against a name
        // the rendered table does not contain.
        // Everything the Relationship cell hides a marker behind, the
        // Component cell can hide a *name* behind: `<span title="`nir-rs`">`
        // renders as its element text while the guard read the attribute. The
        // two cells ask the same question, so they now share the answer --
        // except for code spans, which are stripped only from Relationship:
        // the crate name lives in one here.
        let component = rendered_cell(&cells[0]);
        let name = component
            .split('`')
            .nth(1)
            .unwrap_or_else(|| panic!("a component name must be in backticks: {line}"));
        declared.push(name.to_owned());
    }
    assert!(
        rows > 0,
        "no Ecosystem table found: it must carry the header {COMPONENT_HEADER:?} in that \
         order, with a three-cell delimiter row directly beneath it",
    );
    declared
}

/// `cell` reduced to the text a reader actually sees.
///
/// Markdown hides text in more places than any one strip pass covers, and each
/// was found the same way: as a cell that renders one thing while the guard
/// read another. HTML comments, image descriptions, link destinations and raw
/// HTML tags, in that order -- images before links, or the link pass eats an
/// image's destination and leaves its description behind as if it were text.
///
/// Code spans are deliberately *not* removed here. They hide a marker in the
/// Relationship cell but carry the crate name in the Component cell, so that
/// one strip belongs to the caller that wants it.
fn rendered_cell(cell: &str) -> String {
    outside_html_tags(&outside_link_destinations(&outside_images(
        &without_html_comments(cell),
    )))
}

/// `text` with image spans removed, description and all.
///
/// `![**Declared**](transparent.png)` renders as an image: the description
/// becomes an `alt` attribute, never bold page text. Stripping only the
/// destination -- which is what the link pass does, since `](` looks the same
/// in both -- left `[**Declared**]` on the visible side and the guard counted
/// a marker the rendered table does not show.
fn outside_images(text: &str) -> String {
    let mut visible = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("![") {
        visible.push_str(&rest[..at]);
        let Some(after) = rest[at + 2..]
            .find("](")
            .map(|end| &rest[at + 2 + end + 2..])
        else {
            visible.push_str(&rest[at..]);
            return visible;
        };
        match closing_paren(after) {
            Some(end) => rest = &after[end + 1..],
            None => {
                visible.push_str(&rest[at..]);
                return visible;
            }
        }
    }
    visible.push_str(rest);
    visible
}

/// `cell` with any HTML comment spans removed.
///
/// A marker surviving only inside `<!-- **Declared** -->` renders as nothing,
/// so the contract a reader sees has lost it while a substring check still
/// finds it -- the two sets stay equal and the guard says nothing. An
/// unterminated `<!--` hides everything after it, and is treated that way.
fn without_html_comments(cell: &str) -> String {
    let mut open = false;
    visible_outside_comments(cell, &mut open)
}

/// The part of `line` a reader sees, given whether a comment is already open.
///
/// `open` is carried across calls so a `<!--` on one line hides everything up
/// to a `-->` on a later one. A comment that is never closed hides the rest of
/// the input, which is what Markdown does with it.
///
/// Single-line callers pass a fresh `false` and get the old behaviour; the
/// state is what lets a *block* of lines be read the way it renders.
fn visible_outside_comments(line: &str, open: &mut bool) -> String {
    let mut visible = String::with_capacity(line.len());
    let mut rest = line;
    loop {
        if *open {
            match rest.find("-->") {
                Some(end) => {
                    rest = &rest[end + "-->".len()..];
                    *open = false;
                }
                None => break,
            }
        } else {
            match rest.find("<!--") {
                Some(start) => {
                    visible.push_str(&rest[..start]);
                    rest = &rest[start + "<!--".len()..];
                    *open = true;
                }
                None => {
                    visible.push_str(rest);
                    break;
                }
            }
        }
    }
    visible
}

/// The Ecosystem table's lines, located line by line rather than by an exact
/// substring match on the heading.
///
/// `README.md` carries no `text` attribute, so a Windows checkout with
/// `core.autocrlf=true` gets CRLF and a `"\n## Ecosystem\n"` match finds
/// nothing -- turning the guard above into a panic that never reaches the
/// manifest. `str::lines` strips the `\r`, and `trim_end` absorbs a trailing
/// space on the heading, so neither silences the check.
///
/// Fenced blocks are skipped while looking for the heading and while looking
/// for the section's end. A code sample containing a line that reads exactly
/// `## Ecosystem` is a *sample*, not a heading, and matching it made the guard
/// parse from the wrong place and fail on a README whose real section was
/// untouched. The same state keeps a `## ` inside the section's own fenced
/// block from ending it early.
fn readme_section(readme: &str, heading: &str) -> Vec<String> {
    let lines: Vec<&str> = readme.lines().collect();
    let visible = visible_lines(&lines);
    let located = blank_code(&visible);
    let found = heading_lines(&located, heading);
    // A second visible `## Ecosystem` or `## Files` left the later section
    // entirely unchecked: first-match location validated the good copy and
    // `section_body` stopped before the duplicate, so a copy-paste or a merge
    // could add a conflicting contract that nothing read. There is no useful
    // way to choose between two, so having two is the failure.
    assert!(
        found.len() <= 1,
        "`{heading}` appears {} times; a guarded section must be written once",
        found.len(),
    );
    match found.first() {
        Some(&at) => section_body(&visible, &located, at),
        None => Vec::new(),
    }
}

/// Every line with its HTML comments removed, state carried across lines.
///
/// This is what the whole guard reads. Comment state is a property of the
/// *document*, not of any slice of it, so it is resolved once here and the
/// scanners downstream never restart it. They used to: each ran its own
/// `open = false` pass over the section it was handed, which meant a `<!--`
/// opened on or above the section heading was invisible to them. A hidden
/// table matching `Cargo.toml`, or a hidden fence naming files that exist,
/// was then read as the real one while the rendered section drifted.
///
/// Fenced content is passed through *raw*, and comment state does not advance
/// across it. Inside a fence `<!--` is literal text that Markdown displays, so
/// removing it rewrote what the README says: a tree entry reading
/// `js<!--draft-->on.rs` collapsed to the `json.rs` that exists, and the
/// existence check passed for a path no reader can see. Fences are also left
/// intact for `fenced_block`, which needs their contents; only *locating* has
/// to ignore them, which is [`blank_code`].
///
/// The two states have to be resolved together and in this order. A fence
/// inside a comment is not a fence -- Markdown renders nothing for it -- so
/// comment removal decides what counts as a fence, and being inside a fence
/// then decides that comment syntax is just text.
fn visible_lines(lines: &[&str]) -> Vec<String> {
    let mut open = false;
    let mut fenced = false;
    lines
        .iter()
        .map(|line| {
            if fenced {
                if line.trim_start().starts_with("```") {
                    fenced = false;
                }
                return (*line).to_owned();
            }
            let visible = visible_outside_comments(line, &mut open);
            if visible.trim_start().starts_with("```") {
                fenced = true;
            }
            visible
        })
        .collect()
}

/// The same lines with every line Markdown renders as *code* blanked.
///
/// That is fenced content and both fence markers, and any line indented by
/// four spaces or more -- an indented code block. Only fences were blanked
/// before, so a four-space-indented example of the dependency table was still
/// a candidate for the real one: if the example agreed with `Cargo.toml` the
/// guard passed while the rendered table below it had drifted. The same
/// four-space line that [`dedent`] already refuses to read as a heading.
///
/// Blanking rather than dropping keeps the indices aligned with the input, so
/// a position found here indexes the un-blanked lines too. That is what lets a
/// heading be located, and a section be ended, without a Markdown sample of a
/// heading or a table being mistaken for the real thing.
///
/// Over-blanking is the safe direction: a table this hides is a table not
/// found, which is the loud `no Ecosystem table found` panic rather than a
/// quiet agreement with the wrong rows.
fn blank_code(visible: &[String]) -> Vec<String> {
    let mut fenced = false;
    visible
        .iter()
        .map(|line| {
            if line.trim_start().starts_with("```") {
                fenced = !fenced;
                return String::new();
            }
            if fenced || is_indented_code(line) {
                String::new()
            } else {
                line.clone()
            }
        })
        .collect()
}

/// Whether Markdown reads `line` as an indented code block.
fn is_indented_code(line: &str) -> bool {
    line.starts_with("    ") || line.starts_with('\t')
}

/// The index of `heading`, ignoring any that appear inside a fenced block.
fn heading_lines(located: &[String], heading: &str) -> Vec<usize> {
    located
        .iter()
        .enumerate()
        .filter(|(_, line)| atx_heading(line).as_deref() == Some(heading))
        .map(|(at, _)| at)
        .collect()
}

/// `line` as an ATX heading, with the separator after its `#`s normalised.
///
/// Markdown accepts a tab there as readily as a space, and accepts more than
/// one space. Matching the literal `"## Ecosystem"` meant `##\tEcosystem`
/// was not found at all -- a loud failure on a README that renders the same --
/// while `#\tAppendix` did not *end* a section, which is the silent half: the
/// rest of the document stayed in the body and a table under that heading
/// could be selected once the real one had drifted.
fn atx_heading(line: &str) -> Option<String> {
    let text = dedent(line).trim_end();
    let hashes = text.len() - text.trim_start_matches('#').len();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let body = text.get(hashes..)?.strip_prefix([' ', '\t'])?;
    Some(format!(
        "{} {}",
        "#".repeat(hashes),
        without_closing_hashes(body).trim_start()
    ))
}

/// `body` without Markdown's optional closing `#` sequence.
///
/// `## Ecosystem ##` is the same heading as `## Ecosystem`. Keeping the
/// trailing hashes meant a duplicate written that way did not match the
/// guarded heading, so the once-only check never saw it -- while
/// `section_body` still stopped there, leaving the second contract read by
/// nothing. The closing run has to be preceded by whitespace, so `Ecosystem##`
/// is a name, not a heading with a closing sequence.
fn without_closing_hashes(body: &str) -> &str {
    let body = body.trim_end();
    let closed = body.trim_end_matches('#');
    if closed.len() < body.len() && (closed.is_empty() || closed.ends_with([' ', '\t'])) {
        closed.trim_end()
    } else {
        body
    }
}

/// `line` with the indentation Markdown permits before an ATX heading removed.
///
/// Up to three spaces still opens a heading; a fourth makes it an indented code
/// block instead. Comparing the raw line meant a heading a human had nudged
/// right rendered identically but was not found at all, and the guard panicked
/// on a document whose meaning had not changed.
fn dedent(line: &str) -> &str {
    let body = line.trim_start_matches(' ');
    if line.len() - body.len() <= 3 {
        body
    } else {
        line
    }
}

/// The lines after `at`, up to the next `## ` that is not inside a fence.
///
/// Fence state is tracked here too, so a `## ` inside the section's own fenced
/// block -- a Markdown sample, a shell comment -- does not end the section
/// early.
fn section_body(visible: &[String], located: &[String], at: usize) -> Vec<String> {
    let ends = located[at + 1..]
        .iter()
        .position(|line| {
            // A level-one heading closes a level-two section too. Stopping
            // only at `## ` left the rest of the document in the body, so a
            // component table under a later `# ` could be selected after the
            // real one had drifted.
            atx_heading(line)
                .is_some_and(|heading| heading.starts_with("# ") || heading.starts_with("## "))
        })
        .map_or(visible.len(), |offset| at + 1 + offset);
    visible[at + 1..ends].to_vec()
}

/// A CRLF checkout, or a stray trailing space, must not make the dependency
/// guard disappear into a panic.
#[test]
fn the_ecosystem_section_is_found_whatever_the_line_endings() {
    let lf =
        "intro\n\n## Ecosystem\n\n| Component | Role | Relationship |\n\n## The Story\nprose\n";
    let crlf = lf
        .replace('\n', "\r\n")
        .replace("## Ecosystem", "## Ecosystem ");

    let found = readme_section(lf, "## Ecosystem");
    assert_eq!(
        found,
        readme_section(&crlf, "## Ecosystem"),
        "CRLF must locate the same rows"
    );
    assert!(
        found.iter().any(|line| line.starts_with('|')),
        "the section must carry the table: {found:?}",
    );
    assert!(
        !found.iter().any(|line| line.contains("prose")),
        "the section must stop at the next heading: {found:?}",
    );
}

/// Every path the `## Files` tree names must exist.
///
/// That tree is the repository's own account of what it ships, and it had
/// drifted three ways at once: a shipped `config.json` missing, the crate root
/// `src/lib.rs` missing, and an eight-module `tools/` package shown as a single
/// script. A rename would have been just as invisible. Existence is the claim
/// the tree actually makes about every one of its entries, and this holds it
/// to that.
///
/// The reverse direction -- every shipped file must be *named* -- was here too,
/// scoped to `dataset/merged_v2/`, and is deliberately gone. The tree is a
/// reader's map, not a manifest: it names 12 of the 35 tracked files and the
/// other 23 are absent by design. A guard that enforced set equality over one
/// subdirectory while the rest of the tree was admittedly partial was drawing a
/// line the document does not draw, and it made every new file under that one
/// directory a documentation obligation. The kind check (`/` means directory)
/// went with it for the same reason: it is a claim about formatting, not about
/// the repository.
///
/// What survives is narrow on purpose. `escaping_paths` stays because it is
/// what makes existence mean anything -- `Path::join` discards a leading `/`,
/// so a tree naming `/etc/passwd` would be checked against the host filesystem
/// and pass on any runner.
///
/// Only the fenced tree block is read. The section also carries prose and a
/// `### Loading on FPGA` Verilog example, and parsing those would let an
/// ordinary documentation edit -- a line beginning `$readmemh("...")` -- fail
/// this test with a "path" that was never a path.
#[test]
fn the_files_tree_names_only_paths_that_exist() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");

    let named = files_tree_paths(&readme_section(&readme, "## Files"));
    assert!(
        named.len() >= 5,
        "the Files tree named too little: {named:?}"
    );

    let escaping = escaping_paths(&named);
    assert!(
        escaping.is_empty(),
        "`## Files` names {escaping:?}, which leave the repository; \
         every entry must be a path inside it",
    );

    let missing: Vec<&String> = named.iter().filter(|p| !root.join(p).exists()).collect();
    assert!(
        missing.is_empty(),
        "`## Files` names {missing:?}, which do not exist; {} paths checked",
        named.len(),
    );

    let unrooted = unrooted_paths(root, &named);
    assert!(
        unrooted.is_empty(),
        "`## Files` names {unrooted:?}, which resolve outside the repository \
         through a symlink; existence must mean the repository ships it",
    );
}

/// Entries that resolve outside the repository once symlinks are followed.
///
/// [`escaping_paths`] reads the path as written, which a symlink defeats: with
/// a tracked `outside -> /etc`, the entry `outside/passwd` is neither absolute
/// nor a `..` walk, so it was accepted -- and `exists()` follows the link, so
/// the guard passed on a host file the repository does not ship. Lexical
/// rejection and resolved rejection are different questions and both have to
/// be asked.
///
/// A path that does not resolve at all is not reported here; that is the
/// existence check's finding, and naming it twice would only obscure it.
fn unrooted_paths<'a>(root: &Path, named: &'a [String]) -> Vec<&'a String> {
    let Ok(real_root) = root.canonicalize() else {
        return Vec::new();
    };
    named
        .iter()
        .filter(|p| {
            root.join(p)
                .canonicalize()
                .is_ok_and(|real| !real.starts_with(&real_root))
        })
        .collect()
}

/// Entries that name something outside the repository.
///
/// `Path::join` with an absolute path discards the root, so a tree naming
/// `/etc/passwd` is checked against the host filesystem and passes on any CI
/// runner -- the existence invariant silently not holding for that row. `..`
/// walks out of the checkout the same way. Neither is a repository path, so
/// both are rejected before the join rather than resolved by it.
fn escaping_paths(named: &[String]) -> Vec<&String> {
    named
        .iter()
        .filter(|p| {
            let path = Path::new(p.as_str());
            path.is_absolute() || path.components().any(|c| c == Component::ParentDir)
        })
        .collect()
}

/// The repository paths named inside the `## Files` fenced tree.
///
/// A line ending in `/` sets the directory the entries below it hang from, a
/// box-drawing entry is a file in that directory, and any other line is a file
/// at the repository root -- named with or without an extension, since
/// `LICENSE` and `Makefile` are as much files as `config.json` is.
fn files_tree_paths(section: &[String]) -> Vec<String> {
    let mut dir = String::new();
    let mut nested: Vec<(usize, String)> = Vec::new();
    let mut paths = Vec::new();
    for line in fenced_block(section) {
        match classify_tree_line(&line) {
            Some(TreeLine::Dir(entry)) => {
                dir = entry.to_owned();
                nested.clear();
                paths.push(dir.clone());
            }
            Some(TreeLine::Nested { indent, name }) => {
                // Anything at this column or deeper is a sibling or a closed
                // branch, not an ancestor.
                while nested.last().is_some_and(|&(at, _)| at >= indent) {
                    nested.pop();
                }
                let prefix = nested
                    .last()
                    .map_or(dir.as_str(), |(_, path)| path.as_str());
                let path = format!("{prefix}{name}");
                if name.ends_with('/') {
                    nested.push((indent, path.clone()));
                }
                paths.push(path);
            }
            Some(TreeLine::Root(name)) => paths.push(name.to_owned()),
            None => {}
        }
    }
    paths
}

/// The lines inside a section's first fenced block, fences excluded.
///
/// A section can hold several fenced blocks -- `## Files` holds the tree and a
/// Verilog example -- so this takes the first and stops at its closing fence
/// rather than skipping fence markers wherever they appear.
///
/// Comment state is carried across the section, so a fence wrapped in a
/// multiline `<!-- ... -->` is not the first *visible* one. Reading a
/// commented-out fence is the quiet failure: the tree would vanish from the
/// rendered README while every path and artifact check still passed against
/// the hidden copy, so the guard would go silent at exactly the moment the
/// documentation it guards disappeared.
fn fenced_block(section: &[String]) -> Vec<String> {
    section
        .iter()
        .skip_while(|line| !line.trim_start().starts_with("```"))
        .skip(1)
        .take_while(|line| !line.trim_start().starts_with("```"))
        .cloned()
        .collect()
}

/// The box-drawing prefixes a `## Files` entry can carry: tee, then elbow for
/// the last child of a directory.
const TREE_ENTRY_GLYPHS: [&str; 2] = ["\u{251c}\u{2500}\u{2500} ", "\u{2514}\u{2500}\u{2500} "];

/// `line` up to the annotation comment that follows it, if any.
///
/// Only a `#` with whitespace in front of it opens an annotation. Cutting at
/// *every* `#` truncates a path that legitimately contains one -- `model#1.json`
/// becomes `model` -- and the existence check then passes against whatever the
/// truncation happens to name. `src/model#2.rs` would reduce to `src/model`,
/// which exists as a directory, so a documented file that is not there at all
/// would be silently validated. The loud direction is a wrong set comparison;
/// this one is the direction that says nothing, which is why it is worth the
/// narrower rule.
///
/// A `#` in the first column is therefore a filename, not a comment. Both
/// READMEs write every annotation and every wrapped continuation with leading
/// whitespace, so nothing that is a comment today stops being one.
fn strip_annotation(line: &str) -> &str {
    line.char_indices()
        .find(|&(at, character)| character == '#' && line[..at].ends_with(char::is_whitespace))
        .map_or(line, |(at, _)| &line[..at])
}

/// What one line of the `## Files` tree names.
enum TreeLine<'a> {
    /// A directory heading, keeping its trailing `/`.
    Dir(&'a str),
    /// An entry under the heading above it, with the column its glyph sits
    /// in. The indentation is what says whether it hangs from the heading or
    /// from a nested directory listed between them.
    Nested { indent: usize, name: &'a str },
    /// A file at the repository root, written without a tree glyph.
    Root(&'a str),
}

/// Classify one tree line, or `None` for a blank or a wrapped comment
/// continuation.
///
/// Comments are cut first, which is what reduces the `#` continuation lines to
/// nothing rather than letting their prose look like filenames.
fn classify_tree_line(line: &str) -> Option<TreeLine<'_>> {
    // The glyph comes off *before* the annotation. Doing it the other way
    // round made the glyph's own mandatory trailing space the whitespace that
    // opens an annotation, so `|-- #generated.rs` was cut down to the bare
    // glyph and reported as the missing path `|--` -- a real filename the tree
    // cannot express, and a failure message naming punctuation instead of the
    // entry. Stripping the glyph first leaves `#generated.rs` as the name, and
    // a leading `#` there is no longer preceded by anything.
    if let Some(entry) = TREE_ENTRY_GLYPHS
        .iter()
        .find_map(|glyph| line.trim_start().strip_prefix(glyph))
    {
        let name = strip_annotation(entry).trim();
        let indent = line.len() - line.trim_start().len();
        return (!name.is_empty()).then_some(TreeLine::Nested { indent, name });
    }
    let trimmed = strip_annotation(line).trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.ends_with('/') {
        return Some(TreeLine::Dir(trimmed));
    }
    Some(TreeLine::Root(trimmed))
}

/// The exact header the Ecosystem table must carry, in order.
const COMPONENT_HEADER: [&str; 3] = ["Component", "Role", "Relationship"];

/// Whether `line` is a Markdown delimiter row of exactly three cells.
///
/// Alignment colons are allowed, since `|:---|---:|:---:|` renders the same
/// table.
fn is_delimiter_row(line: &str) -> bool {
    let cells = table_cells(line);
    cells.len() == COMPONENT_HEADER.len()
        && cells.iter().all(|cell| {
            let dashes = cell.trim().trim_start_matches(':').trim_end_matches(':');
            // Three, not one. GitHub's own documentation says three, every
            // example in the GFM spec uses three, and the spec's normative
            // text says "hyphens" without giving a minimum -- so this cannot
            // be settled from the spec alone. Accepting one was the silent
            // reading: if GitHub does require three, `|-|-|-|` renders as
            // paragraph text and the guard went on parsing rows from a table
            // that is no longer there. Requiring three is the loud reading,
            // and its failure is a contributor typing two more dashes.
            dashes.len() >= 3 && dashes.chars().all(|c| c == '-')
        })
}

/// The rows of the Ecosystem table, from its header to its end, or nothing if
/// the section holds no such table.
///
/// The section is not the table. Filtering every `|` line in it would feed a
/// second table -- or a fenced example using pipes -- to the row parser, so an
/// unrelated documentation edit could trip the three-cell assertion or
/// contribute stray component names.
///
/// The header is matched by its *cells*, not by the literal `| Component `
/// spelling: a Markdown formatter may drop the optional padding or the leading
/// delimiter, and `|Component|Role|Relationship|` is the same table.
///
/// Two things this deliberately insists on, because recognising the table by
/// its first cell alone let the *rendered* contract change with the guard
/// still agreeing with itself:
///
/// - **The whole header, in order.** Reordering it to
///   `Component | Relationship | Role` leaves every row untouched, so the parse
///   was identical -- while a reader now sees each **Declared** marker under
///   Role. This repository already has a second `| Component | Spec |` table,
///   so a first-cell match was thin to begin with.
/// - **A delimiter row directly under it.** Delete `|---|---|---|` and Markdown
///   stops rendering a table at all, but the header and its pipe-delimited
///   lines are still there to parse, so the same dependency set came back out
///   of a document that no longer contains the table.
///
/// Comment state is carried across the section *before* the header is located,
/// not per cell afterwards. A whole table inside `<!-- ... -->` placed above
/// the real one was otherwise selected first, and a per-cell strip cannot see
/// that -- it restarts on every cell, so rows wholly inside a comment look
/// visible. Verified with a hidden table matching the manifest while the
/// rendered one had drifted: the guard passed, reading a table nobody can see.
/// That is the failure `fenced_block` already guards against for the file
/// tree, fixed the same way.
fn component_table(section: &[String]) -> Vec<String> {
    let visible = blank_code(section);

    let Some(header) = visible
        .iter()
        .position(|line| table_cells(line) == COMPONENT_HEADER)
    else {
        return Vec::new();
    };
    if !visible
        .get(header + 1)
        .is_some_and(|line| is_delimiter_row(line))
    {
        return Vec::new();
    }
    visible[header..]
        .iter()
        .take_while(|line| line.contains('|'))
        .cloned()
        .collect()
}

/// Split one Markdown table row into cells on *unescaped* pipes.
///
/// A `\\|` inside a cell is table prose, not a column boundary. Splitting on
/// every `|` turned a valid three-column row into four cells and failed the
/// shape assertion, so ordinary wording in the Role column could break the
/// dependency guard.
///
/// Two things this is careful about, both of which were silent holes:
///
/// - **Only the pipe escape is consumed.** Dropping the backslash from every
///   `\\x` reconstructed `**Declared**` out of `\\*\\*Declared\\*\\*`, which
///   Markdown renders as literal asterisks. The rendered table had lost the
///   marker while the guard still counted the row. Escapes of anything else
///   are kept verbatim, so a cell that reads literally stays literal here too.
/// - **At most one outer pipe on each side.** `trim_start_matches('|')` ate
///   every leading pipe, so a row beginning `||` -- which Markdown reads as an
///   empty first cell, changing the column count -- normalised to exactly
///   `Component`, `Role`, `Relationship` and satisfied the header and
///   delimiter checks. Removing a single optional delimiter leaves the empty
///   column visible, and the shape assertion then reports it.
fn table_cells(row: &str) -> Vec<String> {
    let trimmed = row.trim();
    let after_leading = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let body = after_leading.strip_suffix('|').unwrap_or(after_leading);
    let mut cells = vec![String::new()];
    let mut escaped = false;
    for character in body.chars() {
        let cell = cells.last_mut().expect("a cell in progress");
        match (escaped, character) {
            // Only `\|` is a quoted delimiter. Every other escape keeps its
            // backslash so the cell still reads the way it renders.
            (true, '|') => {
                cell.push('|');
                escaped = false;
            }
            (true, _) => {
                cell.push('\\');
                cell.push(character);
                escaped = false;
            }
            (false, '\\') => escaped = true,
            (false, '|') => cells.push(String::new()),
            (false, _) => cell.push(character),
        }
    }
    if escaped {
        cells.last_mut().expect("a cell in progress").push('\\');
    }
    cells.iter().map(|cell| cell.trim().to_owned()).collect()
}

/// `cell` with its inline code spans removed.
///
/// `` `**Declared**` `` renders as literal asterisks inside code, not the bold
/// marker, but a substring check still found it -- the same shape as the
/// escaped-asterisk hole, one syntax over. An unterminated run is literal
/// text, so it stays visible.
///
/// Deliberately *not* applied to the Component cell: the crate name lives in a
/// code span there, so stripping them would erase the thing being read.
fn outside_code_spans(cell: &str) -> String {
    let bytes = cell.as_bytes();
    let mut visible = String::with_capacity(cell.len());
    let mut at = 0;
    while at < bytes.len() {
        let Some(offset) = bytes[at..].iter().position(|&b| b == b'`') else {
            break;
        };
        let open = at + offset;
        visible.push_str(&cell[at..open]);
        let ticks = backtick_run(bytes, open);
        match closing_run(bytes, open + ticks, ticks) {
            Some(close) => at = close + ticks,
            None => {
                visible.push_str(&cell[open..]);
                return visible;
            }
        }
    }
    visible.push_str(&cell[at..]);
    visible
}

/// `text` with the destination of every inline link removed.
///
/// `[not declared](https://example.invalid/**Declared**)` renders as the words
/// "not declared" -- the URL is not shown at all -- but a substring check
/// found the marker in it and counted the row. The dependency contract a
/// reader sees could lose its marker with the two sets still equal.
///
/// Only a `](` that closes a link is consumed, and nesting is tracked so a URL
/// containing parentheses does not end the destination early. Link *text* is
/// deliberately kept: it is what renders, so a marker written there is a real
/// marker.
fn outside_link_destinations(text: &str) -> String {
    let mut visible = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("](") {
        visible.push_str(&rest[..at + 1]);
        let after = &rest[at + 2..];
        match closing_paren(after) {
            Some(end) => rest = &after[end + 1..],
            None => {
                visible.push_str(&rest[at + 1..]);
                return visible;
            }
        }
    }
    visible.push_str(rest);
    visible
}

/// `text` with raw HTML tags, and so their attributes, removed.
///
/// `<span title="**Declared**">not declared</span>` renders as the words "not
/// declared": an attribute value is never page text. The substring check found
/// the marker in the invisible `title` and counted the row, so the rendered
/// contract could say the opposite of what the guard recorded.
///
/// Markdown allows raw HTML in a table cell, so this is reachable without any
/// unusual syntax. Only the tags go -- the text between them is what renders,
/// and a marker written there is a real one.
///
/// An unterminated `<` is literal text (Markdown shows it), so it stays.
fn outside_html_tags(text: &str) -> String {
    let mut visible = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('<') {
        visible.push_str(&rest[..at]);
        match tag_end(&rest[at..]) {
            Some(end) => rest = &rest[at + end + 1..],
            None => {
                visible.push_str(&rest[at..]);
                return visible;
            }
        }
    }
    visible.push_str(rest);
    visible
}

/// Where the tag opened at the start of `tag` ends.
///
/// A `>` inside a quoted attribute value does not close it:
/// `<span title="> **Declared**">not declared</span>` ends at the *second*
/// `>`, and taking the first left the marker and the rest of the tag on the
/// visible side. The same shape as every other finding on this guard -- a
/// delimiter that is only a delimiter outside quoting.
fn tag_end(tag: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    tag.char_indices().find_map(|(at, ch)| {
        match quote {
            Some(open) if ch == open => quote = None,
            Some(_) => {}
            None if ch == '"' || ch == '\'' => quote = Some(ch),
            None if ch == '>' => return Some(at),
            None => {}
        }
        None
    })
}

/// The `)` that closes an inline link, given everything after its `](`.
///
/// CommonMark puts three things between `](` and the closing `)`, and taking
/// any of them literally leaks the marker back into the visible text:
///
/// - **Nested parentheses** in the destination, which must balance.
/// - **Backslash escapes**: `\(` and `\)` are literal characters, not
///   structure. Counting them meant `](host/x\)**Declared**)` ended at the
///   escaped paren and put the rest of the destination back on the visible
///   side, and `](host/x\(y**Declared**)` never found a close at all -- so the
///   marker was read as visible in both.
/// - **A title** after the destination, in quotes. It renders as a tooltip and
///   never as page text, so a marker inside one is not a marker; a `)` inside
///   one is not a close. `](url "a)b **Declared**")` ended at that `)`.
///
/// A title has to be preceded by whitespace, which is what `past_destination`
/// tracks: a quote character inside the destination itself is an ordinary
/// character and must not open one.
fn closing_paren(after: &str) -> Option<usize> {
    let mut scan = LinkScan::default();
    after.char_indices().find_map(|(at, ch)| scan.step(at, ch))
}

/// The state [`closing_paren`] carries from one character to the next.
#[derive(Default)]
struct LinkScan {
    depth: usize,
    escaped: bool,
    quote: Option<char>,
    past_destination: bool,
}

impl LinkScan {
    /// Consumes one character, yielding the offset when it closes the link.
    fn step(&mut self, at: usize, ch: char) -> Option<usize> {
        if std::mem::take(&mut self.escaped) {
            return None;
        }
        if ch == '\\' {
            self.escaped = true;
            return None;
        }
        if self.titling(ch) {
            return None;
        }
        match ch {
            '(' => self.depth += 1,
            ')' if self.depth == 0 => return Some(at),
            ')' => self.depth -= 1,
            _ => {}
        }
        None
    }

    /// Whether `ch` belongs to the title rather than the destination.
    ///
    /// That covers the quote characters that open and close one, everything
    /// between them, and the whitespace that ends the destination and makes a
    /// title possible in the first place.
    fn titling(&mut self, ch: char) -> bool {
        if self.quote == Some(ch) {
            self.quote = None;
            return true;
        }
        if self.quote.is_some() {
            return true;
        }
        if ch.is_whitespace() {
            self.past_destination = true;
            return true;
        }
        if self.past_destination && (ch == '"' || ch == '\'') {
            self.quote = Some(ch);
            return true;
        }
        false
    }
}

/// The length of the run of backticks beginning at `at`.
fn backtick_run(bytes: &[u8], at: usize) -> usize {
    bytes[at..].iter().take_while(|&&b| b == b'`').count()
}

/// Where the next run of *exactly* `ticks` backticks starts, at or after `from`.
///
/// Per CommonMark a span opened with N backticks closes only on a run of N --
/// not on the first backtick of a longer one. Searching for the substring
/// instead let a three-backtick run close a one-backtick span, which ended the
/// span early and put the rest of the cell back on the visible side. A marker
/// the reader sees rendered as code was then read as the **Declared** marker,
/// so the README could lose its dependency contract with the sets still equal.
/// A longer run is skipped whole rather than re-entered, so its interior
/// backticks cannot be mistaken for a closer either.
fn closing_run(bytes: &[u8], from: usize, ticks: usize) -> Option<usize> {
    let mut at = from;
    while at < bytes.len() {
        if bytes[at] != b'`' {
            at += 1;
            continue;
        }
        let run = backtick_run(bytes, at);
        if run == ticks {
            return Some(at);
        }
        at += run;
    }
    None
}

/// A component name hidden in an HTML comment does not become the declared
/// crate.
///
/// The marker check already read the visible Relationship cell, but the name
/// was taken from the raw Component cell, so a commented-out crate ahead of
/// the real one won `nth(1)`. The manifest comparison then held against a name
/// the rendered table does not contain -- and because the hidden name is the
/// one that *is* in `[dependencies]`, the guard would agree with itself while
/// the README advertised something else.
#[test]
fn a_commented_out_component_name_is_not_the_declared_one() {
    let readme = concat!(
        "## Ecosystem\n",
        "\n",
        "| Component | Role | Relationship |\n",
        "|---|---|---|\n",
        "| <!-- `nir-rs` --> `replacement` | graph interchange | **Declared** |\n",
    );

    assert_eq!(
        readme_declared_components(&readme_section(readme, "## Ecosystem")),
        vec!["replacement".to_owned()],
        "the declared crate must be the one a reader sees, not one inside a comment",
    );
}

/// A fenced block inside an HTML comment is not the section's first block.
///
/// Markdown renders nothing for it, so reading it would let the whole file
/// tree be commented out while every path and artifact assertion still passed
/// against the hidden copy. That is the silent direction: the guard keeps
/// saying the documentation is accurate after the documentation is gone.
#[test]
fn a_fence_inside_an_html_comment_is_not_the_first_block() {
    let readme = concat!(
        "## Files\n",
        "\n",
        "<!--\n",
        "```text\n",
        "hidden/tree.rs\n",
        "```\n",
        "-->\n",
        "```text\n",
        "visible/tree.rs\n",
        "```\n",
    );

    assert_eq!(
        fenced_block(&readme_section(readme, "## Files")),
        vec!["visible/tree.rs".to_owned()],
        "a commented-out fence must not be read as the section's tree",
    );
}

/// A comment opened on the heading line still hides what follows it.
///
/// The heading renders -- its visible prefix is intact -- so it is found, and
/// the section below it begins *inside* an open comment. Every scanner used to
/// restart its own comment state from the section it was handed, which put the
/// `<!--` one line out of reach and made the hidden fence the section's first
/// block. Resolving comment state over the whole document before slicing is
/// what closes it.
#[test]
fn a_comment_opened_on_the_heading_line_hides_the_section_below_it() {
    let readme = concat!(
        "## Files <!--\n",
        "\n",
        "```text\n",
        "hidden/tree.rs\n",
        "```\n",
        "\n",
        "-->\n",
        "\n",
        "```text\n",
        "visible/tree.rs\n",
        "```\n",
    );

    assert_eq!(
        fenced_block(&readme_section(readme, "## Files")),
        vec!["visible/tree.rs".to_owned()],
        "a comment opened on the heading line must hide the fence below it",
    );
}

/// A longer backtick run does not close a shorter inline code span.
///
/// Per CommonMark a span opened with N backticks closes on a run of exactly N.
/// Accepting the first backtick of a longer run ended the span early and put
/// the rest of the cell back on the visible side, so a marker the reader sees
/// rendered as code was read as the **Declared** marker.
#[test]
fn a_longer_backtick_run_does_not_close_a_shorter_code_span() {
    assert_eq!(
        outside_code_spans("`x```**Declared**`"),
        "",
        "the three-backtick run cannot close a one-backtick span",
    );
    assert_eq!(outside_code_spans("a `b` c ``d`` e"), "a  c  e");
    assert_eq!(
        outside_code_spans("**Declared** in `Cargo.toml`"),
        "**Declared** in ",
        "a marker outside code stays visible",
    );
}

/// An HTML comment written inside a fence is part of the text, not a comment.
///
/// Markdown displays it verbatim there. Stripping it rewrote the tree: an
/// entry reading `js<!--draft-->on.rs` collapsed to `json.rs`, which exists,
/// so the existence check passed for a path no reader can see. Silent, and in
/// the direction that matters -- the document drifted and the guard agreed.
#[test]
fn an_html_comment_inside_a_fence_is_part_of_the_path() {
    let readme = concat!(
        "## Files\n",
        "\n",
        "```text\n",
        "src/\n",
        "└── js<!--draft-->on.rs\n",
        "```\n",
    );

    assert!(
        fenced_block(&readme_section(readme, "## Files"))
            .iter()
            .any(|line| line.contains("<!--draft-->")),
        "a comment inside a fence is literal text and must survive",
    );
}

/// A four-space-indented table is a code sample, not the dependency table.
///
/// `dedent` already refuses to read a four-space-indented `## ` as a heading,
/// for the same reason: at that indentation Markdown is rendering code. An
/// indented example agreeing with `Cargo.toml` could otherwise be selected
/// ahead of the real table.
#[test]
fn an_indented_table_example_is_code_not_the_table() {
    let readme = concat!(
        "## Ecosystem\n",
        "\n",
        "    | Component | Role | Relationship |\n",
        "    |---|---|---|\n",
        "    | `example` | sample | **Declared** in `Cargo.toml` |\n",
        "\n",
        "| Component | Role | Relationship |\n",
        "|---|---|---|\n",
        "| `real` | the table | **Declared** in `Cargo.toml` |\n",
    );

    assert_eq!(
        readme_declared_components(&readme_section(readme, "## Ecosystem")),
        vec!["real".to_owned()],
        "the indented sample must not be read as the dependency table",
    );
}

/// A marker inside a link destination is not a declaration.
///
/// `[not declared](https://example.invalid/**Declared**)` renders as the words
/// "not declared". The URL is never shown, so a marker hidden in it is not a
/// marker a reader can see.
#[test]
fn a_marker_in_a_link_destination_is_not_a_declaration() {
    assert_eq!(
        outside_link_destinations("[not declared](https://example.invalid/**Declared**)"),
        "[not declared]",
    );
    assert_eq!(
        outside_link_destinations("**Declared** — [#8](https://example.invalid/a(b)c) done"),
        "**Declared** — [#8] done",
        "a URL may contain balanced parentheses",
    );
    assert_eq!(
        outside_link_destinations("plain **Declared** text"),
        "plain **Declared** text",
    );
}

/// Escapes and titles inside a link do not end its destination early.
///
/// Each of these left a marker on the visible side, and each is silent: the
/// rendered cell has no **Declared** and the guard counted one anyway.
#[test]
fn a_link_destination_ends_where_commonmark_ends_it() {
    assert_eq!(
        outside_link_destinations(r#"[#8](https://example.invalid/x "a)b **Declared**")"#),
        "[#8]",
        "a `)` inside a quoted title does not close the link",
    );
    assert_eq!(
        outside_link_destinations(r"[not declared](https://host/x\)**Declared**)"),
        "[not declared]",
        "an escaped `)` is a character, not the close",
    );
    assert_eq!(
        outside_link_destinations(r"[not declared](https://host/x\(y**Declared**)"),
        "[not declared]",
        "an escaped `(` does not open a nesting level",
    );
    // Written with an escaped quote rather than a raw string on purpose:
    // Lizard -- which Codacy runs -- mis-parses a raw string holding an odd
    // number of `"`, taking the inner one as the terminator and folding the
    // next two tests into this one. That reports this function at 77 NLOC and
    // fails the build on a limit it is nowhere near.
    assert_eq!(
        outside_link_destinations("[a](https://host/a\"b) **Declared**"),
        "[a] **Declared**",
        "a quote inside the destination is an ordinary character, not a title",
    );
}

/// A path that leaves the repository through a symlink is not in it.
///
/// `escaping_paths` reads the path as written, which a symlink defeats, and
/// `exists()` follows the link -- so the guard passed on a host file the
/// repository does not ship.
///
/// Built under `CARGO_TARGET_TMPDIR` rather than `std::env::temp_dir()`. The
/// shared temp directory is world-writable with predictable names, so a test
/// that creates a symlink there is itself the attack it is testing for -- and
/// this one runs in CI. Cargo gives integration tests a private directory
/// under `target/` for exactly this.
#[cfg(unix)]
#[test]
fn a_path_through_a_symlink_out_of_the_repository_is_rejected() {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("unrooted-paths");
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    let elsewhere = base.join("elsewhere");
    std::fs::create_dir_all(&repo).expect("create the fake repository root");
    std::fs::create_dir_all(&elsewhere).expect("create the directory outside it");
    std::fs::write(repo.join("inside.txt"), "").expect("write a file inside");
    std::fs::write(elsewhere.join("outside.txt"), "").expect("write a file outside");
    std::os::unix::fs::symlink("../elsewhere", repo.join("link")).expect("link out of the repo");

    let named = ["inside.txt".to_owned(), "link/outside.txt".to_owned()];
    let unrooted = unrooted_paths(&repo, &named);

    std::fs::remove_dir_all(&base).expect("clean up");

    assert_eq!(
        unrooted,
        vec![&"link/outside.txt".to_owned()],
        "only the path that resolves outside the repository is rejected",
    );
}

/// A marker inside a raw HTML attribute is not a declaration.
///
/// `<span title="**Declared**">not declared</span>` renders as "not declared".
/// An attribute value is never page text, so a marker in one is not a marker.
#[test]
fn a_marker_in_an_html_attribute_is_not_a_declaration() {
    assert_eq!(
        outside_html_tags("<span title=\"**Declared**\">not declared</span>"),
        "not declared",
    );
    assert_eq!(
        outside_html_tags("**Declared** in `Cargo.toml`"),
        "**Declared** in `Cargo.toml`",
        "text outside any tag is untouched",
    );
    assert_eq!(
        outside_html_tags("a < b and **Declared**"),
        "a < b and **Declared**",
        "an unterminated `<` is literal text",
    );
}

/// A tab after the hashes still opens a heading.
///
/// Both directions matter: `##\tEcosystem` must be *found* as the section --
/// missing it is a loud failure on a README that renders identically -- and
/// `#\tAppendix` must *end* one, which is the silent half.
#[test]
fn a_tab_after_the_hashes_is_still_a_heading() {
    assert_eq!(
        atx_heading("##\tEcosystem").as_deref(),
        Some("## Ecosystem")
    );
    assert_eq!(
        atx_heading("##   Ecosystem").as_deref(),
        Some("## Ecosystem")
    );
    assert_eq!(atx_heading("#\tAppendix").as_deref(), Some("# Appendix"));
    assert_eq!(atx_heading("##Ecosystem"), None, "a separator is required");
    assert_eq!(atx_heading("not a heading"), None);
}

/// A guarded section written twice is an error, not a first-match.
///
/// The later copy was read by nothing: location took the first, and the
/// section ended before the duplicate. A copy-paste or a merge could add a
/// second contract that no assertion ever saw.
#[test]
#[should_panic(expected = "a guarded section must be written once")]
fn a_guarded_section_written_twice_is_rejected() {
    let readme = concat!(
        "## Ecosystem\n",
        "\n",
        "| Component | Role | Relationship |\n",
        "|---|---|---|\n",
        "| `real` | the table | **Declared** in `Cargo.toml` |\n",
        "\n",
        "## Ecosystem\n",
        "\n",
        "| Component | Role | Relationship |\n",
        "|---|---|---|\n",
        "| `other` | a second contract | **Declared** in `Cargo.toml` |\n",
    );

    readme_section(readme, "## Ecosystem");
}

/// A delimiter cell needs three dashes, not one.
#[test]
fn a_delimiter_cell_needs_three_dashes() {
    assert!(is_delimiter_row("|---|---|---|"));
    assert!(is_delimiter_row("| :--- | ---: | :---: |"));
    assert!(
        !is_delimiter_row("|-|-|-|"),
        "one dash is not a delimiter row"
    );
    assert!(!is_delimiter_row("|--|--|--|"), "two dashes are not either");
}

/// A component name hidden in an HTML attribute is not the component.
///
/// `<span title="`nir-rs`">replacement</span>` renders as "replacement". The
/// Component cell can hide a name exactly where the Relationship cell hides a
/// marker, so both now read the same rendered text.
#[test]
fn a_name_in_an_html_attribute_is_not_the_component() {
    assert_eq!(
        rendered_cell("<span title=\"`nir-rs`\">replacement</span>"),
        "replacement",
    );
    assert_eq!(
        rendered_cell("[`nir-rs`](https://crates.io/crates/nir-rs) 0.4.2"),
        "[`nir-rs`] 0.4.2",
        "a linked name still renders, so its code span must survive",
    );
}

/// A `>` inside a quoted attribute does not end the tag.
#[test]
fn a_bracket_inside_a_quoted_attribute_is_not_the_tag_end() {
    assert_eq!(
        outside_html_tags("<span title=\"> **Declared**\">not declared</span>"),
        "not declared",
    );
    assert_eq!(
        outside_html_tags("a < b"),
        "a < b",
        "an unterminated `<` stays"
    );
}

/// An image description is not page text, so a marker in one is not a marker.
#[test]
fn an_image_description_is_not_a_declaration() {
    assert_eq!(outside_images("![**Declared**](transparent.png)"), "");
    assert_eq!(
        outside_images("before ![alt](a(b)c.png) after"),
        "before  after",
        "an image destination may contain balanced parentheses",
    );
    assert_eq!(
        outside_images("[**Declared**](x) is a link, not an image"),
        "[**Declared**](x) is a link, not an image",
    );
}

/// Markdown's optional closing hashes do not change which heading a line is.
#[test]
fn closing_hashes_do_not_change_a_heading() {
    assert_eq!(
        atx_heading("## Ecosystem ##").as_deref(),
        Some("## Ecosystem")
    );
    assert_eq!(
        atx_heading("## Ecosystem  ####  ").as_deref(),
        Some("## Ecosystem")
    );
    assert_eq!(
        atx_heading("## Ecosystem##").as_deref(),
        Some("## Ecosystem##"),
        "a closing run must be preceded by whitespace, or it is part of the name",
    );
}

/// A nested directory prefixes the entries listed under it.
///
/// The tree named `src/model/graph.rs`; dropping the nesting checked
/// `src/model/` and `src/graph.rs`, which both exist, so a file the tree
/// claims and the repository lacks went unnoticed.
#[test]
fn a_nested_directory_prefixes_the_entries_below_it() {
    let readme = concat!(
        "## Files\n",
        "\n",
        "```text\n",
        "src/\n",
        "├── model/\n",
        "    └── graph.rs\n",
        "└── json.rs\n",
        "```\n",
    );

    assert_eq!(
        files_tree_paths(&readme_section(readme, "## Files")),
        vec![
            "src/".to_owned(),
            "src/model/".to_owned(),
            "src/model/graph.rs".to_owned(),
            "src/json.rs".to_owned(),
        ],
        "depth decides the prefix, and returning to it clears the nesting",
    );
}

/// An HTML comment opened on one line and closed on a later one hides every
/// line between them.
///
/// This is the state-carrying half of the two fixes above: without it a
/// multiline comment only hides its opening line, which is exactly the case a
/// single-line strip gets wrong.
#[test]
fn an_html_comment_spans_lines() {
    let mut open = false;
    let lines = ["before <!-- start", "swallowed", "end --> after"];
    let visible: Vec<String> = lines
        .iter()
        .map(|line| visible_outside_comments(line, &mut open))
        .collect();

    assert_eq!(visible, vec!["before ", "", " after"]);
    assert!(!open, "the comment closed, so nothing should still be open");
}

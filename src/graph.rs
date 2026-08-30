// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mapping [`SnnModel`] onto a `nir-rs` [`NirGraph`].
//!
//! The graph is the canonical NIR shape for a feed-forward layer of leaky
//! integrate-and-fire units:
//!
//! ```text
//! Input [16] ──▶ Linear [16×16] ──▶ LIF [16] ──▶ Output [16]
//! ```
//!
//! four nodes and three edges, using the NIR wire type names `Input`, `Linear`,
//! `LIF` and `Output`. Parameter mapping:
//!
//! | `snn_model.json`     | NIR parameter     | Notes                                                    |
//! | -------------------- | ----------------- | -------------------------------------------------------- |
//! | `weights`            | `Linear.weight`   | the learned 16×16 matrix, row-major, `y = W x`            |
//! | `decay_rate`         | `LIF.tau`         | `tau = -dt / ln(decay_rate)`, `dt = 1 / clock_hz` seconds |
//! | `decay_rate`         | `LIF.r`           | `1 / (1 - decay_rate)`; see [Membrane resistance](#membrane-resistance) |
//! | `threshold`          | `LIF.v_threshold` | copied verbatim                                          |
//! | —                    | `LIF.v_leak`      | `0.0`: the model leaks toward a zero resting potential    |
//! | —                    | `LIF.v_reset`     | absent; NIR defaults it to `zeros_like(v_threshold)`      |
//! | `membrane_potential` | —                 | simulator state, not a graph parameter                    |
//! | `last_spike`         | —                 | simulator state, not a graph parameter                    |
//!
//! # The `Linear` node
//!
//! The 256 stored weights are the layer's learned input weights: the README
//! records that the hidden layer is purely excitatory and that the network has
//! *no* recurrent feedback, so there is nothing recurrent for them to be. They
//! belong on the wire between `Input` and the population — without the
//! `Linear` node the graph is identical for every possible weight matrix, and
//! nothing a consumer runs would depend on what the model learned.
//!
//! `nir-rs` 0.4.2 offers both `Affine` (`y = W x + b`) and `Linear`
//! (`y = W x`). The model stores no bias, so `Linear` is the exact fit; an
//! `Affine` node would have to invent a zero bias tensor that is not part of
//! the artifact.
//!
//! # Membrane resistance
//!
//! NIR's LIF integrates `tau * dv/dt = (v_leak - v) + R*I`. Over one step of
//! length `dt`, holding `I` constant and with `v_leak = 0`, that is
//!
//! ```text
//! v[t+1] = decay * v[t] + R * (1 - decay) * I     where decay = exp(-dt / tau)
//! ```
//!
//! Because [`tau_from_decay`] sets `tau = -dt / ln(decay_rate)`, the state term
//! `decay` *is* the stored `decay_rate`, exactly. The input term is not: it
//! carries an extra factor of `(1 - decay_rate)`.
//!
//! The artifact this crate loads is a fixed-point LIF whose recurrence adds the
//! weighted input unscaled,
//!
//! ```text
//! v[t+1] = decay_rate * v[t] + I
//! ```
//!
//! — the form [`Neuron::decay_rate`] documents, and the form the Q8.8 hardware
//! implements: there is one per-neuron coefficient in the shipped artifacts
//! (`parameters_decay.mem`), not two, so no `(1 - decay)` input gain exists to
//! be applied.
//!
//! Leaving `R = 1` would therefore hand consumers a different network. The
//! shipped decay rates run from `0.796875` to `0.94921875`, so `(1 - decay)`
//! attenuates every input to between 20% and 5% of its trained magnitude, and
//! the units with the longest memory are starved the hardest. Setting
//! `R = 1 / (1 - decay_rate)` per unit cancels the factor and makes one NIR
//! step reproduce the model's step exactly. [`resistance_from_decay`] does it.
//!
//! # Provenance
//!
//! The `source` and `provenance` metadata keys say which artifact a graph's
//! numbers came from. That is a claim about history, and a graph builder given
//! a `&SnnModel` cannot verify it: the model may have been loaded from another
//! path, hand-constructed from public fields, or mutated after loading. So the
//! builders do not guess.
//!
//! - [`load_default_lif_graph`] stamps [`Provenance::MERGED_V2`]. It loads the
//!   shipped artifact and consumes it in the same call, so the claim holds by
//!   construction.
//! - [`build_lif_graph`] and [`build_lif_graph_with_timestep`] stamp nothing.
//!   An unlabelled graph is the honest result for parameters this crate cannot
//!   vouch for.
//! - [`build_lif_graph_with_provenance`] takes a [`Provenance`] from a caller
//!   who does know — an experiment exporter labelling its own run, say.
//!
//! Reading it back: a `provenance` value equal to [`MERGED_V2_PROVENANCE`]
//! means the shipped artifact, unmodified. Absent means unknown, *not* absent
//! means default. Anything else is the caller's own label.
//!
//! This is why the distinction is worth the extra constructor: an exported
//! graph outlives the process that built it, and a mislabelled one quietly
//! corrupts the evidence trail it exists to support.

use nir_rs::nodes::{Input, Lif, Linear, Output};
use nir_rs::types::{MetadataValue, Tensor};
use nir_rs::{NirGraph, NirNode};

use crate::model::{
    MERGED_V2_PROVENANCE, MODEL_RELATIVE_PATH, ModelError, SnnModel, TIMESTEP_SECONDS, check_q8_8,
};
#[cfg(doc)]
use crate::model::{Neuron, tau_from_decay};

/// Name of the graph's input node.
pub const INPUT_NODE: &str = "input";

/// Name of the `Linear` node carrying the model's learned input weights.
pub const LINEAR_NODE: &str = "linear";

/// Name of the graph's LIF population node.
pub const LIF_NODE: &str = "lif";

/// Name of the graph's output node.
pub const OUTPUT_NODE: &str = "output";

/// Resting potential every unit leaks toward.
const RESTING_POTENTIAL: f64 = 0.0;

/// The NIR membrane resistance that preserves the model's discrete input term.
///
/// One NIR step of a LIF with `v_leak = 0` is
/// `v[t+1] = decay * v[t] + R * (1 - decay) * I`, while the shipped model steps
/// as `v[t+1] = decay_rate * v[t] + I`. Returning `1 / (1 - decay_rate)`
/// cancels the `(1 - decay)` factor so the two agree.
///
/// See [Membrane resistance](self#membrane-resistance) for why this, and not
/// `R = 1`, is the faithful mapping.
///
/// # Errors
///
/// Returns [`ModelError::Schema`] unless `decay_rate` lies in `(0, 1)`, outside
/// which the scaling is undefined or negative. Decoding already enforces this;
/// the check covers hand-built [`SnnModel`] values.
///
/// Unlike [`tau_from_decay`], the *result* needs no check: once `decay_rate` is
/// finite and inside `(0, 1)`, the largest representable `decay_rate` is
/// `1 - 2^-53`, so `1 - decay_rate` is at least `2^-53` and never zero, and the
/// quotient is bounded by `2^53`. The reciprocal is therefore always finite and
/// strictly greater than 1. `resistance_boundary_stays_finite` pins that bound.
pub fn resistance_from_decay(decay_rate: f64) -> Result<f64, ModelError> {
    if !(decay_rate.is_finite() && decay_rate > 0.0 && decay_rate < 1.0) {
        return Err(ModelError::Schema(format!(
            "decay_rate must lie in (0, 1) to scale the membrane resistance, got {decay_rate}"
        )));
    }
    Ok(1.0 / (1.0 - decay_rate))
}

/// Which artifact a graph's parameters came from.
///
/// Stamped into [`NirGraph::metadata`] under the `source` and `provenance`
/// keys. Both are claims about where the numbers originated, so only the caller
/// who knows the answer can make them: [`build_lif_graph`] and
/// [`build_lif_graph_with_timestep`] stamp neither, and an unlabelled graph is
/// the correct result for a model this crate cannot vouch for.
///
/// See [Provenance](self#provenance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Provenance<'a> {
    /// Where the parameters were read from — the `source` metadata value.
    ///
    /// A path, URL, or experiment identifier; whatever lets a reader find the
    /// same numbers again.
    pub source: &'a str,
    /// What the artifact *is* — the `provenance` metadata value.
    ///
    /// Prose for a human reading an exported graph, including the negative
    /// claims that matter: which retrain or encoder this is *not*.
    pub description: &'a str,
}

impl Provenance<'static> {
    /// The shipped `merged_v2` artifact at [`MODEL_RELATIVE_PATH`].
    ///
    /// [`load_default_lif_graph`] stamps this because it loads that artifact
    /// itself and hands it straight to the builder, so nothing can have altered
    /// it in between. Passing it for any other model asserts something this
    /// crate cannot check — that the parameters really are the shipped ones.
    pub const MERGED_V2: Self = Self {
        source: MODEL_RELATIVE_PATH,
        description: MERGED_V2_PROVENANCE,
    };
}

/// Load the shipped `merged_v2` model and build its NIR graph.
///
/// This is the only constructor that stamps [`Provenance::MERGED_V2`], and the
/// stamp is true by construction: the model is loaded and consumed here, so no
/// caller can have modified it. The graph is the repository's 16-neuron LIF
/// artifact, not a post-exp-009 legal-encoder retrain and not session-holdout
/// 5-ch v3.
///
/// # Errors
///
/// See [`SnnModel::load_default`] and [`build_lif_graph_with_provenance`].
pub fn load_default_lif_graph() -> Result<NirGraph, ModelError> {
    build_lif_graph_with_provenance(
        &SnnModel::load_default()?,
        TIMESTEP_SECONDS,
        Some(Provenance::MERGED_V2),
    )
}

/// Build the `Input → Linear → LIF → Output` NIR graph for `model`, stepping
/// at the shipped 1 kHz clock.
///
/// The graph carries no provenance metadata; see
/// [`build_lif_graph_with_provenance`] to attach some, or
/// [`load_default_lif_graph`] for the shipped artifact.
///
/// # Errors
///
/// See [`build_lif_graph_with_timestep`].
pub fn build_lif_graph(model: &SnnModel) -> Result<NirGraph, ModelError> {
    build_lif_graph_with_timestep(model, TIMESTEP_SECONDS)
}

/// Build the `Input → Linear → LIF → Output` NIR graph for `model` at an
/// explicit simulation timestep, in seconds.
///
/// The timestep only affects `tau`: the stored decay rates are per-step
/// multipliers, so the same model integrated on a different clock has different
/// time constants.
///
/// The graph carries no provenance metadata; see
/// [`build_lif_graph_with_provenance`].
///
/// # Errors
///
/// See [`build_lif_graph_with_provenance`].
pub fn build_lif_graph_with_timestep(
    model: &SnnModel,
    timestep_seconds: f64,
) -> Result<NirGraph, ModelError> {
    build_lif_graph_with_provenance(model, timestep_seconds, None)
}

/// Build the `Input → Linear → LIF → Output` NIR graph for `model`, stamping
/// `provenance` into the graph metadata.
///
/// `None` leaves the `source` and `provenance` keys off entirely, which is what
/// every builder that cannot vouch for the model does. `timestep_seconds` is
/// always recorded: it describes how this graph was built, not which artifact
/// the parameters came from.
///
/// The returned graph has passed [`NirGraph::validate_structure`].
///
/// # Errors
///
/// - [`ModelError::Schema`] if the model has no units, a weight row is not
///   `units` long, or a decay rate cannot be inverted into a time constant or
///   scaled into a resistance
/// - [`ModelError::Nir`] if `nir-rs` rejects a parameter tensor or the graph
pub fn build_lif_graph_with_provenance(
    model: &SnnModel,
    timestep_seconds: f64,
    provenance: Option<Provenance<'_>>,
) -> Result<NirGraph, ModelError> {
    let units = model.len();
    if units == 0 {
        return Err(ModelError::Schema("model has no neurons".into()));
    }
    let shape = vec![units];

    let weight = model.weight_tensor()?;
    let lif = lif_parameters(model, timestep_seconds, &shape)?;

    let mut graph = NirGraph::new();
    insert_population_nodes(&mut graph, shape, weight, lif)?;

    graph.add_edge(INPUT_NODE, LINEAR_NODE);
    graph.add_edge(LINEAR_NODE, LIF_NODE);
    graph.add_edge(LIF_NODE, OUTPUT_NODE);

    stamp_metadata(&mut graph, timestep_seconds, provenance);

    graph.validate_structure()?;
    Ok(graph)
}

/// Build the LIF node's four parameter tensors from the model.
///
/// Every value is checked against the Q8.8 grid on the way through: the
/// shipped artifact is a Q8.8 export, so a parameter that cannot be
/// represented in Q8.8 did not come from it.
/// Check every value in a per-neuron column against the Q8.8 grid.
///
/// The shipped artifact is a Q8.8 export, so a parameter that cannot be
/// represented in Q8.8 did not come from it. Errors name the offending neuron.
/// Insert the four nodes of the population, in graph order.
fn insert_population_nodes(
    graph: &mut NirGraph,
    shape: Vec<usize>,
    weight: Tensor,
    lif: Lif,
) -> Result<(), ModelError> {
    graph.insert_node(
        INPUT_NODE,
        NirNode::Input(Input {
            shape: shape.clone(),
            metadata: Default::default(),
        }),
    )?;
    graph.insert_node(
        LINEAR_NODE,
        NirNode::Linear(Linear {
            // Row `i` weights the input channels feeding unit `i`, so the node
            // computes `y = W x` for the population below it.
            weight,
            metadata: Default::default(),
        }),
    )?;
    graph.insert_node(LIF_NODE, NirNode::Lif(lif))?;
    graph.insert_node(
        OUTPUT_NODE,
        NirNode::Output(Output {
            shape,
            metadata: Default::default(),
        }),
    )?;
    Ok(())
}

/// Membrane resistance per unit, as a tensor.
///
/// NIR steps a LIF as `v[t+1] = decay*v[t] + R*(1-decay)*I`, so `R` must be
/// `1/(1-decay)` for the input to arrive at its trained magnitude.
fn resistance_tensor(decay_rates: Vec<f64>, shape: &[usize]) -> Result<Tensor, ModelError> {
    let resistances = decay_rates
        .into_iter()
        .map(resistance_from_decay)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::from_f64(shape.to_vec(), resistances)?)
}

/// Every value in one per-unit column must sit exactly on the Q8.8 grid.
fn check_q8_8_column(field: &str, values: &[f64]) -> Result<(), ModelError> {
    for (index, &value) in values.iter().enumerate() {
        check_q8_8(&format!("neuron {index} {field}"), value)?;
    }
    Ok(())
}

fn lif_parameters(
    model: &SnnModel,
    timestep_seconds: f64,
    shape: &[usize],
) -> Result<Lif, ModelError> {
    let tau = Tensor::from_f64(shape.to_vec(), model.taus_seconds(timestep_seconds)?)?;

    let thresholds = model.thresholds();
    check_q8_8_column("threshold", &thresholds)?;
    let v_threshold = Tensor::from_f64(shape.to_vec(), thresholds)?;

    let decay_rates = model.decay_rates();
    check_q8_8_column("decay_rate", &decay_rates)?;

    Ok(Lif {
        tau,
        r: resistance_tensor(decay_rates, shape)?,
        v_leak: Tensor::from_f64(shape.to_vec(), vec![RESTING_POTENTIAL; shape[0]])?,
        v_threshold,
        // Absent on the wire: NIR reads it as zeros_like(v_threshold),
        // which is the zero baseline this model resets to.
        v_reset: None,
        metadata: Default::default(),
    })
}

/// Stamp how the graph was built, and where its parameters came from.
///
/// `timestep_seconds` is a fact about this call and is always recorded. The
/// provenance is a claim only the caller can make, so it is stamped only when
/// one was supplied.
fn stamp_metadata(graph: &mut NirGraph, timestep_seconds: f64, provenance: Option<Provenance<'_>>) {
    graph.metadata.insert(
        "timestep_seconds".into(),
        MetadataValue::F64(timestep_seconds),
    );
    if let Some(Provenance {
        source,
        description,
    }) = provenance
    {
        graph
            .metadata
            .insert("source".into(), MetadataValue::String(source.into()));
        graph.metadata.insert(
            "provenance".into(),
            MetadataValue::String(description.into()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NEURON_COUNT, Neuron};
    use nir_rs::types::TensorData;

    #[test]
    fn graph_is_input_linear_lif_output() {
        let graph = load_default_lif_graph().unwrap();
        assert_eq!(
            graph.edges,
            [
                (INPUT_NODE.to_owned(), LINEAR_NODE.to_owned()),
                (LINEAR_NODE.to_owned(), LIF_NODE.to_owned()),
                (LIF_NODE.to_owned(), OUTPUT_NODE.to_owned()),
            ],
        );
    }

    #[test]
    fn builds_four_node_three_edge_graph() {
        let graph = load_default_lif_graph().unwrap();
        assert_eq!(graph.len(), 4);
        assert_eq!(graph.edges.len(), 3);
        assert_eq!(graph.get(INPUT_NODE).unwrap().type_name(), "Input");
        assert_eq!(graph.get(LINEAR_NODE).unwrap().type_name(), "Linear");
        assert_eq!(graph.get(LIF_NODE).unwrap().type_name(), "LIF");
        assert_eq!(graph.get(OUTPUT_NODE).unwrap().type_name(), "Output");
    }

    /// Helper: the f64 payload of a LIF parameter, or a panic.
    fn lif_values(graph: &NirGraph, pick: impl Fn(&Lif) -> &Tensor) -> Vec<f64> {
        let Some(NirNode::Lif(lif)) = graph.get(LIF_NODE) else {
            panic!("expected a LIF node");
        };
        match pick(lif).data() {
            TensorData::F64(values) => values.clone(),
            other => panic!("expected an f64 parameter, got {other:?}"),
        }
    }

    #[test]
    fn linear_node_carries_every_learned_weight() {
        let model = SnnModel::load_default().unwrap();
        let graph = build_lif_graph(&model).unwrap();
        let Some(NirNode::Linear(linear)) = graph.get(LINEAR_NODE) else {
            panic!("expected a Linear node");
        };
        assert_eq!(linear.weight.shape(), [NEURON_COUNT, NEURON_COUNT]);
        assert_eq!(linear.weight, model.weight_tensor().unwrap());
        assert_eq!(
            linear.weight.data(),
            model.weight_tensor().unwrap().data(),
            "all {} learned weights must reach the graph",
            NEURON_COUNT * NEURON_COUNT,
        );
    }

    /// The whole point of the `Linear` node: the graph must be a function of the
    /// learned weights. Before it existed, changing all 256 left the graph bit
    /// for bit identical.
    #[test]
    fn perturbing_one_weight_changes_the_graph() {
        let model = SnnModel::load_default().unwrap();
        let baseline = build_lif_graph(&model).unwrap();

        let mut perturbed = model.clone();
        // One Q8.8 code, the smallest change the hardware can represent.
        perturbed.neurons[9].weights[4] += 1.0 / 256.0;
        assert_ne!(perturbed, model);

        let changed = build_lif_graph(&perturbed).unwrap();
        assert_ne!(changed, baseline, "a changed weight must change the graph");
        assert_ne!(changed.get(LINEAR_NODE), baseline.get(LINEAR_NODE));
        // Nothing else moved: the weights only reach the graph through Linear.
        assert_eq!(changed.get(LIF_NODE), baseline.get(LIF_NODE));
        assert_eq!(changed.edges, baseline.edges);
    }

    /// NIR steps a LIF as `v[t+1] = decay*v[t] + r*(1 - decay)*I`; the model
    /// steps as `v[t+1] = decay*v[t] + I`. With `r = 1/(1 - decay)` the two
    /// agree, so one step from rest reproduces the weighted input exactly.
    ///
    /// The decay-only tests cannot see this: they never drive an input.
    #[test]
    fn one_step_from_rest_reproduces_the_weighted_input() {
        let model = SnnModel::load_default().unwrap();
        let graph = build_lif_graph(&model).unwrap();
        let Some(NirNode::Linear(linear)) = graph.get(LINEAR_NODE) else {
            panic!("expected a Linear node");
        };
        let TensorData::F64(weight) = linear.weight.data() else {
            panic!("expected an f64 weight");
        };
        let taus = lif_values(&graph, |lif| &lif.tau);
        let rs = lif_values(&graph, |lif| &lif.r);

        // A unit spike on input channel 5 and nothing else.
        let mut input = [0.0; NEURON_COUNT];
        input[5] = 1.0;

        for unit in 0..NEURON_COUNT {
            // `Linear`: I = W x.
            let current: f64 = (0..NEURON_COUNT)
                .map(|channel| weight[unit * NEURON_COUNT + channel] * input[channel])
                .sum();
            assert_eq!(current, model.neurons[unit].weights[5]);

            // `LIF`: one step from v = v_leak = 0.
            let decay = (-TIMESTEP_SECONDS / taus[unit]).exp();
            let v_next = decay * RESTING_POTENTIAL + rs[unit] * (1.0 - decay) * current;

            assert!(
                (v_next - current).abs() < 1e-12,
                "unit {unit}: one NIR step gave {v_next}, the model gives {current}",
            );
            // And the mapping is not vacuous: R = 1 would lose 80-95% of it.
            assert!(rs[unit] > 4.0, "unit {unit}: resistance {}", rs[unit]);
        }
    }

    #[test]
    fn resistance_inverts_the_discrete_input_attenuation() {
        // 1 / (1 - decay).
        assert_eq!(resistance_from_decay(0.5).unwrap(), 2.0);
        assert_eq!(resistance_from_decay(0.75).unwrap(), 4.0);
        // A slower decay needs a larger resistance to keep the input intact.
        assert!(
            resistance_from_decay(0.94921875).unwrap() > resistance_from_decay(0.796875).unwrap()
        );
        for decay in [0.0, 1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            assert!(resistance_from_decay(decay).is_err());
        }
    }

    #[test]
    fn lif_parameters_come_from_the_model() {
        let model = SnnModel::load_default().unwrap();
        let graph = build_lif_graph(&model).unwrap();
        let Some(NirNode::Lif(lif)) = graph.get(LIF_NODE) else {
            panic!("expected a LIF node");
        };
        assert_eq!(lif.v_threshold.shape(), [NEURON_COUNT]);
        assert_eq!(lif.v_threshold.data(), &TensorData::F64(model.thresholds()));

        let expected: Vec<f64> = model
            .decay_rates()
            .into_iter()
            .map(|decay| resistance_from_decay(decay).unwrap())
            .collect();
        assert_eq!(lif.r.shape(), [NEURON_COUNT]);
        assert_eq!(lif.r.data(), &TensorData::F64(expected));

        assert_eq!(lif.v_leak.data(), &TensorData::F64(vec![0.0; NEURON_COUNT]));
        assert!(lif.v_reset.is_none());
    }

    #[test]
    fn timestep_scales_tau() {
        let model = SnnModel::load_default().unwrap();
        let fast = build_lif_graph_with_timestep(&model, TIMESTEP_SECONDS).unwrap();
        let slow = build_lif_graph_with_timestep(&model, TIMESTEP_SECONDS * 2.0).unwrap();
        let taus = |graph: &NirGraph| match graph.get(LIF_NODE) {
            Some(NirNode::Lif(lif)) => match lif.tau.data() {
                TensorData::F64(values) => values.clone(),
                other => panic!("expected f64 tau, got {other:?}"),
            },
            _ => panic!("expected a LIF node"),
        };
        let (fast, slow) = (taus(&fast), taus(&slow));
        for (a, b) in fast.iter().zip(&slow) {
            assert!((b - a * 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn graph_metadata_records_provenance() {
        let graph = load_default_lif_graph().unwrap();
        assert_eq!(
            graph.metadata.get("source"),
            Some(&MetadataValue::String(MODEL_RELATIVE_PATH.into()))
        );
        assert_eq!(
            graph.metadata.get("provenance"),
            Some(&MetadataValue::String(MERGED_V2_PROVENANCE.into()))
        );
        assert_eq!(
            graph.metadata.get("timestep_seconds"),
            Some(&MetadataValue::F64(TIMESTEP_SECONDS))
        );
    }

    /// A model this crate cannot vouch for must not come out wearing the
    /// shipped artifact's label. The perturbed model is the exact case: a
    /// deliberately changed graph that used to carry unchanged provenance.
    #[test]
    fn a_perturbed_model_is_not_labelled_as_the_shipped_artifact() {
        let mut perturbed = SnnModel::load_default().unwrap();
        perturbed.neurons[9].weights[4] += 1.0 / 256.0;

        let graph = build_lif_graph(&perturbed).unwrap();
        assert_ne!(graph, load_default_lif_graph().unwrap());

        assert_eq!(
            graph.metadata.get("provenance"),
            None,
            "a changed model must not claim to be the shipped merged_v2 artifact",
        );
        assert_eq!(
            graph.metadata.get("source"),
            None,
            "nor claim to have been read from the shipped artifact's path",
        );
        // The build parameters are still recorded: they describe this call, not
        // which artifact the numbers came from.
        assert_eq!(
            graph.metadata.get("timestep_seconds"),
            Some(&MetadataValue::F64(TIMESTEP_SECONDS))
        );
    }

    /// Same rule for a model that never came from the shipped artifact at all.
    #[test]
    fn a_caller_built_model_is_not_labelled_as_the_shipped_artifact() {
        let hand_built = SnnModel {
            neurons: vec![Neuron {
                decay_rate: 0.5,
                membrane_potential: 0.0,
                threshold: 1.0,
                last_spike: false,
                weights: vec![0.25],
            }],
        };
        let graph = build_lif_graph(&hand_built).unwrap();
        assert_eq!(graph.metadata.get("provenance"), None);
        assert_eq!(graph.metadata.get("source"), None);

        // Even at the shipped timestep, and even for an unmodified load: the
        // builder cannot tell the difference, so it never guesses.
        let unlabelled = build_lif_graph(&SnnModel::load_default().unwrap()).unwrap();
        assert_eq!(unlabelled.metadata.get("provenance"), None);
        assert_eq!(unlabelled.metadata.get("source"), None);
    }

    /// A caller who does know the origin can say so, and gets their own label
    /// rather than this repository's.
    #[test]
    fn callers_can_stamp_their_own_provenance() {
        let mut model = SnnModel::load_default().unwrap();
        model.neurons[0].threshold = 2.0;

        let graph = build_lif_graph_with_provenance(
            &model,
            TIMESTEP_SECONDS,
            Some(Provenance {
                source: "exp-042/threshold-sweep.json",
                description: "threshold sweep over merged_v2; not the shipped artifact",
            }),
        )
        .unwrap();

        assert_eq!(
            graph.metadata.get("source"),
            Some(&MetadataValue::String(
                "exp-042/threshold-sweep.json".into()
            ))
        );
        let Some(MetadataValue::String(stamp)) = graph.metadata.get("provenance") else {
            panic!("expected a provenance string");
        };
        assert!(stamp.contains("threshold sweep"));
        assert_ne!(stamp, MERGED_V2_PROVENANCE);
    }

    /// The shipped stamp is exactly what `load_default_lif_graph` writes, so a
    /// reader can compare against the constant to recognise the artifact.
    #[test]
    fn merged_v2_provenance_constant_matches_the_default_graph() {
        assert_eq!(Provenance::MERGED_V2.source, MODEL_RELATIVE_PATH);
        assert_eq!(Provenance::MERGED_V2.description, MERGED_V2_PROVENANCE);

        let stamped = build_lif_graph_with_provenance(
            &SnnModel::load_default().unwrap(),
            TIMESTEP_SECONDS,
            Some(Provenance::MERGED_V2),
        )
        .unwrap();
        assert_eq!(stamped, load_default_lif_graph().unwrap());
    }

    /// `resistance_from_decay` needs no result check: the `(0, 1)` guard bounds
    /// the reciprocal at `2^53`. This pins the worst case.
    #[test]
    fn resistance_boundary_stays_finite() {
        // The largest f64 strictly below 1.0.
        let largest = 1.0 - f64::EPSILON / 2.0;
        assert!(largest < 1.0);
        let r = resistance_from_decay(largest).unwrap();
        assert!(r.is_finite() && r > 0.0, "resistance {r}");
        assert_eq!(r, 2.0_f64.powi(53));

        // And the other end: a decay just above zero gives a resistance just
        // above 1, never zero or negative.
        let r = resistance_from_decay(f64::MIN_POSITIVE).unwrap();
        assert!(r.is_finite() && r >= 1.0, "resistance {r}");
    }

    /// A timestep extreme enough to make `tau` non-finite must fail the build
    /// rather than put `inf` on the LIF node.
    #[test]
    fn rejects_timesteps_that_make_tau_non_finite() {
        let model = SnnModel::load_default().unwrap();
        let err = build_lif_graph_with_timestep(&model, f64::MAX).unwrap_err();
        assert!(err.to_string().contains("time constant"), "got {err}",);
    }

    /// Ragged weight rows must not reach the `Linear` node.
    #[test]
    fn rejects_a_model_with_ragged_weight_rows() {
        let ragged = SnnModel {
            neurons: vec![
                Neuron {
                    decay_rate: 0.5,
                    membrane_potential: 0.0,
                    threshold: 1.0,
                    last_spike: false,
                    weights: vec![1.0, 2.0, 3.0],
                },
                Neuron {
                    decay_rate: 0.5,
                    membrane_potential: 0.0,
                    threshold: 1.0,
                    last_spike: false,
                    weights: vec![4.0],
                },
            ],
        };
        let err = build_lif_graph(&ragged).unwrap_err();
        assert!(err.to_string().contains("3 weights"), "got {err}");
    }

    /// Every unit's resistance is per-neuron, not a shared constant: the
    /// shipped decay rates are graduated, so no two are the same.
    #[test]
    fn resistance_is_per_unit() {
        let graph = load_default_lif_graph().unwrap();
        let rs = lif_values(&graph, |lif| &lif.r);
        assert_eq!(rs.len(), NEURON_COUNT);
        for window in rs.windows(2) {
            assert!(
                window[1] > window[0],
                "graduated decays give strictly increasing resistances, got {rs:?}",
            );
        }
    }

    #[test]
    fn rejects_a_model_with_no_units() {
        let empty = SnnModel { neurons: vec![] };
        let err = build_lif_graph(&empty).unwrap_err();
        assert!(err.to_string().contains("no neurons"));
    }
}

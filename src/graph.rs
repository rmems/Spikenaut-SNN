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

use nir_rs::nodes::{Input, Lif, Linear, Output};
use nir_rs::types::{MetadataValue, Tensor};
use nir_rs::{NirGraph, NirNode};

use crate::model::{
    MERGED_V2_PROVENANCE, MODEL_RELATIVE_PATH, ModelError, SnnModel, TIMESTEP_SECONDS,
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
pub fn resistance_from_decay(decay_rate: f64) -> Result<f64, ModelError> {
    if !(decay_rate.is_finite() && decay_rate > 0.0 && decay_rate < 1.0) {
        return Err(ModelError::Schema(format!(
            "decay_rate must lie in (0, 1) to scale the membrane resistance, got {decay_rate}"
        )));
    }
    Ok(1.0 / (1.0 - decay_rate))
}

/// Load the shipped `merged_v2` model and build its NIR graph.
///
/// The graph metadata records [`MERGED_V2_PROVENANCE`]: this is the
/// repository's 16-neuron LIF artifact, not a post-exp-009 legal-encoder
/// retrain and not session-holdout 5-ch v3.
///
/// # Errors
///
/// See [`SnnModel::load_default`] and [`build_lif_graph`].
pub fn load_default_lif_graph() -> Result<NirGraph, ModelError> {
    build_lif_graph(&SnnModel::load_default()?)
}

/// Build the `Input → Linear → LIF → Output` NIR graph for `model`, stepping
/// at the shipped 1 kHz clock.
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
/// The returned graph has passed [`NirGraph::validate_structure`].
///
/// # Errors
///
/// - [`ModelError::Schema`] if a decay rate cannot be inverted into a time
///   constant or scaled into a resistance, or the model has no units
/// - [`ModelError::Nir`] if `nir-rs` rejects a parameter tensor or the graph
pub fn build_lif_graph_with_timestep(
    model: &SnnModel,
    timestep_seconds: f64,
) -> Result<NirGraph, ModelError> {
    let units = model.len();
    if units == 0 {
        return Err(ModelError::Schema("model has no neurons".into()));
    }
    let shape = vec![units];

    let weight = model.weight_tensor()?;
    let tau = Tensor::from_f64(shape.clone(), model.taus_seconds(timestep_seconds)?)?;
    let v_threshold = Tensor::from_f64(shape.clone(), model.thresholds())?;
    let resistances = model
        .decay_rates()
        .into_iter()
        .map(resistance_from_decay)
        .collect::<Result<Vec<_>, _>>()?;
    let r = Tensor::from_f64(shape.clone(), resistances)?;
    let v_leak = Tensor::from_f64(shape.clone(), vec![RESTING_POTENTIAL; units])?;

    let mut graph = NirGraph::new();

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
    graph.insert_node(
        LIF_NODE,
        NirNode::Lif(Lif {
            tau,
            r,
            v_leak,
            v_threshold,
            // Absent on the wire: NIR reads it as zeros_like(v_threshold),
            // which is the zero baseline this model resets to.
            v_reset: None,
            metadata: Default::default(),
        }),
    )?;
    graph.insert_node(
        OUTPUT_NODE,
        NirNode::Output(Output {
            shape,
            metadata: Default::default(),
        }),
    )?;

    graph.add_edge(INPUT_NODE, LINEAR_NODE);
    graph.add_edge(LINEAR_NODE, LIF_NODE);
    graph.add_edge(LIF_NODE, OUTPUT_NODE);

    graph.metadata.insert(
        "source".into(),
        MetadataValue::String(MODEL_RELATIVE_PATH.into()),
    );
    graph.metadata.insert(
        "provenance".into(),
        MetadataValue::String(MERGED_V2_PROVENANCE.into()),
    );
    graph.metadata.insert(
        "timestep_seconds".into(),
        MetadataValue::F64(timestep_seconds),
    );

    graph.validate_structure()?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NEURON_COUNT;
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

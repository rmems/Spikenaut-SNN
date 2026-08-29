// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mapping [`SnnModel`] onto a `nir-rs` [`NirGraph`].
//!
//! The graph is the canonical NIR shape for a single population of leaky
//! integrate-and-fire units:
//!
//! ```text
//! Input [16] ──▶ LIF [16] ──▶ Output [16]
//! ```
//!
//! three nodes and two edges, using the NIR wire type names `Input`, `LIF` and
//! `Output`. Parameter mapping:
//!
//! | `snn_model.json`     | NIR LIF parameter | Notes                                                    |
//! | -------------------- | ----------------- | -------------------------------------------------------- |
//! | `decay_rate`         | `tau`             | `tau = -dt / ln(decay_rate)`, `dt = 1 / clock_hz` seconds |
//! | `threshold`          | `v_threshold`     | copied verbatim                                          |
//! | —                    | `r`               | `1.0`: the model applies input current unscaled          |
//! | —                    | `v_leak`          | `0.0`: the model leaks toward a zero resting potential    |
//! | —                    | `v_reset`         | absent; NIR defaults it to `zeros_like(v_threshold)`      |
//! | `membrane_potential` | —                 | simulator state, not a graph parameter                    |
//! | `last_spike`         | —                 | simulator state, not a graph parameter                    |
//! | `weights`            | —                 | recurrent matrix; see [`SnnModel::weight_tensor`]         |

use nir_rs::nodes::{Input, Lif, Output};
use nir_rs::types::{MetadataValue, Tensor};
use nir_rs::{NirGraph, NirNode};

use crate::model::{
    MERGED_V2_PROVENANCE, MODEL_RELATIVE_PATH, ModelError, SnnModel, TIMESTEP_SECONDS,
};

/// Name of the graph's input node.
pub const INPUT_NODE: &str = "input";

/// Name of the graph's LIF population node.
pub const LIF_NODE: &str = "lif";

/// Name of the graph's output node.
pub const OUTPUT_NODE: &str = "output";

/// Membrane resistance applied to every unit: the model scales input current by 1.
const MEMBRANE_RESISTANCE: f64 = 1.0;

/// Resting potential every unit leaks toward.
const RESTING_POTENTIAL: f64 = 0.0;

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

/// Build the `Input → LIF → Output` NIR graph for `model`, stepping at the
/// shipped 1 kHz clock.
///
/// # Errors
///
/// See [`build_lif_graph_with_timestep`].
pub fn build_lif_graph(model: &SnnModel) -> Result<NirGraph, ModelError> {
    build_lif_graph_with_timestep(model, TIMESTEP_SECONDS)
}

/// Build the `Input → LIF → Output` NIR graph for `model` at an explicit
/// simulation timestep, in seconds.
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
///   constant, or the model has no units
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

    let tau = Tensor::from_f64(shape.clone(), model.taus_seconds(timestep_seconds)?)?;
    let v_threshold = Tensor::from_f64(shape.clone(), model.thresholds())?;
    let r = Tensor::from_f64(shape.clone(), vec![MEMBRANE_RESISTANCE; units])?;
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

    graph.add_edge(INPUT_NODE, LIF_NODE);
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
    fn builds_three_node_two_edge_graph() {
        let graph = load_default_lif_graph().unwrap();
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.get(INPUT_NODE).unwrap().type_name(), "Input");
        assert_eq!(graph.get(LIF_NODE).unwrap().type_name(), "LIF");
        assert_eq!(graph.get(OUTPUT_NODE).unwrap().type_name(), "Output");
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
        assert_eq!(lif.r.data(), &TensorData::F64(vec![1.0; NEURON_COUNT]));
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

    #[test]
    fn rejects_a_model_with_no_units() {
        let empty = SnnModel { neurons: vec![] };
        let err = build_lif_graph(&empty).unwrap_err();
        assert!(err.to_string().contains("no neurons"));
    }
}

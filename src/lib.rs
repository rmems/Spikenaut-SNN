// SPDX-License-Identifier: MIT OR Apache-2.0

//! Spikenaut-SNN → NIR.
//!
//! Spikenaut-SNN is a weights and model repository. This crate is the thin Rust
//! surface that lifts the shipped `merged_v2` model out of its ad-hoc JSON and
//! into the [Neuromorphic Intermediate Representation][nir], so the graph can
//! leave the repository in the standard interchange form.
//!
//! It has exactly one dependency, [`nir_rs`] from crates.io.
//!
//! The graph [`load_default_lif_graph`] returns is the shipped `merged_v2`
//! artifact ([`model::MERGED_V2_PROVENANCE`]): 16-neuron LIF with known
//! training-path defects. It is not a post-exp-009 legal-encoder retrain and
//! not the session-holdout 5-ch v3 encoder.
//!
//! That claim is stamped into the graph metadata only by
//! [`load_default_lif_graph`], which loads the artifact itself. Graphs built
//! from a caller's own [`SnnModel`] are unlabelled unless the caller supplies a
//! [`Provenance`]; see [Provenance](graph#provenance).
//!
//! # Example
//!
//! ```
//! use spikenaut_snn::graph::{self, INPUT_NODE, LIF_NODE, LINEAR_NODE, OUTPUT_NODE};
//! use spikenaut_snn::model::{NEURON_COUNT, SnnModel};
//!
//! let model = SnnModel::load_default()?;
//! assert_eq!(model.len(), NEURON_COUNT);
//!
//! let nir = graph::build_lif_graph(&model)?;
//! assert_eq!(nir.len(), 4);
//! assert_eq!(nir.edges.len(), 3);
//! assert_eq!(nir.get(LINEAR_NODE).unwrap().type_name(), "Linear");
//! assert_eq!(nir.get(LIF_NODE).unwrap().type_name(), "LIF");
//! assert_eq!(nir.edges, [
//!     (INPUT_NODE.to_string(), LINEAR_NODE.to_string()),
//!     (LINEAR_NODE.to_string(), LIF_NODE.to_string()),
//!     (LIF_NODE.to_string(), OUTPUT_NODE.to_string()),
//! ]);
//! # Ok::<(), spikenaut_snn::ModelError>(())
//! ```
//!
//! # Scope
//!
//! [`graph`] builds the `Input → Linear → LIF → Output` layer graph and nothing
//! else: the learned 16×16 weights ride the `Linear` node, and the LIF
//! resistances are scaled so one NIR step reproduces the model's fixed-point
//! step. Numbers are snapped onto the Q8.8 grid as they are decoded
//! ([`model::quantize_q8_8`]), so the graph carries exactly the values in the
//! `.mem` artifacts.
//!
//! Writing `.nir` files (the `nir-rs` `hdf5` feature links the system libhdf5)
//! and the output-layer weights in `parameters_output_weights.mem` belong to
//! their own tickets.
//!
//! [nir]: https://neuroir.org/

#![warn(missing_docs)]

pub mod graph;
pub mod json;
pub mod model;

pub use graph::{
    Provenance, build_lif_graph, build_lif_graph_with_provenance, load_default_lif_graph,
    resistance_from_decay,
};
pub use model::{MERGED_V2_PROVENANCE, ModelError, Neuron, SnnModel, quantize_q8_8};

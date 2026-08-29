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
//! # Example
//!
//! ```
//! use spikenaut_snn::graph::{self, INPUT_NODE, LIF_NODE, OUTPUT_NODE};
//! use spikenaut_snn::model::{NEURON_COUNT, SnnModel};
//!
//! let model = SnnModel::load_default()?;
//! assert_eq!(model.len(), NEURON_COUNT);
//!
//! let nir = graph::build_lif_graph(&model)?;
//! assert_eq!(nir.len(), 3);
//! assert_eq!(nir.edges.len(), 2);
//! assert_eq!(nir.get(LIF_NODE).unwrap().type_name(), "LIF");
//! assert_eq!(nir.edges, [
//!     (INPUT_NODE.to_string(), LIF_NODE.to_string()),
//!     (LIF_NODE.to_string(), OUTPUT_NODE.to_string()),
//! ]);
//! # Ok::<(), spikenaut_snn::ModelError>(())
//! ```
//!
//! # Scope
//!
//! [`graph`] builds the `Input → LIF → Output` population graph and nothing
//! else. The recurrent 16×16 weight matrix is decoded and exposed through
//! [`SnnModel::weight_tensor`], but placing it on a NIR `Affine` / `Linear`
//! node, writing `.nir` files (the `nir-rs` `hdf5` feature links the system
//! libhdf5), and Q8.8 / FPGA mapping all belong to their own tickets.
//!
//! [nir]: https://neuroir.org/

#![warn(missing_docs)]

pub mod graph;
pub mod json;
pub mod model;

pub use graph::{build_lif_graph, load_default_lif_graph};
pub use model::{ModelError, Neuron, SnnModel};

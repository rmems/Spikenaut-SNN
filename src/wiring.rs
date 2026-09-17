// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic 16-neuron Dale recurrent-topology experiment.
//!
//! This mesh is deliberately parallel to the shipped exp-025 feed-forward
//! input matrix. It proposes recurrent wiring; it does not reinterpret or
//! rewrite the model's weights or FPGA `.mem` layout.

use synaptic_wiring::topology::generate_small_world;
use synaptic_wiring::{
    MeshError, Polarity, SynapseDescriptor, SynapticGraph, SynapticMesh, TopologyDigest,
};

use crate::model::NEURON_COUNT;

/// First source-neuron index assigned inhibitory outgoing polarity.
pub const INHIBITORY_START: usize = 12;

/// Parameters for the deterministic small-world wiring proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DaleMeshConfig {
    /// Directed outgoing neighbors per neuron; must be even and in `[2, 15]`.
    pub neighbors: usize,
    /// Watts–Strogatz rewiring probability in `[0, 1]`.
    pub rewire_probability: f32,
    /// Maximum discrete axonal delay.
    pub max_delay: u16,
}

impl Default for DaleMeshConfig {
    fn default() -> Self {
        Self {
            neighbors: 4,
            rewire_probability: 0.1,
            max_delay: 1,
        }
    }
}

/// Host-side 12-excitatory / 4-inhibitory recurrent mesh proposal.
pub struct ExperimentalDaleMesh {
    mesh: SynapticMesh,
}

impl ExperimentalDaleMesh {
    /// Build a deterministic small-world topology with neurons 12–15
    /// inhibitory under Dale's outgoing-polarity rule.
    pub fn new(config: DaleMeshConfig) -> Result<Self, MeshError> {
        // Generate targets, delays, and magnitudes without assigning an
        // inhibitory prefix; Spikenaut's documented inhibitory bank is the
        // final four neurons, so polarity is applied explicitly below.
        let generated = generate_small_world(
            NEURON_COUNT,
            config.neighbors,
            config.rewire_probability,
            config.max_delay,
            0.0,
        )?;

        let mut descriptors = Vec::with_capacity(generated.synapse_count());
        for source in 0..NEURON_COUNT {
            let polarity = if source < INHIBITORY_START {
                Polarity::Excitatory
            } else {
                Polarity::Inhibitory
            };
            descriptors.extend(
                generated
                    .outgoing(source)
                    .map(|(target, weight, delay, _)| SynapseDescriptor {
                        source: source as u32,
                        target,
                        weight: weight.abs(),
                        delay,
                        polarity,
                    }),
            );
        }

        let graph = SynapticGraph::from_descriptors(NEURON_COUNT, &descriptors)?;
        Ok(Self {
            mesh: SynapticMesh::new(graph),
        })
    }

    /// Propagate a 16-neuron binary spike frame through one mesh tick.
    pub fn propagate(&mut self, spikes: &[bool; NEURON_COUNT]) -> Result<Vec<f32>, MeshError> {
        self.mesh.propagate(spikes)
    }

    /// Allocation-free propagation into a caller-owned 16-neuron buffer.
    pub fn propagate_into(
        &mut self,
        spikes: &[bool; NEURON_COUNT],
        output: &mut [f32; NEURON_COUNT],
    ) -> Result<(), MeshError> {
        self.mesh.propagate_into(spikes, output)
    }

    /// Reset ticks and delayed events while retaining the topology.
    pub fn reset(&mut self) {
        self.mesh.reset();
    }

    /// Shared access to the published mesh.
    #[must_use]
    pub fn inner(&self) -> &SynapticMesh {
        &self.mesh
    }

    /// Stable digest of the generated topology.
    #[must_use]
    pub fn topology_digest(&self) -> TopologyDigest {
        self.mesh.topology_digest()
    }
}

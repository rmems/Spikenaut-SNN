// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional host-side training sessions through `plasticity-lab` 0.2.1.
//!
//! [`HostTrainingSession`] applies the published trainer to a synthetic,
//! non-negative 16-LIF network built with the same dimensions and initial
//! weights as [`crate::HostNetwork`]. `plasticity-lab` 0.2.1 still depends on
//! `neuromod` 0.6, so this session names that crate (`neuromod06`) and never
//! passes a 0.7 [`neuromod::SpikingNetwork`] into the trainer. It never loads,
//! rewrites, or exports the shipped signed exp-025 bank; `SynapticDistill.jl`
//! remains the artifact-producing sidecar until an explicit parity path exists.

use neuromod06::{SeedableRng, SpikingNetwork, StdRng};
use plasticity_lab::PlasticityTrainer;
pub use plasticity_lab::{TrainerError, TrainingConfig, TrainingExample, TrainingSummary};

use crate::encode::CHANNEL_COUNT;
use crate::neuromod_host::{HOST_NETWORK_INITIAL_WEIGHT, HOST_NETWORK_NEURONS};

/// A caller-seeded `plasticity-lab` session over a synthetic neuromod 0.6 network.
pub struct HostTrainingSession {
    network: SpikingNetwork,
    rng: StdRng,
    seed: u64,
    trainer: PlasticityTrainer,
}

impl HostTrainingSession {
    /// Construct a fresh synthetic host network and trainer.
    #[must_use]
    pub fn new(seed: u64, config: TrainingConfig) -> Self {
        let mut network = SpikingNetwork::with_dimensions(HOST_NETWORK_NEURONS, 0, CHANNEL_COUNT);
        for neuron in &mut network.neurons {
            neuron.weights.fill(HOST_NETWORK_INITIAL_WEIGHT);
        }
        Self {
            network,
            rng: StdRng::seed_from_u64(seed),
            seed,
            trainer: PlasticityTrainer::new(config),
        }
    }

    /// Run a validated batch through one persistent caller-seeded RNG stream.
    pub fn run_session(
        &mut self,
        examples: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        self.trainer
            .run_session_with_rng(&mut self.network, examples, &mut self.rng)
    }

    /// Inspect the mutable in-memory experiment after a training session.
    #[must_use]
    pub fn network(&self) -> &SpikingNetwork {
        &self.network
    }

    /// Reset dynamic state, learned weights, traces, and RNG to the seed.
    pub fn reset(&mut self) {
        let config = self.trainer.config;
        *self = Self::new(self.seed, config);
    }
}

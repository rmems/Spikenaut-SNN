// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional host-side training sessions through `plasticity-lab` 0.2.1.
//!
//! [`HostTrainingSession`] is the optional trainer for the synthetic host
//! experiment. [`crate::HostNetwork`] is the neuromod 0.7 network (16 LIF,
//! uniform `2.0/16` weights). `plasticity-lab` 0.2.1 still requires neuromod
//! 0.6, so this session cannot wrap that 0.7 [`neuromod::SpikingNetwork`]. It
//! names the 0.6 crate (`neuromod06`) and builds a same-shape bank for the
//! trainer only. It never loads, rewrites, or exports the shipped signed
//! exp-025 bank; `SynapticDistill.jl` remains the artifact-producing sidecar
//! until an explicit parity path exists.

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

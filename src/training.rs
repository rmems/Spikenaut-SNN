// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional host-side training sessions through `plasticity-lab` 0.2.
//!
//! [`HostTrainingSession`] applies the published trainer only to the synthetic,
//! non-negative [`crate::HostNetwork`] experiment. It never loads, rewrites, or
//! exports the shipped signed exp-025 bank; `SynapticDistill.jl` remains the
//! artifact-producing sidecar until an explicit parity path exists.

use neuromod::SpikingNetwork;
use plasticity_lab::PlasticityTrainer;
pub use plasticity_lab::{TrainerError, TrainingConfig, TrainingExample, TrainingSummary};

use crate::HostNetwork;

/// A caller-seeded `plasticity-lab` session over the synthetic host network.
pub struct HostTrainingSession {
    network: HostNetwork,
    trainer: PlasticityTrainer,
}

impl HostTrainingSession {
    /// Construct a fresh synthetic host network and trainer.
    #[must_use]
    pub fn new(seed: u64, config: TrainingConfig) -> Self {
        Self {
            network: HostNetwork::new(seed),
            trainer: PlasticityTrainer::new(config),
        }
    }

    /// Run a validated batch through one persistent caller-seeded RNG stream.
    pub fn run_session(
        &mut self,
        examples: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        let (network, rng) = self.network.training_parts();
        self.trainer.run_session_with_rng(network, examples, rng)
    }

    /// Inspect the mutable in-memory experiment after a training session.
    #[must_use]
    pub fn network(&self) -> &SpikingNetwork {
        self.network.inner()
    }

    /// Reset dynamic state, learned weights, traces, and RNG to the seed.
    pub fn reset(&mut self) {
        self.network.reset();
    }
}

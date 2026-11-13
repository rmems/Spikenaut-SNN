// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional host-side training sessions through `plasticity-lab` 0.3.
//!
//! [`HostTrainingSession`] is the optional trainer for the synthetic host
//! experiment. [`crate::HostNetwork`] and this session now share the same
//! neuromod 0.7 [`neuromod::SpikingNetwork`] type (16 LIF, uniform
//! `2.0/16` weights): the 0.3 trainer accepts the 0.7 engine directly, so the
//! old `neuromod06` alias is gone. Its learned parameters can be exported
//! with [`HostTrainingSession::export_q8_8`]. It never loads, rewrites, or
//! exports the shipped signed exp-025 bank; `SynapticDistill.jl` remains
//! that bank's established trainer. Neither trainer parity nor FPGA parity
//! is claimed.
//!
//! The public API is the 0.3 surface: seeded sessions, frozen held-out
//! evaluation (including an isolated variant that leaves runtime and RNG
//! state untouched), atomic batch admission, non-finite reward rejection, and
//! validated `RewardMapping`. The 0.3.0 release was validated here as a
//! Git-pinned release candidate before it published (RM-1720).

use neuromod::{NeuroModulators, SeedableRng, SpikingNetwork, StdRng};
use plasticity_lab::PlasticityTrainer;
pub use plasticity_lab::{
    EvaluationExample, EvaluationSummary, RewardMapping, RewardMappingBuilder, RewardMappingError,
    TrainerError, TrainingConfig, TrainingExample, TrainingSummary,
};
use silicon_bridge::{FpgaParameterExporter, FpgaParameters, ParameterShapeError};

use crate::encode::CHANNEL_COUNT;
use crate::neuromod_host::{HOST_NETWORK_INITIAL_WEIGHT, HOST_NETWORK_NEURONS};

/// A caller-seeded `plasticity-lab` session over a synthetic neuromod 0.7 network.
pub struct HostTrainingSession {
    network: SpikingNetwork,
    rng: StdRng,
    seed: u64,
    trainer: PlasticityTrainer,
    isolated_eval_calls: u64,
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
            isolated_eval_calls: 0,
        }
    }

    /// Run a validated batch through one persistent caller-seeded RNG stream.
    ///
    /// # Errors
    ///
    /// Any [`TrainerError`] from `plasticity-lab`: an empty batch, a sample
    /// that fails preflight admission, a non-finite reward, or a network
    /// step failure. Batches rejected during admission are atomic: neither
    /// the network nor the RNG advances. A step failure after admission can
    /// leave earlier examples applied.
    pub fn run_session(
        &mut self,
        examples: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        self.trainer
            .run_session_with_rng(&mut self.network, examples, &mut self.rng)
    }

    /// Run a caller-owned held-out batch through frozen stepping.
    ///
    /// Delegates to `plasticity-lab`'s `run_eval_with_rng`: runtime dynamics
    /// and spike observations advance, but weights, eligibility traces,
    /// thresholds, decay parameters, persistent modulators, and the R-STDP
    /// configuration are preserved bit-for-bit. Runtime timestamps still
    /// advance and can influence a later training step, so evaluate on a
    /// separate session when training must resume from the original state.
    ///
    /// # Errors
    ///
    /// Any [`TrainerError`] from `plasticity-lab`: an empty batch, a sample
    /// that fails preflight admission, or a frozen-step failure. Batches
    /// rejected during admission are atomic; a frozen-step failure after
    /// admission can leave earlier examples applied.
    pub fn run_eval(
        &mut self,
        examples: &[EvaluationExample],
        modulators: &NeuroModulators,
    ) -> Result<EvaluationSummary, TrainerError> {
        self.trainer
            .run_eval_with_rng(&mut self.network, examples, modulators, &mut self.rng)
    }

    /// Run held-out evaluation on a private copy of the session state.
    ///
    /// The cloned network and a dedicated evaluation RNG derived from the
    /// session seed are used, so neither the session network nor its training
    /// RNG stream advances — including when evaluation fails after admission.
    /// Use this when later training must resume from the exact pre-evaluation
    /// state. The dedicated stream makes repeated isolated evaluations
    /// deterministic for a given seed and successful-call order; it is not
    /// the same RNG trajectory as [`Self::run_eval`].
    ///
    /// # Errors
    ///
    /// Any [`TrainerError`] from `plasticity-lab`, with the session left
    /// unchanged.
    pub fn run_eval_isolated(
        &mut self,
        examples: &[EvaluationExample],
        modulators: &NeuroModulators,
    ) -> Result<EvaluationSummary, TrainerError> {
        let mut network = clone_network(&self.network);
        let mut rng = StdRng::seed_from_u64(
            self.seed
                ^ 0x9e37_79b9_7f4a_7c15
                ^ self.isolated_eval_calls.wrapping_mul(0x0000_0001_0000_01b3),
        );
        let result = self
            .trainer
            .run_eval_with_rng(&mut network, examples, modulators, &mut rng);
        if result.is_ok() {
            self.isolated_eval_calls = self.isolated_eval_calls.wrapping_add(1);
        }
        result
    }

    /// Inspect the mutable in-memory experiment after a training session.
    #[must_use]
    pub fn network(&self) -> &SpikingNetwork {
        &self.network
    }

    /// Export the current synthetic LIF parameters as checked signed Q8.8.
    ///
    /// Returns an owned `silicon-bridge` bundle, with weights flattened in
    /// neuron-by-input order and no readout (this trainer has none). Current
    /// thresholds are captured, not their resting baselines. Neuromod's
    /// fractional **loss** is converted to the FPGA's **keep** multiplier,
    /// `1 - decay_rate`.
    ///
    /// Encoding uses silicon-bridge's truncation toward zero, not the shipped
    /// JSON decoder's nearest-grid rounding. This is a parameter snapshot,
    /// not a resumable trainer checkpoint: membrane state, traces, modulators,
    /// and RNG are excluded. Neuromod's current scaling, integration order,
    /// and lateral inhibition are not implemented by this export; matching
    /// parameter words does not establish behavioral or FPGA parity.
    ///
    /// This never mutates the session or writes files, and is also available
    /// before training or after reset.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterShapeError`] for empty or inconsistent dimensions,
    /// non-finite values, or values outside signed Q8.8 range. Invalid values
    /// are rejected rather than saturated.
    pub fn export_q8_8(&self) -> Result<FpgaParameters, ParameterShapeError> {
        let thresholds = self.network.neurons.iter().map(|n| n.threshold).collect();
        let weights = self
            .network
            .neurons
            .iter()
            .map(|n| n.weights.clone())
            .collect();
        let decay_rates = self
            .network
            .neurons
            .iter()
            .map(|n| 1.0 - n.decay_rate)
            .collect();
        FpgaParameterExporter::from_params(thresholds, weights, decay_rates).try_export()
    }

    /// Reset dynamic state, learned weights, traces, and RNG to the seed.
    pub fn reset(&mut self) {
        let config = self.trainer.config;
        *self = Self::new(self.seed, config);
    }
}

fn clone_network(network: &SpikingNetwork) -> SpikingNetwork {
    SpikingNetwork {
        neurons: network.neurons.clone(),
        iz_neurons: network.iz_neurons.clone(),
        modulators: network.modulators,
        global_step: network.global_step,
        num_channels: network.num_channels,
        input_spike_times: network.input_spike_times.clone(),
        predictive_state: network.predictive_state.clone(),
        stdp_config: network.stdp_config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_preserves_signed_words_and_neuron_major_order() {
        let mut session = HostTrainingSession::new(1, TrainingConfig::default());
        session.network.neurons[0].weights[..4].copy_from_slice(&[
            -128.0,
            i16::MAX as f32 / 256.0,
            -1.001_953_1,
            1.001_953_1,
        ]);
        session.network.neurons[1].weights[0] = -0.5;
        session.network.neurons[0].threshold = 0.5;
        session.network.neurons[0].base_threshold = 0.25;
        session.network.neurons[0].decay_rate = 0.25;

        let image = session.export_q8_8().unwrap();
        assert_eq!(&image.weights[..4], &[i16::MIN, i16::MAX, -256, 256]);
        assert_eq!(image.weights[CHANNEL_COUNT], -128);
        assert_eq!(image.thresholds[0], 128);
        assert_eq!(image.decay_rates[0], 192);
    }

    #[test]
    fn export_rejects_invalid_parameters_without_changing_them() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -256.0, 256.0] {
            for field in 0..3 {
                let mut session = HostTrainingSession::new(1, TrainingConfig::default());
                let neuron = &mut session.network.neurons[0];
                match field {
                    0 => neuron.weights[0] = value,
                    1 => neuron.threshold = value,
                    _ => neuron.decay_rate = value,
                }
                assert!(session.export_q8_8().is_err(), "field {field}: {value}");
                let neuron = &session.network.neurons[0];
                let after = match field {
                    0 => neuron.weights[0],
                    1 => neuron.threshold,
                    _ => neuron.decay_rate,
                };
                assert_eq!(after.to_bits(), value.to_bits());
                assert_eq!(session.network.global_step, 0);
            }
        }

        let mut session = HostTrainingSession::new(1, TrainingConfig::default());
        session.network.neurons[1].weights.pop();
        assert!(matches!(
            session.export_q8_8(),
            Err(ParameterShapeError::RaggedWeights { .. })
        ));
        session.network.neurons.clear();
        assert!(session.export_q8_8().is_err());
    }
}

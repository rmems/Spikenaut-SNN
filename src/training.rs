// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional host-side training sessions through `plasticity-lab` 0.3.
//!
//! [`HostTrainingSession`] is the optional trainer for the synthetic host
//! experiment. [`crate::HostNetwork`] is the neuromod 0.7 network (16 LIF,
//! uniform `2.0/16` weights). `plasticity-lab` 0.3 requires the same
//! published neuromod 0.7, so this session trains a real
//! [`neuromod::SpikingNetwork`] of that exact type — the same engine
//! [`crate::HostNetwork`] wraps. Its learned parameters can be exported with
//! [`HostTrainingSession::export_q8_8`]. It never loads, rewrites, or exports
//! the shipped signed exp-025 bank; `SynapticDistill.jl` remains that bank's
//! established trainer. Neither trainer parity nor FPGA parity is claimed.

use neuromod::{NeuroModulators, SeedableRng, SpikingNetwork, StdRng};
use plasticity_lab::PlasticityTrainer;
pub use plasticity_lab::{
    EvaluationExample, EvaluationSummary, TrainerError, TrainingConfig, TrainingExample,
    TrainingSummary,
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
    ///
    /// # Errors
    ///
    /// Any [`TrainerError`] from `plasticity-lab`: an empty batch, a sample
    /// that fails preflight admission (neither leaves the network or the RNG
    /// advanced), a non-finite reward, or a network step failure.
    pub fn run_session(
        &mut self,
        examples: &[TrainingExample],
    ) -> Result<TrainingSummary, TrainerError> {
        self.trainer
            .run_session_with_rng(&mut self.network, examples, &mut self.rng)
    }

    /// Evaluate a caller-owned held-out batch without retaining plasticity.
    ///
    /// Delegates to `plasticity-lab`'s frozen stepping: runtime dynamics and
    /// spike observations advance while weights, eligibility traces,
    /// thresholds, adaptive decay state, persistent modulators, and other
    /// plasticity-controlled state are preserved bit-for-bit. The whole batch
    /// is admitted before the first step. Spike timestamps still advance, so
    /// evaluate on a separate network if later training must be unaffected by
    /// held-out stimuli. Uses the session's persistent seeded RNG stream.
    ///
    /// # Errors
    ///
    /// Any [`TrainerError`] from `plasticity-lab`: an empty batch, a stimulus
    /// that fails preflight admission, or a network step failure.
    pub fn run_eval(
        &mut self,
        examples: &[EvaluationExample],
        modulators: &NeuroModulators,
    ) -> Result<EvaluationSummary, TrainerError> {
        self.trainer
            .run_eval_with_rng(&mut self.network, examples, modulators, &mut self.rng)
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

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side experiments over published [`neuromod`](https://crates.io/crates/neuromod) 0.6.x.
//!
//! [`HostLif`] preserves the original single-cell adapter. [`HostNetwork`]
//! exercises the checked, caller-seeded `SpikingNetwork` and its reward-
//! modulated eligibility traces using a deliberately synthetic, non-negative
//! weight bank. [`HostGifLayer`] exposes the new sparse GIF layer as a separate
//! experiment.
//!
//! ```text
//! neuromod 0.6 host experiments  ← this module
//!              ✕
//! exp-025 signed weights         ← not loaded into the non-negative engine
//! Distill / FPGA / train         ← not touched
//! ```
//!
//! [`crate::encode`] and [`crate::kinetic`] stay independent of this path.

use neuromod::{
    GifLayerError, LifNeuron, NeuroModulators, RmStdpConfig, SeedableRng, SparseGifHiddenLayer,
    SparseGifLayerConfig, SpikingNetwork, StdRng, StepError,
};

use crate::encode::CHANNEL_COUNT;

/// Number of experimental LIF neurons in [`HostNetwork`].
pub const HOST_NETWORK_NEURONS: usize = 16;

/// Uniform non-negative starting weight used by the host experiment.
///
/// Sixteen weights sum to the published engine's default L1 budget of `2.0`.
/// This value is synthetic and is not decoded from the exp-025 artifact.
pub const HOST_NETWORK_INITIAL_WEIGHT: f32 = 2.0 / CHANNEL_COUNT as f32;

/// Sparse fan-in per neuron in [`HostGifLayer`].
pub const HOST_GIF_FAN_IN: usize = 4;

/// A single host-side LIF cell wrapping `neuromod::LifNeuron`.
///
/// The wrapper exists so this crate names the published type and the step
/// contract without adopting the engine, STDP, or neuromodulator banks.
#[derive(Clone, Debug)]
pub struct HostLif {
    neuron: LifNeuron,
}

impl HostLif {
    /// Default published LIF cell (`LifNeuron::new`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            neuron: LifNeuron::new(),
        }
    }

    /// Integrate one stimulus through the published leaky-integrate step.
    pub fn integrate(&mut self, stimulus: f32) {
        self.neuron.integrate(stimulus);
    }

    /// Fire if the published threshold is crossed, then hard-reset.
    ///
    /// Returns `Some(peak_potential)` on a spike, `None` otherwise -- the
    /// `neuromod` 0.6.x `LifNeuron::check_fire` contract.
    pub fn check_fire(&mut self) -> Option<f32> {
        self.neuron.check_fire()
    }

    /// Current membrane potential after the last integrate / reset.
    #[must_use]
    pub fn membrane_potential(&self) -> f32 {
        self.neuron.membrane_potential
    }

    /// Shared access to the wrapped published neuron, for inspection.
    #[must_use]
    pub fn inner(&self) -> &LifNeuron {
        &self.neuron
    }
}

impl Default for HostLif {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic host-side exercise of `neuromod`'s network engine.
///
/// The engine has 16 LIF neurons, no Izhikevich bank, 16 input channels, and
/// uniform non-negative weights. It is intentionally not a runtime for the
/// shipped signed exp-025 bank.
pub struct HostNetwork {
    network: SpikingNetwork,
    rng: StdRng,
    seed: u64,
}

impl HostNetwork {
    /// Construct the experimental network and seed its caller-owned RNG.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut network = SpikingNetwork::with_dimensions(HOST_NETWORK_NEURONS, 0, CHANNEL_COUNT);
        for neuron in &mut network.neurons {
            neuron.weights.fill(HOST_NETWORK_INITIAL_WEIGHT);
        }
        Self {
            network,
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Advance one checked stochastic step using the persistent seeded RNG.
    pub fn step(
        &mut self,
        stimuli: &[f32; CHANNEL_COUNT],
        modulators: &NeuroModulators,
    ) -> Result<Vec<usize>, StepError> {
        self.network
            .step_with_rng(stimuli, modulators, &mut self.rng)
    }

    /// Replace the reward-modulated STDP configuration and re-tau traces.
    pub fn set_rm_stdp_config(&mut self, config: RmStdpConfig) {
        self.network.set_rm_stdp_config(config);
    }

    /// Shared access to the published network state.
    #[must_use]
    pub fn inner(&self) -> &SpikingNetwork {
        &self.network
    }

    /// Mutable access for explicit host experiments and checkpoint handling.
    pub fn inner_mut(&mut self) -> &mut SpikingNetwork {
        &mut self.network
    }

    /// Borrow the network and its persistent RNG together for crate-owned
    /// orchestration layers that must preserve the seeded replay stream.
    #[cfg(feature = "training")]
    pub(crate) fn training_parts(&mut self) -> (&mut SpikingNetwork, &mut StdRng) {
        (&mut self.network, &mut self.rng)
    }

    /// Reset dynamic state, learned weights, traces, and the RNG to the seed.
    pub fn reset(&mut self) {
        let config = self.network.stdp_config;
        *self = Self::new(self.seed);
        self.network.set_rm_stdp_config(config);
    }
}

/// Sparse 16-input, 16-neuron GIF layer showcase from `neuromod` 0.6.
///
/// This alternative neuron model is not the shipped LIF bank and never reads
/// or rewrites its model artifacts.
pub struct HostGifLayer {
    layer: SparseGifHiddenLayer,
}

impl HostGifLayer {
    /// Build the deterministic sparse layer for `seed`.
    pub fn new(seed: u64) -> Result<Self, GifLayerError> {
        let layer = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
            num_inputs: CHANNEL_COUNT,
            num_neurons: HOST_NETWORK_NEURONS,
            fan_in: HOST_GIF_FAN_IN,
            seed,
            ..SparseGifLayerConfig::default()
        })?;
        Ok(Self { layer })
    }

    /// Advance one GIF step and return the indices that fired.
    pub fn step(&mut self, stimuli: &[f32]) -> Result<Vec<usize>, GifLayerError> {
        self.layer.step(stimuli)
    }

    /// Clear dynamic GIF state while preserving topology and weights.
    pub fn reset(&mut self) {
        self.layer.reset();
    }

    /// Shared access to the published sparse layer.
    #[must_use]
    pub fn inner(&self) -> &SparseGifHiddenLayer {
        &self.layer
    }

    /// Mutable access to GIF parameters and weights.
    pub fn inner_mut(&mut self) -> &mut SparseGifHiddenLayer {
        &mut self.layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_wraps_a_published_lif_neuron() {
        let cell = HostLif::new();
        assert_eq!(cell.membrane_potential(), cell.inner().membrane_potential);
        assert!(cell.inner().threshold.is_finite());
        assert!(cell.inner().decay_rate.is_finite());
        assert!(
            cell.inner().decay_rate > 0.0,
            "a zero decay would pin the membrane and hide a missing crate"
        );
    }

    #[test]
    fn a_supthreshold_pulse_fires_and_resets() {
        let mut cell = HostLif::new();
        let threshold = cell.inner().threshold;
        let decay = cell.inner().decay_rate;
        // Published 0.6.x step: V += stimulus; V -= V * decay_rate.
        // So V' = stimulus * (1 - decay_rate) from rest. Cross that gate.
        let keep = 1.0 - decay;
        assert!(keep > 0.0, "decay_rate must leave some charge");
        cell.integrate(threshold / keep + 1.0);
        let peak = cell
            .check_fire()
            .expect("a pulse above threshold / (1 - decay) must spike");
        assert!(peak >= threshold, "peak {peak} below threshold {threshold}");
        assert_eq!(cell.membrane_potential(), 0.0);
        assert!(
            cell.check_fire().is_none(),
            "reset cell must not fire again"
        );
    }
}

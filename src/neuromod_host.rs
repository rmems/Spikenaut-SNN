// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side adapter over published [`neuromod`](https://crates.io/crates/neuromod) 0.5.x.
//!
//! This is the smallest honest wiring for issue #5: construct a real crates.io
//! [`LifNeuron`] and step it. The crate is therefore load-bearing — removing
//! it fails to compile — without rewriting Spikenaut weights, Distill, FPGA,
//! or training loops.
//!
//! ```text
//! neuromod::LifNeuron     ← this module (construct / integrate / check_fire)
//!         ✕
//! merged_v2 weights       ← not touched
//! Distill / FPGA / train  ← not touched
//! ```
//!
//! [`crate::encode`] and [`crate::kinetic`] stay independent of this path.
//! Neuromodulator / limbic-critic adapter surface is issue #10 and is not
//! opened here.

use neuromod::LifNeuron;

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
    /// `neuromod` 0.5.x `LifNeuron::check_fire` contract.
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
        // Published 0.5.x step: V += stimulus; V -= V * decay_rate.
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

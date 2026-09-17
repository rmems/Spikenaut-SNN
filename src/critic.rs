// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked reward-shaping bridge from `limbic-critic` to `neuromod`.
//!
//! This module owns only the application-level type conversion. The critic
//! crate stays independent of neuron dynamics, while domain reward collection
//! and policy/training orchestration stay outside Spikenaut-SNN.

use limbic_critic::{CriticError, CriticField, Environment, InvalidAlpha, TDCritic};
use neuromod::NeuroModulators;

/// Normalized host observation consumed by [`HostCritic`].
///
/// `limbic-critic` clamps the auxiliary channels to its documented ranges.
/// Non-finite values are rejected through `TDCritic::try_assess`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupervisorObservation {
    /// Scalar objective whose step-to-step change drives dopamine.
    pub objective: f32,
    /// Risk or volatility signal mapped to serotonin.
    pub volatility: f32,
    /// Novelty signal. The bridge validates it even though `TDCritic` derives
    /// acetylcholine from the absolute objective delta and otherwise ignores
    /// this channel.
    pub surprise: f32,
    /// System stress mapped to norepinephrine.
    pub stress: f32,
}

impl Environment for SupervisorObservation {
    fn objective(&self) -> f32 {
        self.objective
    }

    fn volatility(&self) -> f32 {
        self.volatility
    }

    fn surprise(&self) -> f32 {
        self.surprise
    }

    fn stress(&self) -> f32 {
        self.stress
    }
}

/// Stateful checked temporal-difference reward shaper.
pub struct HostCritic {
    critic: TDCritic,
}

impl HostCritic {
    /// Create a critic with EMA rate `alpha`, which must be finite and in
    /// `(0, 1]`.
    pub fn new(alpha: f32) -> Result<Self, InvalidAlpha> {
        Ok(Self {
            critic: TDCritic::new(alpha)?,
        })
    }

    /// Assess one observation without committing state on invalid input.
    ///
    /// The conversion copies the four fields exactly, including signed TD
    /// dopamine; no clamp or semantic reinterpretation is added here.
    pub fn assess(
        &mut self,
        observation: &SupervisorObservation,
    ) -> Result<NeuroModulators, CriticError> {
        if let Some(error) =
            CriticError::from_non_finite(CriticField::Surprise, observation.surprise)
        {
            return Err(error);
        }
        let values = self.critic.try_assess(observation)?;
        Ok(NeuroModulators {
            dopamine: values.dopamine,
            serotonin: values.serotonin,
            acetylcholine: values.acetylcholine,
            norepinephrine: values.norepinephrine,
        })
    }

    /// Shared access to the published critic state.
    #[must_use]
    pub fn inner(&self) -> &TDCritic {
        &self.critic
    }
}

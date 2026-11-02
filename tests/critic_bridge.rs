// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked `limbic-critic` to `neuromod` bridge contract.

use std::path::Path;

use limbic_critic::CriticField;
use spikenaut_snn::{HostCritic, SupervisorObservation};

fn observation(
    objective: f32,
    volatility: f32,
    surprise: f32,
    stress: f32,
) -> SupervisorObservation {
    SupervisorObservation {
        objective,
        volatility,
        surprise,
        stress,
    }
}

#[test]
fn checked_td_fields_map_exactly_into_neuromodulators() {
    let mut critic = HostCritic::new(1.0).unwrap();
    let modulators = critic.assess(&observation(0.5, 0.25, 0.9, 0.75)).unwrap();

    assert_eq!(modulators.dopamine, 0.5_f32.tanh());
    assert_eq!(modulators.serotonin, 0.25);
    assert_eq!(modulators.acetylcholine, 0.5_f32.tanh());
    assert_eq!(modulators.norepinephrine, 0.75);
}

#[test]
fn td_dopamine_preserves_negative_objective_changes() {
    let mut critic = HostCritic::new(1.0).unwrap();
    critic.assess(&observation(0.75, 0.0, 0.0, 0.0)).unwrap();
    let declined = critic.assess(&observation(-0.25, 0.0, 0.0, 0.0)).unwrap();

    assert_eq!(declined.dopamine, (-1.0_f32).tanh());
    assert!(declined.dopamine < 0.0);
}

#[test]
fn invalid_alpha_is_rejected_at_construction() {
    assert!(HostCritic::new(0.0).is_err());
    assert!(HostCritic::new(f32::NAN).is_err());
    assert!(HostCritic::new(1.1).is_err());
}

#[test]
fn a_failed_assessment_does_not_change_td_history() {
    let mut victim = HostCritic::new(0.5).unwrap();
    let mut control = HostCritic::new(0.5).unwrap();
    let first = observation(0.2, 0.1, 0.0, 0.1);
    assert_eq!(
        victim.assess(&first).unwrap(),
        control.assess(&first).unwrap()
    );

    let err = victim
        .assess(&observation(0.9, f32::INFINITY, 0.0, 0.0))
        .unwrap_err();
    assert_eq!(err.field(), CriticField::Volatility);

    let next = observation(0.5, 0.2, 0.0, 0.3);
    assert_eq!(
        victim.assess(&next).unwrap(),
        control.assess(&next).unwrap()
    );
}

#[test]
fn non_finite_observation_reports_its_field() {
    let cases = [
        (observation(f32::NAN, 0.0, 0.0, 0.0), CriticField::Objective),
        (
            observation(0.0, f32::NEG_INFINITY, 0.0, 0.0),
            CriticField::Volatility,
        ),
        (
            observation(0.0, 0.0, f32::INFINITY, 0.0),
            CriticField::Surprise,
        ),
        (observation(0.0, 0.0, 0.0, f32::NAN), CriticField::Stress),
    ];

    for (bad, expected_field) in cases {
        let mut critic = HostCritic::new(0.2).unwrap();
        let err = critic.assess(&bad).unwrap_err();
        assert_eq!(err.field(), expected_field);
    }
}

#[test]
fn multiple_non_finite_fields_report_the_first_observation_channel() {
    let mut critic = HostCritic::new(0.2).unwrap();
    let all_bad = observation(f32::NAN, f32::INFINITY, f32::NAN, f32::NEG_INFINITY);
    assert_eq!(
        critic.assess(&all_bad).unwrap_err().field(),
        CriticField::Objective
    );

    let auxiliary_bad = observation(0.0, f32::INFINITY, f32::NAN, f32::NEG_INFINITY);
    assert_eq!(
        critic.assess(&auxiliary_bad).unwrap_err().field(),
        CriticField::Volatility
    );
}

#[test]
fn alpha_bounds_are_half_open() {
    assert!(HostCritic::new(1.0).is_ok());
    assert!(HostCritic::new(f32::MIN_POSITIVE).is_ok());
    for alpha in [-0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(HostCritic::new(alpha).is_err(), "alpha {alpha}");
    }
}

/// `surprise` is the one channel `TDCritic::try_assess` never reads, so the
/// bridge's own preflight is all that stops it. A refusal there must leave
/// the TD history exactly where it was.
#[test]
fn a_non_finite_surprise_is_refused_without_advancing_history() {
    let mut victim = HostCritic::new(0.5).unwrap();
    let mut control = HostCritic::new(0.5).unwrap();
    let first = observation(0.4, 0.0, 0.0, 0.0);
    victim.assess(&first).unwrap();
    control.assess(&first).unwrap();

    let err = victim
        .assess(&observation(0.9, 0.0, f32::NAN, 0.0))
        .unwrap_err();
    assert_eq!(err.field(), CriticField::Surprise);

    let next = observation(-0.3, 0.1, 0.0, 0.2);
    assert_eq!(
        victim.assess(&next).unwrap(),
        control.assess(&next).unwrap()
    );
}

/// Finite surprise is validated but ignored: acetylcholine is the absolute
/// TD error, and the auxiliary channels are clamped into `[0, 1]`.
#[test]
fn surprise_is_ignored_and_auxiliary_channels_are_clamped() {
    let mut quiet = HostCritic::new(1.0).unwrap();
    let mut loud = HostCritic::new(1.0).unwrap();
    let a = quiet.assess(&observation(0.25, 2.0, 0.0, -3.0)).unwrap();
    let b = loud.assess(&observation(0.25, 2.0, 1e30, -3.0)).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.acetylcholine, 0.25_f32.tanh());
    assert_eq!(a.serotonin, 1.0);
    assert_eq!(a.norepinephrine, 0.0);
}

/// Extreme finite objectives saturate rather than fail: the overflowing TD
/// delta is clamped to `±f32::MAX`, so dopamine pins at its `[-1, 1]` ends.
#[test]
fn extreme_finite_objectives_saturate() {
    let mut critic = HostCritic::new(1.0).unwrap();
    let up = critic
        .assess(&observation(f32::MAX, 0.0, 0.0, 0.0))
        .unwrap();
    assert_eq!(up.dopamine, 1.0);
    let down = critic
        .assess(&observation(-f32::MAX, 0.0, 0.0, 0.0))
        .unwrap();
    assert_eq!(down.dopamine, -1.0);
    assert_eq!(down.acetylcholine, 1.0);
}

#[test]
fn limbic_critic_resolves_from_the_registry_at_zero_three() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "limbic-critic""#))
        .expect("Cargo.lock has limbic-critic");
    assert!(entry.contains(r#"version = "0.3."#), "{entry}");
    assert!(
        entry.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "{entry}"
    );
}

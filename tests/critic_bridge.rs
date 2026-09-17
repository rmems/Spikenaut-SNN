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

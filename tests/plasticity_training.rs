// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-gated host training contract for `plasticity-lab` 0.2.

#![cfg(feature = "training")]

use plasticity_lab::{TrainingConfig, TrainingExample};
use spikenaut_snn::{HOST_NETWORK_INITIAL_WEIGHT, HostTrainingSession};

fn rewarded_batch() -> Vec<TrainingExample> {
    vec![
        TrainingExample {
            stimuli: vec![1.0; 16],
            reward: 0.0,
        },
        TrainingExample {
            stimuli: vec![0.0; 16],
            reward: 10.0,
        },
    ]
}

fn weights(session: &HostTrainingSession) -> Vec<Vec<f32>> {
    session
        .network()
        .neurons
        .iter()
        .map(|neuron| neuron.weights.clone())
        .collect()
}

#[test]
fn a_seeded_session_reports_and_applies_real_weight_deltas() {
    let mut session = HostTrainingSession::new(99, TrainingConfig::default());
    let initial = weights(&session);

    assert!(
        initial
            .iter()
            .flatten()
            .all(|&weight| { weight == HOST_NETWORK_INITIAL_WEIGHT })
    );

    let summary = session.run_session(&rewarded_batch()).unwrap();
    let trained = weights(&session);

    assert_eq!(summary.steps_processed, 2);
    assert!(
        summary
            .weight_drifts
            .iter()
            .flatten()
            .any(|&delta| delta != 0.0),
        "the published trainer must pay out neuromod eligibility into weights"
    );
    assert_ne!(trained, initial);

    for (neuron_index, (before, after)) in initial.iter().zip(&trained).enumerate() {
        for (channel, (&before, &after)) in before.iter().zip(after).enumerate() {
            assert_eq!(
                summary.weight_drifts[neuron_index][channel],
                after - before,
                "summary drift must describe the applied update"
            );
        }
    }
}

#[test]
fn reset_replays_the_same_seeded_training_session() {
    let mut session = HostTrainingSession::new(0x5eed, TrainingConfig::default());
    let batch = rewarded_batch();

    let first_summary = session.run_session(&batch).unwrap();
    let first_weights = weights(&session);

    session.reset();

    let replay_summary = session.run_session(&batch).unwrap();
    assert_eq!(replay_summary, first_summary);
    assert_eq!(weights(&session), first_weights);
}

#[test]
fn plasticity_lab_resolves_from_crates_io_at_zero_two() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"))
        .expect("read Cargo.lock");
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "plasticity-lab""#))
        .expect("Cargo.lock has plasticity-lab");

    assert!(
        entry.contains("source = \"registry+https://github.com/rust-lang/crates.io-index\""),
        "plasticity-lab must resolve from crates.io, got:\n{entry}"
    );
    assert!(
        entry.contains("version = \"0.2."),
        "plasticity-lab must resolve to 0.2.x, got:\n{entry}"
    );
}

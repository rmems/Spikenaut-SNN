// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-gated host training contract for `plasticity-lab` 0.2.1.

#![cfg(feature = "training")]

use spikenaut_snn::{
    HOST_NETWORK_INITIAL_WEIGHT, HostNetwork, HostTrainingSession, TrainerError, TrainingConfig,
    TrainingExample,
};

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
fn a_rejected_batch_is_atomic_and_does_not_advance_the_rng() {
    let seed = 0xa70c;
    let mut rejected = HostTrainingSession::new(seed, TrainingConfig::default());
    let initial_weights = weights(&rejected);
    let invalid = vec![
        TrainingExample {
            stimuli: vec![0.5; 16],
            reward: 1.0,
        },
        TrainingExample {
            stimuli: vec![0.5; 15],
            reward: 1.0,
        },
    ];

    let error = rejected.run_session(&invalid).unwrap_err();
    assert!(
        matches!(error, TrainerError::InvalidSample { index: 1, .. }),
        "the later malformed example must fail batch admission: {error}"
    );
    assert_eq!(rejected.network().global_step, 0);
    assert_eq!(weights(&rejected), initial_weights);

    let mut fresh = HostTrainingSession::new(seed, TrainingConfig::default());
    let rejected_then_valid = rejected.run_session(&rewarded_batch()).unwrap();
    let fresh_valid = fresh.run_session(&rewarded_batch()).unwrap();
    assert_eq!(rejected_then_valid, fresh_valid);
    assert_eq!(weights(&rejected), weights(&fresh));
    assert_eq!(
        rejected.network().input_spike_times,
        fresh.network().input_spike_times
    );
    assert_eq!(
        rejected.network().predictive_state,
        fresh.network().predictive_state
    );
}

#[test]
fn reset_preserves_a_non_default_training_configuration() {
    let config = TrainingConfig {
        use_reward_modulation: false,
    };
    let mut session = HostTrainingSession::new(0xc0ffee, config);
    let batch = rewarded_batch();

    let first_summary = session.run_session(&batch).unwrap();
    let first_weights = weights(&session);
    session.reset();
    let replay_summary = session.run_session(&batch).unwrap();

    assert_eq!(replay_summary, first_summary);
    assert_eq!(weights(&session), first_weights);
}

/// The 0.6 trainer bank mirrors the 0.7 [`HostNetwork`] shape exactly.
#[test]
fn the_trainer_bank_mirrors_the_host_network_shape() {
    let session = HostTrainingSession::new(1, TrainingConfig::default());
    let host = HostNetwork::new(1);
    let (trainer, host) = (session.network(), host.inner());

    assert_eq!(trainer.neurons.len(), host.neurons.len());
    assert_eq!(trainer.iz_neurons.len(), host.iz_neurons.len());
    assert!(trainer.iz_neurons.is_empty());
    assert_eq!(trainer.num_channels, host.num_channels);
    assert_eq!(trainer.global_step, 0);
    for (trainer_neuron, host_neuron) in trainer.neurons.iter().zip(&host.neurons) {
        assert_eq!(trainer_neuron.weights, host_neuron.weights);
    }
}

#[test]
fn an_empty_batch_is_rejected_without_advancing_anything() {
    let seed = 0xe1;
    let mut rejected = HostTrainingSession::new(seed, TrainingConfig::default());
    let initial_weights = weights(&rejected);

    assert!(matches!(
        rejected.run_session(&[]),
        Err(TrainerError::EmptyBatch)
    ));
    assert_eq!(rejected.network().global_step, 0);
    assert_eq!(weights(&rejected), initial_weights);

    // The RNG did not move either: the next real batch matches a fresh run.
    let mut fresh = HostTrainingSession::new(seed, TrainingConfig::default());
    assert_eq!(
        rejected.run_session(&rewarded_batch()).unwrap(),
        fresh.run_session(&rewarded_batch()).unwrap()
    );
    assert_eq!(weights(&rejected), weights(&fresh));
}

#[test]
fn plasticity_lab_resolves_from_crates_io_at_zero_two_one() {
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
        entry.contains("version = \"0.2.1\""),
        "plasticity-lab must resolve to the reviewed 0.2.1 release, got:\n{entry}"
    );
}

#[test]
fn a_completed_session_exports_its_learned_q88_parameters() {
    let mut session = HostTrainingSession::new(99, TrainingConfig::default());
    let initial = session.export_q8_8().unwrap();
    assert_eq!(initial.weights, vec![32; 16 * 16]);

    session.run_session(&rewarded_batch()).unwrap();
    let trained_weights = weights(&session);
    let image = session.export_q8_8().unwrap();
    assert_eq!(image.metadata.num_neurons, 16);
    assert_eq!(image.metadata.num_channels, 16);
    assert!(image.output_weights.is_none(), "do not invent a readout");
    assert_ne!(image.weights, initial.weights, "real drift reaches Q8.8");
    assert_eq!(image.thresholds.len(), 16);
    assert_eq!(image.decay_rates.len(), 16);

    for (row, neuron) in session.network().neurons.iter().enumerate() {
        assert_eq!(
            image.thresholds[row],
            (neuron.threshold * 256.0).trunc() as i16
        );
        assert_eq!(
            image.decay_rates[row],
            ((1.0 - neuron.decay_rate) * 256.0).trunc() as i16
        );
        for (column, &weight) in neuron.weights.iter().enumerate() {
            assert_eq!(
                image.weights[row * 16 + column],
                (weight * 256.0).trunc() as i16
            );
        }
    }
    assert_eq!(weights(&session), trained_weights);
    assert_eq!(session.network().global_step, 2);
    assert_eq!(session.export_q8_8().unwrap().weights, image.weights);

    // Exporting between batches must not consume RNG or alter trainer state.
    let mut uninterrupted = HostTrainingSession::new(99, TrainingConfig::default());
    uninterrupted.run_session(&rewarded_batch()).unwrap();
    assert_eq!(
        session.run_session(&rewarded_batch()).unwrap(),
        uninterrupted.run_session(&rewarded_batch()).unwrap()
    );
    assert_eq!(weights(&session), weights(&uninterrupted));

    session.reset();
    assert_eq!(session.export_q8_8().unwrap().weights, initial.weights);
    session.run_session(&rewarded_batch()).unwrap();
    let replay = session.export_q8_8().unwrap();
    assert_eq!(replay.weights, image.weights);
    assert_eq!(replay.thresholds, image.thresholds);
    assert_eq!(replay.decay_rates, image.decay_rates);
}

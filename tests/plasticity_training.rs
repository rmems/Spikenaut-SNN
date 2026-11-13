// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-gated host training contract for `plasticity-lab` 0.3.

#![cfg(feature = "training")]

use neuromod::{NeuroModulators, SpikingNetwork};
use plasticity_lab::RewardMapping;
use spikenaut_snn::{
    EvaluationExample, HOST_NETWORK_INITIAL_WEIGHT, HostNetwork, HostTrainingSession, TrainerError,
    TrainingConfig, TrainingExample,
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
        ..TrainingConfig::default()
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

/// The 0.7 trainer bank is the same engine type [`HostNetwork`] wraps.
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
fn plasticity_lab_resolves_from_crates_io_at_zero_three() {
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
        entry.contains("version = \"0.3.0\""),
        "plasticity-lab must resolve to the published 0.3.0 release, got:\n{entry}"
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

#[derive(Debug, PartialEq, Eq)]
struct PlasticityBits {
    modulators: [u32; 4],
    stdp_config: [u32; 4],
    neurons: Vec<NeuronPlasticityBits>,
}

#[derive(Debug, PartialEq, Eq)]
struct NeuronPlasticityBits {
    decay_rate: u32,
    threshold: u32,
    base_threshold: u32,
    weights: Vec<u32>,
    eligibility: Vec<[u32; 2]>,
}

/// Every plasticity-controlled field `plasticity-lab` 0.3's frozen
/// evaluation contract promises to preserve bit-for-bit.
fn plasticity_bits(network: &SpikingNetwork) -> PlasticityBits {
    PlasticityBits {
        modulators: [
            network.modulators.dopamine.to_bits(),
            network.modulators.serotonin.to_bits(),
            network.modulators.acetylcholine.to_bits(),
            network.modulators.norepinephrine.to_bits(),
        ],
        stdp_config: [
            network.stdp_config.tau_eligibility.to_bits(),
            network.stdp_config.reward_lr.to_bits(),
            network.stdp_config.w_min.to_bits(),
            network.stdp_config.w_max.to_bits(),
        ],
        neurons: network
            .neurons
            .iter()
            .map(|neuron| NeuronPlasticityBits {
                decay_rate: neuron.decay_rate.to_bits(),
                threshold: neuron.threshold.to_bits(),
                base_threshold: neuron.base_threshold.to_bits(),
                weights: neuron.weights.iter().map(|value| value.to_bits()).collect(),
                eligibility: neuron
                    .eligibility
                    .iter()
                    .map(|trace| [trace.value.to_bits(), trace.tau.to_bits()])
                    .collect(),
            })
            .collect(),
    }
}

#[test]
fn frozen_evaluation_preserves_every_plasticity_bit_while_runtime_advances() {
    let mut session = HostTrainingSession::new(0xE7A1, TrainingConfig::default());
    // Learn first so weights, thresholds, traces, and modulators are all
    // non-default when evaluation runs.
    session.run_session(&rewarded_batch()).unwrap();

    let before_plasticity = plasticity_bits(session.network());
    let before_step = session.network().global_step;
    let before_prediction = session.network().predictive_state.clone();
    let before_spike_times = session.network().input_spike_times.clone();

    let held_out = vec![
        EvaluationExample {
            stimuli: vec![1.0; 16],
        },
        EvaluationExample {
            stimuli: vec![0.5; 16],
        },
        EvaluationExample {
            stimuli: vec![0.25; 16],
        },
    ];
    let modulators = NeuroModulators {
        dopamine: 0.9,
        serotonin: 0.3,
        acetylcholine: 0.7,
        norepinephrine: 0.2,
    };

    let summary = session.run_eval(&held_out, &modulators).unwrap();

    assert_eq!(summary.steps_processed, held_out.len());
    assert_eq!(
        summary.total_spikes,
        summary.per_neuron_spikes.iter().sum::<u64>()
    );
    assert_eq!(plasticity_bits(session.network()), before_plasticity);
    assert_eq!(
        session.network().global_step,
        before_step + held_out.len() as i64
    );
    assert!(
        session.network().predictive_state != before_prediction
            || session.network().input_spike_times != before_spike_times,
        "frozen evaluation must still advance runtime dynamics"
    );
}

#[test]
fn a_non_finite_reward_is_rejected_before_any_mutation() {
    for reward in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let seed = 0xA11CE_u64.wrapping_add(reward.to_bits() as u64);
        let mut rejected = HostTrainingSession::new(seed, TrainingConfig::default());
        let initial_weights = weights(&rejected);
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.5; 16],
                reward: 1.0,
            },
            TrainingExample {
                stimuli: vec![0.5; 16],
                reward,
            },
        ];

        let error = rejected.run_session(&batch).unwrap_err();
        assert!(
            matches!(error, TrainerError::NonFiniteReward { index: Some(1) }),
            "reward {reward} must fail batch admission at index 1: {error}"
        );
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
}

#[test]
fn a_custom_validated_reward_mapping_changes_the_training_outcome() {
    let mapping = RewardMapping::builder()
        .dopamine_gain(0.5)
        .positive_norepinephrine_suppression(0.2)
        .negative_norepinephrine_gain(0.4)
        .build()
        .expect("in-range coefficients must validate");
    let custom_config = TrainingConfig::default().with_reward_mapping(mapping);

    let mut custom = HostTrainingSession::new(0xC057, custom_config);
    let mut baseline = HostTrainingSession::new(0xC057, TrainingConfig::default());
    // Reward 1.0 stays inside the mapping's linear range: the default
    // dopamine gain (0.1) and the custom one (0.5) land on different
    // modulator values instead of both saturating at the [0, 1] clamp.
    let batch = vec![
        TrainingExample {
            stimuli: vec![1.0; 16],
            reward: 0.0,
        },
        TrainingExample {
            stimuli: vec![0.0; 16],
            reward: 1.0,
        },
    ];

    let custom_summary = custom.run_session(&batch).unwrap();
    let baseline_summary = baseline.run_session(&batch).unwrap();

    assert_eq!(
        custom_summary.steps_processed,
        baseline_summary.steps_processed
    );
    assert_ne!(
        custom_summary.weight_drifts, baseline_summary.weight_drifts,
        "a non-default RewardMapping must reach the network"
    );
    assert_ne!(weights(&custom), weights(&baseline));
}

#[test]
fn an_invalid_reward_mapping_is_rejected_at_construction() {
    for coefficient in [-0.1, f32::NAN, f32::INFINITY] {
        assert!(
            RewardMapping::builder()
                .dopamine_gain(coefficient)
                .build()
                .is_err(),
            "dopamine_gain {coefficient} must be rejected"
        );
    }
}

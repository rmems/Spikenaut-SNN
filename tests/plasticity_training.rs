// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-gated host training contract for `plasticity-lab` 0.3.
//!
//! Validates the RM-1720 consumer pair: the published 0.3.0 release over
//! the neuromod 0.7 engine, both resolved from crates.io.

#![cfg(feature = "training")]

use neuromod::NeuroModulators;
use spikenaut_snn::{
    EvaluationExample, HOST_NETWORK_INITIAL_WEIGHT, HostNetwork, HostTrainingSession,
    RewardMapping, RewardMappingError, TrainerError, TrainingConfig, TrainingExample,
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

/// Everything frozen stepping must preserve, in one comparable snapshot:
/// per-neuron weights, thresholds, decay rates, and eligibility traces plus
/// the persistent modulators and R-STDP configuration.
#[derive(Debug, PartialEq)]
struct PlasticitySnapshot {
    neurons: Vec<NeuronPlasticity>,
    modulators: NeuroModulators,
    stdp_config: neuromod::RmStdpConfig,
}

#[derive(Debug, PartialEq)]
struct NeuronPlasticity {
    weights: Vec<f32>,
    threshold: f32,
    base_threshold: f32,
    decay_rate: f32,
    eligibility: Vec<(f32, f32)>,
}

fn plasticity_snapshot(session: &HostTrainingSession) -> PlasticitySnapshot {
    let network = session.network();
    PlasticitySnapshot {
        neurons: network
            .neurons
            .iter()
            .map(|neuron| NeuronPlasticity {
                weights: neuron.weights.clone(),
                threshold: neuron.threshold,
                base_threshold: neuron.base_threshold,
                decay_rate: neuron.decay_rate,
                eligibility: neuron
                    .eligibility
                    .iter()
                    .map(|trace| (trace.value, trace.tau))
                    .collect(),
            })
            .collect(),
        modulators: network.modulators,
        stdp_config: network.stdp_config,
    }
}

fn runtime_snapshot(session: &HostTrainingSession) -> (i64, Vec<i64>, Vec<f32>) {
    let network = session.network();
    (
        network.global_step,
        network.input_spike_times.clone(),
        network.predictive_state.clone(),
    )
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
fn a_non_finite_reward_is_rejected_atomically_before_any_step() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let seed = 0xfeed;
        let mut rejected = HostTrainingSession::new(seed, TrainingConfig::default());
        let initial_weights = weights(&rejected);
        let initial_step = rejected.network().global_step;
        let batch = vec![
            TrainingExample {
                stimuli: vec![0.5; 16],
                reward: 1.0,
            },
            TrainingExample {
                stimuli: vec![0.5; 16],
                reward: bad,
            },
        ];

        let error = rejected.run_session(&batch).unwrap_err();
        assert_eq!(
            error,
            TrainerError::NonFiniteReward { index: Some(1) },
            "reward {bad} must name its sample index"
        );
        // Admission precedes the first step: the valid sample did not run.
        assert_eq!(rejected.network().global_step, initial_step);
        assert_eq!(weights(&rejected), initial_weights);

        // The RNG stream did not move either: the next real batch matches a
        // fresh session exactly.
        let mut fresh = HostTrainingSession::new(seed, TrainingConfig::default());
        assert_eq!(
            rejected.run_session(&rewarded_batch()).unwrap(),
            fresh.run_session(&rewarded_batch()).unwrap()
        );
        assert_eq!(weights(&rejected), weights(&fresh));
    }
}

#[test]
fn frozen_evaluation_preserves_every_plasticity_controlled_field() {
    let seed = 0xf207;
    let modulators = NeuroModulators::default();
    let mut session = HostTrainingSession::new(seed, TrainingConfig::default());

    // Give the network real plasticity state first: a training session leaves
    // non-zero eligibility traces and retuned thresholds to protect.
    session.run_session(&rewarded_batch()).unwrap();
    let before = plasticity_snapshot(&session);
    assert!(
        before
            .neurons
            .iter()
            .any(|neuron| neuron.eligibility.iter().any(|(value, _)| *value != 0.0)),
        "training must leave non-zero eligibility traces to protect"
    );
    assert!(
        before
            .neurons
            .iter()
            .any(|neuron| neuron.threshold != neuron.base_threshold),
        "training must leave retuned thresholds to protect"
    );
    let step_before = session.network().global_step;

    let held_out = vec![
        EvaluationExample {
            stimuli: vec![0.75; 16],
        },
        EvaluationExample {
            stimuli: vec![0.25; 16],
        },
        EvaluationExample {
            stimuli: vec![0.5; 16],
        },
    ];
    let summary = session.run_eval(&held_out, &modulators).unwrap();

    assert_eq!(summary.steps_processed, 3);
    assert_eq!(summary.per_neuron_spikes.len(), 16);
    assert_eq!(
        summary.total_spikes,
        summary.per_neuron_spikes.iter().sum::<u64>()
    );

    // The held-out stimuli must drive the observation: the same trained
    // state evaluated on quiet inputs cannot produce the same spike counts.
    let mut quiet_reference = HostTrainingSession::new(seed, TrainingConfig::default());
    quiet_reference.run_session(&rewarded_batch()).unwrap();
    let quiet = vec![
        EvaluationExample {
            stimuli: vec![0.0; 16],
        },
        EvaluationExample {
            stimuli: vec![0.0; 16],
        },
        EvaluationExample {
            stimuli: vec![0.0; 16],
        },
    ];
    let quiet_summary = quiet_reference.run_eval(&quiet, &modulators).unwrap();
    assert!(
        summary.total_spikes > quiet_summary.total_spikes,
        "held-out stimuli must affect evaluation: {summary:?} vs {quiet_summary:?}"
    );
    assert_ne!(summary.per_neuron_spikes, quiet_summary.per_neuron_spikes);

    let after = plasticity_snapshot(&session);
    assert_eq!(
        after, before,
        "frozen eval must not retain plasticity changes"
    );

    // Runtime dynamics did advance: the step counter and spike timestamps are
    // observable state, not plasticity.
    assert_eq!(session.network().global_step, step_before + 3);
}

#[test]
fn an_invalid_eval_batch_is_atomic_and_leaves_plasticity_untouched() {
    let modulators = NeuroModulators::default();
    let seed = 0xe7a1;
    let mut session = HostTrainingSession::new(seed, TrainingConfig::default());
    let mut reference = HostTrainingSession::new(seed, TrainingConfig::default());
    session.run_session(&rewarded_batch()).unwrap();
    reference.run_session(&rewarded_batch()).unwrap();
    let before = plasticity_snapshot(&session);
    let runtime_before = runtime_snapshot(&session);

    let invalid = vec![
        EvaluationExample {
            stimuli: vec![0.5; 16],
        },
        EvaluationExample {
            stimuli: vec![0.5; 15],
        },
    ];
    let error = session.run_eval(&invalid, &modulators).unwrap_err();
    assert!(
        matches!(error, TrainerError::InvalidSample { index: 1, .. }),
        "the later malformed example must fail eval admission: {error}"
    );
    assert_eq!(plasticity_snapshot(&session), before);
    assert_eq!(runtime_snapshot(&session), runtime_before);

    // A preflight rejection must not consume the RNG stream either: the next
    // valid evaluation matches an untouched session exactly.
    let held_out = vec![
        EvaluationExample {
            stimuli: vec![0.75; 16],
        },
        EvaluationExample {
            stimuli: vec![0.25; 16],
        },
        EvaluationExample {
            stimuli: vec![0.5; 16],
        },
    ];
    let summary = session.run_eval(&held_out, &modulators).unwrap();
    let reference_summary = reference.run_eval(&held_out, &modulators).unwrap();
    assert_eq!(summary, reference_summary);
    assert_eq!(runtime_snapshot(&session), runtime_snapshot(&reference));
    assert_eq!(
        plasticity_snapshot(&session),
        plasticity_snapshot(&reference)
    );
}

#[test]
fn isolated_evaluation_preserves_the_training_trajectory() {
    let seed = 0x150d;
    let modulators = NeuroModulators::default();
    let mut session = HostTrainingSession::new(seed, TrainingConfig::default());
    let mut untouched = HostTrainingSession::new(seed, TrainingConfig::default());
    let mut isolated_reference = HostTrainingSession::new(seed, TrainingConfig::default());
    let training_batch = rewarded_batch();
    session.run_session(&training_batch).unwrap();
    untouched.run_session(&training_batch).unwrap();
    isolated_reference.run_session(&training_batch).unwrap();
    let before = plasticity_snapshot(&session);
    let runtime_before = runtime_snapshot(&session);

    let held_out = vec![
        EvaluationExample {
            stimuli: vec![0.75; 16],
        },
        EvaluationExample {
            stimuli: vec![0.25; 16],
        },
        EvaluationExample {
            stimuli: vec![0.5; 16],
        },
    ];
    let isolated = session.run_eval_isolated(&held_out, &modulators).unwrap();
    let expected = isolated_reference
        .run_eval_isolated(&held_out, &modulators)
        .unwrap();
    assert_eq!(isolated, expected);
    assert_eq!(plasticity_snapshot(&session), before);
    assert_eq!(runtime_snapshot(&session), runtime_before);

    // Continue training on both sides of the evaluation: identical summaries
    // and weights prove the private eval did not consume network or RNG state.
    let next_batch = vec![TrainingExample {
        stimuli: vec![0.5; 16],
        reward: 2.0,
    }];
    let session_summary = session.run_session(&next_batch).unwrap();
    let untouched_summary = untouched.run_session(&next_batch).unwrap();
    assert_eq!(session_summary, untouched_summary);
    assert_eq!(weights(&session), weights(&untouched));
}

#[test]
fn a_custom_reward_mapping_changes_the_applied_updates() {
    // Gains below the defaults keep the reward-10 dopamine delta inside the
    // [0, 1] clamp (0.05 * 10 = 0.5 vs the default 1.0), and the -4 sample
    // exercises the negative-reward norepinephrine branch (0.5 * 4 = 2.0
    // clamps to 1.0 vs the default 0.8).
    let mapping = RewardMapping::builder()
        .dopamine_gain(0.05)
        .positive_norepinephrine_suppression(0.2)
        .negative_norepinephrine_gain(0.5)
        .build()
        .expect("non-negative finite coefficients validate");
    assert_eq!(mapping.dopamine_gain(), 0.05);

    let custom = TrainingConfig::default().with_reward_mapping(mapping);
    let mut custom_session = HostTrainingSession::new(0xc0a7, custom);
    let mut default_session = HostTrainingSession::new(0xc0a7, TrainingConfig::default());

    // The rewarded_batch stimuli are the proven drift-producing pattern; the
    // appended negative-reward sample exercises the NE branch of the mapping.
    let mut batch = rewarded_batch();
    batch.push(TrainingExample {
        stimuli: vec![0.5; 16],
        reward: -4.0,
    });
    let custom_summary = custom_session.run_session(&batch).unwrap();
    let default_summary = default_session.run_session(&batch).unwrap();

    assert_ne!(
        weights(&custom_session),
        weights(&default_session),
        "a validated custom mapping must steer different weight updates"
    );
    assert_ne!(custom_summary.weight_drifts, default_summary.weight_drifts);
}

#[test]
fn reward_mapping_validation_rejects_negative_and_non_finite_coefficients() {
    for (build, expected) in [
        (
            RewardMapping::builder().dopamine_gain(-0.1).build(),
            RewardMappingError::NegativeCoefficient {
                name: "dopamine_gain",
            },
        ),
        (
            RewardMapping::builder()
                .positive_norepinephrine_suppression(f32::NAN)
                .build(),
            RewardMappingError::NonFiniteCoefficient {
                name: "positive_norepinephrine_suppression",
            },
        ),
        (
            RewardMapping::builder()
                .negative_norepinephrine_gain(f32::INFINITY)
                .build(),
            RewardMappingError::NonFiniteCoefficient {
                name: "negative_norepinephrine_gain",
            },
        ),
    ] {
        assert_eq!(build.unwrap_err(), expected);
    }
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

/// The 0.3 trainer bank is the same neuromod 0.7 [`HostNetwork`] shape.
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

    let neuromod_entries: Vec<&str> = lock
        .split("[[package]]")
        .filter(|block| block.contains(r#"name = "neuromod""#))
        .collect();
    assert_eq!(
        neuromod_entries.len(),
        1,
        "the neuromod 0.6 alias is gone; exactly one neuromod may resolve"
    );
    assert!(
        neuromod_entries[0].contains("version = \"0.7.0\""),
        "neuromod must resolve to the published 0.7.0 release, got:\n{}",
        neuromod_entries[0]
    );
    assert!(
        neuromod_entries[0]
            .contains("source = \"registry+https://github.com/rust-lang/crates.io-index\""),
        "neuromod must come from the crates.io registry, got:\n{}",
        neuromod_entries[0]
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

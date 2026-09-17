// SPDX-License-Identifier: MIT OR Apache-2.0

//! Behavioral contract for the `neuromod` 0.6 host experiments.

use neuromod::{GifLayerError, NeuroModulators, NonFiniteClass, StepError};
use spikenaut_snn::{HostGifLayer, HostNetwork};

const STIMULI: [f32; 16] = [0.55; 16];

fn assert_same_network(left: &HostNetwork, right: &HostNetwork) {
    assert_eq!(left.inner().global_step, right.inner().global_step);
    assert_eq!(
        left.inner().get_membrane_potentials(),
        right.inner().get_membrane_potentials()
    );
    for (left_neuron, right_neuron) in left.inner().neurons.iter().zip(&right.inner().neurons) {
        assert_eq!(left_neuron.weights, right_neuron.weights);
        assert_eq!(left_neuron.eligibility, right_neuron.eligibility);
    }
}

#[test]
fn equal_seeds_replay_the_same_network_trajectory() {
    let mut left = HostNetwork::new(0x5eed);
    let mut right = HostNetwork::new(0x5eed);
    let modulators = NeuroModulators::default();

    for _ in 0..12 {
        assert_eq!(
            left.step(&STIMULI, &modulators).unwrap(),
            right.step(&STIMULI, &modulators).unwrap()
        );
        assert_same_network(&left, &right);
    }
}

#[test]
fn a_rejected_step_changes_neither_network_state_nor_rng_position() {
    let mut victim = HostNetwork::new(7);
    let mut control = HostNetwork::new(7);
    let modulators = NeuroModulators::default();
    let mut invalid = STIMULI;
    invalid[6] = f32::NAN;

    assert_eq!(
        victim.step(&invalid, &modulators),
        Err(StepError::NonFiniteStimulus {
            index: 6,
            class: NonFiniteClass::Nan,
        })
    );
    assert_same_network(&victim, &control);

    assert_eq!(
        victim.step(&STIMULI, &modulators).unwrap(),
        control.step(&STIMULI, &modulators).unwrap()
    );
    assert_same_network(&victim, &control);
}

#[test]
fn dopamine_pays_out_eligibility_that_accumulated_without_reward() {
    let mut network = HostNetwork::new(99);
    let initial_weights = network.inner().neurons[0].weights.clone();
    let no_reward = NeuroModulators::default();

    network.step(&[1.0; 16], &no_reward).unwrap();
    assert_eq!(network.inner().neurons[0].weights, initial_weights);
    assert!(
        network.inner().neurons[0]
            .eligibility
            .iter()
            .any(|trace| trace.value != 0.0)
    );

    let rewarded = NeuroModulators {
        dopamine: 1.0,
        ..NeuroModulators::default()
    };
    network.step(&[0.0; 16], &rewarded).unwrap();
    assert_ne!(network.inner().neurons[0].weights, initial_weights);
    assert!(
        network.inner().neurons[0]
            .weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
    );
}

#[test]
fn reset_replays_the_original_seeded_run() {
    let mut network = HostNetwork::new(42);
    let modulators = NeuroModulators::default();
    let first: Vec<Vec<usize>> = (0..8)
        .map(|_| network.step(&STIMULI, &modulators).unwrap())
        .collect();

    network.reset();
    let replay: Vec<Vec<usize>> = (0..8)
        .map(|_| network.step(&STIMULI, &modulators).unwrap())
        .collect();

    assert_eq!(replay, first);
}

#[test]
fn equal_gif_seeds_have_identical_topology_and_replay() {
    let mut left = HostGifLayer::new(123).unwrap();
    let mut right = HostGifLayer::new(123).unwrap();

    assert_eq!(left.inner(), right.inner());
    for _ in 0..10 {
        assert_eq!(
            left.step(&[1.0; 16]).unwrap(),
            right.step(&[1.0; 16]).unwrap()
        );
        assert_eq!(left.inner(), right.inner());
    }

    left.reset();
    right.reset();
    assert_eq!(
        left.step(&[1.0; 16]).unwrap(),
        right.step(&[1.0; 16]).unwrap()
    );
}

#[test]
fn gif_layer_rejects_the_wrong_input_width() {
    let mut layer = HostGifLayer::new(1).unwrap();
    assert_eq!(
        layer.step(&[1.0; 15]),
        Err(GifLayerError::InputLenMismatch {
            expected: 16,
            got: 15,
        })
    );
    assert_eq!(layer.inner().step_count(), 0);
}

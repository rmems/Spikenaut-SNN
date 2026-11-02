// SPDX-License-Identifier: MIT OR Apache-2.0

//! Behavioral contract for the `neuromod` 0.7 host experiments.

use neuromod::{
    ChannelVector, GifLayerError, NeuroModulators, NonFiniteClass, RmStdpConfig, StepError,
};
use spikenaut_snn::neuromod_host::{HOST_GIF_FAN_IN, HOST_NETWORK_NEURONS};
use spikenaut_snn::{CHANNEL_COUNT, HOST_NETWORK_INITIAL_WEIGHT, HostGifLayer, HostNetwork};

const STIMULI: [f32; 16] = [0.55; 16];

#[test]
fn a_fresh_network_has_the_documented_synthetic_shape() {
    let network = HostNetwork::new(1);
    let inner = network.inner();
    assert_eq!(inner.neurons.len(), HOST_NETWORK_NEURONS);
    assert!(inner.iz_neurons.is_empty(), "no Izhikevich bank");
    assert_eq!(inner.num_channels, CHANNEL_COUNT);
    assert_eq!(inner.global_step, 0);
    for (index, neuron) in inner.neurons.iter().enumerate() {
        assert_eq!(neuron.weights.len(), CHANNEL_COUNT, "neuron {index}");
        assert!(
            neuron
                .weights
                .iter()
                .all(|&weight| weight == HOST_NETWORK_INITIAL_WEIGHT),
            "neuron {index} must start uniform and non-negative",
        );
        // The documented L1 budget: sixteen equal weights summing to 2.0.
        let l1: f32 = neuron.weights.iter().sum();
        assert!((l1 - 2.0).abs() < 1e-6, "neuron {index} L1 = {l1}");
    }
}

#[test]
fn gif_layer_has_the_documented_sparse_topology() {
    let layer = HostGifLayer::new(77).unwrap();
    let inner = layer.inner();
    assert_eq!(inner.num_inputs(), CHANNEL_COUNT);
    assert_eq!(inner.num_neurons(), HOST_NETWORK_NEURONS);
    assert_eq!(inner.num_synapses(), HOST_NETWORK_NEURONS * HOST_GIF_FAN_IN);
    assert_eq!(inner.seed(), 77);
    for neuron in 0..HOST_NETWORK_NEURONS {
        let (sources, weights) = inner.fan_in_of(neuron).expect("in range");
        assert_eq!(sources.len(), HOST_GIF_FAN_IN, "neuron {neuron}");
        assert_eq!(weights.len(), HOST_GIF_FAN_IN, "neuron {neuron}");
        assert!(
            sources
                .iter()
                .all(|&source| (source as usize) < CHANNEL_COUNT),
            "neuron {neuron} reads past the input width: {sources:?}",
        );
    }
    assert!(inner.fan_in_of(HOST_NETWORK_NEURONS).is_none());
}

#[test]
fn gif_reset_replays_a_fresh_layer() {
    let mut used = HostGifLayer::new(9).unwrap();
    let mut fresh = HostGifLayer::new(9).unwrap();
    for _ in 0..5 {
        used.step(&[1.0; 16]).unwrap();
    }
    assert_eq!(used.inner().step_count(), 5);

    used.reset();
    assert_eq!(used.inner().step_count(), 0);
    assert_eq!(
        used.inner(),
        fresh.inner(),
        "reset restores the seeded layer"
    );
    for _ in 0..5 {
        assert_eq!(
            used.step(&[1.0; 16]).unwrap(),
            fresh.step(&[1.0; 16]).unwrap()
        );
    }
}

fn assert_same_network(left: &HostNetwork, right: &HostNetwork) {
    let left = left.inner();
    let right = right.inner();
    assert_eq!(left.global_step, right.global_step);
    assert_eq!(left.num_channels, right.num_channels);
    assert_eq!(left.input_spike_times, right.input_spike_times);
    assert_eq!(left.predictive_state, right.predictive_state);
    assert_eq!(left.modulators, right.modulators);
    assert_eq!(left.stdp_config, right.stdp_config);
    assert_eq!(left.neurons.len(), right.neurons.len());
    assert_eq!(left.iz_neurons.len(), right.iz_neurons.len());
    for (left_neuron, right_neuron) in left.neurons.iter().zip(&right.neurons) {
        assert_eq!(
            left_neuron.membrane_potential,
            right_neuron.membrane_potential
        );
        assert_eq!(left_neuron.decay_rate, right_neuron.decay_rate);
        assert_eq!(left_neuron.threshold, right_neuron.threshold);
        assert_eq!(left_neuron.base_threshold, right_neuron.base_threshold);
        assert_eq!(left_neuron.last_spike, right_neuron.last_spike);
        assert_eq!(left_neuron.last_spike_time, right_neuron.last_spike_time);
        assert_eq!(left_neuron.weights, right_neuron.weights);
        assert_eq!(left_neuron.eligibility, right_neuron.eligibility);
    }
}

/// Compile-time lock: a new `StepError` variant fails this crate until named.
fn assert_known_step_error(error: StepError) {
    match error {
        StepError::InputLenMismatch { .. }
        | StepError::NonFiniteStimulus { .. }
        | StepError::NonFiniteModulator { .. }
        | StepError::NonFinitePredictiveState { .. }
        | StepError::StepCounterExhausted { .. }
        | StepError::CheckpointShapeMismatch { .. } => {}
    }
}

/// Compile-time lock: a new `GifLayerError` variant fails this crate until named.
fn assert_known_gif_layer_error(error: GifLayerError) {
    match error {
        GifLayerError::NonFiniteParam { .. }
        | GifLayerError::NonFiniteWeight { .. }
        | GifLayerError::NonFiniteInput { .. }
        | GifLayerError::NumericOverflow { .. }
        | GifLayerError::FanInExceedsInputs { .. }
        | GifLayerError::InvalidWeightRange { .. }
        | GifLayerError::InputLenMismatch { .. }
        | GifLayerError::OutputLenMismatch { .. }
        | GifLayerError::SourceOutOfRange { .. }
        | GifLayerError::TooManyInputs { .. }
        | GifLayerError::MalformedCheckpoint { .. }
        | GifLayerError::StepCounterExhausted { .. }
        | GifLayerError::RasterTooLarge { .. } => {}
    }
}

#[test]
fn published_error_enums_name_the_0_7_variants() {
    assert_known_step_error(StepError::InputLenMismatch {
        expected: 16,
        got: 1,
    });
    assert_known_gif_layer_error(GifLayerError::InputLenMismatch {
        expected: 16,
        got: 15,
    });
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
fn a_rejected_modulator_changes_neither_network_state_nor_rng_position() {
    let mut victim = HostNetwork::new(17);
    let mut control = HostNetwork::new(17);
    let invalid = NeuroModulators {
        dopamine: f32::NAN,
        ..NeuroModulators::default()
    };

    assert!(victim.step(&STIMULI, &invalid).is_err());
    assert_same_network(&victim, &control);

    let valid = NeuroModulators::default();
    assert_eq!(
        victim.step(&STIMULI, &valid).unwrap(),
        control.step(&STIMULI, &valid).unwrap()
    );
    assert_same_network(&victim, &control);
}

#[test]
fn a_non_finite_predictive_state_is_rejected_without_mutation() {
    let mut victim = HostNetwork::new(21);
    let mut control = HostNetwork::new(21);
    let modulators = NeuroModulators::default();
    victim.inner_mut().predictive_state[2] = f32::NAN;

    assert_eq!(
        victim.step(&STIMULI, &modulators),
        Err(StepError::NonFinitePredictiveState {
            index: 2,
            class: NonFiniteClass::Nan,
        })
    );
    assert_eq!(victim.inner().global_step, 0);
    assert_eq!(
        victim.inner().input_spike_times,
        control.inner().input_spike_times
    );

    victim.inner_mut().predictive_state[2] = 0.0;
    assert_eq!(
        victim.step(&STIMULI, &modulators).unwrap(),
        control.step(&STIMULI, &modulators).unwrap()
    );
    assert_same_network(&victim, &control);
}

#[test]
fn a_malformed_checkpoint_shape_is_rejected_without_mutation() {
    let mut victim = HostNetwork::new(23);
    let control = HostNetwork::new(23);
    let modulators = NeuroModulators::default();
    victim.inner_mut().predictive_state.pop();

    assert_eq!(
        victim.step(&STIMULI, &modulators),
        Err(StepError::CheckpointShapeMismatch {
            field: ChannelVector::PredictiveState,
            expected: 16,
            got: 15,
        })
    );
    assert_eq!(victim.inner().global_step, 0);
    assert_eq!(victim.inner().predictive_state.len(), 15);
    assert_eq!(
        victim.inner().input_spike_times,
        control.inner().input_spike_times
    );
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
fn reset_preserves_the_selected_r_stdp_configuration() {
    let config = RmStdpConfig {
        tau_eligibility: 91.0,
        reward_lr: 0.025,
        w_min: 0.01,
        w_max: 1.25,
    };
    let mut network = HostNetwork::new(4242);
    let mut fresh_control = HostNetwork::new(4242);
    network.set_rm_stdp_config(config);
    fresh_control.set_rm_stdp_config(config);

    for _ in 0..4 {
        network
            .step(
                &STIMULI,
                &NeuroModulators {
                    dopamine: 0.5,
                    ..NeuroModulators::default()
                },
            )
            .unwrap();
    }
    network.reset();

    assert_eq!(network.inner().stdp_config, config);
    assert_same_network(&network, &fresh_control);
    assert!(network.inner().neurons.iter().all(|neuron| {
        neuron
            .eligibility
            .iter()
            .all(|trace| trace.tau == config.tau_eligibility)
    }));
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

#[test]
fn gif_layer_rejects_a_non_finite_input_without_advancing() {
    let mut layer = HostGifLayer::new(3).unwrap();
    let mut invalid = [1.0; 16];
    invalid[4] = f32::NAN;
    assert_eq!(
        layer.step(&invalid),
        Err(GifLayerError::NonFiniteInput {
            index: 4,
            class: NonFiniteClass::Nan,
        })
    );
    assert_eq!(layer.inner().step_count(), 0);
}

#[test]
fn gif_layer_rejects_a_non_finite_weight_without_advancing() {
    let mut layer = HostGifLayer::new(5).unwrap();
    layer.inner_mut().weights_mut()[0] = f32::INFINITY;
    assert_eq!(
        layer.step(&[1.0; 16]),
        Err(GifLayerError::NonFiniteWeight {
            index: 0,
            class: NonFiniteClass::PosInfinity,
        })
    );
    assert_eq!(layer.inner().step_count(), 0);
}

#[test]
fn gif_layer_rejects_a_non_finite_parameter_without_advancing() {
    let mut layer = HostGifLayer::new(7).unwrap();
    let mut params = *layer.inner().params();
    params.leak = f32::NAN;
    *layer.inner_mut().params_mut() = params;
    assert_eq!(
        layer.step(&[1.0; 16]),
        Err(GifLayerError::NonFiniteParam {
            field: "leak",
            class: NonFiniteClass::Nan,
        })
    );
    assert_eq!(layer.inner().step_count(), 0);
}

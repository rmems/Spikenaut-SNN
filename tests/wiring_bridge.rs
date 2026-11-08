// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic 12:4 Dale recurrent-topology experiment.

//! This bridge is not a router integration test. The shipped consumer is the
//! five-column analog exp-025 bank, whose unused axons remain zero; the
//! historical 16-channel coin map has no compatible consumer and is retired.

use std::path::Path;

use spikenaut_snn::{DaleMeshConfig, ExperimentalDaleMesh};
use synaptic_wiring::Polarity;

#[test]
fn default_mesh_is_sixteen_neurons_with_four_edges_each() {
    let mesh = ExperimentalDaleMesh::new(DaleMeshConfig::default()).unwrap();
    assert_eq!(mesh.inner().neuron_count(), 16);
    assert_eq!(mesh.inner().synapse_count(), 64);
    assert_eq!(mesh.inner().mean_degree(), 4.0);
}

#[test]
fn the_last_four_sources_are_inhibitory_and_the_first_twelve_excitatory() {
    let mesh = ExperimentalDaleMesh::new(DaleMeshConfig::default()).unwrap();

    for source in 0..16 {
        let expected = if source < 12 {
            Polarity::Excitatory
        } else {
            Polarity::Inhibitory
        };
        let outgoing: Vec<_> = mesh.inner().graph().outgoing(source).collect();
        assert_eq!(outgoing.len(), 4, "source {source}");
        for (_, weight, _, polarity) in outgoing {
            assert_eq!(polarity, expected, "source {source}");
            match expected {
                Polarity::Excitatory => assert!(weight > 0.0),
                Polarity::Inhibitory => assert!(weight < 0.0),
            }
        }
    }
}

#[test]
fn identical_configs_have_identical_topology_digests() {
    let left = ExperimentalDaleMesh::new(DaleMeshConfig::default()).unwrap();
    let right = ExperimentalDaleMesh::new(DaleMeshConfig::default()).unwrap();
    assert_eq!(left.topology_digest(), right.topology_digest());
}

#[test]
fn delay_one_delivery_preserves_source_polarity() {
    let mut excitatory = ExperimentalDaleMesh::new(DaleMeshConfig::default()).unwrap();
    let mut source_zero = [false; 16];
    source_zero[0] = true;
    assert_eq!(excitatory.propagate(&source_zero).unwrap(), vec![0.0; 16]);
    let delivered = excitatory.propagate(&[false; 16]).unwrap();
    assert_eq!(delivered.iter().filter(|&&value| value > 0.0).count(), 4);
    assert!(delivered.iter().all(|&value| value >= 0.0));

    let mut inhibitory = ExperimentalDaleMesh::new(DaleMeshConfig::default()).unwrap();
    let mut source_twelve = [false; 16];
    source_twelve[12] = true;
    assert_eq!(inhibitory.propagate(&source_twelve).unwrap(), vec![0.0; 16]);
    let delivered = inhibitory.propagate(&[false; 16]).unwrap();
    assert_eq!(delivered.iter().filter(|&&value| value < 0.0).count(), 4);
    assert!(delivered.iter().all(|&value| value <= 0.0));
}

#[test]
fn reset_clears_in_flight_spikes_and_tick() {
    let mut mesh = ExperimentalDaleMesh::new(DaleMeshConfig::default()).unwrap();
    let mut spike = [false; 16];
    spike[3] = true;
    mesh.propagate(&spike).unwrap();
    assert_eq!(mesh.inner().tick(), 1);

    mesh.reset();
    assert_eq!(mesh.inner().tick(), 0);
    assert_eq!(mesh.propagate(&[false; 16]).unwrap(), vec![0.0; 16]);
}

#[test]
fn invalid_small_world_configuration_is_rejected() {
    assert!(
        ExperimentalDaleMesh::new(DaleMeshConfig {
            neighbors: 3,
            ..DaleMeshConfig::default()
        })
        .is_err()
    );
    assert!(
        ExperimentalDaleMesh::new(DaleMeshConfig {
            rewire_probability: 1.1,
            ..DaleMeshConfig::default()
        })
        .is_err()
    );
}

/// The boundaries the `neighbors` doc promises: even values in `[2, 15]`
/// build, and NaN is not a probability.
#[test]
fn neighbor_and_probability_bounds_match_the_documentation() {
    for neighbors in [2, 14] {
        let mesh = ExperimentalDaleMesh::new(DaleMeshConfig {
            neighbors,
            ..DaleMeshConfig::default()
        })
        .unwrap_or_else(|err| panic!("neighbors {neighbors}: {err}"));
        assert_eq!(mesh.inner().synapse_count(), 16 * neighbors);
    }
    for neighbors in [0, 1, 15, 16, 17] {
        assert!(
            ExperimentalDaleMesh::new(DaleMeshConfig {
                neighbors,
                ..DaleMeshConfig::default()
            })
            .is_err(),
            "neighbors {neighbors} must be rejected"
        );
    }
    for rewire_probability in [-0.1, f32::NAN] {
        assert!(
            ExperimentalDaleMesh::new(DaleMeshConfig {
                rewire_probability,
                ..DaleMeshConfig::default()
            })
            .is_err(),
            "rewire probability {rewire_probability} must be rejected"
        );
    }
}

/// Dale's rule holds for every rewiring and delay setting, not just the
/// default.
#[test]
fn polarity_follows_the_source_under_any_rewiring() {
    for rewire_probability in [0.0, 0.5, 1.0] {
        let mesh = ExperimentalDaleMesh::new(DaleMeshConfig {
            neighbors: 6,
            rewire_probability,
            max_delay: 4,
        })
        .unwrap();
        for source in 0..16 {
            for (target, weight, delay, polarity) in mesh.inner().graph().outgoing(source) {
                assert_ne!(target as usize, source, "no self-loops");
                assert!((1..=4).contains(&delay), "delay {delay}");
                if source < spikenaut_snn::wiring::INHIBITORY_START {
                    assert_eq!(polarity, Polarity::Excitatory);
                    assert!(weight > 0.0);
                } else {
                    assert_eq!(polarity, Polarity::Inhibitory);
                    assert!(weight < 0.0);
                }
            }
        }
    }
}

/// `propagate_into` delivers what `propagate` does, overwriting the buffer.
#[test]
fn propagate_into_matches_propagate() {
    let config = DaleMeshConfig {
        max_delay: 3,
        ..DaleMeshConfig::default()
    };
    let mut allocating = ExperimentalDaleMesh::new(config).unwrap();
    let mut in_place = ExperimentalDaleMesh::new(config).unwrap();
    let mut buffer = [f32::NAN; 16];

    let frames: [[bool; 16]; 5] = [
        std::array::from_fn(|i| i == 0 || i == 13),
        [false; 16],
        std::array::from_fn(|i| i % 2 == 0),
        [false; 16],
        [false; 16],
    ];
    for frame in &frames {
        let expected = allocating.propagate(frame).unwrap();
        in_place.propagate_into(frame, &mut buffer).unwrap();
        assert_eq!(buffer.as_slice(), expected.as_slice());
    }
    assert_eq!(allocating.inner().tick(), in_place.inner().tick());
}

#[test]
fn synaptic_wiring_resolves_from_the_registry_at_zero_three() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "synaptic-wiring""#))
        .expect("Cargo.lock has synaptic-wiring");
    assert!(entry.contains(r#"version = "0.3."#), "{entry}");
    assert!(
        entry.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "{entry}"
    );
}

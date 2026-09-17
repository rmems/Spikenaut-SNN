// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed, transport-free `corpus-ipc` bridge contract.

use std::path::Path;

use corpus_ipc::{IpcMessage, NeuromodulatorSnapshot};
use spikenaut_snn::{
    IpcBatchContext, IpcBridgeError, decode_ipc_message, encode_ipc_message,
    neuromodulator_message, spike_message, stimulus_message,
};

fn context() -> IpcBatchContext {
    IpcBatchContext {
        session_id: Some("exp-025-shadow".to_owned()),
        batch_id: 7,
        timestamp_ns: 123_456,
    }
}

fn stimuli() -> [f32; 16] {
    [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
}

#[test]
fn a_complete_stimulus_round_trips_without_inventing_missingness() {
    let message = stimulus_message(&context(), stimuli(), None).unwrap();
    let bytes = encode_ipc_message(&message).unwrap();
    assert_eq!(decode_ipc_message(&bytes).unwrap(), message);

    let IpcMessage::Stimuli(batch) = message else {
        panic!("stimulus builder returned the wrong message variant")
    };
    assert_eq!(batch.values, stimuli());
    assert_eq!(batch.valid_mask, None);
    assert_eq!(batch.session_id.as_deref(), Some("exp-025-shadow"));
    assert_eq!(batch.batch_id, 7);
    assert_eq!(batch.timestamp, 123_456);
}

#[test]
fn an_explicit_validity_mask_round_trips_exactly() {
    let mut mask = vec![true; 16];
    mask[4] = false;
    let message = stimulus_message(&context(), stimuli(), Some(mask.clone())).unwrap();
    let decoded = decode_ipc_message(&encode_ipc_message(&message).unwrap()).unwrap();

    let IpcMessage::Stimuli(batch) = decoded else {
        panic!("decoded the wrong message variant")
    };
    assert_eq!(batch.valid_mask, Some(mask));
}

#[test]
fn malformed_or_non_finite_stimuli_are_rejected() {
    assert!(matches!(
        stimulus_message(&context(), stimuli(), Some(vec![true; 15])),
        Err(IpcBridgeError::Validation(_))
    ));

    let mut invalid = stimuli();
    invalid[2] = f32::NAN;
    assert!(matches!(
        stimulus_message(&context(), invalid, None),
        Err(IpcBridgeError::Validation(_))
    ));
}

#[test]
fn fired_neurons_become_checked_unit_strength_spike_events() {
    let message = spike_message(&context(), 42, &[1, 7, 15]).unwrap();
    let IpcMessage::Spikes(batch) = message else {
        panic!("spike builder returned the wrong message variant")
    };
    assert_eq!(batch.spikes.len(), 3);
    assert_eq!(batch.spikes[0].channel, 1);
    assert_eq!(batch.spikes[1].channel, 7);
    assert_eq!(batch.spikes[2].channel, 15);
    assert!(
        batch
            .spikes
            .iter()
            .all(|spike| spike.time == 42 && spike.strength == 1.0)
    );
}

#[test]
fn a_large_spike_channel_is_not_truncated() {
    let channel = usize::from(u16::MAX) + 1;
    assert!(matches!(
        spike_message(&context(), 0, &[channel]),
        Err(IpcBridgeError::ChannelOutOfRange { channel: bad }) if bad == channel
    ));
}

#[test]
fn a_spike_channel_outside_the_host_network_is_rejected() {
    assert!(spike_message(&context(), 0, &[15]).is_ok());
    assert!(matches!(
        spike_message(&context(), 0, &[16]),
        Err(IpcBridgeError::ChannelOutOfRange { channel: 16 })
    ));
}

#[test]
fn explicit_ipc_neuromodulators_validate_and_round_trip() {
    let snapshot = NeuromodulatorSnapshot {
        tick: 9,
        dopamine: 0.4,
        cortisol: 0.2,
        acetylcholine: 0.7,
        tempo: 1.25,
    };
    let message = neuromodulator_message(snapshot.clone()).unwrap();
    assert_eq!(
        decode_ipc_message(&encode_ipc_message(&message).unwrap()).unwrap(),
        message
    );

    let invalid = NeuromodulatorSnapshot {
        cortisol: 1.1,
        ..snapshot
    };
    assert!(matches!(
        neuromodulator_message(invalid),
        Err(IpcBridgeError::Validation(_))
    ));
}

#[test]
fn generic_encoding_revalidates_directly_constructed_messages() {
    let invalid = IpcMessage::Neuromodulators(NeuromodulatorSnapshot {
        tick: 1,
        dopamine: 0.0,
        cortisol: 1.1,
        acetylcholine: 0.0,
        tempo: 1.0,
    });
    assert!(matches!(
        encode_ipc_message(&invalid),
        Err(IpcBridgeError::Validation(_))
    ));
}

#[test]
fn strict_decoder_rejects_legacy_unversioned_messages() {
    let legacy_unit = br#""Ping""#;
    let legacy_object = br#"{"Neuromodulators":{"tick":9,"dopamine":0.4,"cortisol":0.2,"acetylcholine":0.7,"tempo":1.25}}"#;

    assert!(matches!(
        decode_ipc_message(legacy_unit),
        Err(IpcBridgeError::Envelope(_))
    ));
    assert!(matches!(
        decode_ipc_message(legacy_object),
        Err(IpcBridgeError::Envelope(_))
    ));
}

#[test]
fn strict_decoder_rejects_unsupported_envelope_versions() {
    for bytes in [
        br#"{"wire_version":0,"payload":"Ping"}"#.as_slice(),
        br#"{"wire_version":2,"payload":"Ping"}"#.as_slice(),
    ] {
        assert!(matches!(
            decode_ipc_message(bytes),
            Err(IpcBridgeError::Envelope(_))
        ));
    }
}

#[test]
fn corpus_ipc_is_registry_only_without_transport_features() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");
    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "corpus-ipc""#))
        .expect("Cargo.lock has corpus-ipc");

    assert!(entry.contains(r#"version = "0.1."#), "{entry}");
    assert!(
        entry.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "{entry}"
    );
    assert!(!entry.contains("\n \"zmq\""), "{entry}");
    assert!(!entry.contains("\n \"axum\""), "{entry}");
    assert!(!entry.contains("\n \"tokio\""), "{entry}");
}

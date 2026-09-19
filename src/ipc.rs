// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed, transport-free `corpus-ipc` message adapters.
//!
//! This module builds and validates versioned wire messages but opens no
//! sockets. In particular, it does not invent a conversion between
//! `neuromod`'s dopamine/serotonin/acetylcholine/norepinephrine vocabulary and
//! `corpus-ipc`'s dopamine/cortisol/acetylcholine/tempo vocabulary.

use std::error::Error;
use std::fmt;

use corpus_ipc::{
    EnvelopeError, IpcMessage, NeuromodulatorSnapshot, SpikeBatch, SpikeEvent, StimulusBatch,
    Validate, ValidationError, WireEnvelope, encode_ipc_message_json,
};

use crate::encode::CHANNEL_COUNT;
use crate::neuromod_host::HOST_NETWORK_NEURONS;

/// Correlation and timestamp fields shared by stimulus and spike batches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpcBatchContext {
    /// Optional experiment/session identifier.
    pub session_id: Option<String>,
    /// Caller-owned batch correlation identifier.
    pub batch_id: u64,
    /// UTC or relative timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

/// Failures while adapting Spikenaut data into validated IPC messages.
#[derive(Debug)]
pub enum IpcBridgeError {
    /// A typed corpus payload violated its validation contract.
    Validation(ValidationError),
    /// A neuron index is outside the 16-neuron host network.
    ChannelOutOfRange {
        /// The unrepresentable host index.
        channel: usize,
    },
    /// A versioned JSON envelope could not be encoded or decoded.
    Envelope(EnvelopeError),
}

impl fmt::Display for IpcBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => write!(formatter, "invalid IPC payload: {error}"),
            Self::ChannelOutOfRange { channel } => {
                write!(
                    formatter,
                    "IPC spike channel {channel} is outside host range 0..{HOST_NETWORK_NEURONS}"
                )
            }
            Self::Envelope(error) => write!(formatter, "invalid IPC envelope: {error}"),
        }
    }
}

impl Error for IpcBridgeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            Self::ChannelOutOfRange { .. } => None,
            Self::Envelope(error) => Some(error),
        }
    }
}

impl From<ValidationError> for IpcBridgeError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

impl From<EnvelopeError> for IpcBridgeError {
    fn from(error: EnvelopeError) -> Self {
        Self::Envelope(error)
    }
}

/// Build a validated 16-channel continuous-stimulus message.
///
/// A `None` mask means every value is valid, including architecturally unused
/// axons whose valid stimulus is zero. Numeric zero is never interpreted as
/// missingness.
pub fn stimulus_message(
    context: &IpcBatchContext,
    values: [f32; CHANNEL_COUNT],
    valid_mask: Option<Vec<bool>>,
) -> Result<IpcMessage, IpcBridgeError> {
    let batch = StimulusBatch {
        session_id: context.session_id.clone(),
        batch_id: context.batch_id,
        timestamp: context.timestamp_ns,
        values: values.to_vec(),
        valid_mask,
        metadata: None,
    };
    batch.validate()?;
    Ok(IpcMessage::Stimuli(batch))
}

/// Build a validated unit-strength spike message for one engine tick.
pub fn spike_message(
    context: &IpcBatchContext,
    time: u32,
    fired: &[usize],
) -> Result<IpcMessage, IpcBridgeError> {
    let spikes = fired
        .iter()
        .map(|&channel| {
            if channel >= HOST_NETWORK_NEURONS {
                return Err(IpcBridgeError::ChannelOutOfRange { channel });
            }
            let channel = u16::try_from(channel)
                .map_err(|_| IpcBridgeError::ChannelOutOfRange { channel })?;
            Ok(SpikeEvent {
                channel,
                time,
                strength: 1.0,
            })
        })
        .collect::<Result<Vec<_>, IpcBridgeError>>()?;
    let batch = SpikeBatch {
        session_id: context.session_id.clone(),
        batch_id: context.batch_id,
        timestamp: context.timestamp_ns,
        spikes,
        metadata: None,
    };
    batch.validate()?;
    Ok(IpcMessage::Spikes(batch))
}

/// Validate and wrap an explicitly IPC-domain neuromodulator snapshot.
///
/// Callers must supply cortisol and tempo under their `corpus-ipc` meanings;
/// this function intentionally accepts no `neuromod::NeuroModulators` value.
pub fn neuromodulator_message(
    snapshot: NeuromodulatorSnapshot,
) -> Result<IpcMessage, IpcBridgeError> {
    snapshot.validate()?;
    Ok(IpcMessage::Neuromodulators(snapshot))
}

/// Validate and encode one message in the current versioned JSON wire envelope.
pub fn encode_ipc_message(message: &IpcMessage) -> Result<Vec<u8>, IpcBridgeError> {
    message.validate()?;
    Ok(encode_ipc_message_json(message)?)
}

/// Strictly decode and validate one versioned JSON wire envelope.
///
/// Legacy unversioned payloads accepted by `corpus-ipc`'s compatibility
/// decoder are deliberately rejected at this boundary.
pub fn decode_ipc_message(bytes: &[u8]) -> Result<IpcMessage, IpcBridgeError> {
    let message = WireEnvelope::<IpcMessage>::decode_json(bytes)?.into_payload();
    message.validate()?;
    Ok(message)
}

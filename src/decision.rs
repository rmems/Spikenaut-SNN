// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic output-row → decision contract (Linear RM-1328 / GH #6).
//!
//! Supervisor v3 needs one conversion from a readout score row to a typed
//! shadow-policy decision. Downstream code used to pick argmax, a threshold,
//! or a tie rule independently; this module is the software replay path those
//! consumers — and FPGA parity — have to match.
//!
//! It is pure: one finite score row in, a [`Decision`] or a fail-closed
//! [`DecisionError`] out. It does not step a membrane, does not read model
//! weights, and does not actuate the host. The action vocabulary is supplied
//! by [`DecisionConfig`], not by this function inventing process-control
//! names.
//!
//! # Shipped head (exp-025 / Distill `a1fa491`)
//!
//! The live bank's readout is **3-wide**, neuron-major, trained as a
//! regression onto Distill `sample_readout_target`:
//!
//! ```text
//! index 0  comfort     clamp(1 - 0.5 * temp_u - 0.5 * power_u, 0, 1)
//! index 1  temp        frozen-minmax `gpu_temp_c` in [0, 1]
//! index 2  power       frozen-minmax `power_w` in [0, 1]
//! pred   = readout * spikes     (3 × 16) · (16 × 1)
//! ```
//!
//! That order is the checkpoint's intended ordering. It is **not** the
//! RM-1150 Stage-1 vocabulary (`ALLOW` / `WARN` / `THROTTLE` / `PAUSE` /
//! `YIELD_GPU`), which is five-wide and unbound: mapping those five names
//! onto these three regression channels would be a guess. [`HeadChannel`]
//! names the shipped rows; [`SupervisorAction`] documents the Stage-1 list
//! without wiring it to `merged_v2`.
//!
//! # Contract
//!
//! [`decide`] (and [`replay_output_row`], which is `decide` on
//! [`DecisionConfig::shipped`]):
//!
//! - **Argmax**, algebraic (negative scores are allowed; Distill inhibitory
//!   columns are ≤ 0).
//! - **Ties** (including all-equal): lowest index wins, [`Diagnostics::tied`]
//!   is set. Optional [`DecisionConfig::abstain_on_tie`] turns that into
//!   [`AbstainReason::Tie`] instead of a proposal. The default proposes, so
//!   replay and FPGA see a stable winner rather than a coin flip.
//! - **Confidence** is `margin / (|winner| + |runner_up|)`, or `0` when the
//!   denominator is 0, or `1` on a width-1 row. No `exp`, no softmax: the
//!   formula is exact on Q8.8 and on Distill `f32` promoted to `f64`.
//! - **Abstention**: confidence strictly below the configured floor, or an
//!   opted-in tie. The would-be winner stays in the diagnostics.
//! - **Fail closed**: empty row, width mismatch, any `NaN` / `±Inf` in the
//!   input, a finite row whose derived margin or confidence overflows
//!   to `NaN` / `±Inf`, a finite margin whose `|winner| + |runner_up|`
//!   denominator overflows, and a readout whose accumulated channel
//!   sums overflow all return [`DecisionError`]. Nothing is
//!   substituted, and no action is proposed. Same for an invalid config
//!   (empty or duplicate vocabulary, empty label, non-finite floor
//!   outside `[0, 1]`).
//!
//! Diagnostics copy the finite scores and the winning index/label. They do
//! not borrow membrane, weights, or any other mutable model state.

use std::fmt;

use crate::model::NEURON_COUNT;

#[path = "decision_helpers.rs"]
mod decision_helpers;
use decision_helpers::{validate_readout_shape, write_non_finite_indices};

/// Width of the shipped Distill readout (`n_outputs` in `snn_model.json`).
pub const OUTPUT_WIDTH: usize = 3;

/// Neuron-major length of `parameters_output_weights.mem` (16 × 3).
pub const OUTPUT_WEIGHT_COUNT: usize = NEURON_COUNT * OUTPUT_WIDTH;

/// Checkpoint-native score-channel labels, Distill row order.
///
/// These are regression heads, not mutually exclusive classes. Argmax still
/// names a winner so replay and FPGA agree on *which channel dominated*; it
/// is not a claim that Distill trained a classifier.
pub const SHIPPED_VOCABULARY: [&str; OUTPUT_WIDTH] = ["comfort", "temp", "power"];

/// Contract id recorded on the golden pin both languages read.
pub const CONTRACT_ID: &str = "spikenaut-output-row-v1";

/// Shipped readout channels, in Distill `sample_readout_target` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadChannel {
    /// Index 0: comfort `clamp(1 - 0.5 * temp_u - 0.5 * power_u, 0, 1)`.
    Comfort,
    /// Index 1: frozen-minmax `gpu_temp_c`.
    Temp,
    /// Index 2: frozen-minmax `power_w`.
    Power,
}

impl HeadChannel {
    /// Every shipped channel, in index order.
    pub const ALL: [Self; OUTPUT_WIDTH] = [Self::Comfort, Self::Temp, Self::Power];

    /// The Distill row this channel occupies.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Comfort => 0,
            Self::Temp => 1,
            Self::Power => 2,
        }
    }

    /// The vocabulary label [`SHIPPED_VOCABULARY`] stores at [`Self::index`].
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Comfort => "comfort",
            Self::Temp => "temp",
            Self::Power => "power",
        }
    }

    /// The channel at `index`, or `None` if it is not one of the three.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Comfort),
            1 => Some(Self::Temp),
            2 => Some(Self::Power),
            _ => None,
        }
    }
}

/// RM-1150 Stage-1 suggested vocabulary. Width 5; the shipped head is width 3.
///
/// Binding these names onto Distill `(comfort, temp, power)` is unresolved
/// and is not performed here. A caller with a 5-wide head may pass
/// [`Self::VOCABULARY`] to [`DecisionConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupervisorAction {
    /// Continue without intervention.
    Allow,
    /// Raise a warning; no throttle.
    Warn,
    /// Reduce clocks / power.
    Throttle,
    /// Pause the workload.
    Pause,
    /// Yield the GPU to another tenant.
    YieldGpu,
}

impl SupervisorAction {
    /// The five Stage-1 labels, in RM-1150 order. Not the shipped head.
    pub const VOCABULARY: [&str; 5] = ["ALLOW", "WARN", "THROTTLE", "PAUSE", "YIELD_GPU"];

    /// Every Stage-1 action, in the same order as [`Self::VOCABULARY`].
    pub const ALL: [Self; 5] = [
        Self::Allow,
        Self::Warn,
        Self::Throttle,
        Self::Pause,
        Self::YieldGpu,
    ];

    /// The RM-1150 label for this action.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Allow => "ALLOW",
            Self::Warn => "WARN",
            Self::Throttle => "THROTTLE",
            Self::Pause => "PAUSE",
            Self::YieldGpu => "YIELD_GPU",
        }
    }
}

/// Why a finite, well-shaped row was not turned into a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbstainReason {
    /// [`Diagnostics::confidence`] is strictly below the configured floor.
    LowConfidence,
    /// Two or more scores share the max and the config asked to abstain.
    Tie,
}

/// Proposal vs fail-closed abstention. Invalid rows are [`DecisionError`],
/// not a kind of decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecisionKind {
    /// Argmax produced an action. Ties still land here unless the config
    /// abstains on them.
    Propose,
    /// The row was well-formed but the contract refused to propose.
    Abstain(AbstainReason),
}

/// Diagnostics for one decision. Copied scores only; no model state.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    /// Lowest index among the scores that share the maximum.
    pub winning_index: usize,
    /// Score at [`Self::winning_index`].
    pub winning_score: f64,
    /// Vocabulary label at [`Self::winning_index`]. Owned so the decision
    /// does not borrow the config.
    pub winning_action: String,
    /// First index of the runner-up: the next tied max, or the highest
    /// remaining score. `None` on a width-1 row.
    pub runner_up_index: Option<usize>,
    /// Score at [`Self::runner_up_index`].
    pub runner_up_score: Option<f64>,
    /// `winner - runner_up`, or `0` on a width-1 row.
    pub margin: f64,
    /// `margin / (|winner| + |runner_up|)`, `0` when that denominator is 0,
    /// `1` on a width-1 row.
    pub confidence: f64,
    /// More than one index holds the maximum score.
    pub tied: bool,
    /// Copy of the finite input row, in index order.
    pub scores: Vec<f64>,
}

/// One output-row decision plus the diagnostics a replay log needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// Propose or abstain. Never an error: those are [`DecisionError`].
    pub kind: DecisionKind,
    /// Winning index/score/confidence, and a copy of the row.
    pub diagnostics: Diagnostics,
}

/// Vocabulary and abstention knobs for [`decide`].
///
/// Fields stay private so a constructed config cannot be mutated into an
/// invalid one. [`DecisionConfig::new`] is the check; [`Self::shipped`] is
/// the Distill three-channel layout.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionConfig {
    vocabulary: Vec<String>,
    confidence_floor: f64,
    abstain_on_tie: bool,
}

impl DecisionConfig {
    /// Build a config. The vocabulary length is the row width [`decide`]
    /// will demand. The labels are an ordered sequence (a slice); a set or
    /// other unordered iterator cannot be passed, because channel *i* is
    /// vocabulary *i*.
    ///
    /// # Errors
    ///
    /// - [`DecisionError::EmptyVocabulary`] if `vocabulary` is empty
    /// - [`DecisionError::InvalidLabel`] if a label is empty
    /// - [`DecisionError::DuplicateLabel`] if a label repeats
    /// - [`DecisionError::InvalidConfidenceFloor`] if `confidence_floor` is
    ///   not finite or not in `[0, 1]`
    pub fn new<S: AsRef<str>>(
        vocabulary: impl AsRef<[S]>,
        confidence_floor: f64,
        abstain_on_tie: bool,
    ) -> Result<Self, DecisionError> {
        let vocabulary: Vec<String> = vocabulary
            .as_ref()
            .iter()
            .map(|label| label.as_ref().to_string())
            .collect();
        validate_config(&vocabulary, confidence_floor)?;
        Ok(Self {
            vocabulary,
            confidence_floor,
            abstain_on_tie,
        })
    }

    /// Distill `(comfort, temp, power)`, floor `0` (never abstain on
    /// confidence), ties propose the lowest index.
    #[must_use]
    pub fn shipped() -> Self {
        Self::new(SHIPPED_VOCABULARY, 0.0, false).expect("shipped Distill vocabulary is valid")
    }

    /// Action labels, in row-index order.
    #[must_use]
    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    /// Row width this config will accept.
    #[must_use]
    pub fn width(&self) -> usize {
        self.vocabulary.len()
    }

    /// Confidence strictly below this value abstains.
    #[must_use]
    pub fn confidence_floor(&self) -> f64 {
        self.confidence_floor
    }

    /// When true, a tied max becomes [`AbstainReason::Tie`] rather than a
    /// lowest-index proposal.
    #[must_use]
    pub fn abstain_on_tie(&self) -> bool {
        self.abstain_on_tie
    }
}

/// Failures that refuse to produce a [`Decision`].
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionError {
    /// The score row had length 0.
    EmptyRow,
    /// Row or spike/weight vector length did not match the contract.
    WidthMismatch {
        /// Length that arrived.
        got: usize,
        /// Length the config or the shipped head required.
        expected: usize,
    },
    /// One or more values were `NaN` or `±Inf`. Indices are in increasing
    /// order, every offender, not just the first.
    NonFinite {
        /// Positions in the input vector that were not finite.
        indices: Vec<usize>,
    },
    /// [`DecisionConfig::new`] was given no labels.
    EmptyVocabulary,
    /// A vocabulary entry was the empty string.
    InvalidLabel {
        /// Index in the vocabulary.
        index: usize,
    },
    /// The same label appeared twice.
    DuplicateLabel {
        /// The repeated label.
        label: String,
    },
    /// Confidence floor was not a finite value in `[0, 1]`.
    InvalidConfidenceFloor {
        /// The rejected floor.
        value: f64,
    },
    /// Input scores were finite, but `winner - runner_up` or
    /// `margin / (|winner| + |runner_up|)` overflowed to `NaN` / `±Inf`.
    /// Example: `[f64::MAX, -f64::MAX, -f64::MAX]`. The contract refuses to
    /// propose with non-finite diagnostics (`NaN < floor` is false).
    DerivedNonFinite {
        /// True when the derived margin is not finite.
        margin: bool,
        /// True when the derived confidence is not finite.
        confidence: bool,
    },
}

impl fmt::Display for DecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRow | Self::WidthMismatch { .. } | Self::EmptyVocabulary => {
                self.fmt_shape(f)
            }
            other => other.fmt_value(f),
        }
    }
}

impl DecisionError {
    fn fmt_shape(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRow => f.write_str("output row is empty"),
            Self::WidthMismatch { got, expected } => {
                write!(f, "output row width {got}, expected {expected}")
            }
            Self::EmptyVocabulary => f.write_str("decision vocabulary is empty"),
            _ => unreachable!("fmt_shape only formats shape errors"),
        }
    }

    fn fmt_value(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { indices } => write_non_finite_indices(f, indices),
            Self::InvalidLabel { index } => {
                write!(f, "decision vocabulary label {index} is empty")
            }
            Self::DuplicateLabel { label } => {
                write!(f, "decision vocabulary repeats {label:?}")
            }
            Self::InvalidConfidenceFloor { value } => {
                write!(
                    f,
                    "confidence floor {value} is not a finite value in [0, 1]"
                )
            }
            Self::DerivedNonFinite { margin, confidence } => {
                f.write_str(derived_non_finite_message(*margin, *confidence))
            }
            _ => unreachable!("fmt_value only formats value errors"),
        }
    }
}

impl std::error::Error for DecisionError {}

/// Convert one score row into a decision under `config`.
///
/// This is the generic form. Software replay of the shipped bank uses
/// [`replay_output_row`].
///
/// # Errors
///
/// See [`DecisionError`]: empty/wrong-width rows, any non-finite score, a
/// finite row whose derived margin/confidence overflowed, and an invalid
/// config all fail closed. A well-formed low-confidence or tied row is
/// [`Ok`] with [`DecisionKind::Abstain`], not an error.
pub fn decide(row: &[f64], config: &DecisionConfig) -> Result<Decision, DecisionError> {
    validate_config(&config.vocabulary, config.confidence_floor)?;
    let expected = config.width();
    let scores = validate_row(row, expected)?;
    let (winning_index, runner_up_index, tied) = pick_winner(&scores);
    let winning_score = scores[winning_index];
    let runner_up_score = runner_up_index.map(|index| scores[index]);
    let (margin, confidence) = margin_and_confidence(winning_score, runner_up_score)?;
    let diagnostics = Diagnostics {
        winning_index,
        winning_score,
        winning_action: config.vocabulary[winning_index].clone(),
        runner_up_index,
        runner_up_score,
        margin,
        confidence,
        tied,
        scores,
    };
    let kind = if tied && config.abstain_on_tie {
        DecisionKind::Abstain(AbstainReason::Tie)
    } else if confidence < config.confidence_floor {
        DecisionKind::Abstain(AbstainReason::LowConfidence)
    } else {
        DecisionKind::Propose
    };
    Ok(Decision { kind, diagnostics })
}

/// Software replay path for the shipped Distill head.
///
/// Equivalent to [`decide`] with [`DecisionConfig::shipped`]. FPGA action
/// parity compares against this function, not against an ad-hoc argmax.
///
/// # Errors
///
/// [`DecisionError::EmptyRow`], [`DecisionError::WidthMismatch`] (not width
/// 3), [`DecisionError::NonFinite`], or [`DecisionError::DerivedNonFinite`].
pub fn replay_output_row(row: &[f64]) -> Result<Decision, DecisionError> {
    decide(row, &DecisionConfig::shipped())
}

/// Score one spike vector through the shipped readout, then decide.
///
/// Software replay step downstream of a keep-LIF tick: spikes in, Distill-
/// ordered decision out. It does not actuate the host. Custom knobs stay on
/// [`score_readout`] plus [`decide`]; this path is always
/// [`DecisionConfig::shipped`].
///
/// # Errors
///
/// [`score_readout`] errors, or [`replay_output_row`] errors on the scores.
pub fn replay_tick(neuron_major: &[f64], spikes: &[bool]) -> Result<Decision, DecisionError> {
    let row = score_readout(neuron_major, spikes)?;
    replay_output_row(&row)
}

/// Distill `pred = readout * spikes` on the neuron-major 16×3 image.
///
/// `neuron_major` is the layout of `parameters_output_weights.mem` and of
/// concatenated `neurons[i].output_weights`. A spike contributes that
/// neuron's three weights; a silent neuron contributes nothing. Addition is
/// `f64`. This is the analog of Distill's `Float32` readout, not a Q8.8
/// accumulator — [`decide`] is the contract, and it consumes the scores this
/// returns (or any other finite 3-wide row).
///
/// # Errors
///
/// Empty weights, a length other than [`OUTPUT_WEIGHT_COUNT`], a spike
/// vector other than [`NEURON_COUNT`] long, any non-finite weight, or a
/// finite image whose accumulated channel sums overflow to `NaN` / `±Inf`.
pub fn score_readout(
    neuron_major: &[f64],
    spikes: &[bool],
) -> Result<[f64; OUTPUT_WIDTH], DecisionError> {
    validate_readout_shape(neuron_major, spikes)?;
    refuse_non_finite(neuron_major)?;
    let mut scores = [0.0; OUTPUT_WIDTH];
    for (neuron, &spiked) in spikes.iter().enumerate() {
        if !spiked {
            continue;
        }
        let base = neuron * OUTPUT_WIDTH;
        for (channel, score) in scores.iter_mut().enumerate() {
            *score += neuron_major[base + channel];
        }
    }
    refuse_non_finite(&scores)?;
    Ok(scores)
}

fn validate_config(vocabulary: &[String], confidence_floor: f64) -> Result<(), DecisionError> {
    if vocabulary.is_empty() {
        return Err(DecisionError::EmptyVocabulary);
    }
    for (index, label) in vocabulary.iter().enumerate() {
        if label.is_empty() {
            return Err(DecisionError::InvalidLabel { index });
        }
        if vocabulary[..index].iter().any(|seen| seen == label) {
            return Err(DecisionError::DuplicateLabel {
                label: label.clone(),
            });
        }
    }
    if !(confidence_floor.is_finite() && (0.0..=1.0).contains(&confidence_floor)) {
        return Err(DecisionError::InvalidConfidenceFloor {
            value: confidence_floor,
        });
    }
    Ok(())
}

fn validate_row(row: &[f64], expected: usize) -> Result<Vec<f64>, DecisionError> {
    if row.is_empty() {
        return Err(DecisionError::EmptyRow);
    }
    if row.len() != expected {
        return Err(DecisionError::WidthMismatch {
            got: row.len(),
            expected,
        });
    }
    refuse_non_finite(row)?;
    Ok(row.to_vec())
}

fn refuse_non_finite(values: &[f64]) -> Result<(), DecisionError> {
    let indices: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (!value.is_finite()).then_some(index))
        .collect();
    if indices.is_empty() {
        Ok(())
    } else {
        Err(DecisionError::NonFinite { indices })
    }
}

/// Lowest-index argmax, plus the runner-up used for margin/confidence.
///
/// Runner-up is the next index that also holds the max when tied, otherwise
/// the index of the highest remaining score. Width 1 has no runner-up.
fn pick_winner(scores: &[f64]) -> (usize, Option<usize>, bool) {
    let mut winning_index = 0usize;
    let mut winning_score = scores[0];
    let mut tied = false;
    for (index, &score) in scores.iter().enumerate().skip(1) {
        if score > winning_score {
            winning_score = score;
            winning_index = index;
            tied = false;
        } else if score == winning_score {
            tied = true;
        }
    }
    let runner_up_index = runner_up(scores, winning_index);
    (winning_index, runner_up_index, tied)
}

fn runner_up(scores: &[f64], winning_index: usize) -> Option<usize> {
    if scores.len() == 1 {
        return None;
    }
    let mut best: Option<usize> = None;
    for (index, &score) in scores.iter().enumerate() {
        if index == winning_index {
            continue;
        }
        match best {
            None => best = Some(index),
            Some(current) if score > scores[current] => best = Some(index),
            Some(_) => {}
        }
    }
    best
}

fn derived_non_finite_message(margin: bool, confidence: bool) -> &'static str {
    match (margin, confidence) {
        (true, true) => "derived margin and confidence are not finite",
        (true, false) => "derived margin is not finite",
        (false, true) => "derived confidence is not finite",
        (false, false) => "derived diagnostics are not finite",
    }
}

fn margin_and_confidence(
    winning_score: f64,
    runner_up_score: Option<f64>,
) -> Result<(f64, f64), DecisionError> {
    let Some(runner) = runner_up_score else {
        return Ok((0.0, 1.0));
    };
    let margin = winning_score - runner;
    let denom = winning_score.abs() + runner.abs();
    // Same-sign extremes such as `[f64::MAX, f64::MAX / 2.0]` keep a finite
    // margin while `|winner| + |runner|` overflows to Inf. `margin / Inf`
    // is 0.0, which a positive floor would treat as a low-confidence
    // abstention. Refuse a non-finite denominator instead of substituting.
    let denom_bad = !denom.is_finite();
    let confidence = if denom_bad {
        f64::NAN
    } else if denom > 0.0 {
        margin / denom
    } else {
        0.0
    };
    let margin_bad = !margin.is_finite();
    let confidence_bad = denom_bad || !confidence.is_finite();
    if margin_bad || confidence_bad {
        return Err(DecisionError::DerivedNonFinite {
            margin: margin_bad,
            confidence: confidence_bad,
        });
    }
    Ok((margin, confidence))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_vocabulary_matches_head_channel_labels() {
        for (index, channel) in HeadChannel::ALL.iter().enumerate() {
            assert_eq!(channel.index(), index);
            assert_eq!(channel.label(), SHIPPED_VOCABULARY[index]);
            assert_eq!(HeadChannel::from_index(index), Some(*channel));
        }
        assert_eq!(HeadChannel::from_index(3), None);
        assert_eq!(OUTPUT_WIDTH, 3);
        assert_eq!(OUTPUT_WEIGHT_COUNT, 48);
    }

    #[test]
    fn supervisor_actions_are_five_wide_and_unbound() {
        assert_eq!(SupervisorAction::VOCABULARY.len(), 5);
        for (index, action) in SupervisorAction::ALL.iter().enumerate() {
            assert_eq!(action.label(), SupervisorAction::VOCABULARY[index]);
            assert_ne!(
                action.label(),
                SHIPPED_VOCABULARY[0],
                "Stage-1 names must not silently replace Distill comfort/temp/power"
            );
        }
        assert_ne!(SupervisorAction::VOCABULARY.len(), SHIPPED_VOCABULARY.len());
    }

    #[test]
    fn shipped_config_is_the_distill_head() {
        let config = DecisionConfig::shipped();
        assert_eq!(config.vocabulary(), &["comfort", "temp", "power"]);
        assert_eq!(config.width(), OUTPUT_WIDTH);
        assert_eq!(config.confidence_floor(), 0.0);
        assert!(!config.abstain_on_tie());
    }

    #[test]
    fn overflow_confidence_fails_closed() {
        let err = replay_output_row(&[f64::MAX, -f64::MAX, -f64::MAX]).unwrap_err();
        match err {
            DecisionError::DerivedNonFinite { margin, confidence } => {
                assert!(margin);
                assert!(confidence);
            }
            other => panic!("expected DerivedNonFinite, got {other:?}"),
        }
        let err = replay_output_row(&[f64::MAX, f64::MAX / 2.0, 0.0]).unwrap_err();
        match err {
            DecisionError::DerivedNonFinite { margin, confidence } => {
                assert!(!margin);
                assert!(confidence);
            }
            other => panic!("expected DerivedNonFinite on overflowing denom, got {other:?}"),
        }
        let floor = DecisionConfig::new(SHIPPED_VOCABULARY, 0.5, false).expect("floor 0.5");
        let err = decide(&[f64::MAX, f64::MAX / 2.0, 0.0], &floor).unwrap_err();
        match err {
            DecisionError::DerivedNonFinite { margin, confidence } => {
                assert!(!margin);
                assert!(confidence);
            }
            other => panic!(
                "overflowing denom must not become a low-confidence abstention, got {other:?}"
            ),
        }
        let ok = replay_output_row(&[f64::MAX, 0.0, 0.0]).expect("MAX vs 0 stays finite");
        assert_eq!(ok.kind, DecisionKind::Propose);
        assert_eq!(ok.diagnostics.winning_index, 0);
        assert_eq!(ok.diagnostics.confidence, 1.0);
        assert!(ok.diagnostics.margin.is_finite());
    }

    #[test]
    fn overflow_readout_fails_closed() {
        let mut weights = [0.0; OUTPUT_WEIGHT_COUNT];
        weights[0] = f64::MAX;
        weights[OUTPUT_WIDTH] = f64::MAX;
        let mut spikes = [false; NEURON_COUNT];
        spikes[0] = true;
        spikes[1] = true;
        let err = score_readout(&weights, &spikes).unwrap_err();
        match err {
            DecisionError::NonFinite { indices } => assert_eq!(indices, [0]),
            other => panic!("expected NonFinite on overflowing readout, got {other:?}"),
        }
        let replayed = replay_tick(&weights, &spikes).unwrap_err();
        match replayed {
            DecisionError::NonFinite { indices } => assert_eq!(indices, [0]),
            other => panic!("replay_tick must fail closed on overflowing readout, got {other:?}"),
        }
    }
}

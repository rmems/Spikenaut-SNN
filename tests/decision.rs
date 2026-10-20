// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acceptance for the output-row decision contract (Linear RM-1328):
//! Python `tools/decision_core.py` and Rust `spikenaut_snn::decision` must
//! agree, case for case, on the golden pin both languages read.
//!
//! `tools/decision_parity.py` regenerates `tools/fixtures/decision/expected.json`
//! from the Python reference. This file reads that pin and asserts the Rust
//! API reproduces it, including Distill `(comfort, temp, power)` ordering on
//! the shipped checkpoint. CI runs both halves.

use spikenaut_snn::json::{self, Json};
use spikenaut_snn::{
    AbstainReason, CONTRACT_ID, Decision, DecisionConfig, DecisionError, DecisionKind,
    OUTPUT_WEIGHT_COUNT, OUTPUT_WIDTH, SHIPPED_VOCABULARY, SupervisorAction, decide,
    replay_output_row, replay_tick, score_readout,
};

const EXPECTED_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tools/fixtures/decision/expected.json"
));

const PIN_REL: &str = "tools/fixtures/decision/expected.json";

struct PinCase {
    name: String,
    row: Vec<f64>,
    config: Result<DecisionConfig, String>,
    expected: Expected,
    spikes: Option<Vec<bool>>,
}

enum Expected {
    Ok(Decision),
    Err {
        code: String,
        indices: Vec<usize>,
        got: Option<usize>,
        expected_width: Option<usize>,
        margin: Option<bool>,
        confidence: Option<bool>,
    },
}

fn pin() -> Json {
    json::parse(EXPECTED_JSON).unwrap_or_else(|err| panic!("{PIN_REL}: {err}"))
}

fn bits_f64(hex: &str, what: &str) -> f64 {
    f64::from_bits(
        u64::from_str_radix(hex, 16).unwrap_or_else(|err| panic!("{what}: {hex:?}: {err}")),
    )
}

fn require_object<'a>(value: &'a Json, what: &str) -> &'a Json {
    value
        .as_object()
        .map(|_| value)
        .unwrap_or_else(|| panic!("{what}: expected an object"))
}

fn string_field(value: &Json, key: &str, what: &str) -> String {
    value
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("{what}: missing string {key}"))
        .to_owned()
}

fn bool_field(value: &Json, key: &str, what: &str) -> bool {
    value
        .get(key)
        .and_then(Json::as_bool)
        .unwrap_or_else(|| panic!("{what}: missing bool {key}"))
}

fn f64_field(value: &Json, key: &str, what: &str) -> f64 {
    value
        .get(key)
        .and_then(Json::as_f64)
        .unwrap_or_else(|| panic!("{what}: missing number {key}"))
}

fn usize_field(value: &Json, key: &str, what: &str) -> usize {
    let number = f64_field(value, key, what);
    assert_eq!(number, number.trunc(), "{what}: {key} is not an integer");
    number as usize
}

fn optional_usize(value: &Json, key: &str, what: &str) -> Option<usize> {
    match value.get(key) {
        None | Some(Json::Null) => None,
        Some(Json::Number(number)) => {
            assert_eq!(*number, number.trunc(), "{what}: {key} is not an integer");
            Some(*number as usize)
        }
        Some(_) => panic!("{what}: {key} must be a number or null"),
    }
}

fn optional_bool(value: &Json, key: &str, what: &str) -> Option<bool> {
    match value.get(key) {
        None | Some(Json::Null) => None,
        Some(Json::Bool(flag)) => Some(*flag),
        Some(_) => panic!("{what}: {key} must be a bool or null"),
    }
}

fn optional_f64(value: &Json, key: &str, what: &str) -> Option<f64> {
    match value.get(key) {
        None | Some(Json::Null) => None,
        Some(Json::Number(number)) => Some(*number),
        Some(_) => panic!("{what}: {key} must be a number or null"),
    }
}

fn string_list(value: &Json, key: &str, what: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("{what}: missing array {key}"))
        .iter()
        .map(|item| {
            item.as_str()
                .unwrap_or_else(|| panic!("{what}: {key} entries are strings"))
                .to_owned()
        })
        .collect()
}

fn config_of(value: &Json, what: &str) -> Result<DecisionConfig, String> {
    let vocabulary = string_list(value, "vocabulary", what);
    let floor = f64_field(value, "confidence_floor", what);
    let abstain = bool_field(value, "abstain_on_tie", what);
    DecisionConfig::new(vocabulary, floor, abstain).map_err(|err| err.to_string())
}

fn row_of(value: &Json, what: &str) -> Vec<f64> {
    value
        .get("row_bits")
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("{what}: missing row_bits"))
        .iter()
        .map(|item| {
            bits_f64(
                item.as_str()
                    .unwrap_or_else(|| panic!("{what}: row_bits entries are strings")),
                what,
            )
        })
        .collect()
}

fn expected_of(value: &Json, what: &str) -> Expected {
    if bool_field(value, "ok", what) {
        Expected::Ok(decision_from_pin(value, what))
    } else {
        Expected::Err {
            code: string_field(value, "error", what),
            indices: value
                .get("indices")
                .and_then(Json::as_array)
                .map(|items| {
                    items
                        .iter()
                        .map(|item| {
                            let number = item
                                .as_f64()
                                .unwrap_or_else(|| panic!("{what}: indices entries are numbers"));
                            number as usize
                        })
                        .collect()
                })
                .unwrap_or_default(),
            got: optional_usize(value, "got", what),
            expected_width: optional_usize(value, "expected", what),
            margin: optional_bool(value, "margin", what),
            confidence: optional_bool(value, "confidence", what),
        }
    }
}

fn decision_from_pin(value: &Json, what: &str) -> Decision {
    let kind = match string_field(value, "kind", what).as_str() {
        "propose" => DecisionKind::Propose,
        "abstain" => {
            let reason = match value
                .get("abstain_reason")
                .and_then(Json::as_str)
                .unwrap_or_else(|| panic!("{what}: abstain_reason required"))
            {
                "low_confidence" => AbstainReason::LowConfidence,
                "tie" => AbstainReason::Tie,
                other => panic!("{what}: unknown abstain_reason {other}"),
            };
            DecisionKind::Abstain(reason)
        }
        other => panic!("{what}: unknown kind {other}"),
    };
    let scores = value
        .get("scores")
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("{what}: missing scores"))
        .iter()
        .map(|item| {
            item.as_f64()
                .unwrap_or_else(|| panic!("{what}: scores are numbers"))
        })
        .collect();
    Decision {
        kind,
        diagnostics: spikenaut_snn::Diagnostics {
            winning_index: usize_field(value, "winning_index", what),
            winning_score: f64_field(value, "winning_score", what),
            winning_action: string_field(value, "winning_action", what),
            runner_up_index: optional_usize(value, "runner_up_index", what),
            runner_up_score: optional_f64(value, "runner_up_score", what),
            margin: f64_field(value, "margin", what),
            confidence: f64_field(value, "confidence", what),
            tied: bool_field(value, "tied", what),
            scores,
        },
    }
}

fn cases() -> Vec<PinCase> {
    let document = pin();
    let cases = document
        .get("cases")
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("{PIN_REL}: missing cases"));
    assert!(!cases.is_empty(), "an empty pin would pass vacuously");
    cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let what = format!("{PIN_REL} cases[{index}]");
            let case = require_object(case, &what);
            let name = string_field(case, "name", &what);
            let labelled = format!("{PIN_REL} cases[{name}]");
            let spikes = case.get("spikes").and_then(Json::as_array).map(|items| {
                items
                    .iter()
                    .map(|item| match item.as_f64() {
                        Some(0.0) => false,
                        Some(1.0) => true,
                        _ => panic!("{labelled}: spikes entries are 0 or 1"),
                    })
                    .collect()
            });
            PinCase {
                name,
                row: row_of(case, &labelled),
                config: case
                    .get("config")
                    .map(|config| config_of(config, &labelled))
                    .unwrap_or_else(|| Ok(DecisionConfig::shipped())),
                expected: expected_of(case, &labelled),
                spikes,
            }
        })
        .collect()
}

fn classify_error(err: &DecisionError) -> (&str, Vec<usize>, Option<usize>, Option<usize>) {
    match err {
        DecisionError::EmptyRow => ("empty_row", Vec::new(), None, None),
        DecisionError::WidthMismatch { got, expected } => {
            ("width_mismatch", Vec::new(), Some(*got), Some(*expected))
        }
        DecisionError::NonFinite { indices } => ("non_finite", indices.clone(), None, None),
        DecisionError::EmptyVocabulary => ("empty_vocabulary", Vec::new(), None, None),
        DecisionError::InvalidLabel { index } => ("invalid_label", vec![*index], None, None),
        DecisionError::DuplicateLabel { .. } => ("duplicate_label", Vec::new(), None, None),
        DecisionError::InvalidConfidenceFloor { .. } => {
            ("invalid_confidence_floor", Vec::new(), None, None)
        }
        DecisionError::DerivedNonFinite { .. } => ("derived_non_finite", Vec::new(), None, None),
    }
}

fn run_case(case: &PinCase) -> Result<Decision, DecisionError> {
    match &case.config {
        Ok(config) => decide(&case.row, config),
        Err(_) => {
            // The pin recorded a config that `DecisionConfig::new` itself
            // refuses. Reconstruct that refusal by calling new the same way
            // the pin's `config` object describes -- already failed above.
            unreachable!("run_case is not called when config construction failed")
        }
    }
}

/// Metadata the two languages stamp on the pin must stay in lockstep.
#[test]
fn pin_names_the_shipped_contract() {
    let document = pin();
    assert_eq!(string_field(&document, "contract", PIN_REL), CONTRACT_ID);
    assert_eq!(
        string_list(&document, "vocabulary", PIN_REL),
        SHIPPED_VOCABULARY
    );
    assert_eq!(
        usize_field(&document, "output_width", PIN_REL),
        OUTPUT_WIDTH
    );
    assert_eq!(
        usize_field(&document, "n_outputs_json", PIN_REL),
        OUTPUT_WIDTH
    );
    assert_eq!(
        usize_field(&document, "output_weight_count", PIN_REL),
        OUTPUT_WEIGHT_COUNT
    );
    assert_eq!(
        string_list(&document, "supervisor_vocabulary_unbound", PIN_REL),
        SupervisorAction::VOCABULARY
    );
    assert_eq!(
        string_field(&document, "tie_break", PIN_REL),
        "lowest-index"
    );
}

/// Every golden case: Rust `decide` matches the Python pin.
#[test]
fn rust_matches_the_python_pin() {
    let cases = cases();
    assert!(!cases.is_empty());
    for case in &cases {
        assert_pin_case(case);
    }
}

fn assert_pin_case(case: &PinCase) {
    match (&case.config, &case.expected) {
        (Err(message), Expected::Err { code, .. }) => {
            assert_config_error(case, message, code);
        }
        (Err(message), Expected::Ok(_)) => {
            panic!("{}: config failed ({message}) but the pin is ok", case.name)
        }
        (Ok(_), expected) => assert_run_case(case, expected),
    }
}

fn assert_config_error(case: &PinCase, message: &str, code: &str) {
    let needle = match code {
        "empty_vocabulary" => "vocabulary is empty",
        "invalid_label" => "label",
        "duplicate_label" => "repeats",
        "invalid_confidence_floor" => "confidence floor",
        other => panic!("{}: config error code {other}", case.name),
    };
    assert!(
        message.contains(needle),
        "{}: config error {message:?} did not match {code}",
        case.name
    );
}

fn assert_run_case(case: &PinCase, expected: &Expected) {
    match (run_case(case), expected) {
        (Ok(got), Expected::Ok(want)) => {
            assert_eq!(got, *want, "{}", case.name);
            assert_eq!(
                got.diagnostics.scores, case.row,
                "{}: diagnostics must copy the finite row",
                case.name
            );
        }
        (Err(err), Expected::Err { .. }) => assert_pin_error(case, &err),
        (Ok(got), Expected::Err { code, .. }) => {
            panic!("{}: Rust proposed {got:?}, pin error {code}", case.name)
        }
        (Err(err), Expected::Ok(_)) => {
            panic!("{}: Rust error {err}, pin is ok", case.name)
        }
    }
}

fn assert_pin_error(case: &PinCase, err: &DecisionError) {
    let Expected::Err {
        code,
        indices,
        got,
        expected_width,
        margin,
        confidence,
    } = &case.expected
    else {
        panic!("{}: assert_pin_error on an ok pin", case.name);
    };
    let (got_code, got_indices, got_got, got_expected) = classify_error(err);
    assert_eq!(got_code, code, "{}", case.name);
    if *code == "non_finite" {
        assert_eq!(got_indices, *indices, "{}", case.name);
    }
    if *code == "width_mismatch" {
        assert_eq!(got_got, *got, "{}", case.name);
        assert_eq!(got_expected, *expected_width, "{}", case.name);
    }
    if *code == "derived_non_finite" {
        match err {
            DecisionError::DerivedNonFinite {
                margin: got_margin,
                confidence: got_confidence,
            } => {
                assert_eq!(*got_margin, margin.unwrap_or(true), "{}", case.name);
                assert_eq!(*got_confidence, confidence.unwrap_or(true), "{}", case.name);
            }
            other => panic!(
                "{}: classified derived_non_finite but got {other:?}",
                case.name
            ),
        }
    }
}

/// `replay_output_row` is decide-with-shipped, and the checkpoint case keeps
/// Distill comfort at index 0.
#[test]
fn shipped_replay_keeps_distill_ordering() {
    let comfort = replay_output_row(&[0.9, 0.2, 0.1]).expect("finite 3-wide row");
    assert_eq!(comfort.kind, DecisionKind::Propose);
    assert_eq!(comfort.diagnostics.winning_index, 0);
    assert_eq!(comfort.diagnostics.winning_action, "comfort");
    assert!(!comfort.diagnostics.tied);

    let temp = replay_output_row(&[0.1, 0.8, 0.2]).expect("temp wins");
    assert_eq!(temp.diagnostics.winning_action, "temp");

    let power = replay_output_row(&[0.1, 0.2, 0.9]).expect("power wins");
    assert_eq!(power.diagnostics.winning_action, "power");

    let tied = replay_output_row(&[0.4, 0.4, 0.4]).expect("all-equal is a stable tie");
    assert_eq!(tied.kind, DecisionKind::Propose);
    assert!(tied.diagnostics.tied);
    assert_eq!(tied.diagnostics.winning_index, 0);
    assert_eq!(tied.diagnostics.winning_action, "comfort");
    assert_eq!(tied.diagnostics.confidence, 0.0);

    let checkpoint = cases()
        .into_iter()
        .find(|case| case.name == "checkpoint_neuron0_spike")
        .expect("the pin covers shipped ordering");
    let got = replay_output_row(&checkpoint.row).expect("neuron-0 scores are finite");
    assert_eq!(got.diagnostics.winning_action, "comfort");
    assert_eq!(got.diagnostics.winning_index, 0);
    if let Some(spikes) = checkpoint.spikes {
        let mem = read_output_mem();
        let scored = score_readout(&mem, &spikes).expect("shipped readout scores");
        assert_eq!(scored.as_slice(), checkpoint.row.as_slice());
        let replayed = replay_tick(&mem, &spikes).expect("replay_tick");
        assert_eq!(replayed, got);
    }
}

/// Empty, wrong-width, and non-finite rows fail closed: never a proposal.
#[test]
fn malformed_rows_do_not_propose() {
    assert!(matches!(
        replay_output_row(&[]),
        Err(DecisionError::EmptyRow)
    ));
    assert!(matches!(
        replay_output_row(&[0.1, 0.2]),
        Err(DecisionError::WidthMismatch {
            got: 2,
            expected: OUTPUT_WIDTH
        })
    ));
    assert_non_finite_row(&[f64::NAN, 0.0, 0.0], &[0]);
    assert_non_finite_row(&[0.0, f64::INFINITY, f64::NEG_INFINITY], &[1, 2]);
    assert_derived_row(&[f64::MAX, -f64::MAX, -f64::MAX], true, true);
    assert_derived_row(&[f64::MAX, f64::MAX / 2.0, 0.0], false, true);
    assert_overflowing_readout();
}

fn assert_non_finite_row(row: &[f64], want: &[usize]) {
    match replay_output_row(row) {
        Err(DecisionError::NonFinite { indices }) => assert_eq!(indices, want),
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

fn assert_derived_row(row: &[f64], margin: bool, confidence: bool) {
    match replay_output_row(row) {
        Err(DecisionError::DerivedNonFinite {
            margin: got_margin,
            confidence: got_confidence,
        }) => {
            assert_eq!(got_margin, margin);
            assert_eq!(got_confidence, confidence);
        }
        other => panic!("expected DerivedNonFinite, got {other:?}"),
    }
}

fn assert_overflowing_readout() {
    let mut weights = vec![0.0; OUTPUT_WEIGHT_COUNT];
    weights[0] = f64::MAX;
    weights[OUTPUT_WIDTH] = f64::MAX;
    let neuron_count = OUTPUT_WEIGHT_COUNT / OUTPUT_WIDTH;
    let mut spikes = vec![false; neuron_count];
    spikes[0] = true;
    spikes[1] = true;
    match score_readout(&weights, &spikes) {
        Err(DecisionError::NonFinite { indices }) => assert_eq!(indices, [0]),
        other => panic!("expected NonFinite on overflowing readout, got {other:?}"),
    }
}

/// Config validation paths the pin does not cover: only `empty_vocabulary`
/// is pinned, so the other three refusals are asserted here.
#[test]
fn invalid_configs_are_refused() {
    assert_eq!(
        DecisionConfig::new(Vec::<String>::new(), 0.0, false),
        Err(DecisionError::EmptyVocabulary)
    );
    assert_eq!(
        DecisionConfig::new(["a", "", "c"], 0.0, false),
        Err(DecisionError::InvalidLabel { index: 1 })
    );
    assert_eq!(
        DecisionConfig::new(["a", "b", "a"], 0.0, false),
        Err(DecisionError::DuplicateLabel { label: "a".into() })
    );
    // An empty label is reported before a later duplicate.
    assert_eq!(
        DecisionConfig::new(["", ""], 0.0, false),
        Err(DecisionError::InvalidLabel { index: 0 })
    );
    for floor in [
        -0.1,
        1.0 + f64::EPSILON,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        match DecisionConfig::new(SHIPPED_VOCABULARY, floor, false) {
            Err(DecisionError::InvalidConfidenceFloor { value }) => {
                assert!(value.to_bits() == floor.to_bits(), "{value} vs {floor}");
            }
            other => panic!("floor {floor}: expected InvalidConfidenceFloor, got {other:?}"),
        }
    }
    // Both ends of `[0, 1]` are legal floors.
    for floor in [0.0, -0.0, 1.0] {
        let config = DecisionConfig::new(SHIPPED_VOCABULARY, floor, true).expect("legal floor");
        assert_eq!(config.confidence_floor(), floor);
        assert!(config.abstain_on_tie());
    }
    // Owned and borrowed label sequences build the same config.
    let owned: Vec<String> = SHIPPED_VOCABULARY.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(
        DecisionConfig::new(owned, 0.0, false).unwrap(),
        DecisionConfig::shipped()
    );
}

/// Confidence *strictly* below the floor abstains; equal to it proposes.
#[test]
fn the_confidence_floor_is_strict() {
    // margin 2, denominator 4: confidence is exactly 0.5.
    let row = [3.0, 1.0, 0.0];
    let at = DecisionConfig::new(SHIPPED_VOCABULARY, 0.5, false).unwrap();
    let decision = decide(&row, &at).unwrap();
    assert_eq!(decision.diagnostics.confidence, 0.5);
    assert_eq!(decision.kind, DecisionKind::Propose);

    let above = DecisionConfig::new(SHIPPED_VOCABULARY, 0.5 + f64::EPSILON, false).unwrap();
    let decision = decide(&row, &above).unwrap();
    assert_eq!(
        decision.kind,
        DecisionKind::Abstain(AbstainReason::LowConfidence)
    );
    // Abstention keeps the would-be winner in the diagnostics.
    assert_eq!(decision.diagnostics.winning_action, "comfort");
    assert_eq!(decision.diagnostics.margin, 2.0);
}

/// A tie has confidence 0, so with a positive floor it abstains either way;
/// opting into tie abstention names the tie rather than the confidence.
#[test]
fn tie_abstention_takes_precedence_over_low_confidence() {
    let row = [0.25, 0.25, 0.0];
    let floor_only = DecisionConfig::new(SHIPPED_VOCABULARY, 0.1, false).unwrap();
    assert_eq!(
        decide(&row, &floor_only).unwrap().kind,
        DecisionKind::Abstain(AbstainReason::LowConfidence)
    );
    let both = DecisionConfig::new(SHIPPED_VOCABULARY, 0.1, true).unwrap();
    assert_eq!(
        decide(&row, &both).unwrap().kind,
        DecisionKind::Abstain(AbstainReason::Tie)
    );
    // A clear winner is not a tie, whatever the tie knob says.
    let clear = decide(&[0.9, 0.1, 0.0], &both).unwrap();
    assert_eq!(clear.kind, DecisionKind::Propose);
    assert!(!clear.diagnostics.tied);
}

/// `0.0 == -0.0` under IEEE comparison, so signed zeros tie and the lower
/// index wins -- the same answer Python's `==` gives.
#[test]
fn signed_zeros_tie() {
    let decision = replay_output_row(&[-0.0, 0.0, -1.0]).unwrap();
    assert!(decision.diagnostics.tied);
    assert_eq!(decision.diagnostics.winning_index, 0);
    assert_eq!(decision.diagnostics.runner_up_index, Some(1));
    assert_eq!(decision.diagnostics.confidence, 0.0);
}

/// Width 1 has no runner-up: margin 0, confidence 1, never abstains on
/// confidence even at the maximum floor.
#[test]
fn a_width_one_row_proposes_with_full_confidence() {
    let config = DecisionConfig::new(["only"], 1.0, true).unwrap();
    let decision = decide(&[-3.5], &config).unwrap();
    assert_eq!(decision.kind, DecisionKind::Propose);
    let diagnostics = &decision.diagnostics;
    assert_eq!(diagnostics.winning_index, 0);
    assert_eq!(diagnostics.winning_action, "only");
    assert_eq!(diagnostics.runner_up_index, None);
    assert_eq!(diagnostics.runner_up_score, None);
    assert_eq!(diagnostics.margin, 0.0);
    assert_eq!(diagnostics.confidence, 1.0);
    assert!(!diagnostics.tied);
}

/// Without a tie, the runner-up is the highest remaining score, lowest index
/// first among equals, wherever it sits relative to the winner.
#[test]
fn runner_up_is_the_lowest_index_of_the_next_highest_score() {
    let config = DecisionConfig::new(["a", "b", "c", "d", "e"], 0.0, false).unwrap();
    let decision = decide(&[2.0, 5.0, 1.0, 2.0, -1.0], &config).unwrap();
    let diagnostics = &decision.diagnostics;
    assert_eq!(diagnostics.winning_index, 1);
    assert_eq!(diagnostics.runner_up_index, Some(0));
    assert_eq!(diagnostics.runner_up_score, Some(2.0));
    assert_eq!(diagnostics.margin, 3.0);
    assert_eq!(diagnostics.confidence, 3.0 / 7.0);
    assert!(!diagnostics.tied);

    // A later max clears an earlier tie.
    let decision = decide(&[1.0, 1.0, 4.0, 0.0, 0.0], &config).unwrap();
    assert_eq!(decision.diagnostics.winning_index, 2);
    assert!(!decision.diagnostics.tied);
    assert_eq!(decision.diagnostics.runner_up_index, Some(0));
}

/// `score_readout` shape and value refusals, and the silent-row baseline.
#[test]
fn score_readout_refuses_malformed_images() {
    let weights = read_output_mem();
    let spikes = vec![false; OUTPUT_WEIGHT_COUNT / OUTPUT_WIDTH];

    assert_eq!(score_readout(&[], &spikes), Err(DecisionError::EmptyRow));
    assert_eq!(
        score_readout(&weights[1..], &spikes),
        Err(DecisionError::WidthMismatch {
            got: OUTPUT_WEIGHT_COUNT - 1,
            expected: OUTPUT_WEIGHT_COUNT
        })
    );
    assert_eq!(
        score_readout(&weights, &spikes[1..]),
        Err(DecisionError::WidthMismatch {
            got: spikes.len() - 1,
            expected: spikes.len()
        })
    );

    // A non-finite weight is refused even on a silent neuron: the image is
    // checked whole, not only the rows a spike happens to read.
    let mut poisoned = weights.clone();
    poisoned[5] = f64::NAN;
    poisoned[OUTPUT_WEIGHT_COUNT - 1] = f64::NEG_INFINITY;
    assert_eq!(
        score_readout(&poisoned, &spikes),
        Err(DecisionError::NonFinite {
            indices: vec![5, OUTPUT_WEIGHT_COUNT - 1]
        })
    );

    // No spikes, no score: an all-zero row, which replays as a full tie.
    assert_eq!(score_readout(&weights, &spikes), Ok([0.0; OUTPUT_WIDTH]));
    let silent = replay_tick(&weights, &spikes).unwrap();
    assert_eq!(silent.kind, DecisionKind::Propose);
    assert!(silent.diagnostics.tied);
    assert_eq!(silent.diagnostics.winning_action, "comfort");
    assert_eq!(silent.diagnostics.confidence, 0.0);
}

/// Every spike adds exactly its neuron's three weights, in channel order.
#[test]
fn score_readout_sums_the_spiking_neurons_rows() {
    let weights = read_output_mem();
    let neurons = OUTPUT_WEIGHT_COUNT / OUTPUT_WIDTH;
    let spikes: Vec<bool> = (0..neurons).map(|neuron| neuron % 3 == 1).collect();
    let mut expected = [0.0; OUTPUT_WIDTH];
    for neuron in (0..neurons).filter(|neuron| neuron % 3 == 1) {
        for (channel, score) in expected.iter_mut().enumerate() {
            *score += weights[neuron * OUTPUT_WIDTH + channel];
        }
    }
    assert_eq!(score_readout(&weights, &spikes), Ok(expected));
}

/// Display text mirrors the messages `tools/decision_core.py` raises for the
/// same refusals. The one spelling difference is quoting: Python's `repr`
/// writes a duplicate label as `'a'`, Rust's `Debug` as `"a"`.
#[test]
fn error_messages_match_the_python_reference() {
    for (err, want) in [
        (DecisionError::EmptyRow, "output row is empty"),
        (
            DecisionError::WidthMismatch {
                got: 2,
                expected: 3,
            },
            "output row width 2, expected 3",
        ),
        (
            DecisionError::NonFinite { indices: vec![0] },
            "non-finite output score at index 0",
        ),
        (
            DecisionError::NonFinite {
                indices: vec![1, 2],
            },
            "non-finite output scores at indexes 1, 2",
        ),
        (
            DecisionError::EmptyVocabulary,
            "decision vocabulary is empty",
        ),
        (
            DecisionError::InvalidLabel { index: 1 },
            "decision vocabulary label 1 is empty",
        ),
        (
            DecisionError::DuplicateLabel { label: "a".into() },
            "decision vocabulary repeats \"a\"",
        ),
        (
            DecisionError::InvalidConfidenceFloor { value: 1.5 },
            "confidence floor 1.5 is not a finite value in [0, 1]",
        ),
        (
            DecisionError::DerivedNonFinite {
                margin: true,
                confidence: true,
            },
            "derived margin and confidence are not finite",
        ),
        (
            DecisionError::DerivedNonFinite {
                margin: true,
                confidence: false,
            },
            "derived margin is not finite",
        ),
        (
            DecisionError::DerivedNonFinite {
                margin: false,
                confidence: true,
            },
            "derived confidence is not finite",
        ),
        (
            DecisionError::DerivedNonFinite {
                margin: false,
                confidence: false,
            },
            "derived diagnostics are not finite",
        ),
    ] {
        assert_eq!(err.to_string(), want);
    }
}

fn read_output_mem() -> Vec<f64> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("dataset/merged_v2/parameters_output_weights.mem");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let words: Vec<f64> = text
        .split_whitespace()
        .map(|code| {
            let bits = u16::from_str_radix(code, 16)
                .unwrap_or_else(|err| panic!("{}: bad code {code:?}: {err}", path.display()));
            f64::from(bits.cast_signed()) / 256.0
        })
        .collect();
    assert_eq!(words.len(), OUTPUT_WEIGHT_COUNT);
    words
}

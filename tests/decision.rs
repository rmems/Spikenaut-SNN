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
    let mut checked = 0usize;
    for case in &cases {
        checked += 1;
        match (&case.config, &case.expected) {
            (Err(message), Expected::Err { code, .. }) => {
                assert!(
                    message.contains(match code.as_str() {
                        "empty_vocabulary" => "vocabulary is empty",
                        "invalid_label" => "label",
                        "duplicate_label" => "repeats",
                        "invalid_confidence_floor" => "confidence floor",
                        other => panic!("{}: config error code {other}", case.name),
                    }),
                    "{}: config error {message:?} did not match {code}",
                    case.name
                );
            }
            (Err(message), Expected::Ok(_)) => {
                panic!("{}: config failed ({message}) but the pin is ok", case.name)
            }
            (Ok(_), expected) => match (run_case(case), expected) {
                (Ok(got), Expected::Ok(want)) => {
                    assert_eq!(got, *want, "{}", case.name);
                    assert_eq!(
                        got.diagnostics.scores, case.row,
                        "{}: diagnostics must copy the finite row",
                        case.name
                    );
                }
                (
                    Err(err),
                    Expected::Err {
                        code,
                        indices,
                        got,
                        expected_width,
                    },
                ) => {
                    let (got_code, got_indices, got_got, got_expected) = classify_error(&err);
                    assert_eq!(got_code, code, "{}", case.name);
                    if *code == "non_finite" {
                        assert_eq!(got_indices, *indices, "{}", case.name);
                    }
                    if *code == "width_mismatch" {
                        assert_eq!(got_got, *got, "{}", case.name);
                        assert_eq!(got_expected, *expected_width, "{}", case.name);
                    }
                }
                (Ok(got), Expected::Err { code, .. }) => {
                    panic!("{}: Rust proposed {got:?}, pin error {code}", case.name)
                }
                (Err(err), Expected::Ok(_)) => {
                    panic!("{}: Rust error {err}, pin is ok", case.name)
                }
            },
        }
    }
    assert_eq!(checked, cases.len());
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
    let err = replay_output_row(&[f64::NAN, 0.0, 0.0]).unwrap_err();
    match err {
        DecisionError::NonFinite { indices } => assert_eq!(indices, [0]),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    let err = replay_output_row(&[0.0, f64::INFINITY, f64::NEG_INFINITY]).unwrap_err();
    match err {
        DecisionError::NonFinite { indices } => assert_eq!(indices, [1, 2]),
        other => panic!("expected NonFinite, got {other:?}"),
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

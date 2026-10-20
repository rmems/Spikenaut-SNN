// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for [`super`]: the decoder, the Q8.8 grid, and the builders.
//!
//! Split out of `model.rs` rather than reorganized: every test is unchanged
//! and still a child module of `model`, so `use super::*` continues to reach
//! the private decode helpers it exercises.

use super::*;
use nir_rs::types::TensorData;

fn stub_json(units: usize) -> String {
    let rows: Vec<String> = (0..units)
        .map(|i| {
            let weights: Vec<String> = (0..units).map(|j| format!("{}.0", i + j)).collect();
            format!(
                r#"{{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [{}], "last_spike": false}}"#,
                weights.join(", ")
            )
        })
        .collect();
    format!(r#"{{"neurons": [{}]}}"#, rows.join(", "))
}

#[test]
fn loads_shipped_model() {
    let model = SnnModel::load_default().unwrap();
    assert_eq!(model.len(), NEURON_COUNT);
    assert!(!model.is_empty());
    assert_eq!(model.neurons[0].weights.len(), NEURON_COUNT);
    assert_eq!(model.neurons[0].threshold, 410.0 / 256.0);
    assert_eq!(model.neurons[0].decay_rate, 218.0 / 256.0);
    assert_eq!(model.neurons[0].membrane_potential, 280.0 / 256.0);
    assert!(!model.neurons[0].last_spike);
    assert_eq!(model.thresholds().len(), NEURON_COUNT);
    assert_eq!(model.decay_rates().len(), NEURON_COUNT);
    assert!(MERGED_V2_PROVENANCE.contains("shipped merged_v2 artifact"));
    assert!(MERGED_V2_PROVENANCE.contains("exp-025 Dale health-PASS"));
    assert!(MERGED_V2_PROVENANCE.contains("post-exp-009 legal-encoder retrain"));
    assert!(!MERGED_V2_PROVENANCE.contains("not a post-exp-009"));
}

#[test]
fn load_default_matches_the_checkout_file() {
    let embedded = SnnModel::load_default().unwrap();
    let from_file = SnnModel::from_path(default_model_path()).unwrap();
    assert_eq!(embedded, from_file);
}

#[test]
fn shipped_loader_requires_sixteen_units() {
    let stub = SnnModel::from_json_str(&stub_json(3)).unwrap();
    let err = require_merged_v2_width(&stub).unwrap_err();
    assert!(err.to_string().contains("must have 16 neurons"));
    let shipped = SnnModel::from_path(default_model_path()).unwrap();
    require_merged_v2_width(&shipped).unwrap();
}

#[test]
fn weight_tensor_is_square_and_row_major() {
    let model = SnnModel::from_json_str(&stub_json(3)).unwrap();
    let tensor = model.weight_tensor().unwrap();
    assert_eq!(tensor.shape(), [3, 3]);
    assert_eq!(tensor.numel(), 9);
    let TensorData::F64(values) = tensor.data() else {
        panic!("expected an f64 tensor");
    };
    // Row i, column j holds i + j for the stub.
    assert_eq!(values, &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
}

/// Flattening throws away the row boundaries, so the total length is not a
/// sufficient check: rows of 3 and 1 also total 4 and would be accepted as
/// a 2×2 tensor with `4.0` shifted from unit 1 column 0 into column 1.
/// `SnnModel`'s fields are public, so decoding is not the only way in.
#[test]
fn weight_tensor_rejects_ragged_rows() {
    let row = |weights: Vec<f64>| Neuron {
        decay_rate: 0.5,
        membrane_potential: 0.0,
        threshold: 1.0,
        last_spike: false,
        weights,
    };

    let ragged = SnnModel {
        neurons: vec![row(vec![1.0, 2.0, 3.0]), row(vec![4.0])],
    };
    // The corrupting case: 3 + 1 == 2 * 2, so the length check alone passes.
    assert_eq!(
        ragged
            .neurons
            .iter()
            .map(|n| n.weights.len())
            .sum::<usize>(),
        ragged.len() * ragged.len(),
    );
    let err = ragged.weight_tensor().unwrap_err();
    assert!(matches!(err, ModelError::Schema(_)), "{err}");
    let message = err.to_string();
    assert!(message.contains("neuron 0"), "{message}");
    assert!(message.contains("3 weights"), "{message}");
    assert!(message.contains("expected 2"), "{message}");

    // A short row that does not sum to units² is caught too, and named.
    let short = SnnModel {
        neurons: vec![row(vec![1.0, 2.0]), row(vec![3.0])],
    };
    let message = short.weight_tensor().unwrap_err().to_string();
    assert!(message.contains("neuron 1"), "{message}");

    // Square rows still build.
    let square = SnnModel {
        neurons: vec![row(vec![1.0, 2.0]), row(vec![3.0, 4.0])],
    };
    let tensor = square.weight_tensor().unwrap();
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensor.data(), &TensorData::F64(vec![1.0, 2.0, 3.0, 4.0]));
}

/// A model mutated after decoding must not corrupt the matrix either.
#[test]
fn weight_tensor_rejects_a_mutated_row() {
    let mut model = SnnModel::load_default().unwrap();
    assert!(model.weight_tensor().is_ok());

    // Move one weight from unit 3 to unit 4: the total is still 256.
    let moved = model.neurons[3].weights.pop().unwrap();
    model.neurons[4].weights.push(moved);
    assert_eq!(
        model.neurons.iter().map(|n| n.weights.len()).sum::<usize>(),
        NEURON_COUNT * NEURON_COUNT,
    );

    let message = model.weight_tensor().unwrap_err().to_string();
    assert!(message.contains("neuron 3"), "{message}");
    assert!(message.contains("15 weights"), "{message}");
}

#[test]
fn tau_inverts_the_decay_multiplier() {
    let tau = tau_from_decay(0.5, TIMESTEP_SECONDS).unwrap();
    // Re-applying one step of decay must return the original multiplier.
    assert!(((-TIMESTEP_SECONDS / tau).exp() - 0.5).abs() < 1e-12);
    // A slower decay integrates over a longer time constant.
    assert!(tau_from_decay(0.9, TIMESTEP_SECONDS).unwrap() > tau);
}

#[test]
fn tau_rejects_out_of_range_inputs() {
    for decay in [0.0, 1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        assert!(tau_from_decay(decay, TIMESTEP_SECONDS).is_err());
    }
    assert!(tau_from_decay(0.5, 0.0).is_err());
    assert!(tau_from_decay(0.5, -1.0).is_err());
    assert!(tau_from_decay(0.5, f64::NAN).is_err());
    assert!(tau_from_decay(0.5, f64::INFINITY).is_err());
}

/// Both inputs can be finite and in range while the quotient is not. The
/// documented contract is that an uninvertible input errors, so a
/// non-finite `tau` must not escape as `Ok` — it would reach a NIR `LIF`
/// node as a meaningless time constant.
#[test]
fn tau_rejects_non_finite_results() {
    // Overflow: -f64::MAX / ln(0.5) is about 1.44 * f64::MAX.
    let err = tau_from_decay(0.5, f64::MAX).unwrap_err();
    assert!(matches!(err, ModelError::Schema(_)));
    assert!(err.to_string().contains("time constant"), "{err}");
    assert!(tau_from_decay(0.5, 1e308).is_ok(), "1e308 still inverts");

    // Underflow: a subnormal timestep against the smallest decay rate the
    // Q8.8 grid can hold (1/256) divides away to exactly zero.
    let smallest_on_grid = 1.0 / Q8_8_SCALE;
    let subnormal: f64 = 5e-324;
    assert!(subnormal > 0.0 && subnormal.is_finite());
    let err = tau_from_decay(smallest_on_grid, subnormal).unwrap_err();
    assert!(matches!(err, ModelError::Schema(_)));
    assert!(err.to_string().contains("time constant"), "{err}");

    // Nothing in between is rejected: every value that does invert into a
    // finite positive tau still succeeds.
    for dt in [1e-300, 1e-10, TIMESTEP_SECONDS, 1.0, 1e100] {
        for decay in [1.0 / 256.0, 0.5, 0.796875, 0.94921875] {
            let tau = tau_from_decay(decay, dt).unwrap();
            assert!(tau.is_finite() && tau > 0.0, "tau({decay}, {dt}) = {tau}");
        }
    }
}

/// A non-finite `tau` must not reach the graph either.
#[test]
fn taus_seconds_rejects_extreme_timesteps() {
    let model = SnnModel::load_default().unwrap();
    assert!(model.taus_seconds(f64::MAX).is_err());
    assert!(model.taus_seconds(TIMESTEP_SECONDS).is_ok());

    // The shipped decays are keep `00DA` = 0.8515625, so `|ln(decay)|` is
    // about 0.161 and a subnormal timestep still divides to a nonzero
    // subnormal — no underflow for this model.
    assert!(model.taus_seconds(5e-324).is_ok());

    // Underflow needs a decay far from 1. `1/256` is the smallest the Q8.8
    // grid holds, so this is a decodable model, not an impossible one.
    let fast = SnnModel {
        neurons: vec![Neuron {
            decay_rate: 1.0 / Q8_8_SCALE,
            membrane_potential: 0.0,
            threshold: 1.0,
            last_spike: false,
            weights: vec![0.0],
        }],
    };
    assert!(fast.taus_seconds(5e-324).is_err());
    assert!(fast.taus_seconds(TIMESTEP_SECONDS).is_ok());
}

#[test]
fn taus_cover_every_unit() {
    let model = SnnModel::load_default().unwrap();
    let taus = model.taus_seconds(TIMESTEP_SECONDS).unwrap();
    assert_eq!(taus.len(), NEURON_COUNT);
    assert!(taus.iter().all(|tau| *tau > 0.0 && tau.is_finite()));
}

#[test]
fn rejects_malformed_models() {
    let cases = [
        (r#"{}"#, "missing top-level"),
        (r#"{"neurons": 3}"#, "not an array"),
        (r#"{"neurons": []}"#, "is empty"),
        (
            r#"{"neurons": [{"membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "missing `decay_rate`",
        ),
        (
            r#"{"neurons": [{"decay_rate": "x", "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "expected a number",
        ),
        (
            r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": 1.0, "last_spike": false}]}"#,
            "expected an array",
        ),
        (
            r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0, 2.0], "last_spike": false}]}"#,
            "expected 1",
        ),
        (
            r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": 0}]}"#,
            "expected a boolean",
        ),
        (r#"[]"#, "the document is not an object (found array)"),
        (
            r#"{"neurons": [3]}"#,
            "neuron 0 is not an object (found number)",
        ),
        (
            r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}], "output_weights": [[0.1]]}"#,
            "the document carries unknown member(s) `output_weights`",
        ),
        (
            r#"{"neurons": [{"cell_type": "E", "decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "neuron 0 carries unknown member(s) `cell_type`",
        ),
    ];
    for (text, expected) in cases {
        let err = SnnModel::from_json_str(text).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(expected),
            "expected {message:?} to contain {expected:?}"
        );
    }
}

/// `quantize_q8_8` rounds and does not saturate, and that is on purpose.
///
/// Clamping an out-of-range input down to the nearest code would turn a
/// wrong number into a plausible weight and destroy the evidence — the
/// silent clamp-to-range failure `tools/verify_q88.py` was written to
/// catch. It would also disarm `q8_8_field`, which refuses a bad value
/// precisely *because* the snapped result is still out of range: under a
/// saturating quantizer, 200.0 would load as `Q8_8_MAX` and the model
/// would be silently wrong rather than loudly rejected.
///
/// So this pins the behaviour against a well-meaning future fix.
#[test]
fn quantizing_never_silently_clamps_into_range() {
    for out_of_range in [128.0_f64, 200.0, -128.5, -1000.0] {
        let snapped = quantize_q8_8(out_of_range);
        assert!(
            !is_q8_8(snapped),
            "{out_of_range} must not be rounded into a valid code, got {snapped}"
        );
        // And the decode path is what turns that into a refusal.
        let err = q8_8_field("weight", out_of_range).unwrap_err();
        assert!(
            err.to_string().contains("outside the Q8.8 range"),
            "unexpected error for {out_of_range}: {err}"
        );
    }

    // The documented edges of the contract, stated so they cannot drift.
    assert!(!is_q8_8(f64::NAN) && quantize_q8_8(f64::NAN).is_nan());
    assert!(!is_q8_8(quantize_q8_8(f64::MAX)));
    assert!(
        quantize_q8_8(f64::MAX).is_infinite(),
        "the product overflows"
    );

    // In range, it does exactly what the name says.
    assert!(
        is_q8_8(quantize_q8_8(0.7539062)),
        "the printed form snaps on-grid"
    );
    assert!(
        is_q8_8(Q8_8_MAX) && is_q8_8(Q8_8_MIN),
        "both extremes are codes"
    );
    assert!(!is_q8_8(0.7539062), "off-grid before snapping");
}

/// The extreme Q8.8 codes must survive being printed.
///
/// `snn_model.json` is written at seven significant digits — that is why
/// `0.7539062` appears in it for code 193 — and at that precision the
/// largest code, 127.996_093_75, is written `127.9961`, which is 6.25e-6
/// *above* `Q8_8_MAX`. Checking the literal rejected it, so a model holding
/// a saturated weight or threshold could not be loaded at all. The shipped
/// artifact tops out at 1.59375 and never reached this, but the retrain in
/// #2/#3 regenerates these files with a wider weight range.
///
/// The check is on what the number encodes to, which does not widen the
/// range: anything that snaps past the top code still fails.
#[test]
fn the_extreme_codes_survive_seven_digit_printing() {
    // Both ends, as a producer would print them.
    for (printed, expected) in [(127.9961_f64, Q8_8_MAX), (-128.0_f64, Q8_8_MIN)] {
        let value =
            q8_8_field("threshold", printed).unwrap_or_else(|e| panic!("{printed} must load: {e}"));
        assert_eq!(value, expected, "{printed} must snap to the extreme code");
    }

    // Half an LSB is the whole of the tolerance: these still round onto a
    // real code. 127.998 scales to 32767.488, which is code 32767.
    for inside in [127.998_f64, -128.001] {
        q8_8_field("threshold", inside)
            .unwrap_or_else(|e| panic!("{inside} rounds onto a code: {e}"));
    }

    // And the range is still a range. Each of these snaps past an end:
    // 127.998_046_875 scales to exactly 32767.5 and rounds away from zero
    // to code 32768, which 128.0 cannot hold.
    for beyond in [
        127.998_046_875_f64,
        -128.001_953_125,
        200.0,
        -200.0,
        f64::MAX,
        f64::MIN,
    ] {
        let err = q8_8_field("threshold", beyond).unwrap_err();
        assert!(
            err.to_string().contains("outside the Q8.8 range"),
            "unexpected error for {beyond}: {err}"
        );
    }

    // The public-API boundary is stricter on purpose. It takes the exact
    // code, and refuses the printed form: a caller setting a threshold in
    // code is not working around a formatter, so there is no rounding to
    // forgive. Which of the two true complaints it makes — off-grid, or
    // above the top code, both of which `127.9961` is — is not pinned here.
    check_q8_8("threshold", Q8_8_MAX).expect("the exact code is representable");
    check_q8_8("threshold", 127.9961).expect_err("the printed form is not exact");
}

/// A member this decoder does not read must not pass silently, because the
/// builder stamps `Provenance::MERGED_V2` on what comes out.
///
/// exp-025 allowlists sidecar `output_weights` / `inhibitory`. A member
/// outside that list must still fail, or a later revision would decode by
/// silent truncation while the graph kept the `merged_v2` stamp.
#[test]
fn an_unknown_member_is_never_dropped_silently() {
    let shipped = std::fs::read_to_string(default_model_path()).unwrap();
    SnnModel::from_json_str(&shipped).expect("the shipped artifact decodes");

    // The same artifact, with one member the decoder does not read.
    let widened = shipped.replacen(
        r#""decay_rate""#,
        r#""output_weight": 0.5, "decay_rate""#,
        1,
    );
    assert_ne!(widened, shipped, "the fixture must actually differ");
    let err = SnnModel::from_json_str(&widened).unwrap_err();
    assert!(
        err.to_string()
            .contains("unknown member(s) `output_weight`"),
        "a widened artifact must be refused, got: {err}"
    );
}

/// The stored decimals are truncated Q8.8 codes; decoding must restore the
/// exact values, or the graph is not the model the FPGA runs.
#[test]
fn decoding_restores_exact_q8_8_values() {
    let model = SnnModel::load_default().unwrap();

    // `0.85` in the JSON; every `parameters_decay.mem` line is `00DA`.
    assert_eq!(model.neurons[1].decay_rate, 218.0 / 256.0);
    assert_eq!(model.neurons[1].decay_rate, 0.851_562_5);
    // First hidden row, column 1; `parameters_weights.mem` line 2 is `000E`.
    assert_eq!(model.neurons[0].weights[1], 14.0 / 256.0);
    assert_eq!(model.neurons[0].weights[1], 0.054_687_5);

    // No stored number is left off the grid.
    for neuron in &model.neurons {
        for value in [
            neuron.decay_rate,
            neuron.threshold,
            neuron.membrane_potential,
        ]
        .into_iter()
        .chain(neuron.weights.iter().copied())
        {
            assert_eq!(value * Q8_8_SCALE, (value * Q8_8_SCALE).round(), "{value}");
        }
    }
}

#[test]
fn quantize_q8_8_snaps_to_the_nearest_code() {
    assert_eq!(quantize_q8_8(0.808_594), 207.0 / 256.0);
    assert_eq!(quantize_q8_8(0.753_906_2), 193.0 / 256.0);
    // Already on the grid: unchanged.
    assert_eq!(quantize_q8_8(1.125), 1.125);
    assert_eq!(quantize_q8_8(-0.027_343_75), -0.027_343_75);
    // Never moves by more than half an LSB.
    for code in -2000..2000 {
        let value = f64::from(code) / 512.0;
        assert!((quantize_q8_8(value) - value).abs() <= 0.5 / 256.0);
    }
}

/// `decay_rate` is documented as lying in `(0, 1)`; decoding is the boundary
/// that enforces it, so a caller reading the fields or the weight tensor
/// directly can never hold an invalid model.
#[test]
fn decoding_enforces_numeric_invariants() {
    let cases = [
        (
            r#"{"neurons": [{"decay_rate": 0.0, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "decay multiplier in (0, 1)",
        ),
        (
            r#"{"neurons": [{"decay_rate": 1.0, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "decay multiplier in (0, 1)",
        ),
        (
            r#"{"neurons": [{"decay_rate": -0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "decay multiplier in (0, 1)",
        ),
        (
            r#"{"neurons": [{"decay_rate": 1.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "decay multiplier in (0, 1)",
        ),
        (
            r#"{"neurons": [{"decay_rate": 1e300, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "outside the Q8.8 range",
        ),
        (
            r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 500.0, "weights": [1.0], "last_spike": false}]}"#,
            "outside the Q8.8 range",
        ),
        (
            r#"{"neurons": [{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [-1e9], "last_spike": false}]}"#,
            "outside the Q8.8 range",
        ),
    ];
    for (text, expected) in cases {
        let err = SnnModel::from_json_str(text).unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, ModelError::Schema(_)), "{message}");
        assert!(
            message.contains(expected),
            "expected {message:?} to contain {expected:?}"
        );
    }

    // A decay rate that rounds *onto* a boundary is rejected too: 0.999 is
    // one LSB short of 1.0 and snaps to exactly 1.0.
    let err = SnnModel::from_json_str(
        r#"{"neurons": [{"decay_rate": 0.999, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
    )
    .unwrap_err();
    assert!(err.to_string().contains("decay multiplier in (0, 1)"));
}

#[test]
fn reports_json_and_io_failures() {
    let err = SnnModel::from_json_str("not json").unwrap_err();
    assert!(matches!(err, ModelError::Json(_)));
    assert!(err.to_string().contains("cannot parse model JSON"));

    let missing = default_model_path().with_file_name("does-not-exist.json");
    let err = SnnModel::from_path(&missing).unwrap_err();
    assert!(matches!(err, ModelError::Io { .. }));
    assert!(err.to_string().contains("does-not-exist.json"));
}

/// `std::error::Error::source` chains to the underlying failure where there is
/// one, and a schema error is its own root cause.
#[test]
fn model_errors_expose_their_source() {
    use std::error::Error as _;

    let json = SnnModel::from_json_str("not json").unwrap_err();
    assert!(json.source().is_some(), "{json}");

    let io = SnnModel::from_path(default_model_path().with_file_name("does-not-exist.json"))
        .unwrap_err();
    assert!(io.source().is_some(), "{io}");

    let schema = SnnModel::from_json_str(r#"{"neurons": []}"#).unwrap_err();
    assert!(matches!(schema, ModelError::Schema(_)));
    assert!(schema.source().is_none(), "{schema}");
    assert!(schema.to_string().starts_with("invalid SNN model: "));
}

/// Ties round half away from zero, symmetric about zero: the rule
/// `tools/q88_core.py::encode_q88` uses (`floor(x + 0.5)` / `ceil(x - 0.5)`).
/// A banker's or floor rounder agrees with it everywhere except exactly here,
/// and a floor rounder would bias every negative weight.
#[test]
fn quantize_q8_8_rounds_ties_away_from_zero() {
    let lsb = 1.0 / Q8_8_SCALE;
    assert_eq!(quantize_q8_8(0.5 * lsb), lsb);
    assert_eq!(quantize_q8_8(-0.5 * lsb), -lsb);
    // 2.5 LSB: round-half-to-even would give 2, away-from-zero gives 3.
    assert_eq!(quantize_q8_8(2.5 * lsb), 3.0 * lsb);
    assert_eq!(quantize_q8_8(-2.5 * lsb), -3.0 * lsb);

    // Symmetric about zero on the whole half-LSB lattice, ties included.
    for k in -2000..=2000 {
        let value = f64::from(k) / 512.0;
        assert_eq!(quantize_q8_8(-value), -quantize_q8_8(value), "{value}");
    }

    // Negative zero is code 0 and keeps its sign through the snap.
    assert!(quantize_q8_8(-0.0).is_sign_negative());
    assert!(is_q8_8(-0.0));
}

/// `is_q8_8` is true for every code and nothing else: off-grid, out of range
/// and non-finite values are all refused.
#[test]
fn is_q8_8_accepts_exactly_the_representable_codes() {
    for code in [
        i16::MIN,
        i16::MIN + 1,
        -1,
        0,
        1,
        193,
        i16::MAX - 1,
        i16::MAX,
    ] {
        let value = f64::from(code) / Q8_8_SCALE;
        assert!(is_q8_8(value), "code {code} ({value})");
    }
    assert_eq!(f64::from(i16::MAX) / Q8_8_SCALE, Q8_8_MAX);
    assert_eq!(f64::from(i16::MIN) / Q8_8_SCALE, Q8_8_MIN);

    for value in [
        0.1,
        0.5 / Q8_8_SCALE,
        Q8_8_MAX + 1.0 / Q8_8_SCALE,
        Q8_8_MIN - 1.0 / Q8_8_SCALE,
        f64::MIN_POSITIVE,
        5e-324,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        assert!(!is_q8_8(value), "{value}");
    }
}

/// The public-API check never rounds, and says which way the value missed.
#[test]
fn check_q8_8_reports_each_failure_without_rounding() {
    // 0.1 * 256 = 25.6, between codes 25 and 26.
    let message = check_q8_8("w", 0.1).unwrap_err().to_string();
    assert!(message.contains("not Q8.8-representable"), "{message}");
    assert!(
        message.contains("nearest codes 25/256 and 26/256"),
        "{message}"
    );

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let message = check_q8_8("w", bad).unwrap_err().to_string();
        assert!(message.contains("expected a finite number"), "{message}");
    }
    for out in [128.0, Q8_8_MIN - 1.0 / Q8_8_SCALE] {
        let message = check_q8_8("w", out).unwrap_err().to_string();
        assert!(message.contains("outside the Q8.8 range"), "{message}");
    }

    for code in [Q8_8_MIN, -0.0, 0.0, Q8_8_MAX] {
        check_q8_8("w", code).unwrap_or_else(|e| panic!("{code} is a code: {e}"));
    }
}

/// `weight_tensor` re-checks hand-built weights rather than trusting them,
/// and never rounds one on the caller's behalf.
#[test]
fn weight_tensor_refuses_a_hand_built_weight_off_the_q8_8_grid() {
    let unit = |weight: f64| Neuron {
        decay_rate: 0.5,
        membrane_potential: 0.0,
        threshold: 1.0,
        last_spike: false,
        weights: vec![weight],
    };

    for (weight, expected) in [
        (0.1, "not Q8.8-representable"),
        (f64::NAN, "expected a finite number"),
        (f64::INFINITY, "expected a finite number"),
        (200.0, "outside the Q8.8 range"),
    ] {
        let model = SnnModel {
            neurons: vec![unit(weight)],
        };
        let err = model.weight_tensor().unwrap_err();
        assert!(matches!(err, ModelError::Schema(_)), "{err}");
        let message = err.to_string();
        assert!(message.contains("neuron 0 weight 0"), "{message}");
        assert!(message.contains(expected), "{weight}: {message}");
    }

    let on_grid = SnnModel {
        neurons: vec![unit(193.0 / 256.0)],
    };
    assert_eq!(
        on_grid.weight_tensor().unwrap().data(),
        &TensorData::F64(vec![193.0 / 256.0]),
    );
}

/// Every type error names the unit and the field it came from, and a neuron
/// with several faults reports them in the documented precedence: weights,
/// then `last_spike`, then the scalars.
#[test]
fn decoding_names_the_unit_and_field_of_every_fault() {
    let ok = r#"{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0, 2.0], "last_spike": false}"#;
    let cases = [
        (
            r#"{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": ["x", 2.0], "last_spike": false}"#,
            "neuron 1 weight 0 is a string, expected a number",
        ),
        (
            r#"{"decay_rate": 0.5, "threshold": 1.0, "weights": [1.0, 2.0], "last_spike": false}"#,
            "neuron 1 is missing `membrane_potential`",
        ),
        (
            r#"{"decay_rate": 0.5, "membrane_potential": 0.0, "weights": [1.0, 2.0], "last_spike": false}"#,
            "neuron 1 is missing `threshold`",
        ),
        (
            r#"{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "last_spike": false}"#,
            "neuron 1 is missing `weights`",
        ),
        (
            r#"{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0, 2.0]}"#,
            "neuron 1 is missing `last_spike`",
        ),
        (
            r#"{"decay_rate": 0.5, "membrane_potential": 0.0, "threshold": null, "weights": [1.0, 2.0], "last_spike": false}"#,
            "neuron 1 field `threshold` is a null, expected a number",
        ),
        (
            r#"{"decay_rate": 0.5, "membrane_potential": 300.0, "threshold": 1.0, "weights": [1.0, 2.0], "last_spike": false}"#,
            "neuron 1 field `membrane_potential` is 300, outside the Q8.8 range",
        ),
        // Precedence: a bad weight wins over a bad decay and a bad flag.
        (
            r#"{"decay_rate": 5.0, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0, 999.0], "last_spike": 1}"#,
            "neuron 1 weight 1 is 999, outside the Q8.8 range",
        ),
        // Then `last_spike` wins over a bad scalar.
        (
            r#"{"decay_rate": 5.0, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0, 2.0], "last_spike": 1}"#,
            "neuron 1 field `last_spike` is a number, expected a boolean",
        ),
    ];
    for (second, expected) in cases {
        let text = format!(r#"{{"neurons": [{ok}, {second}]}}"#);
        let err = SnnModel::from_json_str(&text).unwrap_err();
        assert!(matches!(err, ModelError::Schema(_)), "{err}");
        let message = err.to_string();
        assert!(
            message.contains(expected),
            "expected {message:?} to contain {expected:?}"
        );
    }

    // And the healthy pair decodes, so the cases above fail on their fault.
    let pair = SnnModel::from_json_str(&format!(r#"{{"neurons": [{ok}, {ok}]}}"#)).unwrap();
    assert_eq!(pair.len(), 2);
}

/// The decay rate is snapped onto the Q8.8 grid like every other number, and
/// the `(0, 1)` check runs on the snapped value: the smallest positive code
/// loads, and anything that snaps to zero does not.
#[test]
fn decay_rate_is_snapped_before_its_open_interval_check() {
    let model_with_decay = |decay: f64| {
        SnnModel::from_json_str(&format!(
            r#"{{"neurons": [{{"decay_rate": {decay}, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}}]}}"#
        ))
    };

    let smallest = model_with_decay(0.003_906_25).unwrap();
    assert_eq!(smallest.neurons[0].decay_rate, 1.0 / Q8_8_SCALE);
    let largest = model_with_decay(0.996_093_8).unwrap();
    assert_eq!(largest.neurons[0].decay_rate, 255.0 / Q8_8_SCALE);

    // 0.001 snaps to code 0, which is not a decay multiplier.
    let message = model_with_decay(0.001).unwrap_err().to_string();
    assert!(message.contains("decay multiplier in (0, 1)"), "{message}");
}

#[test]
fn time_base_constants_agree() {
    assert_eq!(TIMESTEP_SECONDS, 0.001);
    assert_eq!(TIMESTEP_SECONDS * CLOCK_HZ, 1.0);
    assert!(default_model_path().ends_with(MODEL_RELATIVE_PATH));
}

/// A hand-built neuron is re-checked by `tau_seconds` rather than trusted.
#[test]
fn tau_seconds_rechecks_a_hand_built_decay() {
    let unit = |decay_rate: f64| Neuron {
        decay_rate,
        membrane_potential: 0.0,
        threshold: 1.0,
        last_spike: false,
        weights: vec![0.0],
    };
    for decay in [0.0, 1.0, -0.25, f64::NAN] {
        assert!(
            unit(decay).tau_seconds(TIMESTEP_SECONDS).is_err(),
            "{decay}"
        );
    }
    assert_eq!(
        unit(0.5).tau_seconds(TIMESTEP_SECONDS).unwrap(),
        tau_from_decay(0.5, TIMESTEP_SECONDS).unwrap(),
    );
}

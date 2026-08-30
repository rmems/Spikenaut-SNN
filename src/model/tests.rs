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
    assert_eq!(model.neurons[0].threshold, 1.125);
    assert_eq!(model.neurons[0].decay_rate, 0.796875);
    assert_eq!(model.neurons[0].membrane_potential, 0.0);
    assert!(!model.neurons[0].last_spike);
    assert_eq!(model.thresholds().len(), NEURON_COUNT);
    assert_eq!(model.decay_rates().len(), NEURON_COUNT);
    assert!(MERGED_V2_PROVENANCE.contains("shipped merged_v2 artifact"));
    assert!(MERGED_V2_PROVENANCE.contains("not a post-exp-009 legal-encoder retrain"));
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

    // The shipped decays run 0.796875..=0.94921875, so `|ln(decay)|` is at
    // most 0.227 and a subnormal timestep still divides to a nonzero
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
            r#"{"neurons": [{"decay_rate": 0.5, "inhibitory": true, "membrane_potential": 0.0, "threshold": 1.0, "weights": [1.0], "last_spike": false}]}"#,
            "neuron 0 carries unknown member(s) `inhibitory`",
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
/// The retrain tracked by issues #2 and #3 is expected to add per-neuron
/// parameters — signed output weights, an excitatory/inhibitory flag. If a
/// revised artifact decoded by dropping them, the graph would keep claiming
/// to be the shipped model while describing strictly less of it. Loudly
/// refusing an unknown member is what makes that revision a visible schema
/// change rather than a silent truncation.
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

    // `0.808594` in the JSON; `parameters_decay.mem` line 2 is `00CF`.
    assert_eq!(model.neurons[1].decay_rate, 207.0 / 256.0);
    assert_eq!(model.neurons[1].decay_rate, 0.808_593_75);
    // `0.7539062` in the JSON; `parameters_weights.mem` line 2 is `00C1`.
    assert_eq!(model.neurons[0].weights[1], 193.0 / 256.0);
    assert_eq!(model.neurons[0].weights[1], 0.753_906_25);

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

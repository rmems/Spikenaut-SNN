// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for [`super`]: the strict JSON reader.
//!
//! Split out of `json.rs` but still a child module of `json`, so
//! `use super::*` continues to reach the private parser helpers. The split also keeps `json.rs`
//! under the 500 non-comment-line ceiling the repository's static analysis
//! enforces.

use super::*;

#[test]
fn parses_scalars() {
    assert_eq!(parse("null").unwrap(), Json::Null);
    assert_eq!(parse(" true ").unwrap(), Json::Bool(true));
    assert_eq!(parse("false").unwrap(), Json::Bool(false));
    assert_eq!(parse("-1.5e2").unwrap(), Json::Number(-150.0));
    assert_eq!(parse("0").unwrap(), Json::Number(0.0));
}

#[test]
fn parses_nested_containers() {
    let value = parse(r#"{"a": [1, {"b": false}], "c": null}"#).unwrap();
    assert_eq!(value.get("a").unwrap().as_array().unwrap().len(), 2);
    assert_eq!(
        value.get("a").unwrap().as_array().unwrap()[1]
            .get("b")
            .unwrap()
            .as_bool(),
        Some(false)
    );
    assert_eq!(value.get("c"), Some(&Json::Null));
    assert_eq!(parse("[]").unwrap(), Json::Array(vec![]));
    assert_eq!(parse("{}").unwrap().as_object().unwrap().len(), 0);
}

#[test]
fn q8_8_values_written_in_full_parse_exactly() {
    // Q8.8 fixed point is dyadic, so it is exact in binary floating point.
    let value = parse("[0.75390625, 1.125, 0.796875]").unwrap();
    let items = value.as_array().unwrap();
    assert_eq!(items[0].as_f64(), Some(193.0 / 256.0));
    assert_eq!(items[1].as_f64(), Some(1.125));
    assert_eq!(items[2].as_f64(), Some(0.796875));

    // The shipped file writes them truncated, which no parser can undo;
    // `model::quantize_q8_8` restores the grid value at decode time.
    let truncated = parse("0.7539062").unwrap().as_f64().unwrap();
    assert_ne!(truncated, 193.0 / 256.0);
    assert_eq!(crate::model::quantize_q8_8(truncated), 193.0 / 256.0);
}

#[test]
fn parses_string_escapes() {
    let value = parse(r#""a\"b\\c\/d\b\f\n\r\teA😀""#).unwrap();
    assert_eq!(value.as_str(), Some("a\"b\\c/d\u{8}\u{c}\n\r\teA\u{1F600}"));
}

#[test]
fn rejects_malformed_input() {
    for bad in [
        "",
        "tru",
        "01",
        "1.",
        "1e",
        "[1,]",
        "[1 2]",
        r#"{"a" 1}"#,
        r#"{"a": 1,}"#,
        r#"{"a": 1} extra"#,
        r#""unterminated"#,
        r#""bad \q escape""#,
        r#""\uD800""#,
        r#""\uDC00""#,
        r#""\u00G0""#,
        "\"raw\ncontrol\"",
    ] {
        assert!(parse(bad).is_err(), "expected {bad:?} to be rejected");
    }
}

/// A bad hex digit reports its own offset, not the escape's first digit.
///
/// `parse_hex4` reads four bytes before advancing `self.pos`, so an error
/// built from `self.pos` alone points at the first digit whichever one is
/// actually wrong. These four inputs differ only in which position holds
/// the `G`, so they pin the offset apart; before the fix all four reported
/// the same offset.
#[test]
fn a_bad_hex_digit_reports_its_own_offset() {
    for (src, expected) in [
        (r#""\uG000""#, 3),
        (r#""\u0G00""#, 4),
        (r#""\u00G0""#, 5),
        (r#""\u000G""#, 6),
    ] {
        let err = parse(src).unwrap_err();
        assert_eq!(
            err.offset, expected,
            "{src} should fault at offset {expected}, got {}",
            err.offset
        );
        assert_eq!(
            src.as_bytes()[err.offset],
            b'G',
            "offset must land on the G"
        );
    }
}

#[test]
fn rejects_duplicate_keys() {
    let err = parse(r#"{"a": 1, "a": 2}"#).unwrap_err();
    assert!(err.message.contains("duplicate object key"));
}

#[test]
fn rejects_excessive_nesting() {
    let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
    let err = parse(&deep).unwrap_err();
    assert!(err.message.contains("nesting"));
}

/// `MAX_DEPTH` containers parse and one more does not, whatever the innermost
/// one holds.
///
/// The limit used to be checked on every value with `depth > MAX_DEPTH`, so a
/// 65th level got through when it was empty and was refused once it held a
/// scalar: the same nesting both parsed and failed.
#[test]
fn nesting_limit_counts_containers_not_their_contents() {
    let nest =
        |levels: usize, inner: &str| format!("{}{inner}{}", "[".repeat(levels), "]".repeat(levels));
    for inner in ["", "1", "null", r#""s""#] {
        assert!(
            parse(&nest(MAX_DEPTH, inner)).is_ok(),
            "{MAX_DEPTH} levels around {inner:?} must parse"
        );
    }
    // A container inside MAX_DEPTH arrays is itself level MAX_DEPTH + 1.
    for inner in ["", "1", r#""s""#, "[]", "{}"] {
        let levels = if inner.starts_with(['[', '{']) {
            MAX_DEPTH
        } else {
            MAX_DEPTH + 1
        };
        let src = nest(levels, inner);
        let err = parse(&src).unwrap_err();
        assert!(err.message.contains("nesting"), "{inner:?}: {err}");
        // Reported at the bracket that would open the extra level.
        assert_eq!(err.offset, MAX_DEPTH, "{inner:?}: {err}");
    }

    // Objects count the same way as arrays.
    let objects =
        |levels: usize| format!("{}null{}", r#"{"k":"#.repeat(levels), "}".repeat(levels));
    assert!(parse(&objects(MAX_DEPTH)).is_ok());
    assert!(parse(&objects(MAX_DEPTH + 1)).is_err());
}

/// `\u` escapes on the success path: BMP characters, both hex cases, NUL,
/// and a surrogate pair combined into one supplementary character.
#[test]
fn decodes_unicode_escapes_and_surrogate_pairs() {
    let value = parse(r#""Aéé中\u0000😀􏿿""#).unwrap();
    assert_eq!(
        value.as_str(),
        Some("A\u{e9}\u{e9}\u{4e2d}\u{0}\u{1F600}\u{10FFFF}")
    );
}

#[test]
fn rejects_malformed_surrogates() {
    for bad in [
        // High surrogate, then something other than an escape.
        r#""\uD800x""#,
        // High surrogate, then a non-surrogate escape.
        r#""\uD800A""#,
        // Two high surrogates.
        r#""\uD800\uD800""#,
        // High surrogate, then a non-`\u` escape.
        r#""\uD800\n""#,
        // Low surrogate first.
        r#""\uDC00\uD800""#,
        // Truncated second half.
        r#""\uD800\uDC0""#,
        // Truncated first escape.
        r#""\u12""#,
        r#""\u12"#,
    ] {
        assert!(parse(bad).is_err(), "expected {bad:?} to be rejected");
    }
}

/// Non-ASCII text outside escapes passes through byte for byte.
#[test]
fn literal_utf8_and_del_pass_through() {
    let value = parse("\"caf\u{e9} \u{4e2d}\u{6587} \u{1F600}\u{7f}\"").unwrap();
    assert_eq!(
        value.as_str(),
        Some("caf\u{e9} \u{4e2d}\u{6587} \u{1F600}\u{7f}")
    );
}

/// An unknown escape names the whole character and points at it.
#[test]
fn an_unknown_escape_reports_the_whole_character() {
    let err = parse("\"a\\\u{e9}\"").unwrap_err();
    assert_eq!(err.offset, 3, "{err}");
    assert!(err.message.contains("`\\\u{e9}`"), "{err}");

    let err = parse(r#""\q""#).unwrap_err();
    assert_eq!(err.offset, 2, "{err}");
    assert!(err.message.contains(r"`\q`"), "{err}");
}

#[test]
fn number_edge_cases() {
    for (src, want) in [
        ("-0", -0.0),
        ("0.0", 0.0),
        ("1E+2", 100.0),
        ("1e-2", 0.01),
        ("-0.5e1", -5.0),
        // Integers wider than 53 bits round to the nearest f64.
        ("1000000000000000000000000000000", 1e30),
        // Underflow rounds to zero rather than failing, as in serde_json.
        ("1e-400", 0.0),
        ("1.7976931348623157e308", f64::MAX),
    ] {
        let got = parse(src).unwrap().as_f64().unwrap();
        assert_eq!(got, want, "{src}");
        assert_eq!(got.is_sign_negative(), want.is_sign_negative(), "{src}");
    }

    for bad in [
        "+1",
        ".5",
        "-",
        "--1",
        "-01",
        "00",
        "1.e5",
        "0x10",
        "1e+",
        "Infinity",
        "-Infinity",
        "NaN",
        "1e309",
        "-1e309",
        "1_000",
    ] {
        assert!(parse(bad).is_err(), "expected {bad:?} to be rejected");
    }
}

/// Only the four RFC 8259 whitespace bytes separate tokens, and anything
/// after the value is an error located where it starts.
#[test]
fn trailing_input_and_foreign_whitespace_are_rejected() {
    for (src, offset) in [
        ("nullx", 4),
        ("[1]]", 3),
        ("{}{}", 2),
        ("1 2", 2),
        ("true\n,", 5),
    ] {
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("trailing"), "{src:?}: {err}");
        assert_eq!(err.offset, offset, "{src:?}: {err}");
    }
    assert_eq!(parse(" \t\r\n1 \t\r\n").unwrap(), Json::Number(1.0));
    // NBSP, form feed, vertical tab, and a byte-order mark are not JSON
    // whitespace. Rejecting the BOM is RFC 8259 section 8.1's strict option.
    for bad in ["\u{a0}1", "\u{c}1", "\u{b}1", "\u{feff}1", "1\u{a0}"] {
        assert!(parse(bad).is_err(), "expected {bad:?} to be rejected");
    }
}

/// A duplicate is located at the second key, not where its value ends.
#[test]
fn a_duplicate_key_reports_the_second_key_offset() {
    let src = r#"{"a": 1, "a": [2, 3]}"#;
    let err = parse(src).unwrap_err();
    assert_eq!(err.offset, 9, "{err}");
    assert_eq!(&src[err.offset..=err.offset + 2], r#""a""#);
    assert_eq!(err.message, r#"duplicate object key "a""#);
    // Keys that differ only before escape decoding are still duplicates:
    // `a` decodes to `a`.
    let escaped = parse(r#"{"a": 1, "a": 2}"#).unwrap_err();
    assert_eq!(escaped.message, r#"duplicate object key "a""#);
}

#[test]
fn error_display_names_the_offset() {
    let err = parse("[1,]").unwrap_err();
    assert_eq!(
        err.to_string(),
        "invalid JSON at byte 3: expected a JSON value"
    );
    assert!(std::error::Error::source(&err).is_none());
}

#[test]
fn accessors_return_their_own_type() {
    let value = parse(r#"{"b": true, "n": 2.5, "s": "x", "a": [null], "o": {}}"#).unwrap();
    assert_eq!(value.type_name(), "object");
    assert_eq!(value.get("b").and_then(Json::as_bool), Some(true));
    assert_eq!(value.get("n").and_then(Json::as_f64), Some(2.5));
    assert_eq!(value.get("s").and_then(Json::as_str), Some("x"));
    assert_eq!(
        value.get("a").and_then(Json::as_array),
        Some([Json::Null].as_slice())
    );
    assert!(value.get("o").and_then(Json::as_object).unwrap().is_empty());
    assert!(value.get("missing").is_none());
    for (key, name) in [
        ("b", "boolean"),
        ("n", "number"),
        ("s", "string"),
        ("a", "array"),
        ("o", "object"),
    ] {
        assert_eq!(value.get(key).unwrap().type_name(), name);
    }
    // `get` on a non-object is `None`, not a panic.
    assert!(value.get("a").unwrap().get("0").is_none());
}

#[test]
fn accessors_return_none_for_other_types() {
    let value = Json::Null;
    assert!(value.as_object().is_none());
    assert!(value.as_array().is_none());
    assert!(value.as_f64().is_none());
    assert!(value.as_bool().is_none());
    assert!(value.as_str().is_none());
    assert!(value.get("missing").is_none());
    assert_eq!(value.type_name(), "null");
}

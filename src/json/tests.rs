// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for [`super`]: the strict JSON reader.
//!
//! Split out of `json.rs` rather than reorganized: every test is unchanged
//! and still a child module of `json`, so `use super::*` continues to reach
//! the private parser helpers it exercises. The split also keeps `json.rs`
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

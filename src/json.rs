// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal RFC 8259 JSON reader.
//!
//! This repository deliberately depends on exactly one crate (`nir-rs`), so the
//! model loader parses `snn_model.json` itself rather than pulling in a serde
//! stack for one small, fixed-shape file.
//!
//! The grammar is complete (objects, arrays, strings with `\u` escapes and
//! surrogate pairs, numbers, `true` / `false` / `null`) and strict: leading
//! zeros, unescaped control characters, lone surrogates, trailing bytes and
//! nesting deeper than [`MAX_DEPTH`] are all rejected. Numbers decode to `f64`.
//!
//! `f64` represents every Q8.8 value exactly, but `snn_model.json` prints those
//! values as truncated decimals, so the parser alone does not recover them.
//! [`crate::model::quantize_q8_8`] snaps them back onto the grid at decode
//! time.

use std::collections::BTreeMap;
use std::fmt;

/// Maximum object/array nesting accepted before parsing fails.
///
/// The parser is recursive, so this bounds stack usage on hostile input.
pub const MAX_DEPTH: usize = 64;

/// A decoded JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// Any JSON number, decoded as `f64`.
    Number(f64),
    /// A UTF-8 string with escapes resolved.
    String(String),
    /// An ordered list of values.
    Array(Vec<Json>),
    /// An object, keyed by member name.
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// The object members, or `None` if this is not an object.
    #[must_use]
    pub fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }

    /// The array elements, or `None` if this is not an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The numeric value, or `None` if this is not a number.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// The boolean value, or `None` if this is not a boolean.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// The string value, or `None` if this is not a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// The member named `key`, or `None` if this is not an object or has no
    /// such member.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?.get(key)
    }

    /// The JSON type name, for error messages.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "boolean",
            Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }
}

/// A syntax error, located by byte offset into the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    /// What the parser expected or rejected.
    pub message: String,
    /// Byte offset into the input where the problem was found.
    pub offset: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for JsonError {}

/// Parse a complete JSON document.
///
/// # Errors
///
/// Returns [`JsonError`] if `input` is not a single well-formed JSON value,
/// or if it nests deeper than [`MAX_DEPTH`].
pub fn parse(input: &str) -> Result<Json, JsonError> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    parser.skip_whitespace();
    let value = parser.parse_value(0)?;
    parser.skip_whitespace();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error("trailing characters after the JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: impl Into<String>) -> JsonError {
        JsonError {
            message: message.into(),
            offset: self.pos,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(format!("expected {:?}", byte as char)))
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error(format!("expected `{word}`")))
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth > MAX_DEPTH {
            return Err(self.error(format!("nesting deeper than {MAX_DEPTH} levels")));
        }
        match self.peek() {
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.parse_string().map(Json::String),
            Some(b'[') => self.parse_array(depth),
            Some(b'{') => self.parse_object(depth),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error("expected a JSON value")),
            None => Err(self.error("unexpected end of input")),
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.parse_value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.error("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_whitespace();
            let key_offset = self.pos;
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.parse_value(depth + 1)?;
            if map.insert(key.clone(), value).is_some() {
                return Err(JsonError {
                    message: format!("duplicate object key {key:?}"),
                    offset: key_offset,
                });
            }
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Object(map));
                }
                _ => return Err(self.error("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            // JSON forbids leading zeros: `0` stands alone.
            Some(b'0') => self.pos += 1,
            Some(c) if c.is_ascii_digit() => self.skip_digits(),
            _ => return Err(self.error("expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.error("expected a digit after the decimal point"));
            }
            self.skip_digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.error("expected a digit in the exponent"));
            }
            self.skip_digits();
        }

        // The span was matched byte-by-byte against the ASCII number grammar.
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("number span is ASCII by construction");
        let value: f64 = text
            .parse()
            .map_err(|_| self.error(format!("number {text} is not representable as f64")))?;
        if !value.is_finite() {
            return Err(self.error(format!("number {text} overflows f64")));
        }
        Ok(Json::Number(value))
    }

    fn skip_digits(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    self.parse_escape(&mut out)?;
                }
                Some(c) if c < 0x20 => {
                    return Err(self.error("unescaped control character in string"));
                }
                Some(_) => {
                    // The input is a `&str`, so the bytes from here to the next
                    // ASCII delimiter form whole UTF-8 sequences.
                    let start = self.pos;
                    while self
                        .peek()
                        .is_some_and(|c| c != b'"' && c != b'\\' && c >= 0x20)
                    {
                        self.pos += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..self.pos])
                            .expect("slice of a &str at char boundaries"),
                    );
                }
            }
        }
    }

    fn parse_escape(&mut self, out: &mut String) -> Result<(), JsonError> {
        let escape = self
            .peek()
            .ok_or_else(|| self.error("unterminated escape"))?;
        self.pos += 1;
        let unescaped = match escape {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => return self.parse_unicode_escape(out),
            other => {
                return Err(self.error(format!("unknown escape `\\{}`", other as char)));
            }
        };
        out.push(unescaped);
        Ok(())
    }

    fn parse_unicode_escape(&mut self, out: &mut String) -> Result<(), JsonError> {
        let first = self.parse_hex4()?;
        let code_point = match first {
            // High surrogate: must be followed by `\uDC00`..=`\uDFFF`.
            0xD800..=0xDBFF => {
                if !self.bytes[self.pos..].starts_with(br"\u") {
                    return Err(self.error("high surrogate is not followed by `\\u`"));
                }
                self.pos += 2;
                let second = self.parse_hex4()?;
                if !(0xDC00..=0xDFFF).contains(&second) {
                    return Err(self.error("high surrogate is not followed by a low surrogate"));
                }
                0x1_0000 + ((u32::from(first) - 0xD800) << 10) + (u32::from(second) - 0xDC00)
            }
            0xDC00..=0xDFFF => return Err(self.error("unpaired low surrogate")),
            other => u32::from(other),
        };
        let ch = char::from_u32(code_point)
            .ok_or_else(|| self.error(format!("invalid code point U+{code_point:04X}")))?;
        out.push(ch);
        Ok(())
    }

    fn parse_hex4(&mut self) -> Result<u16, JsonError> {
        let end = self.pos + 4;
        let digits = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| self.error("truncated `\\u` escape"))?;
        let mut value: u16 = 0;
        for &digit in digits {
            let nibble = (digit as char)
                .to_digit(16)
                .ok_or_else(|| self.error("`\\u` escape needs four hex digits"))?;
            // Four hex digits always fit in u16.
            value = value * 16 + nibble as u16;
        }
        self.pos = end;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
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
}

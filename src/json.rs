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
    /// A `JsonError` carrying the current byte offset, so a failure deep in
    /// a nested document can be located in the source text.
    fn error(&self, message: impl Into<String>) -> JsonError {
        JsonError {
            message: message.into(),
            offset: self.pos,
        }
    }

    /// The byte at the cursor, or `None` at end of input.
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Advance past the four bytes RFC 8259 counts as whitespace.
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Consume `byte`, or fail naming what was expected.
    fn expect(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(format!("expected {:?}", byte as char)))
        }
    }

    /// Consume one of the three bare words (`null`, `true`, `false`).
    fn literal(&mut self, word: &str, value: Json) -> Result<Json, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error(format!("expected `{word}`")))
        }
    }

    /// Dispatch on the first byte to the parser for that value type.
    ///
    /// `depth` guards against a document nested deeply enough to overflow
    /// the stack, since this parser recurses.
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

    /// `[` value `,` value `]`, rejecting a trailing comma.
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

    /// `{` member `,` member `}`, rejecting a trailing comma.
    fn parse_object(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.parse_member(&mut map, depth)?;
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

    /// Read one `"key": value` pair into `map`.
    ///
    /// The offset of the *key* is captured before parsing so a duplicate is
    /// reported where the second key starts, not where its value ends.
    fn parse_member(
        &mut self,
        map: &mut BTreeMap<String, Json>,
        depth: usize,
    ) -> Result<(), JsonError> {
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
        Ok(())
    }

    /// Sign, integer, optional fraction, optional exponent, then `f64`.
    ///
    /// The span is matched byte-by-byte against the grammar before it is
    /// handed to `str::parse`, so a JSON-invalid literal Rust happens to
    /// accept -- `1.`, `.5`, `0x10`, `inf` -- is refused here.
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
        self.scan_fraction()?;
        self.scan_exponent()?;

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

    /// The optional fraction: `.` then at least one digit. A bare `1.` is a
    /// syntax error in JSON, unlike in Rust.
    fn scan_fraction(&mut self) -> Result<(), JsonError> {
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.error("expected a digit after the decimal point"));
            }
            self.skip_digits();
        }
        Ok(())
    }

    /// The optional exponent: `e` or `E`, an optional sign, then at least one
    /// digit. A bare `1e` is a syntax error.
    fn scan_exponent(&mut self) -> Result<(), JsonError> {
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
        Ok(())
    }

    /// Advance over a run of ASCII digits.
    fn skip_digits(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
    }

    /// A quoted string, with escapes decoded and control bytes refused.
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
                Some(_) => self.take_literal_run(&mut out),
            }
        }
    }

    /// Consume the run of literal characters up to the next `"`, `\\` or
    /// control byte, and append it whole.
    ///
    /// The input is a `&str`, so the bytes from here to the next ASCII
    /// delimiter form whole UTF-8 sequences -- the slice can never split a
    /// multi-byte character.
    fn take_literal_run(&mut self, out: &mut String) {
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

    /// One escape sequence, the cursor already past the backslash.
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

    /// Exactly four hex digits, as in a `\\uXXXX` escape.
    fn parse_hex4(&mut self) -> Result<u16, JsonError> {
        let end = self.pos + 4;
        let digits = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| self.error("truncated `\\u` escape"))?;
        let mut value: u16 = 0;
        // Enumerated because `self.pos` stays on the first digit for the whole
        // loop -- it only advances once all four are read. Reporting `self.pos`
        // would point every bad digit at the first one, so `\u0G00` and `\uG000`
        // would be indistinguishable in the error.
        for (index, &digit) in digits.iter().enumerate() {
            let nibble = (digit as char).to_digit(16).ok_or_else(|| JsonError {
                message: "`\\u` escape needs four hex digits".into(),
                offset: self.pos + index,
            })?;
            // Four hex digits always fit in u16.
            value = value * 16 + nibble as u16;
        }
        self.pos = end;
        Ok(value)
    }
}

#[cfg(test)]
mod tests;

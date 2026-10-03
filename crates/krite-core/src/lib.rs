//! Krite Protocol v1 (`docs/protocol/v1.md`): request and response types, the checks the JSON
//! Schemas cannot express, error bodies, and the canonical state string.

use std::collections::BTreeMap;
use std::fmt;

use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Choice {
        instructions: String,
        criteria: BTreeMap<String, Option<String>>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    Noul {
        instructions: String,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "true", default)]
    pub yes: Option<String>,
    #[serde(rename = "false", default)]
    pub no: Option<String>,
}

impl Question {
    pub fn kind(&self) -> &'static str {
        match self {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Question::Choice { instructions, .. }
            | Question::Score { instructions, .. }
            | Question::Noul { instructions, .. } => instructions,
        }
    }

    /// Candidates with optional descriptions: choice names (sorted), score levels in order,
    /// noul `true` then `false`.
    pub fn candidates(&self) -> Vec<(String, Option<String>)> {
        match self {
            Question::Choice { criteria, .. } => criteria.iter().map(|(n, d)| (n.clone(), d.clone())).collect(),
            Question::Score { criteria, .. } => criteria.iter().map(|n| (n.clone(), None)).collect(),
            Question::Noul { criteria, .. } => {
                let c = criteria.as_ref();
                vec![("true".into(), c.and_then(|c| c.yes.clone())), ("false".into(), c.and_then(|c| c.no.clone()))]
            }
        }
    }
}

/// Server defaults (`docs/protocol/v1.md` §4). Limits are configuration, not protocol.
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_questions: usize,
    pub max_options: usize,
    /// Encoder input tokens, `<bos>`/`<eos>` included.
    pub max_state_tokens: usize,
    pub max_body_bytes: usize,
    /// Tokens of all candidate texts (instructions + criterion) in one request; bounds scoring memory.
    pub max_candidate_tokens: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_questions: 64,
            max_options: 255,
            max_state_tokens: 8192,
            max_body_bytes: 4 * 1024 * 1024,
            max_candidate_tokens: 65_536,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    InvalidRequest,
    StateTooLong,
    TooManyQuestions,
    TooManyOptions,
    UnknownModel,
    InternalError,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub kind: ErrorType,
    pub message: String,
    pub param: Option<String>,
}

impl ApiError {
    pub fn new(kind: ErrorType, message: impl Into<String>, param: Option<String>) -> Self {
        ApiError { kind, message: message.into(), param }
    }

    pub fn invalid(message: impl Into<String>, param: Option<String>) -> Self {
        Self::new(ErrorType::InvalidRequest, message, param)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorType::InternalError, message, None)
    }

    pub fn status(&self) -> u16 {
        if self.kind == ErrorType::InternalError { 500 } else { 422 }
    }

    pub fn body(&self) -> Value {
        serde_json::json!({"error": {"type": self.kind, "message": self.message, "param": self.param}})
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ApiError {}

/// Parses a request body. serde errors carry no path, so `param` is null.
pub fn parse(body: &[u8]) -> Result<Request, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::invalid(format!("invalid request: {e}"), None))
}

fn valid_id(id: &str) -> bool {
    (1..=64).contains(&id.len()) && id.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}

/// Checks the JSON Schemas cannot express, plus the server limits (`v1.md` §4).
pub fn validate(req: &Request, limits: &Limits) -> Result<(), ApiError> {
    if req.model.as_deref() == Some("") {
        return Err(ApiError::invalid("`model` must not be empty", Some("model".into())));
    }
    if req.questions.is_empty() {
        return Err(ApiError::invalid("`questions` is empty", Some("questions".into())));
    }
    if req.questions.len() > limits.max_questions {
        return Err(ApiError::new(
            ErrorType::TooManyQuestions,
            format!("request has {} questions; max is {}", req.questions.len(), limits.max_questions),
            Some("questions".into()),
        ));
    }
    for (id, q) in &req.questions {
        if !valid_id(id) {
            return Err(ApiError::invalid(
                format!("question id {id:?} must match ^[A-Za-z0-9_.-]{{1,64}}$"),
                Some(format!("questions.{id}")),
            ));
        }
        if q.instructions().is_empty() {
            return Err(ApiError::invalid("`instructions` is empty", Some(format!("questions.{id}.instructions"))));
        }
        let param = Some(format!("questions.{id}.criteria"));
        let k = match q {
            Question::Choice { criteria, .. } => {
                if criteria.is_empty() || criteria.keys().any(String::is_empty) {
                    return Err(ApiError::invalid("choice criteria need at least one non-empty name", param));
                }
                criteria.len()
            }
            Question::Score { criteria, .. } => {
                let mut seen = std::collections::BTreeSet::new();
                if criteria.is_empty() || criteria.iter().any(|l| l.is_empty() || !seen.insert(l)) {
                    return Err(ApiError::invalid("score criteria need unique non-empty levels", param));
                }
                criteria.len()
            }
            Question::Noul { .. } => 2,
        };
        if k > limits.max_options {
            return Err(ApiError::new(
                ErrorType::TooManyOptions,
                format!("question '{id}' has {k} options; max is {}", limits.max_options),
                param,
            ));
        }
    }
    Ok(())
}

/// String states pass through; object states become their RFC 8785 (JCS) canonical string.
pub fn canonical_state(state: &Value) -> Result<String, ApiError> {
    match state {
        Value::String(s) => Ok(s.clone()),
        Value::Object(_) => {
            let mut out = String::new();
            jcs(state, &mut out);
            Ok(out)
        }
        _ => Err(ApiError::invalid("`state` must be a string or an object", Some("state".into()))),
    }
}

/// RFC 8785: members sorted by UTF-16 code units, ECMAScript number formatting, JSON.stringify string escapes.
fn jcs(v: &Value, out: &mut String) {
    match v {
        Value::Null | Value::Bool(_) => out.push_str(&v.to_string()),
        // serde_json escapes exactly what JSON.stringify escapes (", \, \b \f \n \r \t, other controls as \u00xx).
        Value::String(s) => out.push_str(&Value::String(s.clone()).to_string()),
        Value::Number(n) => out.push_str(&es_number(n.as_f64().expect("JSON numbers convert to f64"))),
        Value::Array(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                jcs(x, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push(':');
                jcs(&map[k], out);
            }
            out.push('}');
        }
    }
}

/// ECMAScript Number::toString for a finite double (RFC 8785 §3.2.2.3).
fn es_number(x: f64) -> String {
    if x == 0.0 {
        return "0".into(); // also -0
    }
    // Rust's `{:e}` gives the shortest round-trip digits: "d.ddde±x".
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp.parse::<i32>().expect("integer exponent") + 1; // x = 0.digits × 10^n
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let e = n - 1;
        let sign = if e >= 0 { "+" } else { "-" };
        let frac = if k == 1 { String::new() } else { format!(".{}", &digits[1..]) };
        format!("{}{frac}e{sign}{}", &digits[..1], e.abs())
    };
    if x < 0.0 { format!("-{body}") } else { body }
}

/// A JSON object serialized in the given order (score levels, legend "0".."K-1").
#[derive(Debug, Clone, PartialEq)]
pub struct Ordered<V>(pub Vec<(String, V)>);

impl<V: Serialize> Serialize for Ordered<V> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice { choice: String, probabilities: Ordered<f64>, confidence: f64 },
    Score { score: f64, legend: Ordered<String>, probabilities: Ordered<f64>, confidence: f64 },
    Noul { noul: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
    pub latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn example(name: &str) -> Vec<u8> {
        let p = format!("{}/../../docs/protocol/examples/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
    }

    fn req(v: Value) -> Result<Request, ApiError> {
        parse(v.to_string().as_bytes())
    }

    fn check(v: Value) -> Result<(), ApiError> {
        validate(&req(v)?, &Limits::default())
    }

    fn choice(k: usize) -> Value {
        let c: serde_json::Map<String, Value> = (0..k).map(|i| (format!("o{i}"), Value::Null)).collect();
        json!({"type": "choice", "instructions": "i", "criteria": c})
    }

    #[test]
    fn example_requests_are_valid() {
        for name in ["choice.request.json", "multi.request.json"] {
            validate(&parse(&example(name)).unwrap(), &Limits::default()).unwrap();
        }
    }

    #[test]
    fn invalid_examples_are_rejected() {
        let dir = format!("{}/../../docs/protocol/examples/invalid", env!("CARGO_MANIFEST_DIR"));
        let mut n = 0;
        for f in std::fs::read_dir(dir).unwrap() {
            let body = std::fs::read(f.unwrap().path()).unwrap();
            assert!(parse(&body).and_then(|r| validate(&r, &Limits::default())).is_err());
            n += 1;
        }
        assert!(n >= 3);
    }

    #[test]
    fn question_ids() {
        let ok = |id: &str| check(json!({"state": "s", "questions": {id: choice(2)}})).is_ok();
        assert!(ok("ok_id.-1"));
        assert!(!ok("a b"));
        assert!(!ok(&"x".repeat(65)));
    }

    #[test]
    fn limits() {
        let qs: serde_json::Map<String, Value> = (0..65).map(|i| (format!("q{i}"), choice(2))).collect();
        let e = check(json!({"state": "s", "questions": qs})).unwrap_err();
        assert_eq!((e.kind, e.param.as_deref()), (ErrorType::TooManyQuestions, Some("questions")));
        let e = check(json!({"state": "s", "questions": {"q": choice(256)}})).unwrap_err();
        assert_eq!((e.kind, e.param.as_deref()), (ErrorType::TooManyOptions, Some("questions.q.criteria")));
        assert_eq!(e.message, "question 'q' has 256 options; max is 255");
        assert!(check(json!({"state": "s", "questions": {"q": choice(255)}})).is_ok());
    }

    #[test]
    fn score_levels_unique_non_empty() {
        for c in [json!(["a", "a"]), json!([""]), json!([])] {
            let q = json!({"type": "score", "instructions": "i", "criteria": c});
            assert_eq!(
                check(json!({"state": "s", "questions": {"q": q}})).unwrap_err().kind,
                ErrorType::InvalidRequest
            );
        }
    }

    #[test]
    fn noul_criteria() {
        let noul = |c: Option<Value>| {
            let mut q = json!({"type": "noul", "instructions": "i"});
            if let Some(c) = c {
                q["criteria"] = c;
            }
            check(json!({"state": "s", "questions": {"q": q}}))
        };
        assert!(noul(None).is_ok());
        assert!(noul(Some(json!({}))).is_ok());
        assert!(noul(Some(json!({"true": "x"}))).is_ok());
        assert!(noul(Some(json!({"maybe": 1}))).is_err());
    }

    #[test]
    fn empty_fields() {
        assert!(check(json!({"state": "s", "questions": {}})).is_err());
        assert!(check(json!({"state": "s", "model": "", "questions": {"q": choice(1)}})).is_err());
        let q = json!({"type": "choice", "instructions": "", "criteria": {"a": null}});
        let e = check(json!({"state": "s", "questions": {"q": q}})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("questions.q.instructions"));
        assert!(check(json!({"state": "s", "questions": {"q": choice(0)}})).is_err());
    }

    #[test]
    fn jcs_rfc8785_vectors() {
        // RFC 8785 §3.2.2 sample input and output.
        let input = r#"{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001],
            "string":"\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/","literals":[null,true,false]}"#;
        let want = r#"{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}"#;
        assert_eq!(canonical_state(&serde_json::from_str(input).unwrap()).unwrap(), want);
        // §3.2.3: properties sort by UTF-16 code units (U+1F600 sorts before U+FB33).
        let keys = r#"{"\u20ac":0,"\r":0,"\ufb33":0,"1":0,"\ud83d\ude00":0,"\u0080":0,"\u00f6":0}"#;
        let got = canonical_state(&serde_json::from_str(keys).unwrap()).unwrap();
        assert_eq!(got, "{\"\\r\":0,\"1\":0,\"\u{80}\":0,\"ö\":0,\"€\":0,\"😀\":0,\"\u{fb33}\":0}");
        // Appendix B number vectors (IEEE 754 bits → ES Number::toString).
        for (bits, want) in [
            (0x0000000000000000u64, "0"),
            (0x8000000000000000, "0"),
            (0x0000000000000001, "5e-324"),
            (0x8000000000000001, "-5e-324"),
            (0x7fefffffffffffff, "1.7976931348623157e+308"),
            (0x4340000000000000, "9007199254740992"),
            (0x4430000000000000, "295147905179352830000"),
            (0x44b52d02c7e14af5, "9.999999999999997e+22"),
            (0x44b52d02c7e14af6, "1e+23"),
            (0x3eb0c6f7a0b5ed8d, "0.000001"),
            (0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"),
            (0x444b1ae4d6e2ef50, "1e+21"),
            (0x3e7ad7f29abcaf48, "1e-7"),
        ] {
            assert_eq!(es_number(f64::from_bits(bits)), want, "{bits:#x}");
        }
    }

    #[test]
    fn equivalent_numbers_canonicalize_alike() {
        let a: Value = serde_json::from_str(r#"{"x":1,"y":[1e2,0.5]}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"y":[100.0,5E-1],"x":1.0}"#).unwrap();
        assert_eq!(canonical_state(&a).unwrap(), r#"{"x":1,"y":[100,0.5]}"#);
        assert_eq!(canonical_state(&a).unwrap(), canonical_state(&b).unwrap());
    }

    #[test]
    fn canonical_state_ignores_key_order() {
        let a: Value = serde_json::from_str(r#"{"b":1,"a":{"d":2,"c":3}}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a":{"c":3,"d":2},"b":1}"#).unwrap();
        assert_eq!(canonical_state(&a).unwrap(), r#"{"a":{"c":3,"d":2},"b":1}"#);
        assert_eq!(canonical_state(&a).unwrap(), canonical_state(&b).unwrap());
        assert_eq!(canonical_state(&json!("é")).unwrap(), "é");
        for bad in [json!([1, 2]), json!(3)] {
            assert_eq!(canonical_state(&bad).unwrap_err().param.as_deref(), Some("state"));
        }
    }

    #[test]
    fn ordered_serialization_keeps_level_order() {
        let legend: Vec<(String, String)> = (0..11).map(|i| (i.to_string(), format!("l{i}"))).collect();
        let a = Answer::Score {
            score: 0.0,
            legend: Ordered(legend),
            probabilities: Ordered(vec![("z".into(), 0.5), ("a".into(), 0.5)]),
            confidence: 0.0,
        };
        let s = serde_json::to_string(&a).unwrap();
        assert!(s.find("\"2\"").unwrap() < s.find("\"10\"").unwrap());
        assert!(s.find("\"z\"").unwrap() < s.find("\"a\"").unwrap());
        assert!(s.starts_with(r#"{"type":"score""#));
    }

    #[test]
    fn error_body_shape() {
        let e = ApiError::new(ErrorType::TooManyOptions, "m", Some("questions.route.criteria".into()));
        let want: Value = serde_json::from_slice(&example("too-many-options.error.json")).unwrap();
        let got = e.body();
        assert_eq!(got["error"]["type"], want["error"]["type"]);
        assert_eq!(got["error"]["param"], want["error"]["param"]);
        assert_eq!(e.status(), 422);
        assert_eq!(ApiError::internal("x").status(), 500);
        assert_eq!(ApiError::invalid("x", None).body()["error"]["param"], Value::Null);
    }
}

//! Request pipeline (ARCHITECTURE.md §1): canonical state → tokenize → state cache → encode on a miss
//! → one energy per candidate → softmax → calibration → Choice / Score / Noul answers.

pub mod cache;
pub mod calibrate;

use std::collections::BTreeMap;
use std::time::Instant;

use krite_core::{Answer, ApiError, ErrorType, Limits, Ordered, Request, Response, Usage};
use sha2::{Digest, Sha256};

pub use cache::StateCache;
pub use calibrate::{BUCKETS, Calibrator, bucket};

/// One candidate as the backend sees it: its question type and its ids from `Backend::candidate_ids`.
/// No question id: ids are not model inputs (I4).
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub kind: &'static str,
    pub ids: Vec<u32>,
}

/// A model backend. The state encoder never sees questions, so its output can be cached across requests.
pub trait Backend: Send + 'static {
    type State: Clone + Send;

    fn model_id(&self) -> &str;
    /// Spec §1 backend value, e.g. `candle-metal`.
    fn backend_name(&self) -> &str;
    /// Part of the state cache key, so a tokenizer change invalidates cached states.
    fn tokenizer_version(&self) -> &str;
    fn tokenize(&self, text: &str, special_tokens: bool) -> anyhow::Result<Vec<u32>>;
    fn encode(&mut self, state_ids: &[u32]) -> anyhow::Result<Self::State>;
    fn state_bytes(state: &Self::State) -> usize;
    /// Model input for each criterion `(name, desc)` of one question (`desc` is None when absent or empty).
    /// One call per question, so shared instructions are processed once and the work stays linear in
    /// the request body.
    fn candidate_ids(&self, instructions: &str, criteria: &[(String, Option<String>)])
    -> anyhow::Result<Vec<Vec<u32>>>;
    /// One energy per candidate; each depends only on the state and that candidate alone.
    fn energies(&mut self, state: &Self::State, candidates: &[Candidate]) -> anyhow::Result<Vec<f32>>;
}

/// Per-stage milliseconds for the `x-krite-timing` header.
#[derive(Debug, Clone, Default)]
pub struct Timing {
    pub tokenize_ms: f64,
    pub encode_ms: f64,
    pub decide_ms: f64,
    pub calibrate_ms: f64,
}

impl Timing {
    pub fn header(&self) -> String {
        format!(
            "tokenize={:.2};encode={:.2};decide={:.2};calibrate={:.2}",
            self.tokenize_ms, self.encode_ms, self.decide_ms, self.calibrate_ms
        )
    }
}

#[derive(Debug)]
pub struct Decision {
    pub response: Response,
    pub cache_hit: bool,
    pub timing: Timing,
}

pub struct Runtime<B: Backend> {
    backend: B,
    cache: StateCache<B::State>,
    calibrator: Calibrator,
    limits: Limits,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn internal(e: anyhow::Error) -> ApiError {
    ApiError::internal(format!("{e:#}"))
}

/// Normalized-entropy confidence (ARCHITECTURE.md Glossary); 1.0 when K = 1.
pub fn confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let h: f64 = -p.iter().filter(|&&x| x > 0.0).map(|x| x * x.ln()).sum::<f64>();
    (1.0 - h / (p.len() as f64).ln()).clamp(0.0, 1.0)
}

pub fn softmax(e: &[f32], temperature: f64) -> Vec<f64> {
    let z: Vec<f64> = e.iter().map(|&x| x as f64 / temperature).collect();
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = z.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

fn answer(kind: &str, names: Vec<String>, p: Vec<f64>) -> Answer {
    match kind {
        "noul" => Answer::Noul { noul: p[0] },
        "choice" => {
            // Ties go to the smallest name in codepoint order (String order is UTF-8 byte order).
            let top = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let choice = names.iter().zip(&p).filter(|(_, x)| **x == top).map(|(n, _)| n).min().unwrap().clone();
            let confidence = confidence(&p);
            Answer::Choice { choice, probabilities: Ordered(names.into_iter().zip(p).collect()), confidence }
        }
        _ => Answer::Score {
            score: p.iter().enumerate().map(|(i, x)| i as f64 * x).sum(),
            legend: Ordered(names.iter().enumerate().map(|(i, n)| (i.to_string(), n.clone())).collect()),
            confidence: confidence(&p),
            probabilities: Ordered(names.into_iter().zip(p).collect()),
        },
    }
}

impl<B: Backend> Runtime<B> {
    pub fn new(backend: B, cache_bytes: usize) -> Self {
        Runtime {
            backend,
            cache: StateCache::new(cache_bytes),
            calibrator: Calibrator::identity(),
            limits: Limits::default(),
        }
    }

    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_calibrator(mut self, calibrator: Calibrator) -> Self {
        self.calibrator = calibrator;
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    fn cache_key(&self, canonical: &str) -> cache::Key {
        let mut h = Sha256::new();
        h.update(self.backend.model_id());
        h.update([0]);
        h.update(self.backend.tokenizer_version());
        h.update([0]);
        h.update(canonical);
        h.finalize().into()
    }

    pub fn decide(&mut self, req: &Request) -> Result<Decision, ApiError> {
        krite_core::validate(req, &self.limits)?;
        if let Some(m) = req.model.as_deref().filter(|m| *m != self.backend.model_id()) {
            let msg = format!("unknown model {m:?}; this server runs {:?}", self.backend.model_id());
            return Err(ApiError::new(ErrorType::UnknownModel, msg, Some("model".into())));
        }
        let mut timing = Timing::default();

        let t = Instant::now();
        let canonical = krite_core::canonical_state(&req.state)?;
        let state_ids = self.backend.tokenize(&canonical, true).map_err(internal)?;
        if state_ids.len() > self.limits.max_state_tokens {
            let msg = format!("state has {} tokens; max is {}", state_ids.len(), self.limits.max_state_tokens);
            return Err(ApiError::new(ErrorType::StateTooLong, msg, Some("state".into())));
        }
        // Question ids never reach the backend (I4): only instructions, names, and descriptions do.
        // Scoring memory grows with the total candidate tokens, which the body limit alone does not bound.
        let mut spans = Vec::with_capacity(req.questions.len());
        let mut texts = Vec::new();
        let mut candidate_tokens = 0;
        for q in req.questions.values() {
            let cands: Vec<(String, Option<String>)> =
                q.candidates().into_iter().map(|(n, d)| (n, d.filter(|d| !d.is_empty()))).collect();
            spans.push((texts.len(), cands.len()));
            let ids = self.backend.candidate_ids(q.instructions(), &cands).map_err(internal)?;
            if ids.len() != cands.len() {
                return Err(ApiError::internal(format!("{} inputs for {} criteria", ids.len(), cands.len())));
            }
            for ids in ids {
                candidate_tokens += ids.len();
                if candidate_tokens > self.limits.max_candidate_tokens {
                    let msg = format!(
                        "instructions and criteria exceed {} tokens in total; split the request",
                        self.limits.max_candidate_tokens
                    );
                    return Err(ApiError::invalid(msg, Some("questions".into())));
                }
                texts.push(Candidate { kind: q.kind(), ids });
            }
        }
        timing.tokenize_ms = ms(t);

        let key = self.cache_key(&canonical);
        let (state, cache_hit) = match self.cache.get(&key) {
            Some(s) => (s, true),
            None => {
                let t = Instant::now();
                let s = self.backend.encode(&state_ids).map_err(internal)?;
                timing.encode_ms = ms(t);
                self.cache.insert(key, s.clone(), B::state_bytes(&s));
                (s, false)
            }
        };

        let t = Instant::now();
        let energies = self.backend.energies(&state, &texts).map_err(internal)?;
        if energies.len() != texts.len() {
            return Err(ApiError::internal(format!("{} energies for {} candidates", energies.len(), texts.len())));
        }
        timing.decide_ms = ms(t);

        let t = Instant::now();
        let mut answers = BTreeMap::new();
        for ((id, q), (start, k)) in req.questions.iter().zip(spans) {
            let names = q.candidates().into_iter().map(|(n, _)| n).collect();
            let e = &energies[start..start + k];
            let p = if k == 1 { vec![1.0] } else { softmax(e, self.calibrator.temperature(bucket(q.kind(), k))) };
            answers.insert(id.clone(), answer(q.kind(), names, p));
        }
        timing.calibrate_ms = ms(t);

        let input_tokens = (state_ids.len() + texts.iter().map(|c| c.ids.len()).sum::<usize>()) as u64;
        let response = Response {
            model: self.backend.model_id().to_string(),
            answers,
            usage: Usage { input_tokens, output_tokens: 0 },
            latency_ms: 0.0,
        };
        Ok(Decision { response, cache_hit, timing })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krite_core::parse;
    use serde_json::{Value, json};

    /// Deterministic, weight-free backend: each energy hashes (state ids, candidate ids).
    struct HashBackend {
        encodes: usize,
        /// Calls that turn a question's instructions into model input.
        instruction_calls: std::cell::Cell<usize>,
    }

    impl Backend for HashBackend {
        type State = Vec<u32>;
        fn model_id(&self) -> &str {
            "hash"
        }
        fn backend_name(&self) -> &str {
            "none"
        }
        fn tokenizer_version(&self) -> &str {
            "bytes"
        }
        fn tokenize(&self, text: &str, special: bool) -> anyhow::Result<Vec<u32>> {
            let ids = text.bytes().map(u32::from);
            Ok(if special { std::iter::once(2).chain(ids).chain([1]).collect() } else { ids.collect() })
        }
        fn encode(&mut self, ids: &[u32]) -> anyhow::Result<Vec<u32>> {
            self.encodes += 1;
            Ok(ids.to_vec())
        }
        fn state_bytes(s: &Vec<u32>) -> usize {
            s.len() * 4
        }
        fn candidate_ids(
            &self,
            instructions: &str,
            criteria: &[(String, Option<String>)],
        ) -> anyhow::Result<Vec<Vec<u32>>> {
            self.instruction_calls.set(self.instruction_calls.get() + 1);
            criteria
                .iter()
                .map(|(name, desc)| match desc {
                    Some(d) => self.tokenize(&format!("{instructions}\n{name}: {d}"), false),
                    None => self.tokenize(&format!("{instructions}\n{name}"), false),
                })
                .collect()
        }
        fn energies(&mut self, s: &Vec<u32>, cands: &[Candidate]) -> anyhow::Result<Vec<f32>> {
            Ok(cands
                .iter()
                .map(|c| {
                    let h = Sha256::new().chain_update(format!("{s:?}|{}|{:?}", c.kind, c.ids)).finalize();
                    u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as f32 / u32::MAX as f32 * 4.0
                })
                .collect())
        }
    }

    fn rt() -> Runtime<HashBackend> {
        Runtime::new(HashBackend { encodes: 0, instruction_calls: Default::default() }, 1 << 20)
    }

    fn decide(rt: &mut Runtime<HashBackend>, v: Value) -> Result<Decision, ApiError> {
        rt.decide(&parse(v.to_string().as_bytes())?)
    }

    fn body(d: &Decision) -> Value {
        serde_json::to_value(&d.response).unwrap()
    }

    fn route() -> Value {
        json!({"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": null, "other": null}})
    }

    #[test]
    fn answer_math() {
        let p = softmax(&[0.0, 0.0], 1.0);
        assert_eq!(p, vec![0.5, 0.5]);
        let Answer::Choice { choice, confidence, .. } = answer("choice", vec!["b".into(), "a".into()], p) else {
            panic!()
        };
        assert_eq!((choice.as_str(), confidence), ("a", 0.0));
        let Answer::Score { score, legend, .. } = answer("score", vec!["l".into(), "h".into()], vec![0.25, 0.75])
        else {
            panic!()
        };
        assert_eq!(score, 0.75);
        assert_eq!(legend.0[1], ("1".into(), "h".into()));
        assert!((softmax(&[1.0, 0.0], 2.0)[0] - 1.0 / (1.0 + (-0.5f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn single_candidate_is_certain() {
        let q = json!({"type": "choice", "instructions": "i", "criteria": {"only": null}});
        let d = decide(&mut rt(), json!({"state": "s", "questions": {"q": q}})).unwrap();
        assert_eq!(body(&d)["answers"]["q"]["probabilities"]["only"], 1.0);
        assert_eq!(body(&d)["answers"]["q"]["confidence"], 1.0);
    }

    #[test]
    fn types_and_sums() {
        let req = json!({"state": {"b": 1, "a": "x"}, "questions": {
            "route": route(),
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high", "max"]},
            "refund": {"type": "noul", "instructions": "Refund?"}
        }});
        let b = body(&decide(&mut rt(), req).unwrap());
        for id in ["route", "urgency"] {
            let s: f64 =
                b["answers"][id]["probabilities"].as_object().unwrap().values().map(|v| v.as_f64().unwrap()).sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
        let u = &b["answers"]["urgency"];
        let p: Vec<f64> =
            ["low", "mid", "high", "max"].iter().map(|n| u["probabilities"][n].as_f64().unwrap()).collect();
        assert!(
            (u["score"].as_f64().unwrap() - p.iter().enumerate().map(|(i, x)| i as f64 * x).sum::<f64>()).abs() < 1e-12
        );
        assert!(b["answers"]["refund"].get("confidence").is_none());
        assert_eq!(b["usage"]["output_tokens"], 0);
    }

    #[test]
    fn question_isolation_and_id_irrelevance() {
        let mut r = rt();
        let alone = body(&decide(&mut r, json!({"state": "s", "questions": {"a": route()}})).unwrap());
        let mut qs = serde_json::Map::new();
        qs.insert("zz".into(), route());
        for i in 0..15 {
            qs.insert(format!("f{i}"), json!({"type": "noul", "instructions": format!("filler {i}?")}));
        }
        let crowd = body(&decide(&mut r, json!({"state": "s", "questions": qs})).unwrap());
        assert_eq!(alone["answers"]["a"], crowd["answers"]["zz"]);
    }

    #[test]
    fn cold_and_warm_agree_and_object_key_order_hits() {
        let mut r = rt();
        let a = decide(&mut r, json!({"state": {"x": 1, "y": 2}, "questions": {"q": route()}})).unwrap();
        let b = decide(&mut r, json!({"state": {"y": 2, "x": 1}, "questions": {"q": route()}})).unwrap();
        assert!(!a.cache_hit && b.cache_hit);
        assert_eq!(b.timing.encode_ms, 0.0);
        assert_eq!(body(&a), body(&b));
        assert_eq!(r.backend().encodes, 1);
        assert_eq!(b.timing.header().split(';').count(), 4);
    }

    #[test]
    fn request_errors() {
        let e = decide(&mut rt(), json!({"model": "other", "state": "s", "questions": {"q": route()}})).unwrap_err();
        assert_eq!((e.kind, e.param.as_deref()), (ErrorType::UnknownModel, Some("model")));
        assert!(decide(&mut rt(), json!({"model": "hash", "state": "s", "questions": {"q": route()}})).is_ok());
        let mut r = rt().with_limits(Limits { max_state_tokens: 8, ..Limits::default() });
        let e = decide(&mut r, json!({"state": "longer than eight", "questions": {"q": route()}})).unwrap_err();
        assert_eq!((e.kind, e.param.as_deref()), (ErrorType::StateTooLong, Some("state")));
    }

    #[test]
    fn candidate_token_budget_rejects_before_scoring() {
        // Byte tokenizer: each route candidate text is ~20 tokens, 3 candidates per question.
        let mut r = rt().with_limits(Limits { max_candidate_tokens: 100, ..Limits::default() });
        let qs: serde_json::Map<String, Value> = (0..2).map(|i| (format!("q{i}"), route())).collect();
        let e = decide(&mut r, json!({"state": "s", "questions": qs})).unwrap_err();
        assert_eq!((e.kind, e.param.as_deref()), (ErrorType::InvalidRequest, Some("questions")));
        assert_eq!(r.backend().encodes, 0);
        let mut r = rt().with_limits(Limits { max_candidate_tokens: 100, ..Limits::default() });
        assert!(decide(&mut r, json!({"state": "s", "questions": {"q": route()}})).is_ok());
    }

    #[test]
    fn instructions_are_processed_once_per_question() {
        // Tokenization work must stay linear in the request body: 255 criteria must not repeat
        // the (possibly long) instructions 255 times.
        let criteria: serde_json::Map<String, Value> = (0..255).map(|i| (format!("c{i}"), Value::Null)).collect();
        let q = json!({"type": "choice", "instructions": "Pick one.", "criteria": criteria});
        let mut r = rt();
        decide(&mut r, json!({"state": "s", "questions": {"a": q.clone(), "b": q}})).unwrap();
        assert_eq!(r.backend().instruction_calls.get(), 2);
    }

    #[test]
    fn temperature_is_applied_per_bucket() {
        let mut r = rt();
        let q = json!({"type": "choice", "instructions": "i", "criteria": {"a": null, "b": null}});
        let req = json!({"state": "s", "questions": {"q": q}});
        let raw = body(&decide(&mut r, req.clone()).unwrap());
        let mut hot = rt().with_calibrator(Calibrator::identity().with("choice/2", 1e6));
        let flat = body(&decide(&mut hot, req).unwrap());
        assert_ne!(raw["answers"]["q"]["probabilities"]["a"], 0.5);
        assert!((flat["answers"]["q"]["probabilities"]["a"].as_f64().unwrap() - 0.5).abs() < 1e-5);
    }
}

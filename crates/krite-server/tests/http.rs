//! HTTP-level checks of `POST /v1/systemone` against a weight-free backend.

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use krite_runtime::{Backend, Runtime};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

struct HashBackend;

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
    fn tokenize(&self, text: &str, _special: bool) -> anyhow::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }
    fn encode(&mut self, ids: &[u32]) -> anyhow::Result<Vec<u32>> {
        Ok(ids.to_vec())
    }
    fn state_bytes(s: &Vec<u32>) -> usize {
        s.len() * 4
    }
    fn energies(&mut self, s: &Vec<u32>, cands: &[Vec<u32>]) -> anyhow::Result<Vec<f32>> {
        Ok(cands.iter().map(|c| Sha256::digest(format!("{s:?}|{c:?}"))[0] as f32 / 64.0).collect())
    }
}

async fn post(app: &axum::Router, body: impl Into<Body>) -> (StatusCode, axum::http::HeaderMap, Value) {
    let req = Request::post("/v1/systemone").header("content-type", "application/json").body(body.into()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = resp.into_parts();
    let bytes = to_bytes(body, usize::MAX).await.unwrap();
    (parts.status, parts.headers, serde_json::from_slice(&bytes).unwrap())
}

fn multi_without_model() -> String {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/protocol/examples/multi.request.json");
    let mut v: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
    v.as_object_mut().unwrap().remove("model");
    v.to_string()
}

#[tokio::test]
async fn answers_and_reports_cache_state() {
    let app = krite_server::router(Runtime::new(HashBackend, 1 << 20));
    let (status, h, body) = post(&app, multi_without_model()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(h["x-krite-request-id"], h["x-typesafe-request-id"]);
    assert_eq!(h["x-krite-state-cache"], "miss");
    let timing = h["x-krite-timing"].to_str().unwrap();
    let keys: Vec<&str> = timing.split(';').map(|kv| kv.split('=').next().unwrap()).collect();
    assert_eq!(keys, ["tokenize", "encode", "decide", "calibrate"]);
    assert_eq!(body["model"], "hash");
    assert_eq!(body["answers"].as_object().unwrap().len(), 3);
    assert!(body["latency_ms"].as_f64().unwrap() >= 0.0);
    let (_, h2, body2) = post(&app, multi_without_model()).await;
    assert_eq!(h2["x-krite-state-cache"], "hit");
    assert_ne!(h["x-krite-request-id"], h2["x-krite-request-id"]);
    assert_eq!(body["answers"], body2["answers"]);
}

#[tokio::test]
async fn protocol_errors() {
    let app = krite_server::router(Runtime::new(HashBackend, 1 << 20));
    let unprocessable = |status: StatusCode, body: &Value, kind: &str| {
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["error"]["type"], kind);
    };

    let (status, _, body) = post(&app, "{").await;
    unprocessable(status, &body, "invalid_request");
    assert_eq!(body["error"]["param"], Value::Null);

    let big = format!(r#"{{"state":"{}","questions":{{}}}}"#, "x".repeat(4 * 1024 * 1024));
    let (status, _, body) = post(&app, big).await;
    unprocessable(status, &body, "invalid_request");

    let qs: serde_json::Map<String, Value> =
        (0..65).map(|i| (format!("q{i}"), json!({"type": "noul", "instructions": "i"}))).collect();
    let (status, _, body) = post(&app, json!({"state": "s", "questions": qs}).to_string()).await;
    unprocessable(status, &body, "too_many_questions");

    let q = json!({"q": {"type": "noul", "instructions": "i"}});
    let (status, _, body) =
        post(&app, json!({"model": "krite-0.15b-v1", "state": "s", "questions": q}).to_string()).await;
    unprocessable(status, &body, "unknown_model");
}

//! HTTP-level checks of `POST /v1/systemone` and `GET /v1/models` against a weight-free backend.

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use krite_runtime::Runtime;
use serde_json::{Value, json};
use tower::ServiceExt;

#[path = "support/hash.rs"]
mod hash;

use hash::HashBackend;

async fn post(app: &axum::Router, body: impl Into<Body>) -> (StatusCode, axum::http::HeaderMap, Value) {
    let req = Request::post("/v1/systemone").header("content-type", "application/json").body(body.into()).unwrap();
    send(app, req).await
}

async fn send(app: &axum::Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Value) {
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

#[tokio::test]
async fn lists_the_model_and_its_jev_alias() {
    let app = krite_server::router(Runtime::new(HashBackend, 1 << 20));
    let (status, h, body) = send(&app, Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(h["x-krite-request-id"], h["x-typesafe-request-id"]);
    let names: Vec<&str> = body["models"].as_array().unwrap().iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["hash", "jev-latest"]);
    for m in body["models"].as_array().unwrap() {
        assert!(m["description"].is_string() && m["release_date"].is_string(), "{m}");
    }
}

#[tokio::test]
async fn answers_the_sdk_request_shape() {
    let app = krite_server::router(Runtime::new(HashBackend, 1 << 20));
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/protocol/examples/sdk.request.json");
    let (status, _, body) = post(&app, std::fs::read(p).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "hash"); // the request asked for `jev-latest`
    let u = &body["answers"]["urgency"];
    assert_eq!(u["legend"]["1"], json!({"level": "Needs attention this week"}));
    let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(keys(&u["probabilities"]), ["0", "1", "2"]);
    assert_eq!(keys(&u["probabilities"]), keys(&u["legend"]));
}

//! `POST /v1/systemone` (docs/protocol/v1.md) over a `Runtime`. Binds loopback only; no auth.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use krite_core::ApiError;
use krite_runtime::{Backend, Runtime};

/// axum's own limit; bodies between the protocol limit (4 MiB) and this get the Protocol 422, not a plain 413.
const AXUM_BODY_LIMIT: usize = 16 * 1024 * 1024;

struct App<B: Backend> {
    // ponytail: one global lock (spec default concurrency 1; one Metal queue); per-device workers if concurrency matters
    rt: Arc<Mutex<Runtime<B>>>,
    max_body: usize,
}

impl<B: Backend> Clone for App<B> {
    fn clone(&self) -> Self {
        App { rt: self.rt.clone(), max_body: self.max_body }
    }
}

pub fn router<B: Backend>(rt: Runtime<B>) -> Router {
    let app = App { max_body: rt.limits().max_body_bytes, rt: Arc::new(Mutex::new(rt)) };
    Router::new()
        .route("/v1/systemone", post(systemone::<B>))
        .layer(DefaultBodyLimit::max(AXUM_BODY_LIMIT))
        .with_state(app)
}

/// Serves on 127.0.0.1:`port` until the process is killed.
pub async fn serve<B: Backend>(rt: Runtime<B>, port: u16) -> anyhow::Result<()> {
    let label = format!("{} on 127.0.0.1:{port} ({})", rt.backend().model_id(), rt.backend().backend_name());
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    eprintln!("serving {label}");
    axum::serve(listener, router(rt)).await?;
    Ok(())
}

fn request_id() -> String {
    static START: OnceLock<u64> = OnceLock::new();
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let start = *START.get_or_init(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64));
    format!("{start:016x}{:016x}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn error(e: ApiError, headers: HeaderMap) -> Response {
    if e.status() == 500 {
        eprintln!("internal_error: {}", e.message);
    }
    let status = StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, headers, Json(e.body())).into_response()
}

async fn systemone<B: Backend>(State(app): State<App<B>>, body: Bytes) -> Response {
    let mut headers = HeaderMap::new();
    let rid = HeaderValue::from_str(&request_id()).expect("hex is a valid header value");
    headers.insert("x-krite-request-id", rid.clone());
    headers.insert("x-typesafe-request-id", rid);
    if body.len() > app.max_body {
        let msg = format!("request body is {} bytes; max is {}", body.len(), app.max_body);
        return error(ApiError::invalid(msg, None), headers);
    }
    let req = match krite_core::parse(&body) {
        Ok(r) => r,
        Err(e) => return error(e, headers),
    };
    // `latency_ms` covers parsed request to answers, before serialization (Protocol v1 §3).
    let t0 = Instant::now();
    let rt = app.rt.clone();
    let decided = tokio::task::spawn_blocking(move || {
        let mut rt = rt.lock().map_err(|_| ApiError::internal("runtime lock poisoned by an earlier panic"))?;
        rt.decide(&req)
    })
    .await
    .unwrap_or_else(|e| Err(ApiError::internal(format!("decision task failed: {e}"))));
    match decided {
        Ok(mut d) => {
            d.response.latency_ms = t0.elapsed().as_secs_f64() * 1e3;
            headers.insert("x-krite-state-cache", HeaderValue::from_static(if d.cache_hit { "hit" } else { "miss" }));
            headers.insert("x-krite-timing", HeaderValue::from_str(&d.timing.header()).expect("ascii"));
            (StatusCode::OK, headers, Json(d.response)).into_response()
        }
        Err(e) => error(e, headers),
    }
}

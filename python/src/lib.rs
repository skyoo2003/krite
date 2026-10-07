//! `krite._krite`: the Krite runtime in process; `krite/__init__.py` is the public API.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use krite_candle::{CandleBackend, serving_runtime};
use krite_core::ApiError;
use krite_runtime::{Backend, Runtime};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// A loaded model. Calls run one at a time (one runtime lock) with the GIL released.
#[pyclass(module = "krite._krite", frozen)]
struct Engine {
    rt: Mutex<Runtime<CandleBackend>>,
    model_id: String,
    max_body: usize,
}

impl Engine {
    /// Mirrors the `POST /v1/systemone` handler in krite-server.
    fn answer(&self, body: &[u8]) -> Result<String, ApiError> {
        if body.len() > self.max_body {
            let msg = format!("request body is {} bytes; max is {}", body.len(), self.max_body);
            return Err(ApiError::invalid(msg, None));
        }
        let req = krite_core::parse(body)?;
        // `latency_ms` covers parsed request to answers, before serialization (Protocol v1 §3).
        let t0 = Instant::now();
        let mut rt = self.rt.lock().map_err(|_| ApiError::internal("runtime lock poisoned by an earlier panic"))?;
        let mut d = rt.decide(&req)?;
        drop(rt);
        d.response.latency_ms = t0.elapsed().as_secs_f64() * 1e3;
        serde_json::to_string(&d.response).map_err(|e| ApiError::internal(format!("serialize response: {e}")))
    }
}

#[pymethods]
impl Engine {
    #[new]
    fn new(
        py: Python<'_>,
        model: PathBuf,
        device: &str,
        state_cache_mb: usize,
        candidate_cache_mb: usize,
        raw: bool,
    ) -> PyResult<Self> {
        let cpu = match device {
            "auto" => false,
            "cpu" => true,
            _ => return Err(PyValueError::new_err(format!("device must be \"auto\" or \"cpu\", not {device:?}"))),
        };
        let rt = py
            .detach(|| serving_runtime(&model, cpu, state_cache_mb << 20, candidate_cache_mb << 20, raw))
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        let (model_id, max_body) = (rt.backend().model_id().to_string(), rt.limits().max_body_bytes);
        Ok(Self { rt: Mutex::new(rt), model_id, max_body })
    }

    #[getter]
    fn model_id(&self) -> &str {
        &self.model_id
    }

    /// (status, JSON body), exactly as `POST /v1/systemone` answers.
    fn decide(&self, py: Python<'_>, body: &[u8]) -> (u16, String) {
        py.detach(|| match self.answer(body) {
            Ok(json) => (200, json),
            Err(e) => (e.status(), e.body().to_string()),
        })
    }
}

#[pymodule]
fn _krite(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()
}

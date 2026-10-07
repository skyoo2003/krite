//! `krite serve`, `krite bench-encoder`, and `krite bench-decide`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use krite_candle::{CandleBackend, bench_candidates, bench_ids, device, serving_runtime};
use krite_runtime::Backend;
use serde_json::json;

/// 15 minutes: above this, model-layer cells drop to 50 measured runs (benchmark-spec §12 `truncated`).
const BUDGET_S: f64 = 15.0 * 60.0;

#[derive(Parser)]
#[command(name = "krite", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Dev {
    /// Metal when available, else CPU
    Auto,
    Cpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve Protocol v1 (POST /v1/systemone) on 127.0.0.1
    Serve {
        /// Model directory from `python -m krite_train.export`
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 8110)]
        port: u16,
        #[arg(long, value_enum, default_value_t = Dev::Auto)]
        device: Dev,
        /// State cache budget (MiB of state keys and values); 0 disables
        #[arg(long, default_value_t = 1024)]
        state_cache_mb: usize,
        /// Candidate lower-layer cache budget (MiB); 0 disables
        #[arg(long, default_value_t = 64)]
        candidate_cache_mb: usize,
        /// Serve uncalibrated probabilities (every temperature 1.0), for fitting temperatures
        #[arg(long)]
        raw: bool,
    },
    /// Model-layer encoder timings (benchmark-spec §2) as one JSON object on stdout
    BenchEncoder {
        /// Model directory from `python -m krite_train.export`
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "64,512,2048")]
        tokens: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        warmup: usize,
        #[arg(long, default_value_t = 200)]
        n: usize,
        #[arg(long, value_enum, default_value_t = Dev::Auto)]
        device: Dev,
    },
    /// Model-layer timings of state encoding and warm candidate scoring (one JSON object on stdout)
    BenchDecide(BenchDecide),
}

#[derive(Args)]
struct BenchDecide {
    /// Model directory from `python -m krite_train.export`
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 512)]
    state_tokens: usize,
    #[arg(long, value_delimiter = ',', default_value = "1,10,30")]
    questions: Vec<usize>,
    #[arg(long, default_value_t = 4)]
    options: usize,
    #[arg(long, default_value_t = 5)]
    warmup: usize,
    #[arg(long, default_value_t = 50)]
    n: usize,
    /// Free-form tag for the JSON row (e.g. the optimization step)
    #[arg(long, default_value = "")]
    label: String,
    #[arg(long, value_enum, default_value_t = Dev::Auto)]
    device: Dev,
}

fn load(model: &Path, dev: Dev, candidate_cache_mb: usize) -> anyhow::Result<CandleBackend> {
    CandleBackend::load(model, device(matches!(dev, Dev::Cpu))?, candidate_cache_mb << 20)
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn bench_encoder(model: &Path, tokens: &[usize], warmup: usize, n: usize, dev: Dev) -> anyhow::Result<()> {
    let b = load(model, dev, 0)?;
    let mut results = BTreeMap::new();
    for &s in tokens {
        anyhow::ensure!(s >= 2, "--tokens values must be at least 2 (<bos> and <eos>)");
        let ids = bench_ids(s);
        let warm = (0..warmup).map(|_| Ok(b.forward_timed(&ids)?.0)).collect::<anyhow::Result<Vec<f64>>>()?;
        let truncated = median(&warm) * (warmup + n) as f64 / 1000.0 > BUDGET_S;
        let runs = if truncated { n.min(50) } else { n };
        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            times.push(b.forward_timed(&ids)?.0);
        }
        let h = b.forward_timed(&ids)?.1.squeeze(0)?;
        let probe = [0, s / 2, s - 1]
            .iter()
            .map(|&p| h.get(p)?.narrow(0, 0, 8)?.to_vec1::<f32>())
            .collect::<candle_core::Result<Vec<_>>>()?;
        eprintln!("{s} tokens: p50 {:.1} ms over {runs} runs", median(&times));
        results.insert(s.to_string(), json!({"times_ms": times, "truncated": truncated, "probe": probe}));
    }
    println!("{}", json!({"backend": b.backend_name(), "precision": "fp32", "results": results}));
    Ok(())
}

/// Warm timings like the HTTP warm cell: the candidate cache is filled during warmup, so only the late
/// layers and the scorer run in the timed calls.
fn bench_decide(a: &BenchDecide) -> anyhow::Result<()> {
    anyhow::ensure!(a.state_tokens >= 2, "--state-tokens must be at least 2 (<bos> and <eos>)");
    let mut b = load(&a.model, a.device, 64)?;
    let ids = bench_ids(a.state_tokens);
    let mut encode_ms = Vec::with_capacity(a.n);
    for i in 0..a.warmup + a.n {
        let t = Instant::now();
        b.encode(&ids)?;
        if i >= a.warmup {
            encode_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    eprintln!("encode {} tokens: p50 {:.1} ms", a.state_tokens, median(&encode_ms));
    let state = b.encode(&ids)?;
    let mut results = BTreeMap::new();
    for &q in &a.questions {
        let cands = bench_candidates(&b, q, a.options)?;
        for _ in 0..a.warmup.max(1) {
            b.energies(&state, &cands)?;
        }
        let mut times = Vec::with_capacity(a.n);
        for _ in 0..a.n {
            let t = Instant::now();
            b.energies(&state, &cands)?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let tokens: usize = cands.iter().map(|c| c.ids.len()).sum();
        eprintln!("{q} questions ({} candidates): p50 {:.1} ms", cands.len(), median(&times));
        results.insert(q.to_string(), json!({"decide_ms": times, "candidates": cands.len(), "tokens": tokens}));
    }
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!(
        "{}",
        json!({
            "label": a.label, "commit": commit, "backend": b.backend_name(), "precision": "fp32",
            "state_tokens": a.state_tokens, "options": a.options, "encode_ms": encode_ms, "results": results,
        })
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve { model, port, device, state_cache_mb, candidate_cache_mb, raw } => {
            let cpu = matches!(device, Dev::Cpu);
            let rt = serving_runtime(&model, cpu, state_cache_mb << 20, candidate_cache_mb << 20, raw)?;
            tokio::runtime::Runtime::new()?.block_on(krite_server::serve(rt, port))
        }
        Cmd::BenchEncoder { model, tokens, warmup, n, device } => bench_encoder(&model, &tokens, warmup, n, device),
        Cmd::BenchDecide(a) => bench_decide(&a),
    }
}

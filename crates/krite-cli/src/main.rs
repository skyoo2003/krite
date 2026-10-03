//! `krite serve` and `krite bench-encoder`.

use std::collections::BTreeMap;

use clap::{Parser, Subcommand, ValueEnum};
use krite_candle::{CandleBackend, bench_ids, device};
use krite_runtime::{Backend, Runtime};
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
        #[arg(long, default_value_t = 8110)]
        port: u16,
        #[arg(long, value_enum, default_value_t = Dev::Auto)]
        device: Dev,
        /// State cache budget (MiB of encoder hidden states)
        #[arg(long, default_value_t = 1024)]
        state_cache_mb: usize,
    },
    /// Model-layer encoder timings (benchmark-spec §2) as one JSON object on stdout
    BenchEncoder {
        #[arg(long, value_delimiter = ',', default_value = "64,512,2048")]
        tokens: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        warmup: usize,
        #[arg(long, default_value_t = 200)]
        n: usize,
        #[arg(long, value_enum, default_value_t = Dev::Auto)]
        device: Dev,
    },
}

fn load(dev: Dev) -> anyhow::Result<CandleBackend> {
    CandleBackend::load(device(matches!(dev, Dev::Cpu))?)
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn bench_encoder(tokens: &[usize], warmup: usize, n: usize, dev: Dev) -> anyhow::Result<()> {
    let b = load(dev)?;
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

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve { port, device, state_cache_mb } => {
            let mut backend = load(device)?;
            backend.warmup()?;
            let rt = Runtime::new(backend, state_cache_mb * 1024 * 1024);
            tokio::runtime::Runtime::new()?.block_on(krite_server::serve(rt, port))
        }
        Cmd::BenchEncoder { tokens, warmup, n, device } => bench_encoder(&tokens, warmup, n, device),
    }
}

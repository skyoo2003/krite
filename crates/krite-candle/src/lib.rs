//! Candle backend: mmBERT-small (ModernBERT) state encoder on Metal or CPU, plus the slice head.

pub mod modernbert;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use krite_runtime::Backend;
use modernbert::{Config, ModernBert};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "krite-0.15b-v0";
const REPO: &str = "jhu-clsp/mmBERT-small";
/// config.json and tokenizer.json (this revision ships only pytorch_model.bin).
const CONFIG_REV: &str = "abc32620dd4f6ab06f5fbe905dc25f310618e09f";
/// refs/pr/12: model.safetensors converted from the same weights.
const WEIGHTS_REV: &str = "461475a70b192efcbb760df5541d3825b07c4d5b";

/// A file of `REPO` at `rev` in the local Hugging Face cache.
fn hub_file(rev: &str, file: &str) -> anyhow::Result<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")))
        .context("no HF_HUB_CACHE, HF_HOME, or HOME")?;
    let path = hub.join(format!("models--{}", REPO.replace('/', "--"))).join("snapshots").join(rev).join(file);
    if !path.exists() {
        bail!(
            "{file} not in the Hugging Face cache; run: uvx --from huggingface_hub hf download {REPO} {file} --revision {rev}"
        );
    }
    Ok(path)
}

/// Metal when available unless `cpu` is set.
pub fn device(cpu: bool) -> anyhow::Result<Device> {
    if !cpu && candle_core::utils::metal_is_available() { Ok(Device::new_metal(0)?) } else { Ok(Device::Cpu) }
}

/// Token ids for model-layer timing; keep in sync with `benchmarks/baselines/encoder_torch.py`.
pub fn bench_ids(n: usize) -> Vec<u32> {
    let mut v = vec![2u32];
    v.extend((0..n.saturating_sub(2)).map(|i| ((1000 + 7919 * i) % 256_000) as u32));
    v.push(1);
    v
}

pub struct CandleBackend {
    model: ModernBert,
    tokenizer: Tokenizer,
    device: Device,
    tokenizer_version: String,
}

impl CandleBackend {
    pub fn load(device: Device) -> anyhow::Result<Self> {
        let config: Config = serde_json::from_slice(&std::fs::read(hub_file(CONFIG_REV, "config.json")?)?)?;
        let tok_path = hub_file(CONFIG_REV, "tokenizer.json")?;
        let tok_bytes = std::fs::read(&tok_path)?;
        let tokenizer = Tokenizer::from_bytes(&tok_bytes).map_err(anyhow::Error::msg)?;
        let tokenizer_version = Sha256::digest(&tok_bytes)[..8].iter().map(|b| format!("{b:02x}")).collect();
        let weights = hub_file(WEIGHTS_REV, "model.safetensors")?;
        // SAFETY: the mmapped file is a read-only snapshot in the HF cache that is not modified while serving.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        // Tied embeddings: the input embedding is stored once, under the decoder name.
        let vb = vb.rename_f(|n| {
            if n == "model.embeddings.tok_embeddings.weight" { "decoder.weight".to_string() } else { n.to_string() }
        });
        let model = ModernBert::load(vb, &config)?;
        Ok(Self { model, tokenizer, device, tokenizer_version })
    }

    /// Compiles the Metal pipelines before the first request (the very first forward takes seconds).
    pub fn warmup(&mut self) -> anyhow::Result<()> {
        let ids = self.tokenize("warm up the encoder and the head", true)?;
        let state = self.encode(&ids)?;
        let cands = [self.tokenize("yes", false)?, self.tokenize("no", false)?];
        self.energies(&state, &cands)?;
        Ok(())
    }

    /// Model layer (spec §2): encoder forward on ids, timed until the device has finished.
    pub fn forward_timed(&self, ids: &[u32]) -> anyhow::Result<(f64, Tensor)> {
        let x = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let t = Instant::now();
        let h = self.model.forward(&x)?;
        self.device.synchronize()?;
        Ok((t.elapsed().as_secs_f64() * 1e3, h))
    }
}

impl Backend for CandleBackend {
    type State = Tensor;

    fn model_id(&self) -> &str {
        MODEL_ID
    }

    fn backend_name(&self) -> &str {
        if self.device.is_metal() { "candle-metal" } else { "candle-cpu" }
    }

    fn tokenizer_version(&self) -> &str {
        &self.tokenizer_version
    }

    fn tokenize(&self, text: &str, special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, special_tokens).map_err(anyhow::Error::msg)?.get_ids().to_vec())
    }

    fn encode(&mut self, state_ids: &[u32]) -> anyhow::Result<Tensor> {
        let h = self.model.forward(&Tensor::new(state_ids, &self.device)?.unsqueeze(0)?)?;
        self.device.synchronize()?;
        Ok(h)
    }

    fn state_bytes(state: &Tensor) -> usize {
        state.elem_count() * state.dtype().size_in_bytes()
    }

    /// Slice head, not the Krite decision tower: energy_i = 10 · cos(mean_t H_state[t], mean_j E[c_i, j]).
    /// One shared function per candidate, so it is order-invariant and question-isolated; its quality is
    /// meaningless.
    fn energies(&mut self, state: &Tensor, candidates: &[Vec<u32>]) -> anyhow::Result<Vec<f32>> {
        let n = candidates.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if candidates.iter().any(Vec::is_empty) {
            bail!("a candidate text has no tokens");
        }
        // Segment mean pooling: memory is O((T + n)·d); the runtime bounds T (`max_candidate_tokens`).
        let ids: Vec<u32> = candidates.concat();
        let segment: Vec<u32> =
            candidates.iter().enumerate().flat_map(|(i, c)| std::iter::repeat_n(i as u32, c.len())).collect();
        let lens: Vec<f32> = candidates.iter().map(|c| c.len() as f32).collect();
        let d = self.model.embeddings().dim(1)?;
        let emb = self.model.embeddings().index_select(&Tensor::new(ids, &self.device)?, 0)?; // (T, d)
        let segment = Tensor::new(segment, &self.device)?;
        let sums = Tensor::zeros((n, d), DType::F32, &self.device)?.index_add(&segment, &emb, 0)?; // (n, d)
        let c = sums.broadcast_div(&Tensor::from_vec(lens, (n, 1), &self.device)?)?;
        let h = state.mean(1)?.squeeze(0)?; // (d)
        let dot = c.matmul(&h.unsqueeze(1)?)?.squeeze(1)?;
        let norms = (c.sqr()?.sum(1)?.sqrt()? * h.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64)?;
        Ok(((dot / norms)? * 10.0)?.to_vec1::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    //! These need the weights in the Hugging Face cache: `cargo test -p krite-candle --release -- --ignored`.
    use super::*;

    fn backend() -> CandleBackend {
        CandleBackend::load(device(false).unwrap()).unwrap()
    }

    #[test]
    fn bench_ids_frame_with_bos_eos() {
        let ids = bench_ids(64);
        assert_eq!((ids.len(), ids[0], ids[63], ids[1]), (64, 2, 1, 1000));
    }

    #[test]
    #[ignore]
    fn encoder_matches_torch_reference() {
        // hidden[0, 0, 0..4] of bench_ids(64) from `encoder_torch.py --probe` (torch MPS, fp32).
        let want = std::env::var("KRITE_TORCH_PROBE").ok();
        let b = backend();
        let (_, h) = b.forward_timed(&bench_ids(64)).unwrap();
        assert_eq!(h.dims(), &[1, 64, 384]);
        let got: Vec<f32> = h.get(0).unwrap().get(0).unwrap().narrow(0, 0, 4).unwrap().to_vec1().unwrap();
        assert!(got.iter().all(|x| x.is_finite()));
        if let Some(w) = want {
            let w: Vec<f32> = serde_json::from_str(&w).unwrap();
            let diff = got.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(diff <= 1e-4, "max |Δ| {diff}: {got:?} vs {w:?}");
        }
    }

    #[test]
    #[ignore]
    fn head_is_order_invariant_and_isolated() {
        let mut b = backend();
        let state = b.encode(&b.tokenize("The invoice was charged twice.", true).unwrap()).unwrap();
        let c: Vec<Vec<u32>> =
            ["billing", "technical", "account", "other"].iter().map(|t| b.tokenize(t, false).unwrap()).collect();
        let fwd = b.energies(&state, &c).unwrap();
        let mut rev_c = c.clone();
        rev_c.reverse();
        let mut rev = b.energies(&state, &rev_c).unwrap();
        rev.reverse();
        let mut crowd = c.clone();
        crowd.extend((0..15).map(|i| b.tokenize(&format!("filler {i}"), false).unwrap()));
        let crowded = b.energies(&state, &crowd).unwrap();
        for i in 0..4 {
            assert!((fwd[i] - rev[i]).abs() <= 1e-5);
            assert!((fwd[i] - crowded[i]).abs() <= 1e-5);
        }
    }
}

//! Candle backend: the late-interaction decision tower (mmBERT-small encoder) on Metal or CPU, loaded
//! from a model directory written by `training/krite_train/export.py`.

pub mod late;
pub mod modernbert;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use krite_runtime::{Backend, Candidate, StateCache};
use late::{LateModel, LateState};
use modernbert::Config;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

/// `krite.json` in a model directory.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model_id: String,
    pub late_layers: usize,
    /// Candidate ids per question criterion, `<bos>`/`<eos>` included.
    pub max_candidate_tokens: usize,
    /// Ids kept from `"\n" + name[: desc]` before the instructions get the rest.
    pub max_option_tokens: usize,
    /// Calibration temperature per bucket; missing buckets use 1.0.
    pub temperatures: BTreeMap<String, f64>,
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

/// `<bos> instructions option <eos>` within `max` ids; the option keeps up to `max_option` ids first.
fn frame(bos: u32, mut ins: Vec<u32>, mut opt: Vec<u32>, eos: u32, max: usize, max_option: usize) -> Vec<u32> {
    opt.truncate(max_option);
    ins.truncate(max.saturating_sub(2 + opt.len()));
    std::iter::once(bos).chain(ins).chain(opt).chain([eos]).collect()
}

/// Candidate cache key: sha256 of the ids.
fn ids_key(ids: &[u32]) -> [u8; 32] {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    h.finalize().into()
}

fn qtype(kind: &str) -> u32 {
    match kind {
        "choice" => 0,
        "score" => 1,
        _ => 2,
    }
}

fn model_file(dir: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let path = dir.join(name);
    if !path.exists() {
        bail!(
            "{} missing; export a checkpoint: cd training && uv run python -m krite_train.export --ckpt ckpt/late8",
            path.display()
        );
    }
    Ok(path)
}

pub struct CandleBackend {
    model: LateModel,
    manifest: Manifest,
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
    tokenizer_version: String,
    /// Lower-layer candidate states by their ids; a 0-byte budget caches nothing.
    candidates: StateCache<Tensor>,
}

impl CandleBackend {
    pub fn load(dir: &Path, device: Device, candidate_cache_bytes: usize) -> anyhow::Result<Self> {
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(model_file(dir, "krite.json")?)?).context("parse krite.json")?;
        let config: Config =
            serde_json::from_slice(&std::fs::read(model_file(dir, "config.json")?)?).context("parse config.json")?;
        ensure!(
            manifest.max_candidate_tokens <= config.local_attention / 2,
            "candidates must fit in half the local attention window ({})",
            config.local_attention / 2
        );
        let tok_bytes = std::fs::read(model_file(dir, "tokenizer.json")?)?;
        let tokenizer = Tokenizer::from_bytes(&tok_bytes).map_err(anyhow::Error::msg)?;
        let tokenizer_version = Sha256::digest(&tok_bytes)[..8].iter().map(|b| format!("{b:02x}")).collect();
        let weights = model_file(dir, "model.safetensors")?;
        // SAFETY: the mmapped file is an exported model file that is not modified while serving.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let model = LateModel::load(vb, &config, manifest.late_layers, manifest.max_candidate_tokens)?;
        Ok(Self {
            model,
            manifest,
            config,
            tokenizer,
            device,
            tokenizer_version,
            candidates: StateCache::new(candidate_cache_bytes),
        })
    }

    pub fn temperatures(&self) -> &BTreeMap<String, f64> {
        &self.manifest.temperatures
    }

    /// Compiles the Metal pipelines before the first request (the very first forward takes seconds).
    pub fn warmup(&mut self) -> anyhow::Result<()> {
        let ids = self.tokenize("warm up the encoder and the decision tower", true)?;
        let state = self.encode(&ids)?;
        let criteria = [("yes".to_string(), None), ("no".to_string(), None)];
        let mut cands = Vec::new();
        for kind in ["choice", "score", "noul"] {
            for ids in self.candidate_ids("Warm up?", &criteria)? {
                cands.push(Candidate { kind, ids });
            }
        }
        self.energies(&state, &cands)?;
        Ok(())
    }

    /// Model layer (spec §2): encoder forward on ids, timed until the device has finished.
    pub fn forward_timed(&self, ids: &[u32]) -> anyhow::Result<(f64, Tensor)> {
        let x = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let t = Instant::now();
        let h = self.model.encoder().forward(&x)?;
        self.device.synchronize()?;
        Ok((t.elapsed().as_secs_f64() * 1e3, h))
    }
}

impl Backend for CandleBackend {
    type State = LateState;

    fn model_id(&self) -> &str {
        &self.manifest.model_id
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

    fn encode(&mut self, state_ids: &[u32]) -> anyhow::Result<LateState> {
        let s = self.model.encode_state(state_ids, &self.device)?;
        self.device.synchronize()?;
        Ok(s)
    }

    fn state_bytes(state: &LateState) -> usize {
        state.bytes()
    }

    /// Same layout as `candidate_ids` in training/krite_train/model.py: instructions and option are
    /// tokenized separately; the instructions once per question.
    fn candidate_ids(
        &self,
        instructions: &str,
        criteria: &[(String, Option<String>)],
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let m = &self.manifest;
        let mut ins = self.tokenize(instructions, false)?;
        ins.truncate(m.max_candidate_tokens);
        let (bos, eos) = (self.config.bos_token_id, self.config.eos_token_id);
        criteria
            .iter()
            .map(|(name, desc)| {
                let opt = match desc {
                    Some(d) => format!("\n{name}: {d}"),
                    None => format!("\n{name}"),
                };
                let opt = self.tokenize(&opt, false)?;
                Ok(frame(bos, ins.clone(), opt, eos, m.max_candidate_tokens, m.max_option_tokens))
            })
            .collect()
    }

    fn energies(&mut self, state: &LateState, candidates: &[Candidate]) -> anyhow::Result<Vec<f32>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        if candidates.iter().any(|c| c.ids.is_empty()) {
            bail!("a candidate has no tokens");
        }
        let keys: Vec<[u8; 32]> = candidates.iter().map(|c| ids_key(&c.ids)).collect();
        let mut lowers: Vec<Option<Tensor>> = keys.iter().map(|k| self.candidates.get(k)).collect();
        // Distinct misses run the lower layers together, in bounded batches.
        let mut miss: Vec<usize> = Vec::new();
        for (i, l) in lowers.iter().enumerate() {
            if l.is_none() && !miss.iter().any(|&j| keys[j] == keys[i]) {
                miss.push(i);
            }
        }
        if !miss.is_empty() {
            let ids: Vec<&[u32]> = miss.iter().map(|&i| candidates[i].ids.as_slice()).collect();
            let fresh = self.model.lower(&ids, &self.device)?;
            for (&i, x) in miss.iter().zip(fresh) {
                for (j, l) in lowers.iter_mut().enumerate() {
                    if l.is_none() && keys[j] == keys[i] {
                        *l = Some(x.clone());
                    }
                }
                let bytes = x.elem_count() * x.dtype().size_in_bytes();
                self.candidates.insert(keys[i], x, bytes);
            }
        }
        let lowers: Vec<Tensor> = lowers.into_iter().map(|l| l.expect("every miss was filled")).collect();
        let kinds: Vec<u32> = candidates.iter().map(|c| qtype(c.kind)).collect();
        Ok(self.model.energies(state, &lowers, &kinds)?)
    }
}

#[cfg(test)]
mod tests {
    //! Ignored tests need an exported model directory:
    //! `KRITE_MODEL=$PWD/training/ckpt/late8/candle cargo test -p krite-candle --release -- --ignored`.
    use super::*;
    use serde_json::Value;

    #[test]
    fn bench_ids_frame_with_bos_eos() {
        let ids = bench_ids(64);
        assert_eq!((ids.len(), ids[0], ids[63], ids[1]), (64, 2, 1, 1000));
    }

    #[test]
    fn frame_keeps_the_option_first() {
        let f = frame(2, (100..140).collect(), (200..220).collect(), 1, 32, 14);
        assert_eq!(f.len(), 32);
        assert_eq!((f[0], f[31]), (2, 1));
        assert_eq!(&f[17..31], &(200..214).collect::<Vec<u32>>()[..]);
        assert_eq!(&f[1..17], &(100..116).collect::<Vec<u32>>()[..]);
        assert_eq!(frame(2, vec![7, 8], vec![9], 1, 32, 14), vec![2, 7, 8, 9, 1]);
    }

    fn model_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("KRITE_MODEL").map(PathBuf::from);
        if dir.is_none() {
            eprintln!("KRITE_MODEL not set; skipping");
        }
        dir
    }

    fn probe(dir: &Path) -> Value {
        serde_json::from_slice(&std::fs::read(dir.join("probe.json")).unwrap()).unwrap()
    }

    fn probe_candidates(b: &CandleBackend, p: &Value) -> Vec<Candidate> {
        p["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let kind = match c["kind"].as_str().unwrap() {
                    "choice" => "choice",
                    "score" => "score",
                    _ => "noul",
                };
                let criterion = (c["name"].as_str().unwrap().to_string(), c["desc"].as_str().map(str::to_string));
                let ids = b.candidate_ids(c["instructions"].as_str().unwrap(), &[criterion]).unwrap().remove(0);
                let want: Vec<u32> = serde_json::from_value(c["ids"].clone()).unwrap();
                assert_eq!(ids, want, "candidate ids of {c}");
                Candidate { kind, ids }
            })
            .collect()
    }

    fn max_dev(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    #[ignore]
    fn lower_rows_do_not_pin_the_batch() {
        // The candidate cache charges each row its own bytes, so a row must not keep the batch alive.
        let Some(dir) = model_dir() else { return };
        let b = CandleBackend::load(&dir, Device::Cpu, 0).unwrap();
        let cands: [&[u32]; 3] = [&[2, 10, 11, 12, 1], &[2, 20, 1], &[2, 30, 31, 1]];
        for (x, c) in b.model.lower(&cands, &Device::Cpu).unwrap().iter().zip(cands) {
            let (storage, _) = x.storage_and_layout();
            let candle_core::Storage::Cpu(s) = &*storage else { unreachable!("CPU device") };
            assert_eq!(s.as_slice::<f32>().unwrap().len(), c.len() * x.dim(1).unwrap(), "row {c:?} pins the batch");
        }
    }

    #[test]
    #[ignore]
    fn matches_torch_probe() {
        let Some(dir) = model_dir() else { return };
        let p = probe(&dir);
        let mut b = CandleBackend::load(&dir, device(false).unwrap(), 1 << 26).unwrap();
        let ids = b.tokenize(p["state_text"].as_str().unwrap(), true).unwrap();
        assert_eq!(ids, serde_json::from_value::<Vec<u32>>(p["state_ids"].clone()).unwrap());
        let cands = probe_candidates(&b, &p);
        let state = b.encode(&ids).unwrap();
        let got = b.energies(&state, &cands).unwrap();
        let want: Vec<f32> = serde_json::from_value(p["energies"].clone()).unwrap();
        let dev = max_dev(&got, &want);
        eprintln!("max |Δ energy| vs torch: {dev:.2e}");
        assert!(dev <= 1e-4, "max |Δ energy| {dev}: {got:?} vs {want:?}");
    }

    #[test]
    #[ignore]
    fn order_invariant_isolated_and_cache_free() {
        let Some(dir) = model_dir() else { return };
        let p = probe(&dir);
        let mut b = CandleBackend::load(&dir, device(false).unwrap(), 1 << 26).unwrap();
        let state = b.encode(&b.tokenize(p["state_text"].as_str().unwrap(), true).unwrap()).unwrap();
        let cands = probe_candidates(&b, &p);
        let fwd = b.energies(&state, &cands).unwrap();
        let warm = b.energies(&state, &cands).unwrap();
        let mut rev_c = cands.clone();
        rev_c.reverse();
        let mut rev = b.energies(&state, &rev_c).unwrap();
        rev.reverse();
        let mut crowd = cands.clone();
        let fillers: Vec<(String, Option<String>)> = (0..15).map(|i| (format!("filler {i}"), None)).collect();
        for ids in b.candidate_ids("Pick one.", &fillers).unwrap() {
            crowd.push(Candidate { kind: "choice", ids });
        }
        let crowded = b.energies(&state, &crowd).unwrap()[..cands.len()].to_vec();
        let mut cold = CandleBackend::load(&dir, device(false).unwrap(), 0).unwrap();
        // Its own state: tensors of another Metal device instance do not mix.
        let cold_state = cold.encode(&cold.tokenize(p["state_text"].as_str().unwrap(), true).unwrap()).unwrap();
        let uncached = cold.energies(&cold_state, &cands).unwrap();
        for (name, e) in [("warm", &warm), ("reversed", &rev), ("crowded", &crowded), ("uncached", &uncached)] {
            let dev = max_dev(&fwd, e);
            assert!(dev <= 1e-5, "{name}: max |Δ| {dev}");
        }
    }
}

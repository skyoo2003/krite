//! Late-interaction decision tower (ARCHITECTURE.md §1); a port of `LateModel` in
//! training/krite_train/model.py. Candidates run the lower encoder layers alone. In the top layers their
//! tokens attend to their own tokens and to the state's keys and values at that layer, at positions after
//! the state. A masked mean plus a question-type embedding goes to an MLP scorer.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};

use crate::modernbert::{Config, ModernBert};

/// Bytes of attention scores per top-layer batch (heads × padded tokens × (state + tokens) × f32).
const SCORE_BUDGET: usize = 256 << 20;

/// Padded candidate tokens per batch, in the lower and the top layers; bounds the MLP intermediate
/// (tokens × 2 × intermediate × f32) near 150 MiB whatever the mix of candidate lengths in a request.
const BATCH_TOKENS: usize = 16_384;

/// Candidate indices grouped into batches whose padded size (count × longest) stays within `budget`.
/// Sorting by length keeps short candidates out of long candidates' padding; a candidate longer than
/// the budget runs alone.
fn batches(lens: &[usize], budget: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| lens[i]);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for i in order {
        match batches.last_mut() {
            Some(b) if (b.len() + 1) * lens[i] <= budget => b.push(i),
            _ => batches.push(vec![i]),
        }
    }
    batches
}

/// State memory: per top layer, the state's rotated keys transposed (heads, head_dim, S) and its values
/// (heads, S, head_dim). It never sees questions, so it is cached across requests.
#[derive(Clone)]
pub struct LateState {
    kv: Vec<(Tensor, Tensor)>,
    len: usize,
}

impl LateState {
    pub fn bytes(&self) -> usize {
        self.kv.iter().map(|(k, v)| (k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes()).sum()
    }
}

pub struct LateModel {
    bert: ModernBert,
    split: usize,
    type_emb: Embedding,
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// Additive key-padding mask (n, 1, 1, t): 0 on real tokens, -inf on padding.
fn key_padding(lens: &[usize], t: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> =
        lens.iter().flat_map(|&l| (0..t).map(move |j| if j < l { 0.0 } else { f32::NEG_INFINITY })).collect();
    Tensor::from_vec(v, (lens.len(), 1, 1, t), dev)?.to_dtype(dtype)
}

/// Softmax over [state keys ; own keys] for every candidate, without copying the state per candidate.
/// q, k, v: (H, n, t, dh); kt: (H, dh, S); sv: (H, S, dh); pad: additive, broadcastable to (H, n, t, t).
/// Returns (H, n, t, dh).
///
/// Each score block gets its own fused softmax; the two are then weighted by exp(lse_block - lse).
/// Concatenating and slicing, or broadcasting over the (H, n, t, S) block, runs Candle's strided Metal
/// kernels, which are about 15× slower than the contiguous ones.
pub(crate) fn late_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kt: &Tensor,
    sv: &Tensor,
    pad: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let (h, n, t, dh) = q.dims4()?;
    let s = kt.dim(2)?;
    let q = (q * scale)?;
    let state = q.reshape((h, n * t, dh))?.matmul(kt)?.reshape((h, n, t, s))?;
    let own = q.matmul(&k.t()?.contiguous()?)?.broadcast_add(pad)?;
    // log Σ exp(x) = max x - log max softmax(x); max softmax(x) ≥ 1/len, so the log never underflows.
    let softmax_lse = |x: &Tensor| -> Result<(Tensor, Tensor)> {
        let p = softmax_last_dim(x)?;
        let lse = (x.max_keepdim(D::Minus1)? - p.max_keepdim(D::Minus1)?.log()?)?;
        Ok((p, lse))
    };
    let ((ps, ls), (po, lo)) = (softmax_lse(&state)?, softmax_lse(&own)?);
    let m = ls.maximum(&lo)?;
    let lse = (&m + ((&ls - &m)?.exp()? + (&lo - &m)?.exp()?)?.log()?)?;
    let from_state = ps.reshape((h, n * t, s))?.matmul(sv)?.reshape((h, n, t, dh))?;
    let from_own = po.matmul(v)?;
    from_state.broadcast_mul(&(ls - &lse)?.exp()?)? + from_own.broadcast_mul(&(lo - &lse)?.exp()?)?
}

impl LateModel {
    /// `max_candidate_tokens` extends the RoPE table: candidates sit after a state of up to
    /// `max_position_embeddings` tokens.
    pub fn load(vb: VarBuilder, config: &Config, late_layers: usize, max_candidate_tokens: usize) -> Result<Self> {
        let d = config.hidden_size;
        let bert = ModernBert::load(vb.pp("encoder"), config, max_candidate_tokens)?;
        if late_layers == 0 || late_layers > bert.num_layers() {
            candle_core::bail!("late_layers {late_layers} not in 1..={}", bert.num_layers());
        }
        let split = bert.num_layers() - late_layers;
        Ok(Self {
            split,
            bert,
            type_emb: embedding(3, d, vb.pp("type_emb"))?,
            norm: layer_norm(d, 1e-5, vb.pp("scorer.0"))?,
            fc1: linear(d, d, vb.pp("scorer.1"))?,
            fc2: linear(d, 1, vb.pp("scorer.3"))?,
        })
    }

    pub fn encoder(&self) -> &ModernBert {
        &self.bert
    }

    /// State memory for one state, `<bos>`/`<eos>` included.
    pub fn encode_state(&self, ids: &[u32], dev: &Device) -> Result<LateState> {
        let ids = Tensor::new(ids, dev)?.unsqueeze(0)?;
        let band = self.bert.band(ids.dim(1)?, dev)?;
        let mut h = self.bert.run(&self.bert.embed(&ids)?, 0..self.split, None, Some(&band))?;
        let mut kv = Vec::with_capacity(self.bert.num_layers() - self.split);
        for l in self.split..self.bert.num_layers() {
            let layer = self.bert.layer(l);
            let (_, k, v) = layer.attn.qkv(&layer.attn_input(&h)?, 0)?;
            kv.push((k.squeeze(0)?.t()?.contiguous()?, v.squeeze(0)?));
            if l + 1 < self.bert.num_layers() {
                h = self.bert.run(&h, l..l + 1, None, Some(&band))?;
            }
        }
        Ok(LateState { kv, len: ids.dim(1)? })
    }

    /// Lower-layer hidden states (len_i, hidden) per candidate; they depend on the candidate's ids only.
    /// Candidates run in length-sorted batches of at most `BATCH_TOKENS` padded tokens.
    pub fn lower(&self, cands: &[&[u32]], dev: &Device) -> Result<Vec<Tensor>> {
        let lens: Vec<usize> = cands.iter().map(|c| c.len()).collect();
        let mut out: Vec<Option<Tensor>> = vec![None; cands.len()];
        for batch in batches(&lens, BATCH_TOKENS) {
            let rows: Vec<&[u32]> = batch.iter().map(|&i| cands[i]).collect();
            for (i, x) in batch.into_iter().zip(self.lower_batch(&rows, dev)?) {
                out[i] = Some(x);
            }
        }
        Ok(out.into_iter().map(|x| x.expect("every candidate is in one batch")).collect())
    }

    /// One padded batch. Candidates fit in half the local window, so key padding is the only mask. Each
    /// row is copied out of the batch: a view would keep the whole batch alive in the candidate cache,
    /// which charges a row only its own bytes.
    fn lower_batch(&self, cands: &[&[u32]], dev: &Device) -> Result<Vec<Tensor>> {
        let t = cands.iter().map(|c| c.len()).max().unwrap_or(0);
        let lens: Vec<usize> = cands.iter().map(|c| c.len()).collect();
        let ids: Vec<u32> =
            cands.iter().flat_map(|c| c.iter().copied().chain(std::iter::repeat_n(0, t - c.len()))).collect();
        let ids = Tensor::from_vec(ids, (cands.len(), t), dev)?;
        let x = self.bert.embed(&ids)?;
        let pad = self.bert.mask(&key_padding(&lens, t, x.dtype(), dev)?, cands.len(), t)?;
        let x = self.bert.run(&x, 0..self.split, Some(&pad), Some(&pad))?;
        lens.iter().enumerate().map(|(i, &l)| x.get(i)?.narrow(0, 0, l)?.force_contiguous()).collect()
    }

    /// One energy per candidate from its lower-layer states and question type (choice 0, score 1, noul 2).
    pub fn energies(&self, state: &LateState, lowers: &[Tensor], kinds: &[u32]) -> Result<Vec<f32>> {
        let lens = lowers.iter().map(|x| x.dim(0)).collect::<Result<Vec<_>>>()?;
        let t = lens.iter().copied().max().unwrap_or(0);
        let heads = self.bert.layer(self.split).attn.heads;
        // ponytail: fixed score budget; tile over S if long states need more candidates per chunk
        let budget = BATCH_TOKENS.min(SCORE_BUDGET / (heads * (state.len + t) * 4));
        let mut out = vec![0.0; lowers.len()];
        for b in batches(&lens, budget) {
            let xs: Vec<Tensor> = b.iter().map(|&i| lowers[i].clone()).collect();
            let ks: Vec<u32> = b.iter().map(|&i| kinds[i]).collect();
            for (i, e) in b.into_iter().zip(self.chunk(state, &xs, &ks)?) {
                out[i] = e;
            }
        }
        Ok(out)
    }

    fn chunk(&self, state: &LateState, lowers: &[Tensor], kinds: &[u32]) -> Result<Vec<f32>> {
        let lens = lowers.iter().map(|x| x.dim(0)).collect::<Result<Vec<_>>>()?;
        let (n, t) = (lens.len(), lens.iter().copied().max().unwrap_or(0));
        let rows = lowers.iter().map(|x| x.pad_with_zeros(0, 0, t - x.dim(0)?)).collect::<Result<Vec<_>>>()?;
        let mut x = Tensor::stack(&rows, 0)?; // (n, t, d)
        let (dev, dtype, d) = (x.device().clone(), x.dtype(), x.dim(2)?);
        x = x.reshape((n * t, d))?;
        let pad = key_padding(&lens, t, dtype, &dev)?.reshape((1, n, 1, t))?;
        for (i, (kt, sv)) in state.kv.iter().enumerate() {
            let layer = self.bert.layer(self.split + i);
            let a = &layer.attn;
            let (q, k, v) = a.project(&layer.attn_input(&x)?.unsqueeze(0)?)?; // (H, n·t, dh)
            let split = |x: Tensor| x.reshape((a.heads, n, t, a.head_dim)); // (H, n, t, dh)
            let (q, k) = (a.rope(&split(q)?, state.len)?, a.rope(&split(k)?, state.len)?);
            let o = late_attention(&q, &k, &split(v)?, kt, sv, &pad, (a.head_dim as f64).powf(-0.5))?;
            x = (&x + o.permute((1, 2, 0, 3))?.reshape((n * t, d))?.apply(&a.proj)?)?;
            x = (&x + x.apply(&layer.mlp_norm)?.apply(&layer.mlp)?)?;
        }
        let x = x.apply(self.bert.final_norm())?.reshape((n, t, d))?;
        let w: Vec<f32> = lens.iter().flat_map(|&l| (0..t).map(move |j| if j < l { 1.0 } else { 0.0 })).collect();
        let w = Tensor::from_vec(w, (n, t, 1), &dev)?.to_dtype(dtype)?;
        let len =
            Tensor::from_vec(lens.iter().map(|&l| l as f32).collect::<Vec<_>>(), (n, 1), &dev)?.to_dtype(dtype)?;
        let c = x.broadcast_mul(&w)?.sum(1)?.broadcast_div(&len)?;
        let c = (c + Tensor::new(kinds, &dev)?.apply(&self.type_emb)?)?;
        let e = c.apply(&self.norm)?.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)?;
        e.squeeze(1)?.to_dtype(DType::F32)?.to_vec1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_bound_padded_tokens() {
        // Many short candidates plus one long one: one padded batch would hold 13,001 × 32 tokens.
        let mut lens = vec![5; 13_000];
        lens.insert(6_500, 32);
        let got = batches(&lens, BATCH_TOKENS);
        let mut seen: Vec<usize> = got.concat();
        seen.sort_unstable();
        assert_eq!(seen, (0..lens.len()).collect::<Vec<_>>());
        for b in &got {
            let t = b.iter().map(|&i| lens[i]).max().unwrap();
            assert!(b.len() * t <= BATCH_TOKENS, "batch of {} × {t}", b.len());
        }
        assert_eq!(batches(&[BATCH_TOKENS + 1], BATCH_TOKENS), vec![vec![0]]);
    }

    #[test]
    fn late_attention_matches_concatenated_reference() {
        let dev = Device::Cpu;
        let (h, n, t, s, dh) = (2, 3, 4, 5, 8);
        let r = |shape: &[usize]| Tensor::randn(0f32, 1.0, shape, &dev).unwrap();
        let (q, k, v, sk, sv) =
            (r(&[h, n, t, dh]), r(&[h, n, t, dh]), r(&[h, n, t, dh]), r(&[h, s, dh]), r(&[h, s, dh]));
        let lens = [4, 2, 3];
        let pad = key_padding(&lens, t, DType::F32, &dev).unwrap().reshape((1, n, 1, t)).unwrap();
        let kt = sk.t().unwrap().contiguous().unwrap();
        let got = late_attention(&q, &k, &v, &kt, &sv, &pad, 0.3).unwrap();
        for (c, &len) in lens.iter().enumerate() {
            let (qc, kc, vc) = (q.narrow(1, c, 1).unwrap(), k.narrow(1, c, 1).unwrap(), v.narrow(1, c, 1).unwrap());
            let keys = Tensor::cat(&[&sk, &kc.squeeze(1).unwrap().narrow(1, 0, len).unwrap()], 1).unwrap();
            let vals = Tensor::cat(&[&sv, &vc.squeeze(1).unwrap().narrow(1, 0, len).unwrap()], 1).unwrap();
            let att = (qc.squeeze(1).unwrap() * 0.3).unwrap().matmul(&keys.t().unwrap()).unwrap();
            let want = softmax_last_dim(&att).unwrap().matmul(&vals).unwrap();
            let diff = (got.narrow(1, c, 1).unwrap().squeeze(1).unwrap() - want).unwrap().abs().unwrap();
            let max = diff.flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            assert!(max <= 1e-5, "candidate {c}: max |Δ| {max}");
        }
    }
}

// Adapted from candle-transformers 0.11.0 (MIT OR Apache-2.0), models/modernbert.rs.
// Changes: backbone only (no MLM/classifier heads), weight names without the `model.` prefix; fused
// `sdpa` attention on Metal; additive masks chosen per call (global layers of one unpadded sequence run
// unmasked); the local band mask is built in the weight dtype; RoPE at a position offset, with a table
// longer than `max_position_embeddings` so decision-tower candidates can sit after a full-length state;
// layer-level access for the late-interaction tower. Fused projections are split at load: q/k/v into
// per-head (heads, hidden, head_dim) weights, so one batched matmul yields head-major q/k/v, and the GeGLU
// input into gate and up halves. Slicing or permuting fused outputs runs Candle's strided Metal kernels,
// which are an order of magnitude slower than the contiguous ones.

use std::ops::Range;
use std::sync::Arc;

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::ops::{sdpa, softmax_last_dim};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm_no_bias, linear_no_bias};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(vb: &VarBuilder, config: &Config, rope_theta: f64, len: usize) -> Result<Self> {
        let (dtype, dev) = (vb.dtype(), vb.device());
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> =
            (0..dim).step_by(2).map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32).collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, len as u32, dev)?.to_dtype(dtype)?.reshape((len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }

    /// Rotates x (b, heads, s, head_dim) as if its tokens sat at positions offset..offset + s.
    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let s = x.dim(2)?;
        let cos = self.cos.narrow(0, offset, s)?.contiguous()?;
        let sin = self.sin.narrow(0, offset, s)?.contiguous()?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}

pub(crate) struct Attention {
    /// (heads, hidden, head_dim) each.
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    pub(crate) proj: Linear,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn load(vb: VarBuilder, config: &Config, rotary: Arc<RotaryEmbedding>) -> Result<Self> {
        let (d, h) = (config.hidden_size, config.num_attention_heads);
        let w = vb.get((3 * d, d), "Wqkv.weight")?;
        let per_head = |i: usize| w.narrow(0, i * d, d)?.reshape((h, d / h, d))?.transpose(1, 2)?.contiguous();
        Ok(Self {
            wq: per_head(0)?,
            wk: per_head(1)?,
            wv: per_head(2)?,
            proj: linear_no_bias(d, d, vb.pp("Wo"))?,
            heads: h,
            head_dim: d / h,
            rotary,
        })
    }

    /// Unrotated q, k, v of rows x (1, rows, hidden), head-major: (heads, rows, head_dim) each.
    pub(crate) fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((x.broadcast_matmul(&self.wq)?, x.broadcast_matmul(&self.wk)?, x.broadcast_matmul(&self.wv)?))
    }

    /// Rotates x (b, h, s, head_dim) as if its tokens sat at positions offset..offset + s; `h` may be any
    /// batch-like dimension.
    pub(crate) fn rope(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        self.rotary.apply(x, offset)
    }

    /// q and k rotated at positions offset..offset + s, and v; each (b, heads, s, head_dim).
    pub(crate) fn qkv(&self, xs: &Tensor, offset: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, d) = xs.dims3()?;
        let (q, k, v) = self.project(&xs.reshape((1, b * s, d))?)?;
        // (heads, b, s, head_dim): RoPE treats `heads` as the batch; then batch-major for attention.
        let split = |x: Tensor| x.reshape((self.heads, b, s, self.head_dim));
        let (q, k, v) = (self.rotary.apply(&split(q)?, offset)?, self.rotary.apply(&split(k)?, offset)?, split(v)?);
        let major = |x: Tensor| {
            if b == 1 { x.reshape((1, self.heads, s, self.head_dim)) } else { x.transpose(0, 1)?.contiguous() }
        };
        Ok((major(q)?, major(k)?, major(v)?))
    }

    /// `mask` is additive and broadcastable to (b, heads, s, s), or None for full attention.
    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, d) = xs.dims3()?;
        let (q, k, v) = self.qkv(xs, 0)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let xs = if q.device().is_metal() {
            // Fused kernel; it has no CPU implementation.
            let mask = mask.map(|m| m.broadcast_as((b, self.heads, s, s))?.contiguous()).transpose()?;
            sdpa(&q, &k, &v, mask.as_ref(), false, scale as f32, 1.0)?
        } else {
            let att = (q * scale)?.matmul(&k.t()?.contiguous()?)?;
            let att = match mask {
                Some(m) => att.broadcast_add(m)?,
                None => att,
            };
            softmax_last_dim(&att)?.matmul(&v)?
        };
        xs.transpose(1, 2)?.reshape((b, s, d))?.apply(&self.proj)
    }
}

pub(crate) struct Mlp {
    gate: Linear,
    up: Linear,
    wo: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (d, i) = (config.hidden_size, config.intermediate_size);
        let wi = vb.get((2 * i, d), "Wi.weight")?;
        Ok(Self {
            gate: Linear::new(wi.narrow(0, 0, i)?.contiguous()?, None),
            up: Linear::new(wi.narrow(0, i, i)?.contiguous()?, None),
            wo: linear_no_bias(i, d, vb.pp("Wo"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.gate)?.gelu_erf()? * xs.apply(&self.up)?)?.apply(&self.wo) // GeGLU
    }
}

pub(crate) struct Layer {
    pub(crate) attn: Attention,
    pub(crate) mlp: Mlp,
    attn_norm: Option<LayerNorm>,
    pub(crate) mlp_norm: LayerNorm,
    local: bool,
}

impl Layer {
    /// Attention input: layer 0 has no attention norm.
    pub(crate) fn attn_input(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.attn_norm {
            Some(norm) => xs.apply(norm),
            None => Ok(xs.clone()),
        }
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let xs = (self.attn.forward(&self.attn_input(xs)?, mask)? + xs)?;
        let mlp = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        xs + mlp
    }
}

pub struct ModernBert {
    embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    half_window: usize,
    heads: usize,
}

impl ModernBert {
    /// `extra_positions`: RoPE rows beyond `max_position_embeddings`.
    pub fn load(vb: VarBuilder, config: &Config, extra_positions: usize) -> Result<Self> {
        let embeddings = embedding(config.vocab_size, config.hidden_size, vb.pp("embeddings.tok_embeddings"))?;
        let norm = layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("embeddings.norm"))?;
        let len = config.max_position_embeddings + extra_positions;
        let global = Arc::new(RotaryEmbedding::new(&vb, config, config.global_rope_theta, len)?);
        let local = Arc::new(RotaryEmbedding::new(&vb, config, config.local_rope_theta, len)?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let vb = vb.pp(format!("layers.{i}"));
            let is_local = i % config.global_attn_every_n_layers != 0;
            let rotary = if is_local { local.clone() } else { global.clone() };
            layers.push(Layer {
                attn: Attention::load(vb.pp("attn"), config, rotary)?,
                mlp: Mlp::load(vb.pp("mlp"), config)?,
                // Layer 0 has no attention norm.
                attn_norm: layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("attn_norm")).ok(),
                mlp_norm: layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?,
                local: is_local,
            });
        }
        let final_norm = layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("final_norm"))?;
        Ok(Self {
            embeddings,
            norm,
            layers,
            final_norm,
            half_window: config.local_attention / 2,
            heads: config.num_attention_heads,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, i: usize) -> &Layer {
        &self.layers[i]
    }

    pub fn final_norm(&self) -> &LayerNorm {
        &self.final_norm
    }

    /// Normalized token embeddings (b, s, hidden) of ids (b, s).
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        ids.apply(&self.embeddings)?.apply(&self.norm)
    }

    /// Additive sliding-window mask of the local layers for one sequence of s tokens, in the weight dtype,
    /// laid out by `mask`.
    pub fn band(&self, s: usize, dev: &Device) -> Result<Tensor> {
        // ponytail: dense O(S²) band mask, expanded per head (~1.6 GB at 8192 tokens) for the whole encode; windowed kernel later
        let w = self.half_window;
        let band: Vec<f32> =
            (0..s).flat_map(|i| (0..s).map(move |j| if i.abs_diff(j) > w { f32::NEG_INFINITY } else { 0.0 })).collect();
        self.mask(&Tensor::from_vec(band, (s, s), dev)?.to_dtype(self.embeddings.embeddings().dtype())?, 1, s)
    }

    /// An additive mask broadcastable to (b, heads, s, s), in the layout attention reads. The fused Metal
    /// kernel reads a dense mask; expanding it once per call instead of in every layer saves a strided
    /// copy per layer (1.4 ms per local layer at 512 tokens). The CPU path broadcasts as it adds.
    pub fn mask(&self, m: &Tensor, b: usize, s: usize) -> Result<Tensor> {
        if m.device().is_metal() { m.broadcast_as((b, self.heads, s, s))?.contiguous() } else { Ok(m.clone()) }
    }

    /// Runs `layers` on xs (b, s, hidden) with `global` / `local` masks on the global / local layers.
    pub fn run(
        &self,
        xs: &Tensor,
        layers: Range<usize>,
        global: Option<&Tensor>,
        local: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers[layers] {
            xs = layer.forward(&xs, if layer.local { local } else { global })?;
        }
        Ok(xs)
    }

    /// Per-token hidden states (1, S, hidden) for one unpadded sequence of ids (1, S).
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let band = self.band(ids.dim(1)?, ids.device())?;
        self.run(&self.embed(ids)?, 0..self.layers.len(), None, Some(&band))?.apply(&self.final_norm)
    }
}

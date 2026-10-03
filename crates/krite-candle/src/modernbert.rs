// Adapted from candle-transformers 0.11.0 (MIT OR Apache-2.0), models/modernbert.rs.
// Changes: backbone only (no MLM/classifier heads); fused `sdpa` attention on Metal; global layers run
// unmasked (one unpadded sequence per call); the local band mask is built in the weight dtype.

use std::sync::Arc;

use candle_core::{D, DType, Device, Module, Result, Tensor};
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
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &Config, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> =
            (0..dim).step_by(2).map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32).collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?.to_dtype(dtype)?.reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, &self.cos, &self.sin)
    }
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn load(vb: VarBuilder, config: &Config, rotary: Arc<RotaryEmbedding>) -> Result<Self> {
        Ok(Self {
            qkv: linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?,
            proj: linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?,
            heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            rotary,
        })
    }

    /// `mask` is an additive (seq, seq) mask, or None for full attention.
    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, d) = xs.dims3()?;
        let qkv = xs.apply(&self.qkv)?.reshape((b, s, 3, self.heads, self.head_dim))?.permute((2, 0, 3, 1, 4))?;
        let q = self.rotary.apply(&qkv.get(0)?)?;
        let k = self.rotary.apply(&qkv.get(1)?)?;
        let v = qkv.get(2)?.contiguous()?;
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

struct Mlp {
    wi: Linear,
    wo: Linear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let xs = xs.chunk(2, D::Minus1)?;
        (&xs[0].gelu_erf()? * &xs[1])?.apply(&self.wo) // GeGLU
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    local: bool,
}

impl Layer {
    fn forward(&self, xs: &Tensor, local_mask: &Tensor) -> Result<Tensor> {
        let h = match &self.attn_norm {
            Some(norm) => xs.apply(norm)?,
            None => xs.clone(),
        };
        let xs = (self.attn.forward(&h, self.local.then_some(local_mask))? + xs)?;
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
}

impl ModernBert {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embeddings = embedding(config.vocab_size, config.hidden_size, vb.pp("model.embeddings.tok_embeddings"))?;
        let norm = layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("model.embeddings.norm"))?;
        let global = Arc::new(RotaryEmbedding::new(vb.dtype(), config, config.global_rope_theta, vb.device())?);
        let local = Arc::new(RotaryEmbedding::new(vb.dtype(), config, config.local_rope_theta, vb.device())?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let vb = vb.pp(format!("model.layers.{i}"));
            let is_local = i % config.global_attn_every_n_layers != 0;
            let rotary = if is_local { local.clone() } else { global.clone() };
            layers.push(Layer {
                attn: Attention::load(vb.pp("attn"), config, rotary)?,
                mlp: Mlp {
                    wi: linear_no_bias(config.hidden_size, config.intermediate_size * 2, vb.pp("mlp.Wi"))?,
                    wo: linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("mlp.Wo"))?,
                },
                // Layer 0 has no attention norm.
                attn_norm: layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("attn_norm")).ok(),
                mlp_norm: layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?,
                local: is_local,
            });
        }
        let final_norm = layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("model.final_norm"))?;
        Ok(Self { embeddings, norm, layers, final_norm, half_window: config.local_attention / 2 })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.embeddings.embeddings()
    }

    /// Per-token hidden states (1, S, hidden) for one unpadded sequence of ids (1, S).
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let s = ids.dim(1)?;
        let dtype = self.embeddings().dtype();
        // ponytail: dense O(S²) band mask (~1.6 GB at 8192 tokens after the head broadcast); windowed kernel later
        let w = self.half_window;
        let band: Vec<f32> =
            (0..s).flat_map(|i| (0..s).map(move |j| if i.abs_diff(j) > w { f32::NEG_INFINITY } else { 0.0 })).collect();
        let local_mask = Tensor::from_vec(band, (s, s), ids.device())?.to_dtype(dtype)?;
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &local_mask)?;
        }
        xs.apply(&self.final_norm)
    }
}

//! Deterministic, weight-free backend: each energy hashes (state ids, candidate kind, candidate ids).
//! Shared by the HTTP tests and the `hash_server` example that the Jev conformance suite runs against.

use krite_runtime::{Backend, Candidate};
use sha2::{Digest, Sha256};

pub struct HashBackend;

impl Backend for HashBackend {
    type State = Vec<u32>;
    fn model_id(&self) -> &str {
        "hash"
    }
    fn backend_name(&self) -> &str {
        "none"
    }
    fn tokenizer_version(&self) -> &str {
        "bytes"
    }
    fn tokenize(&self, text: &str, _special: bool) -> anyhow::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }
    fn encode(&mut self, ids: &[u32]) -> anyhow::Result<Vec<u32>> {
        Ok(ids.to_vec())
    }
    fn state_bytes(s: &Vec<u32>) -> usize {
        s.len() * 4
    }
    fn candidate_ids(
        &self,
        instructions: &str,
        criteria: &[(String, Option<String>)],
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        criteria
            .iter()
            .map(|(n, d)| self.tokenize(&format!("{instructions}\n{n}: {}", d.as_deref().unwrap_or("")), false))
            .collect()
    }
    fn energies(&mut self, s: &Vec<u32>, cands: &[Candidate]) -> anyhow::Result<Vec<f32>> {
        Ok(cands.iter().map(|c| Sha256::digest(format!("{s:?}|{}|{:?}", c.kind, c.ids))[0] as f32 / 64.0).collect())
    }
}

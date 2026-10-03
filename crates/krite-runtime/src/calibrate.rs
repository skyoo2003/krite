//! `temperature[type][cardinality_bucket]` calibration (ARCHITECTURE.md Glossary).

use std::collections::HashMap;

/// Calibration bucket; mirrors `benchmarks/krite_bench/data.py::bucket`.
pub fn bucket(kind: &str, k: usize) -> &'static str {
    match (kind, k) {
        ("noul", _) => "noul",
        ("choice", 0..=2) => "choice/2",
        ("choice", 3) => "choice/3",
        ("choice", 4) => "choice/4",
        ("choice", 5..=8) => "choice/5-8",
        ("choice", _) => "choice/9+",
        (_, 0..=2) => "score/2",
        (_, 3) => "score/3",
        (_, 4) => "score/4",
        (_, 5) => "score/5",
        _ => "score/6+",
    }
}

/// Per-bucket temperatures; buckets without one use 1.0.
#[derive(Debug, Clone, Default)]
pub struct Calibrator {
    temps: HashMap<&'static str, f64>,
}

impl Calibrator {
    /// No calibration: every temperature is 1.0 (the slice head is untrained).
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn with(mut self, bucket: &'static str, temperature: f64) -> Self {
        self.temps.insert(bucket, temperature);
        self
    }

    pub fn temperature(&self, bucket: &str) -> f64 {
        self.temps.get(bucket).copied().unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_match_the_glossary_table() {
        let cases = [
            ("choice", 2, "choice/2"),
            ("choice", 4, "choice/4"),
            ("choice", 5, "choice/5-8"),
            ("choice", 8, "choice/5-8"),
            ("choice", 9, "choice/9+"),
            ("score", 2, "score/2"),
            ("score", 5, "score/5"),
            ("score", 6, "score/6+"),
            ("noul", 2, "noul"),
        ];
        for (kind, k, want) in cases {
            assert_eq!(bucket(kind, k), want, "{kind}/{k}");
        }
    }
}

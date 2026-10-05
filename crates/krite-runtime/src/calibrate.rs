//! `temperature[type][cardinality_bucket]` calibration (ARCHITECTURE.md Glossary).

use std::collections::HashMap;

/// Every bucket name, in ARCHITECTURE.md Glossary order.
pub const BUCKETS: [&str; 11] = [
    "choice/2",
    "choice/3",
    "choice/4",
    "choice/5-8",
    "choice/9+",
    "noul",
    "score/2",
    "score/3",
    "score/4",
    "score/5",
    "score/6+",
];

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

    /// From `(bucket, temperature)` pairs, e.g. a model's `krite.json`; unknown buckets and
    /// non-positive or non-finite temperatures are errors.
    pub fn from_temperatures<'a>(temps: impl IntoIterator<Item = (&'a str, f64)>) -> anyhow::Result<Self> {
        let mut c = Self::default();
        for (name, t) in temps {
            let Some(&b) = BUCKETS.iter().find(|&&b| b == name) else {
                anyhow::bail!("unknown calibration bucket {name:?}")
            };
            anyhow::ensure!(t.is_finite() && t > 0.0, "temperature of {name} must be positive, got {t}");
            c = c.with(b, t);
        }
        Ok(c)
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
            assert!(BUCKETS.contains(&want));
        }
    }

    #[test]
    fn from_temperatures_checks_names_and_values() {
        let c = Calibrator::from_temperatures([("choice/4", 1.5), ("noul", 0.5)]).unwrap();
        assert_eq!((c.temperature("choice/4"), c.temperature("noul"), c.temperature("score/5")), (1.5, 0.5, 1.0));
        assert!(Calibrator::from_temperatures([("choice/10", 1.0)]).is_err());
        assert!(Calibrator::from_temperatures([("noul", 0.0)]).is_err());
        assert!(Calibrator::from_temperatures([("noul", f64::NAN)]).is_err());
    }
}

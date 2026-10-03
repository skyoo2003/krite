//! Cross-request state cache, bounded by bytes, least-recently-used eviction.

use std::collections::HashMap;

pub type Key = [u8; 32];

// ponytail: O(n) min-scan eviction; switch to a linked LRU if the cache holds >10k states
pub struct StateCache<S> {
    map: HashMap<Key, (S, usize, u64)>,
    used: usize,
    budget: usize,
    tick: u64,
}

impl<S: Clone> StateCache<S> {
    pub fn new(budget_bytes: usize) -> Self {
        StateCache { map: HashMap::new(), used: 0, budget: budget_bytes, tick: 0 }
    }

    pub fn get(&mut self, key: &Key) -> Option<S> {
        self.tick += 1;
        let tick = self.tick;
        self.map.get_mut(key).map(|e| {
            e.2 = tick;
            e.0.clone()
        })
    }

    /// States larger than the whole budget are not cached.
    pub fn insert(&mut self, key: Key, state: S, bytes: usize) {
        if bytes > self.budget {
            return;
        }
        if let Some((_, old, _)) = self.map.remove(&key) {
            self.used -= old;
        }
        while self.used + bytes > self.budget {
            let oldest = *self.map.iter().min_by_key(|(_, e)| e.2).expect("used > 0 implies entries").0;
            self.used -= self.map.remove(&oldest).expect("key from map").1;
        }
        self.tick += 1;
        self.used += bytes;
        self.map.insert(key, (state, bytes, self.tick));
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_least_recently_used_by_bytes() {
        let mut c = StateCache::new(20);
        c.insert([1; 32], "a", 10);
        c.insert([2; 32], "b", 10);
        assert_eq!(c.get(&[1; 32]), Some("a"));
        c.insert([3; 32], "c", 10);
        assert_eq!(c.get(&[2; 32]), None);
        assert_eq!(c.get(&[1; 32]), Some("a"));
        assert_eq!(c.get(&[3; 32]), Some("c"));
        c.insert([4; 32], "huge", 21);
        assert_eq!(c.get(&[4; 32]), None);
        assert_eq!(c.len(), 2);
    }
}

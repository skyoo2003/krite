"""Quality, calibration, selectivity, and invariance metrics (benchmark-spec §7-10).

A prediction is a probability vector over the question's candidates in request order
(noul: [P(true), P(false)]) plus the gold candidate index. Vectors may differ in length.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EPS = 1e-12
BINS = 15


def normalize(p: Sequence[float]) -> tuple[np.ndarray, float]:
    """Clip, renormalize, and return the original |sum - 1| so violations can be reported."""
    a = np.asarray(p, dtype=np.float64)
    dev = abs(float(a.sum()) - 1.0)
    a = np.clip(a, EPS, 1.0)
    return a / a.sum(), dev


def argmax(p: Sequence[float], names: Sequence[str]) -> int:
    """Argmax with ties broken by ascending codepoint order of names (Protocol v1 §3)."""
    a = np.asarray(p)
    top = np.flatnonzero(a == a.max())
    return int(min(top, key=lambda i: names[i]))


# --- quality -------------------------------------------------------------------------------------


def accuracy(pred: Sequence[int], gold: Sequence[int]) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(gold)))


def macro_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    pred, gold = np.asarray(pred), np.asarray(gold)
    f1s = []
    for label in np.unique(gold):
        tp = np.sum((pred == label) & (gold == label))
        fp = np.sum((pred == label) & (gold != label))
        fn = np.sum((pred != label) & (gold == label))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def auroc(score: Sequence[float], y: Sequence[bool]) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    s, y = np.asarray(score, dtype=np.float64), np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def qwk(pred: Sequence[int], gold: Sequence[int], k: int) -> float:
    """Quadratic weighted kappa over levels 0..k-1."""
    obs = np.zeros((k, k))
    for p, g in zip(pred, gold, strict=True):
        obs[g, p] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def rps(probs: Sequence[np.ndarray], gold: Sequence[int]) -> float:
    """Ranked probability score (discrete CRPS), normalized by k - 1."""
    vals = []
    for p, g in zip(probs, gold, strict=True):
        cdf = np.cumsum(p)
        target = (np.arange(len(p)) >= g).astype(float)
        vals.append(np.sum((cdf - target) ** 2) / (len(p) - 1))
    return float(np.mean(vals))


# --- calibration --------------------------------------------------------------------------------


def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = BINS) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    c, y = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(c[m].mean() - y[m].mean())
    return float(total / len(c))


def adaptive_ece(conf: Sequence[float], correct: Sequence[bool], bins: int = BINS) -> float:
    """ECE with equal-mass bins."""
    c, y = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    order = np.argsort(c, kind="mergesort")
    total = 0.0
    for chunk in np.array_split(order, min(bins, len(c))):
        total += len(chunk) * abs(c[chunk].mean() - y[chunk].mean())
    return float(total / len(c))


def nll(probs: Sequence[np.ndarray], gold: Sequence[int]) -> float:
    return float(-np.mean([math.log(max(p[g], EPS)) for p, g in zip(probs, gold, strict=True)]))


def brier(probs: Sequence[np.ndarray], gold: Sequence[int]) -> float:
    """Multi-class Brier: sum over candidates of squared error, averaged over cases."""
    vals = []
    for p, g in zip(probs, gold, strict=True):
        t = np.zeros(len(p))
        t[g] = 1.0
        vals.append(np.sum((p - t) ** 2))
    return float(np.mean(vals))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, EPS, 1.0)) / t
    z = np.exp(z - z.max())
    return z / z.sum()


def fit_temperature(probs: Sequence[np.ndarray], gold: Sequence[int], lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature minimizing NLL, by golden-section search on log T."""

    def loss(log_t: float) -> float:
        return nll([apply_temperature(p, math.exp(log_t)) for p in probs], gold)

    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)


def calib_split(n: int, seed: int = 13) -> np.ndarray:
    """Boolean mask: True = calibration-fit half, False = evaluation half."""
    mask = np.zeros(n, dtype=bool)
    mask[np.random.default_rng(seed).permutation(n)[: n // 2]] = True
    return mask


# --- selectivity --------------------------------------------------------------------------------


def aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve, sorted by descending confidence."""
    c, y = np.asarray(conf), np.asarray(correct, dtype=np.float64)
    order = np.argsort(-c, kind="mergesort")
    risk = np.cumsum(1 - y[order]) / np.arange(1, len(c) + 1)
    return float(risk.mean())


# --- invariance ---------------------------------------------------------------------------------


def flip_rate(orig: Sequence[str], perm: Sequence[str]) -> float:
    """Fraction of (question, permutation) pairs whose argmax name differs from the original order."""
    return float(np.mean(np.asarray(orig) != np.asarray(perm)))


def max_prob_dev(a: dict[str, float], b: dict[str, float]) -> float:
    """Max absolute probability difference for the same candidate names."""
    return max(abs(a[k] - b[k]) for k in a)


# --- suite summary ------------------------------------------------------------------------------


def calibration_block(probs: list[np.ndarray], gold: list[int], names: list[list[str]]) -> dict:
    tops = [argmax(p, n) for p, n in zip(probs, names, strict=True)]
    conf = [float(p[t]) for p, t in zip(probs, tops, strict=True)]
    correct = [t == g for t, g in zip(tops, gold, strict=True)]
    return {
        "ece": ece(conf, correct),
        "adaptive_ece": adaptive_ece(conf, correct),
        "nll": nll(probs, gold),
        "brier": brier(probs, gold),
        "aurc": aurc(conf, correct),
    }


def summarize(qtype: str, preds: list[dict]) -> dict:
    """Quality and raw-calibration metrics for one suite. preds: [{probs, gold, names}] for successful cases."""
    probs = [normalize(p["probs"])[0] for p in preds]
    gold = [p["gold"] for p in preds]
    names = [p["names"] for p in preds]
    tops = [argmax(p, n) for p, n in zip(probs, names, strict=True)]
    out: dict = {"n_scored": len(preds), "max_sum_dev": max(normalize(p["probs"])[1] for p in preds)}
    out["accuracy"] = accuracy(tops, gold)
    if qtype == "choice":
        out["macro_f1"] = macro_f1(
            [n[t] for n, t in zip(names, tops, strict=True)], [n[g] for n, g in zip(names, gold, strict=True)]
        )
    elif qtype == "noul":
        y = [g == 0 for g in gold]
        out["auroc"] = auroc([p[0] for p in probs], y)
        out["brier_binary"] = float(np.mean([(p[0] - yy) ** 2 for p, yy in zip(probs, y, strict=True)]))
    else:
        k = len(probs[0])
        out["mae"] = float(np.mean([abs(np.dot(np.arange(k), p) - g) for p, g in zip(probs, gold, strict=True)]))
        out["qwk"] = qwk(tops, gold, k)
        out["rps"] = rps(probs, gold)
    out["raw"] = calibration_block(probs, gold, names)
    return out


def pooled_calibration(suite_preds: dict[str, tuple[list[dict], int]]) -> dict[str, dict]:
    """One temperature per calibration bucket, shared by every suite in it (ARCHITECTURE.md Glossary).

    suite_preds: {suite_id: (preds, n_cases)} for one calibrator (one engine model); preds are
    [{probs, gold, names, bucket, index}] for successful cases, `index` being the case position in the
    suite. Each suite is split in halves keyed on case index; the fit halves of all suites in a bucket
    are pooled to fit the temperature, and each suite is scored on its own evaluation half.
    """
    halves = {}
    for sid, (preds, n_cases) in suite_preds.items():
        split = calib_split(n_cases)
        halves[sid] = [p for p in preds if split[p["index"]]], [p for p in preds if not split[p["index"]]]
    temps = {}
    for b in sorted({p["bucket"] for preds, _ in suite_preds.values() for p in preds}):
        fit = [p for fit_half, _ in halves.values() for p in fit_half if p["bucket"] == b]
        probs = [normalize(p["probs"])[0] for p in fit]
        temps[b] = fit_temperature(probs, [p["gold"] for p in fit]) if fit else 1.0
    out = {}
    for sid, (_, ev) in halves.items():
        raw = [normalize(p["probs"])[0] for p in ev]
        scaled = [apply_temperature(r, temps[p["bucket"]]) for r, p in zip(raw, ev, strict=True)]
        gold, names = [p["gold"] for p in ev], [p["names"] for p in ev]
        buckets = sorted({p["bucket"] for p in suite_preds[sid][0]})
        out[sid] = {
            "temperature": {b: temps[b] for b in buckets},
            "n_fit_pooled": {b: sum(p["bucket"] == b for h, _ in halves.values() for p in h) for b in buckets},
            "raw_eval_half": calibration_block(raw, gold, names),
            "scaled_eval_half": calibration_block(scaled, gold, names),
        }
    return out

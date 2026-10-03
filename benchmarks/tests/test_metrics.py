import math

import numpy as np
import pytest

from krite_bench import metrics as m
from krite_bench import runners


def test_ece_calibrated_is_near_zero():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 20000)
    correct = rng.uniform(0, 1, 20000) < conf
    assert m.ece(conf, correct) < 0.02
    assert m.adaptive_ece(conf, correct) < 0.02


def test_ece_hand_example():
    # One bin [0.8, 0.867): mean conf 0.8, accuracy 0.5 -> |0.8 - 0.5| = 0.3
    assert math.isclose(m.ece([0.8, 0.8, 0.8, 0.8], [True, False, True, False]), 0.3)


def test_auroc_perfect_reversed_ties():
    y = [True, True, False, False]
    assert m.auroc([0.9, 0.8, 0.2, 0.1], y) == 1.0
    assert m.auroc([0.1, 0.2, 0.8, 0.9], y) == 0.0
    assert m.auroc([0.5, 0.5, 0.5, 0.5], y) == 0.5


def test_qwk():
    assert m.qwk([0, 1, 2, 3], [0, 1, 2, 3], 4) == 1.0
    assert m.qwk([1, 2, 3, 3], [0, 1, 2, 3], 4) < 1.0


def test_rps_one_hot_correct_is_zero():
    assert m.rps([np.array([0, 0, 1.0, 0])], [2]) == 0.0


def test_fit_temperature_recovers_t2():
    rng = np.random.default_rng(1)
    probs, gold = [], []
    for _ in range(4000):
        logits = rng.normal(0, 3, 4)
        true_p = np.exp(logits / 2) / np.exp(logits / 2).sum()  # data generated at T = 2
        probs.append(np.exp(logits) / np.exp(logits).sum())  # model reports T = 1
        gold.append(int(rng.choice(4, p=true_p)))
    assert abs(m.fit_temperature(probs, gold) - 2.0) < 0.1


def test_zero_probabilities_give_finite_nll():
    p, dev = m.normalize([1.0, 0.0])
    assert dev == 0.0
    assert math.isfinite(m.nll([p], [1]))


def test_argmax_tie_break_by_codepoint():
    assert m.argmax([0.5, 0.5], ["b", "a"]) == 1


def test_summarize_choice_has_raw_block():
    preds = [{"probs": [0.7, 0.3], "gold": i % 2, "names": ["a", "b"]} for i in range(20)]
    out = m.summarize("choice", preds)
    assert out["accuracy"] == 0.5
    assert set(out["raw"]) == {"ece", "adaptive_ece", "nll", "brier", "aurc"}


def _preds(n, conf, bucket="choice/2"):
    return [
        {"probs": [conf, 1 - conf], "gold": i % 2, "names": ["a", "b"], "bucket": bucket, "index": i} for i in range(n)
    ]


def test_pooled_calibration_shares_one_temperature_per_bucket():
    # Two suites in one bucket with opposite miscalibration get the same, pooled temperature.
    out = m.pooled_calibration(
        {"s1": (_preds(20, 0.7), 20), "s2": (_preds(20, 0.9), 20), "s3": (_preds(20, 0.6, "noul"), 20)}
    )
    assert out["s1"]["temperature"] == out["s2"]["temperature"]
    assert set(out["s3"]["temperature"]) == {"noul"}
    assert out["s1"]["n_fit_pooled"] == {"choice/2": 20}
    # A 50%-accurate 0.7 confidence: scaling must reduce ECE.
    assert out["s1"]["scaled_eval_half"]["ece"] < out["s1"]["raw_eval_half"]["ece"]


def test_calibration_split_is_keyed_on_case_index():
    preds = _preds(20, 0.7)
    fit_case = int(np.flatnonzero(m.calib_split(20))[0])
    full = m.pooled_calibration({"s": (preds, 20)})["s"]
    without = m.pooled_calibration({"s": ([p for p in preds if p["index"] != fit_case], 20)})["s"]
    # Losing a fit-half case must not reshuffle the evaluation half.
    assert without["raw_eval_half"] == full["raw_eval_half"]


def _encoder_out(backend: str, probe_shift: float) -> dict:
    probe = [[0.1 + probe_shift] * 8, [0.2] * 8, [0.3] * 8]
    return {
        "backend": backend,
        "precision": "fp32",
        "results": {"64": {"times_ms": [1.0, 2.0, 3.0], "truncated": False, "probe": probe}},
    }


def test_encoder_rows_are_model_layer_and_parity_checked():
    rows = runners.encoder_rows(
        {"krite": _encoder_out("candle-metal", 5e-5), "classifier": _encoder_out("torch-mps", 0.0)}, "env.json"
    )
    by_engine = {r["engine"]: r for r in rows}
    k = by_engine["krite"]
    assert (k["layer"], k["state_tokens"], k["questions"], k["decisions_per_sec"]) == ("model", 64, 0, None)
    assert (k["backend"], k["p50_ms"], k["component"]) == ("candle-metal", 2.0, "encoder")
    assert math.isclose(k["max_abs_diff_vs_torch"], 5e-5, rel_tol=1e-6)
    assert "max_abs_diff_vs_torch" not in by_engine["classifier"]
    with pytest.raises(RuntimeError, match="differs from torch"):
        runners.encoder_rows(
            {"krite": _encoder_out("candle-metal", 1e-3), "classifier": _encoder_out("torch-mps", 0.0)}, "e"
        )

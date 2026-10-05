"""Calibration temperatures picked for the exported model directory."""

import json

from krite_train.export import temperatures


def test_latest_row_of_the_engine_wins(tmp_path):
    rows = [
        {"engine": "krite", "timestamp": "2026-10-05T02:00:00+00:00", "temperature": {"choice/4": 1.5}},
        {"engine": "krite", "timestamp": "2026-10-05T01:00:00+00:00", "temperature": {"choice/4": 9.0, "noul": 2.0}},
        {"engine": "other", "timestamp": "2026-10-05T03:00:00+00:00", "temperature": {"choice/4": 7.0}},
    ]
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert temperatures(path, "krite") == {"choice/4": 1.5, "noul": 2.0}
    assert temperatures(path, "missing") == {}
    assert temperatures(None, "krite") == {}


def test_calibration_engine_must_serve_raw_probabilities(tmp_path):
    from krite_train.export import require_raw

    engines = tmp_path / "engines.toml"
    engines.write_text(
        '[krite]\nstart = "krite serve --port 8110 --model m"\n'
        '[krite-raw]\nstart = "krite serve --port 8138 --model m --raw"\n'
    )
    require_raw(engines, "krite-raw")
    for engine in ("krite", "missing"):
        try:
            require_raw(engines, engine)
        except SystemExit as e:
            assert engine in str(e)
        else:
            raise AssertionError(f"{engine} accepted")


def test_failed_export_keeps_the_previous_model(tmp_path):
    from krite_train.export import publish

    out = tmp_path / "candle"
    out.mkdir()
    (out / "model.safetensors").write_text("old")

    def fail(staging):
        (staging / "model.safetensors").write_text("new")
        raise RuntimeError("download failed")

    try:
        publish(out, fail)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failure swallowed")
    assert (out / "model.safetensors").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candle"]

    publish(out, lambda staging: (staging / "model.safetensors").write_text("new"))
    assert (out / "model.safetensors").read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candle"]

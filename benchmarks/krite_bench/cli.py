"""krite-bench command line."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import jsonschema

from . import data, env, runners


def _suites(a) -> list[str]:
    return a.suites.split(",") if a.suites else runners.suite_ids(a.langs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="krite-bench", description=__doc__)
    ap.add_argument("--out", type=Path, default=runners.RESULTS_DIR, help="results directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="build suites into benchmarks/data (seed 13)")
    p.add_argument("--only", help="comma-separated suite ids")
    p.add_argument("--verify", action="store_true", help="rebuild and compare against manifest.json")

    sub.add_parser("env", help="write a §1 environment record")

    for name in ("quality", "invariance"):
        p = sub.add_parser(name)
        p.add_argument("--engine", required=True)
        p.add_argument("--suites", help="comma-separated suite ids (default: by --langs)")
        p.add_argument("--langs", choices=["all", "en", "non-en"], default="all")
        if name == "quality":
            p.add_argument("--memory", action="store_true", help="sample peak memory during the run")
        else:
            p.add_argument("--n", type=int, default=100)

    p = sub.add_parser("calibrate", help="refit pooled per-bucket temperatures from the engine's pred files")
    p.add_argument("--engine", required=True)

    p = sub.add_parser("interference")
    p.add_argument("--engine", required=True)

    p = sub.add_parser("latency")
    p.add_argument("--engines", required=True, help="comma-separated; interleaved per call in burst mode")
    p.add_argument("--mode", choices=["burst", "sustained"], default="burst")
    p.add_argument("--cells", choices=sorted(runners.CELLS), default="primary")
    p.add_argument("--cache", choices=["cold", "warm", "startup"], default="cold")
    p.add_argument("--seconds", type=int, default=600)

    p = sub.add_parser("report", help="render docs/baselines.md tables")
    p.add_argument("--dest", type=Path, default=data.ROOT / "docs" / "baselines.md")

    a = ap.parse_args(argv)
    if a.cmd == "data":
        only = a.only.split(",") if a.only else None
        if a.verify:
            old = json.loads((data.DATA_DIR / "manifest.json").read_text())["suites"]
            with tempfile.TemporaryDirectory() as tmp:
                new = data.build_all(Path(tmp), only or list(old))["suites"]
            bad = [s for s in new if new[s]["cases_sha256"] != old[s]["cases_sha256"]]
            if bad:
                raise SystemExit(f"FAIL: rebuild differs for {bad}")
            validator = jsonschema.Draft202012Validator(
                json.loads((data.SCHEMA_DIR / "request.schema.json").read_text())
            )
            invalid = [c["case_id"] for s in new for c in data.load_suite(s) if not validator.is_valid(c["request"])]
            if invalid:
                raise SystemExit(f"FAIL: {len(invalid)} requests violate request.schema.json, e.g. {invalid[:3]}")
            print(f"data ok ({len(new)} suites reproduce, all requests schema-valid)")
        else:
            m = data.build_all(only=only)
            print(f"built {len(m['suites'])} suites into {data.DATA_DIR}")
    elif a.cmd == "env":
        print(env.record(a.out))
    elif a.cmd == "quality":
        runners.run_quality(a.engine, _suites(a), a.out, a.memory)
    elif a.cmd == "invariance":
        runners.run_invariance(a.engine, _suites(a), a.n, a.out)
    elif a.cmd == "calibrate":
        runners.run_calibration(a.engine, a.out)
    elif a.cmd == "interference":
        runners.run_interference(a.engine, out=a.out)
    elif a.cmd == "latency":
        engines = a.engines.split(",")
        if a.cache == "startup":
            for e in engines:
                runners.run_startup(e, a.out)
        elif a.mode == "sustained":
            for e in engines:
                runners.run_sustained(e, a.seconds, a.out)
        else:
            runners.run_burst(engines, a.cells, a.cache, a.out)
    elif a.cmd == "report":
        from . import report

        report.write(a.out, a.dest)
        print(a.dest)


if __name__ == "__main__":
    main()

"""Environment record (benchmark-spec §1)."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from .data import ROOT

VENVS = ROOT / "benchmarks" / "baselines" / ".venvs"


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def commit() -> str:
    return sh("git", "-C", str(ROOT), "rev-parse", "HEAD")


def therm() -> str:
    return sh("pmset", "-g", "therm")


def power_source() -> str:
    """`AC Power` or `Battery Power`, from `pmset -g batt`."""
    first = sh("pmset", "-g", "batt").splitlines()[:1]
    return first[0].split("'")[1] if first and "'" in first[0] else "unknown"


def record(out_dir: Path, label: str = "") -> Path:
    pmset = sh("pmset", "-g")
    engines = {}
    for f in sorted(VENVS.glob("*/INSTALLED.json")):
        engines[f.parent.name] = json.loads(f.read_text())
    rec = {
        "timestamp": now(),
        "label": label,
        "hw_model": sh("sysctl", "-n", "hw.model"),
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0),
        "macos": sh("sw_vers", "-productVersion"),
        "power_source": power_source(),
        "low_power_mode": next((ln.split()[-1] for ln in pmset.splitlines() if "lowpowermode" in ln), "unknown"),
        "pmset_therm": therm(),
        "ambient_note": "",
        "commit": commit(),
        "engines": engines,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"env-{rec['timestamp'].replace(':', '')}.json"
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return path


def latest(out_dir: Path) -> str:
    files = sorted(out_dir.glob("env-*.json"))
    if not files:
        raise FileNotFoundError(f"no env record in {out_dir}; run `krite-bench env` first")
    f = files[-1]
    return str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f)

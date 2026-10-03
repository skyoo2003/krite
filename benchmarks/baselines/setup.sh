#!/usr/bin/env bash
# Creates the isolated environment for one baseline engine (see engines.toml).
set -euo pipefail
cd "$(dirname "$0")"
engine="${1:?usage: setup.sh <engine>}"
cfg() { python3 -c 'import sys, tomllib; c = tomllib.load(open("engines.toml", "rb"))[sys.argv[1]]; v = c.get(sys.argv[2], ""); print(v if not isinstance(v, bool) else str(v).lower())' "$engine" "$1"; }

repo=$(cfg repo)
if [[ -n "$repo" && ! -d "vendor/$(basename "$repo")" ]]; then
  mkdir -p vendor
  git clone -q "$repo" "vendor/$(basename "$repo")"
fi

venv=".venvs/$engine"
mkdir -p "$venv"
if [[ "$(cfg venv)" == "true" ]]; then
  from=$(cfg venv_from)
  if [[ -n "$from" ]]; then
    [[ -d ".venvs/$from/bin" ]] || { echo "FAIL: set up $from first" >&2; exit 1; }
    echo "$engine shares the $from environment"
    cp ".venvs/$from/INSTALLED.json" "$venv/INSTALLED.json"
    exit 0
  fi
  uv venv -q --python 3.11 "$venv"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
fi
install=$(cfg install)
[[ -n "$install" ]] && bash -c "$install"

freeze=$(if [[ "$(cfg venv)" == "true" ]]; then uv pip freeze; elif [[ -n "$repo" ]]; then (cd "vendor/$(basename "$repo")" && uv pip freeze); fi || true)
python3 - "$engine" "$venv/INSTALLED.json" "$freeze" "$repo" <<'PY'
import hashlib, json, subprocess, sys, datetime
engine, path, freeze, repo = sys.argv[1:]
commit = ""
if repo:
    d = "vendor/" + repo.rstrip("/").rsplit("/", 1)[-1]
    commit = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"engine": engine, "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
           "packages": freeze.splitlines(), "repo_commit": commit,
           "installed_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")},
          open(path, "w"), indent=2)
PY
echo "installed $engine"

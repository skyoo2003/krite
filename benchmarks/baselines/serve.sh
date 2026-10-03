#!/usr/bin/env bash
# Starts one engine on its port in the background and waits until POST /v1/systemone returns 200.
set -euo pipefail
cd "$(dirname "$0")"
engine="${1:?usage: serve.sh <engine>}"
timeout_s="${SERVE_TIMEOUT:-600}"
eval "$(python3 - "$engine" <<'PY'
import shlex, sys, tomllib
c = tomllib.load(open("engines.toml", "rb"))[sys.argv[1]]
envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in c.get("env", {}).items())
for k, v in {"port": c["port"], "venv": str(c.get("venv", False)).lower(), "from": c.get("venv_from", ""),
             "start": c["start"], "envs": envs, "probe_model": c.get("request_model", "").format(dataset="boolq")}.items():  # probe is a noul question
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

if lsof -ti "tcp:$port" -sTCP:LISTEN >/dev/null; then
  echo "FAIL: port $port already in use" >&2; exit 1
fi
mkdir -p ".venvs/$engine"
log=".venvs/$engine/serve.log"
(
  if [[ "$venv" == "true" ]]; then
    [[ -z "$from" ]] && from="$engine"
    # shellcheck disable=SC1090
    source ".venvs/$from/bin/activate"
  fi
  [[ -n "$envs" ]] && eval "export $envs"
  exec bash -c "$start"
) >"$log" 2>&1 &
engine_pid=$!

probe='{"state":"ping","questions":{"q":{"type":"noul","instructions":"Is this a test?"}}'
# Engines that route on the request `model` (request_model in engines.toml) need it in the probe too.
if [[ -n "$probe_model" ]]; then probe="${probe},\"model\":\"${probe_model}\"}"; else probe="${probe}}"; fi
deadline=$((SECONDS + timeout_s))
while ((SECONDS < deadline)); do
  code=$(curl -s --max-time 10 -o /dev/null -w '%{http_code}' -H 'content-type: application/json' -d "$probe" "http://127.0.0.1:$port/v1/systemone" || true)
  if [[ "$code" == "200" ]]; then
    lsof -ti "tcp:$port" -sTCP:LISTEN | head -1
    exit 0
  fi
  sleep 0.2
done
# Do not leave a half-started engine holding the port and unified memory.
pkill -P "$engine_pid" 2>/dev/null || true
kill "$engine_pid" 2>/dev/null || true
lsof -ti "tcp:$port" -sTCP:LISTEN | xargs kill 2>/dev/null || true
echo "FAIL: $engine did not answer 200 on port $port within ${timeout_s}s; see $log" >&2
exit 1

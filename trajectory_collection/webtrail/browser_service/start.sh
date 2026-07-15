#!/usr/bin/env bash
# Launch one or more browser-service workers on consecutive ports.
#
#   WORKERS=8 BASE_PORT=9300 ./start.sh
#
# Ports are cleaned before starting so a crashed previous run never blocks
# the next one. Logs go to logs/service-<port>.log.

set -euo pipefail

cd "$(dirname "$0")"

WORKERS="${WORKERS:-4}"
BASE_PORT="${BASE_PORT:-9300}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

# Find PIDs holding a TCP port, using whatever tool the host has. Port cleanup
# is best-effort: a fresh host with none of these simply skips it.
port_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$port" 2>/dev/null
  elif command -v fuser >/dev/null 2>&1; then
    fuser "$port"/tcp 2>/dev/null
  elif command -v ss >/dev/null 2>&1; then
    ss -tlnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2
  fi
}

for ((i = 0; i < WORKERS; i++)); do
  port=$((BASE_PORT + i))
  pids=$(port_pids "$port" || true)
  if [[ -n "$pids" ]]; then
    echo "port $port busy (pids: $pids), killing"
    kill -9 $pids 2>/dev/null || true
  fi
done

sleep 0.5

pids=()
for ((i = 0; i < WORKERS; i++)); do
  port=$((BASE_PORT + i))
  node src/server.js "$port" >>"$LOG_DIR/service-$port.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "worker $i -> port $port (pid $pid)"
done

trap 'echo "stopping workers"; kill "${pids[@]}" 2>/dev/null || true' INT TERM

# wait for readiness
for ((i = 0; i < WORKERS; i++)); do
  port=$((BASE_PORT + i))
  for _ in $(seq 1 50); do
    if curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
      echo "worker on :$port ready"
      break
    fi
    sleep 0.2
  done
done

echo "all workers started; Ctrl-C to stop"
wait

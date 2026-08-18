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
RECLAIM_PORTS="${RECLAIM_PORTS:-1}"
PID_FILE="${PID_FILE:-}"

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
    if [[ "$RECLAIM_PORTS" == "1" ]]; then
      echo "port $port busy (pids: $pids), reclaiming"
      kill $pids 2>/dev/null || true
      sleep 0.5
      remaining=$(port_pids "$port" || true)
      [[ -z "$remaining" ]] || kill -9 $remaining 2>/dev/null || true
    else
      echo "port $port is already owned by pid(s): $pids" >&2
      exit 1
    fi
  fi
done

sleep 0.5

pids=()
cleaned=0
cleanup() {
  [[ "$cleaned" -eq 0 ]] || return 0
  cleaned=1
  trap - EXIT INT TERM
  echo "stopping browser workers: ${pids[*]:-none}"
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      alive=()
      for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive+=("$pid")
      done
      [[ ${#alive[@]} -gt 0 ]] || break
      sleep 0.2
    done
    [[ ${#alive[@]} -eq 0 ]] || kill -9 "${alive[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  [[ -z "$PID_FILE" ]] || rm -f -- "$PID_FILE"
}
trap cleanup EXIT INT TERM

for ((i = 0; i < WORKERS; i++)); do
  port=$((BASE_PORT + i))
  node src/server.js "$port" >>"$LOG_DIR/service-$port.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "worker $i -> port $port (pid $pid)"
done
if [[ -n "$PID_FILE" ]]; then
  printf '%s\n' "${pids[@]}" > "$PID_FILE"
fi

# wait for readiness
for ((i = 0; i < WORKERS; i++)); do
  port=$((BASE_PORT + i))
  ready=0
  for _ in $(seq 1 50); do
    if curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
      echo "worker on :$port ready"
      ready=1
      break
    fi
    sleep 0.2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "worker on :$port failed readiness" >&2
    exit 1
  fi
done

echo "all workers started; Ctrl-C to stop"
wait

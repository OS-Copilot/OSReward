#!/bin/bash
# Gracefully shut down the Windows VM, then stop the container.
# This flushes installed software / disk changes into vm/storage/.
#
# Usage:
#   ./stop-desktop.sh
#   ./stop-desktop.sh --container-name winarena

set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

container_name="winarena"
wait_seconds=180

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-name)
            container_name="$2"
            shift 2
            ;;
        --wait-seconds)
            wait_seconds="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--container-name NAME] [--wait-seconds N]"
            echo "Sends Windows shutdown via the arena server, waits for disk flush, then docker stop."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
    echo "Container $container_name is not running."
    # Still try to remove exited leftover if any
    if docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
        docker rm -f "$container_name" 2>/dev/null || true
    fi
    exit 0
fi

echo "Requesting graceful Windows shutdown inside $container_name ..."
code=$(docker exec "$container_name" curl --write-out '%{http_code}' --silent --output /dev/null \
  -X POST http://20.20.20.21:5000/shutdown || echo "000")

if [ "$code" = "200" ]; then
    echo "Shutdown accepted (HTTP 200). Waiting up to ${wait_seconds}s for disk flush..."
    sleep "$wait_seconds"
else
    echo "Warning: shutdown endpoint returned HTTP ${code}."
    echo "If you just closed Windows from the desktop UI, waiting ${wait_seconds}s anyway..."
    sleep "$wait_seconds"
fi

echo "Stopping container $container_name ..."
docker stop "$container_name" || true
echo "Done. Installed software should persist in:"
echo "  $SCRIPT_DIR/../src/win-arena-container/vm/storage/"
echo "Next start with ./start-desktop.sh or collection will reuse this image."

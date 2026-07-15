#!/bin/bash
# Start WinArena desktop only: no evaluation, no collection.
# Use the browser (http://localhost:8006) or RDP to install software.
# Disk state is persisted under src/win-arena-container/vm/storage/.
#
# Usage:
#   ./start-desktop.sh
#   ./start-desktop.sh --browser-port 8006 --rdp-port 3390

set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

browser_port=8006
rdp_port=3390
container_name="winarena"
ram_size=8G
cpu_cores=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --browser-port)
            browser_port="$2"
            shift 2
            ;;
        --rdp-port)
            rdp_port="$2"
            shift 2
            ;;
        --container-name)
            container_name="$2"
            shift 2
            ;;
        --ram-size)
            ram_size="$2"
            shift 2
            ;;
        --cpu-cores)
            cpu_cores="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--browser-port N] [--rdp-port N] [--container-name NAME]"
            echo "Starts Windows VM without collection/evaluation."
            echo "Open http://localhost:${browser_port} to use the desktop."
            echo "When finished installing software, run: ./stop-desktop.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Stop existing container if present
if docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
    echo "Container $container_name already exists. Stopping it first..."
    "$SCRIPT_DIR/stop-desktop.sh" --container-name "$container_name" || true
    sleep 2
fi

echo "Starting desktop-only WinArena (no collection)..."
echo "  Browser: http://localhost:${browser_port}"
echo "  RDP:     localhost:${rdp_port}"
echo "  Storage: ../src/win-arena-container/vm/storage (persisted)"
echo ""
echo "After Windows is ready, install software in the VM."
echo "Then run: ./stop-desktop.sh   # graceful shutdown so installs are saved"
echo ""

./run-local.sh \
  --skip-build true \
  --start-client false \
  --prepare-image false \
  --container-name "$container_name" \
  --browser-port "$browser_port" \
  --rdp-port "$rdp_port" \
  --ram-size "$ram_size" \
  --cpu-cores "$cpu_cores" \
  --mount-vm-storage true

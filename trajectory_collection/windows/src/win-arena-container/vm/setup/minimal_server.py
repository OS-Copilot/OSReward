"""Minimal WAA server for prepare-image: /probe and /shutdown only."""
from flask import Flask
import argparse
import os
import subprocess
import threading
import time

app = Flask(__name__)


@app.route("/probe", methods=["GET"])
def probe():
    return "OK", 200


@app.route("/shutdown", methods=["POST"])
def shutdown():
    def _do():
        time.sleep(1)
        # Graceful shutdown of Windows
        subprocess.Popen(
            ["shutdown", "/s", "/t", "5", "/f"],
            shell=False,
        )

    threading.Thread(target=_do, daemon=True).start()
    return "Shutting down", 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    # Bind all interfaces so host.lan can reach the guest
    app.run(host="0.0.0.0", port=args.port, threaded=True)

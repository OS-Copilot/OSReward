#!/usr/bin/env python3
"""Open Win+R and run a command via QEMU monitor keystrokes."""
import socket
import time
import string
import sys

HOST, PORT = "127.0.0.1", 7100

shift_map = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8",
    "(": "9", ")": "0", "_": "minus", "+": "equal", "{": "bracket_left", "}": "bracket_right",
    "|": "backslash", ":": "semicolon", '"': "apostrophe", "<": "comma", ">": "dot",
    "?": "slash", "~": "grave_accent",
}
base = {
    "-": "minus", "=": "equal", "[": "bracket_left", "]": "bracket_right", "\\": "backslash",
    ";": "semicolon", "'": "apostrophe", ",": "comma", ".": "dot", "/": "slash",
    "`": "grave_accent", " ": "spc", "\n": "ret", "\t": "tab",
}


def mon(cmd, wait=0.08):
    s = socket.create_connection((HOST, PORT), timeout=5)
    try:
        s.recv(4096)
    except Exception:
        pass
    s.sendall((cmd + "\n").encode())
    time.sleep(wait)
    try:
        s.settimeout(0.15)
        s.recv(4096)
    except Exception:
        pass
    s.close()


def send_char(ch):
    if ch in string.ascii_uppercase:
        mon(f"sendkey shift-{ch.lower()}")
        return
    if ch in shift_map:
        mon(f"sendkey shift-{shift_map[ch]}")
        return
    if ch in base:
        mon(f"sendkey {base[ch]}")
        return
    if ch in string.ascii_lowercase or ch in string.digits:
        mon(f"sendkey {ch}")
        return
    print("skip", repr(ch), file=sys.stderr)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else (
        r'powershell -NoProfile -ExecutionPolicy Bypass -Command '
        r'"& \"$env:LOCALAPPDATA\Programs\Python\Python310\python.exe\" '
        r'\\host.lan\Data\minimal_server.py --port 5000"'
    )
    # Win+R
    mon("sendkey meta_l-r", 0.5)
    time.sleep(1.0)
    for ch in cmd:
        send_char(ch)
    time.sleep(0.3)
    mon("sendkey ret", 0.2)
    print("ran via Win+R:", cmd)


if __name__ == "__main__":
    main()

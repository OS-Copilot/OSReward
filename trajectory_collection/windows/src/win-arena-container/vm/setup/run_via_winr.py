#!/usr/bin/env python3
"""Win+R launch a path, then confirm Open File Security Warning with Run."""
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


def mon(cmd, wait=0.12):
    s = socket.create_connection((HOST, PORT), timeout=5)
    try:
        s.recv(4096)
    except Exception:
        pass
    s.sendall((cmd + "\n").encode())
    time.sleep(wait)
    try:
        s.settimeout(0.1)
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


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else r"\\host.lan\Data\run_fix_pywin32.bat"
    mon("sendkey esc", 0.3)
    mon("sendkey esc", 0.3)
    for key in ("meta_l-r", "super_l-r"):
        mon(f"sendkey {key}", 0.2)
    time.sleep(1.0)
    mon("sendkey ctrl-a", 0.2)
    mon("sendkey backspace", 0.2)
    for ch in target:
        send_char(ch)
    time.sleep(0.2)
    mon("sendkey ret", 0.3)
    # Confirm security warning: focus is often on Cancel, move left to Run
    time.sleep(2.0)
    mon("sendkey left", 0.3)
    mon("sendkey ret", 0.3)
    print("launched:", target)


if __name__ == "__main__":
    main()

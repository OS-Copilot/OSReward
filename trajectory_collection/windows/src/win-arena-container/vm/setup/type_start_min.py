#!/usr/bin/env python3
"""Focus terminal and start minimal_server slowly."""
import socket
import time
import string

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
    # Ctrl+C twice to ensure prompt
    mon("sendkey ctrl-c", 0.3)
    mon("sendkey ctrl-c", 0.5)
    mon("sendkey ret", 0.4)
    time.sleep(1)
    # Use cmd start to open a new process so this session stays free
    cmd = r'start "" "\\host.lan\Data\start_min_server.bat"'
    for ch in cmd:
        send_char(ch)
    mon("sendkey ret", 0.3)
    print("done:", cmd)


if __name__ == "__main__":
    main()

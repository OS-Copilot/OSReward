#!/usr/bin/env python3
import socket
import time
import string

HOST, PORT = "127.0.0.1", 7100


def mon(cmd, wait=0.05):
    s = socket.create_connection((HOST, PORT), timeout=5)
    s.recv(4096)
    s.sendall((cmd + "\n").encode())
    time.sleep(wait)
    try:
        s.settimeout(0.2)
        s.recv(4096)
    except Exception:
        pass
    s.close()


shift_map = {
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "minus",
    "+": "equal",
    "{": "bracket_left",
    "}": "bracket_right",
    "|": "backslash",
    ":": "semicolon",
    '"': "apostrophe",
    "<": "comma",
    ">": "dot",
    "?": "slash",
    "~": "grave_accent",
}
base = {
    "-": "minus",
    "=": "equal",
    "[": "bracket_left",
    "]": "bracket_right",
    "\\": "backslash",
    ";": "semicolon",
    "'": "apostrophe",
    ",": "comma",
    ".": "dot",
    "/": "slash",
    "`": "grave_accent",
    " ": "spc",
    "\n": "ret",
    "\t": "tab",
}


def send_char(ch):
    if ch in string.ascii_uppercase:
        mon(f"sendkey shift-{ch.lower()}", 0.03)
        return
    if ch in shift_map:
        mon(f"sendkey shift-{shift_map[ch]}", 0.03)
        return
    if ch in base:
        mon(f"sendkey {base[ch]}", 0.03)
        return
    if ch in string.ascii_lowercase or ch in string.digits:
        mon(f"sendkey {ch}", 0.03)
        return
    print("skip", repr(ch))


def main():
    cmd = r"powershell -ExecutionPolicy Bypass -File \\host.lan\Data\fix_pywin32.ps1"
    mon("sendkey ret", 0.2)
    mon("sendkey ret", 0.2)
    time.sleep(0.5)
    for ch in cmd:
        send_char(ch)
    mon("sendkey ret", 0.2)
    print("typed", cmd)


if __name__ == "__main__":
    main()

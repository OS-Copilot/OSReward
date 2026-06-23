"""Read raw agent runs and normalise heterogeneous trajectory formats.

Two orthogonal axes keep new benchmarks/agents cheap to add:

1. SOURCES — how task files are read:
     ZipSource(zip_path)   OSWorld-style zips (possibly deeply nested)
     DirSource(root)       unpacked dirs (WindowsAgentArena / MacOSArena / OSWorld)
   Both expose ``.tasks() -> [(domain, task_id, handle)]`` and
   ``.read(handle, rel) -> bytes|None`` / ``.read_text(handle, rel) -> str|None``.

2. FORMATS — how one agent's ``traj.jsonl`` encodes a step and where the
   instruction lives. Auto-detected from the first step; registry is ordered.
     kimi        steps have ``natural_language_action`` + pyautogui ``action``;
                 instruction in ``runtime.log``.
     ossymphony  steps carry an ``instruction`` field + pyautogui ``action`` +
                 ``response.plan`` (covers WAA-interngui and macOS-ossymphony).
     claude      ``action`` is an Anthropic tool dict; ``response`` is the
                 thought text; instruction is not in the run (map fallback).

WebArena ships as a merged JSONL instead of per-task dirs and is handled
separately by :func:`webarena_records`.
"""
import ast
import json
import os
import re
import zipfile

_CLICK_RE = re.compile(
    r"pyautogui\.(?:click|doubleClick|rightClick|moveTo|dragTo|mouseDown|mouseUp)"
    r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)")


# ----------------------------- sources -----------------------------
class ZipSource:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        self._names = self.zf.namelist()

    def tasks(self):
        out = []
        for name in self._names:
            if name.endswith("/result.txt"):
                prefix = name[: -len("result.txt")]
                parts = prefix.rstrip("/").split("/")
                if len(parts) >= 2:
                    out.append((parts[-2], parts[-1], prefix))
        return sorted(out)

    def read(self, handle, rel):
        try:
            return self.zf.read(handle + rel)
        except KeyError:
            return None

    def read_text(self, handle, rel):
        b = self.read(handle, rel)
        return b.decode("utf-8", "replace") if b is not None else None


class DirSource:
    def __init__(self, root):
        self.root = root

    def tasks(self):
        out = []
        for dirpath, _dirs, files in os.walk(self.root):
            if "result.txt" in files and "traj.jsonl" in files:
                parts = dirpath.rstrip(os.sep).split(os.sep)
                if len(parts) >= 2:
                    out.append((parts[-2], parts[-1], dirpath))
        return sorted(out)

    def read(self, handle, rel):
        p = os.path.join(handle, rel)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def read_text(self, handle, rel):
        b = self.read(handle, rel)
        return b.decode("utf-8", "replace") if b is not None else None


# ----------------------------- helpers -----------------------------
def parse_score(text):
    try:
        return float((text or "").strip())
    except (ValueError, TypeError):
        return None


def load_steps(traj_text):
    """Parse a ``traj.jsonl`` body into a list of step dicts (bad lines skipped)."""
    steps = []
    for line in (traj_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            steps.append(json.loads(line))
        except Exception:
            pass
    return steps


def _maybe_literal(v):
    """traj fields are sometimes python-repr strings; try to parse to an object."""
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str) and v.strip()[:1] in "{[":
        try:
            return ast.literal_eval(v.strip())
        except Exception:
            return v
    return v


def click_from_pyautogui(code):
    m = _CLICK_RE.search(code or "")
    return (int(float(m.group(1))), int(float(m.group(2)))) if m else None


def _cap(text, cap=8000):
    """Keep head+tail of a long sub-agent output, dropping the middle."""
    text = str(text)
    if len(text) <= cap:
        return text
    head = int(cap * 0.6)
    return f"{text[:head]}\n...[truncated {len(text) - cap} chars]...\n{text[-(cap - head):]}"


def _instr_from_runtime(read_text, handle):
    rt = read_text(handle, "runtime.log") or ""
    m = re.search(r"Instruction:\s*\n(.*?)\nModel Output:", rt, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"Instruction:\s*(.+)", rt)
    return m.group(1).strip() if m else None


def _describe_anthropic(act):
    """Anthropic computer-use action -> (text, click_xy).

    ``input.coordinate`` is Anthropic's downscaled (1280x720) tool space while the
    screenshots are full-res (1920x1080); the real executed pixels live in
    ``act['command']`` (pyautogui). Parse those; fall back to coordinate*1.5.
    """
    if not isinstance(act, dict):
        return (str(act) if act else ""), None
    inp = act.get("input", {}) if isinstance(act.get("input"), dict) else {}
    a = inp.get("action", act.get("name", ""))
    text = inp.get("text")
    xy = click_from_pyautogui(act.get("command") or "")
    if xy is None:
        coord = inp.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            xy = (int(float(coord[0]) * 1.5), int(float(coord[1]) * 1.5))
    parts = [str(a)]
    if xy is not None:
        parts.append(f"({xy[0]}, {xy[1]})")
    if text:
        parts.append(f'text="{text}"')
    return " ".join(parts), xy


# ----------------------------- formats -----------------------------
def _kimi_step(s, full=False):
    a = s.get("action") or ""
    return (s.get("natural_language_action") or ""), a, click_from_pyautogui(a)


def _oss_step(s, full=False):
    resp = _maybe_literal(s.get("response"))
    th = ""
    if isinstance(resp, dict):
        th = resp.get("plan") or resp.get("refined_instruction") or ""
        if full:  # fold in multi-agent sub-results omitted by `plan`
            for key in ("code_agent_output", "search_agent_output"):
                if resp.get(key):
                    th += f"\n[{key}]\n{_cap(resp[key])}"
    a = s.get("action") or ""
    return th, a, click_from_pyautogui(a)


def _claude_step(s, full=False):
    th = s.get("response") or ""
    if isinstance(th, (dict, list)):
        th = json.dumps(th, ensure_ascii=False)
    action, xy = _describe_anthropic(_maybe_literal(s.get("action")))
    return th, action, xy


FORMATS = {
    "kimi": {
        "detect": lambda s0: "natural_language_action" in s0,
        "instruction": lambda steps, rt, h: _instr_from_runtime(rt, h),
        "step": _kimi_step,
    },
    "ossymphony": {  # also covers WAA-interngui and macOS-ossymphony (same schema)
        "detect": lambda s0: "instruction" in s0,
        "instruction": lambda steps, rt, h: next(
            (s["instruction"] for s in steps if s.get("instruction")), None),
        "step": _oss_step,
    },
    "claude": {  # Anthropic computer-use; instruction not in run -> map fallback
        "detect": lambda s0: True,
        "instruction": lambda steps, rt, h: None,
        "step": _claude_step,
    },
}
_FORMAT_ORDER = ("kimi", "ossymphony", "claude")


def detect_format(steps):
    if not steps:
        return "ossymphony"
    s0 = steps[0]
    for name in _FORMAT_ORDER:
        if FORMATS[name]["detect"](s0):
            return name
    return "claude"


# ----------------------------- webarena -----------------------------
def _wa_action_str(step):
    a = step.get("action")
    try:
        obj = json.loads(a) if isinstance(a, str) else a
        ak = obj.get("action_key", "")
        raw = (obj.get("action_kwargs") or {}).get("raw_action", "")
        return f"{ak}. Tool Call: {raw}" if raw else str(ak)
    except Exception:
        return str(a)


def webarena_records(set_dir, last_n=5):
    """Yield normalised webarena records from a merged-JSONL rollout set.

    Layout::
        set_dir/judged_last5_gemini-3-flash-preview.jsonl
        set_dir/images/<trace_id>/step_<k>_img_0.png

    Each yielded dict: trace_id, instruction, golden_label, history, n_steps,
    domain, and ``read_frames()`` -> list of ``(png_bytes, name, None)`` (web
    actions are element-id based, so there is no click coordinate to mark).
    """
    jsonl = None
    for fn in sorted(os.listdir(set_dir)):
        if fn.endswith(".jsonl"):
            jsonl = os.path.join(set_dir, fn)
            break
    if jsonl is None:
        return
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        traj = r.get("trajectory", [])
        lines = []
        for i, s in enumerate(traj, 1):
            th = (s.get("thought") or "").strip().strip('"').strip("'")
            lines.append(f"Step {i}: Thought: {th} Action: {_wa_action_str(s)}")
        idir = os.path.join(set_dir, "images", r["trace_id"])
        imgs = []
        if os.path.isdir(idir):
            imgs = sorted([fn for fn in os.listdir(idir) if fn.endswith(".png")],
                          key=lambda fn: int(re.search(r"step_(\d+)", fn).group(1))
                          if re.search(r"step_(\d+)", fn) else 0)
        last = imgs[-last_n:] if len(imgs) > last_n else imgs

        def make_reader(idir=idir, last=last):
            def _read():
                out = []
                for fn in last:
                    with open(os.path.join(idir, fn), "rb") as fh:
                        out.append((fh.read(), fn, None))
                return out
            return _read

        ts = r.get("task_success")
        yield {"trace_id": r["trace_id"], "domain": "web",
               "instruction": r.get("goal"), "history": "\n".join(lines),
               "golden_label": "SUCCESS" if ts == 1 else ("FAIL" if ts == 0 else None),
               "n_steps": len(traj), "read_frames": make_reader()}


def _wa_instr_from_text(text):
    m = re.search(r"The user instruction:\s*(.+)", text or "")
    return m.group(1).strip() if m else None


def webarena_judgements_records(set_dir, judge_subdir=None):
    """Yield records from the legacy 'judgements' webarena format.

    Layout::
        set_dir/judgements/last_5_<judge-model>/<trace>__last5__<model>.json
        set_dir/images/<trace_id>/...

    Each per-trace JSON is an already-built judge input: ``messages`` [system,
    user, assistant], ``images`` (relpaths into set_dir/images/...), and a stored
    ``golden_label``. The judge *input* is identical across the last_5_<model>
    subdirs (they differ only in the recorded judge output), so any one is used;
    by default a gemini-3-flash subdir is preferred. The user message is already
    a fully composed prompt, returned verbatim as ``user_text``.
    """
    jdir = os.path.join(set_dir, "judgements")
    subs = sorted(d for d in os.listdir(jdir) if os.path.isdir(os.path.join(jdir, d)))
    if judge_subdir is None:
        subs_pref = [d for d in subs if "gemini-3-flash" in d] or subs
        judge_subdir = subs_pref[0]
    sdir = os.path.join(jdir, judge_subdir)
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
        user = next((m.get("content", "") for m in (r.get("messages") or [])
                     if m.get("role") == "user"), "")
        user_text = "\n".join(ln for ln in user.split("\n") if ln.strip() != "<image>")
        imgs = [os.path.normpath(os.path.join(sdir, rel)) for rel in r.get("images", [])]
        tid = (r.get("metadata") or {}).get("trace_id") or fn.split("__")[0]
        yield {"trace_id": tid, "domain": "web", "golden_label": r.get("golden_label"),
               "instruction": _wa_instr_from_text(user_text), "user_text": user_text,
               "image_paths": imgs}


# ----------------------------- androidworld -----------------------------
def _aw_click_xy_norm(tool_call):
    """(x, y) in 0..1000 for click/long_press actions, else None."""
    try:
        tc = json.loads(tool_call) if isinstance(tool_call, str) else tool_call
        args = (tc or {}).get("arguments", {})
        if args.get("action") in ("click", "long_press"):
            c = args.get("coordinate")
            if isinstance(c, (list, tuple)) and len(c) == 2:
                return float(c[0]), float(c[1])
    except Exception:
        pass
    return None


def _aw_step_line(i, s):
    th = (s.get("thought") or "").strip()
    ac = (s.get("action") or "").strip().strip('"')
    tc = s.get("tool_call") or ""
    try:
        tc_str = json.dumps(json.loads(tc).get("arguments", {}), ensure_ascii=False) if tc else ""
    except Exception:
        tc_str = str(tc)
    return f"Step {i}: Thought: {th} Action: {ac}. Tool Call: {tc_str}."


def androidworld_records(jsonl_path, last_n=5, root=None):
    """Yield normalised AndroidWorld (mobile) records from a merged JSONL.

    ``jsonl_path`` is the normalized one-trajectory-per-line file (e.g.
    ``.../results/merged_normalized_300.jsonl``). Each line carries goal,
    task_success (1/0 gold), save_dir, and trajectory[{screenshot_path, thought,
    action, tool_call}]. ``screenshot_path`` is resolved against ``root`` (default:
    two levels up from the JSONL — the dir holding ``results/`` and the per-agent
    screenshot folders). Click/long_press coordinates are 0..1000 and are carried
    through (3rd tuple slot) so a red mark can be scaled to each image.
    """
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(jsonl_path)))
    for line in open(jsonl_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        traj = r.get("trajectory", [])
        if not traj:
            continue
        lines, shots = [], []
        for i, s in enumerate(traj, 1):
            lines.append(_aw_step_line(i, s))
            sp = s.get("screenshot_path")
            if sp:
                shots.append((sp, _aw_click_xy_norm(s.get("tool_call"))))
        last = shots[-last_n:] if len(shots) > last_n else shots

        def make_reader(last=last):
            def _read():
                out = []
                for sp, xy in last:
                    ap = os.path.join(root, sp)
                    if not os.path.exists(ap):
                        continue
                    with open(ap, "rb") as fh:
                        out.append((fh.read(), os.path.basename(sp), xy))
                return out
            return _read

        ts = r.get("task_success")
        yield {"trace_id": r.get("save_dir") or r.get("task_name"), "domain": "mobile",
               "instruction": r.get("goal"), "history": "\n".join(lines),
               "golden_label": "SUCCESS" if ts == 1 else ("FAIL" if ts == 0 else None),
               "n_steps": len(traj), "read_frames": make_reader()}

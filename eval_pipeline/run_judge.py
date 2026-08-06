#!/usr/bin/env python3
"""
OSReward standalone judge: evaluate GUI-agent trajectories with a VLM through
any OpenAI-compatible API.

Two output modes, selected with --prompt_type:
  binary  ->  Judge: SUCCESS | FAIL                       (OSReward)
  multi   ->  Judge + Alignment + Efficiency sub-scores   (OSReward-Multi)

Each mode loads its canonical system prompt from prompts/ (binary_v1.txt /
multi_v4.txt); --prompt_file overrides both.

Usage
-----
  # Multi output (default) on the bundled example trace
  python run_judge.py --traces example/example_trace.json \
      --models gpt-4o --base_url https://api.openai.com/v1 --api_key $OPENAI_API_KEY

  # Binary output
  python run_judge.py --traces example/example_trace.json \
      --models gpt-4o --prompt_type binary

  # A directory of traces, several judge models, higher concurrency
  python run_judge.py --traces /path/to/traces_dir --models gpt-4o claude-sonnet-4-6 \
      --version my_run --max_workers 8

Trace format (one JSON per trajectory; see README.md):
  {"trace_id": ..., "task_id": ..., "platform": ..., "agent": ...,
   "instruction": ..., "trajectory": [{"step_index", "screenshot_path",
   "thought", "action", "coordinate"}, ...]}
Optional gold fields "human_label" / "human_alignment" / "human_efficiency"
enable accuracy scoring in the final report.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import mimetypes
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILES = {
    "multi":  SCRIPT_DIR / "prompts" / "multi_v4.txt",
    "binary": SCRIPT_DIR / "prompts" / "binary_v1.txt",
}
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_FIRST_N = 0
DEFAULT_LAST_N = 5
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_WORKERS = 4
ALL_STEPS = 2**31

SUBSCORE_LEVELS = (0.0, 0.5, 1.0)


def _int_or_all(value: str) -> int:
    if value.lower() == "all":
        return ALL_STEPS
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer or 'all'")
    return n


# ── Prompt loading ──────────────────────────────────────────────────────────


def load_system_prompt(path: Path) -> tuple[str, str]:
    """Return (prompt_text, version_tag). version_tag is the filename stem."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip(), path.stem


# ── Trace discovery ─────────────────────────────────────────────────────────


def discover_traces(paths: list[Path]) -> list[dict]:
    """Load trace JSONs from files and/or directories (recursive scan).

    A JSON file counts as a trace when its top-level object is a dict with
    both "trace_id" and "task_id".
    """
    json_paths: list[Path] = []
    for p in paths:
        if p.is_dir():
            json_paths.extend(sorted(p.rglob("*.json")))
        elif p.is_file():
            json_paths.append(p)
        else:
            raise FileNotFoundError(f"Trace path not found: {p}")

    records: list[dict] = []
    seen: set[str] = set()
    for jp in json_paths:
        if jp.name.startswith("."):
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  [skip] invalid JSON: {jp} ({exc})")
            continue
        if not (isinstance(data, dict) and data.get("trace_id") and data.get("task_id")):
            continue
        tid = str(data["trace_id"])
        if tid in seen:
            continue
        seen.add(tid)
        records.append({
            "trace_id": tid,
            "task_id": str(data["task_id"]),
            "platform": str(data.get("platform", "")),
            "agent": str(data.get("agent", "")),
            "instruction": str(data.get("instruction", "")),
            "trajectory": data.get("trajectory") or [],
            "source_json_path": str(jp),
            "human_label": data.get("human_label"),
            "human_alignment": data.get("human_alignment"),
            "human_efficiency": data.get("human_efficiency"),
        })
    return sorted(records, key=lambda r: r["trace_id"])


# ── Image processing ────────────────────────────────────────────────────────


def _is_valid_norm_point(value) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return False
    return 0 <= x <= 1000 and 0 <= y <= 1000


def image_to_data_url(image_path: Path, step: dict | None = None) -> str:
    """Base64-encode a screenshot; if the step has a click coordinate
    (normalized 0-1000), draw a red circle around the action point."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if step and _is_valid_norm_point(step.get("coordinate")):
        with Image.open(image_path) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size
            nx, ny = step["coordinate"]
            x, y = float(nx) / 1000.0 * w, float(ny) / 1000.0 * h
            radius = max(10, int(min(w, h) * 0.05))
            lw = max(4, int(min(w, h) * 0.008))
            draw = ImageDraw.Draw(img)
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                         outline="red", width=lw)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        mime_type = "image/png"
    else:
        mime_type = mime_type or "image/png"
        image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# ── Message building ────────────────────────────────────────────────────────


def _norm(value) -> str:
    return str(value).strip() if value is not None else ""


def build_history_text(trajectory: list[dict], *, include_thought: bool = True) -> str:
    lines: list[str] = []
    for idx, step in enumerate(trajectory):
        si = step.get("step_index", idx)
        action = _norm(step.get("action")) or "[empty]"
        if include_thought:
            thought = _norm(step.get("thought")) or "[empty]"
            lines.append(f"Step {si}:\nThought: {thought}\nAction: {action}")
        else:
            lines.append(f"Step {si}:\nAction: {action}")
    return "\n\n".join(lines)


def resolve_screenshots(
    source_json_path: str, trajectory: list[dict], first_n: int, last_n: int,
) -> list[dict]:
    """Select first_n + last_n steps and resolve their screenshot paths
    (relative to the trace JSON's directory)."""
    if not trajectory:
        return []
    total = len(trajectory)
    indices: set[int] = set()
    if first_n > 0:
        indices.update(range(min(first_n, total)))
    if last_n > 0:
        indices.update(range(max(0, total - last_n), total))
    if not indices:
        return []
    selected = [trajectory[i] for i in sorted(indices)]
    json_dir = Path(source_json_path).resolve().parent
    items: list[dict] = []
    for step in selected:
        rel = step.get("screenshot_path")
        if not rel or not isinstance(rel, str):
            raise ValueError(f"Missing screenshot_path in step {step.get('step_index')}")
        abs_path = (json_dir / rel).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {abs_path}")
        items.append({"image_path": abs_path, "step": step})
    return items


def build_messages(
    trace: dict, system_prompt: str, first_n: int, last_n: int, *,
    history_mode: str = "full", show_action_mark: bool = True,
    include_thought: bool = True,
) -> list[dict]:
    traj = trace.get("trajectory") or []
    items = resolve_screenshots(trace["source_json_path"], traj, first_n, last_n)
    if history_mode == "full":
        history_text = build_history_text(traj, include_thought=include_thought)
    elif history_mode == "selected":
        history_text = build_history_text(
            [it["step"] for it in items], include_thought=include_thought,
        )
    else:
        history_text = None

    content: list[dict] = [
        {"type": "image_url",
         "image_url": {"url": image_to_data_url(
             it["image_path"], it["step"] if show_action_mark else None)}}
        for it in items
    ]
    text_parts = [
        f"{len(items)} screenshots from the agent's trajectory have been provided.",
        f"Platform: {trace['platform']}",
        f"User instruction: {trace['instruction']}",
    ]
    if history_text is not None:
        text_parts.append(f"Full action history:\n{history_text}")
    content.append({"type": "text", "text": "\n".join(text_parts)})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


# ── API call ────────────────────────────────────────────────────────────────


def _backoff_sleep(attempt: int, base: float = 2.0, cap: float = 60.0):
    delay = min(base ** attempt, cap)
    time.sleep(delay * (0.5 + 0.5 * random.random()))


def call_judge_api(
    messages: list[dict],
    base_url: str,
    api_key: str,
    model_name: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"model": model_name, "messages": messages,
                      "temperature": temperature, "timeout": timeout}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            completion = client.chat.completions.create(**kwargs)
            if not hasattr(completion, "choices") or not completion.choices:
                raise RuntimeError(f"Unexpected API response (no choices): {completion}")
            text = completion.choices[0].message.content or ""
            if text.strip():
                return text
            raise RuntimeError("Empty model response")
        except Exception as exc:
            last_error = exc
            logger.debug("Attempt %d/%d failed for %s: %s",
                         attempt, max_retries, model_name, exc)
            if attempt < max_retries:
                _backoff_sleep(attempt)
    raise RuntimeError(f"Judge API failed after {max_retries} attempts: {last_error}")


# ── Response parsing ────────────────────────────────────────────────────────

# Prefer line-anchored "Judge: SUCCESS|FAIL" (allowing Markdown emphasis);
# fall back to a global search. The prompt puts the answer at the END of the
# response, so the LAST match wins over incidental mentions in the Thought.
_JUDGE_LINE_RE = re.compile(
    r"^[\s>*_`#-]*judge\s*:\s*\**\s*(success|fail)\w*",
    flags=re.IGNORECASE | re.MULTILINE,
)
_JUDGE_ANY_RE = re.compile(
    r"\bjudge\s*:\s*\**\s*(success|fail)\w*", flags=re.IGNORECASE
)
THOUGHT_RE = re.compile(
    r"thought\s*:\s*(.*?)(?=\n\s*judge\s*:)", flags=re.IGNORECASE | re.DOTALL
)


def _last_match(pattern: re.Pattern, text: str):
    last = None
    for m in pattern.finditer(text):
        last = m
    return last


def extract_judge_label(text: str) -> str:
    text = text or ""
    m = _last_match(_JUDGE_LINE_RE, text) or _last_match(_JUDGE_ANY_RE, text)
    if not m:
        raise ValueError(f"Cannot parse judge label from: {text[:200]!r}")
    label = m.group(1).strip().upper()
    if label.startswith("SUCC"):
        return "SUCCESS"
    if label.startswith("FAIL"):
        return "FAIL"
    raise ValueError(f"Unknown judge label: {label}")


def extract_judge_thought(text: str) -> str | None:
    m = THOUGHT_RE.search(text or "")
    if m:
        return m.group(1).strip() or None
    return (text or "").strip() or None


def _parse_subscore_value(raw: str | None) -> float | str | None:
    """Map a captured literal to a number in {0, 0.5, 1} or 'N/A' or None."""
    if raw is None:
        return None
    s = raw.strip().lower().rstrip(".")
    if s in {"n/a", "na", "n.a.", "n.a"}:
        return "N/A"
    try:
        f = float(s)
    except ValueError:
        return None
    snapped = min(SUBSCORE_LEVELS, key=lambda v: abs(v - f))
    return snapped if abs(snapped - f) <= 0.05 else None


def extract_subscore(text: str, key: str) -> tuple[float | None, str]:
    """Return (value, status) for an "Alignment:" / "Efficiency:" line.
    value in {0, 0.5, 1.0} or None; status in {'ok', 'na', 'missing', 'malformed'}.
    """
    text = text or ""
    strict = re.compile(
        rf"^[\s>*_`-]*{key}(?!_)\s*:\s*\**\s*(1(?:\.0+)?|0\.5+|0(?:\.0+)?|\.5|n/?a)\b",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    m = _last_match(strict, text)
    if not m:
        loose = re.compile(
            rf"\b{key}(?!_)\s*:\s*\**\s*(1(?:\.0+)?|0\.5+|0(?:\.0+)?|\.5|n/?a)\b",
            flags=re.IGNORECASE,
        )
        m = _last_match(loose, text)
        if not m:
            return None, "missing"
    parsed = _parse_subscore_value(m.group(1))
    if parsed == "N/A":
        return None, "na"
    if parsed is None:
        return None, "malformed"
    return parsed, "ok"


# ── Per-sample scoring (OSReward-Multi rules) ───────────────────────────────


def score_subscore(gt: float | None, pred: float | None) -> float | None:
    """Alignment/Efficiency score for one trace: only scored when the gold
    label is SUCCESS (gt is a number). Missed SUCCESS (pred is None because
    the model said FAIL or refused to score) scores 0.0; otherwise
    1 - |gt - pred|, which lives on {0.0, 0.5, 1.0}."""
    if gt is None:
        return None
    if pred is None:
        return 0.0
    return round(1.0 - abs(gt - pred), 3)


# ── Result I/O ──────────────────────────────────────────────────────────────


def output_path_for_model(output_dir: Path, version: str, prompt_type: str,
                          model_name: str) -> Path:
    safe = model_name.replace("/", "_").replace("\\", "_")
    return output_dir / f"judge_{version}_{prompt_type}_{safe}.jsonl"


def load_existing_results(path: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    if not path.exists():
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tid = rec.get("trace_id")
            if tid:
                recs[str(tid)] = rec
    return recs


def write_results_jsonl(path: Path, records: dict[str, dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for tid in sorted(records):
            f.write(json.dumps(records[tid], ensure_ascii=False) + "\n")


# ── Single-trace judge ──────────────────────────────────────────────────────

PARSE_RETRIES = 1


def judge_single_trace(
    trace: dict, args: argparse.Namespace,
    system_prompt: str, prompt_version: str,
    base_url: str, api_key: str, model_name: str,
) -> dict:
    started = time.time()
    prompt_type = args.prompt_type
    result = {
        "trace_id": trace["trace_id"],
        "task_id": trace["task_id"],
        "platform": trace["platform"],
        "agent": trace["agent"],
        "instruction": trace["instruction"],
        "source_json_path": trace["source_json_path"],
        "num_steps": len(trace.get("trajectory") or []),
        "first_n": "all" if args.first_n >= ALL_STEPS else args.first_n,
        "last_n": "all" if args.last_n >= ALL_STEPS else args.last_n,
        "history_mode": args.history,
        "show_action_mark": not args.no_mark,
        "include_thought": not args.no_thought,
        "version": args.version,
        "prompt_type": prompt_type,
        "prompt_version": prompt_version,
        "judge_model": model_name,
        "human_label": trace.get("human_label"),
        "human_alignment": trace.get("human_alignment"),
        "human_efficiency": trace.get("human_efficiency"),
        "judge_thought": None,
        "judge_label": None,
        "judge_alignment": None,
        "judge_efficiency": None,
        "judge_alignment_status": None,
        "judge_efficiency_status": None,
        "judge_raw_response": None,
        "binary_correct": None,
        "alignment_score": None,
        "efficiency_score": None,
        "status": "error",
        "error": None,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": None,
    }

    try:
        messages = build_messages(
            trace, system_prompt, args.first_n, args.last_n,
            history_mode=args.history,
            show_action_mark=not args.no_mark,
            include_thought=not args.no_thought,
        )
    except Exception as exc:
        # Bad trace (missing screenshot, malformed step): record the error row
        # and keep the run going.
        result["error"] = str(exc)
        result["elapsed_seconds"] = round(time.time() - started, 3)
        return result

    last_exc: Exception | None = None
    for attempt in range(1, PARSE_RETRIES + 2):
        try:
            raw = call_judge_api(
                messages, base_url, api_key, model_name,
                temperature=args.temperature, max_tokens=args.max_tokens,
                timeout=args.timeout, max_retries=args.max_retries,
            )
            result["judge_raw_response"] = raw
            result["judge_thought"] = extract_judge_thought(raw)
            result["judge_label"] = extract_judge_label(raw)
            if prompt_type == "multi":
                align_val, align_status = extract_subscore(raw, "alignment")
                effic_val, effic_status = extract_subscore(raw, "efficiency")
                result["judge_alignment"] = align_val
                result["judge_alignment_status"] = align_status
                result["judge_efficiency"] = effic_val
                result["judge_efficiency_status"] = effic_status
            result["status"] = "ok"
            last_exc = None
            break
        except ValueError as parse_exc:
            # API succeeded but the response was unparseable: retry once.
            last_exc = parse_exc
            if attempt < PARSE_RETRIES + 1:
                _backoff_sleep(attempt, base=2.0, cap=10.0)
                continue
            break
        except Exception as api_exc:
            last_exc = api_exc
            break

    if last_exc is not None:
        result["error"] = str(last_exc)

    # Score against gold when the trace carries it.
    human_label = trace.get("human_label")
    if result["status"] == "ok" and human_label and result["judge_label"]:
        result["binary_correct"] = int(result["judge_label"] == human_label)
        if prompt_type == "multi" and human_label == "SUCCESS":
            if result["judge_label"] == "SUCCESS":
                result["alignment_score"] = score_subscore(
                    trace.get("human_alignment"), result["judge_alignment"])
                result["efficiency_score"] = score_subscore(
                    trace.get("human_efficiency"), result["judge_efficiency"])
            else:
                result["alignment_score"] = 0.0
                result["efficiency_score"] = 0.0

    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


# ── Per-model run ───────────────────────────────────────────────────────────


def run_model(
    model_name: str, traces: list[dict], args: argparse.Namespace,
    system_prompt: str, prompt_version: str, base_url: str, api_key: str,
) -> Path:
    out_path = output_path_for_model(args.output_dir, args.version,
                                     args.prompt_type, model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(out_path)
    done_ids = {tid for tid, r in existing.items() if r.get("status") == "ok"}
    pending = [t for t in traces if t["trace_id"] not in done_ids]
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"\n{'=' * 64}")
    print(f"Model:        {model_name}")
    print(f"Base URL:     {base_url}")
    print(f"Prompt:       type={args.prompt_type}  version={prompt_version}")
    print(f"Output:       {out_path}")
    print(f"Traces:       {len(traces)}  |  Resumed: {len(done_ids)}  |  Pending: {len(pending)}")
    print(f"{'=' * 64}", flush=True)

    if not pending:
        print("  Nothing to do.")
        return out_path

    ok, err = 0, 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(judge_single_trace, t, args, system_prompt,
                        prompt_version, base_url, api_key, model_name)
            for t in pending
        ]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  {model_name}"):
            res = fut.result()
            existing[res["trace_id"]] = res
            write_results_jsonl(out_path, existing)
            if res["status"] == "ok":
                ok += 1
            else:
                err += 1

    print(f"  OK: {ok}  |  Errors: {err}  |  Written: {out_path}")
    return out_path


# ── Report ──────────────────────────────────────────────────────────────────


def print_report(model_paths: dict[str, Path], prompt_type: str) -> None:
    print(f"\n{'#' * 72}")
    print(f"#  Summary  ({prompt_type} output)")
    print(f"{'#' * 72}")
    for model_name, path in model_paths.items():
        rows = [r for r in load_existing_results(path).values()
                if r.get("status") == "ok"]
        if not rows:
            print(f"\n{model_name}: no successful results.")
            continue
        n_succ = sum(1 for r in rows if r.get("judge_label") == "SUCCESS")
        n_fail = sum(1 for r in rows if r.get("judge_label") == "FAIL")
        print(f"\n{model_name}  (N={len(rows)})")
        print(f"  Judge labels:      SUCCESS={n_succ}  FAIL={n_fail}")
        gold_rows = [r for r in rows if r.get("binary_correct") is not None]
        if gold_rows:
            acc = sum(r["binary_correct"] for r in gold_rows) / len(gold_rows)
            print(f"  Binary accuracy:   {acc * 100:.1f}%  (n={len(gold_rows)} with gold)")
        if prompt_type == "multi":
            a = [r["alignment_score"] for r in rows if r.get("alignment_score") is not None]
            e = [r["efficiency_score"] for r in rows if r.get("efficiency_score") is not None]
            if a:
                print(f"  Alignment score:   {sum(a) / len(a):.3f}  (n={len(a)}, gold SUCCESS only)")
            if e:
                print(f"  Efficiency score:  {sum(e) / len(e):.3f}  (n={len(e)}, gold SUCCESS only)")
        for r in rows:
            align = f"  Alignment={r.get('judge_alignment')}" if prompt_type == "multi" else ""
            effic = f"  Efficiency={r.get('judge_efficiency')}" if prompt_type == "multi" else ""
            print(f"    {r['trace_id']}: Judge={r.get('judge_label')}{align}{effic}")


# ── CLI / main ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSReward standalone judge (binary / multi output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--traces", nargs="+", type=Path, required=True,
                   help="Trace JSON file(s) and/or directory(ies) to scan recursively.")
    p.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                   help="Judge model name(s) served by the endpoint.")
    p.add_argument("--base_url",
                   default=os.environ.get("JUDGE_BASE_URL")
                   or os.environ.get("OPENAI_BASE_URL")
                   or "https://api.openai.com/v1",
                   help="OpenAI-compatible endpoint (env: JUDGE_BASE_URL / OPENAI_BASE_URL).")
    p.add_argument("--api_key",
                   default=os.environ.get("JUDGE_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"),
                   help="API key (env: JUDGE_API_KEY / OPENAI_API_KEY).")
    p.add_argument("--prompt_type", choices=["binary", "multi"], default="multi",
                   help="Output mode switch. 'multi' (default) = Judge + Alignment "
                        "+ Efficiency (OSReward-Multi); 'binary' = Judge only (OSReward).")
    p.add_argument("--prompt_file", type=Path, default=None,
                   help="Explicit system prompt file; overrides the --prompt_type default.")
    p.add_argument("--version", default="demo",
                   help="Run tag used in output filenames (default: demo).")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--first_n", type=_int_or_all, default=DEFAULT_FIRST_N,
                   help="Number of initial screenshots to include, or 'all' (default: 0).")
    p.add_argument("--last_n", type=_int_or_all, default=DEFAULT_LAST_N,
                   help="Number of final screenshots to include, or 'all' (default: 5).")
    p.add_argument("--history", default="full", choices=["full", "selected", "none"],
                   help="Action history text scope: all steps / only steps with "
                        "screenshots / none (vision-only).")
    p.add_argument("--no_thought", action="store_true",
                   help="Strip agent thoughts from the history text, keeping only actions.")
    p.add_argument("--no_mark", action="store_true",
                   help="Disable red-circle action markers on screenshots.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Optional max_tokens cap (useful for thinking models).")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Per-call API timeout in seconds (default: {DEFAULT_TIMEOUT}).")
    p.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
                   help="Concurrent API calls per model.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max traces per model (smoke testing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("No API key. Pass --api_key or set JUDGE_API_KEY / OPENAI_API_KEY.")
    if args.first_n == 0 and args.last_n == 0:
        raise SystemExit("At least one of --first_n or --last_n must be > 0.")

    prompt_file = args.prompt_file or DEFAULT_PROMPT_FILES[args.prompt_type]
    system_prompt, prompt_version = load_system_prompt(prompt_file)
    print(f"Prompt: type={args.prompt_type}  file={prompt_file}  version={prompt_version}")

    traces = discover_traces(args.traces)
    if not traces:
        raise SystemExit("No traces found.")
    n_gold = sum(1 for t in traces if t.get("human_label"))
    print(f"Traces: {len(traces)}  (with gold labels: {n_gold})")

    model_paths: dict[str, Path] = {}
    for model_name in args.models:
        model_paths[model_name] = run_model(
            model_name, traces, args, system_prompt, prompt_version,
            args.base_url, args.api_key,
        )

    print_report(model_paths, args.prompt_type)


if __name__ == "__main__":
    main()

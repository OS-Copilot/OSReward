"""The VLM-as-a-judge core: build prompts, call the model, score, resume.

A judge run sends, per trajectory, the system prompt plus a user message of
[last-N screenshots] + [instruction + full thought/action history] to an
OpenAI-compatible chat model, parses a binary ``Judge: SUCCESS|FAIL`` out of the
reply, and stores it next to the golden label. Runs are concurrent across API
channels and resume-safe (only successfully-judged traces count as done, so
errors are retried on re-run).
"""
import base64
import itertools
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from . import config, metrics, prepare

PLATFORM_TAG = {
    "osworld": "desktop (Ubuntu)", "macos": "desktop (macOS)",
    "windows": "desktop (Windows)", "androidworld": "mobile (Android)",
    "webarena": "web",
}

_print_lock = threading.Lock()
_write_lock = threading.Lock()


def extract_judge(text):
    """Pull a binary SUCCESS/FAIL out of a judge reply (None if absent)."""
    if not text:
        return None
    m = re.search(r"judge\s*:\s*([a-zA-Z]+)", text, flags=re.IGNORECASE)
    if m:
        lab = m.group(1).strip().upper()
        if lab.startswith("SUCC"):
            return "SUCCESS"
        if lab.startswith("FAIL"):
            return "FAIL"
    if re.search(r"\bsuccess\b", text, flags=re.IGNORECASE):
        return "SUCCESS"
    if re.search(r"\bfail\b", text, flags=re.IGNORECASE):
        return "FAIL"
    return None


def encode_image_path(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


def build_messages(rec, system_prompt, platform):
    """Compose the system+user chat messages for one judge-ready record.

    If the record carries a pre-composed ``user_text`` (legacy webarena
    judgements), it is used verbatim; otherwise the user message is built from
    the instruction + thought/action history.
    """
    imgs = rec.get("images", [])
    user_text = rec.get("user_text") or (
        f"The screenshots of the last {len(imgs)} states of the operating system "
        f"have been provided.\n"
        f"Platform: {PLATFORM_TAG.get(platform, platform)}.\n"
        f"The user instruction: {rec.get('instruction')}.\n"
        f"The agent action history with thought:\n{rec.get('history', '')}"
    )
    content = [{"type": "image_url", "image_url": {"url": encode_image_path(p)}}
               for p in imgs]
    content.append({"type": "text", "text": user_text})
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content}]


class JudgeRunner:
    """Concurrent, fail-over caller over one or more API channels."""

    def __init__(self, model, workers=12, timeout=120, temperature=0, max_retry=6):
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retry = max_retry
        self.channels = config.get_channels()
        self.clients = [OpenAI(base_url=c["base_url"], api_key=c["api_key"])
                        for c in self.channels]
        self.sems = [threading.Semaphore(c["max_concurrency"]) for c in self.channels]
        self._cycle = itertools.cycle(range(len(self.channels)))
        self._cycle_lock = threading.Lock()
        self.dead = set()  # channels lacking this model / with a dead key
        self.workers = min(workers, config.GLOBAL_MAX_CONCURRENCY,
                           sum(c["max_concurrency"] for c in self.channels))

    def _next_ci(self):
        with self._cycle_lock:
            for _ in range(len(self.channels)):
                ci = next(self._cycle)
                if ci not in self.dead:
                    return ci
            return next(self._cycle)

    def call(self, messages):
        """Return (content, usage, duration_ms, retries, channel_name, error)."""
        last_err = last_chan = None
        for attempt in range(self.max_retry):
            ci = self._next_ci()
            ch, client, sem = self.channels[ci], self.clients[ci], self.sems[ci]
            last_chan = ch["name"]
            with sem:
                t0 = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=self.model, messages=messages,
                        temperature=self.temperature, timeout=self.timeout)
                    dur = (time.time() - t0) * 1000.0
                    content = resp.choices[0].message.content or ""
                    usage = {}
                    if resp.usage:
                        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                                 "completion_tokens": resp.usage.completion_tokens,
                                 "total_tokens": resp.usage.total_tokens}
                    return content, usage, dur, attempt, ch["name"], None
                except Exception as e:
                    last_err = str(e)
                    es = last_err.lower()
                    if ("no available channel" in es or "model_not_found" in es
                            or "model not found" in es or "401" in es
                            or ("invalid" in es and "token" in es)):
                        self.dead.add(ci)
            if attempt < self.max_retry - 1:
                time.sleep(min(2 ** attempt, 20))
        return "", {}, 0.0, self.max_retry, last_chan, last_err


def run_batch(records, build_fn, out_path, runner, version, setting):
    """Judge a list of records (resume-safe), append rows to ``out_path``, report.

    ``records`` are dicts with at least ``trace_id`` and ``golden_label`` (plus
    whatever ``build_fn`` reads); an optional ``extra`` dict is merged onto each
    output row. ``build_fn(rec) -> messages``.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = metrics.load_done_ids(out_path)
    todo = [r for r in records if r["trace_id"] not in done]
    gold = {"SUCCESS": 0, "FAIL": 0, None: 0}
    for r in records:
        gold[r.get("golden_label")] = gold.get(r.get("golden_label"), 0) + 1
    print(f"model={runner.model} setting={setting} version={version}")
    print(f"records={len(records)} gold={{S:{gold.get('SUCCESS', 0)},F:{gold.get('FAIL', 0)}}} "
          f"done={len(done)} todo={len(todo)} "
          f"channels={[c['name'] for c in runner.channels]} workers={runner.workers}")
    print(f"out={out_path}")
    if not todo:
        return metrics.report(out_path)

    def work(rec):
        content, usage, dur, retries, chan, err = runner.call(build_fn(rec))
        judge = extract_judge(content)
        correct = int(judge == rec["golden_label"]) if (judge and rec["golden_label"]) else None
        row = {"trace_id": rec["trace_id"], "model": runner.model, "setting": setting,
               "version": version, "channel": chan, "instruction": rec.get("instruction"),
               "judge_response": content, "judge": judge,
               "golden_label": rec["golden_label"], "binary_correct": correct,
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"),
               "total_tokens": usage.get("total_tokens"), "duration_ms": dur,
               "retry_count": retries, "error": err}
        row.update(rec.get("extra", {}))
        return row

    completed = 0
    with open(out_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=runner.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fut in as_completed(futs):
            row = fut.result()
            with _write_lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
            completed += 1
            with _print_lock:
                tag = ("ERR:" + (row["error"][:40] if row["error"] else "")) if row["error"] \
                    else f"{row['judge']}|gold={row['golden_label']}|ok={row['binary_correct']}"
                print(f"[{completed}/{len(todo)}] {str(row['trace_id'])[:12]} "
                      f"{row['channel']:5s} {tag}")
    return metrics.report(out_path)


def judge_platform(platform, agent, model, version="v1", setting="last5",
                   prompt=config.DEFAULT_PROMPT, concurrency=12, timeout=120,
                   limit=0, sample="head", seed=0):
    """High-level: judge a prepared (platform, agent) with ``model``. Returns metrics."""
    system_prompt = config.load_system_prompt(prompt)
    records = prepare.load_judge_ready(platform, agent)
    if sample == "random":
        random.Random(seed).shuffle(records)
    if limit:
        records = records[:limit]
    out_path = os.path.join(config.platform_dir(platform), "results",
                            f"judge_{version}_{setting}_{agent}_{model}.jsonl")
    runner = JudgeRunner(model, workers=concurrency, timeout=timeout)
    return run_batch(records, lambda r: build_messages(r, system_prompt, platform),
                     out_path, runner, version, setting)

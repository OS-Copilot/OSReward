"""`tasks` subcommand: bring tasks in from other datasets into webtrail JSONL.

Collection input is a small JSONL of ``{url, instruction, ...}`` records. Most
existing web-agent datasets store the same information under different column
names (``website`` / ``confirmed_task`` / ``ques`` / ``goal`` …). ``webtrail
tasks import`` maps those onto webtrail's schema, drops records missing a URL or
instruction, de-duplicates, and writes a ready-to-collect task file.

Sources: a local ``.jsonl`` / ``.json`` / ``.parquet`` file, or a Hugging Face
dataset id (``owner/name``) when the optional ``datasets`` package is installed.
Column names are auto-detected; override with ``--url-field`` / ``--instruction-field``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from .types import Task

logger = logging.getLogger(__name__)

URL_FIELDS = ["url", "urls", "website", "web", "start_url", "site", "domain"]
INSTR_FIELDS = ["instruction", "task", "confirmed_task", "ques", "question",
                "goal", "query", "intent"]


def _load_records(source: str, split: str) -> Iterable[dict]:
    path = Path(source)
    if path.exists():
        if path.suffix == ".parquet":
            return _load_parquet(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl" or "\n{" in text.strip():
            return [json.loads(l) for l in text.splitlines() if l.strip()]
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("tasks", data.get("data", []))
    return _load_hf(source, split)


def _load_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as err:
        raise SystemExit("reading .parquet needs pyarrow: pip install pyarrow") from err
    return pq.read_table(path).to_pylist()


def _load_hf(source: str, split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise SystemExit(
            f"'{source}' is not a local file; loading a Hugging Face dataset "
            "needs the datasets package: pip install datasets") from err
    logger.info("loading HF dataset %s (split=%s)", source, split)
    ds = load_dataset(source, split=split)
    return [dict(row) for row in ds]


def _pick(record: dict, candidates: list[str], override: str | None) -> str | None:
    if override:
        return override if override in record else None
    lowered = {k.lower(): k for k in record}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def normalize(records: Iterable[dict], url_field: str | None,
              instr_field: str | None, default_url: str | None) -> list[dict]:
    out: list[dict] = []
    dropped = 0
    last_keys = (None, None)
    for record in records:
        # detect column names per record: HF datasets are uniform, but mixed
        # files (or a fallback column on some rows) should not silently drop
        ukey = _pick(record, URL_FIELDS, url_field)
        ikey = _pick(record, INSTR_FIELDS, instr_field)
        last_keys = (ukey, ikey)
        url = (record.get(ukey) if ukey else None) or default_url
        instruction = record.get(ikey) if ikey else None
        if not url or not instruction:
            dropped += 1
            continue
        task: dict = {"url": url, "instruction": str(instruction).strip()}
        for extra in ("id", "task_id", "steps", "criteria", "max_steps"):
            if record.get(extra) is not None:
                task[extra] = record[extra]
        out.append(task)
    if dropped:
        logger.warning("dropped %d records missing url/instruction "
                       "(last seen url_field=%s instruction_field=%s)",
                       dropped, *last_keys)
    return out


def dedupe(tasks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for task in tasks:
        try:
            parsed = Task.from_record(task, 0)
        except ValueError:
            continue
        key = parsed.instruction.lower().strip() + "|" + parsed.start_url.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)
    return unique


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "tasks", help="import tasks from another dataset into webtrail JSONL"
    )
    sub = parser.add_subparsers(required=True)
    imp = sub.add_parser("import", help="convert a file or HF dataset into a task file")
    imp.add_argument("--source", required=True,
                     help="local .jsonl/.json/.parquet, or a HF dataset id owner/name")
    imp.add_argument("--out", required=True, help="output task JSONL")
    imp.add_argument("--split", default="train", help="HF split (default train)")
    imp.add_argument("--url-field", help="column holding the start URL (else auto)")
    imp.add_argument("--instruction-field", help="column holding the instruction (else auto)")
    imp.add_argument("--default-url",
                     help="URL to use when a record has none (e.g. a search engine "
                          "for open-ended tasks)")
    imp.add_argument("--limit", type=int, help="keep at most this many tasks")
    imp.add_argument("--no-dedupe", action="store_true")
    imp.set_defaults(handler=main_import)


def main_import(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = _load_records(args.source, args.split)
    tasks = normalize(records, args.url_field, args.instruction_field, args.default_url)
    if not args.no_dedupe:
        before = len(tasks)
        tasks = dedupe(tasks)
        if before != len(tasks):
            logger.info("deduped %d -> %d tasks", before, len(tasks))
    if args.limit:
        tasks = tasks[:args.limit]
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks -> {out}")

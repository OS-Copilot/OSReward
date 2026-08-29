#!/usr/bin/env python3
"""Convert canonical OSReward trajectories into Hugging Face SFT data."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PathLike = str | Path
LABELS = {"SUCCESS", "FAIL"}
METRICS = ("success", "alignment_score", "efficiency", "self_correction")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


@dataclass(frozen=True)
class Trace:
    trace_id: str
    platform: str
    instruction: str
    steps: tuple[dict, ...]
    root: Path


@dataclass(frozen=True)
class Vote:
    source: str
    model: str
    label: str
    thought: str
    metrics: dict


def _records(path: Path) -> list[dict]:
    """Read a JSON object/array or JSONL file."""

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                records.append(record)
        return records

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"expected a JSON object or array of objects: {path}")


def _trajectory_files(value: PathLike) -> tuple[Path, list[Path]]:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"unsupported trajectory file: {path}")
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise ValueError("trajectory input contains no JSON or JSONL files")
    return path, files


def _load_traces(value: PathLike) -> list[Trace]:
    root, files = _trajectory_files(value)
    traces = []
    seen = set()
    for path in files:
        for record in _records(path):
            trace_id = str(record.get("trace_id") or "").strip()
            instruction = str(record.get("instruction") or "").strip()
            steps = record.get("trajectory")
            if not trace_id or not instruction or not isinstance(steps, list):
                raise ValueError(f"malformed canonical trajectory in {path}")
            if trace_id in seen:
                raise ValueError(f"duplicate trace_id: {trace_id}")
            seen.add(trace_id)
            traces.append(
                Trace(
                    trace_id=trace_id,
                    platform=str(record.get("platform") or ""),
                    instruction=instruction,
                    steps=tuple(step for step in steps if isinstance(step, dict)),
                    root=root,
                )
            )
    return sorted(traces, key=lambda trace: trace.trace_id)


def _load_votes(judgments: Mapping[str, PathLike]) -> tuple[list[str], dict]:
    if not isinstance(judgments, Mapping) or not judgments:
        raise ValueError("judgments must map judge names to JSON/JSONL files")
    sources = [str(source) for source in judgments]
    votes: dict[str, dict[str, Vote]] = {source: {} for source in sources}
    for raw_source, raw_path in judgments.items():
        source = str(raw_source)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"invalid judgment file for {source}: {path}")
        for record in _records(path):
            status = str(record.get("status") or "ok").lower()
            if status != "ok":
                continue
            trace_id = str(record.get("trace_id") or "").strip()
            model = str(record.get("judge_model") or "").strip()
            label = str(record.get("judge_label") or "").strip().upper()
            thought = str(record.get("judge_thought") or "").strip()
            if not trace_id or not model or label not in LABELS or not thought:
                continue
            if trace_id in votes[source]:
                raise ValueError(
                    f"duplicate judgment for source={source}, trace_id={trace_id}"
                )
            raw_metrics = record.get("metrics")
            metrics = raw_metrics if isinstance(raw_metrics, dict) else record
            votes[source][trace_id] = Vote(
                source=source,
                model=model,
                label=label,
                thought=thought,
                metrics={key: metrics.get(key) for key in METRICS},
            )
    return sources, votes


def _majority(
    trace_id: str,
    sources: list[str],
    votes_by_source: dict,
    preferred_judge: str | None,
) -> tuple[list[Vote], Vote] | None:
    votes = [
        vote
        for source in sources
        if (vote := votes_by_source[source].get(trace_id)) is not None
    ]
    if not votes:
        return None
    label, count = Counter(vote.label for vote in votes).most_common(1)[0]
    if count * 2 <= len(votes):
        return None
    agreeing = [vote for vote in votes if vote.label == label]
    target = next(
        (vote for vote in agreeing if vote.source == preferred_judge), agreeing[0]
    )
    return votes, target


def _screenshots(trace: Trace) -> list[tuple[dict, Path]]:
    screenshots = []
    for step in trace.steps:
        state = step.get("state")
        value = state.get("screenshot_path") if isinstance(state, dict) else None
        if not value:
            continue
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"screenshot_path must be relative for {trace.trace_id}: {value}"
            )
        path = (trace.root / relative).resolve()
        try:
            path.relative_to(trace.root)
        except ValueError as exc:
            raise ValueError(
                f"screenshot_path escapes input root for {trace.trace_id}: {value}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"screenshot not found for {trace.trace_id}: {value}"
            )
        screenshots.append((step, path))
    return screenshots


def _coordinate(step: dict) -> list[int] | None:
    action = step.get("action")
    if not isinstance(action, dict):
        return None
    coordinates = action.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        return None
    first = coordinates[0]
    relative = first.get("relative") if isinstance(first, dict) else None
    if not isinstance(relative, (list, tuple)) or len(relative) != 2:
        return None
    try:
        x, y = float(relative[0]), float(relative[1])
    except (TypeError, ValueError):
        return None
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        return None
    return [round(x * 1000), round(y * 1000)]


def _action_text(action: object) -> str:
    if isinstance(action, str):
        return action.strip() or "[empty]"
    if action is None:
        return "[empty]"
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def _user_message(trace: Trace, selected: list[tuple[dict, Path]]) -> str:
    count = len(selected)
    placeholders = "\n".join(
        f"Visual state {index}: <image>" for index in range(1, count + 1)
    )
    history = "\n".join(
        f"Step {step.get('step_index', index)}: {_action_text(step.get('action'))}"
        for index, (step, _path) in enumerate(selected)
    )
    return "\n\n".join(
        (
            placeholders,
            f"The screenshots of the last {count} states are provided above, "
            "oldest first.",
            f"Platform: {trace.platform or '(unspecified)'}",
            f"User instruction: {trace.instruction}",
            f"Action history for the shown states:\n{history}",
        )
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.") or "unknown"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_image(source: Path, output: Path, cache: dict[Path, str]) -> str:
    if source in cache:
        return cache[source]
    suffix = source.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported screenshot extension: {source.name}")
    digest = _digest(source)
    relative = (
        Path("osreward_rm_train_bundle")
        / "images"
        / digest[:2]
        / f"{digest}{suffix}"
    )
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or _digest(target) != digest:
            raise RuntimeError(f"conflicting image: {relative}")
    else:
        shutil.copy2(source, target)
    cache[source] = relative.as_posix()
    return cache[source]


def _contains_credential(text: str) -> bool:
    return any(pattern.search(text) for pattern in CREDENTIAL_PATTERNS)


def _prompt() -> str:
    path = Path(__file__).resolve().parents[1] / "eval_pipeline/prompts/binary_v1.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def _row(
    trace: Trace,
    selected: list[tuple[dict, Path]],
    votes: list[Vote],
    target: Vote,
    prompt: str,
    output: Path,
    image_cache: dict[Path, str],
) -> dict:
    setting = f"last{len(selected)}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": _user_message(trace, selected)},
        {
            "role": "assistant",
            "content": f"Thought: {target.thought}\nJudge: {target.label}",
        },
    ]
    release_text = json.dumps(
        {"messages": messages, "votes": [vote.thought for vote in votes]},
        ensure_ascii=False,
    )
    if _contains_credential(release_text):
        raise ValueError(f"credential-like text in trace {trace.trace_id}")
    return {
        "id": "__".join(
            (_slug(trace.trace_id), setting, _slug(target.model))
        ),
        "messages": messages,
        "images": [
            _copy_image(path, output, image_cache) for _step, path in selected
        ],
        "coordinates": [_coordinate(step) for step, _path in selected],
        "results": [
            {
                "judge": vote.label,
                "setting": setting,
                "model": vote.model,
                "metrics": vote.metrics,
            }
            for vote in votes
        ],
    }


def export_judged_sft(
    trajectories: PathLike,
    judgments: Mapping[str, PathLike],
    output_dir: PathLike,
    *,
    sampling: str = "last5",
    preferred_judge: str | None = None,
    overwrite: bool = False,
) -> int:
    """Export majority-judged trajectories in OS-Shepherd SFT format."""

    if sampling not in {"last5", "incremental"}:
        raise ValueError("sampling must be 'last5' or 'incremental'")
    if preferred_judge is not None and preferred_judge not in judgments:
        raise ValueError(f"unknown preferred_judge: {preferred_judge}")

    output = Path(output_dir).expanduser().resolve()
    dataset = output / "dataset.json"
    if dataset.exists() and not overwrite:
        raise FileExistsError(dataset)

    traces = _load_traces(trajectories)
    sources, votes_by_source = _load_votes(judgments)
    prompt = _prompt()
    output.mkdir(parents=True, exist_ok=True)
    image_cache: dict[Path, str] = {}
    count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output,
            prefix=".dataset.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write("[\n")
            for trace in traces:
                majority = _majority(
                    trace.trace_id, sources, votes_by_source, preferred_judge
                )
                if majority is None:
                    continue
                votes, target = majority
                screenshots = _screenshots(trace)
                limit = min(5, len(screenshots))
                if not limit:
                    continue
                sizes = range(1, limit + 1) if sampling == "incremental" else (limit,)
                for size in sizes:
                    row = _row(
                        trace,
                        screenshots[-size:],
                        votes,
                        target,
                        prompt,
                        output,
                        image_cache,
                    )
                    if count:
                        handle.write(",\n")
                    handle.write(json.dumps(row, ensure_ascii=False))
                    count += 1
            handle.write("\n]\n")
        temporary.replace(dataset)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


__all__ = ["export_judged_sft"]

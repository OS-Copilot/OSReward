#!/usr/bin/env python3
"""Run a binary OSReward judge through OpenAI or Anthropic API protocols."""

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

from anthropic import Anthropic
from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm


logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompts" / "binary_v1.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_FIRST_N = 0
DEFAULT_LAST_N = 5
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_WORKERS = 4
ALL_STEPS = 2**31
PARSE_RETRIES = 1
# A response cut off by the completion budget usually ends inside the thought,
# before the "Judge:" line ever appears. Such traces are retried with a larger
# budget instead of an identical request.
TRUNCATION_RETRIES = 2
ESCALATION_FACTOR = 4
ESCALATED_MAX_TOKENS_FLOOR = 4096
ESCALATED_MAX_TOKENS_CAP = 16384
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
PARQUET_METADATA_COLUMNS = [
    "trace_id",
    "task_id",
    "platform",
    "agent",
    "instruction",
    "frame_semantics",
    "trajectory_length",
    "trajectory",
    "human_label",
]


def int_or_all(value: str) -> int:
    if value.lower() == "all":
        return ALL_STEPS
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative or 'all'")
    return number


def load_system_prompt(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip(), path.stem


def trace_record(data: dict, **source: object) -> dict:
    return {
        "trace_id": str(data["trace_id"]),
        "task_id": str(data["task_id"]),
        "platform": str(data.get("platform", "")),
        "agent": str(data.get("agent", "")),
        "instruction": str(data.get("instruction", "")),
        "trajectory": data.get("trajectory") or [],
        "human_label": data["human_label"],
        **source,
    }


def is_binary_trace(data: object) -> bool:
    return bool(
        isinstance(data, dict)
        and data.get("trace_id")
        and data.get("task_id")
        and data.get("human_label") in {"SUCCESS", "FAIL"}
    )


def discover_json_traces(json_paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for json_path in json_paths:
        if json_path.name.startswith("."):
            continue
        try:
            with json_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"  [skip] invalid JSON: {json_path} ({exc})")
            continue
        if not is_binary_trace(data):
            continue
        trace_id = str(data["trace_id"])
        if trace_id in seen:
            raise ValueError(f"Duplicate trace_id in input: {trace_id}")
        seen.add(trace_id)
        records.append(trace_record(data, source_json_path=str(json_path)))
    return sorted(records, key=lambda record: record["trace_id"])


def discover_parquet_traces(parquet_paths: list[Path]) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for Parquet input; install eval_pipeline/requirements.txt"
        ) from exc

    records: list[dict] = []
    seen: set[str] = set()
    required = set(PARQUET_METADATA_COLUMNS) | {"screenshots"}
    for parquet_path in sorted(parquet_paths):
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        missing = required - available
        if missing:
            raise ValueError(f"Missing Parquet columns in {parquet_path}: {sorted(missing)}")
        table = parquet_file.read(columns=PARQUET_METADATA_COLUMNS)
        locations = [
            (row_group, row_index)
            for row_group in range(parquet_file.num_row_groups)
            for row_index in range(parquet_file.metadata.row_group(row_group).num_rows)
        ]
        rows = table.to_pylist()
        if len(rows) != len(locations):
            raise ValueError(f"Parquet row metadata mismatch in {parquet_path}")
        for data, (row_group, row_index) in zip(rows, locations):
            if not is_binary_trace(data):
                raise ValueError(
                    f"Invalid binary trace in {parquet_path}, row group {row_group}, "
                    f"row {row_index}"
                )
            trace_id = str(data["trace_id"])
            if trace_id in seen:
                raise ValueError(f"Duplicate trace_id in input: {trace_id}")
            seen.add(trace_id)
            records.append(
                trace_record(
                    data,
                    source_parquet_path=str(parquet_path),
                    source_parquet_row_group=row_group,
                    source_parquet_row_index=row_index,
                )
            )
    return sorted(records, key=lambda record: record["trace_id"])


def discover_traces(paths: list[Path]) -> list[dict]:
    json_paths: list[Path] = []
    parquet_paths: list[Path] = []
    for path in paths:
        if path.is_dir():
            json_paths.extend(sorted(path.rglob("*.json")))
            parquet_paths.extend(sorted(path.rglob("*.parquet")))
        elif path.is_file():
            if path.suffix.lower() == ".parquet":
                parquet_paths.append(path)
            else:
                json_paths.append(path)
        else:
            raise FileNotFoundError(f"Trace path not found: {path}")
    if json_paths and parquet_paths:
        raise ValueError(
            "Input contains both JSON and Parquet traces; pass one format at a time."
        )
    if parquet_paths:
        return discover_parquet_traces(parquet_paths)
    return discover_json_traces(json_paths)


def discover_hf_dataset_traces(
    dataset_id: str,
    subset: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
) -> list[dict]:
    local_candidate = Path(dataset_id).expanduser()
    if local_candidate.exists():
        dataset_root = local_candidate.resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required for --dataset; install "
                "eval_pipeline/requirements.txt"
            ) from exc
        dataset_root = Path(
            snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=[
                    f"parquet/{subset}/*.parquet",
                    f"data/{subset}/*.parquet",
                ],
            )
        )

    parquet_paths = sorted((dataset_root / "parquet" / subset).glob("*.parquet"))
    if not parquet_paths:
        parquet_paths = sorted((dataset_root / "data" / subset).glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No Parquet shards for subset {subset!r} under {dataset_root}"
        )
    return discover_parquet_traces(parquet_paths)


def is_valid_norm_point(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return False
    return 0 <= x <= 1000 and 0 <= y <= 1000


def image_to_data_url(
    image_source: Path | dict | bytes,
    step: dict | None = None,
    *,
    base_dir: Path | None = None,
) -> str:
    source_name = ""
    if isinstance(image_source, Path):
        source_name = str(image_source)
        image_bytes = image_source.read_bytes()
    elif isinstance(image_source, bytes):
        image_bytes = image_source
    elif isinstance(image_source, dict):
        source_name = str(image_source.get("path") or "")
        image_bytes = image_source.get("bytes")
        if image_bytes is None and source_name:
            image_path = Path(source_name)
            if not image_path.is_absolute() and base_dir is not None:
                image_path = base_dir / image_path
            image_bytes = image_path.read_bytes()
    else:
        raise TypeError(f"Unsupported image value: {type(image_source).__name__}")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError(f"Embedded image has no bytes: {source_name or '[unnamed]'}")

    mime_type, _ = mimetypes.guess_type(source_name)
    if step and is_valid_norm_point(step.get("coordinate")):
        with Image.open(io.BytesIO(image_bytes)) as raw_image:
            image = raw_image.convert("RGB")
            width, height = image.size
            nx, ny = step["coordinate"]
            x = float(nx) / 1000.0 * width
            y = float(ny) / 1000.0 * height
            radius = max(10, int(min(width, height) * 0.05))
            line_width = max(4, int(min(width, height) * 0.008))
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                outline="red",
                width=line_width,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        mime_type = "image/png"
    else:
        mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def norm(value: object) -> str:
    return str(value).strip() if value is not None else ""


def build_history_text(trajectory: list[dict], include_thought: bool = True) -> str:
    lines: list[str] = []
    for position, step in enumerate(trajectory):
        step_index = step.get("step_index", position)
        action = norm(step.get("action")) or "[empty]"
        if include_thought:
            thought = norm(step.get("thought")) or "[empty]"
            lines.append(f"Step {step_index}:\nThought: {thought}\nAction: {action}")
        else:
            lines.append(f"Step {step_index}:\nAction: {action}")
    return "\n\n".join(lines)


def resolve_screenshots(
    trace: dict,
    trajectory: list[dict],
    first_n: int,
    last_n: int,
) -> tuple[list[dict], int]:
    """Resolve selected screenshots, skipping explicitly unavailable frames."""
    if not trajectory:
        return [], 0
    total = len(trajectory)
    indices: set[int] = set()
    if first_n > 0:
        indices.update(range(min(first_n, total)))
    if last_n > 0:
        indices.update(range(max(0, total - last_n), total))
    embedded_screenshots: list[object] | None = None
    parquet_path_value = trace.get("source_parquet_path")
    if parquet_path_value:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read embedded screenshots") from exc
        parquet_path = Path(str(parquet_path_value))
        parquet_file = pq.ParquetFile(parquet_path)
        table = parquet_file.read_row_group(
            int(trace["source_parquet_row_group"]), columns=["screenshots"]
        )
        rows = table.to_pylist()
        embedded_screenshots = rows[int(trace["source_parquet_row_index"])][
            "screenshots"
        ]
        if len(embedded_screenshots) != total:
            raise ValueError(
                f"Screenshot/trajectory length mismatch for {trace['trace_id']}: "
                f"{len(embedded_screenshots)} != {total}"
            )

    json_dir = None
    if trace.get("source_json_path"):
        json_dir = Path(str(trace["source_json_path"])).resolve().parent
    items: list[dict] = []
    missing = 0
    for index in sorted(indices):
        step = trajectory[index]
        if embedded_screenshots is not None:
            image_source = embedded_screenshots[index]
            if image_source is None:
                missing += 1
                continue
            items.append(
                {
                    "image_source": image_source,
                    "image_base_dir": Path(str(parquet_path_value)).parent,
                    "step": step,
                }
            )
        else:
            relative_path = step.get("screenshot_path")
            if not relative_path or not isinstance(relative_path, str):
                missing += 1
                continue
            if json_dir is None:
                raise ValueError(f"No screenshot source for trace {trace['trace_id']}")
            absolute_path = (json_dir / relative_path).resolve()
            if not absolute_path.is_file():
                raise FileNotFoundError(f"Screenshot not found: {absolute_path}")
            items.append({"image_source": absolute_path, "step": step})
    return items, missing


def build_messages(
    trace: dict,
    system_prompt: str,
    first_n: int,
    last_n: int,
    *,
    history_mode: str,
    show_action_mark: bool,
    include_thought: bool,
) -> tuple[list[dict], int]:
    trajectory = trace.get("trajectory") or []
    items, missing = resolve_screenshots(trace, trajectory, first_n, last_n)
    if history_mode == "full":
        history_text = build_history_text(trajectory, include_thought)
    elif history_mode == "selected":
        history_text = build_history_text(
            [item["step"] for item in items], include_thought
        )
    else:
        history_text = None

    content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(
                    item["image_source"],
                    item["step"] if show_action_mark else None,
                    base_dir=item.get("image_base_dir"),
                )
            },
        }
        for item in items
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
    ], missing


def backoff_sleep(attempt: int, base: float = 2.0, cap: float = 60.0) -> None:
    delay = min(base**attempt, cap)
    time.sleep(delay * (0.5 + 0.5 * random.random()))


def call_judge_api(
    messages: list[dict],
    base_url: str,
    api_key: str,
    model_name: str,
    *,
    api_style: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
    max_retries: int,
) -> tuple[str, bool]:
    """Return (response_text, truncated); truncated means the completion
    budget cut the response off before the model finished."""
    last_error: Exception | None = None
    # ``max_retries`` counts retries after the initial request. Even zero
    # retries must still issue one API call.
    max_attempts = max_retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            if api_style == "anthropic":
                system_prompt, native_messages = to_anthropic_messages(messages)
                client = Anthropic(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=0,
                )
                completion = client.messages.create(
                    model=model_name,
                    system=system_prompt,
                    messages=native_messages,
                    temperature=temperature,
                    max_tokens=max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS,
                )
                text = "\n".join(
                    block.text
                    for block in completion.content
                    if getattr(block, "type", None) == "text"
                    and getattr(block, "text", None)
                )
                truncated = completion.stop_reason == "max_tokens"
            else:
                client = OpenAI(api_key=api_key, base_url=base_url)
                kwargs: dict = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": timeout,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                completion = client.chat.completions.create(**kwargs)
                if not completion.choices:
                    raise RuntimeError("API response contains no choices")
                choice = completion.choices[0]
                text = choice.message.content or ""
                truncated = choice.finish_reason == "length"
            if not text.strip() and not truncated:
                # A truncated-empty response (budget spent on hidden reasoning)
                # is returned so the caller can escalate the budget; an empty
                # response with a normal finish is a transient API failure.
                raise RuntimeError("Empty model response")
            return text, truncated
        except Exception as exc:
            last_error = exc
            logger.debug("Attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                backoff_sleep(attempt)
    raise RuntimeError(f"Judge API failed after {max_attempts} attempts: {last_error}")


def escalate_max_tokens(current: int | None, api_style: str) -> int | None:
    """Next completion budget to try after a truncated response, or None when
    the budget is already at the cap."""
    if current is None:
        current = DEFAULT_ANTHROPIC_MAX_TOKENS if api_style == "anthropic" else 0
    escalated = min(
        max(current * ESCALATION_FACTOR, ESCALATED_MAX_TOKENS_FLOOR),
        ESCALATED_MAX_TOKENS_CAP,
    )
    return escalated if escalated > current else None


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert the internal OpenAI-style multimodal request to Anthropic blocks."""
    system_prompt = ""
    converted: list[dict] = []
    for message in messages:
        if message.get("role") == "system":
            system_prompt = str(message.get("content") or "")
            continue
        blocks: list[dict] = []
        content = message.get("content")
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    blocks.append({"type": "text", "text": str(item.get("text") or "")})
                elif item.get("type") == "image_url":
                    data_url = str((item.get("image_url") or {}).get("url") or "")
                    match = re.fullmatch(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
                    if not match:
                        raise ValueError("Anthropic mode requires base64 data-URL images")
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": match.group(1),
                                "data": match.group(2),
                            },
                        }
                    )
        converted.append({"role": message.get("role", "user"), "content": blocks})
    return system_prompt, converted


JUDGE_LINE_RE = re.compile(
    r"^[\s>*_`#-]*judge\s*:\s*\**\s*(success|fail)\w*",
    flags=re.IGNORECASE | re.MULTILINE,
)
JUDGE_ANY_RE = re.compile(
    r"\bjudge\s*:\s*\**\s*(success|fail)\w*", flags=re.IGNORECASE
)
THOUGHT_RE = re.compile(
    r"thought\s*:\s*(.*?)(?=\n\s*judge\s*:)", flags=re.IGNORECASE | re.DOTALL
)


def last_match(pattern: re.Pattern, text: str):
    match = None
    for candidate in pattern.finditer(text):
        match = candidate
    return match


def extract_judge_label(text: str) -> str:
    match = last_match(JUDGE_LINE_RE, text or "") or last_match(JUDGE_ANY_RE, text or "")
    if not match:
        raise ValueError(f"Cannot parse judge label from: {(text or '')[:200]!r}")
    return "SUCCESS" if match.group(1).upper().startswith("SUCC") else "FAIL"


def extract_judge_thought(text: str) -> str | None:
    match = THOUGHT_RE.search(text or "")
    if match:
        return match.group(1).strip() or None
    return (text or "").strip() or None


def sanitize_error(exc: Exception) -> str:
    """Keep diagnostic value without persisting credentials echoed by APIs."""
    message = str(exc)
    message = re.sub(r"\bsk-[A-Za-z0-9_*.-]+", "[REDACTED_API_KEY]", message)
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,'\"}]+",
        r"\1[REDACTED]",
        message,
    )
    return message


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def output_path_for_model(output_dir: Path, version: str, model_name: str) -> Path:
    return output_dir / f"judge_{version}_binary_{safe_model_name(model_name)}.jsonl"


def load_existing_results(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            if record.get("trace_id"):
                records[str(record["trace_id"])] = record
    return records


def write_results_jsonl(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for trace_id in sorted(records):
            handle.write(json.dumps(records[trace_id], ensure_ascii=False) + "\n")
    temporary.replace(path)


def judge_single_trace(
    trace: dict,
    args: argparse.Namespace,
    system_prompt: str,
    prompt_version: str,
    base_url: str,
    api_key: str,
    model_name: str,
) -> dict:
    started = time.time()
    result = {
        "trace_id": trace["trace_id"],
        "task_id": trace["task_id"],
        "platform": trace["platform"],
        "agent": trace["agent"],
        "instruction": trace["instruction"],
        "num_steps": len(trace.get("trajectory") or []),
        "first_n": "all" if args.first_n >= ALL_STEPS else args.first_n,
        "last_n": "all" if args.last_n >= ALL_STEPS else args.last_n,
        "history_mode": args.history,
        "show_action_mark": not args.no_mark,
        "include_thought": not args.no_thought,
        "version": args.version,
        "prompt_version": prompt_version,
        "judge_model": model_name,
        "api_style": args.api_style,
        "human_label": trace["human_label"],
        "judge_thought": None,
        "judge_label": None,
        "judge_raw_response": None,
        "response_truncated": None,
        "max_tokens_used": None,
        "binary_correct": 0,
        "selected_screenshot_count": 0,
        "missing_selected_screenshots": 0,
        "status": "error",
        "error": None,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": None,
    }

    try:
        messages, missing = build_messages(
            trace,
            system_prompt,
            args.first_n,
            args.last_n,
            history_mode=args.history,
            show_action_mark=not args.no_mark,
            include_thought=not args.no_thought,
        )
        result["selected_screenshot_count"] = sum(
            1 for item in messages[1]["content"] if item.get("type") == "image_url"
        )
        result["missing_selected_screenshots"] = missing
    except Exception as exc:
        result["error"] = sanitize_error(exc)
        result["elapsed_seconds"] = round(time.time() - started, 3)
        return result

    last_error: Exception | None = None
    effective_max_tokens = args.max_tokens
    plain_retries = PARSE_RETRIES
    truncation_retries = TRUNCATION_RETRIES
    truncated = False
    attempt = 0
    while True:
        attempt += 1
        try:
            raw_response, truncated = call_judge_api(
                messages,
                base_url,
                api_key,
                model_name,
                api_style=args.api_style,
                temperature=args.temperature,
                max_tokens=effective_max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            result["judge_raw_response"] = raw_response
            result["response_truncated"] = truncated
            result["max_tokens_used"] = effective_max_tokens
            result["judge_thought"] = extract_judge_thought(raw_response)
            result["judge_label"] = extract_judge_label(raw_response)
            result["binary_correct"] = int(
                result["judge_label"] == trace["human_label"]
            )
            result["status"] = "ok"
            last_error = None
            break
        except ValueError as exc:
            last_error = exc
            if truncated:
                # Retrying a truncated response at the same budget cannot help;
                # either escalate the budget or give up.
                if truncation_retries > 0:
                    escalated = escalate_max_tokens(
                        effective_max_tokens, args.api_style
                    )
                    if escalated is not None:
                        effective_max_tokens = escalated
                        truncation_retries -= 1
                        continue
                break
            if plain_retries > 0:
                plain_retries -= 1
                backoff_sleep(attempt, base=2.0, cap=10.0)
                continue
            break
        except Exception as exc:
            last_error = exc
            break
    if last_error is not None:
        result["error"] = sanitize_error(last_error)
    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def run_model(
    model_name: str,
    traces: list[dict],
    args: argparse.Namespace,
    system_prompt: str,
    prompt_version: str,
    base_url: str,
    api_key: str,
) -> Path:
    output_path = output_path_for_model(args.output_dir, args.version, model_name)
    existing = load_existing_results(output_path)
    expected_ids = {trace["trace_id"] for trace in traces}
    existing = {key: value for key, value in existing.items() if key in expected_ids}
    done_ids = {
        trace_id
        for trace_id, record in existing.items()
        if record.get("status") == "ok"
    }
    pending = [trace for trace in traces if trace["trace_id"] not in done_ids]

    print(f"\n{'=' * 64}")
    print(f"Model:       {model_name}")
    print(f"Prompt:      {prompt_version}")
    print(f"Output:      {output_path}")
    print(f"Traces:      {len(traces)} | Resumed: {len(done_ids)} | Pending: {len(pending)}")
    print(f"{'=' * 64}", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                judge_single_trace,
                trace,
                args,
                system_prompt,
                prompt_version,
                base_url,
                api_key,
                model_name,
            )
            for trace in pending
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {model_name}"):
            result = future.result()
            existing[result["trace_id"]] = result
            write_results_jsonl(output_path, existing)
    return output_path


def compute_metrics(traces: list[dict], result_path: Path) -> dict:
    results = load_existing_results(result_path)
    gold_counts = {"SUCCESS": 0, "FAIL": 0}
    correct_counts = {"SUCCESS": 0, "FAIL": 0}
    prediction_counts = {"SUCCESS": 0, "FAIL": 0}
    covered = 0
    missing_or_error = 0

    for trace in traces:
        label = trace["human_label"]
        gold_counts[label] += 1
        result = results.get(trace["trace_id"])
        if not result or result.get("status") != "ok":
            missing_or_error += 1
            continue
        prediction = result.get("judge_label")
        if prediction not in {"SUCCESS", "FAIL"}:
            missing_or_error += 1
            continue
        covered += 1
        prediction_counts[prediction] += 1
        if prediction == label:
            correct_counts[label] += 1

    total = len(traces)
    correct = sum(correct_counts.values())
    success_recall = (
        correct_counts["SUCCESS"] / gold_counts["SUCCESS"]
        if gold_counts["SUCCESS"]
        else None
    )
    fail_recall = (
        correct_counts["FAIL"] / gold_counts["FAIL"]
        if gold_counts["FAIL"]
        else None
    )
    balanced_accuracy = (
        (success_recall + fail_recall) / 2
        if success_recall is not None and fail_recall is not None
        else None
    )
    return {
        "n": total,
        "gold": gold_counts,
        "predictions_on_covered": prediction_counts,
        "covered": covered,
        "missing_or_error": missing_or_error,
        "coverage": covered / total,
        "accuracy": correct / total,
        "success_recall": success_recall,
        "fail_recall": fail_recall,
        "balanced_accuracy": balanced_accuracy,
        "error_policy": "missing, API-error, and unparseable outputs count as incorrect",
    }


def print_and_save_report(
    model_paths: dict[str, Path], traces: list[dict], args: argparse.Namespace
) -> None:
    print(f"\n{'#' * 72}\n#  Strict binary metrics\n{'#' * 72}")
    for model_name, path in model_paths.items():
        metrics = compute_metrics(traces, path)
        print(f"\n{model_name} (N={metrics['n']})")
        def percent(value: float | None) -> str:
            return "N/A" if value is None else f"{value * 100:.1f}%"

        print(f"  Coverage:          {percent(metrics['coverage'])}")
        print(f"  Balanced Accuracy: {percent(metrics['balanced_accuracy'])}")
        print(f"  Accuracy:          {percent(metrics['accuracy'])}")
        print(f"  SUCCESS Recall:    {percent(metrics['success_recall'])}")
        print(f"  FAIL Recall:       {percent(metrics['fail_recall'])}")
        report_path = args.output_dir / (
            f"metrics_{args.version}_binary_{safe_model_name(model_name)}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "subset": args.subset,
            "model": model_name,
            "version": args.version,
            "metrics": metrics,
            "results_file": path.name,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  Metrics JSON:      {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--traces",
        nargs="+",
        type=Path,
        help="Local legacy JSON files/directories or Parquet shards/directories.",
    )
    inputs.add_argument(
        "--dataset",
        help="Hugging Face dataset ID or local dataset root containing Parquet shards.",
    )
    parser.add_argument("--models", nargs="+", required=True, metavar="MODEL")
    parser.add_argument("--subset", choices=["full", "hard", "custom"], default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument(
        "--api_style",
        choices=["openai", "anthropic"],
        default="openai",
        help="Endpoint wire protocol (default: openai).",
    )
    parser.add_argument(
        "--base_url",
        default=None,
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("JUDGE_API_KEY"),
    )
    parser.add_argument("--prompt_file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--version", default="run")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--first_n", type=int_or_all, default=DEFAULT_FIRST_N)
    parser.add_argument("--last_n", type=int_or_all, default=DEFAULT_LAST_N)
    parser.add_argument("--history", choices=["full", "selected", "none"], default="full")
    parser.add_argument("--no_thought", action="store_true")
    parser.add_argument("--no_mark", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subset is None:
        args.subset = "full" if args.dataset else "custom"
    if args.dataset and args.subset == "custom":
        raise SystemExit("--dataset requires --subset full or --subset hard.")
    if args.api_key is None:
        if args.api_style == "anthropic":
            args.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            args.api_key = os.environ.get("OPENAI_API_KEY")
    if args.base_url is None:
        if args.api_style == "anthropic":
            args.base_url = (
                os.environ.get("JUDGE_BASE_URL")
                or os.environ.get("ANTHROPIC_BASE_URL")
                or "https://api.anthropic.com"
            )
        else:
            args.base_url = (
                os.environ.get("JUDGE_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )
    if not args.api_key:
        raise SystemExit(
            "No API key. Pass --api_key or set JUDGE_API_KEY, OPENAI_API_KEY, "
            "or ANTHROPIC_API_KEY."
        )
    if args.max_retries < 0:
        raise SystemExit("--max_retries must be zero or greater.")
    if args.first_n == 0 and args.last_n == 0:
        raise SystemExit("At least one of --first_n or --last_n must be greater than zero.")

    system_prompt, prompt_version = load_system_prompt(args.prompt_file)
    if args.dataset:
        traces = discover_hf_dataset_traces(
            args.dataset,
            args.subset,
            revision=args.dataset_revision,
            cache_dir=args.cache_dir,
        )
    else:
        traces = discover_traces(args.traces)
    if not traces:
        raise SystemExit("No binary OSReward traces found.")
    if args.limit is not None:
        traces = traces[: args.limit]
    print(f"Prompt: {args.prompt_file} ({prompt_version})")
    print(f"Traces: {len(traces)} | Subset: {args.subset}")

    model_paths: dict[str, Path] = {}
    for model_name in args.models:
        model_paths[model_name] = run_model(
            model_name,
            traces,
            args,
            system_prompt,
            prompt_version,
            args.base_url,
            args.api_key,
        )
    print_and_save_report(model_paths, traces, args)


if __name__ == "__main__":
    main()

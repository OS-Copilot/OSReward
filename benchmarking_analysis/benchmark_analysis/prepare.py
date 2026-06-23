"""Turn raw agent runs into inspectable, judge-ready inputs.

Output layout (under the dataset root)::

    analysis/<platform>/judge_ready/<agent>/<domain>__<task_id>.json
    analysis/<platform>/images/<agent>/<task_id>/<last-N marked PNGs>
    analysis/<platform>/instruction_map.json    (task_id -> instruction, shared)

Each judge-ready record has a uniform schema::

    {trace_id, agent, platform, domain, instruction, raw_score, golden_label,
     n_steps, n_images_used, selected_steps, history, images:[relpath, ...]}

History is the full per-step "Thought: ... Action: ..." log; the last-N step
screenshots get our own red action circle drawn from the parsed click coords.
For desktop benches the golden label is SUCCESS iff the verifier score reaches
``success_threshold``; for webarena it comes from the recorded ``task_success``.
"""
import glob
import io
import json
import os

from PIL import Image, ImageDraw

from . import adapters, config

SUCCESS_THRESHOLD = 0.99


def draw_mark(png_bytes, xy):
    """Draw a red action circle at ``xy`` (no-op if ``xy`` is None)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if xy is not None:
        w, h = img.size
        r = max(12, int(min(w, h) * 0.03))
        ImageDraw.Draw(img).ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r],
                                    outline="red", width=6)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def draw_mark_norm(png_bytes, xy_norm):
    """Like draw_mark, but xy is in 0..1000 normalized coords (mobile)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if xy_norm is not None:
        w, h = img.size
        x, y = xy_norm[0] / 1000.0 * w, xy_norm[1] / 1000.0 * h
        r = max(12, int(min(w, h) * 0.04))
        ImageDraw.Draw(img).ellipse([x - r, y - r, x + r, y + r], outline="red", width=6)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _emit(jr_dir, img_dir, subdir, json_name, rec, frames, mark_fn=draw_mark):
    """Mark+write the selected frames and the judge-ready JSON for one task."""
    task_img_dir = os.path.join(img_dir, subdir)
    os.makedirs(task_img_dir, exist_ok=True)
    rel_imgs = []
    for data, name, xy in frames:
        try:
            data = mark_fn(data, xy)
        except Exception:
            pass
        dst = os.path.join(task_img_dir, os.path.basename(name))
        with open(dst, "wb") as f:
            f.write(data)
        rel_imgs.append(os.path.relpath(dst, jr_dir))
    rec["images"] = rel_imgs
    rec["n_images_used"] = len(rel_imgs)
    with open(os.path.join(jr_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)


def _dirs(platform, agent):
    root = config.platform_dir(platform)
    jr_dir = os.path.join(root, "judge_ready", agent)
    img_dir = os.path.join(root, "images", agent)
    os.makedirs(jr_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    return root, jr_dir, img_dir


def prepare_desktop(platform, agent, source, last_n=5, success_threshold=SUCCESS_THRESHOLD,
                    ossymphony_full=False, instruction_overrides=None, limit=0):
    """Prepare osworld / windows / macos from a Zip- or DirSource."""
    root, jr_dir, img_dir = _dirs(platform, agent)
    map_path = os.path.join(root, "instruction_map.json")
    instr_map = json.load(open(map_path)) if os.path.exists(map_path) else {}
    overrides = instruction_overrides or {}
    stats = {"ok": 0, "skipped": 0, "no_instruction": 0,
             "gold": {"SUCCESS": 0, "FAIL": 0}, "formats": set(), "domains": {}}

    tasks = source.tasks()
    if limit:
        tasks = tasks[:limit]
    for domain, task_id, handle in tasks:
        score = adapters.parse_score(source.read_text(handle, "result.txt"))
        traj = source.read_text(handle, "traj.jsonl")
        if score is None or not traj:
            stats["skipped"] += 1
            continue
        steps = adapters.load_steps(traj)
        fmt = adapters.detect_format(steps)
        F = adapters.FORMATS[fmt]
        stats["formats"].add(fmt)

        instruction = overrides.get(task_id) or F["instruction"](steps, source.read_text, handle)
        if not instruction:
            instruction = instr_map.get(task_id)
        if instruction and task_id not in instr_map:
            instr_map[task_id] = instruction
        if not instruction:
            stats["no_instruction"] += 1

        lines, shots = [], []
        for i, s in enumerate(steps, 1):
            th, ac, xy = F["step"](s, ossymphony_full)
            lines.append(f"Step {i}: Thought: {th} Action: {ac}")
            if s.get("screenshot_file"):
                shots.append((i, s["screenshot_file"], xy))
        last_shots = shots[-last_n:] if len(shots) > last_n else shots

        frames = []
        for _i, sf, xy in last_shots:
            data = source.read(handle, sf)
            if data is not None:
                frames.append((data, sf, xy))

        golden = "SUCCESS" if score >= success_threshold else "FAIL"
        rec = {"trace_id": f"{domain}/{task_id}", "agent": agent, "platform": platform,
               "domain": domain, "task_id": task_id, "format": fmt,
               "instruction": instruction, "raw_score": score, "golden_label": golden,
               "n_steps": len(steps), "selected_steps": [i for i, _, _ in last_shots],
               "history": "\n".join(lines)}
        _emit(jr_dir, img_dir, task_id, f"{domain}__{task_id}.json", rec, frames)
        stats["ok"] += 1
        stats["gold"][golden] += 1
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1

    json.dump(instr_map, open(map_path, "w"), ensure_ascii=False, indent=0)
    stats["formats"] = sorted(stats["formats"])
    return stats


def prepare_webarena(agent, set_dir, last_n=5, limit=0):
    """Prepare webarena from a merged-JSONL rollout set directory."""
    _root, jr_dir, img_dir = _dirs("webarena", agent)
    stats = {"ok": 0, "skipped": 0, "gold": {"SUCCESS": 0, "FAIL": 0}, "domains": {"web": 0}}
    for r in adapters.webarena_records(set_dir, last_n):
        if limit and stats["ok"] >= limit:
            break
        if r["golden_label"] not in ("SUCCESS", "FAIL"):
            stats["skipped"] += 1
            continue
        safe = str(r["trace_id"]).replace("/", "__")
        rec = {"trace_id": r["trace_id"], "agent": agent, "platform": "webarena",
               "domain": "web", "instruction": r["instruction"],
               "golden_label": r["golden_label"], "n_steps": r["n_steps"],
               "history": r["history"]}
        _emit(jr_dir, img_dir, safe, f"{safe}.json", rec, r["read_frames"]())
        stats["ok"] += 1
        stats["gold"][r["golden_label"]] += 1
        stats["domains"]["web"] += 1
    return stats


def prepare_webarena_legacy(agent, set_dir, judge_subdir=None, limit=0):
    """Prepare webarena from the legacy 'judgements' format (pre-built messages).

    The user prompt is already composed, so it is stored verbatim as ``user_text``
    and used as-is at judge time; screenshots are copied in (no click mark).
    """
    _root, jr_dir, img_dir = _dirs("webarena", agent)
    stats = {"ok": 0, "skipped": 0, "gold": {"SUCCESS": 0, "FAIL": 0}, "domains": {"web": 0}}
    for r in adapters.webarena_judgements_records(set_dir, judge_subdir):
        if limit and stats["ok"] >= limit:
            break
        if r["golden_label"] not in ("SUCCESS", "FAIL"):
            stats["skipped"] += 1
            continue
        frames = []
        for p in r["image_paths"]:
            try:
                with open(p, "rb") as fh:
                    frames.append((fh.read(), os.path.basename(p), None))
            except OSError:
                pass
        safe = str(r["trace_id"]).replace("/", "__")
        rec = {"trace_id": r["trace_id"], "agent": agent, "platform": "webarena",
               "domain": "web", "instruction": r["instruction"],
               "golden_label": r["golden_label"], "user_text": r["user_text"]}
        _emit(jr_dir, img_dir, safe, f"{safe}.json", rec, frames)
        stats["ok"] += 1
        stats["gold"][r["golden_label"]] += 1
        stats["domains"]["web"] += 1
    return stats


def prepare_androidworld(agent, jsonl_path, last_n=5, root=None, limit=0):
    """Prepare AndroidWorld (mobile) from a normalized merged-JSONL file.

    ``jsonl_path`` points at the trajectories file (e.g. results/merged_*.jsonl);
    screenshots resolve against ``root`` (default: two levels up). Click marks are
    drawn from 0..1000 normalized coordinates.
    """
    _root, jr_dir, img_dir = _dirs("androidworld", agent)
    stats = {"ok": 0, "skipped": 0, "gold": {"SUCCESS": 0, "FAIL": 0}, "domains": {"mobile": 0}}
    for r in adapters.androidworld_records(jsonl_path, last_n, root):
        if limit and stats["ok"] >= limit:
            break
        if r["golden_label"] not in ("SUCCESS", "FAIL"):
            stats["skipped"] += 1
            continue
        safe = str(r["trace_id"]).replace("/", "__")
        rec = {"trace_id": r["trace_id"], "agent": agent, "platform": "androidworld",
               "domain": "mobile", "instruction": r["instruction"],
               "golden_label": r["golden_label"], "n_steps": r["n_steps"],
               "history": r["history"]}
        _emit(jr_dir, img_dir, safe, f"{safe}.json", rec, r["read_frames"](),
              mark_fn=draw_mark_norm)
        stats["ok"] += 1
        stats["gold"][r["golden_label"]] += 1
        stats["domains"]["mobile"] += 1
    return stats


def load_judge_ready(platform, agent):
    """Read prepared judge-ready records back for judging.

    Returns a list of dicts with absolute image paths, ready for
    :func:`benchmark_analysis.judge.build_messages`.
    """
    jr_dir = os.path.join(config.platform_dir(platform), "judge_ready", agent)
    recs = []
    for f in sorted(glob.glob(os.path.join(jr_dir, "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        recs.append({
            "trace_id": r["trace_id"],
            "golden_label": r.get("golden_label"),
            "instruction": r.get("instruction"),
            "user_text": r.get("user_text"),
            "history": r.get("history", ""),
            "images": [os.path.normpath(os.path.join(jr_dir, rel))
                       for rel in r.get("images", [])],
            "extra": {"platform": platform, "agent": agent, "domain": r.get("domain"),
                      "raw_score": r.get("raw_score"), "n_steps": r.get("n_steps"),
                      "n_images": len(r.get("images", []))},
        })
    return recs

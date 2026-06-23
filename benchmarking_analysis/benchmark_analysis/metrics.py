"""Binary judge scoring — the single source of truth for metrics.

Every judged row carries ``judge`` (SUCCESS / FAIL / None) and ``golden_label``
(SUCCESS / FAIL / None). The judge is scored as a binary classifier against the
golden label:

    acc  = (TP + TN) / N
    sRec = TP / (TP + FN)   success-recall  (high -> not too strict)
    fRec = TN / (TN + FP)   fail-recall     (high -> not too lenient)

where, with gold/judge in {SUCCESS, FAIL}:
    TP gold=SUCCESS judge=SUCCESS    FP gold=FAIL    judge=SUCCESS (lenient)
    TN gold=FAIL    judge=FAIL       FN gold=SUCCESS judge=FAIL    (strict)
"""
import glob
import json
import os
import re

from . import config


def row_ok(r):
    """True if the row produced a usable judgement (no error, judge parsed)."""
    return (not r.get("error")) and (r.get("judge") is not None)


def dedup_rows(rows):
    """Keep one row per trace_id: prefer a usable judgement, else the last seen."""
    best = {}
    for r in rows:
        tid = r.get("trace_id")
        if tid not in best or row_ok(r) or not row_ok(best[tid]):
            best[tid] = r
    return list(best.values())


def load_done_ids(out_path):
    """trace_ids that already have a usable judgement (so failures get retried)."""
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if row_ok(r):
                done.add(r["trace_id"])
    return done


def compute_metrics(rows):
    """Confusion matrix + accuracy / success-recall / fail-recall over scored rows."""
    tp = fp = tn = fn = 0
    for r in rows:
        j, g = r.get("judge"), r.get("golden_label")
        if j is None or g is None:
            continue
        if g == "SUCCESS" and j == "SUCCESS":
            tp += 1
        elif g == "FAIL" and j == "SUCCESS":
            fp += 1
        elif g == "FAIL" and j == "FAIL":
            tn += 1
        elif g == "SUCCESS" and j == "FAIL":
            fn += 1
    n = tp + fp + tn + fn
    return {"n": n, "acc": (tp + tn) / n if n else float("nan"),
            "sRec": tp / (tp + fn) if (tp + fn) else float("nan"),
            "fRec": tn / (tn + fp) if (tn + fp) else float("nan"),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def report(out_path):
    """Print and return pooled metrics for one result file."""
    rows = dedup_rows([json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()])
    errs = sum(1 for r in rows if r.get("error"))
    unparsed = sum(1 for r in rows if not r.get("error") and r.get("judge") is None)
    m = compute_metrics(rows)
    toks = sum(r.get("total_tokens") or 0 for r in rows)
    print("\n==== REPORT ====")
    print(f"unique_traces={len(rows)} errors={errs} unparsed_judge={unparsed}")
    print(f"scored n={m['n']} | acc={m['acc']:.4f} | sRec={m['sRec']:.4f} | fRec={m['fRec']:.4f}")
    print(f"confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']} | total_tokens={toks}")
    return m


def parse_agent_model(fname, version, setting):
    """Split ``judge_<version>_<setting>_[<agent>_]<model>.jsonl`` -> (agent, model)."""
    m = re.match(rf"judge_{re.escape(version)}_{re.escape(setting)}_(.+)\.jsonl",
                 os.path.basename(fname))
    if not m:
        return None, None
    rest = m.group(1)
    for km in config.KNOWN_MODELS:
        if rest == km:
            return None, km
        if rest.endswith("_" + km):
            return rest[: -(len(km) + 1)], km
    return None, rest  # unknown model -> treat whole tail as the model name


def load_results(platform, version, setting):
    """Pool result rows per judge MODEL across a platform's rollout agents.

    Returns ``(stats, per_agent)`` where ``stats[model]`` and
    ``per_agent[(agent, model)]`` are metric dicts. Control arms in
    ``config.EXCLUDE_AGENTS`` are dropped from the pool.
    """
    rdir = os.path.join(config.platform_dir(platform), "results")
    pooled, per_agent = {}, {}
    for f in sorted(glob.glob(os.path.join(rdir, f"judge_{version}_{setting}_*.jsonl"))):
        agent, model = parse_agent_model(f, version, setting)
        if model is None or agent in config.EXCLUDE_AGENTS:
            continue
        rows = dedup_rows([json.loads(l) for l in open(f, encoding="utf-8") if l.strip()])
        pooled.setdefault(model, []).extend(rows)
        if agent:
            per_agent[(agent, model)] = compute_metrics(rows)
    stats = {m: compute_metrics(rows) for m, rows in pooled.items()}
    return stats, per_agent

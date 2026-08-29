# Trajectory Collection

Pipelines for collecting agent trajectories across platforms, used to build
SFT / reward-model training corpora.

| Platform | Directory | Status |
|----------|-----------|--------|
| Web | [`webtrail/`](webtrail/) | Working: task import, rollout, judging, curation, export |
| Windows | [`windows/`](windows/) | Working: WAA-based collect (screenshots + model outputs → JSON) |
| Ubuntu | [`ubuntu/`](ubuntu/) | Placeholder |
| Mobile (Android) | [`mobile/`](mobile/) | Working: AndroidWorld-based collect → extract → reformat → annotate |

Each platform directory is a self-contained pipeline following the same stage
layout as webtrail: tasks import → collect → judge → filter → export.

## Shared trajectory format and judge-SFT export

Platform collectors should emit canonical records with `trace_id`,
`instruction`, and ordered `trajectory` steps. Each screenshot uses
`state.screenshot_path`, relative to the trajectory input root. Judge JSONL
records use `trace_id`, `judge_model`, `judge_label`, and `judge_thought`.

[`export_judged_sft.py`](export_judged_sft.py) converts canonical records plus
`eval_pipeline` judge-result JSONL files into the exact five-field
OS-Shepherd multimodal SFT shape. It is an importable function, not a CLI:

```python
from trajectory_collection.export_judged_sft import export_judged_sft

export_judged_sft(
    trajectories="bundle/trajectories",
    judgments={
        "judge_a": "bundle/judgments/judge_a.jsonl",
        "judge_b": "bundle/judgments/judge_b.jsonl",
    },
    output_dir="bundle/sft",
    sampling="last5",  # or "incremental" for last1 through last5
    preferred_judge="judge_b",
)
```

The exporter keeps traces with a strict judge majority. It writes only
`dataset.json` and content-addressed images under
`osreward_rm_train_bundle/images/`; each row contains exactly `id`, `messages`,
`images`, `coordinates`, and `results`. No source-machine path is serialized.

# data/

Dataset root (override with `OOD_DATA_ROOT`). `raw/` holds the inputs you feed to
`prepare`; `analysis/` holds the outputs. A tiny `*-sample` per platform is
included under `raw/` as a format reference.

```
data/
├── raw/                                  # inputs to `prepare` (point --path here)
│   ├── osworld/<agent>/<domain>/<task_id>/{result.txt, traj.jsonl, runtime.log, step_*.png}
│   ├── windows/<agent>/<domain>/<task_id>/{result.txt, traj.jsonl, step_*.png}
│   ├── webarena/<rollout-set>/            # merged-JSONL (judged_*.jsonl + images/)
│   │                                      #   or legacy judgements/ (auto-detected)
│   └── androidworld/<root>/               # results/*.jsonl + <agent>/<task>/screenshot_*.png
└── analysis/<platform>/                  # outputs (created by the pipeline)
    ├── judge_ready/<agent>/<domain>__<task_id>.json
    ├── images/<agent>/<task_id>/*.png
    ├── results/judge_<version>_<setting>_<agent>_<model>.jsonl
    ├── data/*.csv  ├── figures/*.png
```

To run on your own data, place rollouts in the shapes above (or point `--path`
anywhere) and pass the matching `--platform`/`--agent`. See `../README.md`.

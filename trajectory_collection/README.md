# Trajectory Collection

Pipelines for collecting agent trajectories across platforms, used to build
SFT / reward-model training corpora.

| Platform | Directory | Status |
|----------|-----------|--------|
| Web | [`webtrail/`](webtrail/) | Working: task import, rollout, judging, curation, export |
| Windows | [`windows/`](windows/) | Placeholder |
| Ubuntu | [`ubuntu/`](ubuntu/) | Placeholder |
| Mobile (Android) | [`mobile/`](mobile/) | Placeholder |

Each platform directory is a self-contained pipeline following the same stage
layout as webtrail: tasks import → collect → judge → filter → export.

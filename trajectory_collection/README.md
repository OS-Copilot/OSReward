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

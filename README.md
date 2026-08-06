<p align="center">
  <img src="figures/osreward-banner.png" alt="OSReward" width="850"/>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2607.28609-b31b1b.svg)](https://arxiv.org/abs/2607.28609)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2607.28609)
[![🌐 Website](https://img.shields.io/badge/Website-🌐-informational)](https://os-copilot.github.io/OSReward-Home/)
[![🤗 Benchmark](https://img.shields.io/badge/🤗%20Benchmark-OSReward-ffc107)](https://huggingface.co/datasets/OS-Copilot/OSReward)
[![🤗 Corpus](https://img.shields.io/badge/🤗%20Corpus-OS--Shepherd--100K-ffc107)](https://huggingface.co/datasets/OS-Copilot/OS-Shepherd-100K)
[![🤗 Models](https://img.shields.io/badge/🤗%20Models-OS--Shepherd-ffc107)](https://huggingface.co/collections/OS-Copilot/osreward-and-os-shepherd)

Code, benchmark and data for "OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models"

## 🗞️ Updates

- **2026-07**: Initial release of our [paper](https://arxiv.org/abs/2607.28609) and [🌐 Project Page](https://os-copilot.github.io/OSReward-Home/). Benchmark, corpus, and model checkpoints are on the way! 🚀

## 📏 OSReward

VLM judges are now the de-facto reward signal behind CUA evaluation, data curation, and RL, yet their reliability had gone unexamined. OSReward measures it directly and fairly with human-gold, long-horizon trajectories, built from scratch so judge errors are not confounded with flaws in reused rollouts. Two derived views sharpen the probe: *OSReward-Hard* concentrates the cases current judges commonly fail, and *OSReward-Multi* scores efficiency and alignment beyond the binary verdict.

### Benchmark & Evaluation

The benchmark data ([🤗 OSReward](https://huggingface.co/datasets/OS-Copilot/OSReward)) and the evaluation harness (run any judge on OSReward / -Hard / -Multi and reproduce the leaderboard) are on the way.

### Collection Infrastructure

[`trajectory_collection/`](trajectory_collection/): the cross-platform pipelines behind the benchmark trajectories. Live-web ([`webtrail/`](trajectory_collection/webtrail/)), Windows ([`windows/`](trajectory_collection/windows/), WAA-based), and Android ([`mobile/`](trajectory_collection/mobile/), AndroidWorld-based) are available; Ubuntu is on the way.

### Analysis & Experiments

[`benchmarking_analysis/`](benchmarking_analysis/): the OOD generalization study. It scores how VLM judges agree with the verifiers of existing benchmarks (OSWorld, WindowsAgentArena, WebArena, AndroidWorld) via a prepare → judge → analyze pipeline, producing the accuracy / bias metrics and figures.

## 📚 OS-Shepherd-100K

An open corpus of reasoning-annotated CUA trajectory judgments for training and studying reward models, built by a construction pipeline shaped by the benchmark's findings. The corpus ([🤗 OS-Shepherd-100K](https://huggingface.co/datasets/OS-Copilot/OS-Shepherd-100K)) and its construction pipeline are on the way.

## 🐑 OS-Shepherd (9B / 35B)

Open reward models trained on OS-Shepherd-100K. They supply low-cost, stable, and reliable reward signals for evaluation, data curation, and RL, matching commercial judges at 30–60× lower cost than the frontier. Checkpoints ([🤗 OS-Shepherd](https://huggingface.co/collections/OS-Copilot/osreward-and-os-shepherd)) together with training (SFT + RL) and inference code are on the way.

## 🚧 Open-source Roadmap

**Code**
- [x] Web trajectory collection (`trajectory_collection/webtrail/`)
- [x] Windows trajectory collection (`trajectory_collection/windows/`)
- [x] Android trajectory collection (`trajectory_collection/mobile/`)
- [ ] Ubuntu trajectory collection (`trajectory_collection/ubuntu/`)
- [x] Judge analysis on OOD benchmarks (`benchmarking_analysis/`)
- [ ] OSReward evaluation harness: run any judge on OSReward / -Hard / -Multi and reproduce the leaderboard
- [ ] OS-Shepherd-100K construction pipeline (judgment synthesis + annotation tooling)
- [ ] OS-Shepherd training (SFT + RL) and inference code

**Data & models**
- [ ] OSReward benchmark
- [ ] OS-Shepherd-100K corpus
- [ ] OS-Shepherd-9B / OS-Shepherd-35B checkpoints

## 📃 Citation

```bibtex
@article{sun2026osreward,
  title={OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models},
  author={Qiushi Sun and Kanzhi Cheng and Yian Wang and Bowen Yang and Hang Yan and Liheng Chen and Fangzhi Xu and Zichen Ding and Nuo Chen and Jialin Cao and Xingdong Gong and Zehao Li and Kaiming Jin and Xinfeng Yuan and Zhoumianze Liu and Jingyang Gong and Zhangyue Yin and Jiahui Gao and Zhiyong Wu and Tianbao Xie and Jianbing Zhang and Ben Kao and Lingpeng Kong},
  journal={arXiv preprint arXiv:2607.28609},
  year={2026}
}
```

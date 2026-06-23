"""Reporting: metrics CSV + leaderboard / bias / error figures.

One unified set of plots for every platform. ``in_domain`` (a model -> accuracy
mapping) is optional; when given (e.g. for webarena) the leaderboard overlays the
same judge's in-domain accuracy and the OOD gap.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402

FAMILY_COLOR = {"Claude": "#FC5C30", "GPT": "#01AE58", "Gemini": "#3C90FF",
                "Qwen": "#7372FE", "Doubao": "#00A9BB", "InternLM": "#969DFF",
                "Kimi": "#FF63A0", "Other": "#B8C0FF"}

_FAMILY_PREFIX = [("claude", "Claude"), ("gpt", "GPT"), ("o4", "GPT"), ("o3", "GPT"),
                  ("gemini", "Gemini"), ("qwen", "Qwen"), ("doubao", "Doubao"),
                  ("intern", "InternLM"), ("kimi", "Kimi")]


def family_of(model):
    m = model.lower()
    for pre, fam in _FAMILY_PREFIX:
        if m.startswith(pre):
            return fam
    return "Other"


def _short(model):
    return model.replace("-preview", "").replace("-instruct", "")


def write_metrics_csv(platform, stats, per_agent, version, setting, in_domain=None):
    data_dir = os.path.join(config.platform_dir(platform), "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{platform}_{version}_{setting}_metrics.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("model,family,n,acc,sRec,fRec,tp,fp,tn,fn,in_domain_acc,ood_gap_pp\n")
        for m, s in sorted(stats.items(), key=lambda kv: -kv[1]["acc"]):
            ind = (in_domain or {}).get(m)
            gap = f"{s['acc'] * 100 - ind:.2f}" if ind is not None else ""
            f.write(f"{m},{family_of(m)},{s['n']},{s['acc']:.4f},{s['sRec']:.4f},"
                    f"{s['fRec']:.4f},{s['tp']},{s['fp']},{s['tn']},{s['fn']},"
                    f"{ind if ind is not None else ''},{gap}\n")
    if per_agent:
        ap = os.path.join(data_dir, f"{platform}_{version}_{setting}_by_agent.csv")
        with open(ap, "w", encoding="utf-8") as f:
            f.write("agent,model,n,acc,sRec,fRec\n")
            for (a, m), s in sorted(per_agent.items()):
                f.write(f"{a},{m},{s['n']},{s['acc']:.4f},{s['sRec']:.4f},{s['fRec']:.4f}\n")
    return path


def plot_leaderboard(platform, stats, version, setting, out_png, in_domain=None):
    from matplotlib.lines import Line2D
    items = sorted(stats.items(), key=lambda kv: -kv[1]["acc"])
    models = [m for m, _ in items]
    accs = [s["acc"] * 100 for _, s in items]
    colors = [FAMILY_COLOR[family_of(m)] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(models)), 6.2))
    ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.6, width=0.62, zorder=3)
    for xi, a in zip(x, accs):
        ax.text(xi, a - 2.2, f"{a:.1f}", ha="center", va="top", fontsize=10,
                fontweight="bold", color="white", zorder=6)

    ind_x, ind_y = [], []
    for xi, m in zip(x, models):
        ind = (in_domain or {}).get(m)
        if ind is not None:
            ax.hlines(ind, xi - 0.31, xi + 0.31, color="#333", linestyle="--",
                      linewidth=1.3, zorder=4)
            ind_x.append(xi)
            ind_y.append(ind)
    if ind_x:
        ax.scatter(ind_x, ind_y, marker="D", s=26, color="#333", zorder=5)

    labels = []
    for m in models:
        ind = (in_domain or {}).get(m)
        labels.append(f"{_short(m)}\nΔ{stats[m]['acc'] * 100 - ind:+.1f}"
                      if ind is not None else _short(m))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if in_domain else 18,
                       ha="center" if in_domain else "right", fontsize=8.5)
    ax.set_ylabel("Binary accuracy (%)")
    ax.set_ylim(0, 100)
    n_total = max((s["n"] for s in stats.values()), default=0)
    ax.set_title(f"{platform} OOD — judge binary accuracy "
                 f"({version}/{setting}, pooled n≈{n_total})")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fams = sorted({family_of(m) for m in models})
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in fams]
    labels_fam = list(fams)
    if ind_x:
        handles.append(Line2D([0], [0], marker="D", color="#333", linestyle="--",
                              markersize=7, linewidth=1.3))
        labels_fam.append("in-domain acc")
    ax.legend(handles, labels_fam, title="family", fontsize=8, loc="lower left",
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_bias(platform, stats, version, setting, out_png):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    for m, s in stats.items():
        ax.scatter(s["sRec"], s["fRec"], s=130, color=FAMILY_COLOR[family_of(m)],
                   edgecolor="black", zorder=3)
        ax.annotate(_short(m), (s["sRec"], s["fRec"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("sRec = P(judge SUCCESS | gold SUCCESS)  → strictness↓")
    ax.set_ylabel("fRec = P(judge FAIL | gold FAIL)  → leniency↓")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{platform} OOD — judge bias ({version}/{setting})\n"
                 "lower-right = lenient · upper-left = strict")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_errors(platform, stats, version, setting, out_png):
    items = sorted(stats.items(), key=lambda kv: -kv[1]["acc"])
    models = [m for m, _ in items]
    x = np.arange(len(models))
    fp = [s["fp"] for _, s in items]
    fn = [s["fn"] for _, s in items]
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(models)), 5))
    ax.bar(x, fp, color="#FC5C30", edgecolor="black",
           label="FP (judge SUCCESS, gold FAIL = lenient)")
    ax.bar(x, fn, bottom=fp, color="#3C90FF", edgecolor="black",
           label="FN (judge FAIL, gold SUCCESS = strict)")
    for xi, (m, s) in zip(x, items):
        ax.text(xi, s["fp"] + s["fn"] + 0.5, f"{s['fp'] + s['fn']}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in models], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("# errors")
    ax.set_title(f"{platform} OOD — error composition ({version}/{setting})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def make_figures(platform, stats, version, setting, in_domain=None):
    """Render the three standard figures; returns their paths."""
    fig_dir = os.path.join(config.platform_dir(platform), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sfx = f"{version}_{setting}"
    paths = [os.path.join(fig_dir, f"01_leaderboard_{sfx}.png"),
             os.path.join(fig_dir, f"02_bias_{sfx}.png"),
             os.path.join(fig_dir, f"03_errors_{sfx}.png")]
    plot_leaderboard(platform, stats, version, setting, paths[0], in_domain)
    plot_bias(platform, stats, version, setting, paths[1])
    plot_errors(platform, stats, version, setting, paths[2])
    return paths

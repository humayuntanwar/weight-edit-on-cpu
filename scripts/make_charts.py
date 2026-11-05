"""Charts from runs/summary.json (written by scripts/score.py). Every number plotted comes from the raw runs."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S = json.load(open(os.path.join(ROOT, "runs", "summary.json")))
OUT = os.path.join(ROOT, "images"); os.makedirs(OUT, exist_ok=True)
COLORS = {"base": "#2a78d6", "edited": "#eb6834", "agent_a1": "#1baf7a", "agent_a2": "#eda100", "agent_a3": "#e87ba4"}
LABELS = {"base": "base Qwen3.5-0.8B", "edited": "paper's edit", "agent_a1": "agent edit, attempt 1",
          "agent_a2": "agent edit, attempt 2", "agent_a3": "agent edit, attempt 3"}
COND = {("A", 4096): "paper's sampling\nthinking on, 4,096 cap", ("A", 8192): "paper's sampling\nthinking on, 8,192 cap",
        ("B", 4096): "model card thinking\nsettings, 4,096 cap", ("C", 4096): "model card non-thinking\nsettings, 4,096 cap"}
ORDER = [("A", 4096), ("B", 4096), ("C", 4096), ("A", 8192)]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 15, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": INK2, "figure.facecolor": "white", "axes.facecolor": "white"})


def grid_bars(metric, ci, title, ylabel, fname, note):
    G = {(g["cond"], g["cap"], g["model"]): g for g in S["groups"]}
    conds = [c for c in ORDER if any(k[:2] == c for k in G)]
    models = [m for m in COLORS if any(k[2] == m for k in G)]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        for i, c in enumerate(conds):
            g = G.get((c[0], c[1], m))
            if not g:
                continue
            x = i + (j - (len(models) - 1) / 2) * w
            v, lo, hi = 100 * g[metric], 100 * g[ci][0], 100 * g[ci][1]
            ax.bar(x, v, w * 0.92, color=COLORS[m], label=LABELS[m] if i == next(
                k for k, cc in enumerate(conds) if (cc[0], cc[1], m) in G) else None, zorder=3)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, lw=1.5, zorder=4)
            ax.text(x, hi + 1.2, f"{v:.0f}%\nn={g['generations']}", ha="center", va="bottom", fontsize=11, color=INK2)
    ax.set_xticks(range(len(conds)), [COND[c] for c in conds])
    ax.set_ylabel(ylabel); ax.set_ylim(0, 112); ax.set_title(title, fontsize=20, loc="left", pad=16)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=min(3, len(models)), fontsize=13)
    fig.text(0.01, 0.01, note, fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)


NOTE = ("MATH-500 subset (stratified by level, seed 0), CPU, llama.cpp Q8_0. Error bars: Wilson 95% over generations; "
        "n = generations (problems x samples).")
grid_bars("accuracy", "acc_ci95", "MATH-500 accuracy by setting", "accuracy (%)", "1_accuracy.png", NOTE)
grid_bars("hit_cap_share", "hit_cap_ci95", "Share of answers that ran into the token cap", "hit the cap (%)",
          "2_hit_cap.png", NOTE)
print("charts written to", OUT)

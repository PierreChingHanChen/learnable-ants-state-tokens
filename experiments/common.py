"""實驗共用工具：輸出路徑、CSV、圖表樣式、快速模式。"""
from __future__ import annotations

import csv
import os
import sys
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# QUICK=1 以較少重複次數／週期快速驗證流程；正式數據請用 QUICK=0
QUICK = os.environ.get("QUICK", "0") == "1"

# 參考調色盤（dataviz 技能 references/palette.md，light 模式，固定順序）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def _cjk_font():
    for name in ("Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans CJK SC",
                 "Microsoft JhengHei", "PingFang TC", "Heiti TC", "WenQuanYi Zen Hei"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            return name
    return None


def style():
    f = _cjk_font()
    plt.rcParams.update({
        "font.family": [f, "DejaVu Sans"] if f else ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "figure.dpi": 120, "savefig.dpi": 150,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


style()


def save_csv(name: str, rows: List[Dict]) -> str:
    path = os.path.join(RESULTS, name)
    if not rows:
        return path
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def save_fig(fig, name: str) -> str:
    path = os.path.join(FIGS, name)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_summary(section: str, lines: Iterable[str]) -> None:
    path = os.path.join(RESULTS, f"summary_{section}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- 平行執行與統計工具
def pmap(func, items, workers=None):
    """以多行程平行執行（func 必須是模組層級函式）。"""
    from concurrent.futures import ProcessPoolExecutor
    items = list(items)
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or len(items) <= 1:
        return [func(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(func, items, chunksize=1))


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0, stat=None):
    """平均數（或指定統計量）的 bootstrap 百分位信賴區間。"""
    import numpy as np
    x = np.asarray(x, dtype=float)
    stat = stat or np.mean
    rng = np.random.default_rng(seed)
    bs = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def holm(pvals):
    """Holm–Bonferroni 校正，回傳校正後 p 值（與輸入同順序）。"""
    import numpy as np
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def paired_stats(diff):
    """配對差異的摘要：平均、95% bootstrap CI、Cohen's d_z、Wilcoxon 符號等級檢定 p。"""
    import numpy as np
    from scipy.stats import wilcoxon
    d = np.asarray(diff, dtype=float)
    lo, hi = bootstrap_ci(d)
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else float("nan")
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "d_z": dz, "p": p, "n": len(d)}

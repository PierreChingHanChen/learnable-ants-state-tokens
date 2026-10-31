"""論文圖 2–5（英文標籤）。輸出到 FIGDIR（預設 figures/paper）。"""
import os, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, SURFACE, INK, INK2, GRID, style

style()
plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.titleweight": "bold"})
OUT = os.path.join(RESULTS, "la2")
FIGDIR = os.environ.get("FIGDIR", os.path.join(os.path.dirname(RESULTS), "figures", "paper"))
os.makedirs(FIGDIR, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GRAY, LGRAY = "#7a7975", "#b9b8b3"
STATES = ["η", "δ", "ε", "ν", "φ", "σ", "κ", "υ"]
SNAME = {"η": "η home", "δ": "δ unload", "ε": "ε explore", "ν": "ν vector", "φ": "φ follow trail", "σ": "σ search",
         "κ": "κ scan", "υ": "υ U-turn"}


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"), facecolor="white", bbox_inches="tight",
                metadata={"CreationDate": None})          # 不寫建立時間，重跑時 PDF 位元組相同
    fig.savefig(os.path.join(FIGDIR, name + ".png"), facecolor="white", dpi=160, bbox_inches="tight")
    plt.close(fig)


def label(ax, s):
    ax.text(-0.16, 1.03, s, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom")


# ------------------------------------------------------------------ Figure 2
def fig2():
    d = json.load(open(os.path.join(OUT, "r1_main.json")))
    s = d["summary"]
    occ = d["occupancy_heldout"]
    methods = [("ngram1", "lookup (current o)", "o", LGRAY, "none"), ("ngram8", "8-step n-gram", "s", GRAY, None),
               ("filter", "finite-state filter", "X", PURPLE, None), ("bo_probe", "bo linear probe", "D", GRAY, "none"),
               ("aux", "aux (state supervised, not fed back)", "^", ORANGE, None),
               ("gru_aux", "GRU-aux (recurrent memory)", "v", GREEN, None), ("dual", "dual (free-running)", "o", BLUE, None)]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ys = np.arange(len(STATES))[::-1]
    offs = np.linspace(-0.3, 0.3, len(methods))
    for (k, lab, mk, col, fc), off in zip(methods, offs):
        x = np.array([100 * s[k][st] for st in STATES])
        if "min" in s[k]:
            lo = np.array([100 * s[k]["min"][st] for st in STATES]); hi = np.array([100 * s[k]["max"][st] for st in STATES])
            ax.hlines(ys + off, lo, hi, color=col, lw=1.2, alpha=0.6)
        ax.scatter(x, ys + off, marker=mk, s=26, color=col if fc is None else "white", edgecolor=col, lw=1.2,
                   label=lab, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{SNAME[st]} ({100 * occ[st]:.1f}%)" for st in STATES])
    ax.set_xlim(-2, 102)
    ax.set_xlabel("recovery accuracy of the teacher's state (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.4, -0.13))
    save(fig, "fig2_state_recovery")


# ------------------------------------------------------------------ Figure 3
def fig3():
    c = json.load(open(os.path.join(OUT, "r1_ctx.json")))
    rows = pd.DataFrame(c["rows"])
    nz = json.load(open(os.path.join(OUT, "r1_noise.json")))
    main = json.load(open(os.path.join(OUT, "r1_main.json")))["summary"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax in axs:
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    # A, B: context length, all states and σ
    for ax, key, title in ((axs[0], "all", "all states"), (axs[1], "σ", "σ search")):
        for mode, col, mk, lab in (("ngram", GRAY, "s", "n-gram"), ("aux", ORANGE, "^", "aux"), ("dual", BLUE, "o", "dual")):
            r = rows[rows["mode"] == mode].sort_values("ctx")
            r = r[r.ctx.isin([1, 2, 4, 8, 16])] if mode == "ngram" else r
            ax.plot(r.ctx, 100 * r[key], marker=mk, color=col, label=lab, ms=5, lw=1.6, mfc="white" if mode == "ngram" else col)
            if mode != "ngram":
                lo, hi = (r["min_all"], r["max_all"]) if key == "all" else (r["min_σ"], r["max_σ"])
                ax.vlines(r.ctx, 100 * lo, 100 * hi, color=col, lw=1, alpha=0.6)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("history visible (steps)")
        ax.set_title(title, loc="left", fontsize=9)
    axs[0].set_ylabel("state recovery (%)")
    axs[0].set_ylim(85, 100.5); axs[1].set_ylim(55, 101.5)
    axs[1].legend(loc="lower right")
    label(axs[0], "A"); label(axs[1], "B")
    # C: noise
    ax = axs[2]
    ps = [0] + [r["p"] for r in nz["rows"]]
    for k, col, mk, lab in (("dual", BLUE, "o", "dual"), ("filter", PURPLE, "X", "finite-state filter"),
                            ("ngram8", GRAY, "s", "8-step n-gram")):
        y = [main[k]["all"]] + [r[k]["all"] for r in nz["rows"]]
        ax.plot(np.array(ps) * 100, 100 * np.array(y), marker=mk, color=col, label=lab, ms=5, lw=1.6)
    ydual_phi = [main["dual"]["φ"]] + [r["dual"]["φ"] for r in nz["rows"]]
    ax.plot(np.array(ps) * 100, 100 * np.array(ydual_phi), ls="--", color=BLUE, lw=1.2, marker="o", ms=3.5, mfc="white",
            label="dual, φ only")
    ax.set_xlabel("perception tokens flipped (%)"); ax.set_xticks([0, 5, 10, 20])
    ax.set_title("perceptual noise", loc="left", fontsize=9)
    ax.set_ylim(70, 100.5)
    ax.legend(loc="lower left", fontsize=6.8)
    label(ax, "C")
    fig.tight_layout()
    save(fig, "fig3_memory_noise")




# ------------------------------------------------------------------ Figure 4：群體（two-patch 與雙橋）
COND_LAB = {"bo_frozen": "bo, frozen", "bo_learn": "bo, learning", "aux_frozen": "aux, frozen", "aux_learn": "aux, learning",
            "dual_frozen": "dual, frozen", "dual_learn": "dual, learning", "dual_learn_A": "dual, learning actions only",
            "dual_learn_step": "dual, step-weighted", "dual_learn_sigma": "dual, search-balanced",
            "dual_learn_reh": "dual, teacher rehearsal", "dual_learn_rehNS": "dual, rehearsal without search",
            "dual_learn_rehS": "dual, rehearsal of search only"}
COND_COL = {"bo": GRAY, "aux": ORANGE, "dual": BLUE}


def _rel(d, env, cond, base="teacher"):
    from common import bootstrap_ci
    x = d[d.env == env]
    A = x[x.cond == cond].set_index("seed").delivered
    B = x[x.cond == base].set_index("seed").delivered
    c = A.index.intersection(B.index)
    r = 100 * (A[c] - B[c]) / B[c].mean()
    lo, hi = bootstrap_ci(r.values)
    return r.mean(), lo, hi, len(c)


def fig4():
    d = pd.read_csv(os.path.join(OUT, "r2_per_seed.csv"))
    plt.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.55, 1], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    groups = [("two-patch (food depletes and moves)", "two_patch",
               ["bo_frozen", "bo_learn", "aux_frozen", "aux_learn", "dual_frozen", "dual_learn", "dual_learn_A",
                "dual_learn_step", "dual_learn_sigma", "dual_learn_reh", "dual_learn_rehNS", "dual_learn_rehS"]),
              ("double bridge (fixed food)", "double_bridge",
               ["bo_frozen", "bo_learn", "aux_frozen", "aux_learn", "dual_frozen", "dual_learn"])]
    y = 0; yt, yl = [], []
    for title, env, conds in groups:
        if not (d.env == env).any():
            continue
        ax.text(-14.3, y - 0.35, title, fontsize=7.8, fontweight="bold", va="center", ha="left")
        y -= 1
        for c in conds:
            if not ((d.env == env) & (d.cond == c)).any():
                continue
            m, lo, hi, n = _rel(d, env, c)
            col = GREEN if "_reh" in c else COND_COL[c.split("_")[0]]
            frozen = c.endswith("frozen")
            ax.hlines(y, lo, hi, color=col, lw=1.4)
            ax.plot(m, y, "o", color=col, mfc="white" if frozen else col, ms=6, mew=1.4)
            yt.append(y); yl.append(f"{COND_LAB[c]} (n={n})")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5)
    ax.set_xlabel("food delivered relative to teacher (%)")
    ax.set_xlim(-14.5, 9); ax.grid(axis="y", visible=False); ax.set_ylim(y + 0.9, 0.3)
    label(ax, "A")
    # B：練習到的 σ 比例 vs 學習後 σ 追蹤
    x = d[(d.env == "two_patch")]
    ax2 = fig.add_subplot(gs[0, 1])
    pts = [("dual_learn", "default", BLUE), ("dual_learn_step", "step-weighted", BLUE),
           ("dual_learn_sigma", "search-balanced", BLUE), ("dual_learn_reh", "teacher rehearsal", GREEN),
           ("dual_learn_rehNS", "rehearsal w/o search", GREEN), ("dual_learn_rehS", "rehearsal of search", GREEN)]
    from common import bootstrap_ci
    for c, lab, col in pts:
        g = x[x.cond == c]
        if g.empty:
            continue
        px, py = 100 * g.practiced_sigma.mean(), 100 * g["stab_σ"].mean()
        lo, hi = bootstrap_ci(100 * g["stab_σ"].values)
        ax2.vlines(px, lo, hi, color=col, lw=1.2)
        ax2.plot(px, py, "o", color=col, ms=6)
        ax2.annotate(lab, (px, py), xytext=(5, -2), textcoords="offset points", fontsize=6.8, color=INK2)
    ax2.axhline(100, color=INK2, lw=0.8, ls=":")
    ax2.text(0.5, 100.4, "frozen", fontsize=6.8, color=INK2)
    tsh = 100 * x[x.cond == "teacher"]["occ_σ"].mean()
    ax2.axvline(tsh, color=INK2, lw=0.8, ls=":")
    ax2.text(tsh + 0.4, 84.5, "teacher's\nstep share", fontsize=6.5, color=INK2)
    ax2.set_xlabel("σ share of practiced state targets (%)")
    ax2.set_ylabel("σ tracking after learning (%)")
    ax2.set_ylim(84, 102); ax2.set_xlim(0, 30)
    label(ax2, "B")
    # C：說出的狀態佔比
    ax3 = fig.add_subplot(gs[1, 1])
    conds = [("teacher", "teacher", INK), ("dual_frozen", "dual frozen", LGRAY), ("dual_learn", "dual learning", BLUE),
             ("dual_learn_reh", "teacher rehearsal", GREEN)]
    sts = ["σ", "ε", "φ"]
    wdt = 0.2
    for i, (c, lab, col) in enumerate(conds):
        g = x[x.cond == c]
        vals = [100 * g[f"occ_{s}"].mean() for s in sts]
        ax3.bar(np.arange(3) + (i - 1.5) * wdt, vals, wdt * 0.9, color=col, label=lab)
    ax3.set_xticks(range(3)); ax3.set_xticklabels(["σ search", "ε explore", "φ follow trail"])
    ax3.set_ylabel("share of steps (%)"); ax3.grid(axis="x", visible=False)
    ax3.legend(fontsize=6.3, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    label(ax3, "C")
    save(fig, "fig4_colonies")



# ------------------------------------------------------------------ Figure 5：泛化
ENVLAB = {"train": "two-patch\n(new seed)", "two_patch": "two-patch", "barrier": "barrier\n(unseen map)",
          "food_fast": "food fast\n(10/cell)", "food_slow": "food slow\n(90/cell)", "decay_fast": "decay fast\n(ρ=0.94)",
          "decay_slow": "decay slow\n(ρ=0.99)"}


def fig5():
    from common import bootstrap_ci
    g = json.load(open(os.path.join(OUT, "r1_gen.json")))
    d = pd.read_csv(os.path.join(OUT, "r2_per_seed.csv"))
    plt.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.6, wspace=0.28)
    # A：離線追蹤
    ax = fig.add_subplot(gs[0, :])
    envs = ["train", "barrier", "food_fast", "food_slow", "decay_fast", "decay_slow"]
    meth = [("ngram8", "8-step n-gram", "s", GRAY), ("filter", "finite-state filter", "X", PURPLE),
            ("aux", "aux", "^", ORANGE), ("gru_aux", "GRU-aux", "v", GREEN), ("dual", "dual", "o", BLUE)]
    offs = np.linspace(-0.28, 0.28, len(meth))
    for (k, lab, mk, col), off in zip(meth, offs):
        y = [100 * g[e]["summary"][k]["all"] for e in envs]
        ax.scatter(np.arange(len(envs)) + off, y, marker=mk, color=col, s=30, label=lab, zorder=3)
        if "min" in g[envs[0]]["summary"][k]:
            lo = [100 * g[e]["summary"][k]["min"]["all"] for e in envs]; hi = [100 * g[e]["summary"][k]["max"]["all"] for e in envs]
            ax.vlines(np.arange(len(envs)) + off, lo, hi, color=col, lw=1.1, alpha=0.7)
    ax.set_xticks(range(len(envs))); ax.set_xticklabels([ENVLAB[e] for e in envs], fontsize=7.5)
    ax.set_ylabel("state recovery (%)"); ax.set_ylim(85, 100.5); ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title("offline tracking of teacher trips in each environment", loc="left", fontsize=9, pad=22)
    label(ax, "A")
    # B：凍結群體 vs 教師
    ax2 = fig.add_subplot(gs[1, 0])
    envs2 = ["two_patch", "barrier", "food_fast", "food_slow", "decay_fast", "decay_slow"]
    for j, (m, col) in enumerate((("bo", GRAY), ("aux", ORANGE), ("dual", BLUE))):
        for i, e in enumerate(envs2):
            if not ((d.env == e) & (d.cond == f"{m}_frozen")).any():
                continue
            r = _rel(d, e, f"{m}_frozen")
            ax2.plot(r[0], i + (j - 1) * 0.22, "o", color=col, mfc="white", mew=1.4, ms=5, label=f"{m}, frozen" if i == 0 else None)
            ax2.hlines(i + (j - 1) * 0.22, r[1], r[2], color=col, lw=1.2)
    ax2.axvline(0, color=INK2, lw=0.8)
    ax2.set_yticks(range(len(envs2))); ax2.set_yticklabels([ENVLAB[e].replace("\n", " ") for e in envs2], fontsize=7)
    ax2.invert_yaxis(); ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("food delivered relative to teacher (%)")
    ax2.legend(fontsize=6.5, loc="lower left")
    ax2.set_title("frozen colonies", loc="left", fontsize=9)
    label(ax2, "B")
    # C：學習效果 vs 食物移動頻率
    ax3 = fig.add_subplot(gs[1, 1])
    envs3 = ["food_slow", "two_patch", "food_fast"]
    xs = np.arange(3)
    for j, (c, base, lab, col, mk) in enumerate((("bo_learn", "bo_frozen", "bo", GRAY, "s"),
                                                 ("aux_learn", "aux_frozen", "aux", ORANGE, "^"),
                                                 ("dual_learn", "dual_frozen", "dual", BLUE, "o"),
                                                 ("dual_learn_reh", "dual_frozen", "dual, teacher rehearsal", GREEN, "D"))):
        ms, los, his, ok = [], [], [], []
        for i, e in enumerate(envs3):
            if not ((d.env == e) & (d.cond == c)).any() or not ((d.env == e) & (d.cond == base)).any():
                continue
            r = _rel(d, e, c, base)
            ms.append(r[0]); los.append(r[1]); his.append(r[2]); ok.append(i)
        if ok:
            xx = np.array(ok) + (j - 1.5) * 0.09
            ax3.plot(xx, ms, marker=mk, color=col, label=lab, ms=5, lw=1.4)
            ax3.vlines(xx, los, his, color=col, lw=1.1)
    ax3.axhline(0, color=INK2, lw=0.8)
    ax3.set_xticks(xs); ax3.set_xticklabels(["slow\n(90/cell)", "training\n(30/cell)", "fast\n(10/cell)"], fontsize=7.5)
    ax3.set_xlabel("food relocation frequency"); ax3.set_ylabel("learning − frozen (%)")
    ax3.legend(fontsize=6.3, loc="upper right", ncol=2); ax3.grid(axis="x", visible=False); ax3.set_ylim(-10.5, 7.5)
    ax3.set_title("effect of online learning", loc="left", fontsize=9)
    label(ax3, "C")
    save(fig, "fig5_generalization")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig2", "fig3"]
    for w in which:
        globals()[w]()
        print("saved", w)

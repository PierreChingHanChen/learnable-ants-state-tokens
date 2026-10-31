"""E3／E5 彙整：感知雜訊下的離線追蹤（e124*.json → e3_summary.csv）與消融（e5.json → e5_summary.csv，補充表 S12）。
Markdown 摘要印在標準輸出。"""
import os, json
import numpy as np
import pandas as pd
from common import RESULTS, bootstrap_ci

OUT = os.path.join(RESULTS, "la2")
L = []


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    lo, hi = bootstrap_ci(x)
    return f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def e3():
    rows = []
    for tag, p in (("", 0.0), ("_f05", 0.05), ("_f10", 0.10), ("_f20", 0.20)):
        f = os.path.join(OUT, f"e124{tag}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for r in d["seeds"]:
            rows.append({"flip": p, "seed": r["seed"], "dual_free": r["dual_free"]["all"], "dual_free_sigma": r["dual_free"]["σ"],
                         "dual_free_phi": r["dual_free"]["φ"], "bo_probe": r["bo_probe"]["all"],
                         "obs_probe": r["obs_probe"]["all"], "ngram8": d["ngram"]["8"]["all"],
                         "ngram8_sigma": d["ngram"]["8"]["σ"], "ngram8_phi": d["ngram"]["8"]["φ"],
                         "ngram1": d["ngram"]["1"]["all"], "action_acc_bo": r["bo_eval"]["action_acc"],
                         "action_acc_dual": r["dual_eval"]["action_acc"], "follow": r["intervene"]["follow_frac"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e3_summary.csv"), index=False)
    L.append("## E3 感知翻轉雜訊\n")
    L.append("| 翻轉機率 | dual 自由追蹤 | 8-gram | 查表 | bo 探針 | dual σ | 8-gram σ | dual φ | 8-gram φ | 介入跟隨率 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p, g in df.groupby("flip"):
        L.append(f"| {p} | {ci(g.dual_free)} | {g.ngram8.iloc[0]:.3f} | {g.ngram1.iloc[0]:.3f} | {ci(g.bo_probe)} | "
                 f"{g.dual_free_sigma.mean():.3f} | {g.ngram8_sigma.iloc[0]:.3f} | {g.dual_free_phi.mean():.3f} | "
                 f"{g.ngram8_phi.iloc[0]:.3f} | {g.follow.mean():.3f} |")
    L.append("")


def e5():
    f = os.path.join(OUT, "e5.json")
    if not os.path.exists(f):
        return
    rs = json.load(open(f))
    rows = [{"config": r["config"], "mode": r["mode"], "seed": r["seed"], "action_nll": r["action_nll"],
             "action_acc": r["action_acc"], "free": r.get("free", {}).get("all", np.nan),
             "free_sigma": r.get("free", {}).get("σ", np.nan), "free_phi": r.get("free", {}).get("φ", np.nan)}
            for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e5_summary.csv"), index=False)
    L.append("## E5 消融（各 3 個種子）\n")
    L.append("| 設定 | bo 行動 NLL | dual 行動 NLL | bo 行動一致率 | dual 行動一致率 | dual 自由追蹤 | dual σ | dual φ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cfg in ("ctx4", "ctx8", "main", "ctx32", "embd32", "data30", "data10"):
        g = df[df.config == cfg]
        if g.empty:
            continue
        b, du = g[g["mode"] == "bo"], g[g["mode"] == "dual"]
        L.append(f"| {cfg} | {b.action_nll.mean():.4f} | {du.action_nll.mean():.4f} | {b.action_acc.mean():.4f} | "
                 f"{du.action_acc.mean():.4f} | {du.free.mean():.4f} | {du.free_sigma.mean():.3f} | {du.free_phi.mean():.3f} |")
    L.append("")


if __name__ == "__main__":
    e3(); e5()
    print("\n".join(L))

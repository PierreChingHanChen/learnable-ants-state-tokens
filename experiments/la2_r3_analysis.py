"""R3：彙整群體實驗（r2_jobs）→ 每種子 CSV、配對統計 JSON、摘要 Markdown。"""
import os, glob, pickle, json
import numpy as np
import pandas as pd
from common import RESULTS, paired_stats, bootstrap_ci, holm

OUT = os.path.join(RESULTS, "la2")
R2 = os.path.join(OUT, "r2_jobs")


def load():
    """讀取 r2_jobs/*.pkl（完整紀錄，見大型資料封存）；若不存在則改讀 r2_per_seed.csv。"""
    if not glob.glob(os.path.join(R2, "*.pkl")):
        return pd.read_csv(os.path.join(OUT, "r2_per_seed.csv"))
    rows = []
    for fn in glob.glob(os.path.join(R2, "*.pkl")):
        r = pickle.load(open(fn, "rb"))
        rows.append({k: v for k, v in r.items() if k != "deliv_curve"})
    return pd.DataFrame(rows)


def cond_table(d, env):
    x = d[d.env == env]
    out = {}
    for c, g in x.groupby("cond"):
        row = {"n": int(len(g))}
        for col in ("delivered", "timeouts", "rich_share", "short_share", "oa_legal", "osa_legal", "trans_legal",
                    "dispersion", "mark_q1", "mark_q0", "practiced_sigma", "practiced_sigma_late", "stab_all",
                    "stab_σ", "stab_φ", "stab_ε", "stab_ν", "occ_σ", "occ_ε", "occ_φ", "occ_ν", "occ_η",
                    "sig_in_success", "mark_searched1_q1", "mark_searched0_q1", "mark_q1_strong", "mark_q1_weak"):
            if col in g and g[col].notna().any():
                v = g[col].dropna().values.astype(float)
                lo, hi = bootstrap_ci(v) if len(v) > 1 else (float("nan"), float("nan"))
                row[col] = {"mean": float(v.mean()), "lo": lo, "hi": hi}
        if "mark_q1" in g and "mark_q0" in g and g["mark_q0"].notna().all():
            diff = (g["mark_q1"] - g["mark_q0"]).values
            row["mark_diff"] = paired_stats(diff)
            row["mark_diff"]["n_pos"] = int((diff > 0).sum())
        if "mark_searched1_q1" in g and g["mark_searched1_q1"].notna().all() and g["mark_searched0_q1"].notna().all():
            row["search_boost"] = paired_stats((g["mark_searched1_q1"] - g["mark_searched0_q1"]).values)
        out[c] = row
    return out


def paired(d, env, a, b, col="delivered", rel_to=None):
    x = d[d.env == env]
    A = x[x.cond == a].set_index("seed")[col]
    B = x[x.cond == b].set_index("seed")[col]
    common = A.index.intersection(B.index)
    if len(common) < 3:
        return None
    diff = (A[common] - B[common]).values.astype(float)
    st = paired_stats(diff)
    if rel_to is not None:
        R = x[x.cond == rel_to].set_index("seed")[col]
        st["rel_pct"] = float(100 * diff.mean() / R[common].mean())
    st.update({"a": a, "b": b, "env": env, "col": col})
    return st


COMPARISONS = {
    "two_patch": [("bo_learn", "teacher"), ("aux_learn", "teacher"), ("dual_learn", "teacher"),
                  ("bo_frozen", "teacher"), ("aux_frozen", "teacher"), ("dual_frozen", "teacher"),
                  ("bo_learn", "bo_frozen"), ("aux_learn", "aux_frozen"), ("dual_learn", "dual_frozen"),
                  ("dual_learn", "bo_learn"), ("dual_learn", "aux_learn"), ("aux_learn", "bo_learn"),
                  ("dual_learn_A", "dual_learn"), ("dual_learn_A", "dual_frozen"),
                  ("dual_learn_step", "dual_learn"), ("dual_learn_sigma", "dual_learn"), ("dual_learn_reh", "dual_learn"),
                  ("dual_learn_step", "dual_frozen"), ("dual_learn_sigma", "dual_frozen"), ("dual_learn_reh", "dual_frozen"),
                  ("teacher_nomem", "teacher"),
                  ("dual_learn_rehNS", "dual_learn"), ("dual_learn_rehS", "dual_learn"),
                  ("dual_learn_rehNS", "dual_learn_reh"), ("dual_learn_rehS", "dual_learn_reh"),
                  ("dual_learn_rehNS", "dual_frozen"), ("dual_learn_rehS", "dual_frozen")],
    "double_bridge": [("bo_learn", "teacher"), ("aux_learn", "teacher"), ("dual_learn", "teacher"),
                      ("bo_learn", "bo_frozen"), ("aux_learn", "aux_frozen"), ("dual_learn", "dual_frozen"),
                      ("dual_learn", "bo_learn"), ("dual_learn", "aux_learn")],
}
for e in ("barrier", "food_fast", "food_slow", "decay_fast", "decay_slow"):
    COMPARISONS[e] = [("bo_frozen", "teacher"), ("aux_frozen", "teacher"), ("dual_frozen", "teacher"),
                      ("dual_frozen", "bo_frozen"), ("dual_frozen", "aux_frozen"),
                      ("bo_learn", "bo_frozen"), ("aux_learn", "aux_frozen"), ("dual_learn", "dual_frozen"),
                      ("dual_learn", "bo_learn"), ("dual_learn_reh", "dual_learn"), ("dual_learn_step", "dual_learn"),
                      ("dual_learn_reh", "dual_frozen"), ("dual_learn_step", "dual_frozen")]
STAB_COMPARISONS = [("dual_learn", "dual_frozen"), ("dual_learn_step", "dual_learn"), ("dual_learn_sigma", "dual_learn"),
                    ("dual_learn_reh", "dual_learn"), ("dual_learn_A", "dual_learn"), ("aux_learn", "aux_frozen"),
                    ("dual_learn_reh", "dual_frozen"), ("dual_learn_step", "dual_frozen"), ("dual_learn_sigma", "dual_frozen"),
                    ("dual_learn_rehNS", "dual_learn"), ("dual_learn_rehS", "dual_learn"),
                    ("dual_learn_rehNS", "dual_learn_reh"), ("dual_learn_rehS", "dual_learn_reh"), ("bo_learn", "bo_frozen")]


def export_curves(every=50):
    """每個蟻群的累計運回量（每 50 步一點）→ results/la2/r2_delivery_curves.csv（圖 4）。"""
    rows = []
    for fn in sorted(glob.glob(os.path.join(R2, "*.pkl"))):
        r = pickle.load(open(fn, "rb"))
        rows += [(r["cond"], r["env"], r["seed"], every * (i + 1), v) for i, v in enumerate(r["deliv_curve"])]
    pd.DataFrame(rows, columns=["cond", "env", "seed", "step", "delivered"]).sort_values(
        ["cond", "env", "seed", "step"]).to_csv(os.path.join(OUT, "r2_delivery_curves.csv"), index=False)


def main():
    d = load()
    if glob.glob(os.path.join(R2, "*.pkl")):
        d.to_csv(os.path.join(OUT, "r2_per_seed.csv"), index=False)
        export_curves()
    res = {"tables": {}, "paired": [], "stab_paired": []}
    for env in sorted(d.env.unique()):
        res["tables"][env] = cond_table(d, env)
        for a, b in COMPARISONS.get(env, []):
            st = paired(d, env, a, b, rel_to=b)
            if st:
                res["paired"].append(st)
    for a, b in STAB_COMPARISONS:
        for col in ("stab_all", "stab_σ", "occ_σ", "practiced_sigma", "timeouts"):
            st = paired(d, "two_patch", a, b, col)
            if st:
                res["stab_paired"].append(st)
    # Holm 校正：每個環境的產量比較為一族
    for env in d.env.unique():
        idx = [i for i, s in enumerate(res["paired"]) if s["env"] == env]
        adj = holm([res["paired"][i]["p"] for i in idx])
        for i, p in zip(idx, adj):
            res["paired"][i]["p_holm"] = p
    json.dump(res, open(os.path.join(OUT, "r3_colony.json"), "w"), indent=1, default=float, ensure_ascii=False)
    for env, tab in res["tables"].items():
        print(f"== {env}")
        for c, row in sorted(tab.items()):
            dl = row["delivered"]
            extra = " ".join(f"{k}={row[k]['mean']:.3f}" for k in ("stab_σ", "stab_all", "practiced_sigma", "occ_σ", "timeouts")
                             if k in row)
            print(f"  {c:18s} n={row['n']:2d} delivered {dl['mean']:7.1f} [{dl['lo']:.0f}, {dl['hi']:.0f}]  {extra}")
    print("== paired delivered")
    for s in res["paired"]:
        print(f"  {s['env']:13s} {s['a']:17s} − {s['b']:13s} {s['mean']:+8.1f} [{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}] "
              f"({s.get('rel_pct', float('nan')):+.1f}%) p={s['p']:.2g} holm={s.get('p_holm', float('nan')):.2g} n={s['n']}")
    print("== paired stability (two_patch)")
    for s in res["stab_paired"]:
        print(f"  {s['col']:16s} {s['a']:17s} − {s['b']:13s} {s['mean']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] p={s['p']:.2g}")


if __name__ == "__main__":
    main()

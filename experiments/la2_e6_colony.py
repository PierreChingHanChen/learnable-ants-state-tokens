"""LA2 群體部署的單一工作函式（30 隻螞蟻、共享介質、3000 步），由 la2_e7_followup.py 使用。
條件：teacher、bo＋學習、dual＋學習、bo 凍結、dual 凍結及其變體；地圖 two_patch 與 double_bridge。
論文主要的群體實驗見 la2_r2_colony.py。"""
import os, pickle, json
import numpy as np
from common import pmap, RESULTS, save_csv
from miat_ants.la2 import metrics as M, vocab as V, pretrain as P
from miat_ants.la2.system import LA2Config, LA2System
from miat_ants.foraging.analysis import dispersion

OUT = os.path.join(RESULTS, "la2")
CORPUS = os.path.join(OUT, "corpus.pkl")
SEEDS = int(os.environ.get("LA2_E6_SEEDS", "20"))
FROZEN_SEEDS = int(os.environ.get("LA2_E6_FROZEN", "10"))
STEPS = int(os.environ.get("LA2_E6_STEPS", "3000"))
_cache = {}


def tables():
    if "t" not in _cache:
        tr, va = P.split(pickle.load(open(CORPUS, "rb")), seed=0)
        _cache["t"] = M.teacher_tables(tr)
        _cache["probes"] = {m: M.make_probes(va, m) for m in V.MODES}
    return _cache["t"], _cache["probes"]


JOBDIR = os.path.join(OUT, "e6_jobs")


def job(args):
    cond, mp, seed = args
    fn = os.path.join(JOBDIR, f"{cond}_{mp}_{seed}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    out = _job(args)
    pickle.dump(out, open(fn, "wb"))
    return out


def _job(args):
    cond, mp, seed = args
    tab, probes = tables()
    kw = dict(map_name=mp, seed=seed, max_steps=STEPS, collect_episodes=True)
    if cond.startswith("teacher"):
        kw.update(core="teacher", memory=(cond != "teacher_nomem")); mode = "dual"
    else:
        mode, lr = cond.split("_")
        kw.update(core="gpt", mode=mode, learn=lr.startswith("learn"), probe_contexts=probes[mode],
                  learn_states=(lr != "learnA"))           # learnA：線上學習只學行動、不學狀態詞
    r = LA2System(LA2Config(**kw)).run()
    h = r.history
    eps = r.episodes if (cond == "teacher" or mode == "dual") else r.episodes
    out = {"cond": cond, "map": mp, "seed": seed, "delivered": r.delivered,
           "takes_rich": h["takes_rich"][-1], "takes_poor": h["takes_poor"][-1],
           "gate_short": h["gate_short"][-1], "gate_long": h["gate_long"][-1],
           "timeouts": h["timeouts"][-1], "mark_rate_carry": float(np.mean(h["mark_rate_carry"]))}
    tot = h["takes_rich"][-1] + h["takes_poor"][-1]
    out["rich_share"] = h["takes_rich"][-1] / max(tot, 1)
    g = h["gate_short"][-1] + h["gate_long"][-1]
    out["short_share"] = h["gate_short"][-1] / max(g, 1)
    lg = M.legality(eps, tab, mode)
    out.update({k: v for k, v in lg.items() if k != "occupancy"})
    sc = np.sum(h["state_counts"], 0)
    if sc.sum():
        out.update({f"occ_{s}": float(x) for s, x in zip(V.STATES, sc / sc.sum())})
    mq = M.mark_by_quality(eps, mode)
    out.update({f"mark_{k}": v for k, v in mq.items()})
    if r.probes:
        out["dispersion"] = dispersion(r.probes[-1])
        out["dispersion_start"] = dispersion(r.probes[0])
    if mode == "dual":                                     # 搜尋狀態在成功／失敗回合中的比例（選擇偏差）
        sig, D = V.STOI["σ"], V.STOI["D"]
        succ = [e for e in eps if len(e) > 1 and e[-1] == D]
        fail = [e for e in eps if len(e) > 1 and e[-1] != D]
        has = lambda E: float(np.mean([sig in e for e in E])) if E else float("nan")
        frac = lambda E: float(np.mean([np.mean([t == sig for t in e[2::3]]) for e in E])) if E else float("nan")
        out.update({"sig_in_success": has(succ), "sig_in_fail": has(fail), "sig_in_all": has(eps),
                    "sigfrac_success": frac(succ), "sigfrac_all": frac(eps), "n_success_eps": len(succ),
                    "n_fail_eps": len(fail)})
    out["deliv_curve"] = h["delivered"]
    return out


def main():
    os.makedirs(JOBDIR, exist_ok=True)
    jobs = [(c, m, s) for m in ("two_patch", "double_bridge") for s in range(SEEDS)
            for c in ("dual_learn", "bo_learn", "teacher")]
    jobs += [(c, m, s) for m in ("two_patch", "double_bridge") for s in range(FROZEN_SEEDS)
             for c in ("dual_frozen", "bo_frozen")]
    jobs.sort(key=lambda j: {"dual_learn": 0, "dual_frozen": 1, "bo_learn": 2, "bo_frozen": 3, "teacher": 4}[j[0]])
    res = pmap(job, jobs)
    pickle.dump(res, open(os.path.join(OUT, "e6.pkl"), "wb"))
    import pandas as pd
    pd.DataFrame([{k: v for k, v in r.items() if k != "deliv_curve"} for r in res]).to_csv(
        os.path.join(OUT, "e6_per_seed.csv"), index=False)
    print("done", len(res))


if __name__ == "__main__":
    main()

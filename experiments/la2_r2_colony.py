"""R2：閉環群體實驗（30 隻螞蟻、共享介質、3,000 步）——論文第 4.7–4.10 節。

  * 訓練種子輪替：模擬種子 k 使用訓練種子 k mod 10 的基礎權重。
  * 條件：teacher、teacher_nomem、{bo,aux,dual}_{frozen,learn}，以及 dual 的線上學習變體：
      dual_learn_A                    只學行動（狀態位置不計損失）
      dual_learn_step                 依步數加權取樣（每一步大致等權）
      dual_learn_sigma                搜尋平衡取樣：一半的窗口強制涵蓋一個 σ 步
      dual_learn_reh                  教師複習：每批 25% 的窗口取自教師語料
      dual_learn_rehNS / rehS         教師複習只用不含 σ／只用含 σ 的教師回合
  * 狀態穩定性：學習結束後，以每隻螞蟻的最終權重自由追蹤 150 個固定的教師回合（two-patch、模擬種子 900）。
  * 測試環境：barrier（未見地圖）、food_fast／food_slow（食物塊大小 10／90）、decay_fast／decay_slow（ρ = 0.94／0.99）。
環境變數 LA2_R2_PLAN 指定工作清單（main、reh_split、bridge、gen、gen_learn；以分號串接）。"""
import os, pickle, json, sys, time
import numpy as np
from common import pmap, RESULTS
from miat_ants.la2 import metrics as M, vocab as V, pretrain as P
from miat_ants.la2.system import LA2Config, LA2System, base_path
from miat_ants.foraging.analysis import dispersion

OUT = os.path.join(RESULTS, "la2")
R2 = os.path.join(OUT, "r2_jobs")
os.makedirs(R2, exist_ok=True)
CORPUS = os.path.join(OUT, "corpus.pkl")
STEPS = int(os.environ.get("LA2_R2_STEPS", "3000"))
N_TRAIN_SEEDS = 10
STAB_N = 150
ENVS = {
    "two_patch": dict(map_name="two_patch"),
    "double_bridge": dict(map_name="double_bridge"),
    "barrier": dict(map_name="barrier"),
    "food_fast": dict(map_name="two_patch", map_kw={"per_cell": 10}),
    "food_slow": dict(map_name="two_patch", map_kw={"per_cell": 90}),
    "decay_fast": dict(map_name="two_patch", rho=0.94),
    "decay_slow": dict(map_name="two_patch", rho=0.99),
}
VARIANTS = {"": {}, "A": dict(learn_states=False), "step": dict(weighting="step"),
            "sigma": dict(weighting="sigma", sig_frac=0.5), "reh": dict(rehearse=0.25),
            "rehNS": dict(rehearse=0.25), "rehS": dict(rehearse=0.25)}     # 教師複習：只用不含 σ／只用含 σ 的教師回合
_cache = {}


def shared():
    if "t" not in _cache:
        tr, va = P.split(pickle.load(open(CORPUS, "rb")), seed=0)
        _cache["tr"] = tr
        _cache["t"] = M.teacher_tables(tr)
        _cache["probes"] = {m: M.make_probes(va, m) for m in V.MODES}
        sp = os.path.join(OUT, "r2_stability_eps.pkl")
        if not os.path.exists(sp):
            r = LA2System(LA2Config(map_name="two_patch", core="teacher", seed=900, max_steps=1500,
                                    collect_episodes=True)).run()
            eps = r.episodes
            sel = np.sort(np.random.default_rng(1).choice(len(eps), STAB_N, replace=False))
            pickle.dump([eps[i] for i in sel], open(sp, "wb"))
        _cache["stab"] = pickle.load(open(sp, "rb"))
    return _cache


def job(args):
    cond, env, seed = args
    fn = os.path.join(R2, f"{cond}__{env}__{seed}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    t0 = time.time()
    out = _job(cond, env, seed)
    out["seconds"] = round(time.time() - t0, 1)
    pickle.dump(out, open(fn, "wb"))
    return out


def stability(core, mode, eps, agents):
    model = core.model
    ts = [M.steps_of(e)[1] for e in eps]
    allT = np.concatenate(ts)
    pred = M.free_run_multi(model, eps, agents) if mode == "dual" else M.aux_readout_multi(model, eps, agents)
    accs = [M.per_state_acc(allT, np.concatenate(pred[int(a)])) for a in agents]
    return {f"stab_{k}": float(np.nanmean([a[k] for a in accs])) for k in ("all", "σ", "φ", "ε", "ν")}


def _job(cond, env, seed):
    sh = shared()
    tab, probes = sh["t"], sh["probes"]
    kw = dict(seed=seed, max_steps=STEPS, collect_episodes=True, **ENVS[env])
    tseed = seed % N_TRAIN_SEEDS
    if cond.startswith("teacher"):
        kw.update(core="teacher", memory=(cond != "teacher_nomem"))
        mode = "dual"
    else:
        parts = cond.split("_")
        mode, lr = parts[0], parts[1]
        var = parts[2] if len(parts) > 2 else ""
        kw.update(core="gpt", mode=mode, learn=(lr == "learn"), probe_contexts=probes[mode],
                  base_weights=base_path(mode, f"_s{tseed}"), **VARIANTS[var])
        if var == "reh":
            kw["rehearse_pool"] = sh["tr"]
        elif var in ("rehNS", "rehS"):
            has = lambda e: V.SIGMA_ID in e
            kw["rehearse_pool"] = [e for e in sh["tr"] if has(e) == (var == "rehS")]
    r = LA2System(LA2Config(**kw)).run()
    h = r.history
    eps = r.episodes
    rec_mode = "bo" if mode == "bo" else "dual"          # aux 以 dual 格式記錄（說出的狀態不回饋）
    out = {"cond": cond, "env": env, "seed": seed, "train_seed": tseed if not cond.startswith("teacher") else -1,
           "delivered": r.delivered, "takes_rich": h["takes_rich"][-1], "takes_poor": h["takes_poor"][-1],
           "gate_short": h["gate_short"][-1], "gate_long": h["gate_long"][-1], "timeouts": h["timeouts"][-1],
           "success_eps": h["success_eps"][-1]}
    tot = h["takes_rich"][-1] + h["takes_poor"][-1]
    out["rich_share"] = h["takes_rich"][-1] / max(tot, 1)
    g = h["gate_short"][-1] + h["gate_long"][-1]
    out["short_share"] = h["gate_short"][-1] / max(g, 1) if g else float("nan")
    lg = M.legality(eps, tab, rec_mode)
    out.update({k: v for k, v in lg.items() if k != "occupancy"})
    sc = np.sum(h["state_counts"], 0)
    if sc.sum():
        out.update({f"occ_{s}": float(x) for s, x in zip(V.STATES, sc / sc.sum())})
    mq = M.mark_by_quality(eps, rec_mode)
    out.update({f"mark_{k}": v for k, v in mq.items()})
    if r.probes:
        out["dispersion"] = dispersion(r.probes[-1])
    if rec_mode == "dual":
        sig, D = V.STOI["σ"], V.STOI["D"]
        succ = [e for e in eps if len(e) > 1 and e[-1] == D]
        has = lambda E: float(np.mean([sig in e for e in E])) if E else float("nan")
        out.update({"sig_in_success": has(succ), "sig_in_all": has(eps)})
    if hasattr(r.core, "practiced") and r.core.practiced:
        pr = np.array(r.core.practiced, float)
        out["practiced_sigma"] = float(np.nanmean(pr))
        out["practiced_sigma_late"] = float(np.nanmean(pr[len(pr) // 2:]))
    if mode in ("dual", "aux") and not cond.startswith("teacher") and env == "two_patch":
        agents = np.arange(30) if kw.get("learn") else np.array([0])
        out.update(stability(r.core, mode, sh["stab"], agents))
    out["deliv_curve"] = h["delivered"]
    return out


def plan(name):
    J = []
    if name == "main":
        conds = ["dual_learn", "dual_frozen", "teacher", "bo_learn", "bo_frozen", "aux_learn", "aux_frozen",
                 "dual_learn_step", "dual_learn_sigma", "dual_learn_reh", "dual_learn_A", "teacher_nomem"]
        J = [(c, "two_patch", s) for c in conds for s in range(20)]
    elif name == "bridge":
        conds = ["dual_learn", "dual_frozen", "teacher", "bo_learn", "bo_frozen", "aux_learn", "aux_frozen"]
        J = [(c, "double_bridge", s) for c in conds for s in range(20)]
    elif name == "gen":
        conds = ["teacher", "bo_frozen", "dual_frozen", "aux_frozen"]
        J = [(c, e, s) for e in ("barrier", "food_fast", "food_slow", "decay_fast", "decay_slow")
             for c in conds for s in range(10)]
    elif name == "gen_learn":
        conds = ["dual_learn", "dual_learn_reh", "bo_learn", "aux_learn"]      # 食物移動頻率的劑量—反應檢驗
        J = [(c, e, s) for e in ("food_fast", "food_slow") for c in conds for s in range(10)]
    elif name == "reh_split":
        J = [(c, "two_patch", s) for c in ("dual_learn_rehNS", "dual_learn_rehS") for s in range(20)]
    elif name.startswith("gen_var"):                      # 例如 gen_var:reh,step
        vs = name.split(":")[1].split(",") if ":" in name else ["reh"]
        J = [(f"dual_learn_{v}", e, s) for e in ("food_fast", "food_slow") for v in vs for s in range(10)]
    return J


if __name__ == "__main__":
    plans = os.environ.get("LA2_R2_PLAN", "main").split(";")
    jobs = []
    for p in plans:
        jobs += plan(p)
    cost = lambda j: (0 if "learn" in j[0] and j[0].startswith("dual") else 1 if "learn" in j[0] else 2)
    jobs.sort(key=cost)                                    # 長的工作先跑，減少尾端閒置
    t0 = time.time()
    res = pmap(job, jobs, workers=int(os.environ.get("LA2_WORKERS", "2")))
    print("done", len(res), round(time.time() - t0), "s", flush=True)

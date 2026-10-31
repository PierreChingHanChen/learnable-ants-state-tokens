"""LA2 E5：消融——上下文長度（4/8/32 步）、嵌入寬度 32、訓練資料比例（10%、30%）；bo／dual 各 3 個種子。
主設定（16 步、寬度 16、全部資料）取自 E0 的種子 0–2。指標：保留集行動 NLL／準確率、dual 自由追蹤狀態準確率。"""
import os, pickle, json
import numpy as np
from common import pmap, RESULTS
from miat_ants.la2 import pretrain as P, metrics as M
from miat_ants.la2.system import base_path
from miat_ants.microgpt import MicroGPT

OUT = os.path.join(RESULTS, "la2")
CORPUS = os.path.join(OUT, "corpus.pkl")
SEEDS = int(os.environ.get("LA2_SEEDS", "3"))
CONFIGS = [("ctx4", dict(ctx_steps=4)), ("ctx8", dict(ctx_steps=8)), ("ctx32", dict(ctx_steps=32)),
           ("embd32", dict(n_embd=32)), ("data10", dict(frac=0.1)), ("data30", dict(frac=0.3)),
           ("main", {})]


JOBDIR = os.path.join(OUT, "e5_jobs")


def job(args):
    name, kw, mode, seed = args
    fn = os.path.join(JOBDIR, f"{name}_{mode}_{seed}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    st = _job(args)
    json.dump(st, open(fn, "w"), default=float)
    return st


def _job(args):
    name, kw, mode, seed = args
    eps = pickle.load(open(CORPUS, "rb"))
    tr, va = P.split(eps, seed=0)
    kw = dict(kw)
    frac = kw.pop("frac", 1.0)
    if name == "main":
        model = MicroGPT.load(base_path(mode, f"_s{seed}"))
        st = P.evaluate(model, va, mode, np.random.default_rng(seed + 1))
    else:
        rng = np.random.default_rng(100 + seed)
        trs = [tr[i] for i in rng.choice(len(tr), int(len(tr) * frac), replace=False)] if frac < 1 else tr
        model, st = P.train(mode, trs, va, seed=seed, **kw)
        st = {k: v for k, v in st.items() if k != "curve"}
    vsub = [va[i] for i in np.random.default_rng(seed).choice(len(va), 300, replace=False)]
    if mode == "dual":
        t, p = M.free_run_states(model, vsub)
        st["free"] = M.per_state_acc(t, p)
    st.update({"config": name, "mode": mode, "seed": seed})
    return st


if __name__ == "__main__":
    os.makedirs(JOBDIR, exist_ok=True)
    jobs = [(n, kw, m, s) for n, kw in CONFIGS for m in ("bo", "dual") for s in range(SEEDS)]
    jobs.sort(key=lambda j: -j[1].get("ctx_steps", 16))
    res = pmap(job, jobs)
    json.dump(res, open(os.path.join(OUT, "e5.json"), "w"), indent=1, default=float)
    for r in res:
        print(r["config"], r["mode"], r["seed"], round(r["action_acc"], 4), round(r.get("free", {}).get("all", np.nan), 4))

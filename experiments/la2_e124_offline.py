"""LA2 E1／E2／E4（離線）：平凡性對照、隱藏狀態恢復、讀回控制器。"""
import os, pickle, json
import numpy as np
from common import pmap, RESULTS, bootstrap_ci
from miat_ants.la2 import metrics as M, vocab as V, pretrain as P
from miat_ants.la2.system import base_path
from miat_ants.microgpt import MicroGPT, GPTConfig

OUT = os.path.join(RESULTS, "la2")
CORPUS = os.path.join(OUT, "corpus.pkl")
SEEDS = int(os.environ.get("LA2_SEEDS", "5"))
N_FREE = int(os.environ.get("LA2_NFREE", "400"))
TAG = os.environ.get("LA2_TAG", "")              # 例如 E3 的雜訊語料
CORP = os.environ.get("LA2_CORPUS", CORPUS)
WTAG = os.environ.get("LA2_WTAG", "")            # 權重檔名附加標籤


def load():
    eps = pickle.load(open(CORP, "rb"))
    return P.split(eps, seed=0)


def per_seed(seed):
    tr, va = load()
    rng = np.random.default_rng(seed)
    vsub = [va[i] for i in rng.choice(len(va), min(N_FREE, len(va)), replace=False)]
    out = {"seed": seed}
    dual = MicroGPT.load(base_path("dual", f"{WTAG}_s{seed}"))
    bo = MicroGPT.load(base_path("bo", f"{WTAG}_s{seed}"))
    out["dual_eval"] = P.evaluate(dual, va, "dual", np.random.default_rng(seed + 1))
    out["bo_eval"] = P.evaluate(bo, va, "bo", np.random.default_rng(seed + 1))
    t, p = M.free_run_states(dual, vsub)
    out["dual_free"] = M.per_state_acc(t, p)
    # 線性探針：bo 模型 vs 隨機初始化 vs 只用當步感知詞
    Xtr, ytr, otr = M.hidden_at_obs(bo, tr, 20000, np.random.default_rng(seed + 2))
    Xte, yte, ote = M.hidden_at_obs(bo, va, 10000, np.random.default_rng(seed + 3))
    out["bo_probe"], _ = M.probe_accuracy(Xtr, ytr, Xte, yte)
    rnd = MicroGPT(bo.cfg, n_agents=1, seed=1000 + seed)
    Rtr, rtr, _ = M.hidden_at_obs(rnd, tr, 20000, np.random.default_rng(seed + 2))
    Rte, rte, _ = M.hidden_at_obs(rnd, va, 10000, np.random.default_rng(seed + 3))
    out["rand_probe"], _ = M.probe_accuracy(Rtr, rtr, Rte, rte)
    out["obs_probe"], _ = M.probe_accuracy(M.onehot_obs(otr), ytr, M.onehot_obs(ote), yte)
    # E4：分群純度、因果介入
    for k in (8, 16):
        out[f"purity_bo_k{k}"] = M.kmeans_purity(Xte, yte, k, seed)
        out[f"purity_rand_k{k}"] = M.kmeans_purity(Rte, rte, k, seed)
    tables = M.teacher_tables(tr)
    out["intervene"] = M.intervene_state(dual, vsub, tables, np.random.default_rng(seed + 4))
    return out


def main():
    tr, va = load()
    e1 = {"H_s": M.state_entropy(tr)}
    for kind in ("o", "o,a_prev", "s_prev,o"):
        e1[f"H_s|{kind}"] = M.cond_entropy(tr, kind)
    ngram = {}
    for n in (1, 2, 4, 8, 16):
        ngram[n] = M.eval_ngram(tr, va, n)
    occ = np.bincount(np.concatenate([M.sidx(M.steps_of(e)[1]) for e in tr]), minlength=len(V.STATES))
    e1["occupancy"] = dict(zip(V.STATES, (occ / occ.sum()).round(4).tolist()))
    res = pmap(per_seed, range(SEEDS))
    json.dump({"e1": e1, "ngram": ngram, "seeds": res}, open(os.path.join(OUT, f"e124{TAG}.json"), "w"),
              indent=1, default=float)
    print(json.dumps(e1, indent=1, ensure_ascii=False))
    for n, v in ngram.items():
        print("ngram", n, round(v["all"], 4), round(v["balanced"], 4))
    for key in ("dual_free", "bo_probe", "rand_probe", "obs_probe"):
        a = [r[key]["all"] for r in res]; b = [r[key]["balanced"] for r in res]
        print(key, round(np.mean(a), 4), round(np.mean(b), 4))
    print("intervene", [r["intervene"] for r in res])


if __name__ == "__main__":
    main()

"""R1：離線狀態追蹤——所有方法在「完全相同」的 1,078 個保留回合上評估，並以回合為單位配對比較。

方法（每個預測都是在行動 a_t 之前，對教師狀態 s_t 的估計）：
  lookup / n-gram（n = 1…16）     固定窗口的非參數查表（訓練回合計數）
  filter                          有限狀態估計器（IO-HMM 前向濾波；metrics.BayesStateFilter）
  bo probe                        bo 轉換器最後一層的線性探針（不監督狀態、不回饋）
  aux                             轉換器＋狀態讀出頭（監督狀態、不回饋）
  dual                            轉換器，說出狀態並把它放回輸入（監督＋回饋；自由追蹤）
  GRU-bo probe / GRU-aux          參數量相近的循環網路（記憶不受窗口限制）

輸出：results/la2/r1/*.pkl（逐回合預測）與 r1_summary.json。
環境變數 LA2_R1_PART 指定要跑的部分（main, ctx, noise, gen；預設全部）。"""
import os, pickle, json, sys
import numpy as np
from common import pmap, RESULTS
from miat_ants.la2 import metrics as M, vocab as V, pretrain as P, gru as G
from miat_ants.la2.system import base_path, LA2Config, LA2System, WEIGHTS_DIR
from miat_ants.microgpt import MicroGPT

OUT = os.path.join(RESULTS, "la2")
R1 = os.path.join(OUT, "r1")
os.makedirs(R1, exist_ok=True)
CORPUS = os.path.join(OUT, "corpus.pkl")
N_SEEDS = int(os.environ.get("LA2_NSEEDS", "10"))
PARTS = os.environ.get("LA2_R1_PART", "main,ctx,noise,gen").split(",")
NGRAMS = (1, 2, 4, 8, 16)

# 泛化測試環境（教師在未見過的設定中產生回合；模型與 n-gram／濾波器都只用原訓練語料）
ENVS = {
    "train": dict(map_name="two_patch"),
    "barrier": dict(map_name="barrier"),
    "food_fast": dict(map_name="two_patch", map_kw={"per_cell": 10}),
    "food_slow": dict(map_name="two_patch", map_kw={"per_cell": 90}),
    "decay_fast": dict(map_name="two_patch", rho=0.94),
    "decay_slow": dict(map_name="two_patch", rho=0.99),
}
GEN_SEED, GEN_STEPS, GEN_MAX_TRIPS = 900, 1500, 1000


def gru_path(obj, seed):
    return os.path.join(WEIGHTS_DIR, f"gru_{obj}_s{seed}.npz")


def load_split(path=CORPUS):
    return P.split(pickle.load(open(path, "rb")), seed=0)


def true_states(eps):
    return [M.steps_of(e)[1] for e in eps]


def _probe(Htr, ytr, Hte_list):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Htr, ytr)
    return [V.STATE_IDS[clf.predict(h)] for h in Hte_list], clf


def _train_positions(per_trip_h, eps, n, rng):
    X = np.concatenate(per_trip_h)
    y = M.sidx(np.concatenate(true_states(eps)))
    sel = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[sel], y[sel]


# --------------------------------------------------------------------- 每個訓練種子
def seed_job(args):
    seed, tag, eval_name = args
    fn = os.path.join(R1, f"seed_{eval_name}_{seed}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    tr, va = load_split()
    eps = va if eval_name == "heldout" else pickle.load(open(os.path.join(R1, f"gen_{eval_name}_eps.pkl"), "rb"))
    out = {"seed": seed, "pred": {}, "eval": {}}
    rng = np.random.default_rng(seed + 2)
    # 訓練回合子集：探針訓練（20,000 個位置）
    tr_sub = [tr[i] for i in np.random.default_rng(seed).choice(len(tr), 1500, replace=False)]
    dual = MicroGPT.load(base_path("dual", f"_s{seed}"))
    out["pred"]["dual"] = M.free_run_multi(dual, eps, [0])[0]
    aux = MicroGPT.load(base_path("aux", f"_s{seed}"))
    out["pred"]["aux"] = M.aux_readout_multi(aux, eps, [0])[0]
    bo = MicroGPT.load(base_path("bo", f"_s{seed}"))
    Xtr, ytr = _train_positions(M.bo_hidden_per_trip(bo, tr_sub), tr_sub, 20000, rng)
    out["pred"]["bo_probe"], _ = _probe(Xtr, ytr, M.bo_hidden_per_trip(bo, eps))
    ga = G.GRUModel.load(gru_path("aux", seed))
    res = G.run_eps(ga, eps)
    out["pred"]["gru_aux"] = [np.where(V.STATE_MASK, r["logits"], -np.inf).argmax(-1) for r in res]
    gb = G.GRUModel.load(gru_path("bo", seed))
    htr = [r["h"] for r in G.run_eps(gb, tr_sub)]
    Xtr, ytr = _train_positions(htr, tr_sub, 20000, rng)
    out["pred"]["gru_bo_probe"], _ = _probe(Xtr, ytr, [r["h"] for r in G.run_eps(gb, eps)])
    if eval_name == "heldout":
        # 控制組探針：隨機初始化的轉換器
        rnd = MicroGPT(bo.cfg, n_agents=1, seed=1000 + seed)
        Xtr, ytr = _train_positions(M.bo_hidden_per_trip(rnd, tr_sub), tr_sub, 20000, rng)
        out["pred"]["rand_probe"], _ = _probe(Xtr, ytr, M.bo_hidden_per_trip(rnd, eps))
        # 行動預測（保留回合）
        r_ = np.random.default_rng(seed + 1)
        out["eval"]["dual"] = P.evaluate(dual, va, "dual", np.random.default_rng(seed + 1))
        out["eval"]["bo"] = P.evaluate(bo, va, "bo", np.random.default_rng(seed + 1))
        out["eval"]["aux"] = P.evaluate(aux, va, "aux", np.random.default_rng(seed + 1))
        out["eval"]["gru_aux"] = G.evaluate(ga, va)
        out["eval"]["gru_bo"] = G.evaluate(gb, va)
        # 介入（論文 4.6 節）：每個訓練種子從全部保留回合中抽 4,000 個案例
        out["intervene"] = M.intervene_state(dual, va, M.teacher_tables(tr), np.random.default_rng(seed + 4))
    pickle.dump(out, open(fn, "wb"))
    return out


def deterministic(eps, tr, name):
    fn = os.path.join(R1, f"det_{name}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    out = {f"ngram{n}": M.ngram_per_trip(tr, eps, n) for n in NGRAMS}
    f = M.BayesStateFilter().fit(tr)
    out["filter"] = [f.predict_ep(e) for e in eps]
    pickle.dump(out, open(fn, "wb"))
    return out


def summarize(eps, det, seeds, methods_seeded):
    """每個方法：跨種子平均的整體與各狀態準確率（含最小／最大）；以及 dual 對其他方法的回合配對比較。"""
    ts = true_states(eps)
    allT = np.concatenate(ts)
    summ = {}
    for k, pr in det.items():
        summ[k] = M.per_state_acc(allT, np.concatenate(pr))
    corr = {}
    for m in methods_seeded:
        accs = [M.per_state_acc(allT, np.concatenate(s["pred"][m])) for s in seeds if m in s["pred"]]
        if not accs:
            continue
        summ[m] = {k: float(np.nanmean([a[k] for a in accs])) for k in accs[0]}
        summ[m]["min"] = {k: float(np.nanmin([a[k] for a in accs])) for k in accs[0]}
        summ[m]["max"] = {k: float(np.nanmax([a[k] for a in accs])) for k in accs[0]}
        summ[m]["per_seed"] = [a["all"] for a in accs]
        summ[m]["per_seed_sigma"] = [a["σ"] for a in accs]
        # 每個回合每一步：跨種子的平均正確率
        corr[m] = [np.mean([np.asarray(s["pred"][m][i]) == ts[i] for s in seeds], 0) for i in range(len(eps))]
    for k, pr in det.items():
        corr[k] = [np.asarray(pr[i]) == ts[i] for i in range(len(eps))]
    pairs = {}
    for a in ("dual", "aux", "gru_aux"):
        if a not in corr:
            continue
        for b in corr:
            if b == a:
                continue
            for st in (None, "σ", "φ", "ε", "ν"):
                pairs[f"{a}|{b}|{st or 'all'}"] = M.trip_bootstrap(corr[a], corr[b], ts, st)
    # 各種子對 n-gram-8 的勝負
    if "dual" in methods_seeded and "ngram8" in summ:
        summ["dual_seeds_above_ngram8"] = int(sum(x > summ["ngram8"]["all"] for x in summ["dual"]["per_seed"]))
    return summ, pairs


def export_perseed(va, seeds, methods=("dual", "aux", "gru_aux", "bo_probe", "gru_bo_probe", "rand_probe")):
    """每個訓練種子、每個方法的整體與各狀態準確率 → results/la2/r1_perseed.json（補充表 S8）。"""
    allT = np.concatenate(true_states(va))
    rows = [{"seed": int(s["seed"]), **{m: M.per_state_acc(allT, np.concatenate(s["pred"][m]))
                                         for m in methods if m in s["pred"]}} for s in seeds]
    json.dump(rows, open(os.path.join(OUT, "r1_perseed.json"), "w"), default=float, ensure_ascii=False)
    return rows


# --------------------------------------------------------------------- 部分一：主比較
def part_main():
    tr, va = load_split()
    det = deterministic(va, tr, "heldout")
    seeds = pmap(seed_job, [(s, "", "heldout") for s in range(N_SEEDS)], workers=2)
    summ, pairs = summarize(va, det, seeds, ["dual", "aux", "bo_probe", "gru_aux", "gru_bo_probe", "rand_probe"])
    # 只用當步感知詞的探針（控制組）
    rng = np.random.default_rng(0)
    oh = lambda E: [M.onehot_obs(M.steps_of(e)[0]) for e in E]
    Xtr, ytr = _train_positions(oh(tr[:1500]), tr[:1500], 20000, rng)
    pr, _ = _probe(Xtr, ytr, oh(va))
    summ["obs_probe"] = M.per_state_acc(np.concatenate(true_states(va)), np.concatenate(pr))
    ev = {}
    for m in ("dual", "bo", "aux", "gru_aux", "gru_bo"):
        e = [s["eval"][m] for s in seeds]
        ev[m] = {k: float(np.mean([x[k] for x in e])) for k in e[0]}
    occ = np.bincount(M.sidx(np.concatenate(true_states(va))), minlength=8)
    iv = [s["intervene"] for s in seeds]
    export_perseed(va, seeds)
    json.dump({"summary": summ, "pairs": pairs, "eval": ev, "n_trips": len(va), "intervene": iv,
               "occupancy_heldout": dict(zip(V.STATES, (occ / occ.sum()).tolist()))},
              open(os.path.join(OUT, "r1_main.json"), "w"), indent=1, default=float, ensure_ascii=False)
    for k in ("ngram1", "ngram8", "ngram16", "filter", "bo_probe", "aux", "gru_bo_probe", "gru_aux", "dual"):
        s = summ[k]
        print(f"{k:14s} all {s['all']:.4f} σ {s['σ']:.4f} φ {s['φ']:.4f} ε {s['ε']:.4f} ν {s['ν']:.4f} κ {s['κ']:.3f} υ {s['υ']:.3f}")
    for k in ("dual|ngram8|all", "dual|ngram8|σ", "dual|filter|all", "dual|filter|σ", "dual|gru_aux|all",
              "dual|gru_aux|σ", "dual|aux|all", "dual|aux|σ"):
        print(k, {a: round(b, 4) for a, b in pairs[k].items()})


# --------------------------------------------------------------------- 部分二：上下文長度（監督 vs 回饋）
def ctx_job(args):
    mode, ctx, seed = args
    fn = os.path.join(R1, f"ctx_{mode}_{ctx}_{seed}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    tr, va = load_split()
    tag = (f"_ctx{ctx}" if ctx != 16 else "") + f"_s{seed}"
    m = MicroGPT.load(base_path(mode, tag))
    pred = M.free_run_multi(m, va, [0])[0] if mode == "dual" else M.aux_readout_multi(m, va, [0])[0]
    out = {"mode": mode, "ctx": ctx, "seed": seed, "pred": pred,
           "eval": P.evaluate(m, va, mode, np.random.default_rng(seed + 1))}
    pickle.dump(out, open(fn, "wb"))
    return out


def part_ctx():
    tr, va = load_split()
    det = deterministic(va, tr, "heldout")
    res = pmap(ctx_job, [(m, c, s) for m in ("dual", "aux") for c in (4, 8, 16) for s in range(3)], workers=2)
    ts = true_states(va)
    allT = np.concatenate(ts)
    rows = []
    for m in ("dual", "aux"):
        for c in (4, 8, 16):
            rr = [r for r in res if r["mode"] == m and r["ctx"] == c]
            accs = [M.per_state_acc(allT, np.concatenate(r["pred"])) for r in rr]
            rows.append({"mode": m, "ctx": c, **{k: float(np.mean([a[k] for a in accs])) for k in accs[0]},
                         "min_all": float(min(a["all"] for a in accs)), "max_all": float(max(a["all"] for a in accs)),
                         "min_σ": float(min(a["σ"] for a in accs)), "max_σ": float(max(a["σ"] for a in accs)),
                         "action_nll": float(np.mean([r["eval"]["action_nll"] for r in rr]))})
    for n in NGRAMS:
        rows.append({"mode": "ngram", "ctx": n, **M.per_state_acc(allT, np.concatenate(det[f"ngram{n}"]))})
    # 配對：同一上下文長度下 dual − aux（σ 與整體），回合 bootstrap（跨三個種子平均）
    pairs = {}
    for c in (4, 8, 16):
        cd = [np.mean([np.asarray(r["pred"][i]) == ts[i] for r in res if r["mode"] == "dual" and r["ctx"] == c], 0)
              for i in range(len(va))]
        ca = [np.mean([np.asarray(r["pred"][i]) == ts[i] for r in res if r["mode"] == "aux" and r["ctx"] == c], 0)
              for i in range(len(va))]
        for st in (None, "σ", "ε", "φ"):
            pairs[f"ctx{c}|dual-aux|{st or 'all'}"] = M.trip_bootstrap(cd, ca, ts, st)
    json.dump({"rows": rows, "pairs": pairs}, open(os.path.join(OUT, "r1_ctx.json"), "w"), indent=1, default=float,
              ensure_ascii=False)
    for r in rows:
        print(r["mode"], r["ctx"], round(r["all"], 4), "σ", round(r["σ"], 4), "ε", round(r["ε"], 4), "φ", round(r["φ"], 4))
    for k, v in pairs.items():
        print(k, round(v["diff"], 4), [round(v["lo"], 4), round(v["hi"], 4)])


# --------------------------------------------------------------------- 部分三：感知雜訊（配對重評）
def noise_job(args):
    p, seed = args
    tg = f"_f{int(round(p * 100)):02d}"
    fn = os.path.join(R1, f"noise{tg}_{seed}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    tr, va = load_split(os.path.join(OUT, f"corpus{tg}.pkl"))
    m = MicroGPT.load(base_path("dual", f"{tg}_s{seed}"))
    out = {"p": p, "seed": seed, "pred": M.free_run_multi(m, va, [0])[0]}
    pickle.dump(out, open(fn, "wb"))
    return out


def part_noise():
    levels = (0.05, 0.1, 0.2)
    for p in levels:                                # 雜訊語料（與 la2_e3_noise.py 相同種子；確定性重建）
        path = os.path.join(OUT, f"corpus_f{int(round(p * 100)):02d}.pkl")
        if not os.path.exists(path):
            pickle.dump(P.collect(obs_flip=p), open(path, "wb"))
    res = pmap(noise_job, [(p, s) for p in levels for s in range(3)], workers=2)
    rows, pairs = [], {}
    for p in levels:
        tg = f"_f{int(round(p * 100)):02d}"
        tr, va = load_split(os.path.join(OUT, f"corpus{tg}.pkl"))
        det = deterministic(va, tr, f"noise{tg}")
        ts = true_states(va)
        allT = np.concatenate(ts)
        rr = [r for r in res if r["p"] == p]
        accs = [M.per_state_acc(allT, np.concatenate(r["pred"])) for r in rr]
        row = {"p": p, "n_trips": len(va), "dual": {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}}
        for k in ("ngram1", "ngram8", "filter"):
            row[k] = M.per_state_acc(allT, np.concatenate(det[k]))
        rows.append(row)
        cd = [np.mean([np.asarray(r["pred"][i]) == ts[i] for r in rr], 0) for i in range(len(va))]
        for b in ("ngram8", "filter"):
            cb = [np.asarray(det[b][i]) == ts[i] for i in range(len(va))]
            for st in (None, "σ", "φ"):
                pairs[f"p{p}|dual|{b}|{st or 'all'}"] = M.trip_bootstrap(cd, cb, ts, st)
    json.dump({"rows": rows, "pairs": pairs}, open(os.path.join(OUT, "r1_noise.json"), "w"), indent=1, default=float,
              ensure_ascii=False)
    for r in rows:
        print(r["p"], r["n_trips"], {k: round(r[k]["all"], 4) for k in ("dual", "ngram8", "filter")},
              "σ", {k: round(r[k]["σ"], 4) for k in ("dual", "ngram8", "filter")},
              "φ", {k: round(r[k]["φ"], 4) for k in ("dual", "ngram8", "filter")})


# --------------------------------------------------------------------- 部分四：未見環境的離線泛化
def gen_collect(name):
    path = os.path.join(R1, f"gen_{name}_eps.pkl")
    if os.path.exists(path):
        return path
    kw = dict(ENVS[name])
    r = LA2System(LA2Config(core="teacher", seed=GEN_SEED, max_steps=GEN_STEPS, collect_episodes=True, **kw)).run()
    eps = r.episodes
    if len(eps) > GEN_MAX_TRIPS:
        sel = np.sort(np.random.default_rng(0).choice(len(eps), GEN_MAX_TRIPS, replace=False))
        eps = [eps[i] for i in sel]
    pickle.dump(eps, open(path, "wb"))
    return path


def part_gen():
    tr, va = load_split()
    pmap(gen_collect, list(ENVS), workers=2)
    jobs = [(s, "", name) for name in ENVS for s in range(N_SEEDS)]
    res = pmap(seed_job, jobs, workers=2)
    out = {}
    for name in ENVS:
        eps = pickle.load(open(os.path.join(R1, f"gen_{name}_eps.pkl"), "rb"))
        det = deterministic(eps, tr, f"gen_{name}")
        seeds = [r for (s, _, n), r in zip(jobs, res) if n == name]
        summ, pairs = summarize(eps, det, seeds, ["dual", "aux", "bo_probe", "gru_aux", "gru_bo_probe"])
        occ = np.bincount(M.sidx(np.concatenate(true_states(eps))), minlength=8)
        out[name] = {"summary": summ, "pairs": {k: v for k, v in pairs.items() if k.startswith("dual|")},
                     "n_trips": len(eps), "occupancy": dict(zip(V.STATES, (occ / occ.sum()).tolist()))}
        s = summ
        print(name, len(eps), {k: round(s[k]["all"], 4) for k in ("dual", "aux", "gru_aux", "filter", "ngram8", "bo_probe")},
              "σ", {k: round(s[k]["σ"], 4) for k in ("dual", "aux", "gru_aux", "filter", "ngram8")})
    json.dump(out, open(os.path.join(OUT, "r1_gen.json"), "w"), indent=1, default=float, ensure_ascii=False)


# --------------------------------------------------------------------- 部分五：行動預測（相同位置）
def _tf_action(model, eps, mode, states=None):
    """轉換器在每個保留回合每一步的行動 NLL 與 top-1（滑動上下文）。dual 可給定上下文中的狀態詞（教師或自己預測）。"""
    from collections import defaultdict
    B = model.cfg.block_size
    items = []
    for i, e in enumerate(eps):
        O, S, A = M.steps_of(e)
        if mode == "dual":
            Sx = S if states is None else states[i]
            seq = [V.BOS_ID]
            for t in range(len(O)):
                seq += [int(O[t]), int(Sx[t])]
                items.append((seq[-B:], int(A[t])))
                seq.append(int(A[t]))
        else:
            bo = V.to_bo(e)
            for t in range(len(O)):
                items.append((bo[max(0, 2 * t + 2 - B):2 * t + 2], int(A[t])))
    by = defaultdict(list)
    for c, a in items:
        by[len(c)].append((c, a))
    nll, acc = [], []
    for L, its in by.items():
        for c0 in range(0, len(its), 4000):
            ch = its[c0:c0 + 4000]
            p = model.next_token_probs(np.array([c for c, _ in ch]), np.zeros(len(ch), int), V.ACTION_MASK)
            a = np.array([a for _, a in ch])
            nll += list(-np.log(p[np.arange(len(a)), a] + 1e-12)); acc += list(p.argmax(1) == a)
    return {"action_nll": float(np.mean(nll)), "action_acc": float(np.mean(acc)), "n": len(acc)}


def act_job(seed):
    tr, va = load_split()
    out = {"seed": seed}
    out["bo"] = _tf_action(MicroGPT.load(base_path("bo", f"_s{seed}")), va, "bo")
    out["aux"] = _tf_action(MicroGPT.load(base_path("aux", f"_s{seed}")), va, "aux")
    dual = MicroGPT.load(base_path("dual", f"_s{seed}"))
    out["dual_teacher_states"] = _tf_action(dual, va, "dual")
    own = M.free_run_multi(dual, va, [0])[0]
    out["dual_own_states"] = _tf_action(dual, va, "dual", own)
    for o in ("bo", "aux"):
        e = G.evaluate(G.GRUModel.load(gru_path(o, seed)), va)
        out[f"gru_{o}"] = {"action_nll": e["action_nll"], "action_acc": e["action_acc"]}
    return out


def part_act():
    res = pmap(act_job, range(N_SEEDS), workers=2)
    summ = {}
    for k in ("bo", "aux", "dual_teacher_states", "dual_own_states", "gru_bo", "gru_aux"):
        summ[k] = {m: float(np.mean([r[k][m] for r in res])) for m in ("action_nll", "action_acc")}
        summ[k]["nll_min"] = float(min(r[k]["action_nll"] for r in res)); summ[k]["nll_max"] = float(max(r[k]["action_nll"] for r in res))
        print(k, {a: round(b, 4) for a, b in summ[k].items()})
    json.dump({"summary": summ, "per_seed": res}, open(os.path.join(OUT, "r1_act.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    for part in PARTS:
        print(f"===== {part}", flush=True)
        {"main": part_main, "ctx": part_ctx, "noise": part_noise, "gen": part_gen, "act": part_act}[part]()

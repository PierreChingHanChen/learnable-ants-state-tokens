"""
LA2 分析工具
============
* steps_of：dual 回合 → 每步的 (o, s, a)
* cond_entropy / NGramState：平凡性對照與 n-gram 參考線（E1、E2）
* free_run_states：dual 模型「只看行為、自己說狀態」的追蹤準確率（E2）
* hidden_at_obs / probe_accuracy / kmeans_purity：bo 模型的線性探針與分群（E2、E4）
* intervene_state：替換狀態詞，看行動是否跟著改變（E4 因果檢驗）
* teacher_tables / legality：教師的狀態轉移與 (o,s)→a 支撐集，用於閉環合法率（E6）
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from . import vocab as V

NS = len(V.STATES)


def steps_of(ep: List[int]):
    t = ep[1:]
    n = len(t) // 3
    return np.array(t[0:3 * n:3]), np.array(t[1:3 * n:3]), np.array(t[2:3 * n:3])


def sidx(s_tok):
    return np.array([V.STATE_INDEX[int(x)] for x in np.atleast_1d(s_tok)])


# ------------------------------------------------------------------ E1 熵
def _feat(O, S, A, t, kind, n=1):
    if kind == "o":
        return (O[t],)
    if kind == "o,a_prev":
        return (O[t], A[t - 1] if t > 0 else -1)
    if kind == "s_prev,o":
        return (S[t - 1] if t > 0 else -1, O[t])
    if kind == "hist":
        lo = max(0, t - n + 1)
        return tuple(O[lo:t + 1]) + tuple(A[lo:t]) + ((-2,) if t - n + 1 < 0 else ())
    raise ValueError(kind)


def cond_entropy(eps, kind, n=1) -> float:
    """H(s_t | feature)（bits），以整個語料的計數估計。"""
    joint = defaultdict(Counter)
    for e in eps:
        O, S, A = steps_of(e)
        for t in range(len(O)):
            joint[_feat(O, S, A, t, kind, n)][int(S[t])] += 1
    tot = sum(sum(c.values()) for c in joint.values())
    h = 0.0
    for c in joint.values():
        m = sum(c.values())
        p = np.array(list(c.values())) / m
        h += m / tot * float(-(p * np.log2(p)).sum())
    return h


def state_entropy(eps) -> float:
    c = Counter()
    for e in eps:
        c.update(int(x) for x in steps_of(e)[1])
    p = np.array(list(c.values())) / sum(c.values())
    return float(-(p * np.log2(p)).sum())


class NGramState:
    """以最近 n 步行為歷史 (o…, a…) 預測當前狀態；查無時退回較短歷史。"""

    def __init__(self, n: int):
        self.n = n
        self.tab = [defaultdict(Counter) for _ in range(n + 1)]
        self.glob = Counter()

    def fit(self, eps):
        for e in eps:
            O, S, A = steps_of(e)
            for t in range(len(O)):
                for m in range(1, self.n + 1):
                    self.tab[m][_feat(O, S, A, t, "hist", m)][int(S[t])] += 1
                self.glob[int(S[t])] += 1
        return self

    def predict_ep(self, e):
        O, S, A = steps_of(e)
        out = np.empty(len(O), dtype=int)
        g = self.glob.most_common(1)[0][0]
        for t in range(len(O)):
            out[t] = g
            for m in range(self.n, 0, -1):
                c = self.tab[m].get(_feat(O, S, A, t, "hist", m))
                if c:
                    out[t] = c.most_common(1)[0][0]
                    break
        return out


def per_state_acc(true_s, pred_s) -> Dict:
    ts, ps = sidx(true_s), sidx(pred_s)
    out = {"all": float((ts == ps).mean())}
    for i, s in enumerate(V.STATES):
        m = ts == i
        out[s] = float((ps[m] == i).mean()) if m.any() else float("nan")
    out["balanced"] = float(np.nanmean([out[s] for s in V.STATES]))
    return out


def eval_ngram(train_eps, val_eps, n):
    ng = NGramState(n).fit(train_eps)
    T, P = [], []
    for e in val_eps:
        T.append(steps_of(e)[1]); P.append(ng.predict_ep(e))
    return per_state_acc(np.concatenate(T), np.concatenate(P))


# ------------------------------------------------------------- E2 自由追蹤
def free_run_states(model, val_eps, greedy: bool = True, rng=None):
    """dual 模型：輸入教師的 o 與 a，但上下文中的狀態詞由模型自己預測。回傳 (true, pred)。"""
    block = model.cfg.block_size
    steps = [steps_of(e) for e in val_eps]
    seqs = [[V.BOS_ID] for _ in val_eps]
    preds = [[] for _ in val_eps]
    T = max(len(s[0]) for s in steps)
    for t in range(T):
        act = [i for i, s in enumerate(steps) if len(s[0]) > t]
        for i in act:
            seqs[i].append(int(steps[i][0][t]))
        ctxs = [seqs[i][-block:] for i in act]
        lens = np.array([len(c) for c in ctxs])
        out = np.empty(len(act), dtype=int)
        for L in np.unique(lens):
            sel = np.flatnonzero(lens == L)
            p = model.next_token_probs(np.array([ctxs[j] for j in sel]), np.zeros(len(sel), int), V.STATE_MASK)
            out[sel] = p.argmax(1) if greedy else (rng.random((len(sel), 1)) > p.cumsum(1)).sum(1)
        for j, i in enumerate(act):
            preds[i].append(int(out[j]))
            seqs[i].append(int(out[j]))
            seqs[i].append(int(steps[i][2][t]))
    true = np.concatenate([s[1] for s in steps])
    pred = np.concatenate([np.array(p) for p in preds])
    return true, pred


# ------------------------------------------------------ E2/E4 探針與分群
def hidden_at_obs(model, eps_dual, n_max: int = 20000, rng=None, layer: str = "h3"):
    """bo 模型：在每個感知詞位置取最後一層的殘差表徵，標籤為教師當步狀態。"""
    rng = rng or np.random.default_rng(0)
    block = model.cfg.block_size
    items = []
    for e in eps_dual:
        O, S, A = steps_of(e)
        bo = V.to_bo(e)
        for t in range(len(O)):
            items.append((bo, 2 * t + 2, int(S[t]), int(O[t])))     # bo[:2t+2] 以 o_t 結尾
    if len(items) > n_max:
        items = [items[i] for i in rng.choice(len(items), n_max, replace=False)]
    by_len = defaultdict(list)
    for it in items:
        by_len[min(it[1], block)].append(it)
    X, y, o = [], [], []
    for L, its in by_len.items():
        for c0 in range(0, len(its), 2000):
            chunk = its[c0:c0 + 2000]
            ctx = np.array([it[0][it[1] - L:it[1]] for it in chunk])
            model.forward(ctx[None], np.array([0]), cache=True)
            X.append(model._c[layer][0, :, -1, :])
            y += [it[2] for it in chunk]
            o += [it[3] for it in chunk]
    return np.concatenate(X), sidx(np.array(y)), np.array(o)


def probe_accuracy(Xtr, ytr, Xte, yte) -> Dict:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    out = {"all": float((pred == yte).mean())}
    for i, s in enumerate(V.STATES):
        m = yte == i
        out[s] = float((pred[m] == i).mean()) if m.any() else float("nan")
    out["balanced"] = float(np.nanmean([out[s] for s in V.STATES]))
    return out, clf


def onehot_obs(o):
    X = np.zeros((len(o), V.V))
    X[np.arange(len(o)), o] = 1
    return X


def kmeans_purity(X, y, k: int, seed: int = 0) -> float:
    from sklearn.cluster import KMeans
    lab = KMeans(k, n_init=4, random_state=seed).fit_predict(X)
    return float(sum(np.bincount(y[lab == c], minlength=NS).max() for c in range(k) if (lab == c).any()) / len(y))


# ------------------------------------------------------------ 教師支撐集
def teacher_tables(eps):
    trans = np.zeros((NS, NS))
    osa = defaultdict(Counter)
    oa = defaultdict(Counter)
    for e in eps:
        O, S, A = steps_of(e)
        si = sidx(S)
        for t in range(len(O)):
            if t > 0:
                trans[si[t - 1], si[t]] += 1
            osa[(int(O[t]), int(S[t]))][int(A[t])] += 1
            oa[int(O[t])][int(A[t])] += 1
    return {"trans": trans, "osa": osa, "oa": oa}


def act_dist(counter) -> np.ndarray:
    p = np.zeros(len(V.ACTION_IDS))
    for a, c in counter.items():
        p[list(V.ACTION_IDS).index(a)] = c
    return p / max(p.sum(), 1)


def legality(eps, tables, mode: str, min_frac: float = 0.0) -> Dict:
    """閉環回合的合法率：狀態轉移、(o,s)→a、o→a 是否落在教師支撐集內。"""
    trans_ok = tables["trans"] > 0
    n_tr = ok_tr = n_osa = ok_osa = n_oa = ok_oa = 0
    occ = np.zeros(NS)
    for e in eps:
        if mode == "dual":
            O, S, A = steps_of(e)
            si = sidx(S)
            occ += np.bincount(si, minlength=NS)
            for t in range(len(O)):
                if t > 0:
                    n_tr += 1; ok_tr += trans_ok[si[t - 1], si[t]]
                c = tables["osa"].get((int(O[t]), int(S[t])))
                n_osa += 1; ok_osa += bool(c and c.get(int(A[t]), 0) > 0)
        else:
            t_ = e[1:]
            n = len(t_) // 2
            O, A = np.array(t_[0:2 * n:2]), np.array(t_[1:2 * n:2])
        for t in range(len(O)):
            c = tables["oa"].get(int(O[t]))
            n_oa += 1; ok_oa += bool(c and c.get(int(A[t]), 0) > 0)
    out = {"oa_legal": ok_oa / max(n_oa, 1)}
    if mode == "dual":
        out.update({"trans_legal": ok_tr / max(n_tr, 1), "osa_legal": ok_osa / max(n_osa, 1),
                    "occupancy": occ / max(occ.sum(), 1)})
    return out


def jsd(p, q):
    m = 0.5 * (p + q)
    H = lambda x: -(np.clip(x, 1e-12, 1) * np.log(np.clip(x, 1e-12, 1))).sum(-1)
    return 0.5 * (H(m) - H(p)) + 0.5 * (H(m) - H(q))


# ------------------------------------------------------------ E4 因果介入
def intervene_state(model, val_eps, tables, rng, n_max: int = 4000, js_min: float = 0.2):
    """在 dual 上下文中把最後一個狀態詞 s 換成 s'，比較行動分布更接近教師的 P(a|o,s') 或 P(a|o,s)。"""
    block = model.cfg.block_size
    cases = []
    for e in val_eps:
        O, S, A = steps_of(e)
        for t in range(len(O)):
            o, s = int(O[t]), int(S[t])
            alts = []
            for s2 in V.STATE_IDS:
                c2 = tables["osa"].get((o, int(s2)))
                if int(s2) != s and c2 and sum(c2.values()) >= 50:
                    if jsd(act_dist(tables["osa"][(o, s)]), act_dist(c2)) > js_min:
                        alts.append(int(s2))
            if alts:
                cases.append((e[:3 * t + 3], o, s, alts[rng.integers(len(alts))]))
    if len(cases) > n_max:
        cases = [cases[i] for i in rng.choice(len(cases), n_max, replace=False)]
    follow = []
    d_orig, d_new = [], []
    by_len = defaultdict(list)
    for c in cases:
        by_len[min(len(c[0]), block)].append(c)
    for L, cs in by_len.items():
        ctx = np.array([c[0][-L:] for c in cs])
        ctx2 = ctx.copy()
        ctx2[:, -1] = [c[3] for c in cs]
        p1 = model.next_token_probs(ctx, np.zeros(len(cs), int), V.ACTION_MASK)[:, V.ACTION_IDS]
        p2 = model.next_token_probs(ctx2, np.zeros(len(cs), int), V.ACTION_MASK)[:, V.ACTION_IDS]
        for j, c in enumerate(cs):
            ts = act_dist(tables["osa"][(c[1], c[2])])
            ts2 = act_dist(tables["osa"][(c[1], c[3])])
            follow.append(jsd(p2[j], ts2) < jsd(p2[j], ts))
            d_orig.append(jsd(p1[j], ts)); d_new.append(jsd(p2[j], ts2))
    return {"n": len(follow), "follow_frac": float(np.mean(follow)) if follow else float("nan"),
            "js_orig": float(np.mean(d_orig)) if d_orig else float("nan"),
            "js_swapped": float(np.mean(d_new)) if d_new else float("nan")}


# ------------------------------------------------------------ E6 標記調節
def mark_by_quality(eps, mode: str) -> Dict:
    """返巢移動中的標記率：依食物品質（X/Y）、腳下路徑強弱、距拿取的步數、本趟是否搜尋過。"""
    st = 3 if mode == "dual" else 2
    aoff = st - 1
    acc = defaultdict(lambda: [0, 0])
    X, Y, Z = V.STOI["X"], V.STOI["Y"], V.STOI["z"]
    T = V.STOI["T"]
    move_m = {V.STOI[a] for a in "flr"}
    move_all = {V.STOI[a] for a in "FLRflr"}
    weak = {V.STOI[c] for c in "1234"}
    strong = {V.STOI[c] for c in "5678"}
    for e in eps:
        t_ = e[1:]
        n = len(t_) // st
        O = t_[0:st * n:st]
        A = t_[aoff:st * n:st]
        q, since, searched = None, 0, False
        for i in range(n):
            o, a = O[i], A[i]
            if o == Z:
                searched = True
            if q is None:
                if a == T and o in (X, Y) and i + 1 < n and (O[i + 1] in weak or O[i + 1] in strong):
                    q, since = (1 if o == X else 0), 0
                continue
            since += 1
            if a in move_all and (o in weak or o in strong):
                m = a in move_m
                for key in (f"q{q}", f"q{q}_{'strong' if o in strong else 'weak'}",
                            f"q{q}_{'near' if since <= 8 else 'far'}", f"searched{int(searched)}_q{q}"):
                    acc[key][0] += m
                    acc[key][1] += 1
    return {k: v[0] / v[1] for k, v in acc.items() if v[1] >= 20}


def make_probes(eps_dual, mode: str, n: int = 96, ctx_steps: int = 16, seed: int = 0) -> np.ndarray:
    """探針上下文：bo 以 o 結尾；dual 以 s 結尾（教師狀態）。依結尾感知詞分層抽樣。"""
    rng = np.random.default_rng(seed)
    st = V.stride(mode)
    block = st * ctx_steps
    by_obs = defaultdict(list)
    for e in eps_dual:
        seq = e if mode == "dual" else V.to_bo(e)
        nsteps = (len(seq) - 1) // st
        for t in range(ctx_steps, nsteps):
            end = 1 + st * t + (2 if mode == "dual" else 1)       # 結尾含 o（與 s）
            if end - block >= 0:
                by_obs[seq[1 + st * t]].append(seq[end - block:end])
    keys = sorted(by_obs)
    out = []
    while len(out) < n:
        for k in keys:
            if len(out) >= n:
                break
            out.append(by_obs[k][rng.integers(len(by_obs[k]))])
    return np.array(out)


# ================================================================ 有限狀態濾波器、多 agent 追蹤、配對比較
OBS_I = {int(o): i for i, o in enumerate(V.OBS_IDS)}
ACT_I = {int(a): i for i, a in enumerate(V.ACTION_IDS)}
NO, NA = len(V.OBS_IDS), len(V.ACTION_IDS)


class BayesStateFilter:
    """有限狀態估計器：把教師的八個狀態當成輸入—輸出隱馬可夫模型（IO-HMM）的隱藏狀態，
    以訓練回合的計數估計
        轉移  P(s_t | s_{t-1}, a_{t-1}, o_t)   （階層式 Dirichlet 平滑，退回 P(s_t | s_{t-1}, o_t)、P(s_t | o_t)）
        發射  P(a_t | s_t, o_t)                （退回 P(a_t | s_t)）
        初始  P(s_0 | o_0)
    並以前向濾波逐步推斷：先以 o_t 預測 s_t（取最大後驗，與 dual 模型在行動前說出狀態的時點相同），
    再以觀察到的 a_t 更新信念。記憶不受窗口限制，但只有八個離散狀態（不含教師的內部計數器）。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, eps):
        a = self.alpha
        CT = np.zeros((NS, NA, NO, NS)); CT2 = np.zeros((NS, NO, NS)); CT3 = np.zeros((NO, NS))
        CE = np.zeros((NS, NO, NA)); CE2 = np.zeros((NS, NA)); C0 = np.zeros((NO, NS))
        for e in eps:
            O, S, A = steps_of(e)
            o = np.array([OBS_I[int(x)] for x in O]); s = sidx(S); ac = np.array([ACT_I[int(x)] for x in A])
            C0[o[0], s[0]] += 1
            np.add.at(CE, (s, o, ac), 1); np.add.at(CE2, (s, ac), 1)
            if len(o) > 1:
                np.add.at(CT, (s[:-1], ac[:-1], o[1:], s[1:]), 1)
                np.add.at(CT2, (s[:-1], o[1:], s[1:]), 1)
                np.add.at(CT3, (o[1:], s[1:]), 1)
        nrm = lambda C, prior: (C + a * prior) / (C.sum(-1, keepdims=True) + a)
        P3 = nrm(CT3, np.full(NS, 1 / NS))
        P2 = nrm(CT2, P3[None])
        self.T = nrm(CT, P2[:, None])
        self.P0 = nrm(C0, P3)
        PE2 = nrm(CE2, np.full(NA, 1 / NA))
        self.E = nrm(CE, PE2[:, None])
        return self

    def predict_ep(self, e, return_belief: bool = False):
        O, S, A = steps_of(e)
        o = [OBS_I[int(x)] for x in O]; ac = [ACT_I[int(x)] for x in A]
        out = np.empty(len(o), dtype=int)
        bel = []
        b = self.P0[o[0]].copy()
        for t in range(len(o)):
            if t > 0:
                b = bpost @ self.T[:, ac[t - 1], o[t], :]
            b = b / b.sum()
            out[t] = V.STATE_IDS[int(b.argmax())]
            bel.append(b)
            bpost = b * self.E[:, o[t], ac[t]]
            bpost = bpost / max(bpost.sum(), 1e-300)
        return (out, np.array(bel)) if return_belief else out


def per_trip_correct(true_list, pred_list):
    """每個回合的逐步正確向量（配對分析用）。"""
    return [np.asarray(t) == np.asarray(p) for t, p in zip(true_list, pred_list)]


def free_run_multi(model, eps, agent_ids=None, greedy: bool = True):
    """多 agent 版自由追蹤：每個 (agent, 回合) 組合各自以自己的權重、自己預測的狀態詞追蹤教師回合。
    回傳 preds[agent][trip] → 預測狀態 token 陣列。"""
    block = model.cfg.block_size
    agent_ids = np.arange(model.A) if agent_ids is None else np.asarray(agent_ids)
    steps = [steps_of(e) for e in eps]
    rows = [(a, i) for a in agent_ids for i in range(len(eps))]
    seqs = [[V.BOS_ID] for _ in rows]
    preds = [[] for _ in rows]
    T = max(len(s[0]) for s in steps)
    w = np.array([a for a, _ in rows])
    for t in range(T):
        act = [j for j, (_, i) in enumerate(rows) if len(steps[i][0]) > t]
        for j in act:
            seqs[j].append(int(steps[rows[j][1]][0][t]))
        ctxs = [seqs[j][-block:] for j in act]
        lens = np.array([len(c) for c in ctxs])
        out = np.empty(len(act), dtype=int)
        for L in np.unique(lens):
            sel = np.flatnonzero(lens == L)
            ja = [act[x] for x in sel]
            p = model.next_token_probs(np.array([ctxs[x] for x in sel]), w[ja], V.STATE_MASK)
            out[sel] = p.argmax(1)
        for x, j in enumerate(act):
            i = rows[j][1]
            preds[j].append(int(out[x]))
            seqs[j].append(int(out[x]))
            seqs[j].append(int(steps[i][2][t]))
    res = {int(a): [None] * len(eps) for a in agent_ids}
    for j, (a, i) in enumerate(rows):
        res[int(a)][i] = np.array(preds[j])
    return res


def aux_readout_multi(model, eps, agent_ids=None):
    """aux（或任何以 bo 格式輸入的轉換器）在每個感知詞位置以狀態頭讀出狀態；不回饋。"""
    block = model.cfg.block_size
    agent_ids = np.arange(model.A) if agent_ids is None else np.asarray(agent_ids)
    items = []
    for i, e in enumerate(eps):
        bo = V.to_bo(e)
        n = (len(bo) - 1) // 2
        for t in range(n):
            items.append((i, t, bo[max(0, 2 * t + 2 - block):2 * t + 2]))
    by_len = defaultdict(list)
    for it in items:
        by_len[len(it[2])].append(it)
    res = {int(a): [np.empty((len(steps_of(e)[0]),), dtype=int) for e in eps] for a in agent_ids}
    for L, its in by_len.items():
        for c0 in range(0, len(its), 4000):
            ch = its[c0:c0 + 4000]
            ctx = np.array([c[2] for c in ch])
            for a in agent_ids:
                p = model.next_token_probs(ctx, np.full(len(ch), a), V.STATE_MASK)
                am = p.argmax(1)
                for x, c in enumerate(ch):
                    res[int(a)][c[0]][c[1]] = am[x]
    return res


def bo_hidden_per_trip(model, eps, layer: str = "h3"):
    """bo／aux 轉換器：每個回合每個感知詞位置的最後一層表徵 → list of (n_steps, C)。"""
    block = model.cfg.block_size
    items = []
    for i, e in enumerate(eps):
        bo = V.to_bo(e)
        for t in range((len(bo) - 1) // 2):
            items.append((i, t, bo[max(0, 2 * t + 2 - block):2 * t + 2]))
    out = [np.empty((len(steps_of(e)[0]), model.cfg.n_embd)) for e in eps]
    by_len = defaultdict(list)
    for it in items:
        by_len[len(it[2])].append(it)
    for L, its in by_len.items():
        for c0 in range(0, len(its), 3000):
            ch = its[c0:c0 + 3000]
            model.forward(np.array([c[2] for c in ch])[None], np.array([0]), cache=True)
            H = model._c[layer][0, :, -1, :]
            for x, c in enumerate(ch):
                out[c[0]][c[1]] = H[x]
    return out


def ngram_per_trip(train_eps, eps, n):
    ng = NGramState(n).fit(train_eps)
    return [ng.predict_ep(e) for e in eps]


def trip_bootstrap(correct_a, correct_b, true_s, state=None, n_boot: int = 10000, seed: int = 0):
    """以回合為單位的配對 bootstrap：correct_a/b 為每個回合的逐步正確（布林陣列，或跨種子平均的比例），
    state 指定只計某一狀態的位置。回傳 a、b 的準確率、差異與 95% 區間、以及 a 勝過 b 的回合比例。"""
    rng = np.random.default_rng(seed)
    na, nb, nn = [], [], []
    for ca, cb, ts in zip(correct_a, correct_b, true_s):
        m = np.ones(len(ts), bool) if state is None else (np.asarray(ts) == V.STOI[state])
        na.append(np.asarray(ca, float)[m].sum()); nb.append(np.asarray(cb, float)[m].sum()); nn.append(m.sum())
    na, nb, nn = map(np.array, (na, nb, nn))
    keep = nn > 0
    na, nb, nn = na[keep], nb[keep], nn[keep]
    acc_a, acc_b = na.sum() / nn.sum(), nb.sum() / nn.sum()
    idx = rng.integers(0, len(nn), (n_boot, len(nn)))
    d = (na[idx].sum(1) - nb[idx].sum(1)) / nn[idx].sum(1)
    return {"acc_a": float(acc_a), "acc_b": float(acc_b), "diff": float(acc_a - acc_b),
            "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)), "n_trips": int(len(nn)),
            "n_pos": int(nn.sum()), "p_boot": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

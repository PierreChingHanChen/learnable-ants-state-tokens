"""
GRU 基準
=====================
參數量與 microGPT 螞蟻相近（嵌入 16、隱藏 25 → 4,979 個參數；bo 轉換器為 4,992）的單層 GRU，
numpy 實作、手寫時間反向傳播（完整回合 BPTT，不截斷）。與轉換器不同，GRU 的記憶沒有上下文窗口限制。

兩種訓練目標：
  'bo'  —— 只預測行動（與 bo 轉換器相同；全詞彙 softmax，損失只在行動位置）
  'aux' —— 在每個感知詞位置同時以行動頭與狀態頭（各自的 softmax）預測 a_t 與 s_t；狀態不回饋到輸入
輸入一律是 bo 序列（. o a o a …）；狀態標籤來自教師的 dual 序列（vocab.aux_view）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import vocab as V

PNAMES = ("emb", "wi", "wh", "bi", "bhn", "wo")


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class GRUModel:
    def __init__(self, n_embd: int = 16, n_hidden: int = 25, seed: int = 0, params: Optional[Dict] = None):
        self.E, self.H = n_embd, n_hidden
        if params is not None:
            self.p = {k: np.array(v, dtype=float) for k, v in params.items()}
            self.E, self.H = self.p["emb"].shape[1], self.p["wh"].shape[0]
        else:
            rng = np.random.default_rng(seed)
            k = 1.0 / np.sqrt(n_hidden)
            self.p = {"emb": rng.normal(0, 0.3, (V.V, n_embd)),
                      "wi": rng.uniform(-k, k, (n_embd, 3 * n_hidden)),
                      "wh": rng.uniform(-k, k, (n_hidden, 3 * n_hidden)),
                      "bi": np.zeros(3 * n_hidden), "bhn": np.zeros(n_hidden),
                      "wo": rng.normal(0, 0.08, (n_hidden, V.V))}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.p)

    @classmethod
    def load(cls, path: str) -> "GRUModel":
        z = np.load(path)
        return cls(params={k: z[k] for k in PNAMES})

    # ------------------------------------------------------------ forward
    def forward(self, idx: np.ndarray, cache: bool = False):
        """idx (B,T) → logits (B,T,V)，隱藏狀態 (B,T,H)。"""
        P, H = self.p, self.H
        B, T = idx.shape
        X = P["emb"][idx]
        Ai = X @ P["wi"] + P["bi"]
        h = np.zeros((B, H))
        hs = np.empty((B, T, H))
        if cache:
            rs, zs, ns, hns, hprev = (np.empty((B, T, H)) for _ in range(5))
        for t in range(T):
            hh = h @ P["wh"]
            r = _sig(Ai[:, t, :H] + hh[:, :H])
            z = _sig(Ai[:, t, H:2 * H] + hh[:, H:2 * H])
            hn = hh[:, 2 * H:] + P["bhn"]
            n = np.tanh(Ai[:, t, 2 * H:] + r * hn)
            if cache:
                rs[:, t], zs[:, t], ns[:, t], hns[:, t], hprev[:, t] = r, z, n, hn, h
            h = (1 - z) * n + z * h
            hs[:, t] = h
        logits = hs @ P["wo"]
        if cache:
            self._c = dict(idx=idx, X=X, hs=hs, r=rs, z=zs, n=ns, hn=hns, hp=hprev)
        return logits, hs

    def backward(self, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        c, P, H = self._c, self.p, self.H
        B, T, _ = dlogits.shape
        g = {k: np.zeros_like(v) for k, v in P.items()}
        g["wo"] = c["hs"].reshape(-1, H).T @ dlogits.reshape(-1, V.V)
        dhs = dlogits @ P["wo"].T
        dA = np.empty((B, T, 3 * H))
        dh_next = np.zeros((B, H))
        whT = P["wh"].T
        for t in range(T - 1, -1, -1):
            dh = dh_next + dhs[:, t]
            r, z, n, hn, hp = c["r"][:, t], c["z"][:, t], c["n"][:, t], c["hn"][:, t], c["hp"][:, t]
            dn = dh * (1 - z) * (1 - n * n)
            dz = dh * (hp - n) * z * (1 - z)
            dr = dn * hn * r * (1 - r)
            dhn = dn * r
            dA[:, t, :H], dA[:, t, H:2 * H], dA[:, t, 2 * H:] = dr, dz, dn
            dhh = np.concatenate([dr, dz, dhn], 1)
            g["wh"] += hp.T @ dhh
            g["bhn"] += dhn.sum(0)
            dh_next = dh * z + dhh @ whT
        g["bi"] = dA.sum((0, 1))
        g["wi"] = c["X"].reshape(-1, self.E).T @ dA.reshape(-1, 3 * H)
        dX = dA @ P["wi"].T
        np.add.at(g["emb"], c["idx"], dX)
        return g

    def step(self, g, lr=1e-2, b1=0.9, b2=0.99, clip=1.0):
        norm = np.sqrt(sum((x ** 2).sum() for x in g.values()))
        sc = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k, x in g.items():
            x = x * sc
            self.m[k] = b1 * self.m[k] + (1 - b1) * x
            self.v[k] = b2 * self.v[k] + (1 - b2) * x * x
            self.p[k] -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + 1e-8)


# ---------------------------------------------------------------- data
def make_batch(eps_dual: List[List[int]]):
    """dual 回合 → bo 輸入 idx、行動目標、狀態目標及遮罩（右側補齊）。"""
    views = [V.aux_view(e) for e in eps_dual]
    T = max(len(b) for b, _ in views) - 1
    B = len(views)
    idx = np.full((B, T), V.BOS_ID, dtype=int)
    tgt = np.full((B, T), V.BOS_ID, dtype=int)
    am = np.zeros((B, T), bool)
    tgs = np.full((B, T), V.STATE_IDS[0], dtype=int)
    sm = np.zeros((B, T), bool)
    for i, (bo, lab) in enumerate(views):
        L = len(bo) - 1
        idx[i, :L] = bo[:-1]
        tgt[i, :L] = bo[1:]
        am[i, :L] = V.ACTION_MASK[np.asarray(bo[1:])]
        lb = np.asarray(lab[:L])
        sm[i, :L] = lb >= 0
        tgs[i, :L] = np.where(lb >= 0, lb, V.STATE_IDS[0])
    return idx, tgt, am, tgs, sm


def _ce(logits, tgt, mask, allowed=None):
    lg = logits if allowed is None else np.where(allowed, logits, -np.inf)
    zz = lg - lg.max(-1, keepdims=True)
    ez = np.exp(zz)
    sm = ez / ez.sum(-1, keepdims=True)
    lp = np.log(np.take_along_axis(sm, tgt[..., None], -1)[..., 0] + 1e-300)
    d = sm.copy()
    np.put_along_axis(d, tgt[..., None], np.take_along_axis(d, tgt[..., None], -1) - 1, -1)
    d *= mask[..., None]
    return -(lp * mask).sum(), d


def loss_grad(model: GRUModel, batch, objective: str):
    idx, tgt, am, tgs, sm = batch
    logits, _ = model.forward(idx, cache=True)
    if objective == "bo":
        l, d = _ce(logits, tgt, am)
        n = am.sum()
    else:
        l1, d1 = _ce(logits, tgt, am, V.ACTION_MASK)
        l2, d2 = _ce(logits, tgs, sm, V.STATE_MASK)
        l, d, n = l1 + l2, d1 + d2, am.sum() + sm.sum()
    n = max(int(n), 1)
    return l / n, model.backward(d / n)


def train(objective: str, train_eps, val_eps, seed: int = 0, train_steps: int = 3000, batch: int = 64,
          lr: float = 1e-2, n_embd: int = 16, n_hidden: int = 25, out: Optional[str] = None, verbose=False):
    """完整回合 BPTT。以長度分桶（相近長度的回合同批）減少補齊浪費。"""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    model = GRUModel(n_embd, n_hidden, seed=seed)
    order = np.argsort([len(e) for e in train_eps], kind="stable")
    curve = []
    for s in range(train_steps):
        lr_s = lr * (1 - s / train_steps) + 1e-4
        st = int(rng.integers(0, len(order) - batch + 1))
        sel = order[st:st + batch]
        loss, g = loss_grad(model, make_batch([train_eps[i] for i in sel]), objective)
        model.step(g, lr_s)
        if s % 250 == 0 or s == train_steps - 1:
            curve.append((s, float(loss)))
            if verbose:
                print(f"  [gru-{objective}] step {s:5d} loss {loss:.4f}", flush=True)
    stats = evaluate(model, val_eps)
    stats.update({"mode": f"gru_{objective}", "n_params": model.n_params(), "seconds": round(time.time() - t0, 1),
                  "curve": curve, "seed": seed})
    if out:
        model.save(out)
        with open(out.replace(".npz", ".json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
    return model, stats


def run_eps(model: GRUModel, eps_dual, chunk: int = 128):
    """回傳每個回合在感知詞位置的 (行動 logits, 狀態 logits, 隱藏狀態) 與教師標籤。"""
    order = np.argsort([len(e) for e in eps_dual])
    out = [None] * len(eps_dual)
    for c0 in range(0, len(order), chunk):
        sel = order[c0:c0 + chunk]
        idx, tgt, am, tgs, sm = make_batch([eps_dual[i] for i in sel])
        logits, hs = model.forward(idx)
        for j, i in enumerate(sel):
            pos = np.flatnonzero(sm[j])                # 感知詞位置（之後是行動）
            out[i] = dict(logits=logits[j, pos], h=hs[j, pos], a=tgt[j, pos], s=tgs[j, pos])
    return out


def evaluate(model: GRUModel, eps_dual) -> Dict:
    res = run_eps(model, eps_dual)
    lg = np.concatenate([r["logits"] for r in res])
    a = np.concatenate([r["a"] for r in res])
    s = np.concatenate([r["s"] for r in res])
    out = {}
    for name, tg, m in (("action", a, V.ACTION_MASK), ("state", s, V.STATE_MASK)):
        l = np.where(m, lg, -np.inf)
        l = l - l.max(-1, keepdims=True)
        lp = l - np.log(np.exp(l).sum(-1, keepdims=True))
        out[f"{name}_nll"] = float(-np.take_along_axis(lp, tg[:, None], -1).mean())
        out[f"{name}_acc"] = float((l.argmax(-1) == tg).mean())
    return out
